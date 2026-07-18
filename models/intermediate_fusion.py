"""Intermediate fusion: CLIP ViT tokens + forensic ViT multi-scale features."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSparseAttentionMap(nn.Module):
    """Cross-attention gate between ViT query features and third/last forensic maps."""

    def __init__(self, third_channels=320, last_channels=512, unified_dim=512, target_size=32):
        super().__init__()
        self.target_size = target_size
        self.unified_dim = unified_dim
        self.scale = 1.0 / math.sqrt(unified_dim)

        self.third_proj = nn.Sequential(
            nn.Conv2d(third_channels, unified_dim, 1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=unified_dim),
            nn.GELU(),
        )
        self.last_proj = nn.Sequential(
            nn.Conv2d(last_channels, unified_dim, 1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=unified_dim),
            nn.GELU(),
        )
        self.q_proj = nn.Conv2d(unified_dim, unified_dim, 1, bias=False)
        self.k3_proj = nn.Conv2d(unified_dim, unified_dim, 1, bias=False)
        self.kL_proj = nn.Conv2d(unified_dim, unified_dim, 1, bias=False)
        self.v3_proj = nn.Conv2d(unified_dim, unified_dim, 1, bias=False)
        self.vL_proj = nn.Conv2d(unified_dim, unified_dim, 1, bias=False)
        self.apply(self._init_fn)

    @staticmethod
    def _init_fn(m):
        if isinstance(m, nn.Conv2d):
            if hasattr(m.weight, "device") and m.weight.device.type != "meta":
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            if hasattr(m.weight, "device") and m.weight.device.type != "meta":
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, q_feat, third_feat, last_feat):
        size = (self.target_size, self.target_size)
        q_feat_aligned = F.interpolate(q_feat, size=size, mode="bilinear", align_corners=False)
        third_aligned = F.interpolate(third_feat, size=size, mode="bilinear", align_corners=False)
        last_aligned = F.interpolate(last_feat, size=size, mode="bilinear", align_corners=False)

        third_proj = self.third_proj(third_aligned)
        last_proj = self.last_proj(last_aligned)

        Q = self.q_proj(q_feat_aligned)
        K3 = self.k3_proj(third_proj)
        KL = self.kL_proj(last_proj)
        V3 = self.v3_proj(third_proj)
        VL = self.vL_proj(last_proj)

        l3 = (Q.float() * K3.float()).sum(dim=1, keepdim=True) * self.scale
        lL = (Q.float() * KL.float()).sum(dim=1, keepdim=True) * self.scale
        weights = torch.softmax(torch.cat([l3, lL], dim=1), dim=1).to(Q.dtype)
        alpha = weights[:, 0:1, :, :]
        fused = alpha * V3 + (1 - alpha) * VL
        return alpha, fused


class IntermediateFusion(nn.Module):
    def __init__(self, vit_dim=1024, third_channels=320, last_channels=512):
        super().__init__()
        assert vit_dim > 0 and third_channels > 0 and last_channels > 0
        self.target_size = 32
        self.attention_gen = HybridSparseAttentionMap(
            third_channels=third_channels,
            last_channels=last_channels,
            unified_dim=vit_dim,
            target_size=self.target_size,
        )
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(vit_dim * 2, vit_dim + vit_dim // 2, kernel_size=1),
            nn.GroupNorm(num_groups=32, num_channels=vit_dim + vit_dim // 2),
            nn.GELU(),
            nn.Conv2d(vit_dim + vit_dim // 2, vit_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=vit_dim),
            nn.GELU(),
            nn.Conv2d(vit_dim, vit_dim, kernel_size=1),
            nn.GroupNorm(num_groups=32, num_channels=vit_dim),
        )
        self.gate_logit = nn.Parameter(torch.tensor(0.0))
        self.gate_floor = 0.05
        self.scale_clip = (0.1, 10.0)
        self.apply(self._init_fn)

    @staticmethod
    def _init_fn(m):
        if isinstance(m, nn.Conv2d):
            if hasattr(m.weight, "device") and m.weight.device.type != "meta":
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            if hasattr(m.weight, "device") and m.weight.device.type != "meta":
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, vit_feat, third_feat, last_feat):
        B = vit_feat.shape[0]
        has_cls = False
        cls_token = None

        if len(vit_feat.shape) == 3:
            N, C = vit_feat.shape[1], vit_feat.shape[2]
            H = W = int((N - 1) ** 0.5)
            if N == H * W + 1:
                cls_token = vit_feat[:, :1, :]
                vit_feat_2d = vit_feat[:, 1:, :].transpose(1, 2).reshape(B, C, H, W)
                has_cls = True
            else:
                vit_feat_2d = vit_feat.transpose(1, 2).reshape(B, C, H, W)
        else:
            vit_feat_2d = vit_feat

        _, sparse_fused = self.attention_gen(vit_feat_2d, third_feat, last_feat)
        if sparse_fused.shape[2:] != vit_feat_2d.shape[2:]:
            sparse_fused = F.interpolate(
                sparse_fused, size=vit_feat_2d.shape[2:], mode="bilinear", align_corners=False
            )

        projected = self.fusion_proj(torch.cat([vit_feat_2d, sparse_fused], dim=1))

        def rms(x):
            return x.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt()

        scale = (rms(vit_feat_2d) / (rms(projected) + 1e-6)).clamp_(
            self.scale_clip[0], self.scale_clip[1]
        ).detach()
        gate = self.gate_floor + (1.0 - self.gate_floor) * torch.sigmoid(self.gate_logit)
        enhanced_feat_2d = vit_feat_2d + gate * scale * projected

        if len(vit_feat.shape) == 3:
            enhanced_feat = enhanced_feat_2d.flatten(2).transpose(1, 2)
            if has_cls:
                enhanced_feat = torch.cat([cls_token, enhanced_feat], dim=1)
            return enhanced_feat
        return enhanced_feat_2d
