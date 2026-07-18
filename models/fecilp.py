"""
FECLIP: Forensics-Expert CLIP with Prompt Learning & PCGrad
核心模型 —— 基于 CLIP + ViT Forensic Expert 的多模态融合分类器
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from transformers import CLIPModel, CLIPTokenizer
from peft import LoraConfig, TaskType, LoraModel

try:
    from torchvision.models import vit_b_16, ViT_B_16_Weights
except ImportError:
    vit_b_16 = None
    ViT_B_16_Weights = None


# =========================================================================
# 1. TorchvisionVitExpert — torchvision ViT-B/16 作为取证特征专家
# =========================================================================
class TorchvisionVitExpert(nn.Module):
    def __init__(self, pretrained: bool = True, out_indices: Tuple[int, ...] = (2, 3),
                 img_size: int = 224, local_ckpt: Optional[str] = None):
        super().__init__()
        assert vit_b_16 is not None, "torchvision 未安装或版本过低"

        if pretrained:
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.vit = vit_b_16(weights=None)

        self.out_indices = tuple(out_indices)
        self.img_size = img_size
        if local_ckpt:
            self._load_local_ckpt_best_effort(local_ckpt)

        self.embed_dim = self.vit.conv_proj.out_channels
        patch = self.vit.conv_proj.kernel_size[0]
        assert img_size % patch == 0
        gh = gw = img_size // patch
        self.grid_size = (gh, gw)

    @torch.no_grad()
    def _to_img_size(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] == self.img_size and x.shape[-1] == self.img_size:
            return x
        return F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

    def _load_local_ckpt_best_effort(self, local_ckpt: str):
        print(f"[VitExpert] best-effort load: {local_ckpt}")
        raw = torch.load(local_ckpt, map_location="cpu")

        if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
            sd_raw = raw["state_dict"]
        elif isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
            sd_raw = raw["model"]
        elif isinstance(raw, dict):
            sd_raw = raw
        else:
            raise TypeError(f"Unsupported checkpoint type: {type(raw)}")

        def _strip_prefixes(name):
            for p in ("module.", "model.", "backbone.", "vit."):
                if name.startswith(p): name = name[len(p):]
            return name

        def _is_head_like(name):
            return (name.startswith("heads.") or name.startswith("head.")
                    or name.startswith("classifier.") or name.startswith("fc.") or "cls_head" in name)

        src_sd = {}
        for k, v in sd_raw.items():
            if not torch.is_tensor(v): continue
            k_clean = _strip_prefixes(k)
            if _is_head_like(k_clean): continue
            src_sd[k_clean] = v

        dst_sd = self.vit.state_dict()
        norm2dst_keys = {}
        for dk in dst_sd.keys():
            norm2dst_keys.setdefault(_strip_prefixes(dk), []).append(dk)

        new_sd, used_tgt, matched, skipped_shape, skipped_name = {}, set(), 0, 0, 0
        for sk, sv in src_sd.items():
            if sk in dst_sd and dst_sd[sk].shape == sv.shape and sk not in used_tgt:
                new_sd[sk] = sv; used_tgt.add(sk); matched += 1; continue
            candidates = norm2dst_keys.get(_strip_prefixes(sk), [])
            if not candidates:
                candidates = [dk for dk in dst_sd.keys() if dk.endswith(_strip_prefixes(sk))]
            candidates = [dk for dk in candidates if dst_sd[dk].shape == sv.shape and dk not in used_tgt]
            if candidates:
                new_sd[candidates[0]] = sv; used_tgt.add(candidates[0]); matched += 1
            else:
                if sk in dst_sd: skipped_shape += 1
                else: skipped_name += 1

        print(f"[VitExpert] matched:{matched} skipped_name:{skipped_name} skipped_shape:{skipped_shape}")
        missing, unexpected = self.vit.load_state_dict(new_sd, strict=False)
        if missing: print(f"  missing (example): {missing[:5]}")
        if unexpected: print(f"  unexpected (example): {unexpected[:5]}")

    def _tokens_with_pos(self, img: torch.Tensor) -> torch.Tensor:
        return self.vit._process_input(img)

    def _tokens_to_feat(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N_total, D = tokens.shape
        gh, gw = self.grid_size
        num_patches_expected = gh * gw
        if N_total == num_patches_expected + 1: x = tokens[:, 1:, :]
        elif N_total == num_patches_expected: x = tokens
        else:
            raise AssertionError(f"token count mismatch: N={N_total}, expect {num_patches_expected}")
        return x.reshape(B, gh, gw, D).permute(0, 3, 1, 2).contiguous()

    def forward(self, x_img: torch.Tensor) -> List[torch.Tensor]:
        x_img = x_img.float()
        x_img = self._to_img_size(x_img)
        x_tokens = self._tokens_with_pos(x_img)
        feats: List[torch.Tensor] = []
        for i, blk in enumerate(self.vit.encoder.layers):
            x_tokens = blk(x_tokens)
            if i in self.out_indices:
                feats.append(self._tokens_to_feat(x_tokens))
        return feats


# =========================================================================
# 2. VisionFusionAdapter — CLIP + Forensic Expert 视觉融合
# =========================================================================
class VisionFusionAdapter(nn.Module):
    def __init__(self, clip_vision_model, clip_visual_projection, get_forensic_feats,
                 target_size=32, fusion_mid_layers=None, fusion_mid_layer_index=None,
                 separate_mid_modules=False, vit_hidden_size=None, forensic_channels=None):
        super().__init__()
        self.clip_vision_model = clip_vision_model
        self.clip_visual_projection = clip_visual_projection
        self.get_forensic_feats = get_forensic_feats
        self.target_size = target_size
        self.separate_mid_modules = bool(separate_mid_modules)

        mid_layers_1b = []
        if fusion_mid_layers is not None: mid_layers_1b.extend(list(fusion_mid_layers))
        if fusion_mid_layer_index is not None: mid_layers_1b.append(int(fusion_mid_layer_index) + 1)
        self.fusion_mid_layers_1based = sorted(set(int(x) for x in mid_layers_1b if x is not None))

        vit = self._get_vit_backbone()
        if vit_hidden_size is None:
            vit_hidden_size = getattr(getattr(vit, "config", None), "hidden_size", 768)
        assert forensic_channels is not None, "forensic_channels is required"

        from .intermediate_fusion import IntermediateFusion
        self._ifm_last = IntermediateFusion(vit_dim=vit_hidden_size, third_channels=forensic_channels,
                                             last_channels=forensic_channels)
        if self.separate_mid_modules and len(self.fusion_mid_layers_1based) > 0:
            self._ifm_mid = nn.ModuleList([
                IntermediateFusion(vit_dim=vit_hidden_size, third_channels=forensic_channels, last_channels=forensic_channels)
                for _ in range(len(self.fusion_mid_layers_1based))
            ])
        else:
            self._ifm_mid = IntermediateFusion(vit_dim=vit_hidden_size, third_channels=forensic_channels, last_channels=forensic_channels)
        self._has_cls_cache = None

    def _get_vit_backbone(self):
        vm = self.clip_vision_model
        return vm.vision_model if hasattr(vm, "vision_model") else vm

    def _build_tokens(self, vit, pixel_values):
        if hasattr(vit, "embeddings"):
            tokens = vit.embeddings(pixel_values)
        else:
            raise ValueError("Unsupported CLIP Vision Model structure")
        self._has_cls_cache = (tokens.shape[1] == int(((tokens.shape[1] - 1) ** 0.5)) ** 2 + 1)
        return tokens

    def forward(self, img):
        vit = self._get_vit_backbone()
        forensic_feats = self.get_forensic_feats(img)
        tokens = self._build_tokens(vit, img)

        mid_ptr = 0
        for li, layer in enumerate(vit.encoder.layers):
            out = layer(tokens, attention_mask=None, causal_attention_mask=None, output_attentions=False)
            tokens = out[0]
            if (li + 1) in self.fusion_mid_layers_1based:
                if isinstance(self._ifm_mid, nn.ModuleList):
                    tokens = self._ifm_mid[mid_ptr](tokens, forensic_feats[0], forensic_feats[-1])
                    mid_ptr += 1
                else:
                    tokens = self._ifm_mid(tokens, forensic_feats[0], forensic_feats[-1])

        tokens = self._ifm_last(tokens, forensic_feats[0], forensic_feats[-1])
        if hasattr(vit, "post_layernorm"):
            tokens = vit.post_layernorm(tokens)
        pooled = tokens[:, 0, :] if self._has_cls_cache else tokens.mean(dim=1)
        return self.clip_visual_projection(pooled)


# =========================================================================
# 3. PromptLearner — CoOp 可学习上下文向量
# =========================================================================
class PromptLearner(nn.Module):
    def __init__(self, args, classnames, clip_model, tokenizer):
        super().__init__()
        if isinstance(classnames, dict):
            classnames = classnames.get('all', classnames.get('train', []))
        self.classnames = classnames
        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.config.text_config.hidden_size
        token_embedding = clip_model.text_model.embeddings.token_embedding

        n_ctx = getattr(args, "n_ctx", 16)
        ctx_init = getattr(args, "ctx_init", "")
        print(f"[PromptLearner] n_ctx={n_ctx}, ctx_init='{ctx_init}'")

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(tokenizer(ctx_init, add_special_tokens=False)["input_ids"])
            prompt = tokenizer(ctx_init, add_special_tokens=False, return_tensors="pt")
            with torch.no_grad():
                embedding = token_embedding(prompt["input_ids"]).type(dtype)
            ctx_vectors = embedding[0].clone().detach()
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)
        classnames_with_prompts = [name.replace("_", " ") for name in classnames]
        prompts = [ctx_init + " " + name + "." if ctx_init else prompt_prefix + " " + name + "."
                   for name in classnames_with_prompts]
        tokenized_prompts = tokenizer(prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts["input_ids"]).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.token_embedding = token_embedding
        self.register_buffer("input_ids", tokenized_prompts["input_ids"])

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


# =========================================================================
# 4. FECLIP 主模型
# =========================================================================
class FECLIP(nn.Module):
    def __init__(self, dset, classnames, args, model=None, tokenizer=None, few_shot=False):
        super().__init__()
        self.args = args
        self.dset = dset
        self.classnames = classnames
        self.train_stage = getattr(args, "train_stage", "fusion").lower()
        self.use_swei_fusion = getattr(args, "use_swei_fusion", True)

        if self.train_stage == "clip":
            self.use_swei_fusion = False

        # Prefer --clip_model_path; otherwise load from Hugging Face Hub
        default_clip_path = "openai/clip-vit-base-patch16"
        clip_model_path = getattr(args, "clip_model_path", None) or default_clip_path

        print(f"[FECLIP] Loading CLIP from {clip_model_path}")
        try:
            self.clip_model = CLIPModel.from_pretrained(clip_model_path)
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_path)
        except Exception as e:
            print(f"[FECLIP] CLIP load failed ({clip_model_path}): {e}. Falling back to openai/clip-vit-base-patch16")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

        self.logit_scale = getattr(self.clip_model, "logit_scale", nn.Parameter(torch.ones([]) * 4.6052))

        if self.train_stage in ["clip", "joint"]:
            print("[FECLIP] Initializing Prompt Learner...")
            self.prompt_learner = PromptLearner(args, self.classnames, self.clip_model, self.tokenizer)
        else:
            self.prompt_learner = None

        if self.use_swei_fusion:
            print("[FECLIP] Initializing Fusion Components...")
            vit_ckpt = getattr(args, "vit_ckpt_path", None) or None
            self.forensics_expert = TorchvisionVitExpert(pretrained=False, local_ckpt=vit_ckpt,
                                                          out_indices=(2, 3), img_size=224)
            self.forensics_expert.vit.requires_grad_(False)

            self.vision_fusion = VisionFusionAdapter(
                clip_vision_model=self.clip_model.vision_model,
                clip_visual_projection=self.clip_model.visual_projection,
                get_forensic_feats=lambda im: self.forensics_expert(im),
                fusion_mid_layers=[2, 5, 8, 11],
                separate_mid_modules=False,
                vit_hidden_size=self.clip_model.vision_model.config.hidden_size,
                forensic_channels=self.forensics_expert.embed_dim,
            )
        else:
            self.forensics_expert = None
            self.vision_fusion = None

        self.naive_decoding = getattr(args, "naive_decoding", False)
        if self.naive_decoding:
            print("[FECLIP] Initializing Vision LoRA...")
            vision_peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False,
                r=getattr(args, "lora_rank", 8), lora_alpha=getattr(args, "lora_alpha", 16),
                lora_dropout=getattr(args, "lora_dropout", 0.1),
                target_modules=["layers.{}.self_attn.q_proj".format(i) for i in range(getattr(args, "v_lora_start", 0), getattr(args, "v_lora_end", 12))]
                             + ["layers.{}.self_attn.v_proj".format(i) for i in range(getattr(args, "v_lora_start", 0), getattr(args, "v_lora_end", 12))]
            )
            if getattr(args, "freeze_vit", False):
                self.lora_model = None
            else:
                self.lora_model = nn.ModuleDict(
                    {'default': LoraModel(self.clip_model.vision_model, {'default': vision_peft_config}, 'default')}
                )
        else:
            self.lora_model = None

        self._configure_trainable_params()

    def _configure_trainable_params(self):
        for p in self.parameters():
            p.requires_grad = False
        self.logit_scale.requires_grad = False
        print(f"[FECLIP] Configuring params for stage: {self.train_stage}")

        if self.train_stage == "clip":
            if self.lora_model is not None:
                for name, p in self.lora_model.named_parameters():
                    if "lora_" in name: p.requires_grad = True
            if self.prompt_learner is not None:
                for p in self.prompt_learner.parameters(): p.requires_grad = True

        elif self.train_stage == "fusion":
            if self.vision_fusion is not None:
                for name, p in self.vision_fusion.named_parameters():
                    if "clip_vision_model" not in name and "clip_visual_projection" not in name:
                        p.requires_grad = True
            if self.lora_model is not None:
                for name, p in self.lora_model.named_parameters():
                    if "lora_" in name: p.requires_grad = True

        elif self.train_stage == "joint":
            if self.lora_model is not None:
                for name, p in self.lora_model.named_parameters():
                    if "lora_" in name: p.requires_grad = True
            if self.prompt_learner is not None:
                for p in self.prompt_learner.parameters(): p.requires_grad = True
            if self.vision_fusion is not None:
                for name, p in self.vision_fusion.named_parameters():
                    if "clip_vision_model" not in name and "clip_visual_projection" not in name:
                        p.requires_grad = True

    def extract_image_features(self, img):
        if self.use_swei_fusion and self.vision_fusion is not None:
            return F.normalize(self.vision_fusion(img), dim=-1)
        if self.lora_model is not None:
            outputs = self.lora_model['default'](img)
            image_features = self.clip_model.visual_projection(outputs[1])
        else:
            image_features = self.clip_model.get_image_features(img)
        return F.normalize(image_features, dim=-1)

    def _build_causal_attention_mask(self, bsz, seq_len, dtype, device):
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
        mask.triu_(1)
        return mask.unsqueeze(0).expand(bsz, 1, seq_len, seq_len)

    def get_text_features(self, subset='train'):
        if self.prompt_learner is not None:
            prompts = self.prompt_learner()
            text_model = self.clip_model.text_model
            bsz, seq_len, _ = prompts.shape
            position_ids = torch.arange(seq_len, dtype=torch.long, device=prompts.device).unsqueeze(0).expand(bsz, -1)
            position_embeddings = text_model.embeddings.position_embedding(position_ids)
            hidden_states = prompts + position_embeddings
            causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype, hidden_states.device)
            encoder_outputs = text_model.encoder(inputs_embeds=hidden_states, causal_attention_mask=causal_attention_mask)
            last_hidden_state = text_model.final_layer_norm(encoder_outputs[0])
            input_ids = self.prompt_learner.input_ids
            pooled_output = last_hidden_state[torch.arange(bsz, device=last_hidden_state.device), input_ids.argmax(dim=-1)]
            return F.normalize(self.clip_model.text_projection(pooled_output), dim=-1)
        else:
            if hasattr(self, "cached_text_features") and not self.training:
                return self.cached_text_features
            device = next(self.parameters()).device
            names_list = (self.classnames.get('all', self.classnames.get('train', []))
                          if isinstance(self.classnames, dict) else self.classnames)
            prompts = [self.args.prompt_template.format(c.replace("_", " ")) for c in names_list]
            text_inputs = self.tokenizer(prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
            text_features = F.normalize(self.clip_model.text_projection(self.clip_model.text_model(**text_inputs)[1]), dim=-1)
            if not self.training: self.cached_text_features = text_features
            return text_features

    def forward(self, batch, subset='train'):
        if not isinstance(batch, (list, tuple)):
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        if len(batch) == 2:
            img, label = batch; img_aug = None
        elif len(batch) >= 3:
            a, b, c = batch[0], batch[1], batch[2]
            if torch.is_tensor(b) and b.dim() == 4:
                img, img_aug, label = a, b, c
            else:
                img, label, img_aug = a, b, None
        else:
            raise ValueError(f"Empty batch: len={len(batch)}")

        img = img.to(next(self.parameters()).device)
        if label is not None: label = label.to(next(self.parameters()).device)

        image_features = self.extract_image_features(img)
        text_features = self.get_text_features(subset)

        # L1 consistency loss (LoRA on/off)
        loss_l1_image = torch.tensor(0.0, device=img.device)
        if self.training and self.lora_model is not None:
            try:
                ctx = None
                if hasattr(self.lora_model['default'], "disable_adapter"):
                    ctx = self.lora_model['default'].disable_adapter()
                if ctx is not None:
                    with ctx:
                        with torch.no_grad():
                            orig_outputs = self.clip_model.vision_model(img)
                            orig_feat = F.normalize(self.clip_model.visual_projection(orig_outputs[1]), dim=-1)
                    loss_l1_image = (1.0 - F.cosine_similarity(image_features, orig_feat, dim=-1).mean()) * 5.0
            except Exception:
                pass

        with torch.no_grad():
            self.logit_scale.clamp_(max=math.log(100.0))
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        output_dict = {'logits': logits}

        if self.training and label is not None:
            # Extract domain_id / sample_weight from batch
            domain_id = None
            if isinstance(batch, (list, tuple)) and len(batch) >= 4:
                extra = batch[3]
                if torch.is_tensor(extra) and extra.numel() == label.numel():
                    if extra.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                        domain_id = extra

            ce_per = F.cross_entropy(logits, label, reduction="none",
                                     label_smoothing=getattr(self.args, 'label_smoothing', 0.0))

            # Domain weights
            w = torch.ones_like(ce_per, dtype=torch.float32)
            if domain_id is not None:
                domain_id = domain_id.to(img.device)
                ws = float(getattr(self.args, "source_loss_weight", 1.0))
                wt = float(getattr(self.args, "target_loss_weight", 2.0))
                w = w * torch.where(domain_id > 0, torch.tensor(wt, device=img.device), torch.tensor(ws, device=img.device)).float()

            # Margin-based uncertainty weight
            if getattr(self.args, "use_margin_weight", False):
                with torch.no_grad():
                    probs = torch.softmax(logits.detach(), dim=-1)
                    top2 = probs.topk(2, dim=-1).values
                    p1n, p2n = top2[:, 0] / top2.sum(dim=-1).clamp_min(1e-12), top2[:, 1] / top2.sum(dim=-1).clamp_min(1e-12)
                    ent2 = -(p1n * torch.log2(p1n.clamp_min(1e-12)) + p2n * torch.log2(p2n.clamp_min(1e-12)))
                    w_mw = (1.0 - ent2).clamp(float(getattr(self.args, "mw_min_w", 0.1)),
                                             float(getattr(self.args, "mw_max_w", 1.0)))
                w = w * w_mw.float()

            loss_ce = (w * ce_per.float()).sum() / w.sum().clamp_min(1e-12)
            total_loss = loss_ce + loss_l1_image

            output_dict['loss_ce'] = loss_ce
            output_dict['loss_l1'] = loss_l1_image
            output_dict['loss_total'] = total_loss

            if img_aug is not None and getattr(self.args, 'use_aug_loss', False):
                img_aug_feat = self.extract_image_features(img_aug)
                logits_aug = logit_scale * img_aug_feat @ text_features.t()
                ce_aug = F.cross_entropy(logits_aug, label, reduction="none")
                loss_aug = (w * ce_aug.float()).sum() / w.sum().clamp_min(1e-12)
                output_dict['loss_total'] += loss_aug * 0.5

        return (output_dict, logits) if self.training else (output_dict, logits)
