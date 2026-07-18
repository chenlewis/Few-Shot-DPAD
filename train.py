"""FE-CLIP training: source + few-shot target mix, PCGrad + margin weighting."""

import glob
import inspect
import json
import math
import os
from decimal import Decimal
from functools import partial

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, StateDictType, FullStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data.meta_dataset import MetaDataset
from flags import parser
from models.common import Classification
from models.fecilp import FECLIP
from utils.utils import save_args, load_args, is_main_process

torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)
cudnn.benchmark = True

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False
    roc_auc_score = None
    roc_curve = None


def _pcgrad_flatten_grads(params):
    flats = []
    for p in params:
        if p.grad is None:
            continue
        flats.append(p.grad.detach().view(-1))
    if not flats:
        return None
    return torch.cat(flats, dim=0)


def _pcgrad_assign_flat_grads(params, flat):
    offset = 0
    for p in params:
        if p.grad is None:
            continue
        n = p.grad.numel()
        p.grad.copy_(flat[offset:offset + n].view_as(p.grad))
        offset += n


def _pcgrad_project(g_i, g_j, eps=1e-12):
    dot = torch.dot(g_i, g_j)
    if dot < 0:
        denom = torch.dot(g_j, g_j).clamp_min(eps)
        g_i = g_i - (dot / denom) * g_j
    return g_i


def _pcgrad_multi(grads, seed=None):
    if len(grads) <= 1:
        return grads[0] if grads else None
    if seed is None:
        seed = 0
    g_out = []
    for i, g in enumerate(grads):
        if g is None:
            continue
        idxs = list(range(len(grads)))
        for j in range(len(idxs) - 1, 0, -1):
            k = (seed + i * 997 + j * 101) % (j + 1)
            idxs[j], idxs[k] = idxs[k], idxs[j]
        g_i = g
        for j in idxs:
            if j == i or grads[j] is None:
                continue
            g_i = _pcgrad_project(g_i, grads[j])
        g_out.append(g_i)
    return torch.stack(g_out, dim=0).mean(dim=0)


def _margin_weight_from_logits(logits, min_w=0.1, max_w=1.0):
    with torch.no_grad():
        probs = torch.softmax(logits.detach(), dim=-1)
        top2 = probs.topk(2, dim=-1).values
        p1, p2 = top2[:, 0], top2[:, 1]
        s = (p1 + p2).clamp_min(1e-12)
        p1n, p2n = p1 / s, p2 / s
        ent2 = -(
            p1n * torch.log2(p1n.clamp_min(1e-12))
            + p2n * torch.log2(p2n.clamp_min(1e-12))
        )
        w_mw = (1.0 - ent2).clamp(float(min_w), float(max_w))
    return w_mw


def _compute_auc_eer(probs_np, labels_np):
    import numpy as np
    if not _HAS_SKLEARN:
        return {"auc": None, "eer": None}

    n_classes = probs_np.shape[1]
    if n_classes == 2:
        y_true = labels_np.astype(int)
        y_score = probs_np[:, 1]
        try:
            auc = roc_auc_score(y_true, y_score)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            fnr = 1.0 - tpr
            idx = np.nanargmin(np.abs(fpr - fnr))
            eer = float((fpr[idx] + fnr[idx]) / 2.0)
        except Exception:
            auc, eer = None, None
        return {"auc": auc, "eer": eer}

    classes = np.unique(labels_np)
    auc_list, eer_list = [], []
    for c in classes:
        y_true = (labels_np == c).astype(int)
        y_score = probs_np[:, int(c)]
        try:
            auc_c = roc_auc_score(y_true, y_score)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            fnr = 1.0 - tpr
            idx = np.nanargmin(np.abs(fpr - fnr))
            eer_c = float((fpr[idx] + fnr[idx]) / 2.0)
            auc_list.append(auc_c)
            eer_list.append(eer_c)
        except Exception:
            pass
    return {
        "auc": float(np.mean(auc_list)) if auc_list else None,
        "eer": float(np.mean(eer_list)) if eer_list else None,
    }


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0, last_epoch=-1
):
    def _lr_lambda_fn(current_step: int):
        if num_warmup_steps > 0 and current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    lr_lambda = [_lr_lambda_fn] * len(optimizer.param_groups)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def save_checkpoint(model, epoch, logpath, seed, filename, save_lightweight=True):
    is_fsdp = isinstance(model, FSDP)
    if is_fsdp:
        cfg = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = model.state_dict()
    else:
        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    if not is_main_process():
        return

    if save_lightweight:
        new_state_dict = {}
        for k, v in state_dict.items():
            if "lora_" in k or "vision_fusion" in k or "prompt_learner" in k or "logit_scale" in k:
                new_state_dict[k] = v
            elif "forensics_expert" in k and v.requires_grad:
                new_state_dict[k] = v
            elif "classifier" in k or "head" in k:
                new_state_dict[k] = v
        if new_state_dict:
            print(f"[Save Checkpoint] Lightweight: {len(new_state_dict)} keys")
            state_dict = new_state_dict
        else:
            print("[Save Checkpoint] Warning: filter empty, saving full model")

    torch.save(
        {"net": state_dict, "epoch": epoch},
        os.path.join(logpath, f"ckpt_{filename}_{seed}.t7"),
    )


def build_optimizer_parameters(args, model: nn.Module):
    fusion_params, lora_params, other_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = n.lower()
        if "lora_" in lname or "lora_a" in lname or "lora_b" in lname:
            lora_params.append(p)
        elif "vision_fusion" in lname or "fusion" in lname or "expert" in lname:
            fusion_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if fusion_params:
        print(f"[Optimizer] Fusion: {len(fusion_params)} tensors | LR={args.lr:.1e}")
        param_groups.append({"params": fusion_params, "lr": args.lr, "weight_decay": args.weight_decay})
    lora_lr_val = getattr(args, "lora_lr", args.lr * 0.1)
    if lora_params:
        print(f"[Optimizer] LoRA: {len(lora_params)} tensors | LR={lora_lr_val:.1e}")
        param_groups.append({"params": lora_params, "lr": lora_lr_val, "weight_decay": args.weight_decay})
    if other_params:
        print(f"[Optimizer] Other: {len(other_params)} tensors | LR={args.lr:.1e}")
        param_groups.append({"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay})
    return param_groups


def main():
    args = parser.parse_args()
    cli_load = args.load
    load_args(args.config, args)
    if cli_load is not None:
        args.load = cli_load
    if not args.load:
        args.load = None
    print(f"[Train] args.load = {args.load}")

    stage = str(getattr(args, "train_stage", "fusion")).lower()

    if stage in ["clip", "clip_only", "vlm"]:
        args.use_swei_fusion = False
        phase_tag = "clip"
    elif stage in ["fusion"]:
        args.use_swei_fusion = True
        phase_tag = "fusion_Cert"
    elif stage in ["joint"]:
        args.use_swei_fusion = True
        phase_tag = "joint"
    else:
        raise ValueError(f"Unknown train_stage: {stage}")

    print(f"[Train] phase = {phase_tag}, use_swei_fusion = {args.use_swei_fusion}")

    # Project Name Suffix
    project_name_suffix_o = "{num_shots}Shot-{pair_emb}-{lr:.1E}-{lr_scheuler}-{prompt_type}-{num_prior_tokens}Pr{llm_prompt_depth}x{num_llm_prompts}P{num_text_ctx}T{num_vis_ctx}V-{bias}Init-{dist_type}{lambda_dist:.1f}xDist{ema}{randaug}".format(
        num_shots=args.coop_num_shots,
        pair_emb='{:d}X{}'.format(args.num_decoder_layers, "decode"),
        lr=Decimal(args.lr),
        lr_scheuler=args.lr_scheduler if args.lr_scheduler else "constant",
        prompt_type=args.prompt_type,
        num_prior_tokens=args.num_prior_tokens,
        llm_prompt_depth=args.llm_prompt_depth,
        num_llm_prompts=args.num_llm_prompts,
        num_text_ctx=args.num_text_ctx,
        num_vis_ctx=args.num_vis_ctx,
        dist_type=args.distillation_type or "none",
        lambda_dist=args.lambda_dist if args.lambda_dist is not None else 0.0,
        bias='Bias' if args.token_bias else'No',
        ema=False,
        randaug='-RandAug' if args.rand_aug else '',
    )
    project_name_suffix_o = project_name_suffix_o + '-Skip' if args.decoder_skip_connection else project_name_suffix_o
    project_name_suffix_o = project_name_suffix_o + '-ConcatPrior' if args.concat_fixed_prompts else project_name_suffix_o

    project_name_suffix = project_name_suffix_o + f"-{phase_tag}-NoDS"
    args.name = args.name + project_name_suffix

    base_logpath = os.path.join(args.cv_dir, args.name)
    logpath = os.path.join(base_logpath, phase_tag)

    if not os.path.exists(logpath):
        os.makedirs(logpath, exist_ok=True)
    save_args(args, logpath, args.config)
    stats_log_path = os.path.join(logpath, 'training_stats.json')

    # 1. Source Train Set
    print(f"Loading Source Train Set from: {args.dataset_json_paths}")
    trainset_source = MetaDataset(
        phase='train',
        dataset=args.dataset,
        num_shots=args.coop_num_shots,
        seed=args.coop_seed,
        num_template=args.num_text_template,
        rand_aug=args.rand_aug,
        few_shot=False,
        dataset_json_paths=args.dataset_json_paths
    )
    if hasattr(trainset_source, 'template'):
        args.prompt_template = trainset_source.template

    # 2. Target Support Set
    target_support_path = getattr(args, "target_support_json", "fixed_support.json")
    if not os.path.exists(target_support_path):
        alt_path = os.path.join(os.path.dirname(args.dataset_json_paths), "fixed_support.json")
        if os.path.exists(alt_path):
            target_support_path = alt_path

    print(f"Loading Target Support Set (Few-Shot) from: {target_support_path}")
    trainset_target_support = None
    if os.path.exists(target_support_path):
        trainset_target_support = MetaDataset(
            phase='train',
            dataset=args.dataset,
            num_shots=5,
            seed=args.coop_seed,
            num_template=args.num_text_template,
            rand_aug=args.rand_aug,
            few_shot=True,
            dataset_json_paths=target_support_path
        )

    # 3. Target Test Set
    print(f"Loading Target Test Set from: {args.dataset_json_paths2}")
    testset_target = MetaDataset(
        phase='test',
        dataset=args.dataset,
        num_shots=args.coop_num_shots,
        seed=args.coop_seed,
        num_template=args.num_text_template,
        rand_aug=args.rand_aug,
        few_shot=False,
        dataset_json_paths=args.dataset_json_paths2
    )

    classnames = {'all': trainset_source.classnames}

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    print("Initializing FECLIP Model...")
    model = FECLIP(dset=trainset_source, classnames=classnames, args=args, model=None, tokenizer=None, few_shot=False)


    if stage in ["fusion", "joint"] and args.load is None:
        suffix_len = len(project_name_suffix)
        original_prefix = args.name[:-suffix_len]
        clip_suffix = project_name_suffix_o + "-clip-NoDS"
        nameclip = original_prefix + clip_suffix
        clip_dir = os.path.join(args.cv_dir, nameclip, "clip")
        if not os.path.exists(clip_dir):
            clip_dir_ds = os.path.join(args.cv_dir, original_prefix + project_name_suffix_o + "-clip", "clip")
            if os.path.exists(clip_dir_ds):
                clip_dir = clip_dir_ds
                print(f"[Auto-Load] Found CLIP dir: {clip_dir}")
        if os.path.exists(clip_dir):
            candidates = glob.glob(os.path.join(clip_dir, "ckpt_best_*.t7"))
            if not candidates:
                candidates = glob.glob(os.path.join(clip_dir, "ckpt_last_*.t7"))
            if candidates:
                candidates.sort(key=os.path.getmtime, reverse=True)
                auto_load_path = candidates[0]
                print(f"[Auto-Load] Loading CLIP weights from {auto_load_path}")
                checkpoint = torch.load(auto_load_path, map_location="cpu")
                state_dict = checkpoint["net"]
                new_state_dict = {
                    (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
                }
                missing_keys, _ = model.load_state_dict(new_state_dict, strict=False)
                print(f"[Auto-Load] Missing keys: {len(missing_keys)}")
            else:
                print(f"[Auto-Load] No .t7 found in {clip_dir}")
        else:
            print(f"[Auto-Load] No clip checkpoint dir at {clip_dir}; training from scratch")

    model.to(device)
    sig = inspect.signature(FSDP)
    use_fsdp = "use_orig_params" in sig.parameters
    if use_fsdp:
        print("[Parallel] Using FSDP")
        mp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        mp = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        fsdp_kwargs = dict(
            auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=2_000_000),
            mixed_precision=mp,
            device_id=device,
            cpu_offload=CPUOffload(offload_params=False),
            use_orig_params=True,
        )
        if "limit_all_gathers" in sig.parameters:
            fsdp_kwargs["limit_all_gathers"] = True
        model = FSDP(model, **fsdp_kwargs)
    else:
        print("[Parallel] FSDP unsupported; falling back to DDP/single GPU")
        if distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            print(f"[Parallel] DDP rank {local_rank}")
        else:
            print(f"[Parallel] Single GPU rank {local_rank}")

    parameter_group = build_optimizer_parameters(args, model)
    optimizer = AdamW(lr=args.lr, weight_decay=args.weight_decay, params=parameter_group)
    if torch.cuda.is_bf16_supported() or use_fsdp:
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        print("[AMP] GradScaler disabled")
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        print("[AMP] GradScaler enabled for FP16 DDP")

    attach_param_names_for_optimizer(model)
    summarize_trainable_params(model, title="FECLIP")

    start_epoch = 0
    if args.load is not None:
        checkpoint = torch.load(args.load, map_location="cpu")
        state_dict = checkpoint["net"]
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(state_dict, strict=False)
        start_epoch = checkpoint["epoch"]
        print(f"Loaded resume model from {args.load}")

    train_sampler_source = DistributedSampler(trainset_source, shuffle=True) if distributed else None
    trainloader_source = torch.utils.data.DataLoader(
        trainset_source, batch_size=args.batch_size, shuffle=(train_sampler_source is None),
        sampler=train_sampler_source, num_workers=args.workers, drop_last=True,
    )
    trainloader_target_support = None
    train_sampler_target = None
    if trainset_target_support is not None:
        bs_target = max(1, args.batch_size // 4)
        train_sampler_target = DistributedSampler(trainset_target_support, shuffle=True) if distributed else None
        trainloader_target_support = torch.utils.data.DataLoader(
            trainset_target_support, batch_size=bs_target, shuffle=(train_sampler_target is None),
            sampler=train_sampler_target, num_workers=args.workers, drop_last=False,
        )
    testloader_target = torch.utils.data.DataLoader(
        testset_target, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers
    )
    evaluator_target = Classification(args, testset_target.idx2label)

    best_auc = 0.0
    total_train_steps = len(trainloader_source) * args.max_epochs
    warmup_steps = int(0.03 * total_train_steps)
    lr_scheduler = None
    if args.lr_scheduler and args.lr_scheduler.lower() == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps
        )

    print("Start Training Loop...")
    for epoch in tqdm(range(start_epoch, args.max_epochs), desc="Current epoch"):
        if distributed and train_sampler_target is not None:
            train_sampler_target.set_epoch(epoch)
        avg_train_loss = train(
            epoch, model, optimizer, scaler, trainloader_source,
            trainloader_target_support, lr_scheduler, device, use_fsdp,
        )
        if (epoch + 1) % args.eval_val_every == 0:
            with torch.no_grad():
                print("--> Testing on Target Domain...")
                stats = test(epoch, model, testloader_target, evaluator_target, args, logpath, device, subset="target")
                auc_target = stats.get("auc", 0.0)
                if is_main_process():
                    log_row = {"epoch": epoch, "train_loss": round(avg_train_loss, 6), **stats}
                    log_row.pop("a_epoch", None)
                    with open(stats_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_row, ensure_ascii=False) + "\n")
                    if auc_target is not None and auc_target > best_auc:
                        best_auc = auc_target
                        save_checkpoint(model, epoch, logpath, args.coop_seed, "best")
                        print(f"[Best] epoch {epoch}: auc={auc_target:.4f}")
        torch.cuda.empty_cache()
    save_checkpoint(model, epoch, logpath, args.coop_seed, "last")


def train(epoch, model, optimizer, scaler, loader_source, loader_target, lr_scheduler, device, use_fsdp):
    model.train()
    loss_accumulator = {"loss_total": 0.0}
    epoch_loss_tracker = 0.0
    num_batches = 0
    log_interval = 20
    model_args = model.module.args if hasattr(model, "module") else getattr(model, "args", None)
    target_every = max(1, int(getattr(model_args, "target_every", 4)) if model_args is not None else 4)
    iter_target = iter(loader_target) if loader_target is not None else None
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    for idx, data_s in tqdm(enumerate(loader_source), total=len(loader_source), desc=f"Epoch {epoch} Training"):
        img_s = data_s[0].to(device, non_blocking=True)
        aug_s = data_s[1].to(device, non_blocking=True)
        lbl_s = data_s[2].to(device, non_blocking=True)
        dom_s = torch.zeros(lbl_s.size(0), dtype=torch.long, device=device)

        use_target = (iter_target is not None) and (idx % target_every == 0)
        if use_target:
            try:
                data_t = next(iter_target)
            except StopIteration:
                iter_target = None
                use_target = False
        if use_target:
            img_t = data_t[0].to(device, non_blocking=True)
            aug_t = data_t[1].to(device, non_blocking=True)
            lbl_t = data_t[2].to(device, non_blocking=True)
            dom_t = torch.ones(lbl_t.size(0), dtype=torch.long, device=device)
            imgs = torch.cat([img_s, img_t], dim=0)
            augs = torch.cat([aug_s, aug_t], dim=0)
            labels = torch.cat([lbl_s, lbl_t], dim=0)
            domains = torch.cat([dom_s, dom_t], dim=0)
        else:
            imgs, augs, labels, domains = img_s, aug_s, lbl_s, dom_s

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            outputs = model([imgs, augs, labels, domains])
            logits_out = None
            if isinstance(outputs, tuple):
                if len(outputs) > 1:
                    logits_out = outputs[1]
                losses = outputs[0]
            else:
                losses = outputs
            if not isinstance(losses, dict):
                loss = losses
                losses = {"loss_total": loss}
            else:
                loss = losses["loss_total"]

        model_args = model.module.args if hasattr(model, "module") else getattr(model, "args", None)
        do_pcgrad = (
            (not use_fsdp) and (model_args is not None)
            and bool(getattr(model_args, "use_margin_weight", False))
        )
        if do_pcgrad and logits_out is not None and isinstance(losses, dict) and "loss_ce" in losses:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            ce_per = F.cross_entropy(
                logits_out.float(), labels, reduction="none",
                label_smoothing=float(getattr(model_args, "label_smoothing", 0.0)),
            )
            ws = float(getattr(model_args, "source_loss_weight", 1.0))
            wt = float(getattr(model_args, "target_loss_weight", 2.0))
            w_mw = _margin_weight_from_logits(
                logits_out,
                min_w=float(getattr(model_args, "mw_min_w", 0.1)),
                max_w=float(getattr(model_args, "mw_max_w", 1.0)),
            )
            groups = []
            for d, scale in [(0, ws), (1, wt)]:
                mask_d = domains == d
                if mask_d.sum().item() < 2:
                    continue
                thr = w_mw[mask_d].median()
                mask_easy = mask_d & (w_mw >= thr)
                mask_hard = mask_d & (w_mw < thr)
                if mask_easy.sum().item() < 2 or mask_hard.sum().item() < 2:
                    groups.append((mask_d, scale))
                else:
                    groups.append((mask_easy, scale))
                    groups.append((mask_hard, scale))
            grads = []
            for gi, (gmask, gscale) in enumerate(groups):
                optimizer.zero_grad(set_to_none=False)
                loss_g = (ce_per[gmask] * w_mw[gmask]).mean() * gscale
                loss_g.backward(retain_graph=(gi != len(groups) - 1))
                grads.append(_pcgrad_flatten_grads(trainable_params))
            g_final = _pcgrad_multi(grads, seed=int(epoch * 100000 + idx))
            for p in trainable_params:
                if p.grad is not None:
                    p.grad.zero_()
            if g_final is not None:
                _pcgrad_assign_flat_grads(trainable_params, g_final)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        else:
            scaler.scale(loss).backward()
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        epoch_loss_tracker += loss.item()
        num_batches += 1
        for key, val in losses.items():
            if val is None:
                continue
            v_item = val.mean().item() if isinstance(val, torch.Tensor) else val
            loss_accumulator[key] = loss_accumulator.get(key, 0.0) + v_item
        if (idx + 1) % log_interval == 0:
            log_str = f"Epoch {epoch} | Batch {(idx + 1)}/{len(loader_source)}"
            for key in sorted(loss_accumulator.keys()):
                log_str += f" | {key}: {loss_accumulator[key] / log_interval:.4f}"
            if is_main_process():
                print(log_str)
            loss_accumulator = {}

    if lr_scheduler is not None:
        lr_scheduler.step()
    if is_main_process():
        print(f"Epoch {epoch} finished.")
    return epoch_loss_tracker / max(1, num_batches)


def test(epoch, model, testloader, evaluator, args, logpath, device, subset):
    evaluator.reset()
    model.eval()
    all_probs, all_labels = [], []
    for data in tqdm(testloader, desc="Testing"):
        data = [d.to(device) for d in data]
        with torch.no_grad():
            _, predictions = model(data, subset=subset.lower())
        preds_cpu = predictions.cpu()
        evaluator.process(preds_cpu, data[-1].cpu())
        all_probs.append(torch.softmax(preds_cpu.float(), dim=-1))
        all_labels.append(data[-1].detach().cpu().long())

    stats = evaluator.evaluate()
    stats["a_epoch"] = epoch
    try:
        probs_np = torch.cat(all_probs, dim=0).numpy()
        labels_np = torch.cat(all_labels, dim=0).numpy()
        extra = _compute_auc_eer(probs_np, labels_np)
        if extra["auc"] is not None:
            stats["auc"] = round(float(extra["auc"]), 6)
        if extra["eer"] is not None:
            stats["eer"] = round(float(extra["eer"]), 6)
    except Exception as e:
        print(f"[Warn] Failed to compute AUC/EER: {e}")

    result = " | ".join(
        f"{k} {round(v, 4) if isinstance(v, (int, float)) else v}" for k, v in stats.items()
    )
    if is_main_process():
        print(f"Test Epoch: {epoch}")
        print(f"{result} | {args.name}")
    return stats


def _num_params(t):
    return sum(p.numel() for p in t)


def _pretty_num(n):
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def summarize_trainable_params(model, title="Model"):
    if not is_main_process():
        return
    print(f"\n===== [Trainable Params] {title} =====")
    total = _num_params(model.parameters())
    trainable = _num_params(p for p in model.parameters() if p.requires_grad)
    print(f"Total: {_pretty_num(total)} | Trainable: {_pretty_num(trainable)} | Frozen: {_pretty_num(total - trainable)}")
    fusion_n = other_n = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "fusion" in name.lower() or "expert" in name.lower():
            fusion_n += p.numel()
        else:
            other_n += p.numel()
    print(f"   Fusion: {_pretty_num(fusion_n)}")
    print(f"   Other:  {_pretty_num(other_n)}")


def attach_param_names_for_optimizer(model):
    for name, p in model.named_parameters():
        try:
            setattr(p, "_param_name", name)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
