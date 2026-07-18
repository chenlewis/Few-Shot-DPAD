import argparse

# Override via --data_dir / config; keep empty for portability.
DATA_FOLDER = ""
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Training stage
parser.add_argument("--train_stage", type=str, default="fusion", choices=["clip", "fusion", "joint"])
parser.add_argument("--use_swei_fusion", default=False)

# Experiment / IO
# Checkpoints: https://huggingface.co/willingSZU/Few-Shot-DPAD
parser.add_argument("--config", default="configs/llava/zero-shot/spoof_3class.yml")
parser.add_argument("--dataset", default="spoof_detection")
parser.add_argument("--data_dir", default="")
parser.add_argument("--logpath", default=None)
parser.add_argument("--cv_dir", default="./checkpoints")
parser.add_argument("--name", default="temp")
parser.add_argument("--load", default=None)

# Optimization
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--test_batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--lora_lr", type=float, default=2e-5)
parser.add_argument("--weight_decay", type=float, default=5e-5)
parser.add_argument("--eval_val_every", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=20)
parser.add_argument("--lr_scheduler", type=str, default=None)
parser.add_argument("--label_smoothing", type=float, default=0.1)
parser.add_argument("--device", default="cuda")
parser.add_argument("--seed", default=42, type=int)

# Distributed
parser.add_argument("--world_size", default=1, type=int)
parser.add_argument("--dist_url", default="env://")
parser.add_argument("--distributed", action="store_true")
parser.add_argument("--local_rank", type=int, default=0)

# LoRA / prompt
parser.add_argument("--lora_rank", default=8, type=int)
parser.add_argument("--lora_alpha", default=16, type=int)
parser.add_argument("--lora_dropout", default=0.1, type=float)
parser.add_argument("--freeze_vit", action="store_true")
parser.add_argument("--naive_decoding", action="store_true")
parser.add_argument("--v_lora_start", default=6, type=int)
parser.add_argument("--v_lora_end", default=12, type=int)
parser.add_argument("--n_ctx", type=int, default=16)
parser.add_argument("--ctx_init", type=str, default="")
parser.add_argument("--prompt_template", type=str, default="a photo of a {}.")
parser.add_argument("--num_text_template", type=int, default=11)
parser.add_argument("--rand_aug", action="store_true")

# Few-shot meta
parser.add_argument("--coop_seed", default=1, type=int)
parser.add_argument("--coop_num_shots", default=16, type=int)

# Eval options
parser.add_argument("--test_per_class_result", action="store_true")
parser.add_argument("--test_compute_cmat", action="store_true")
parser.add_argument("--topk", type=int, default=1)

# Naming-only legacy fields (kept for experiment directory naming / resume path matching)
parser.add_argument("--num_decoder_layers", default=1, type=int)
parser.add_argument("--prompt_type", type=str, default="suffix")
parser.add_argument("--num_prior_tokens", default=100, type=int)
parser.add_argument("--llm_prompt_depth", type=int, default=9)
parser.add_argument("--num_llm_prompts", default=16, type=int)
parser.add_argument("--num_text_ctx", type=int, default=4)
parser.add_argument("--num_vis_ctx", type=int, default=4)
parser.add_argument("--distillation_type", default=None, type=str)
parser.add_argument("--lambda_dist", type=float, default=1.0)
parser.add_argument("--token_bias", action="store_true")
parser.add_argument("--decoder_skip_connection", action="store_true")
parser.add_argument("--concat_fixed_prompts", action="store_true")
parser.add_argument("--model_base", type=str, default="")

# FE-CLIP
parser.add_argument("--use_margin_weight", action="store_true")
parser.add_argument("--use_aug_loss", action="store_true")
parser.add_argument("--dataset_json_paths", type=str, default=None)
parser.add_argument("--target_support_json", type=str, default="fixed_support.json")
parser.add_argument("--dataset_json_paths2", type=str, default=None)
parser.add_argument("--source_loss_weight", type=float, default=1.0)
parser.add_argument("--target_loss_weight", type=float, default=2.0)
parser.add_argument("--mw_min_w", type=float, default=0.1)
parser.add_argument("--mw_max_w", type=float, default=1.0)
parser.add_argument("--target_every", type=int, default=4)
parser.add_argument("--clip_model_path", type=str, default=None)
# Local ViT forensics expert ckpt; override with --vit_ckpt_path
parser.add_argument("--vit_ckpt_path", type=str, default="")
