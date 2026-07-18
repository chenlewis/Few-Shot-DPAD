# FE-CLIP: Forensics-Expert CLIP with Prompt Learning

A multimodal few-shot cross-domain image classification framework based on CLIP + ViT Forensic Expert, with PCGrad gradient projection and Margin Weight uncertainty weighting.

## Installation

```bash
conda activate llamp
# Or create a new environment:
# conda create -n feclip python=3.11 pip && conda activate feclip
pip install -r requirements.txt
```

Main dependencies: `torch`, `transformers`, `peft`, `torchvision`

## Pretrained Weights

Released weights are hosted on Hugging Face:

- **Model**: [https://huggingface.co/willingSZU/Few-Shot-DPAD](https://huggingface.co/willingSZU/Few-Shot-DPAD)

```bash
# Download to a local checkpoints directory
huggingface-cli download willingSZU/Few-Shot-DPAD --local-dir ./checkpoints
# Or point WEIGHT_ROOT to the download directory
export WEIGHT_ROOT=./checkpoints
```

## Project Structure

```
Few-Shot-DPAD/                   # Code
├── train.py / test.py
├── run_train.sh / run_test.sh
├── models/ / data/ / utils/ / configs/
checkpoints/                     # Weights (downloaded from Hugging Face; kept separate from code)
└── FE-CLIP_<dataset>_.../fusion_Cert/
    ├── ckpt_best_*.t7
    ├── ckpt_last_*.t7
    └── args_all.yaml
```

Default weight directory: `../checkpoints` (relative to this repo).  
Override via the `WEIGHT_ROOT` environment variable or the `--cv_dir` argument.

## Training

Data paths must be provided via environment variables (no machine-specific absolute paths are committed):

```bash
SOURCE_JSON=/path/to/source.json \
TARGET_SUPPORT=/path/to/target_support.json \
TARGET_TEST=/path/to/target_test.json \
bash run_train.sh

# Custom parameters
GPUS=0 DATASET=spoof_detection TRAIN_STAGE=fusion NUM_SHOTS=5 \
SOURCE_JSON=/path/to/source.json \
TARGET_SUPPORT=/path/to/support.json \
TARGET_TEST=/path/to/test.json \
bash run_train.sh

# Direct invocation
python train.py \
    --config configs/llava/zero-shot/spoof_3class.yml \
    --dataset spoof_detection --train_stage fusion \
    --lr 2e-5 --batch_size 4 --max_epochs 20 \
    --use_margin_weight \
    --cv_dir ./checkpoints \
    --dataset_json_paths /path/to/source.json \
    --target_support_json /path/to/target_support.json \
    --dataset_json_paths2 /path/to/target_test.json
```

### Training Stages

| Stage | Description |
|-------|-------------|
| `clip` | Train CLIP branch only (Prompt + LoRA) |
| `fusion` | Train fusion module + LoRA co-tuning (default) |
| `joint` | Jointly train all modules |

## Testing

```bash
# Specify weight directory and test set
bash run_test.sh ./checkpoints/FE-CLIP_CERT_25Shot_fusion_lr2e-5_s1/fusion_Cert /path/to/test.json

# Or use environment variables
DEFAULT_TEST_JSON=/path/to/test.json bash run_test.sh <WEIGHT_DIR>/fusion_Cert

# If logpath is omitted, the latest experiment under WEIGHT_ROOT is selected (test_json is still required)
DEFAULT_TEST_JSON=/path/to/test.json bash run_test.sh
```

## Key Features

- **PCGrad**: Project gradients grouped by domain × difficulty to reduce conflicts
- **Margin Weight**: Top-2 entropy uncertainty weighting
- **Source + Target mixed training**: Few-shot cross-domain generalization
- **Pure PyTorch**: FSDP/DDP, no DeepSpeed
- **Lightweight checkpoints**: Save LoRA + Fusion parameters only
