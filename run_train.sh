#!/bin/bash
# ============================================================
# FE-CLIP Training Script
# 使用 PCGrad + Margin Weight 的 Few-shot 跨域训练
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 基础配置 ----
GPUS=${GPUS:-0}                    # GPU IDs
DATASET=${DATASET:-spoof_detection}
CONFIG=${CONFIG:-configs/llava/zero-shot/spoof_3class.yml}
SEED=${SEED:-1}

# ---- 训练超参 ----
LR=${LR:-2e-5}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_EPOCHS=${MAX_EPOCHS:-50}
EVAL_EVERY=${EVAL_EVERY:-1}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-96}

# ---- Stage 选择 ----
# clip:   只用 CLIP 分支训练
# fusion: 训练 CLIP + Forensics Expert 融合（默认）
# joint:  联合训练
TRAIN_STAGE=${TRAIN_STAGE:-fusion}

# ---- Few-shot 配置 ----
NUM_SHOTS=${NUM_SHOTS:-5}
NUM_PRIOR=${NUM_PRIOR:-100}
LLM_DEPTH=${LLM_DEPTH:-9}
LLM_PROMPTS=${LLM_PROMPTS:-32}
TEXT_CTX=${TEXT_CTX:-4}
VIS_CTX=${VIS_CTX:-4}

# ---- 数据路径（请通过环境变量覆盖，勿提交本机绝对路径）----
# 示例:
#   SOURCE_JSON=/path/to/source.json \
#   TARGET_SUPPORT=/path/to/target_support.json \
#   TARGET_TEST=/path/to/target_test.json \
#   bash run_train.sh
SOURCE_JSON=${SOURCE_JSON:-}
TARGET_SUPPORT=${TARGET_SUPPORT:-}
TARGET_TEST=${TARGET_TEST:-}

if [ -z "$SOURCE_JSON" ] || [ -z "$TARGET_SUPPORT" ] || [ -z "$TARGET_TEST" ]; then
    echo "Error: please set SOURCE_JSON, TARGET_SUPPORT, and TARGET_TEST."
    echo "Example:"
    echo "  SOURCE_JSON=/path/to/source.json \\"
    echo "  TARGET_SUPPORT=/path/to/support.json \\"
    echo "  TARGET_TEST=/path/to/test.json \\"
    echo "  bash run_train.sh"
    exit 1
fi

# ---- 其他 ----
DISTILL_TYPE=${DISTILL_TYPE:-soft}
LAMBDA_DIST=${LAMBDA_DIST:-1.0}
USE_MARGIN_WEIGHT=${USE_MARGIN_WEIGHT:-True}   # PCGrad margin weight

# ---- 权重保存目录 ----
# 预训练/发布权重托管于 Hugging Face:
#   https://huggingface.co/willingSZU/Few-Shot-DPAD
# 下载示例:
#   huggingface-cli download willingSZU/Few-Shot-DPAD --local-dir ./checkpoints
WEIGHT_ROOT=${WEIGHT_ROOT:-"${SCRIPT_DIR}/../checkpoints"}

# ---- 自动生成实验名 ----
EXP_NAME="FE-CLIP_${DATASET}_${NUM_SHOTS}Shot_${TRAIN_STAGE}_lr${LR}_s${SEED}"

echo "============================================"
echo " FE-CLIP Training"
echo "============================================"
echo " GPUs:        $GPUS"
echo " Dataset:     $DATASET"
echo " Stage:       $TRAIN_STAGE"
echo " Shots:       $NUM_SHOTS"
echo " LR:          $LR"
echo " Epochs:      $MAX_EPOCHS"
echo " Batch size:  $BATCH_SIZE"
echo " Experiment:  $EXP_NAME"
echo " Weight dir:  $WEIGHT_ROOT"
echo "============================================"

CUDA_VISIBLE_DEVICES=$GPUS python train.py \
    --config "$CONFIG" \
    --dataset "$DATASET" \
    --train_stage "$TRAIN_STAGE" \
    --coop_num_shots "$NUM_SHOTS" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --max_epochs "$MAX_EPOCHS" \
    --eval_val_every "$EVAL_EVERY" \
    --coop_seed "$SEED" \
    --num_prior_tokens "$NUM_PRIOR" \
    --llm_prompt_depth "$LLM_DEPTH" \
    --num_llm_prompts "$LLM_PROMPTS" \
    --num_text_ctx "$TEXT_CTX" \
    --num_vis_ctx "$VIS_CTX" \
    --distillation_type "$DISTILL_TYPE" \
    --lambda_dist "$LAMBDA_DIST" \
    --use_margin_weight \
    --dataset_json_paths "$SOURCE_JSON" \
    --target_support_json "$TARGET_SUPPORT" \
    --dataset_json_paths2 "$TARGET_TEST" \
    --cv_dir "$WEIGHT_ROOT" \
    --name "$EXP_NAME"

echo "Done. Checkpoint saved under ${WEIGHT_ROOT}/${EXP_NAME}/"
