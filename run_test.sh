#!/bin/bash
# ============================================================
# FE-CLIP Test Script
# 用法: bash run_test.sh [logpath] [test_json]
#   logpath:  不传则自动找最新的 WEIGHT_ROOT/FE-CLIP_*/fusion_Cert
#   test_json: 不传则用 DEFAULT_TEST_JSON / TEST_JSON 环境变量
#
# 预训练权重托管于 Hugging Face:
#   https://huggingface.co/willingSZU/Few-Shot-DPAD
# 下载示例:
#   huggingface-cli download willingSZU/Few-Shot-DPAD --local-dir ./checkpoints
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU=${GPU:-0}
CKTYPE=${CKTYPE:-best}
# 默认指向本地下载的 HF 权重目录（可用 WEIGHT_ROOT 覆盖）
WEIGHT_ROOT=${WEIGHT_ROOT:-"${SCRIPT_DIR}/../checkpoints"}

# 默认测试 JSON（请通过环境变量设置，勿写本机绝对路径）
# 示例: DEFAULT_TEST_JSON=/path/to/test.json bash run_test.sh
DEFAULT_TEST_JSON=${DEFAULT_TEST_JSON:-${TEST_JSON:-}}

LOGPATH=${1:-""}
TEST_JSON=${2:-$DEFAULT_TEST_JSON}

# 自动找最近的 logpath
if [ -z "$LOGPATH" ]; then
    LOGPATH=$(ls -dt "$WEIGHT_ROOT"/FE-CLIP_*/fusion_Cert 2>/dev/null | head -1)
    if [ -z "$LOGPATH" ]; then
        echo "未找到 logpath，请手动指定: bash run_test.sh <logpath> [test_json]"
        echo "或先从 Hugging Face 下载权重到 WEIGHT_ROOT:"
        echo "  https://huggingface.co/willingSZU/Few-Shot-DPAD"
        exit 1
    fi
    echo "自动选择最近的 logpath: $LOGPATH"
fi

if [ -z "$TEST_JSON" ]; then
    echo "Error: please provide test_json."
    echo "  bash run_test.sh <logpath> /path/to/test.json"
    echo "或: DEFAULT_TEST_JSON=/path/to/test.json bash run_test.sh <logpath>"
    exit 1
fi

echo "============================================"
echo " FE-CLIP Testing"
echo "============================================"
echo " GPU:       $GPU"
echo " Logpath:   $LOGPATH"
echo " Ckpt:      $CKTYPE"
echo " Test JSON: $TEST_JSON"
echo "============================================"

CUDA_VISIBLE_DEVICES=$GPU python test.py \
    --logpath "$LOGPATH" \
    --test-dataset-json-paths "$TEST_JSON"
