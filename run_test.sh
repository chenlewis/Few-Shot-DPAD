#!/bin/bash
# ============================================================
# FE-CLIP Test Script
# Usage: bash run_test.sh [logpath] [test_json]
#   logpath:   If omitted, auto-select the latest WEIGHT_ROOT/FE-CLIP_*/fusion_Cert
#   test_json: If omitted, use DEFAULT_TEST_JSON / TEST_JSON env vars
#
# Pretrained weights are hosted on Hugging Face:
#   https://huggingface.co/willingSZU/Few-Shot-DPAD
# Download example:
#   huggingface-cli download willingSZU/Few-Shot-DPAD --local-dir ./checkpoints
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU=${GPU:-0}
CKTYPE=${CKTYPE:-best}
# Default to locally downloaded HF weights (override with WEIGHT_ROOT)
WEIGHT_ROOT=${WEIGHT_ROOT:-"${SCRIPT_DIR}/../checkpoints"}

# Default test JSON (set via env var; do not hard-code machine-specific absolute paths)
# Example: DEFAULT_TEST_JSON=/path/to/test.json bash run_test.sh
DEFAULT_TEST_JSON=${DEFAULT_TEST_JSON:-${TEST_JSON:-}}

LOGPATH=${1:-""}
TEST_JSON=${2:-$DEFAULT_TEST_JSON}

# Auto-select the most recent logpath
if [ -z "$LOGPATH" ]; then
    LOGPATH=$(ls -dt "$WEIGHT_ROOT"/FE-CLIP_*/fusion_Cert 2>/dev/null | head -1)
    if [ -z "$LOGPATH" ]; then
        echo "logpath not found. Please specify: bash run_test.sh <logpath> [test_json]"
        echo "Or download weights from Hugging Face into WEIGHT_ROOT first:"
        echo "  https://huggingface.co/willingSZU/Few-Shot-DPAD"
        exit 1
    fi
    echo "Auto-selected latest logpath: $LOGPATH"
fi

if [ -z "$TEST_JSON" ]; then
    echo "Error: please provide test_json."
    echo "  bash run_test.sh <logpath> /path/to/test.json"
    echo "Or: DEFAULT_TEST_JSON=/path/to/test.json bash run_test.sh <logpath>"
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
