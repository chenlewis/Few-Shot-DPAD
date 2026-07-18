# FE-CLIP: Forensics-Expert CLIP with Prompt Learning

基于 CLIP + ViT Forensic Expert 的多模态 Few-shot 跨域图像分类框架，使用 PCGrad 梯度投影 + Margin Weight 不确定性加权。

## 环境安装

```bash
conda activate llamp
# 或新建环境:
# conda create -n feclip python=3.11 pip && conda activate feclip
pip install -r requirements.txt
```

主要依赖: `torch`, `transformers`, `peft`, `torchvision`

## 预训练权重

发布权重托管在 Hugging Face：

- **Model**: [https://huggingface.co/willingSZU/Few-Shot-DPAD](https://huggingface.co/willingSZU/Few-Shot-DPAD)

```bash
# 下载到本地 checkpoints 目录
huggingface-cli download willingSZU/Few-Shot-DPAD --local-dir ./checkpoints
# 或设置 WEIGHT_ROOT 指向下载目录
export WEIGHT_ROOT=./checkpoints
```

## 项目结构

```
Few-Shot-DPAD/                   # 代码
├── train.py / test.py
├── run_train.sh / run_test.sh
├── models/ / data/ / utils/ / configs/
checkpoints/                     # 权重（从 Hugging Face 下载，与代码分离）
└── FE-CLIP_<dataset>_.../fusion_Cert/
    ├── ckpt_best_*.t7
    ├── ckpt_last_*.t7
    └── args_all.yaml
```

默认权重目录: `../checkpoints`（相对本仓库）  
可通过环境变量 `WEIGHT_ROOT` 或参数 `--cv_dir` 覆盖。

## 训练

数据路径需通过环境变量传入（仓库中不包含本机绝对路径）：

```bash
SOURCE_JSON=/path/to/source.json \
TARGET_SUPPORT=/path/to/target_support.json \
TARGET_TEST=/path/to/target_test.json \
bash run_train.sh

# 自定义参数
GPUS=0 DATASET=spoof_detection TRAIN_STAGE=fusion NUM_SHOTS=5 \
SOURCE_JSON=/path/to/source.json \
TARGET_SUPPORT=/path/to/support.json \
TARGET_TEST=/path/to/test.json \
bash run_train.sh

# 直接调用
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

### 训练阶段

| Stage | 说明 |
|-------|------|
| `clip` | 仅训练 CLIP 分支 (Prompt + LoRA) |
| `fusion` | 训练融合模块 + LoRA Co-tuning（默认） |
| `joint` | 联合训练所有模块 |

## 测试

```bash
# 指定权重目录与测试集
bash run_test.sh ./checkpoints/FE-CLIP_CERT_25Shot_fusion_lr2e-5_s1/fusion_Cert /path/to/test.json

# 或用环境变量
DEFAULT_TEST_JSON=/path/to/test.json bash run_test.sh <WEIGHT_DIR>/fusion_Cert

# 不传 logpath 则自动选 WEIGHT_ROOT 下最新实验（仍需提供 test_json）
DEFAULT_TEST_JSON=/path/to/test.json bash run_test.sh
```

## 核心特性

- **PCGrad**: 按 domain × difficulty 分组投影梯度，减少梯度冲突
- **Margin Weight**: Top-2 熵不确定性加权
- **Source + Target 混合训练**: Few-shot 跨域泛化
- **纯 PyTorch**: FSDP/DDP，无 DeepSpeed
- **轻量 Checkpoint**: 仅保存 LoRA + Fusion 参数
