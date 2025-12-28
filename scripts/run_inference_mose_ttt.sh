#!/bin/bash
set -e
set -o pipefail

# 激活 conda 环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ttt_sam

# 切换到项目目录
cd /root/autodl-tmp/ttt_rr
export PYTHONPATH=.

# 数据集路径（MOSE train 中的 20% 作为验证）
DATASET_ROOT="/root/autodl-tmp/data_set/MOSE/train/train"
BASE_VIDEO_DIR="${DATASET_ROOT}/JPEGImages"
INPUT_MASK_DIR="${DATASET_ROOT}/Annotations"
VIDEO_LIST_FILE="/root/autodl-tmp/ttt_rr/training/assets/MOSE_val20_list.txt"

# 模型配置（使用训练过的checkpoint）
SAM2_CFG="sam2/configs/sam2.1/sam2_ttt_inference_l.yaml"
SAM2_CHECKPOINT="/root/autodl-tmp/runs/sam_ttt_davis/20251225_230622/checkpoints/checkpoint.pt"
CHECKPOINT_DIR="$(dirname "$SAM2_CHECKPOINT")"
RUN_DIR="$(dirname "$CHECKPOINT_DIR")"
LOG_FILE="${RUN_DIR}/inference.log"

# 输出路径
OUTPUT_MASK_DIR="/root/autodl-tmp/output_mose_val/output_mose_val_ttt"

echo "Running TTT inference on MOSE 20% val split..."
echo "Config: $SAM2_CFG"
echo "Checkpoint: $SAM2_CHECKPOINT"
echo "Output: $OUTPUT_MASK_DIR"
echo "Dataset: $BASE_VIDEO_DIR"
echo "Log: $LOG_FILE"

# 运行推理
python tools/vos_inference_ttt.py \
    --sam2_cfg $SAM2_CFG \
    --sam2_checkpoint $SAM2_CHECKPOINT \
    --base_video_dir $BASE_VIDEO_DIR \
    --input_mask_dir $INPUT_MASK_DIR \
    --video_list_file $VIDEO_LIST_FILE \
    --output_mask_dir $OUTPUT_MASK_DIR \
    2>&1 | tee "$LOG_FILE"

echo "✅ Inference complete! Results saved to $OUTPUT_MASK_DIR"
