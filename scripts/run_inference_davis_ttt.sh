#!/bin/bash
set -e

# 激活 conda 环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ttt_sam

# 切换到项目目录
cd /root/autodl-tmp/ttt_rr
export PYTHONPATH=.

# 数据集路径
DATASET_ROOT="/root/autodl-tmp/dataset_480p"
BASE_VIDEO_DIR="${DATASET_ROOT}/DAVIS/JPEGImages/480p"
INPUT_MASK_DIR="${DATASET_ROOT}/DAVIS/Annotations/480p"
VIDEO_LIST_FILE="${DATASET_ROOT}/DAVIS/ImageSets/2017/val.txt"

# 模型配置（使用训练过的checkpoint）
SAM2_CFG="sam2/configs/sam2.1/sam2_ttt_inference_b+.yaml"
SAM2_CHECKPOINT="/root/autodl-tmp/runs/sam_ttt_davis/20251223_180637/checkpoints/checkpoint.pt"

# 输出路径
OUTPUT_MASK_DIR="/root/autodl-tmp/output_davis_val_ttt_new"

echo "Running TTT inference on DAVIS 2017 val set..."
echo "Config: $SAM2_CFG"
echo "Checkpoint: $SAM2_CHECKPOINT"
echo "Output: $OUTPUT_MASK_DIR"
echo "Dataset: $BASE_VIDEO_DIR"

# 运行推理
python tools/vos_inference_ttt.py \
    --sam2_cfg $SAM2_CFG \
    --sam2_checkpoint $SAM2_CHECKPOINT \
    --base_video_dir $BASE_VIDEO_DIR \
    --input_mask_dir $INPUT_MASK_DIR \
    --video_list_file $VIDEO_LIST_FILE \
    --output_mask_dir $OUTPUT_MASK_DIR

echo "✅ Inference complete! Results saved to $OUTPUT_MASK_DIR"
