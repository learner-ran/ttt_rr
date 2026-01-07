#!/bin/bash
set -e
set -o pipefail

# 激活 conda 环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ttt_sam

# 切换到项目目录
cd /root/autodl-tmp/ttt_rr
export PYTHONPATH=.

# 数据集路径（与 run_inference_mose_ttt.sh 一致）
DATASET_ROOT="/root/autodl-tmp/data_set/MOSE/train/train"
BASE_VIDEO_DIR="${DATASET_ROOT}/JPEGImages"
INPUT_MASK_DIR="${DATASET_ROOT}/Annotations"
VIDEO_LIST_FILE="/root/autodl-tmp/ttt_rr/training/assets/MOSE_val20_list.txt"

# 模型配置（SAM 2.1 Large）
SAM2_CFG="sam2/configs/sam2.1/sam2.1_hiera_l_no_ttt.yaml"
SAM2_CHECKPOINT="/root/autodl-tmp/ttt_rr/checkpoints/sam2.1_hiera_large.pt"

# 输出路径
OUTPUT_MASK_DIR="${OUTPUT_MASK_DIR:-/root/autodl-tmp/output_mose_val/output_mose_test_large}"
LOG_FILE="${OUTPUT_MASK_DIR}/inference.log"
mkdir -p "$OUTPUT_MASK_DIR"

echo "Running SAM 2.1 large inference on MOSE test set..."
echo "Config: $SAM2_CFG"
echo "Checkpoint: $SAM2_CHECKPOINT"
echo "Output: $OUTPUT_MASK_DIR"
echo "Dataset: $BASE_VIDEO_DIR"
echo "Log: $LOG_FILE"

# 运行推理
python tools/vos_inference.py \
    --sam2_cfg $SAM2_CFG \
    --sam2_checkpoint $SAM2_CHECKPOINT \
    --base_video_dir $BASE_VIDEO_DIR \
    --input_mask_dir $INPUT_MASK_DIR \
    --video_list_file $VIDEO_LIST_FILE \
    --output_mask_dir $OUTPUT_MASK_DIR \
    2>&1 | tee "$LOG_FILE"

echo "✅ Inference complete! Results saved to $OUTPUT_MASK_DIR"
