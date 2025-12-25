#!/bin/bash
set -e
set -o pipefail

# Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ttt_sam

cd /root/autodl-tmp/ttt_rr
export PYTHONPATH=.

# DAVIS full-resolution paths
DATASET_ROOT="/root/autodl-tmp/data_set/DAVIS-2017-trainval/DAVIS"
BASE_VIDEO_DIR="${DATASET_ROOT}/JPEGImages/Full-Resolution"
INPUT_MASK_DIR="${DATASET_ROOT}/Annotations/Full-Resolution"
VIDEO_LIST_FILE="${DATASET_ROOT}/ImageSets/2017/val.txt"

# Original SAM 2.1 large checkpoint
SAM2_CFG="sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT="/root/autodl-tmp/ttt_rr/checkpoints/sam2.1_hiera_large.pt"

# Output
OUTPUT_MASK_DIR="${OUTPUT_MASK_DIR:-/root/autodl-tmp/output_davis_val/output_davis_val_large}"
LOG_FILE="${OUTPUT_MASK_DIR}/inference.log"
mkdir -p "$OUTPUT_MASK_DIR"

echo "Running SAM 2.1 large inference on DAVIS 2017 val set..."
echo "Config: $SAM2_CFG"
echo "Checkpoint: $SAM2_CHECKPOINT"
echo "Output: $OUTPUT_MASK_DIR"
echo "Dataset: $BASE_VIDEO_DIR"
echo "Log: $LOG_FILE"

python tools/vos_inference.py \
    --sam2_cfg $SAM2_CFG \
    --sam2_checkpoint $SAM2_CHECKPOINT \
    --base_video_dir $BASE_VIDEO_DIR \
    --input_mask_dir $INPUT_MASK_DIR \
    --video_list_file $VIDEO_LIST_FILE \
    --output_mask_dir $OUTPUT_MASK_DIR \
    2>&1 | tee "$LOG_FILE"

echo "✅ Inference complete! Results saved to $OUTPUT_MASK_DIR"
