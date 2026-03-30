#!/bin/bash
set -e
set -o pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ttt_sam

cd /root/autodl-tmp/ttt_rr
export PYTHONPATH=.

# Semi-supervised BDD100K MOTS val split in DAVIS-style single-PNG format.
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/data_set/BDD100K_MOTS_semi/val}"
BASE_VIDEO_DIR="${BASE_VIDEO_DIR:-${DATASET_ROOT}/JPEGImages}"
INPUT_MASK_DIR="${INPUT_MASK_DIR:-${DATASET_ROOT}/Annotations}"
VIDEO_LIST_FILE="${VIDEO_LIST_FILE:-${DATASET_ROOT}/val_videos.txt}"

SAM2_CFG="${SAM2_CFG:-sam2/configs/sam2.1/sam2.1_hiera_l_no_ttt.yaml}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-/root/autodl-tmp/ttt_rr/checkpoints/sam2.1_hiera_large.pt}"

OUTPUT_MASK_DIR="${OUTPUT_MASK_DIR:-/root/autodl-tmp/output_bdd_semi_val/output_bdd_semi_large}"
LOG_FILE="${OUTPUT_MASK_DIR}/inference.log"
RUN_EVAL_AFTER_INFERENCE="${RUN_EVAL_AFTER_INFERENCE:-1}"
mkdir -p "$OUTPUT_MASK_DIR"

echo "Running SAM 2.1 large inference on semi-supervised BDD100K MOTS val..."
echo "Config: $SAM2_CFG"
echo "Checkpoint: $SAM2_CHECKPOINT"
echo "Dataset: $BASE_VIDEO_DIR"
echo "Input masks: $INPUT_MASK_DIR"
echo "Video list: $VIDEO_LIST_FILE"
echo "Output: $OUTPUT_MASK_DIR"
echo "Log: $LOG_FILE"
echo "Run eval after inference: $RUN_EVAL_AFTER_INFERENCE"

python tools/vos_inference.py \
    --sam2_cfg "$SAM2_CFG" \
    --sam2_checkpoint "$SAM2_CHECKPOINT" \
    --base_video_dir "$BASE_VIDEO_DIR" \
    --input_mask_dir "$INPUT_MASK_DIR" \
    --video_list_file "$VIDEO_LIST_FILE" \
    --output_mask_dir "$OUTPUT_MASK_DIR" \
    2>&1 | tee "$LOG_FILE"

echo "Inference complete. Results saved to $OUTPUT_MASK_DIR"

if [[ "$RUN_EVAL_AFTER_INFERENCE" == "1" ]]; then
    echo "Starting evaluation with predictions from $OUTPUT_MASK_DIR"
    GT_ROOT="${INPUT_MASK_DIR}" \
    PRED_ROOT="${OUTPUT_MASK_DIR}" \
    bash /root/autodl-tmp/ttt_rr/scripts/run_eval_bdd_large_metrics.sh
fi
