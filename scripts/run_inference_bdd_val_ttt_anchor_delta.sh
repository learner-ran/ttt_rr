#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ttt_rr
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ttt_sam

export PYTHONPATH=.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ASYNC_LOADING_FRAMES="${ASYNC_LOADING_FRAMES:-1}"
export NUM_PROCESSES="${NUM_PROCESSES:-20}"

DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/data_set/BDD100K_MOTS_semi/val}"
BASE_VIDEO_DIR="${BASE_VIDEO_DIR:-${DATASET_ROOT}/JPEGImages}"
INPUT_MASK_DIR="${INPUT_MASK_DIR:-${DATASET_ROOT}/Annotations}"
VIDEO_LIST_FILE="${VIDEO_LIST_FILE:-${DATASET_ROOT}/val_videos.txt}"

SAM2_CFG="${SAM2_CFG:-sam2/configs/sam2.1/sam2_ttt_inference_l_restricted_anchor_delta.yaml}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-/root/autodl-tmp/ttt_rr/logs/sam_ttt_bdd_restricted_anchor_delta_run1/checkpoints/checkpoint.pt}"
OUTPUT_MASK_DIR="${OUTPUT_MASK_DIR:-/root/autodl-tmp/output_bdd_semi_val/bdd_restricted_anchor_delta_ttt}"
LOG_FILE="${OUTPUT_MASK_DIR}/inference.log"
RUN_EVAL_AFTER_INFERENCE="${RUN_EVAL_AFTER_INFERENCE:-1}"

mkdir -p "$OUTPUT_MASK_DIR"

echo "Running BDD val inference with anchor/delta TTT enabled..."
echo "Config: $SAM2_CFG"
echo "Checkpoint: $SAM2_CHECKPOINT"
echo "Dataset: $BASE_VIDEO_DIR"
echo "Input masks: $INPUT_MASK_DIR"
echo "Video list: $VIDEO_LIST_FILE"
echo "Output: $OUTPUT_MASK_DIR"
echo "Log: $LOG_FILE"

python tools/vos_inference_ttt.py \
    --sam2_cfg "$SAM2_CFG" \
    --sam2_checkpoint "$SAM2_CHECKPOINT" \
    --base_video_dir "$BASE_VIDEO_DIR" \
    --input_mask_dir "$INPUT_MASK_DIR" \
    --video_list_file "$VIDEO_LIST_FILE" \
    --output_mask_dir "$OUTPUT_MASK_DIR" \
    2>&1 | tee "$LOG_FILE"

echo "Inference complete. Results saved to $OUTPUT_MASK_DIR"

if [[ "$RUN_EVAL_AFTER_INFERENCE" == "1" ]]; then
    GT_ROOT="${INPUT_MASK_DIR}" \
    PRED_ROOT="${OUTPUT_MASK_DIR}" \
    NUM_PROCESSES="${NUM_PROCESSES}" \
    bash /root/autodl-tmp/ttt_rr/scripts/run_eval_bdd_large_metrics.sh \
        2>&1 | tee "${OUTPUT_MASK_DIR}/eval.log"
fi
