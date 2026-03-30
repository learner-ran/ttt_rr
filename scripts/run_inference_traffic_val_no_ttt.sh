#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ttt_rr
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ttt_sam

export PYTHONPATH=.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_PROCESSES="${NUM_PROCESSES:-22}"

DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/data_set/traffic_mots_semi/val}"
BASE_VIDEO_DIR="${BASE_VIDEO_DIR:-${DATASET_ROOT}/JPEGImages}"
INPUT_MASK_DIR="${INPUT_MASK_DIR:-${DATASET_ROOT}/Annotations}"
VIDEO_LIST_FILE="${VIDEO_LIST_FILE:-${DATASET_ROOT}/val_videos.txt}"

SAM2_CFG="${SAM2_CFG:-sam2/configs/sam2.1/sam2.1_hiera_b+_no_ttt.yaml}"
DEFAULT_CHECKPOINT="/root/autodl-tmp/ttt_rr/logs/sam2_traffic_baseplus_15ep_run1/checkpoints/checkpoint.pt"
if [[ -n "${SAM2_CHECKPOINT:-}" ]]; then
  :
elif [[ -n "${EXP_DIR:-}" ]]; then
  SAM2_CHECKPOINT="${EXP_DIR}/checkpoints/checkpoint.pt"
else
  SAM2_CHECKPOINT="${DEFAULT_CHECKPOINT}"
fi
OUTPUT_MASK_DIR="${OUTPUT_MASK_DIR:-/root/autodl-tmp/output_traffic_mots_val/traffic_baseplus_15ep}"
LOG_FILE="${OUTPUT_MASK_DIR}/inference.log"
RUN_EVAL_AFTER_INFERENCE="${RUN_EVAL_AFTER_INFERENCE:-1}"

mkdir -p "$OUTPUT_MASK_DIR"

echo "Running traffic_mots_semi val inference with TTT disabled..."
echo "Config: $SAM2_CFG"
echo "Checkpoint: $SAM2_CHECKPOINT"
echo "Dataset: $BASE_VIDEO_DIR"
echo "Input masks: $INPUT_MASK_DIR"
echo "Video list: $VIDEO_LIST_FILE"
echo "Output: $OUTPUT_MASK_DIR"

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
    GT_ROOT="${INPUT_MASK_DIR}" \
    PRED_ROOT="${OUTPUT_MASK_DIR}" \
    NUM_PROCESSES="${NUM_PROCESSES}" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    bash /root/autodl-tmp/ttt_rr/scripts/run_eval_bdd_large_metrics.sh \
        2>&1 | tee "${OUTPUT_MASK_DIR}/eval.log"
fi
