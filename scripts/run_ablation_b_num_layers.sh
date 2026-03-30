#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ttt_rr
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ttt_sam

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"
export ASYNC_LOADING_FRAMES="${ASYNC_LOADING_FRAMES:-1}"
export NUM_PROCESSES="${NUM_PROCESSES:-22}"

LAYERS="${LAYERS:-1 2 3 6}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/data_set/BDD100K_MOTS_semi/val}"
BASE_VIDEO_DIR="${BASE_VIDEO_DIR:-${DATASET_ROOT}/JPEGImages}"
INPUT_MASK_DIR="${INPUT_MASK_DIR:-${DATASET_ROOT}/Annotations}"
VIDEO_LIST_FILE="${VIDEO_LIST_FILE:-${DATASET_ROOT}/val_videos.txt}"
SUMMARY_CSV="${SUMMARY_CSV:-/root/autodl-tmp/output_bdd_semi_val/bdd_restricted_maskmem_num_layers_summary.csv}"

mkdir -p "$(dirname "$SUMMARY_CSV")"
if [[ ! -f "$SUMMARY_CSV" ]]; then
    echo "num_layers,train_log_dir,output_dir,jf,j,f" > "$SUMMARY_CSV"
fi

for layer in $LAYERS; do
    export EXP_DIR="/root/autodl-tmp/ttt_rr/logs/sam_ttt_bdd_restricted_maskmem_l${layer}_run1"
    TRAIN_CFG="configs/sam2.1_training/sam2_ttt_bdd_large_restricted_maskmem_l${layer}"
    SAM2_CFG="sam2/configs/sam2.1/sam2_ttt_inference_l_restricted_maskmem_l${layer}.yaml"
    SAM2_CHECKPOINT="${EXP_DIR}/checkpoints/checkpoint.pt"
    OUTPUT_MASK_DIR="/root/autodl-tmp/output_bdd_semi_val/bdd_restricted_maskmem_l${layer}"

    echo "============================================================"
    echo "Ablation B: num_layers=${layer}"
    echo "Train cfg: ${TRAIN_CFG}"
    echo "Exp dir:   ${EXP_DIR}"
    echo "Output:    ${OUTPUT_MASK_DIR}"
    echo "============================================================"

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    python training/train.py -c "${TRAIN_CFG}"

    export OMP_NUM_THREADS="${OMP_NUM_THREADS_INFER:-12}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS_INFER:-12}"

    mkdir -p "${OUTPUT_MASK_DIR}"
    python tools/vos_inference_ttt.py \
        --sam2_cfg "${SAM2_CFG}" \
        --sam2_checkpoint "${SAM2_CHECKPOINT}" \
        --base_video_dir "${BASE_VIDEO_DIR}" \
        --input_mask_dir "${INPUT_MASK_DIR}" \
        --video_list_file "${VIDEO_LIST_FILE}" \
        --output_mask_dir "${OUTPUT_MASK_DIR}" \
        2>&1 | tee "${OUTPUT_MASK_DIR}/inference.log"

    GT_ROOT="${INPUT_MASK_DIR}" \
    PRED_ROOT="${OUTPUT_MASK_DIR}" \
    NUM_PROCESSES="${NUM_PROCESSES}" \
    bash /root/autodl-tmp/ttt_rr/scripts/run_eval_bdd_large_metrics.sh \
        2>&1 | tee "${OUTPUT_MASK_DIR}/eval.log"

    python - "${layer}" "${EXP_DIR}" "${OUTPUT_MASK_DIR}" "${SUMMARY_CSV}" <<'PY'
import csv
import sys
from pathlib import Path

layer, exp_dir, out_dir, summary_csv = sys.argv[1:]
result_csv = Path(out_dir) / "results.csv"
with result_csv.open() as f:
    rows = list(csv.reader(f))
global_row = rows[1]
jf = global_row[2].strip()
j = global_row[3].strip()
f_score = global_row[4].strip()
with open(summary_csv, "a", newline="") as out:
    writer = csv.writer(out)
    writer.writerow([layer, exp_dir, out_dir, jf, j, f_score])
print(f"[Ablation B] num_layers={layer}: J&F={jf}, J={j}, F={f_score}")
PY
done

echo "Ablation B complete. Summary saved to ${SUMMARY_CSV}"
