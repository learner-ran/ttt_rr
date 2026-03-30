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
export ASYNC_LOADING_FRAMES="${ASYNC_LOADING_FRAMES:-1}"
export NUM_PROCESSES="${NUM_PROCESSES:-22}"
export EVAL_NUM_PROCESSES="${EVAL_NUM_PROCESSES:-16}"

TRAFFIC_DATASET_ROOT="${TRAFFIC_DATASET_ROOT:-/root/autodl-tmp/data_set/traffic_mots_semi/val}"
TRAFFIC_BASE_VIDEO_DIR="${TRAFFIC_BASE_VIDEO_DIR:-${TRAFFIC_DATASET_ROOT}/JPEGImages}"
TRAFFIC_INPUT_MASK_DIR="${TRAFFIC_INPUT_MASK_DIR:-${TRAFFIC_DATASET_ROOT}/Annotations}"
TRAFFIC_VIDEO_LIST_FILE="${TRAFFIC_VIDEO_LIST_FILE:-${TRAFFIC_DATASET_ROOT}/val_videos.txt}"
SUMMARY_XLSX="${SUMMARY_XLSX:-/root/autodl-tmp/output_traffic_mots_val/traffic_length_bucket_editable_summary.xlsx}"

declare -a EVAL_PIDS=()
declare -a EXPERIMENT_SPECS=()

start_eval_bg() {
  local output_dir="$1"
  local pred_root="$2"

  if [[ -s "${output_dir}/results.csv" ]]; then
    echo "[skip-eval] ${output_dir}"
    return
  fi

  (
    GT_ROOT="${TRAFFIC_INPUT_MASK_DIR}" \
    PRED_ROOT="${pred_root}" \
    NUM_PROCESSES="${EVAL_NUM_PROCESSES}" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    bash /root/autodl-tmp/ttt_rr/scripts/run_eval_bdd_large_metrics.sh \
      > "${output_dir}/eval.log" 2>&1
  ) &
  EVAL_PIDS+=("$!")
  echo "[eval-bg] ${output_dir} pid=$!"
}

run_experiment() {
  local mode="$1"
  local key="$2"
  local description="$3"
  local cfg="$4"
  local checkpoint="$5"
  local output_dir="$6"

  mkdir -p "${output_dir}"
  local results_csv="${output_dir}/results.csv"
  EXPERIMENT_SPECS+=("${key}::${description}::${results_csv}")

  if [[ -s "${results_csv}" ]]; then
    echo "[skip] ${key} already has results.csv"
    return
  fi

  if grep -q "completed VOS prediction on" "${output_dir}/inference.log" 2>/dev/null; then
    echo "[resume-eval] ${key} inference already completed"
    start_eval_bg "${output_dir}" "${output_dir}"
    return
  fi

  local runner
  if [[ "${mode}" == "ttt" ]]; then
    runner="/root/autodl-tmp/ttt_rr/scripts/run_inference_traffic_val_ttt_maskmem_m3_l1.sh"
  else
    runner="/root/autodl-tmp/ttt_rr/scripts/run_inference_traffic_val_no_ttt.sh"
  fi

  echo "[run] ${key}"
  DATASET_ROOT="${TRAFFIC_DATASET_ROOT}" \
  BASE_VIDEO_DIR="${TRAFFIC_BASE_VIDEO_DIR}" \
  INPUT_MASK_DIR="${TRAFFIC_INPUT_MASK_DIR}" \
  VIDEO_LIST_FILE="${TRAFFIC_VIDEO_LIST_FILE}" \
  OUTPUT_MASK_DIR="${output_dir}" \
  SAM2_CFG="${cfg}" \
  SAM2_CHECKPOINT="${checkpoint}" \
  RUN_EVAL_AFTER_INFERENCE=0 \
  bash "${runner}"

  start_eval_bg "${output_dir}" "${output_dir}"
}

run_experiment \
  "no_ttt" \
  "bdd_large_no_ttt_transfer" \
  "迁移: BDD SAM2 large no TTT" \
  "sam2/configs/sam2.1/sam2.1_hiera_l_no_ttt.yaml" \
  "/root/autodl-tmp/ttt_rr/checkpoints/sam2.1_hiera_large.pt" \
  "/root/autodl-tmp/output_traffic_mots_val/transfer_bdd_large_no_ttt"

run_experiment \
  "ttt" \
  "bdd_restricted_maskmem_m1_l1_transfer" \
  "迁移: BDD restricted + TTT(maskmem) + 1 layer + num_maskmem=1" \
  "sam2/configs/sam2.1/sam2_ttt_inference_l_restricted_maskmem_m1_l1.yaml" \
  "/root/autodl-tmp/ttt_rr/logs/sam_ttt_bdd_restricted_maskmem_m1_l1_run1/checkpoints/checkpoint.pt" \
  "/root/autodl-tmp/output_traffic_mots_val/transfer_bdd_restricted_maskmem_m1_l1"

run_experiment \
  "no_ttt" \
  "bdd_baseline_current_only_transfer" \
  "迁移: BDD baseline current-only (num_maskmem=0, no TTT)" \
  "sam2/configs/sam2.1/sam2_ttt_inference_l_baseline_current_only.yaml" \
  "/root/autodl-tmp/ttt_rr/logs/sam_ttt_bdd_baseline_current_only_run1/checkpoints/checkpoint.pt" \
  "/root/autodl-tmp/output_traffic_mots_val/transfer_bdd_baseline_current_only"

run_experiment \
  "ttt" \
  "bdd_restricted_maskmem_l3_transfer" \
  "迁移: BDD restricted + TTT(maskmem) + 3 layers" \
  "sam2/configs/sam2.1/sam2_ttt_inference_l_restricted_maskmem_l3.yaml" \
  "/root/autodl-tmp/ttt_rr/logs/sam_ttt_bdd_restricted_maskmem_l3_run1/checkpoints/checkpoint.pt" \
  "/root/autodl-tmp/output_traffic_mots_val/transfer_bdd_restricted_maskmem_l3"

for pid in "${EVAL_PIDS[@]}"; do
  wait "${pid}"
done

SUMMARY_ARGS=()
for spec in "${EXPERIMENT_SPECS[@]}"; do
  SUMMARY_ARGS+=(--experiment "${spec}")
done

python /root/autodl-tmp/ttt_rr/tools/update_traffic_length_bucket_summary.py \
  --xlsx "${SUMMARY_XLSX}" \
  --dataset-root "${TRAFFIC_DATASET_ROOT}" \
  "${SUMMARY_ARGS[@]}"

echo "All transfer experiments finished."
