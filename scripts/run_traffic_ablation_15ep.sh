#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ttt_rr
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ttt_sam

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_PROCESSES="${NUM_PROCESSES:-22}"
export ASYNC_LOADING_FRAMES="${ASYNC_LOADING_FRAMES:-1}"

python /root/autodl-tmp/ttt_rr/scripts/prepare_traffic_mots_split.py > /dev/null

SUMMARY_CSV="${SUMMARY_CSV:-/root/autodl-tmp/output_traffic_mots_val/traffic_ablation_15ep_summary.csv}"
mkdir -p "$(dirname "$SUMMARY_CSV")"
echo "name,exp_dir,output_dir,jf,j,f" > "$SUMMARY_CSV"

append_summary() {
  local name="$1"
  local exp_dir="$2"
  local output_dir="$3"
  python - "$name" "$exp_dir" "$output_dir" "$SUMMARY_CSV" <<'PY'
import csv, sys
name, exp_dir, output_dir, summary_csv = sys.argv[1:]
with open(f"{output_dir}/results.csv", newline="") as f:
    r = csv.reader(f, skipinitialspace=True)
    next(r)
    row = next(r)
jf, j, fscore = row[2].strip(), row[3].strip(), row[4].strip()
with open(summary_csv, "a", newline="") as f:
    csv.writer(f).writerow([name, exp_dir, output_dir, jf, j, fscore])
print(f"[{name}] J&F={jf}, J={j}, F={fscore}")
PY
}

run_ttt_variant() {
  local name="$1"
  local config="$2"
  local infer_cfg="$3"
  local exp_dir="$4"
  local output_dir="$5"
  export EXP_DIR="$exp_dir"
  export OUTPUT_MASK_DIR="$output_dir"
  export CONFIG="$config"
  export SAM2_CFG="$infer_cfg"
  export SAM2_CHECKPOINT="${EXP_DIR}/checkpoints/checkpoint.pt"
  bash /root/autodl-tmp/ttt_rr/scripts/train_ttt_traffic_restricted_maskmem_m3_l1.sh
  bash /root/autodl-tmp/ttt_rr/scripts/run_inference_traffic_val_ttt_maskmem_m3_l1.sh
  append_summary "$name" "$exp_dir" "$output_dir"
}

run_no_ttt_variant() {
  local name="$1"
  local config="$2"
  local infer_cfg="$3"
  local exp_dir="$4"
  local output_dir="$5"
  export EXP_DIR="$exp_dir"
  export OUTPUT_MASK_DIR="$output_dir"
  export CONFIG="$config"
  export SAM2_CFG="$infer_cfg"
  export SAM2_CHECKPOINT="${EXP_DIR}/checkpoints/checkpoint.pt"
  bash /root/autodl-tmp/ttt_rr/scripts/train_traffic_baseplus_15ep.sh
  bash /root/autodl-tmp/ttt_rr/scripts/run_inference_traffic_val_no_ttt.sh
  append_summary "$name" "$exp_dir" "$output_dir"
}

run_ttt_variant \
  "ttt_no_inner_update" \
  "configs/sam2.1_training/sam2_ttt_traffic_large_restricted_maskmem_m3_l1_15ep_no_inner_update" \
  "sam2/configs/sam2.1/sam2_ttt_inference_l_traffic_restricted_maskmem_m3_l1_15ep_no_inner_update.yaml" \
  "/root/autodl-tmp/ttt_rr/logs/sam_ttt_traffic_m3_l1_15ep_no_inner_update_run1" \
  "/root/autodl-tmp/output_traffic_mots_val/traffic_ttt_m3_l1_15ep_no_inner_update"

run_ttt_variant \
  "ttt_self_target" \
  "configs/sam2.1_training/sam2_ttt_traffic_large_restricted_self_m3_l1_15ep" \
  "sam2/configs/sam2.1/sam2_ttt_inference_l_traffic_restricted_self_m3_l1_15ep.yaml" \
  "/root/autodl-tmp/ttt_rr/logs/sam_ttt_traffic_self_m3_l1_15ep_run1" \
  "/root/autodl-tmp/output_traffic_mots_val/traffic_ttt_self_m3_l1_15ep"

run_ttt_variant \
  "ttt_no_gate" \
  "configs/sam2.1_training/sam2_ttt_traffic_large_restricted_maskmem_m3_l1_15ep_no_gate" \
  "sam2/configs/sam2.1/sam2_ttt_inference_l_traffic_restricted_maskmem_m3_l1_15ep_no_gate.yaml" \
  "/root/autodl-tmp/ttt_rr/logs/sam_ttt_traffic_m3_l1_15ep_no_gate_run1" \
  "/root/autodl-tmp/output_traffic_mots_val/traffic_ttt_m3_l1_15ep_no_gate"

run_no_ttt_variant \
  "baseplus_no_ttt" \
  "configs/sam2.1_training/sam2_traffic_baseplus_15ep" \
  "sam2/configs/sam2.1/sam2.1_hiera_b+_no_ttt.yaml" \
  "/root/autodl-tmp/ttt_rr/logs/sam2_traffic_baseplus_15ep_run1" \
  "/root/autodl-tmp/output_traffic_mots_val/traffic_baseplus_15ep"

echo "Traffic ablation runs complete. Summary: $SUMMARY_CSV"
