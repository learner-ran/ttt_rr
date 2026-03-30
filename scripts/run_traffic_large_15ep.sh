#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ttt_rr
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ttt_sam

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_PROCESSES="${NUM_PROCESSES:-22}"
export ASYNC_LOADING_FRAMES="${ASYNC_LOADING_FRAMES:-1}"

SUMMARY_CSV="${SUMMARY_CSV:-/root/autodl-tmp/output_traffic_mots_val/traffic_ablation_15ep_summary.csv}"
EXP_DIR="${EXP_DIR:-/root/autodl-tmp/ttt_rr/logs/sam2_traffic_large_15ep_run1}"
OUTPUT_MASK_DIR="${OUTPUT_MASK_DIR:-/root/autodl-tmp/output_traffic_mots_val/traffic_large_15ep}"
CONFIG="${CONFIG:-configs/sam2.1_training/sam2_traffic_large_15ep}"
SAM2_CFG="${SAM2_CFG:-sam2/configs/sam2.1/sam2.1_hiera_l_no_ttt.yaml}"

mkdir -p "$(dirname "$SUMMARY_CSV")"

export EXP_DIR OUTPUT_MASK_DIR CONFIG
export SAM2_CFG
export SAM2_CHECKPOINT="${EXP_DIR}/checkpoints/checkpoint.pt"

bash /root/autodl-tmp/ttt_rr/scripts/train_traffic_baseplus_15ep.sh
bash /root/autodl-tmp/ttt_rr/scripts/run_inference_traffic_val_no_ttt.sh

python - "$SUMMARY_CSV" "$EXP_DIR" "$OUTPUT_MASK_DIR" <<'PY'
import csv, sys
summary_csv, exp_dir, output_dir = sys.argv[1:]
with open(f"{output_dir}/results.csv", newline="") as f:
    r = csv.reader(f, skipinitialspace=True)
    next(r)
    row = next(r)
jf, j, fscore = row[2].strip(), row[3].strip(), row[4].strip()
with open(summary_csv, newline="") as f:
    rows = list(csv.reader(f))
header = rows[0] if rows else ["name","exp_dir","output_dir","jf","j","f"]
body = [row for row in rows[1:] if row and row[0] != "large_no_ttt"]
body.append(["large_no_ttt", exp_dir, output_dir, jf, j, fscore])
with open(summary_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(body)
print(f"[large_no_ttt] J&F={jf}, J={j}, F={fscore}")
PY
