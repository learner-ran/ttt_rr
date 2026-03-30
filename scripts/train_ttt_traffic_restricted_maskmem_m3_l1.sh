#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ttt_rr
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ttt_sam

export PYTHONPATH=.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EXP_DIR="${EXP_DIR:-/root/autodl-tmp/ttt_rr/logs/sam_ttt_traffic_restricted_maskmem_m3_l1_run1}"

CONFIG="${CONFIG:-configs/sam2.1_training/sam2_ttt_traffic_large_restricted_maskmem_m3_l1}"

mkdir -p "$EXP_DIR"

echo "Starting traffic_mots_semi training..."
echo "Working dir: $(pwd)"
echo "Config: $CONFIG"
echo "Output: $EXP_DIR"

python training/train.py -c "$CONFIG"
