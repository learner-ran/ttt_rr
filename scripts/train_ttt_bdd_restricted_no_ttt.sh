#!/bin/bash
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ttt_sam

cd /root/autodl-tmp/ttt_rr
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EXP_DIR="${EXP_DIR:-/root/autodl-tmp/ttt_rr/logs/sam_ttt_bdd_restricted_no_ttt_run1}"

CONFIG="configs/sam2.1_training/sam2_ttt_bdd_large_restricted_no_ttt"

mkdir -p "$EXP_DIR"

echo "Starting restricted-memory BDD training with TTT disabled..."
echo "Working dir: $(pwd)"
echo "Config: $CONFIG"
echo "Output: $EXP_DIR"
echo "Conda env: ttt_sam"

python training/train.py -c "$CONFIG"
