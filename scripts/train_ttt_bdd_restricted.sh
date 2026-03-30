#!/bin/bash
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ttt_sam

cd /root/autodl-tmp/ttt_rr
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

CONFIG="configs/sam2.1_training/sam2_ttt_bdd_large_restricted"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export EXP_DIR="/root/autodl-tmp/runs/sam_ttt_bdd_restricted/${TIMESTAMP}"
mkdir -p "$EXP_DIR"
export CUDA_VISIBLE_DEVICES=0

echo "Starting restricted-memory SAM-TTT training on BDD100K MOTS semi..."
echo "Working dir: $(pwd)"
echo "Config: $CONFIG"
echo "Output: $EXP_DIR"
echo "Log: $EXP_DIR/train.log"
echo "Conda env: ttt_sam"

nohup stdbuf -oL -eL python training/train.py \
    -c $CONFIG \
    > "$EXP_DIR/train.log" 2>&1 &

echo $! > "$EXP_DIR/train.pid"
echo "Training started with PID: $(cat $EXP_DIR/train.pid)"
echo "Monitor log: tail -f $EXP_DIR/train.log"
