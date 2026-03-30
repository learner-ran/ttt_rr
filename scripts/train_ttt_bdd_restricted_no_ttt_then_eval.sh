#!/bin/bash
set -e
set -o pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ttt_sam

cd /root/autodl-tmp/ttt_rr
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EXP_DIR="${EXP_DIR:-/root/autodl-tmp/ttt_rr/logs/sam_ttt_bdd_restricted_no_ttt_run1}"

PIPELINE_LOG="${PIPELINE_LOG:-${EXP_DIR}/pipeline.log}"
mkdir -p "$EXP_DIR"

exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "Pipeline started at $(date)"
echo "Train output dir: $EXP_DIR"

bash /root/autodl-tmp/ttt_rr/scripts/train_ttt_bdd_restricted_no_ttt.sh

echo "Training finished at $(date)"
echo "Starting BDD val inference/evaluation for all three checkpoints..."

bash /root/autodl-tmp/ttt_rr/scripts/run_eval_bdd_val_all_experiments.sh

echo "Pipeline finished at $(date)"
