#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ttt_rr
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ttt_sam

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EXP_DIR="${EXP_DIR:-/root/autodl-tmp/ttt_rr/logs/sam_ttt_traffic_restricted_maskmem_m3_l1_run1}"
export OUTPUT_MASK_DIR="${OUTPUT_MASK_DIR:-/root/autodl-tmp/output_traffic_mots_val/traffic_restricted_maskmem_m3_l1}"
export SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${EXP_DIR}/checkpoints/checkpoint.pt}"

python /root/autodl-tmp/ttt_rr/scripts/prepare_traffic_mots_split.py

bash /root/autodl-tmp/ttt_rr/scripts/train_ttt_traffic_restricted_maskmem_m3_l1.sh

NUM_PROCESSES="${NUM_PROCESSES:-22}" \
ASYNC_LOADING_FRAMES="${ASYNC_LOADING_FRAMES:-1}" \
bash /root/autodl-tmp/ttt_rr/scripts/run_inference_traffic_val_ttt_maskmem_m3_l1.sh
