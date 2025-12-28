#!/bin/bash
set -e

# Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ttt_sam

cd /root/autodl-tmp/ttt_rr
export PYTHONPATH=.

# Paths aligned with inference script
GT_ROOT="/root/autodl-tmp/data_set/MOSE/train/train/Annotations"
PRED_ROOT="/root/autodl-tmp/output_mose_val/output_mose_val_ttt"

echo "Running MOSE evaluation..."
echo "GT root: ${GT_ROOT}"
echo "Pred root: ${PRED_ROOT}"

python sav_dataset/sav_evaluator.py \
    --gt_root "${GT_ROOT}" \
    --pred_root "${PRED_ROOT}"

echo "✅ Evaluation finished"
