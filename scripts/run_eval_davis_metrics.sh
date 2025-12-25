#!/bin/bash
set -e

# Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ttt_sam

cd /root/autodl-tmp/ttt_rr
export PYTHONPATH=.

# Paths aligned with inference script
GT_ROOT="/root/autodl-tmp/dataset_480p/DAVIS/Annotations/480p"
PRED_ROOT="/root/autodl-tmp/output_davis_val/output_davis_val_ttt_1223_2"

echo "Running DAVIS evaluation..."
echo "GT root: ${GT_ROOT}"
echo "Pred root: ${PRED_ROOT}"

python sav_dataset/sav_evaluator.py \
    --gt_root "${GT_ROOT}" \
    --pred_root "${PRED_ROOT}"

echo "✅ Evaluation finished"
