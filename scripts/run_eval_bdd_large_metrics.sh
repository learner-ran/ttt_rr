#!/bin/bash
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ttt_sam

cd /root/autodl-tmp/ttt_rr
export PYTHONPATH=.

GT_ROOT="${GT_ROOT:-/root/autodl-tmp/data_set/BDD100K_MOTS_semi/val/Annotations}"
PRED_ROOT="${PRED_ROOT:-/root/autodl-tmp/output_bdd_semi_val/output_bdd_semi_large}"
NUM_PROCESSES="${NUM_PROCESSES:-20}"

echo "Running semi-supervised BDD100K MOTS evaluation (SAM 2.1 large)..."
echo "GT root: ${GT_ROOT}"
echo "Pred root: ${PRED_ROOT}"
echo "Num processes: ${NUM_PROCESSES}"

python sav_dataset/sav_evaluator.py \
    --gt_root "${GT_ROOT}" \
    --pred_root "${PRED_ROOT}" \
    -n "${NUM_PROCESSES}"

echo "Evaluation finished"
