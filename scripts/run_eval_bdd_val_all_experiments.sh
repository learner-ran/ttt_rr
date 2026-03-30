#!/bin/bash
set -e
set -o pipefail

cd /root/autodl-tmp/ttt_rr

OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/output_bdd_semi_val}"

echo "Evaluating three BDD val experiments..."
echo "Output root: $OUTPUT_ROOT"

SAM2_CFG="sam2/configs/sam2.1/sam2_ttt_inference_l.yaml" \
SAM2_CHECKPOINT="/root/autodl-tmp/ttt_rr/logs/sam_ttt_bdd_run1/checkpoints/checkpoint.pt" \
OUTPUT_MASK_DIR="${OUTPUT_ROOT}/bdd_fullmem_ttt" \
RUN_EVAL_AFTER_INFERENCE=1 \
bash /root/autodl-tmp/ttt_rr/scripts/run_inference_bdd_val_ttt.sh

SAM2_CFG="sam2/configs/sam2.1/sam2_ttt_inference_l_restricted.yaml" \
SAM2_CHECKPOINT="/root/autodl-tmp/ttt_rr/logs/sam_ttt_bdd_restricted_run1/checkpoints/checkpoint.pt" \
OUTPUT_MASK_DIR="${OUTPUT_ROOT}/bdd_restricted_ttt" \
RUN_EVAL_AFTER_INFERENCE=1 \
bash /root/autodl-tmp/ttt_rr/scripts/run_inference_bdd_val_ttt.sh

SAM2_CFG="sam2/configs/sam2.1/sam2_ttt_inference_l_restricted_no_ttt.yaml" \
SAM2_CHECKPOINT="/root/autodl-tmp/ttt_rr/logs/sam_ttt_bdd_restricted_no_ttt_run1/checkpoints/checkpoint.pt" \
OUTPUT_MASK_DIR="${OUTPUT_ROOT}/bdd_restricted_no_ttt" \
RUN_EVAL_AFTER_INFERENCE=1 \
bash /root/autodl-tmp/ttt_rr/scripts/run_inference_bdd_val_no_ttt.sh

echo "All three BDD val runs completed."
