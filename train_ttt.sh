#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ttt_sam

export EXP_DIR=./logs/sam_ttt_davis_$(date +%Y%m%d_%H%M%S)
mkdir -p $EXP_DIR

nohup python training/train.py \
    -c sam2/configs/sam2.1_training/sam2_ttt_davis.yaml \
    > $EXP_DIR/train.log 2>&1 &

echo "Training started. Logs in $EXP_DIR/train.log"
