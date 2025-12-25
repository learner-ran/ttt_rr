#!/bin/bash
set -e

# 激活 conda 环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ttt_sam

# 切换到项目目录并设置 PYTHONPATH
cd /root/autodl-tmp/ttt_rr
export PYTHONPATH=.

# 配置（Hydra 格式：去掉 sam2/ 前缀和 .yaml 后缀）
CONFIG="configs/sam2.1_training/sam2_ttt_davis_large"
# 创建按时间命名的子目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export EXP_DIR="/root/autodl-tmp/runs/sam_ttt_davis/${TIMESTAMP}"
mkdir -p "$EXP_DIR"
export CUDA_VISIBLE_DEVICES=0

echo "Starting SAM-TTT training with nohup..."
echo "Working dir: $(pwd)"
echo "Config: $CONFIG"
echo "Output: $EXP_DIR"
echo "Log: $EXP_DIR/train.log"
echo "Conda env: ttt_sam"

# 使用 nohup 后台启动训练（防止 SSH 断开）
nohup python training/train.py \
    -c $CONFIG \
    > "$EXP_DIR/train.log" 2>&1 &

# 保存进程 PID
echo $! > "$EXP_DIR/train.pid"
echo "Training started with PID: $(cat $EXP_DIR/train.pid)"
echo "Monitor log: tail -f $EXP_DIR/train.log"
