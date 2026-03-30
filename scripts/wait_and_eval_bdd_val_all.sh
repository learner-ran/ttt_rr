#!/bin/bash
set -e
set -o pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <train_pid>"
  exit 1
fi

TRAIN_PID="$1"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 60
done

cd /root/autodl-tmp/ttt_rr
bash /root/autodl-tmp/ttt_rr/scripts/run_eval_bdd_val_all_experiments.sh
