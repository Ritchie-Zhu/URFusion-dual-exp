#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/URFusion-main/our_model_1_DualMoE/code

PY=/root/autodl-tmp/conda/envs/urfusion/bin/python
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

DATA=/root/autodl-tmp/URFusion-main/datasets/training_noleak
TEST=/root/autodl-tmp/URFusion-main/datasets/M3FD/test
LOGDIR=/root/autodl-tmp/URFusion-main/our_model_1_DualMoE/train-jobs/console_logs
mkdir -p "$LOGDIR"

run_one() {
  local exp=$1
  shift
  echo "[$(date '+%F %T')] START $exp"
  "$PY" train_fusion_gray.py \
    --device 0 \
    --experiment "$exp" \
    --fusion_model dual_moe_gray \
    --baseDir "$DATA" \
    --testDir "$TEST" \
    --vis_content_ckpt Vis_Content_noleak \
    --ir_content_ckpt ir_Content_noleak \
    --numEpoch 80 \
    --batchsize 12 \
    --patchsize 160 \
    --ckpt_interval 10 \
    --save_fixed_every 20 \
    --num_workers 12 \
    --lambda_int 0.1 \
    --lambda_grad 1.0 \
    --lambda_branch 0.01 \
    --lambda_aux 0.03 \
    --top_k 2 \
    --cudnn_benchmark \
    "$@" \
    > "$LOGDIR/${exp}.log" 2>&1
  echo "[$(date '+%F %T')] DONE $exp"
}

run_one Abl_DualMoE_wo_Lint --lambda_int 0
run_one Abl_DualMoE_wo_Lgrad --lambda_grad 0
run_one Abl_DualMoE_wo_Lbranch --lambda_branch 0

echo "[$(date '+%F %T')] ALL B1 B2 B3 DONE"
echo '[SHUTDOWN] AutoDL instance stopping'
/usr/bin/shutdown
