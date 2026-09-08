#!/usr/bin/env bash
# One GPU: A1 (Abl_wo_Dual) then A2 (Abl_wo_MoE). Shutdown only after both succeed.
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PROJECT=/root/autodl-tmp/URFusion-main
ORCH="$PROJECT/ablation_code"
LOGDIR="$PROJECT/vis-ir-gray/train-jobs/console_logs"
LOG="$LOGDIR/A1_A2_seq.log"
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "========== A1 then A2 START =========="
log "OMP_NUM_THREADS=${OMP_NUM_THREADS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

log "===== A1 Abl_wo_Dual ====="
bash "$ORCH/A1_wo_Dual/run_train.sh"
log "===== A1 DONE ====="

log "===== A2 Abl_wo_MoE ====="
bash "$ORCH/A2_wo_MoE/run_train.sh"
log "===== A2 DONE ====="

date '+%F %T' > "$ORCH/A1_A2_COMPLETE"
log "========== A1 then A2 COMPLETE =========="
sync
log "[SHUTDOWN] both trains succeeded; calling /usr/bin/shutdown"
sleep 3
/usr/bin/shutdown
