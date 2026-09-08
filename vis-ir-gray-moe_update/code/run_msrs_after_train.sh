#!/usr/bin/env bash
# Wait until SpatialResMoE_3 training finishes, then infer+eval LLVIP/MSRS/M3FD.
# Safe to nohup now while training is still running. Shutdown is handled by run_msrs_eval.sh.
set -euo pipefail

PROJECT=/root/autodl-tmp/URFusion-main
ROOT=$PROJECT/vis-ir-gray-moe_update
EXP="${EXP:-SpatialResMoE_3}"
TRAIN_LOG=$ROOT/train-jobs/console_logs/${EXP}.log
CKPT=$ROOT/train-jobs/ckpt/$EXP/${EXP}_ckpt.pth
WAIT_LOG=$ROOT/train-jobs/console_logs/${EXP}_wait_eval.log

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WAIT_LOG"; }
mkdir -p "$(dirname "$WAIT_LOG")"

log "waiting for train DONE exp=$EXP (then LLVIP/MSRS/M3FD eval + shutdown)"
while true; do
	if grep -q "^\[DONE\] ${EXP}$" "$TRAIN_LOG" 2>/dev/null && [[ -f "$CKPT" ]]; then
		if pgrep -f 'vis-ir-gray-moe_update/code/train_fusion_gray.py' >/dev/null; then
			log "log says DONE but train process still up; wait"
		else
			log "train finished, start 3-set eval"
			break
		fi
	fi
	sleep 60
done

export EXP
export SHUTDOWN="${SHUTDOWN:-1}"
bash "$ROOT/code/run_msrs_eval.sh"
log "waiter finished"
