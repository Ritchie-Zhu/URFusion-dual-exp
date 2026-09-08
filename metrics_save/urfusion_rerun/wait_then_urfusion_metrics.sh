#!/usr/bin/env bash
# Wait for URFusion inference to finish, then run metrics + Excel refresh.
set -euo pipefail

ORCH=/root/autodl-tmp/URFusion-main/metrics_save/urfusion_rerun
INF_LOG="$ORCH/urfusion_inference.log"
LOG="$ORCH/wait_metrics.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "Waiting for URFusion inference to complete..."
while ! grep -q 'URFusion INFERENCE COMPLETE' "$INF_LOG" 2>/dev/null; do
	sleep 30
done

log "Inference complete, starting metrics rerun"
bash "$ORCH/run_urfusion_metrics.sh"
