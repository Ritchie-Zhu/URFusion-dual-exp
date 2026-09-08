#!/usr/bin/env bash
# Wait for LRRNet 3-dataset metrics rerun, then M3FD inference + metrics (10 methods).
set -euo pipefail

PROJECT=/root/autodl-tmp/URFusion-main
ORCH="$PROJECT/metrics_save/m3fd_ten_methods"
LRRNET_LOG="$PROJECT/metrics_save/new_methods_gifnet_lrrnet_sage/rerun_lrrnet_metrics.log"
POLL_SEC="${POLL_SEC:-120}"
LOG="$ORCH/wait_pipeline.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

lrrnet_metrics_done() {
	grep -q "LRRNet METRICS RERUN COMPLETE" "$LRRNET_LOG" 2>/dev/null || return 1
	if pgrep -f "run_lrrnet_all_metrics.sh" >/dev/null 2>&1; then return 1; fi
	if pgrep -f "run_eval_from_cache.py.*LRRNet" >/dev/null 2>&1; then return 1; fi
	for ds in LLVIP MSRS DroneVehicle; do
		[[ -f "$PROJECT/metrics_save/LRRNet_${ds}/metrics_summary.json" ]] || return 1
	done
	/root/autodl-tmp/conda/envs/urfusion/bin/python - <<'PY' || return 1
import json
from pathlib import Path
p = Path('/root/autodl-tmp/URFusion-main/metrics_save/LRRNet_LLVIP/metrics_summary.json')
d = json.loads(p.read_text())
if d.get('EN_mean', 0) < 3:
    raise SystemExit('LRRNet LLVIP EN too low (likely bad fused images)')
PY
	return 0
}

log "Watcher started (poll every ${POLL_SEC}s)"
log "Waiting for LRRNet metrics rerun on LLVIP/MSRS/DroneVehicle..."

while true; do
	if lrrnet_metrics_done; then
		log "LRRNet metrics complete. Starting M3FD pipeline..."
		break
	fi
	if [[ -f "$LRRNET_LOG" ]]; then tail -1 "$LRRNET_LOG" | tee -a "$LOG"; fi
	sleep "$POLL_SEC"
done

bash "$ORCH/run_m3fd_inference.sh" 2>&1 | tee -a "$ORCH/m3fd_pipeline.log"
bash "$ORCH/run_m3fd_metrics.sh" 2>&1 | tee -a "$ORCH/m3fd_pipeline.log"

log "========== M3FD FULL PIPELINE DONE =========="
