#!/usr/bin/env bash
# Wait for 7-method metrics, then color Excel (does not modify metrics_py scripts).
set -euo pipefail

PROJECT=/root/autodl-tmp/URFusion-main
PYTHON=/root/autodl-tmp/conda/envs/urfusion/bin/python
LOG="$PROJECT/metrics_save/llvip_sample1000/color_watcher.log"
METHODS=(MUFusion U2Fusion URFusion Fusion_training_1 MetaFusion EMMA Text-IF)
COLORED="$PROJECT/metrics_save/LLVIP_colored.xlsx"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "Color watcher started"

while pgrep -f 'run_eval_from_cache.py' >/dev/null 2>&1; do
	sleep 60
done

log "eval finished, checking summaries..."
for m in "${METHODS[@]}"; do
	while [[ ! -f "$PROJECT/metrics_save/${m}_LLVIP/metrics_summary.json" ]]; do
		sleep 30
	done
done

if [[ -f "$COLORED" ]]; then
	log "LLVIP_colored.xlsx already exists, skip"
	exit 0
fi

log "Running color_methods_metrics via wrapper..."
$PYTHON "$PROJECT/metrics_save/llvip_sample1000/run_color_llvip.py" >> "$LOG" 2>&1
log "Done: $COLORED"
