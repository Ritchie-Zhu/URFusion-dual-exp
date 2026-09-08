#!/usr/bin/env bash
# Resume LLVIP sample1000 metrics (remaining 4 methods). Detached-safe.
set -euo pipefail

PROJECT=/root/autodl-tmp/URFusion-main
PYTHON=/root/autodl-tmp/conda/envs/urfusion/bin/python
LOG_DIR="$PROJECT/metrics_save/llvip_sample1000"
RUNNER="$LOG_DIR/run_eval_from_cache.py"

log() { echo "[$(date '+%F %T')] $*"; }

log "===== Resume metrics (4 methods) ====="
log "Using cached eval_multi_time bytecode (2026-06-26, same as MUFusion/U2Fusion/URFusion)"
log "Metric_torch.py + eval_torch.py: unchanged on disk"

cd "$LOG_DIR"
$PYTHON "$RUNNER" \
	--methods Fusion_training_1 MetaFusion EMMA Text-IF \
	--datasets LLVIP \
	--results_root "$PROJECT/vis-ir-gray/results" \
	--metrics_save "$PROJECT/metrics_save" \
	--project_root "$PROJECT"

log "===== Metrics finished ====="

COLOR_SCRIPT="$PROJECT/metrics_py/color_methods_metrics.py"
if [[ -f "$COLOR_SCRIPT" ]]; then
	log "===== color_methods_metrics ====="
	cd "$PROJECT/metrics_py"
	$PYTHON "$COLOR_SCRIPT" \
		--source "$PROJECT/metrics_save/LLVIP.xlsx" \
		--output "$PROJECT/metrics_save/LLVIP_colored.xlsx" \
		--sheet MeanMetrics
	log "Colored Excel: $PROJECT/metrics_save/LLVIP_colored.xlsx"
else
	log "SKIP colored Excel: color_methods_metrics.py not found (upload to metrics_py/)"
fi

log "===== All done ====="
