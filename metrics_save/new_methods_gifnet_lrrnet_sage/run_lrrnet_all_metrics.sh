#!/usr/bin/env bash
# Re-run LRRNet metrics on LLVIP / MSRS / DroneVehicle and refresh colored Excel.
set -euo pipefail

PROJECT=/root/autodl-tmp/URFusion-main
PYTHON=/root/autodl-tmp/conda/envs/urfusion/bin/python
ORCH="$PROJECT/metrics_save/new_methods_gifnet_lrrnet_sage"
EVAL="$PROJECT/metrics_save/llvip_sample1000/run_eval_from_cache.py"
LOG="$ORCH/rerun_lrrnet_metrics.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

remove_lrrnet_xlsx_row() {
	local dataset=$1
	$PYTHON - "$PROJECT/metrics_save" "$dataset" <<'PY'
import sys
from pathlib import Path
import openpyxl
base, ds = Path(sys.argv[1]), sys.argv[2]
p = base / f'{ds}.xlsx'
wb = openpyxl.load_workbook(p)
ws = wb['MeanMetrics']
for r in range(ws.max_row, 1, -1):
	if ws.cell(r, 1).value == 'LRRNet':
		ws.delete_rows(r)
wb.save(p)
print(f'Removed LRRNet row from {p}')
PY
}

run_metrics() {
	local dataset=$1 test_dir=$2
	log "===== LRRNet metrics @ ${dataset} ====="
	remove_lrrnet_xlsx_row "$dataset"
	rm -rf "$PROJECT/metrics_save/LRRNet_${dataset}"
	$PYTHON "$EVAL" \
		--methods LRRNet \
		--datasets "$dataset" \
		--dataset_roots "$test_dir" \
		--results_root "$PROJECT/vis-ir-gray/results" \
		--metrics_save "$PROJECT/metrics_save" \
		--project_root "$PROJECT" \
		2>&1 | tee -a "$ORCH/eval_LRRNet_${dataset}.log"
	$PYTHON "$ORCH/run_color_xlsx.py" --dataset "$dataset" \
		2>&1 | tee -a "$ORCH/color_${dataset}.log"
	log "  Excel: $PROJECT/metrics_save/${dataset}.xlsx"
	log "  Colored: $PROJECT/metrics_save/${dataset}_colored.xlsx"
}

log "========== LRRNet METRICS RERUN START =========="

run_metrics LLVIP "$PROJECT/datasets/LLVIP/test_sample1000"
run_metrics MSRS "$PROJECT/datasets/MSRS/test"
run_metrics DroneVehicle "$PROJECT/datasets/DroneVehicle/test_sample1000"

log "========== LRRNet METRICS RERUN COMPLETE =========="
