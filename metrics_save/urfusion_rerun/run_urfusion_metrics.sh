#!/usr/bin/env bash
# Re-run URFusion metrics on LLVIP / MSRS / M3FD and refresh Excel files.
set -euo pipefail

PROJECT=/root/autodl-tmp/URFusion-main
PYTHON=/root/autodl-tmp/conda/envs/urfusion/bin/python
ORCH="$PROJECT/metrics_save/urfusion_rerun"
EVAL="$PROJECT/metrics_save/llvip_sample1000/run_eval_from_cache.py"
COLOR="$PROJECT/metrics_save/new_methods_gifnet_lrrnet_sage/run_color_xlsx.py"
LOG="$ORCH/urfusion_metrics.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

remove_urfusion_xlsx_row() {
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
	if ws.cell(r, 1).value == 'URFusion':
		ws.delete_rows(r)
wb.save(p)
print(f'Removed URFusion row from {p}')
PY
}

verify_inference() {
	local dataset=$1 expected=$2
	local n
	n=$(find "$PROJECT/vis-ir-gray/results/URFusion_${dataset}/RGB_fused" -type f 2>/dev/null | wc -l)
	if [[ "$n" -lt "$expected" ]]; then
		log "ERROR: URFusion_${dataset} has ${n}/${expected} fused images"
		exit 1
	fi
	log "  verify URFusion_${dataset}: ${n}/${expected}"
}

run_metrics() {
	local dataset=$1 test_dir=$2 expected=$3
	log "===== URFusion metrics @ ${dataset} ====="
	verify_inference "$dataset" "$expected"
	remove_urfusion_xlsx_row "$dataset"
	rm -rf "$PROJECT/metrics_save/URFusion_${dataset}"
	$PYTHON "$EVAL" \
		--methods URFusion \
		--datasets "$dataset" \
		--dataset_roots "$test_dir" \
		--results_root "$PROJECT/vis-ir-gray/results" \
		--metrics_save "$PROJECT/metrics_save" \
		--project_root "$PROJECT" \
		2>&1 | tee -a "$ORCH/eval_URFusion_${dataset}.log"
	$PYTHON "$COLOR" --dataset "$dataset" \
		2>&1 | tee -a "$ORCH/color_${dataset}.log"
	log "  Excel: $PROJECT/metrics_save/${dataset}.xlsx"
	log "  Colored: $PROJECT/metrics_save/${dataset}_colored.xlsx"
}

log "========== URFusion METRICS RERUN START =========="

run_metrics LLVIP "$PROJECT/datasets/LLVIP/test_sample1000" 1000
run_metrics MSRS "$PROJECT/datasets/MSRS/test" 361
run_metrics M3FD "$PROJECT/datasets/M3FD/test" 300

log "===== rebuild filtered comparison tables ====="
$PYTHON "$PROJECT/metrics_save/build_llvip_msrs_selected_8x8.py" \
	2>&1 | tee -a "$ORCH/rebuild_8x8.log"
$PYTHON "$PROJECT/metrics_save/build_llvip_msrs_m3fd_selected_8x6.py" \
	2>&1 | tee -a "$ORCH/rebuild_8x6.log"

log "========== URFusion METRICS RERUN COMPLETE =========="
