#!/usr/bin/env bash
# M3FD metrics (25 static) + Excel + colored Excel for 10 methods.
# Uses same eval_multi_time bytecode cache as other datasets.
set -euo pipefail

PROJECT=/root/autodl-tmp/URFusion-main
PYTHON=/root/autodl-tmp/conda/envs/urfusion/bin/python
ORCH="$PROJECT/metrics_save/m3fd_ten_methods"
EVAL="$PROJECT/metrics_save/llvip_sample1000/run_eval_from_cache.py"
COLOR="$PROJECT/metrics_save/new_methods_gifnet_lrrnet_sage/run_color_xlsx.py"
TEST_DIR="$PROJECT/datasets/M3FD/test"
METHODS=(MUFusion U2Fusion URFusion Fusion_training_1 MetaFusion EMMA Text-IF GIFNet LRRNet SAGE)
LOG="$ORCH/m3fd_metrics.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

clean_m3fd_metrics() {
	log "Clean prior M3FD metrics artifacts (10 methods only)"
	rm -f "$PROJECT/metrics_save/M3FD.xlsx" "$PROJECT/metrics_save/M3FD_colored.xlsx"
	for m in "${METHODS[@]}"; do
		rm -rf "$PROJECT/metrics_save/${m}_M3FD"
	done
}

verify_inference() {
	local expected=$1
	for m in "${METHODS[@]}"; do
		local n
		n=$(find "$PROJECT/our_model_1_DualMoE/results/${m}_M3FD/RGB_fused" -type f 2>/dev/null | wc -l)
		if [[ "$n" -lt "$expected" ]]; then
			log "ERROR: ${m}_M3FD has ${n}/${expected} fused images"
			exit 1
		fi
		log "  verify ${m}: ${n}/${expected}"
	done
}

log "========== M3FD METRICS START =========="
N=$(find "$TEST_DIR/vis" -type f | wc -l)
log "M3FD test pairs: $N"
verify_inference "$N"
clean_m3fd_metrics

log "===== eval_multi_time @ M3FD (10 methods) ====="
$PYTHON "$EVAL" \
	--methods "${METHODS[@]}" \
	--datasets M3FD \
	--dataset_roots "$TEST_DIR" \
	--results_root "$PROJECT/our_model_1_DualMoE/results" \
	--metrics_save "$PROJECT/metrics_save" \
	--project_root "$PROJECT" \
	2>&1 | tee -a "$ORCH/eval_M3FD.log"

log "===== color_methods_metrics @ M3FD ====="
$PYTHON "$COLOR" --dataset M3FD \
	2>&1 | tee -a "$ORCH/color_M3FD.log"

log "  Excel: $PROJECT/metrics_save/M3FD.xlsx"
log "  Colored: $PROJECT/metrics_save/M3FD_colored.xlsx"
log "========== M3FD METRICS COMPLETE =========="
