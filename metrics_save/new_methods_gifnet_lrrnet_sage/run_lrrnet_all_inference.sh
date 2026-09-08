#!/usr/bin/env bash
# Re-run LRRNet inference on LLVIP / MSRS / DroneVehicle (fixed YCbCr scale).
set -euo pipefail

PROJECT=/root/autodl-tmp/URFusion-main
PYTHON=/root/autodl-tmp/conda/envs/urfusion/bin/python
ORCH="$PROJECT/metrics_save/new_methods_gifnet_lrrnet_sage"
LOG="$ORCH/lrrnet_rerun_inference.log"

export CUDA_VISIBLE_DEVICES=0

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_one() {
	local dataset=$1 test_dir=$2
	local out="$PROJECT/vis-ir-gray/results/LRRNet_${dataset}/RGB_fused"
	log "===== LRRNet @ ${dataset} ====="
	rm -rf "$out"
	mkdir -p "$out"
	$PYTHON "$ORCH/run_lrrnet_infer.py" \
		--dataset "$dataset" \
		--test_dir "$test_dir" \
		--output_dir "$out" \
		--force \
		2>&1 | tee -a "$ORCH/LRRNet_${dataset}.log"
	local n
	n=$(find "$out" -type f | wc -l)
	log "  ${dataset} done: ${n} files -> ${out}"
}

log "========== LRRNet INFERENCE RERUN START =========="

run_one LLVIP "$PROJECT/datasets/LLVIP/test_sample1000"
run_one MSRS "$PROJECT/datasets/MSRS/test"
run_one DroneVehicle "$PROJECT/datasets/DroneVehicle/test_sample1000"

log "========== LRRNet INFERENCE RERUN COMPLETE =========="
