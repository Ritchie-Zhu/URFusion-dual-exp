#!/usr/bin/env bash
# Wait for DroneVehicle inference to finish, then run full metrics + colored Excel.
set -euo pipefail

PROJECT=/root/autodl-tmp/URFusion-main
PYTHON=/root/autodl-tmp/conda/envs/urfusion/bin/python
TOTAL=8980
RESULTS="$PROJECT/our_model_1_DualMoE/results"
LOG_DIR="$PROJECT/metrics_save/dronevehicle_inference"
METRICS_DIR="$PROJECT/metrics_py"
METHODS=(MUFusion U2Fusion URFusion Fusion_training_1 MetaFusion EMMA Text-IF)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/metrics_watch.log"; }

count_rgb() {
	local m=$1
	local d="$RESULTS/${m}_DroneVehicle/RGB_fused"
	if [ -d "$d" ]; then ls "$d" 2>/dev/null | wc -l; else echo 0; fi
}

all_complete() {
	for m in "$METHODS[@]"; do
		local n
		n=$(count_rgb "$m")
		if [ "$n" -lt "$TOTAL" ]; then
			return 1
		fi
	done
	return 0
}

log "Watching DroneVehicle inference ($TOTAL pairs per method)..."

while true; do
	infer_running=false
	if pgrep -f "run_dronevehicle_all7|test_m3fd.*DroneVehicle|test_dronevehicle|test_fusion_gray.*DroneVehicle|test_M3FD.*DroneVehicle" >/dev/null 2>&1; then
		infer_running=true
	fi

	status=""
	for m in "$METHODS[@]"; do
		status+="$m:$(count_rgb "$m)/$TOTAL "
	done
	log "progress: $status"

	if ! $infer_running && all_complete; then
		log "All methods complete. Starting metrics..."
		break
	fi

	sleep 120
done

cd "$METRICS_DIR"
log "===== eval_multi_time @ DroneVehicle (7 methods) ====="
$PYTHON eval_multi_time.py \
	--methods MUFusion U2Fusion URFusion Fusion_training_1 MetaFusion EMMA Text-IF \
	--datasets DroneVehicle \
	2>&1 | tee "$LOG_DIR/eval_dronevehicle.log"

EXCEL="$PROJECT/metrics_save/DroneVehicle.xlsx"
COLORED="$PROJECT/metrics_save/DroneVehicle_colored.xlsx"

log "===== color_methods_metrics ====="
$PYTHON color_methods_metrics.py \
	--source "$EXCEL" \
	--output "$COLORED" \
	--sheet MeanMetrics \
	2>&1 | tee "$LOG_DIR/color_dronevehicle.log"

log "Done. Excel: $EXCEL"
log "Colored: $COLORED"
