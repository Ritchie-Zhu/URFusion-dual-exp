#!/usr/bin/env bash
# After setup: U2Fusion on 1000 sample -> metrics 7 methods -> colored Excel
set -euo pipefail

PROJECT=/root/autodl-tmp/URFusion-main
PYTHON=/root/autodl-tmp/conda/envs/urfusion/bin/python
LOG_DIR="$PROJECT/metrics_save/dronevehicle_sample1000"
SAMPLE_TEST="$PROJECT/datasets/DroneVehicle/test_sample1000"
RESULTS="$PROJECT/our_model_1_DualMoE/results"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/pipeline.log"; }

log "===== U2Fusion @ sample1000 ====="
cd "$PROJECT/U2Fusion"
$PYTHON test_m3fd.py \
	--dataset DroneVehicle \
	--test_dir "$SAMPLE_TEST" \
	--output_dir "$RESULTS/U2Fusion_DroneVehicle/RGB_fused" \
	2>&1 | tee -a "$LOG_DIR/U2Fusion.log"

u2=$(ls "$RESULTS/U2Fusion_DroneVehicle/RGB_fused" | wc -l)
log "U2Fusion finished: $u2 files"

log "===== eval_multi_time @ DroneVehicle (7 methods, sample 1000) ====="
cd "$PROJECT/metrics_py"
$PYTHON eval_multi_time.py \
	--methods MUFusion U2Fusion URFusion Fusion_training_1 MetaFusion EMMA Text-IF \
	--datasets DroneVehicle \
	2>&1 | tee -a "$LOG_DIR/eval.log"

log "===== color_methods_metrics ====="
$PYTHON color_methods_metrics.py \
	--source "$PROJECT/metrics_save/DroneVehicle.xlsx" \
	--output "$PROJECT/metrics_save/DroneVehicle_colored.xlsx" \
	--sheet MeanMetrics \
	2>&1 | tee -a "$LOG_DIR/color.log"

log "Done. Excel: $PROJECT/metrics_save/DroneVehicle.xlsx"
log "Colored: $PROJECT/metrics_save/DroneVehicle_colored.xlsx"
