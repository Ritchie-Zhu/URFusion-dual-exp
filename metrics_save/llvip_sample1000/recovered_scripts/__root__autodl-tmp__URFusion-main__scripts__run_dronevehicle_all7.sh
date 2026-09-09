#!/usr/bin/env bash
# DroneVehicle inference for all 7 MSRS methods (resume-safe: skips existing outputs).
set -euo pipefail

PROJECT=/root/autodl-tmp/URFusion-main
PYTHON=/root/autodl-tmp/conda/envs/urfusion/bin/python
TEST_DIR="$PROJECT/datasets/DroneVehicle/test"
RESULTS="$PROJECT/our_model_1_DualMoE/results"
LOG_DIR="$PROJECT/metrics_save/dronevehicle_inference"
mkdir -p "$LOG_DIR" "$RESULTS"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/all7_pipeline.log"; }

count_outputs() {
	local method=$1
	local d="$RESULTS/${method}_DroneVehicle/RGB_fused"
	if [ -d "$d" ]; then ls "$d" 2>/dev/null | wc -l; else echo 0; fi
}

run_textif() {
	log "===== Text-IF @ DroneVehicle ($(count_outputs Text-IF)/8980) ====="
	cd "$PROJECT/Text-IF"
	$PYTHON test_dronevehicle.py 2>&1 | tee -a "$LOG_DIR/Text-IF.log"
}

run_metafusion() {
	log "===== MetaFusion @ DroneVehicle ($(count_outputs MetaFusion)/8980) ====="
	cd "$PROJECT/MetaFusion"
	$PYTHON test_m3fd.py \
		--test_ir_root "$TEST_DIR/ir" \
		--test_vis_root "$TEST_DIR/vis" \
		--save_path "$RESULTS/MetaFusion_DroneVehicle/RGB_fused" \
		2>&1 | tee -a "$LOG_DIR/MetaFusion.log"
}

run_fusion_training_1() {
	log "===== Fusion_training_1 @ DroneVehicle ($(count_outputs Fusion_training_1)/8980) ====="
	cd "$PROJECT/our_model_1_DualMoE/code"
	$PYTHON test_fusion_gray.py \
		--dataset DroneVehicle \
		--test_dir "$TEST_DIR" \
		--output_root "$RESULTS" \
		2>&1 | tee -a "$LOG_DIR/Fusion_training_1_infer.log"
	$PYTHON colorize_gray.py \
		--method Fusion_training_1 \
		--dataset DroneVehicle \
		--test_dir "$TEST_DIR" \
		--results_root "$RESULTS" \
		2>&1 | tee -a "$LOG_DIR/Fusion_training_1_colorize.log"
}

run_emma() {
	log "===== EMMA @ DroneVehicle ($(count_outputs EMMA)/8980) ====="
	cd "$PROJECT/EMMA"
	$PYTHON test_dronevehicle.py 2>&1 | tee -a "$LOG_DIR/EMMA.log"
}

run_mufusion() {
	log "===== MUFusion @ DroneVehicle ($(count_outputs MUFusion)/8980) ====="
	cd "$PROJECT/MUFusion/ir_vis"
	$PYTHON test_m3fd.py \
		--dataset DroneVehicle \
		--test_dir "$TEST_DIR" \
		--output_dir "$RESULTS/MUFusion_DroneVehicle/RGB_fused" \
		2>&1 | tee -a "$LOG_DIR/MUFusion.log"
}

run_u2fusion() {
	log "===== U2Fusion @ DroneVehicle ($(count_outputs U2Fusion)/8980) ====="
	cd "$PROJECT/U2Fusion"
	$PYTHON test_m3fd.py \
		--dataset DroneVehicle \
		--test_dir "$TEST_DIR" \
		--output_dir "$RESULTS/U2Fusion_DroneVehicle/RGB_fused" \
		2>&1 | tee -a "$LOG_DIR/U2Fusion.log"
}

run_urfusion() {
	local n
	n=$(count_outputs URFusion)
	if [ "$n" -ge 8980 ]; then
		log "===== URFusion @ DroneVehicle: skip (already $n/8980) ====="
		return 0
	fi
	log "===== URFusion @ DroneVehicle ($n/8980) ====="
	cd "$PROJECT/vis-ir/code"
	staging="$PROJECT/vis-ir/results/URFusion_DroneVehicle_staging"
	rm -rf "$staging"
	mkdir -p "$staging"
	$PYTHON test_M3FD.py \
		--fusion_model fusionnet \
		--fusion_pth ../train-jobs/ckpt/content-fusion-msrs_ckpt.pth \
		--vis_mat ../train-jobs/vis.mat \
		--testDir "$TEST_DIR/" \
		--out_name URFusion_DroneVehicle_staging \
		2>&1 | tee -a "$LOG_DIR/URFusion.log"
	mkdir -p "$RESULTS/URFusion_DroneVehicle/RGB_fused"
	rsync -a "$staging/" "$RESULTS/URFusion_DroneVehicle/RGB_fused/"
}

log "DroneVehicle all-7 resume pipeline start (8980 pairs, 840x712)"
log "Methods: Text-IF, MetaFusion, Fusion_training_1, EMMA, MUFusion, U2Fusion, URFusion"

run_textif
run_metafusion
run_fusion_training_1
run_emma
run_mufusion
run_u2fusion
run_urfusion

log "DroneVehicle all-7 pipeline finished"
for m in Text-IF MetaFusion Fusion_training_1 EMMA MUFusion U2Fusion URFusion; do
	log "  $m: $(count_outputs "$m")/8980"
done
