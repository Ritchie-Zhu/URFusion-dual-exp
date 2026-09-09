#!/usr/bin/env bash
# Train ConflictAdd_1, then official 3-set eval. Always shut down when the job ends.
set +e

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PROJECT=/root/autodl-tmp/URFusion-main
ROOT=$PROJECT/our_model_4_ConflictAdd
CODE=$ROOT/code
LOGDIR=$ROOT/train-jobs/console_logs
mkdir -p "$LOGDIR"
PIPE_LOG=$LOGDIR/ConflictAdd_1_pipeline.log

log() { echo "[$(date '+%F %T')] $*" | tee -a "$PIPE_LOG"; }

shutdown_now() {
	log "[SHUTDOWN] pipeline end train_rc=${TRAIN_RC:-na} eval_rc=${EVAL_RC:-na}"
	sync
	sleep 3
	/usr/bin/shutdown
}

TRAIN_RC=1
EVAL_RC=1
log "========== PIPELINE START ConflictAdd_1 =========="
bash "$CODE/run_train.sh"
TRAIN_RC=$?
log "train finished rc=$TRAIN_RC"

if [[ "$TRAIN_RC" -eq 0 ]]; then
	SHUTDOWN=0 bash "$CODE/run_3set_eval.sh"
	EVAL_RC=$?
	log "eval finished rc=$EVAL_RC"
else
	log "skip eval because train failed"
fi

shutdown_now
