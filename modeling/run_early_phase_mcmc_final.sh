#!/usr/bin/env bash
# FINAL production MCMC of the early-phase fit: autonomous, checkpointed, resumable.
#
#   bash run_early_phase_mcmc_final.sh OUTDIR SEED INIT_DIR INIT_START INIT_END
#
# Science: --prior production, wind, xi_e and xi_e_r free, radiative_fireball left at the VegasAfterglow default (True),
#          eps_e + eps_B <= 1 and eps_e_r + eps_B_r <= 1 (all unchanged).
# Numerics: model resolution (0.3, 1, 30), rtol 1e-6 - the setting chosen by the resolution study
#          (result/early_phase_resolution_check: log L within ~0.05 of the converged likelihood).
# Sampler: 64 walkers, 20 workers, DEMove 0.8 + DESnookerMove 0.2, HDF backend, p_r sampled as u_r = log10(p_r - 2)
#          with the prior-preserving Jacobian; walkers start from posterior draws of INIT_DIR, steps [INIT_START, INIT_END).
# Environment: the Python 3.13 conda env yp313 (no NumPy temporary-elision bug).
#
# Stages: 50k (diagnostics only; stop only on a serious failure) -> 100k -> 150k -> 200k (hard maximum). From 100k on,
# stop at the first checkpoint that meets the convergence criteria or reports a serious failure. Re-running the script
# with the same OUTDIR continues the same chain. Decisions: OUTDIR/driver.log; everything else: OUTDIR/run.log.
set -u

OUTDIR=${1:?usage: $0 OUTDIR SEED INIT_DIR INIT_START INIT_END}
SEED=${2:?seed}
INIT_DIR=${3:?init chain directory}
INIT_START=${4:?first init step}
INIT_END=${5:?end init step (0 = end)}
PY=/home/hongyp007/anaconda3/envs/yp313/bin/python
CODE=/extdata01/hongyp007/projects/GRB/code/fitting/modeling
NPOOL=20
NWALKERS=64
RESOLUTION="0.3 1 30"
RTOL=1e-6
STAGES=(50000 100000 150000 200000)
MAX_STEPS=200000

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1

cd "$CODE" || exit 1
mkdir -p "$OUTDIR/environment"
DLOG="$OUTDIR/driver.log"
T_START=$(date +%s)
log() { echo "[$(date '+%F %T')] $*" | tee -a "$DLOG"; }

finish() {
    local elapsed=$(( $(date +%s) - T_START ))
    log "STOP: $1 (driver wall-clock $(( elapsed / 3600 ))h $(( elapsed % 3600 / 60 ))m)"
    local last
    last=$(ls -d "$OUTDIR"/checkpoints/step_* 2>/dev/null | sort | tail -1)
    {
        echo "=== FINAL production MCMC: driver summary ($(date '+%F %T')) ==="
        echo "Stop reason: $1"
        echo
        cat "$DLOG"
        [ -n "$last" ] && { echo; echo "--- Last checkpoint ($last/summary.txt) ---"; cat "$last/summary.txt"; }
    } > "$OUTDIR/FINAL_SUMMARY.txt"
    exit 0
}

# reproducibility record of this run's environment and code
"$PY" -c "import sys, numpy, scipy, emcee, h5py, VegasAfterglow; print(sys.version); print('numpy', numpy.__version__); print('scipy', scipy.__version__); print('emcee', emcee.__version__); print('h5py', h5py.__version__); print('VegasAfterglow', VegasAfterglow.__version__, VegasAfterglow.__file__)" > "$OUTDIR/environment/versions.txt" 2>&1
"$PY" -m pip freeze > "$OUTDIR/environment/pip_freeze.txt" 2>/dev/null
git -C "$CODE" rev-parse HEAD > "$OUTDIR/environment/git_head.txt"
git -C "$CODE" status --porcelain --untracked-files=no > "$OUTDIR/environment/git_modified_tracked_files.txt"
"$PY" "$CODE/test/check_numpy_elision.py" > "$OUTDIR/environment/numpy_elision_check.txt" 2>&1 || {
    log "numpy elision bug present in $PY - refusing to run"; exit 1; }

log "Driver start: OUTDIR=$OUTDIR seed $SEED, init $INIT_DIR steps $INIT_START-$INIT_END, stages ${STAGES[*]} (max $MAX_STEPS)"
log "python $PY ($(head -1 "$OUTDIR/environment/versions.txt")), git $(cat "$OUTDIR/environment/git_head.txt"), resolution ($RESOLUTION) rtol $RTOL, u_r coordinates, $NWALKERS walkers, $NPOOL workers"

for TARGET in "${STAGES[@]}"; do
    [ "$TARGET" -gt "$MAX_STEPS" ] && finish "stage $TARGET would exceed $MAX_STEPS steps"
    if [ ! -f "$OUTDIR/chain.h5" ]; then
        log "Stage to $TARGET: new chain"
        "$PY" early_phase.py --prior production --environment wind --xi both --nwalkers "$NWALKERS" --npool "$NPOOL" \
            --seed "$SEED" --sampling-coords u_r --model-resolution $RESOLUTION --model-rtol "$RTOL" \
            --init-from "$INIT_DIR" --init-window "$INIT_START" "$INIT_END" --nsteps "$TARGET" --outdir "$OUTDIR"
    else
        log "Stage to $TARGET: continuing the same chain $OUTDIR/chain.h5"
        "$PY" early_phase.py --resume "$OUTDIR" --nsteps "$TARGET" --npool "$NPOOL"
    fi
    RC=$?
    [ "$RC" -ne 0 ] && finish "sampler exited with code $RC at stage $TARGET (chain preserved)"

    log "Checkpoint diagnostics at $TARGET steps"
    "$PY" early_phase.py --check-chain "$OUTDIR" --final-check
    CODE_CHECK=$?
    case "$CODE_CHECK" in
        0)  if [ "$TARGET" -lt 100000 ]; then
                log "Convergence criteria met already at $TARGET steps; continuing to the planned 100k anyway"
            else
                finish "convergence criteria met at $TARGET steps"
            fi ;;
        10) [ "$TARGET" -ge "$MAX_STEPS" ] && finish "reached $MAX_STEPS steps without meeting the convergence criteria"
            log "Healthy but not converged at $TARGET steps: continuing the same chain" ;;
        20) finish "SERIOUS SAMPLING FAILURE at $TARGET steps (see the checkpoint summary)" ;;
        *)  finish "checkpoint diagnostics crashed with code $CODE_CHECK at $TARGET steps" ;;
    esac
done
finish "all stages done"
