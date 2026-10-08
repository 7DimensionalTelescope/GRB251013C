#!/usr/bin/env bash
# Autonomous overnight production MCMC of the early-phase fit (wind, xi_e and xi_e_r free, --prior production).
#
#   bash run_early_phase_mcmc_overnight.sh OUTDIR
#
# Stage A: run to 30,000 total steps, then checkpoint diagnostics (early_phase.py --check-chain).
# Stage B: if healthy but not converged, continue the SAME HDF chain (OUTDIR/chain.h5) to 60,000 and re-check.
# Stage C: continue in 20,000-step increments, re-checking each time, never beyond 100,000 steps.
# Stops at the first checkpoint that reports convergence (exit 0) or a serious sampling failure (exit 20), on any
# error, or at 100,000 steps. Never starts a second chain or seed. Re-running the script with the same OUTDIR
# resumes from the existing chain (completed stages are only re-checked).
#
# Decisions are logged to OUTDIR/driver.log; everything printed by Python also goes to OUTDIR/run.log.
# On exit OUTDIR/FINAL_SUMMARY.txt holds the decision log, the last checkpoint summary and the output paths.
set -u

OUTDIR=${1:?usage: $0 OUTDIR}
PY=/home/hongyp007/anaconda3/envs/yp/bin/python
CODE=/extdata01/hongyp007/projects/GRB/code/fitting/modeling
NPOOL=20
NWALKERS=64
SEED=20261008
STAGES=(30000 60000 80000 100000)
MAX_STEPS=100000

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE        # lets the chain be inspected read-only while the sampler writes
export PYTHONUNBUFFERED=1

cd "$CODE" || exit 1
mkdir -p "$OUTDIR"
DLOG="$OUTDIR/driver.log"
T_START=$(date +%s)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$DLOG"; }

finish() {
    local reason=$1
    local elapsed=$(( $(date +%s) - T_START ))
    log "STOP: $reason (driver wall-clock $(( elapsed / 3600 ))h $(( elapsed % 3600 / 60 ))m)"
    local last
    last=$(ls -d "$OUTDIR"/checkpoints/step_* 2>/dev/null | sort | tail -1)
    {
        echo "=== Early-phase production MCMC: final overnight summary ($(date '+%F %T')) ==="
        echo "Stop reason: $reason"
        echo "Driver wall-clock: $(( elapsed / 3600 ))h $(( elapsed % 3600 / 60 ))m"
        echo
        echo "--- Driver decision log ($DLOG) ---"
        cat "$DLOG"
        echo
        if [ -n "$last" ]; then
            echo "--- Last checkpoint summary ($last/summary.txt) ---"
            cat "$last/summary.txt"
            echo
        fi
        echo "--- Outputs ---"
        echo "result directory:       $OUTDIR"
        echo "HDF chain (resumable):  $OUTDIR/chain.h5"
        echo "run log:                $OUTDIR/run.log"
        echo "run configuration:      $OUTDIR/run_config.json"
        echo "initial walkers:        $OUTDIR/initial_walkers.npy (.csv)"
        echo "checkpoints:            $OUTDIR/checkpoints/step_NNNNNN/ (summary.txt, diagnostics.json, trace_params.png,"
        echo "                        trace_log_prob.png, acceptance_autocorr.png, key_correlations.png, key_marginals.png,"
        echo "                        corner_plot.png, bestfit_lc.png, spectral_index_comparison.png, correlation_matrix.csv)"
        echo "latest samples:         $OUTDIR/samples.npy, log_probs.npy, labels.txt, top_k_params.npy, bestfit_params.txt"
    } > "$OUTDIR/FINAL_SUMMARY.txt"
    log "Final summary: $OUTDIR/FINAL_SUMMARY.txt"
    exit 0
}

log "Driver start: OUTDIR=$OUTDIR, stages ${STAGES[*]}, max $MAX_STEPS steps, $NWALKERS walkers, $NPOOL workers, seed $SEED"
log "Host $(hostname), python $PY, pid $$"

for TARGET in "${STAGES[@]}"; do
    if [ "$TARGET" -gt "$MAX_STEPS" ]; then
        finish "stage $TARGET would exceed the $MAX_STEPS-step limit"
    fi
    if [ ! -f "$OUTDIR/chain.h5" ]; then
        log "Stage to $TARGET: new chain (ridge-aware start)"
        "$PY" early_phase.py --prior production --environment wind --xi both --nwalkers "$NWALKERS" \
            --npool "$NPOOL" --seed "$SEED" --nsteps "$TARGET" --outdir "$OUTDIR"
    else
        log "Stage to $TARGET: continuing the same chain $OUTDIR/chain.h5"
        "$PY" early_phase.py --resume "$OUTDIR" --nsteps "$TARGET" --npool "$NPOOL"
    fi
    RC=$?
    if [ "$RC" -ne 0 ]; then
        finish "sampler exited with code $RC at stage $TARGET (chain preserved in $OUTDIR/chain.h5)"
    fi

    log "Checkpoint diagnostics at $TARGET steps"
    "$PY" early_phase.py --check-chain "$OUTDIR" --final-check
    CODE_CHECK=$?
    case "$CODE_CHECK" in
        0)  finish "convergence criteria met at $TARGET steps" ;;
        10) if [ "$TARGET" -ge "$MAX_STEPS" ]; then
                finish "reached $MAX_STEPS steps without meeting the convergence criteria (not extended further)"
            fi
            log "Healthy but not converged at $TARGET steps: continuing the same chain" ;;
        20) finish "SERIOUS SAMPLING FAILURE reported at $TARGET steps (see the checkpoint summary)" ;;
        *)  finish "checkpoint diagnostics crashed with code $CODE_CHECK at $TARGET steps" ;;
    esac
done
finish "all stages done"
