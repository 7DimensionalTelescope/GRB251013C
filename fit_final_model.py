#!/usr/bin/env python3
"""Final model: ALL data + core jet + reverse shock + Norris flare + wing jet.

Thin driver over the `grb` package. Every piece of physics, data loading,
likelihood and plotting lives in `grb/`; this script only parses arguments,
seeds the walkers, runs emcee and writes the run directory.

Replaces the former standalone modeling/final_model.py.
"""
from datetime import datetime
from pathlib import Path
import argparse
import multiprocessing as mp
import os
import sys
from multiprocessing import Pool

# Keep each worker single-threaded so 64 pool workers don't oversubscribe the CPU
# with nested BLAS/OpenMP threads. Must be set before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import emcee

from VegasAfterglow import Scale

from grb.const import FIT_RESULTS_DIR, SI_FLARE_FRAC_MAX
from grb.functions import FLARE_SHAPES
from grb.likelihood import log_probability
from grb.modeling import load_all_optical_data
from grb.params import ENVIRONMENTS, NON_7DT_CAL_DATASETS, cal_param_name, make_param_defs
from grb.plotting import plot_corner, plot_light_curves, plot_spectral_index_comparison
from grb.results import save_bestfit_params, save_run_arrays
from grb.spectral_index import load_xrt_spectral_index

# Initial positions: joint re-optimization of the
# final_flare_wing_20260724_171919 best fit inside the widened bounds
# (logL = -548.0 under the current data + spectral-index likelihood ; scores
# -550.3 as rounded here vs -577.6 for the previous guess).
# NOTE: the previous guess had p_r=3.329 and E_iso_wing=3e51 OUTSIDE their
# own bounds, so every walker started clipped onto those walls.
INITIAL_GUESS = {
    # --- Core jet ---
    "E_iso_core": 9.94629e+52,
    "Gamma0_core": 30.0849,
    "theta_c_core": 0.0391698,

    # --- Circumburst medium (matches environment="wind" above) ---
    # "n_ism": 146.426,
    "A_star": 0.629797,   # uncomment for environment="wind" or "stratified"
    # "n0": 0.447214,        # uncomment for environment="stratified" only

    # --- Forward shock microphysics ---
    "p": 2.08961,
    "eps_e": 0.022096,
    "eps_B": 0.00523188,
    "tau": 22.0893,

    # --- Reverse shock ---
    "p_r": 2.1225,
    "eps_e_r": 0.440575,
    "eps_B_r": 0.495066,

    # --- Host extinction ---
    "A_V": 0.28018,

    # --- Flare (norris; the converged fit) ---
    "t_start_flare": 1139.18,
    "tau_rise_flare": 1222.5471,
    "tau_decay_flare": 4827.58,
    "A_flare": 9.92559e-10,
    "flare_beta": 0.820278,

    # --- Flare (sbpl; no converged fit yet -- rough seed
    # chosen so the profile peaks around the same t~3450s/F~1.1e-9 as the
    # norris fit above, not an optimum) ---
    "t_break_flare": 2300.0,
    "alpha_rise_flare": -3.0,
    "alpha_decay_flare": 2.0,
    "smooth_flare": 1.0,

    # --- Flare (gaussian; no converged fit yet -- rough seed matched to the
    # same norris peak time/amplitude) ---
    "t_peak_flare": 3468.0,
    "sigma_flare": 900.0,

    # Wing
    "E_iso_wing": 1.011e52,
    "Gamma0_wing": 19.2,
    "theta_c_wing": 0.492,
    "p_wing": 3.06,
    "eps_e_wing": 0.303,
    "eps_B_wing": 0.0121,
}
# Cross-cal scales (7DT reference); only used when --include-cal
for _name in NON_7DT_CAL_DATASETS:
    INITIAL_GUESS[cal_param_name(_name)] = 1.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Final model: ALL data + Core + Wing + RS + Norris flare")
    parser.add_argument("--environment", choices=ENVIRONMENTS, default="wind",
                        help="Circumburst medium (default: wind): ism / wind / stratified")
    parser.add_argument("--include-flare", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--flare-function", dest="flare_shape", choices=FLARE_SHAPES, default="norris",
                        help="Flare temporal profile shape when --include-flare (default: norris): norris / sbpl / gaussian")
    parser.add_argument("--include-wing", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--include-cal", default=False, action=argparse.BooleanOptionalAction,
                        help="Fit multiplicative cal scales for non-7DT optical LC datasets "
                             "(i-band, Leavitt_Rc/Ic); 7DT is the flux reference")
    parser.add_argument("--cal-uncert", type=float, default=0.1,
                        help="Half-width of each cal scale about 1 (default 0.1 => [0.9, 1.1]). Enabled when include_cal is True")
    parser.add_argument("--use-spectral-index", default=True, action=argparse.BooleanOptionalAction,
                        help="Constrain the fit with the XRT spectral index (default: on)")
    parser.add_argument("--mark-7dt-time", default=True, action=argparse.BooleanOptionalAction,
                        help="Draw a vertical line at the 7DT spectrum time in the Optical Light Curves panel (default: on)")
    parser.add_argument("--nsteps", type=int, default=34000)
    parser.add_argument("--nwalkers", type=int, default=None)
    parser.add_argument("--ncpus", type=int, default=32)
    parser.add_argument("--burnin", type=int, default=2000,
                        help="Number of initial MCMC steps to discard as burn-in before "
                             "saving samples and plotting (default: 0, no burn-in)")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args(argv)
    if args.burnin >= args.nsteps:
        parser.error(f"--burnin ({args.burnin}) must be less than --nsteps ({args.nsteps})")
    return args


def initial_positions(param_defs, nwalkers):
    """Draw walker start positions around INITIAL_GUESS, clipped to bounds."""
    pos0 = []
    for p in param_defs:
        if p.name in INITIAL_GUESS:
            center = INITIAL_GUESS[p.name]
            # The initial guess is a converged optimum (logL ~ -550), so scatter
            # walkers at roughly the posterior width (~0.1 dex) instead of the
            # 0.3 dex used when the guess was rough - otherwise the ensemble
            # starts ~50 logL downhill and wastes steps re-converging.
            if p.scale is Scale.LOG:
                center_log = np.log10(center)
                pos0.append(np.random.normal(center_log, 0.1, nwalkers))
            else:
                pos0.append(np.random.normal(center, abs(center) * 0.05, nwalkers))
        else:
            if p.scale is Scale.LOG:
                lower_log = np.log10(p.lower)
                upper_log = np.log10(p.upper)
                pos0.append(np.random.uniform(lower_log, upper_log, nwalkers))
            else:
                pos0.append(np.random.uniform(p.lower, p.upper, nwalkers))

    pos0 = np.array(pos0).T

    # Clip to bounds
    for i, p in enumerate(param_defs):
        if p.scale is Scale.LOG:
            lower_log = np.log10(p.lower)
            upper_log = np.log10(p.upper)
            pos0[:, i] = np.clip(pos0[:, i], lower_log, upper_log)
        else:
            pos0[:, i] = np.clip(pos0[:, i], p.lower, p.upper)

    return pos0


class Tee:
    """Duplicate writes to multiple streams (e.g. console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def main(argv=None):
    args = parse_args(argv)

    # Output directory (depends only on args, not on run results) is computed up
    # front so the run log can be captured from the very first print statement.
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    phase_name = f"final_{args.environment}"
    if args.include_flare:
        phase_name += f"_flare-{args.flare_shape}"
    if args.include_wing:
        phase_name += "_wing"
    if args.include_cal:
        phase_name += "_cal"
    outdir = Path(args.outdir) if args.outdir else FIT_RESULTS_DIR / f"{phase_name}_{run_ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    log_file = open(outdir / "run.log", "w")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    # Load ALL data
    print("Loading ALL data (XRT + all optical bands)...")
    xrt_data, optical_datasets = load_all_optical_data()

    # XRT spectral index (photon index) constraint
    xrt_index_data = None
    if args.use_spectral_index:
        try:
            xrt_index_data = load_xrt_spectral_index()
        except Exception as e:
            print(f"  Warning: could not load XRT spectral index ({e}); continuing without it")

    print(f"\nData loaded:")
    print(f"  XRT: {len(xrt_data['time'])} points ({xrt_data['time'].min()/3600:.2f}-{xrt_data['time'].max()/3600:.1f} hr)")
    for dataset in optical_datasets:
        print(f"  {dataset['name']}: {len(dataset['time'])} points " +
              f"({dataset['time'].min()/3600:.2f}-{dataset['time'].max()/3600:.1f} hr)")

    total_optical = sum(len(d['time']) for d in optical_datasets)
    print(f"\nTotal: {len(xrt_data['time'])} XRT + {total_optical} optical = {len(xrt_data['time']) + total_optical} points")
    print(f"Include flare: {args.include_flare}" +
          (f" (function: {args.flare_shape})" if args.include_flare else ""))
    print(f"Include wing: {args.include_wing}")
    if args.include_cal:
        print(f"Cross-cal (7DT reference): ±{args.cal_uncert:.0%} on {', '.join(NON_7DT_CAL_DATASETS)}")
    else:
        print("Cross-cal: not used")
    if xrt_index_data is not None:
        print(f"XRT spectral index: {len(xrt_index_data['time'])} points "
              f"(applied where core+wing dominate XRT, flare < {SI_FLARE_FRAC_MAX:.0%})")
    else:
        print("XRT spectral index: not used")

    # Setup parameters
    param_defs = make_param_defs(
        include_flare=args.include_flare,
        include_wing=args.include_wing,
        include_cal=args.include_cal,
        cal_uncert=args.cal_uncert,
        environment=args.environment,
        flare_shape=args.flare_shape,
    )
    labels = [f"log10_{p.name}" if p.scale is Scale.LOG else p.name for p in param_defs]
    ndim = len(labels)

    # Determine core budget first so we can size the walker ensemble to feed it.
    # emcee evaluates walkers in two half-batches, so effective parallelism is
    # capped at nwalkers/2 -> use ~2 walkers per worker.
    n_cpus = mp.cpu_count()
    n_workers = min(args.ncpus, n_cpus - 2)  # leave headroom on the shared machine
    nwalkers = args.nwalkers or max(2 * ndim, 2 * n_workers)
    nwalkers += nwalkers % 2  # emcee requires an even number of walkers
    n_workers = min(n_workers, nwalkers // 2)  # no worker should sit idle

    print(f"\nParameters: {ndim}")
    print(f"Walkers: {nwalkers}")
    print(f"Steps: {args.nsteps}")

    pos0 = initial_positions(param_defs, nwalkers)

    # Run MCMC with multiprocessing
    print("\nRunning MCMC...")
    print(f"Using {n_workers} CPU cores (out of {n_cpus} available)")

    with Pool(n_workers) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim,
            log_probability,
            args=(param_defs, xrt_data, optical_datasets, args.include_flare, args.include_wing,
                  xrt_index_data, args.flare_shape),
            pool=pool,
        )
        sampler.run_mcmc(pos0, args.nsteps, progress=True)

    tau = sampler.get_autocorr_time(tol=0)

    required_steps = 50 * np.max(tau)

    print("tau:", tau)
    print("nsteps:", sampler.iteration)
    print("50 * max(tau):", required_steps)

    if sampler.iteration > required_steps:
        print("Chain length is sufficient by the 50 tau criterion.")
    else:
        print("Warning: Chain is too short by the 50 tau criterion.")

    print(f"Burn-in: discarding first {args.burnin} of {sampler.iteration} steps")

    # Save results
    samples = sampler.get_chain(discard=args.burnin, flat=True)
    log_probs = sampler.get_log_prob(discard=args.burnin, flat=True)

    top_params, top_log_probs = save_run_arrays(
        outdir, samples, log_probs, labels, top_k=10,
        flare_shape=args.flare_shape if args.include_flare else None,
    )

    print(f"\nBest log probability: {top_log_probs[0]:.3f}")
    print(f"Results saved to: {outdir}")

    save_bestfit_params(outdir, labels, param_defs, top_params, top_log_probs,
                        xrt_data, optical_datasets, include_flare=args.include_flare,
                        include_wing=args.include_wing, flare_shape=args.flare_shape,
                        initial_guess=INITIAL_GUESS, environment=args.environment)

    # Plot light curves with best-fit model
    print("\nPlotting best-fit light curves...")
    plot_light_curves(outdir, mark_7dt_time=args.mark_7dt_time)

    if xrt_index_data is not None:
        print("Plotting spectral index comparison...")
        plot_spectral_index_comparison(outdir)

    plot_corner(outdir, labels, max_samples=20000)
    print(f"Corner plot saved to: {outdir / 'corner_plot.png'}")

    print("\n" + "=" * 60)
    print("✓ All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
