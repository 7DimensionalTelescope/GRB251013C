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
from multiprocessing import Pool

# Keep each worker single-threaded so 64 pool workers don't oversubscribe the CPU
# with nested BLAS/OpenMP threads. Must be set before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import emcee

from VegasAfterglow import Scale

from grb.const import FIT_RESULTS_DIR, SI_FLARE_FRAC_MAX
from grb.likelihood import log_probability
from grb.modeling import load_all_optical_data
from grb.params import make_param_defs
from grb.plotting import plot_corner, plot_light_curves, plot_spectral_index_comparison
from grb.results import save_bestfit_params, save_run_arrays
from grb.spectral_index import load_xrt_spectral_index

# Initial positions (from best previous fits, adjusted for new constraints).
# NOTE: p_r (3.329) and E_iso_wing (3e51) sit outside their declared bounds in
# make_param_defs; the clip below pins those walkers to the boundary. Preserved
# verbatim from modeling/final_model.py so runs stay comparable.
INITIAL_GUESS = {
    "E_iso_core": 1.189e52,
    "Gamma0_core": 522,
    "theta_c_core": 0.02,  # Adjusted: within new 0.001-0.03 rad range (narrower core)
    "n_ism": 18.76,
    "p": 2.158,
    "eps_e": 0.0435,
    "eps_B": 0.0163,
    "xi": 0.943,
    "tau": 15.0,  # Adjusted: within new 5-30s range, reasonable for RS
    "p_r": 3.329,
    "eps_e_r": 0.0422,
    "eps_B_r": 0.20,  # Adjusted: within new 0.1-0.3 range, moderate value
    "xi_r": 0.849,
    "A_V": 0.0254,
    "t_start_flare": 3000,
    "tau_rise_flare": 300,
    "tau_decay_flare": 2000,
    "A_flare": 3e-10,
    "flare_beta": 0.8,
    "E_iso_wing": 3e51,
    "Gamma0_wing": 30,
    "theta_c_wing": 0.3,  # Adjusted: middle of new range (0.2-0.5 rad)
    "p_wing": 2.3,
    "eps_e_wing": 0.9,
    "eps_B_wing": 0.005,
    "xi_wing": 0.8,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Final model: ALL data + Core + Wing + RS + Norris flare")
    parser.add_argument("--include-flare", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--include-wing", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-spectral-index", default=True, action=argparse.BooleanOptionalAction,
                        help="Constrain the fit with the XRT spectral index (default: on)")
    parser.add_argument("--nsteps", type=int, default=3000)
    parser.add_argument("--nwalkers", type=int, default=None)
    parser.add_argument("--ncpus", type=int, default=64)
    parser.add_argument("--outdir", default=None)
    return parser.parse_args(argv)


def initial_positions(param_defs, nwalkers):
    """Draw walker start positions around INITIAL_GUESS, clipped to bounds."""
    pos0 = []
    for p in param_defs:
        if p.name in INITIAL_GUESS:
            center = INITIAL_GUESS[p.name]
            if p.scale is Scale.LOG:
                center_log = np.log10(center)
                pos0.append(np.random.normal(center_log, 0.3, nwalkers))
            else:
                pos0.append(np.random.normal(center, center * 0.2, nwalkers))
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


def main(argv=None):
    args = parse_args(argv)

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
    print(f"Include flare: {args.include_flare}")
    print(f"Include wing: {args.include_wing}")
    if xrt_index_data is not None:
        print(f"XRT spectral index: {len(xrt_index_data['time'])} points "
              f"(applied where core+wing dominate XRT, flare < {SI_FLARE_FRAC_MAX:.0%})")
    else:
        print("XRT spectral index: not used")

    # Setup parameters
    param_defs = make_param_defs(include_flare=args.include_flare, include_wing=args.include_wing)
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
                  xrt_index_data),
            pool=pool,
        )
        sampler.run_mcmc(pos0, args.nsteps, progress=True)

    # Save results
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    phase_name = "final"
    if args.include_flare:
        phase_name += "_flare"
    if args.include_wing:
        phase_name += "_wing"
    outdir = Path(args.outdir) if args.outdir else FIT_RESULTS_DIR / f"{phase_name}_{run_ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    samples = sampler.get_chain(flat=True)
    log_probs = sampler.get_log_prob(flat=True)

    top_params, top_log_probs = save_run_arrays(outdir, samples, log_probs, labels, top_k=10)

    print(f"\nBest log probability: {top_log_probs[0]:.3f}")
    print(f"Results saved to: {outdir}")

    save_bestfit_params(outdir, labels, param_defs, top_params, top_log_probs,
                        xrt_data, optical_datasets)

    # Plot light curves with best-fit model
    print("\nPlotting best-fit light curves...")
    plot_light_curves(outdir)

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
