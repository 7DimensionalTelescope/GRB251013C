#!/usr/bin/env python3
"""Early-phase fit CLI: configuration and dispatch only.

    early_phase_model.py        parameters, priors, physical constraints, model, chi2, likelihood, log_probability
    early_phase_optimize.py     shared robust optimisation utilities
    early_phase_diagnostics.py  mode optimisation and tau / p / xi_e_r / E_iso diagnostics (no MCMC)
    early_phase_mcmc.py         legacy sampler, production sampler (DE moves, ridge-aware start, HDF resume),
                                checkpoint diagnostics, run I/O
    early_phase_plotting.py     plotting

Production MCMC (wind, xi_e and xi_e_r free, see early_phase_model.PRODUCTION_BOUNDS):
    python early_phase.py --prior production --environment wind --xi both --nwalkers 64 --npool 20 --nsteps 30000
    python early_phase.py --resume DIR --nsteps 60000 --npool 20     # continue the SAME chain (DIR/chain.h5)
    python early_phase.py --check-chain DIR                          # checkpoint diagnostics; exit 0/10/20
"""
from datetime import datetime
from pathlib import Path
import json
import os
import sys

import argparse
import numpy as np

from VegasAfterglow import Scale  # noqa: F401  (re-exported for optimize_high_mode.py)
try:
    from VegasAfterglow.fitting.utils import _build_transformer
except ImportError:
    from VegasAfterglow.runner import _build_transformer

os.sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from grb.const import FIT_RESULTS_DIR
from grb.params import ENVIRONMENTS
from grb.prior import ParamDefWithPrior
from grb.plotting import plot_corner
import early_phase_model
from early_phase_model import (  # noqa: F401  (log_probability, _sampled_bounds, param_label: re-exported for optimize_high_mode.py)
    DEFAULT_ENVIRONMENT, HOST_AV_PARAM, PRIORS, SAMPLING_COORDS, XI_MODES, XRT_BAND,
    _evaluate_fn, _init_early_phase_context, _sampled_bounds, build_model_from_params, load_fit_data,
    log_probability, make_fitter, make_param_defs, make_priors, model_spectral_index, param_defs_for_saved_result,
    param_label, xi_flags,
)
from early_phase_diagnostics import (
    TAU_PROFILE_GRID, decompose_modes, optimize_constrained, optimize_eiso_sensitivity, optimize_free_xi_e_r,
    optimize_from_profile, optimize_high_mode, optimize_low_mode, run_p_profile, run_tau_profile, run_xi_profile,
)
from early_phase_mcmc import (
    RUN_CONFIG_FILE, best_params_from_result, check_chain, print_parameter_table, resolve_nwalkers,
    posterior_initial_state, ridge_aware_initial_state, run_custom_emcee, run_production_mcmc, save_bestfit_params, save_result_arrays,
)
from early_phase_plotting import plot_best_fit, plot_spectral_index_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Fit early XRT and i-band data with ISM FS/RS VEGASafterglow model.")
    parser.add_argument("--sampler", default="emcee", help="Sampler to use (emcee, bilby, etc.).")
    parser.add_argument("--npool", type=int, default=32, help="Parallel workers.")
    parser.add_argument("--nsteps", type=int, default=30000, help="Number of MCMC steps (default: 10000).")
    parser.add_argument("--nburn", type=int, default=None, help="Burn-in steps to discard in the legacy sampler (default: 5000). The production sampler uses a provisional, autocorrelation-based discard instead (see --check-chain).")
    parser.add_argument("--nwalkers", type=int, default=None, help="Number of walkers (default: max(2 * ndim, 32)); must be even and >= 2 * ndim.")
    parser.add_argument("--prior", choices=PRIORS, default="final", help="Prior bounds: 'final' = same as the final fit (default), 'wide' = original broad box with xi_e, xi_e_r free, 'production' = final with the production bounds (tau <= 3000 s, eps_e_r / eps_B_r <= 1, p >= 2.01, E_iso <= 1e55 erg; early_phase_model.PRODUCTION_BOUNDS). With 'production' the MCMC uses DE moves, a ridge-aware start and an HDF backend.")
    parser.add_argument("--environment", choices=ENVIRONMENTS, default=DEFAULT_ENVIRONMENT, help="Circumburst medium: ism (n_ism), wind (A_star; default) or stratified (A_star + n_ism + n0).")
    parser.add_argument("--tau-max", type=float, default=None, help="Override only the upper bound of tau [s] (default: the prior's own bound).")
    parser.add_argument("--optimize-high-mode", action="store_true", help="Diagnostic: run a local Powell optimisation from the known high-tau mode and exit (no MCMC).")
    parser.add_argument("--optimize-low-mode", action="store_true", help="Diagnostic: run a local Powell optimisation from the known low-tau mode and exit (no MCMC).")
    parser.add_argument("--decompose-modes", action="store_true", help="Diagnostic: optimise from both known modes and print the chi2 / prior / likelihood decomposition of each optimum (no MCMC).")
    parser.add_argument("--tau-profile", action="store_true", help="Diagnostic: profile the posterior over a fixed grid of tau (max over the other parameters at each tau), forward/reverse continuation, using --npool workers (no MCMC).")
    parser.add_argument("--eps-e-r-max", type=float, default=None, help="Override only the upper bound of eps_e_r (default: the prior's own bound).")
    parser.add_argument("--eps-B-r-max", type=float, default=None, help="Override only the upper bound of eps_B_r (default: the prior's own bound).")
    parser.add_argument("--optimize-profile-best", default=None, metavar="DIR", help="Diagnostic: re-optimise from the best row of DIR/tau_profile.csv at --profile-tau under the current bounds and exit (no MCMC).")
    parser.add_argument("--profile-tau", type=float, default=420.0, help="tau [s] of the tau-profile row used by --optimize-profile-best (default: 420).")
    parser.add_argument("--p-min", type=float, default=None, help="Override only the lower bound of p (default: the prior's own bound).")
    parser.add_argument("--p-profile", default=None, metavar="DIR", help="Diagnostic: profile the posterior over a fixed grid of p, starting from DIR/tau_profile.csv at --profile-tau and the eps_r-widened optimum, using --npool workers (no MCMC).")
    parser.add_argument("--optimize-constrained", default=None, metavar="DIR", help="Diagnostic: multi-start optimisation under eps_e + eps_B <= 1, started from the best row at --start-p of DIR/p_profile.csv, using --npool workers (no MCMC).")
    parser.add_argument("--start-p", type=float, default=2.01, help="p of the p-profile row used by --optimize-constrained (default: 2.01).")
    parser.add_argument("--xi", choices=XI_MODES, default="both", help="Electron acceleration fractions in the final / production prior: 'both' = xi_e and xi_e_r free (0.01-1, linear; default), 'reverse' = only xi_e_r free, 'none' = both fixed at 1. The older fixed-layout diagnostics select their own mode.")
    parser.add_argument("--free-xi-e-r", action="store_true", help="Alias for --xi reverse (only the reverse-shock xi_e_r free).")
    parser.add_argument("--optimize-free-xi-e-r", action="store_true", help="Diagnostic: multi-start optimisation with xi_e_r free (implies --free-xi-e-r), using --npool workers (no MCMC).")
    parser.add_argument("--xi-e-r-min", type=float, default=None, help="Lower bound of the free xi_e_r (default 0.01); only used when xi_e_r is free.")
    parser.add_argument("--xi-profile", action="store_true", help="Diagnostic: profile the posterior over a fixed grid of xi_e_r (implies --free-xi-e-r), forward/reverse continuation, using --npool workers (no MCMC).")
    parser.add_argument("--e-iso-max", type=float, default=None, help="Override only the upper bound of E_iso [erg] (default: the prior's own bound).")
    parser.add_argument("--eiso-sensitivity", default=None, metavar="DIR", help="Diagnostic: E_iso upper-bound sensitivity at fixed xi_e_r = 0.10/0.07/0.05, starting from DIR/xi_profile.csv (implies --free-xi-e-r), using --npool workers (no MCMC).")
    parser.add_argument("--nopt", type=int, default=100, help="Differential-evolution generations used to seed the walkers near the mode; 0 starts from the prior-box centre (default: 100).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--outdir", default=None, help="Output directory.")
    parser.add_argument("--plot-from", default=None, help="Existing run directory with top_k_params.npy to replot.")
    parser.add_argument("--corner-samples", type=int, default=20000, help="Maximum samples to use for corner plot.")
    parser.add_argument("--dry-run", action="store_true", help="Only load data and validate parameters.")
    parser.add_argument("--resume", default=None, metavar="DIR", help="Production: continue the HDF-backed chain DIR/chain.h5 to --nsteps TOTAL steps (configuration from DIR/run_config.json).")
    parser.add_argument("--check-chain", default=None, metavar="DIR", help="Production: checkpoint diagnostics of DIR/chain.h5; exit code 0 = converged, 10 = healthy but continue, 20 = serious failure.")
    parser.add_argument("--final-check", action="store_true", help="With --check-chain: also print the automatic final recommendation.")
    parser.add_argument("--init-only", action="store_true", help="Production: build, check and save the ridge-aware initial ensemble, then exit (no sampling).")
    parser.add_argument("--log-every", type=int, default=500, help="Production: progress line in run.log every N steps (default: 500).")
    parser.add_argument("--sampling-coords", choices=SAMPLING_COORDS, default="physical", help="Production: the sampler's coordinates. 'physical' (default) = the sampled parameters themselves; 'q_E_xir' = log10_E_iso replaced by q_E_xir = log10(E_iso * xi_e_r) (unit Jacobian, identical prior and likelihood); 'u_r' = p_r replaced by u_r = log10(p_r - 2), with the log-Jacobian added so the uniform p_r prior is unchanged (see early_phase_model). A resumed chain keeps its own coordinates.")
    parser.add_argument("--init-from", nargs="+", default=None, metavar="DIR", help="Production: start the walkers from posterior draws (steps in --init-window) of these earlier production chains with the same prior, instead of the ridge-aware anchors.")
    parser.add_argument("--init-window", nargs=2, type=int, default=(50000, 0), metavar=("START", "END"), help="Steps [START, END) of the --init-from chains to draw from (END 0 = to the end; default 50000 0).")
    parser.add_argument("--model-resolution", nargs=3, type=float, default=None, metavar=("PHI", "THETA", "T"), help="Model resolution for this run (default: early_phase_model.MODEL_RESOLUTION = (0.1, 0.25, 10), the historical production value). Recorded in run_config.json; --resume and --check-chain use the chain's own value.")
    parser.add_argument("--model-rtol", type=float, default=None, help="Model rtol for this run (default: early_phase_model.MODEL_RTOL = 1e-5).")
    parser.add_argument(
        "--use-spectral-index",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use XRT spectral index to constrain p via Gaussian prior (default: True).",
    )
    parser.add_argument(
        "--spectral-index-cooling",
        choices=("slow", "fast", "both", "smooth"),
        default="smooth",
        help="Spectral index calculation: 'slow' (p=2β+1), 'fast' (p=2β), 'both' (average), or 'smooth' (Granot & Sari 2001 - recommended).",
    )
    return parser.parse_args()


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


def start_log(log_dir, mode="w"):
    """Mirror stdout/stderr to <log_dir>/run.log from this point on."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / "run.log", mode)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)


def run_production(args, param_defs, xi_mode, xrt_data, i_data, xrt_index_data, outdir):
    """Production MCMC: print the parameter table, then (re)run the HDF-backed DE chain or only build its start."""
    if not (args.use_spectral_index and xrt_index_data is not None):
        raise ValueError("The production configuration includes the XRT spectral-index likelihood term.")
    param_kwargs = dict(prior=args.prior, environment=args.environment, tau_max=args.tau_max,
                        eps_e_r_max=args.eps_e_r_max, eps_B_r_max=args.eps_B_r_max, p_min=args.p_min,
                        free_xi_e_r=xi_flags(xi_mode)[1], xi_e_r_min=args.xi_e_r_min, e_iso_max=args.e_iso_max,
                        free_xi_e=xi_flags(xi_mode)[0])
    print(f"\n=== Production parameter set ({len(param_defs)} free parameters, order as sampled) ===")
    print_parameter_table(param_defs)
    print("xi_e -> forward-shock Radiation(xi_e=...), xi_e_r -> reverse-shock Radiation(xi_e=...) "
          "(early_phase_model.build_model_from_params)\n", flush=True)
    if args.init_only:
        from concurrent.futures import ProcessPoolExecutor
        nwalkers = resolve_nwalkers(args.nwalkers, len(param_defs))
        context = (param_defs, xrt_data, i_data, xrt_index_data, args.spectral_index_cooling)
        _init_early_phase_context(context)
        from early_phase_mcmc import _init_mcmc_worker
        numerics = (early_phase_model.MODEL_RESOLUTION, early_phase_model.MODEL_RTOL)
        with ProcessPoolExecutor(max_workers=args.npool, initializer=_init_mcmc_worker,
                                 initargs=(context, None, None, "physical", numerics)) as pool:
            if args.init_from:
                posterior_initial_state(param_defs, nwalkers, np.random.default_rng(args.seed), pool.map, outdir, args.init_from,
                                        discard=args.init_window[0], end=args.init_window[1] or None)
            else:
                ridge_aware_initial_state(param_defs, nwalkers, np.random.default_rng(args.seed), pool.map, outdir)
        return
    run_config = {"prior": args.prior, "environment": args.environment, "xi": xi_mode, "param_kwargs": param_kwargs,
                  "command": " ".join(sys.argv), "cwd": os.getcwd()}
    run_production_mcmc(param_defs, xrt_data, i_data, xrt_index_data, args.spectral_index_cooling, outdir,
                        nsteps=args.nsteps, nwalkers=args.nwalkers, npool=args.npool, seed=args.seed,
                        resume=bool(args.resume), run_config=run_config, log_every=args.log_every,
                        sampling_coords=args.sampling_coords, init_from=args.init_from,
                        init_window=(args.init_window[0], args.init_window[1] or None))
    print(f"Chain: {outdir / 'chain.h5'}; run checkpoint diagnostics with: --check-chain {outdir}")


def main():
    args = parse_args()
    if args.check_chain:
        start_log(Path(args.check_chain), mode="a")
        code, ckdir = check_chain(args.check_chain, final=args.final_check)
        print(f"Checkpoint written to: {ckdir}")
        sys.exit(code)
    if args.resume:
        saved = json.loads((Path(args.resume) / RUN_CONFIG_FILE).read_text())
        args.prior, args.environment, args.xi, args.free_xi_e_r = saved["prior"], saved["environment"], saved["xi"], False
        for key in ("tau_max", "eps_e_r_max", "eps_B_r_max", "p_min", "xi_e_r_min", "e_iso_max"):
            setattr(args, key, saved["param_kwargs"][key])
        args.outdir = args.resume
        if args.model_resolution is not None or args.model_rtol is not None:
            if (args.model_resolution is not None and [float(v) for v in args.model_resolution] != [float(v) for v in saved["model_resolution"]]) \
                    or (args.model_rtol is not None and float(args.model_rtol) != float(saved["model_rtol"])):
                raise ValueError(f"--resume keeps the chain's model resolution {saved['model_resolution']} / rtol "
                                 f"{saved['model_rtol']}; a different value was given on the command line")
        args.model_resolution, args.model_rtol = saved["model_resolution"], saved["model_rtol"]
    early_phase_model.set_numerics(args.model_resolution, args.model_rtol)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    diagnostic = (args.optimize_high_mode or args.optimize_low_mode or args.decompose_modes or args.optimize_profile_best
                  or args.optimize_constrained or args.optimize_free_xi_e_r)
    run_name = "early_phase_eisosens" if args.eiso_sensitivity else "early_phase_xiprofile" if args.xi_profile else "early_phase_pprofile" if args.p_profile else "early_phase_tauprofile" if args.tau_profile else "early_phase_optimize" if diagnostic else "early_phase_production" if args.prior == "production" else "early_phase"
    if args.tau_profile and args.tau_max is None:
        args.tau_max = float(max(TAU_PROFILE_GRID))
    outdir = Path(args.outdir) if args.outdir else FIT_RESULTS_DIR / f"{run_name}_{run_ts}"

    if not args.dry_run:
        if args.plot_from is None:
            start_log(outdir, mode="a" if args.resume else "w")
        else:
            start_log(Path(args.plot_from), mode="a")

    xrt_data, i_data, xrt_index_data = load_fit_data(include_spectral_index=args.use_spectral_index)
    fitter = make_fitter(xrt_data, i_data, args.environment)
    # The diagnostics were built for exact parameter layouts, so they select their own xi mode
    fixed_layout = (args.optimize_high_mode or args.optimize_low_mode or args.decompose_modes or args.tau_profile
                    or args.p_profile or args.optimize_profile_best or args.optimize_constrained)
    reverse_layout = args.optimize_free_xi_e_r or args.xi_profile or bool(args.eiso_sensitivity)
    xi_mode = "reverse" if args.free_xi_e_r else args.xi
    if fixed_layout and xi_mode != "none":
        print(f"Note: this diagnostic needs xi_e and xi_e_r fixed at 1; overriding --xi {xi_mode}.")
        xi_mode = "none"
    elif reverse_layout and xi_mode != "reverse":
        print(f"Note: this diagnostic needs only xi_e_r free; overriding --xi {xi_mode}.")
        xi_mode = "reverse"
    free_xi_e, free_xi_e_r = xi_flags(xi_mode)
    param_defs = make_param_defs(args.prior, args.environment, args.tau_max, args.eps_e_r_max, args.eps_B_r_max, args.p_min,
                                  free_xi_e_r=free_xi_e_r, xi_e_r_min=args.xi_e_r_min, e_iso_max=args.e_iso_max,
                                  free_xi_e=free_xi_e)
    vegas_param_defs = [p.to_param_def() if isinstance(p, ParamDefWithPrior) else p for p in param_defs]
    fitter.validate_parameters(vegas_param_defs)

    print(f"XRT points used: {len(xrt_data)}")
    print(f"i-band points used: {len(i_data)}")
    print(f"i-band time range: {i_data['time'].min():.1f} - {i_data['time'].max():.1f} s")
    print(f"XRT time range: {xrt_data['time'].min():.1f} - {xrt_data['time'].max():.1f} s")
    if args.use_spectral_index and xrt_index_data is not None:
        print(f"XRT spectral index points: {len(xrt_index_data['time'])}")
    print(f"Prior: {args.prior}, environment: {args.environment}, xi: {xi_mode if args.prior != 'wide' else 'wide prior (both free)'} "
          f"({len(param_defs)} free parameters)")
    print(f"Sampler: {args.sampler}, npool={args.npool}; model resolution {early_phase_model.MODEL_RESOLUTION}, "
          f"rtol {early_phase_model.MODEL_RTOL:g}")
    if args.plot_from is None:
        print(f"Output directory: {outdir}")
    else:
        print(f"Plot source directory: {args.plot_from}")

    if args.dry_run:
        print("Dry run complete: data loaded and parameters validated.")
        return

    if args.eiso_sensitivity:
        optimize_eiso_sensitivity(args.eiso_sensitivity, param_defs, xrt_data, i_data, xrt_index_data,
                                  args.spectral_index_cooling, outdir, args.npool, args.seed)
        return

    if args.xi_profile:
        run_xi_profile(param_defs, xrt_data, i_data, xrt_index_data, args.spectral_index_cooling, outdir, args.npool, args.seed)
        return

    if args.optimize_free_xi_e_r:
        optimize_free_xi_e_r(param_defs, xrt_data, i_data, xrt_index_data, args.spectral_index_cooling, args.npool, args.seed)
        return

    if args.optimize_constrained:
        optimize_constrained(args.optimize_constrained, args.start_p, param_defs, xrt_data, i_data, xrt_index_data,
                             args.spectral_index_cooling, args.npool, args.seed)
        return

    if args.p_profile:
        run_p_profile(args.p_profile, args.profile_tau, param_defs, xrt_data, i_data, xrt_index_data,
                      args.spectral_index_cooling, outdir, args.npool, args.seed)
        return

    if args.tau_profile:
        run_tau_profile(param_defs, xrt_data, i_data, xrt_index_data, args.spectral_index_cooling,
                        outdir, args.npool, args.seed)
        return

    if args.optimize_profile_best:
        optimize_from_profile(args.optimize_profile_best, args.profile_tau, param_defs, xrt_data, i_data,
                              xrt_index_data, args.spectral_index_cooling, args.npool, args.seed)
        return

    if diagnostic:
        diagnostic_args = (param_defs, xrt_data, i_data, xrt_index_data, args.spectral_index_cooling)
        if args.optimize_high_mode:
            optimize_high_mode(*diagnostic_args)
        if args.optimize_low_mode:
            optimize_low_mode(*diagnostic_args)
        if args.decompose_modes:
            decompose_modes(*diagnostic_args)
        return

    if args.plot_from is not None:
        plot_dir = Path(args.plot_from)
        param_defs, args.environment = param_defs_for_saved_result(plot_dir, args.prior, args.environment, args.tau_max, xi_mode)
        vegas_param_defs_plot = [p.to_param_def() if isinstance(p, ParamDefWithPrior) else p for p in param_defs]
        fitter._to_params = _build_transformer(vegas_param_defs_plot)
        best_params = np.load(plot_dir / "top_k_params.npy")[0]
        plot_best_fit(fitter, best_params, xrt_data, i_data, plot_dir, XRT_BAND, HOST_AV_PARAM,
                      evaluate_fn=_evaluate_fn(best_params, param_defs))
        save_bestfit_params(plot_dir, param_defs, xrt_data, i_data, args.environment)
        plot_corner(plot_dir, max_samples=args.corner_samples)
        print(f"Best-fit plot saved to: {plot_dir / 'bestfit_lc.png'}")
        print(f"Best-fit parameters saved to: {plot_dir / 'bestfit_params.txt'}")
        print(f"Corner plot saved to: {plot_dir / 'corner_plot.png'}")
        return

    if args.prior == "production":
        run_production(args, param_defs, xi_mode, xrt_data, i_data, xrt_index_data, outdir)
        return

    # Use custom emcee if spectral index constraint is enabled
    used_custom_emcee = args.use_spectral_index and xrt_index_data is not None
    
    if used_custom_emcee:
        print("Using custom emcee with per-measurement spectral index constraints...")
        result = run_custom_emcee(
            param_defs, xrt_data, i_data, xrt_index_data, 
            args.spectral_index_cooling, npool=args.npool,
            nsteps=args.nsteps, nburn=5000 if args.nburn is None else args.nburn, nopt=args.nopt, seed=args.seed,
            nwalkers=args.nwalkers,
        )
    else:
        result = fitter.fit(
            vegas_param_defs,
            sampler=args.sampler,
            npool=args.npool,
            outdir=str(outdir),
            label="early_phase",
            top_k=10,
            priors=make_priors(param_defs, 
                              xrt_index_data if args.use_spectral_index else None,
                              args.spectral_index_cooling),
        )
    
    # Create output directory only after successful fitting
    outdir.mkdir(parents=True, exist_ok=True)
    
    save_result_arrays(result, outdir)
    best_params = best_params_from_result(result)
    best_log_prob = np.nanmax(result.log_probs)

    save_bestfit_params(outdir, param_defs, xrt_data, i_data, args.environment)
    print(f"Best log probability: {best_log_prob:.3f}")
    print(f"Best-fit parameters saved to: {outdir / 'bestfit_params.txt'}")
    
    # Set up fitter for plotting (even when using custom emcee)
    if used_custom_emcee:
        # When using custom emcee, fitter wasn't fitted, so set up transformer manually
        vegas_param_defs_plot = [p.to_param_def() if isinstance(p, ParamDefWithPrior) else p for p in param_defs]
        fitter._to_params = _build_transformer(vegas_param_defs_plot)
    
    plot_best_fit(fitter, best_params, xrt_data, i_data, outdir, XRT_BAND, HOST_AV_PARAM,
                  evaluate_fn=_evaluate_fn(best_params, param_defs) if used_custom_emcee else None)
    print(f"Best-fit plot saved to: {outdir / 'bestfit_lc.png'}")
    
    plot_corner(outdir, max_samples=args.corner_samples)
    print(f"Corner plot saved to: {outdir / 'corner_plot.png'}")
    
    if args.use_spectral_index and xrt_index_data is not None:
        plot_spectral_index_comparison(
            best_params,
            result.labels,
            outdir,
            xrt_index_data,
            beta_model_fn=(
                (lambda times: model_spectral_index(build_model_from_params(best_params, param_defs)[0], times))
                if args.environment != "ism" else None
            ),
        )
    

if __name__ == "__main__":
    main()
