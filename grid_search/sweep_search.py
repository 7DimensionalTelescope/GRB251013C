#!/usr/bin/env python3
"""Parameter sweeps: how the model changes as each parameter changes.

Holds all parameters at a baseline and varies ONE parameter at a time across
its prior range, writing one family-of-light-curves-plus-spectrum figure per
parameter (XRT, optical light curve, and 7DT spectrum panels, each colored by
the swept value). Contrast with mixing_search.py, which draws joint Sobol
samples over the whole parameter space at once.

The baseline is INITIAL_GUESS (fit_final_model.py) by default, or the
best-fit parameters from a saved run directory via --baseline-dir. Run from
this directory (grid_search/, like `cd modeling` for the modeling/ phase
scripts), e.g.:

    cd grid_search
    python sweep_search.py --ncpus 20              # sweep every parameter
    python sweep_search.py eps_e n_ism --ncpus 8   # sweep just these
    python sweep_search.py --ncpus 20 \
        --baseline-dir /path/to/results/final_flare_20260831_202301
    python sweep_search.py --ncpus 20 \
        --baseline-file /path/to/results/sweep.txt

--baseline-file loads an `overrides = {...}` dict (the same format
manual_model.py/manual_model.ipynb use, e.g. one saved to results/sweep.txt).
Names it doesn't cover fall back to INITIAL_GUESS's fill rule (bounds
midpoint). Mutually exclusive with --baseline-dir.

Coupled physical-validity rules applied on top of the (unchanged) box
bounds, approved 2026-08-23:
  - eps_e + eps_B <= 1 in each shock region (FS, RS, wing)
  - theta_c_wing > theta_c_core (wing must be wider than the core)
  - tau_decay_flare > tau_rise_flare (flare must decay slower than it rises)

Sweep points that violate these rules are marked physical_invalid and
skipped in the family-of-curves figures (their log_likelihood is not
evaluated).

Per-point outcome is one of: success / physical_invalid / numerical_failure.
"""
from datetime import datetime
from pathlib import Path
import argparse
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

# Keep each worker single-threaded so pool workers don't oversubscribe the CPU
# with nested BLAS/OpenMP threads. Must be set before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

os.sys.path.append(os.path.join(os.path.dirname(__file__), "..", "fitting"))

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from VegasAfterglow import Scale

import grb.modeling
from grb.const import XRT_BAND
from grb.likelihood import log_likelihood
from grb.params import dataset_cal_factor
from grb.plotting import C_AA_PER_S, compute_model_components, set_log_y_limits
from grb.results import load_best_fit_params

import grid_search_common
from grid_search_common import (
    DEFAULT_RESULTS_ROOT, STATUS_SUCCESS, STATUS_PHYSICAL_INVALID, STATUS_NUMERICAL_FAILURE,
    STATUS_NAMES, PHYSICAL_VALIDITY_RULES, add_common_args, sampled_bounds, theta_to_params,
    params_to_theta, check_physical_validity, make_progress, _param_defs_json, load_run_inputs,
    _init_worker,
)

BASELINE_KEY = "__baseline__"  # pseudo parameter name for the sweep's reference point


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="One-at-a-time parameter sweep for the final afterglow model")
    parser.add_argument("sweep_params", nargs="*", default=[], metavar="PARAM",
                        help="Parameter name(s) to sweep across their full prior range, holding "
                             "all others at a baseline (default: every model parameter). Valid "
                             "names depend on --environment/--include-flare/--include-wing.")
    add_common_args(parser, outdir_help=f"Run directory (default: {DEFAULT_RESULTS_ROOT}/"
                                          "sweep_<environment>_<timestamp>)")
    parser.add_argument("--sweep-nvalues", type=int, default=15,
                        help="Values per swept parameter")
    # defaults matching the final early-afterglow MCMC / manual_model.ipynb baseline: wind medium, xi_e / xi_e_r free
    parser.set_defaults(environment="wind", free_xi_e=True, free_xi_e_r=True)
    baseline_group = parser.add_mutually_exclusive_group()
    baseline_group.add_argument("--baseline-dir", default=None, metavar="DIR",
                        help="Hold parameters at the best-fit sample from this saved run "
                             "directory (samples.npy/log_probs.npy/labels.txt, e.g. a "
                             "fit_final_model.py output dir) instead of INITIAL_GUESS in "
                             "fit_final_model.py. Must match the shape used by --flare-function's "
                             "run, if given.")
    baseline_group.add_argument("--baseline-file", default="/home/hongyp007/yp_folder/projects/GRB/code/grid_search/sweep.txt", metavar="FILE",
                        help="Hold parameters at the `overrides = {...}` dict in this file "
                             "(same format manual_model.py/manual_model.ipynb use, e.g. "
                             "results/sweep.txt) instead of INITIAL_GUESS in fit_final_model.py")
    return parser.parse_args(argv)


def load_baseline_file(path):
    """Load a partial parameter dict from an `overrides = {...}` file.

    Same format as the overrides cell in manual_model.py/manual_model.ipynb
    (a plain Python dict literal, comments allowed), e.g. one copied out to
    results/sweep.txt. Executed rather than parsed so comments and scientific
    notation just work; only the `overrides` name is read back.
    """
    namespace = {}
    exec(compile(Path(path).read_text(), str(path), "exec"), {}, namespace)
    if "overrides" not in namespace:
        raise ValueError(f"{path} does not define an `overrides` dict")
    return {k: float(v) for k, v in namespace["overrides"].items()}


def build_sweep_baseline(param_defs, baseline_dir=None, baseline_file=None):
    """Baseline parameter dict for the sweep, filtered to exactly the param_defs names,
    made in-bounds.

    Source is INITIAL_GUESS from fit_final_model.py by default, the best-fit sample
    from a saved run directory when baseline_dir is given (same loader grb.plotting
    uses: samples.npy/log_probs.npy/labels.txt), or the `overrides` dict from a file
    like results/sweep.txt when baseline_file is given (baseline_dir/baseline_file
    are mutually exclusive). Restricting to param_defs drops the wing/cal keys when
    those components are off (compute_model_components gates the wing on
    "E_iso_wing" being present). Names missing from the source (e.g. A_star for
    --environment wind, or any name a partial overrides file doesn't cover) get the
    bounds midpoint; out-of-bounds values are clipped. Returns (baseline, notes,
    source_desc) with one note per fill/clip applied.
    """
    if baseline_dir is not None:
        source, _include_flare, _include_wing, _flare_shape = load_best_fit_params(baseline_dir)
        source_desc = f"best-fit params from {baseline_dir}"
    elif baseline_file is not None:
        source = load_baseline_file(baseline_file)
        source_desc = f"overrides from {baseline_file}"
    else:
        from fit_final_model import INITIAL_GUESS  # lazy: pulls in emcee, sweep-only
        source = INITIAL_GUESS
        source_desc = "INITIAL_GUESS from fit_final_model.py"

    baseline, notes = {}, []
    for p in param_defs:
        if p.name in source:
            value = float(source[p.name])
            if not (p.lower <= value <= p.upper):
                if baseline_file is not None:
                    # like manual_model.ipynb: an explicit baseline is kept as given, only warned about
                    notes.append(f"{p.name}: baseline value {value:g} outside "
                                 f"[{p.lower:g}, {p.upper:g}], kept as given")
                else:
                    clipped = min(max(value, p.lower), p.upper)
                    notes.append(f"{p.name}: baseline value {value:g} outside "
                                 f"[{p.lower:g}, {p.upper:g}], clipped to {clipped:g}")
                    value = clipped
        else:
            value = (float(np.sqrt(p.lower * p.upper)) if p.scale is Scale.LOG
                     else 0.5 * (p.lower + p.upper))
            notes.append(f"{p.name}: not in baseline source, set to bounds midpoint {value:g}")
        baseline[p.name] = value
    return baseline, notes, source_desc


def sweep_values(p, nvalues, baseline_value):
    """Sweep grid across the full prior range of one parameter.

    geomspace for LOG parameters, linspace for LINEAR; the baseline value is
    force-included so every sweep contains the delta-logL = 0 anchor point.
    """
    if p.scale is Scale.LOG:
        grid = np.geomspace(p.lower, p.upper, nvalues)
    else:
        grid = np.linspace(p.lower, p.upper, nvalues)
    return np.unique(np.append(grid, baseline_value))


def _evaluate_sweep_point(task):
    """Score one sweep point and compute its display-grid total/FS/RS curves.

    task = (param_name, value_index, theta). Returns
    (param_name, value_index, status, reason, log_likelihood, xrt, opt, sed); each of
    xrt/opt/sed is a float32 (3, n) array stacking (total, core_fs, core_rs) — xrt/opt on
    _CTX["t_grid"] (optical = i-band including its cal factor), sed on _CTX["nu_grid"] at
    _CTX["sed_time"] — and None unless the status is success (sed is also None when there
    is no 7DT spectrum data to plot against).
    """
    param_name, value_index, theta = task
    ctx = grid_search_common._CTX
    param_defs = ctx["param_defs"]

    reason = check_physical_validity(np.asarray(theta)[None, :], param_defs)[0]
    if reason:
        return param_name, value_index, STATUS_PHYSICAL_INVALID, reason, np.nan, None, None, None

    ll = log_likelihood(theta, param_defs, ctx["xrt_data"], ctx["optical_datasets"],
                        ctx["include_flare"], ctx["include_wing"], ctx["xrt_index_data"],
                        ctx["flare_shape"])
    params = theta_to_params(theta, param_defs)
    t_grid = ctx["t_grid"]
    band_names = [name for name, _ in ctx["lc_bands"]]
    iband_pos = band_names.index("i-band") if "i-band" in band_names else 0
    iband_name, iband_nu = ctx["lc_bands"][iband_pos]
    try:
        xrt_comp = compute_model_components(params, t_grid, None, XRT_BAND,
                                            ctx["include_flare"], ctx["include_wing"],
                                            ctx["flare_shape"])
        xrt = np.stack([xrt_comp["total"], xrt_comp["core_fs"], xrt_comp["core_rs"]])
        opt_comp = compute_model_components(params, t_grid, iband_nu, None,
                                            ctx["include_flare"], ctx["include_wing"],
                                            ctx["flare_shape"])
        cal = dataset_cal_factor(params, iband_name)
        opt = np.stack([opt_comp["total"], opt_comp["core_fs"], opt_comp["core_rs"]]) * cal
        sed = None
        if ctx["sed_time"] is not None:
            nu_grid = ctx["nu_grid"]
            sed_times = np.full_like(nu_grid, ctx["sed_time"])
            sed_comp = compute_model_components(params, sed_times, nu_grid, None,
                                                ctx["include_flare"], ctx["include_wing"],
                                                ctx["flare_shape"])
            sed = np.stack([sed_comp["total"], sed_comp["core_fs"], sed_comp["core_rs"]])
    except Exception as e:
        return (param_name, value_index, STATUS_NUMERICAL_FAILURE,
                f"model exception: {e}", ll, None, None, None)
    ok = np.isfinite(ll) and np.all(np.isfinite(xrt)) and np.all(np.isfinite(opt))
    ok = ok and (sed is None or np.all(np.isfinite(sed)))
    if not ok:
        return (param_name, value_index, STATUS_NUMERICAL_FAILURE,
                "non-finite log_likelihood or model curve", ll, None, None, None)
    return (param_name, value_index, STATUS_SUCCESS, "", ll,
            np.asarray(xrt, dtype=np.float32), np.asarray(opt, dtype=np.float32),
            np.asarray(sed, dtype=np.float32) if sed is not None else None)


def render_sweep_figure(outfile, p, values, point_results, baseline_value,
                        baseline_xrt, baseline_opt, baseline_sed, logL0, t_grid, xrt_data,
                        iband_dataset, wavelength_grid, sed_time, sdt_data):
    """Family-of-curves figure for one swept parameter.

    point_results is aligned with values: (status, reason, log_likelihood, xrt, opt, sed)
    per sweep point, where xrt/opt/sed each stack (total, core_fs, core_rs) — see
    _evaluate_sweep_point. baseline_{xrt,opt,sed} are the same (3, n) stack for the
    baseline, or None if the baseline evaluation failed.

    All three panels overlay, for every swept value AND the baseline, the total model
    (solid) plus its forward-shock (dotted) and reverse-shock (dash-dot) components —
    same line styles as grb.plotting.render_light_curve_figure — colored by the
    parameter value (baseline in black) so both the total trend and the FS/RS trends
    are visible as the parameter varies. The spectrum panel (7DT flux density vs
    wavelength at sed_time) is left blank when there is no 7DT data (wavelength_grid
    is None).
    """
    is_log = p.scale is Scale.LOG
    norm = mcolors.LogNorm(p.lower, p.upper) if is_log else mcolors.Normalize(p.lower, p.upper)
    cmap = plt.get_cmap("viridis")
    t_grid_hr = t_grid / 3600

    fig = plt.figure(figsize=(15, 4.5), layout="constrained")
    ax_xrt, ax_opt, ax_sed = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 0.9]})

    def plot_family(ax, x, curves, color, lw_total, alpha_total, zorder):
        ax.plot(x, curves[0], color=color, lw=lw_total, alpha=alpha_total, zorder=zorder)
        ax.plot(x, curves[1], color=color, ls=":", lw=lw_total * 0.7, alpha=alpha_total * 0.6, zorder=zorder)
        ax.plot(x, curves[2], color=color, ls="-.", lw=lw_total * 0.7, alpha=alpha_total * 0.6, zorder=zorder)

    for v, (st, _reason, _ll, xrt_c, opt_c, sed_c) in zip(values, point_results):
        if st != STATUS_SUCCESS:
            continue
        color = cmap(norm(v))
        plot_family(ax_xrt, t_grid_hr, xrt_c, color, 1.0, 0.85, 2)
        plot_family(ax_opt, t_grid_hr, opt_c, color, 1.0, 0.85, 2)
        if sed_c is not None:
            plot_family(ax_sed, wavelength_grid, sed_c, color, 1.0, 0.85, 2)
    if baseline_xrt is not None:
        plot_family(ax_xrt, t_grid_hr, baseline_xrt, "black", 1.8, 1.0, 4)
        plot_family(ax_opt, t_grid_hr, baseline_opt, "black", 1.8, 1.0, 4)
        ax_xrt.plot([], [], color="black", lw=1.8, label="baseline (total)")
        ax_xrt.plot([], [], color="black", ls=":", lw=1.2, label="core FS")
        ax_xrt.plot([], [], color="black", ls="-.", lw=1.2, label="reverse shock")
        ax_opt.plot([], [], color="black", lw=1.8, label="baseline (total)")
        ax_opt.plot([], [], color="black", ls=":", lw=1.2, label="core FS")
        ax_opt.plot([], [], color="black", ls="-.", lw=1.2, label="reverse shock")
    if baseline_sed is not None:
        plot_family(ax_sed, wavelength_grid, baseline_sed, "black", 1.8, 1.0, 4)
        ax_sed.plot([], [], color="black", lw=1.8, label="baseline (total)")
        ax_sed.plot([], [], color="black", ls=":", lw=1.2, label="core FS")
        ax_sed.plot([], [], color="black", ls="-.", lw=1.2, label="reverse shock")

    ax_xrt.errorbar(xrt_data["time"] / 3600, xrt_data["flux"], yerr=xrt_data["flux_error"],
                    fmt=".", color="black", alpha=0.8, markersize=6, zorder=5, label="XRT data")
    ax_opt.errorbar(iband_dataset["time"] / 3600, iband_dataset["flux_mJy"],
                    yerr=iband_dataset["flux_err"], fmt=".", color="black", alpha=0.8,
                    markersize=6, zorder=5, label=f"{iband_dataset['name']} data")

    for ax, title, ylabel in ((ax_xrt, "XRT 0.3-10 keV", r"Flux [erg cm$^{-2}$ s$^{-1}$]"),
                              (ax_opt, f"Optical ({iband_dataset['name']})", "Flux density [mJy]")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time since trigger [hr]", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(which="both", alpha=0.25)
        ax.set_title(title, fontsize=12, fontweight="bold")
    set_log_y_limits(ax_xrt, xrt_data["flux"] - xrt_data["flux_error"],
                     xrt_data["flux"] + xrt_data["flux_error"])
    set_log_y_limits(ax_opt, iband_dataset["flux_mJy"] - iband_dataset["flux_err"],
                     iband_dataset["flux_mJy"] + iband_dataset["flux_err"])

    statuses = np.array([r[0] for r in point_results])
    ok = statuses == STATUS_SUCCESS
    n_valid, n_total = int(np.count_nonzero(ok)), len(values)

    if wavelength_grid is not None:
        sdt_wavelength = sdt_flux = sdt_flux_err = None
        if sdt_data:
            sdt_nu = np.array([d["frequency"] for d in sdt_data])
            sdt_wavelength = C_AA_PER_S / sdt_nu
            sdt_flux = np.array([d["flux_mJy"][0] for d in sdt_data])
            sdt_flux_err = np.array([d["flux_err"][0] for d in sdt_data])
            ax_sed.errorbar(sdt_wavelength, sdt_flux, yerr=sdt_flux_err, fmt=".",
                            color="black", alpha=0.8, markersize=6, zorder=5, label="7DT data")
        ax_sed.set_xscale("log")
        ax_sed.set_yscale("log")
        ax_sed.set_xlabel(r"Wavelength [$\AA$]", fontsize=11)
        ax_sed.set_ylabel("Flux density [mJy]", fontsize=11)
        ax_sed.legend(fontsize=8)
        ax_sed.grid(which="both", alpha=0.25)
        ax_sed.set_title(f"7DT spectrum at {sed_time / 3600:.2f} hr", fontsize=12, fontweight="bold")
        if sdt_flux is not None:
            set_log_y_limits(ax_sed, sdt_flux - sdt_flux_err, sdt_flux + sdt_flux_err)
    else:
        ax_sed.text(0.5, 0.5, "No 7DT spectrum data", ha="center", va="center",
                    transform=ax_sed.transAxes, fontsize=10, color="gray")
        ax_sed.set_xticks([])
        ax_sed.set_yticks([])

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=(ax_xrt, ax_opt, ax_sed),
                 label=p.name, pad=0.01)

    ref = f"logL0 = {logL0:.1f}" if np.isfinite(logL0) else "baseline evaluation failed"
    fig.suptitle(f"One-at-a-time sweep of {p.name} (others at INITIAL_GUESS, {ref}, "
                f"{n_valid}/{n_total} valid)", fontsize=11)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def write_sweep_config(outdir, args, param_defs, run_ts, n_workers, sweep_names,
                       baseline_params, baseline_notes, baseline_source, baseline_logL, counts):
    config = {
        "run_timestamp": run_ts,
        "command_line": " ".join(sys.argv),
        "mode": "sweep",
        "sweep_parameters": list(sweep_names),
        "sweep_nvalues": args.sweep_nvalues,
        "environment": args.environment,
        "include_flare": args.include_flare,
        "flare_shape": args.flare_shape,
        "include_wing": args.include_wing,
        "include_cal": args.include_cal,
        "free_xi_e": args.free_xi_e,
        "free_xi_e_r": args.free_xi_e_r,
        "cal_uncert": args.cal_uncert,
        "use_spectral_index": args.use_spectral_index,
        "ncpus_requested": args.ncpus,
        "n_workers": n_workers,
        "baseline": f"{baseline_source}, filtered to the model parameters",
        "baseline_params": baseline_params,
        "baseline_notes": baseline_notes,
        "baseline_log_likelihood": float(baseline_logL) if np.isfinite(baseline_logL) else None,
        "physical_validity_rules": PHYSICAL_VALIDITY_RULES,
        "parameters": _param_defs_json(param_defs),
        "counts": counts,
    }
    (Path(outdir) / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")


def main(argv=None):
    args = parse_args(argv)
    # Model grid of the final MCMC (as in manual_model.ipynb); Pool workers fork after this, so they inherit it.
    grb.modeling.MODEL_RESOLUTIONS = (0.3, 1.0, 30.0)
    ctx = load_run_inputs(args)
    param_defs, param_names = ctx.param_defs, ctx.param_names

    requested = args.sweep_params if args.sweep_params else param_names
    unknown = sorted(set(requested) - set(param_names))
    if unknown:
        sys.exit(f"Unknown sweep parameter(s): {', '.join(unknown)}. "
                 f"Valid names: {', '.join(param_names)}")
    sweep_names = [n for n in param_names if n in set(requested)]  # param_defs order, deduped

    baseline_params, baseline_notes, baseline_source = build_sweep_baseline(
        param_defs, args.baseline_dir, args.baseline_file)
    theta0 = params_to_theta(baseline_params, param_defs)
    print(f"Baseline: {baseline_source}")
    for note in baseline_notes:
        print(f"  Baseline note: {note}")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = (Path(args.outdir) if args.outdir
              else DEFAULT_RESULTS_ROOT / f"sweep_{args.environment}_{run_ts}")
    results_dir = outdir / "results"
    plots_dir = outdir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEnvironment: {args.environment}")
    print(f"Sweep parameters: {len(sweep_names)} ({', '.join(sweep_names)})")
    print(f"Values per parameter: {args.sweep_nvalues} (baseline value force-included)")
    print(f"Include flare: {args.include_flare}" +
          (f" ({args.flare_shape})" if args.include_flare else "") +
          f", wing: {args.include_wing}, cal: {args.include_cal}")
    print(f"XRT spectral index: {'used' if ctx.xrt_index_data is not None else 'not used'}")
    print(f"Workers: {ctx.n_workers} (out of {ctx.n_cpus} CPUs)")
    print(f"Output: {outdir}")

    col = {name: j for j, name in enumerate(param_names)}
    values_by_param = {}
    tasks = [(BASELINE_KEY, 0, theta0)]
    for name in sweep_names:
        p = param_defs[col[name]]
        values = sweep_values(p, args.sweep_nvalues, baseline_params[name])
        values_by_param[name] = values
        for i, v in enumerate(values):
            theta = theta0.copy()
            theta[col[name]] = np.log10(v) if p.scale is Scale.LOG else v
            tasks.append((name, i, theta))

    init_args = (param_defs, ctx.xrt_data, ctx.optical_datasets, args.include_flare,
                 args.include_wing, ctx.xrt_index_data, args.flare_shape, ctx.t_grid, ctx.lc_bands,
                 ctx.sed_time, ctx.nu_grid)

    print(f"\nEvaluating {len(tasks)} sweep points on {ctx.n_workers} workers...")
    results = {}
    progress = make_progress(len(tasks))
    with Pool(ctx.n_workers, initializer=_init_worker, initargs=init_args) as pool:
        for name, i, st, reason, ll, xrt_c, opt_c, sed_c in pool.imap_unordered(
                _evaluate_sweep_point, tasks, chunksize=1):
            results[(name, i)] = (st, reason, ll, xrt_c, opt_c, sed_c)
            if progress is not None:
                progress.update(1)
    if progress is not None:
        progress.close()

    base_st, base_reason, logL0, base_xrt, base_opt, base_sed = results.pop((BASELINE_KEY, 0))
    if base_st == STATUS_SUCCESS:
        print(f"\nBaseline logL = {logL0:.2f}")
    else:
        print(f"\nWarning: baseline evaluation failed ({STATUS_NAMES[base_st]}: {base_reason}); "
              "figures render without the baseline overlay")
        logL0 = np.nan

    band_names = [name for name, _ in ctx.lc_bands]
    iband_pos = band_names.index("i-band") if "i-band" in band_names else 0
    iband_dataset = ctx.lc_data[iband_pos]

    # --- numerical results (rows aligned between the CSV and sweep_curves.npz) ---
    nt = len(ctx.t_grid)
    nan_curve = np.full((3, nt), np.nan, dtype=np.float32)
    nan_sed = (np.full((3, len(ctx.nu_grid)), np.nan, dtype=np.float32)
              if ctx.nu_grid is not None else None)
    rows, xrt_rows, opt_rows, sed_rows = [], [], [], []
    tally = Counter()
    for name in sweep_names:
        p = param_defs[col[name]]
        for i, v in enumerate(values_by_param[name]):
            st, reason, ll, xrt_c, opt_c, sed_c = results[(name, i)]
            tally[st] += 1
            delta = ll - logL0 if (np.isfinite(ll) and np.isfinite(logL0)) else np.nan
            rows.append({"param": name, "value_index": i, "value": v,
                         "sampled_value": np.log10(v) if p.scale is Scale.LOG else v,
                         "status": STATUS_NAMES[st], "reason": reason,
                         "log_likelihood": ll, "delta_logL": delta})
            xrt_rows.append(xrt_c if xrt_c is not None else nan_curve)
            opt_rows.append(opt_c if opt_c is not None else nan_curve)
            if nan_sed is not None:
                sed_rows.append(sed_c if sed_c is not None else nan_sed)

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "sweep_results.csv", index=False)
    # (total, core_fs, core_rs) stacks -> split into separate named arrays for the npz
    xrt_stack = np.stack(xrt_rows)  # (n_rows, 3, nt)
    opt_stack = np.stack(opt_rows)
    base_xrt_stack = base_xrt if base_xrt is not None else nan_curve
    base_opt_stack = base_opt if base_opt is not None else nan_curve
    npz_payload = dict(
        t_grid=ctx.t_grid,
        param=df["param"].to_numpy(dtype=str),
        value=df["value"].to_numpy(),
        status=df["status"].to_numpy(dtype=str),
        xrt_total=xrt_stack[:, 0], xrt_fs=xrt_stack[:, 1], xrt_rs=xrt_stack[:, 2],
        opt_total=opt_stack[:, 0], opt_fs=opt_stack[:, 1], opt_rs=opt_stack[:, 2],
        baseline_xrt=base_xrt_stack[0], baseline_xrt_fs=base_xrt_stack[1], baseline_xrt_rs=base_xrt_stack[2],
        baseline_opt=base_opt_stack[0], baseline_opt_fs=base_opt_stack[1], baseline_opt_rs=base_opt_stack[2],
        baseline_logL=logL0,
        opt_band=np.array(band_names[iband_pos]),
    )
    if ctx.nu_grid is not None:
        sed_stack = np.stack(sed_rows)
        base_sed_stack = base_sed if base_sed is not None else nan_sed
        npz_payload.update(
            wavelength_grid=ctx.wavelength_grid,
            sed_time=ctx.sed_time,
            sed_total=sed_stack[:, 0], sed_fs=sed_stack[:, 1], sed_rs=sed_stack[:, 2],
            baseline_sed=base_sed_stack[0], baseline_sed_fs=base_sed_stack[1], baseline_sed_rs=base_sed_stack[2],
        )
    np.savez_compressed(results_dir / "sweep_curves.npz", **npz_payload)
    print(f"Numerical results saved to {results_dir}")

    print("\nRendering one figure per swept parameter...")
    for name in sweep_names:
        p = param_defs[col[name]]
        values = values_by_param[name]
        point_results = [results[(name, i)] for i in range(len(values))]
        render_sweep_figure(plots_dir / f"sweep_{name}.png", p, values, point_results,
                            baseline_params[name], base_xrt, base_opt, base_sed, logL0,
                            ctx.t_grid, ctx.xrt_data, iband_dataset,
                            ctx.wavelength_grid, ctx.sed_time, ctx.sdt_data)
    print(f"  {len(sweep_names)} figures in {plots_dir}")

    counts = {"total_points": len(rows),
              "successful": tally[STATUS_SUCCESS],
              "physically_rejected": tally[STATUS_PHYSICAL_INVALID],
              "numerical_failures": tally[STATUS_NUMERICAL_FAILURE]}
    write_sweep_config(outdir, args, param_defs, run_ts, ctx.n_workers, sweep_names,
                       baseline_params, baseline_notes, baseline_source, logL0, counts)

    print("\n" + "=" * 60)
    print(f"✓ Parameter sweep done: {outdir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
