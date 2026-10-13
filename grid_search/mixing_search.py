#!/usr/bin/env python3
"""Sobol-sequence exploration of the afterglow parameter space before MCMC.

This is "mixing mode": every sample mixes all parameters together in one
joint draw (contrast with sweep_search.py, which varies one parameter at a
time around a baseline). Draws a scrambled Sobol sequence over the existing
prior box (uniform in log10 for Scale.LOG parameters), scores every sample
with the existing log_prior + log_likelihood from `grb`, and saves ranked
numerical results plus light-curve plots for a small, systematically chosen
subset.

Not a sampler and not an optimizer: every sample is independent, and the
ranking is only meant to locate useful starting regions for MCMC.

Each run explores ONE circumburst environment (--environment ism|wind|stratified,
see grb.params.medium_param_defs); run the script once per environment to
compare them. Run from this directory (grid_search/, like `cd modeling` for
the modeling/ phase scripts), e.g.:

    cd grid_search
    python mixing_search.py --nsamples 100000 --ncpus 20 --seed 42
    python mixing_search.py --environment wind --nsamples 100000 --ncpus 20

Coupled physical-validity rules applied on top of the (unchanged) box
bounds, approved 2026-08-23:
  - eps_e + eps_B <= 1 in each shock region (FS, RS, wing)
  - theta_c_wing > theta_c_core (wing must be wider than the core)
  - tau_decay_flare > tau_rise_flare (flare must decay slower than it rises)

Validity is checked right after generation, before any model evaluation:
rejected draws are recorded and replaced by continuing the Sobol sequence,
so --nsamples means the number of physically valid samples evaluated.

Per-sample outcome is one of: success / physical_invalid / numerical_failure.
"""
from datetime import datetime
from pathlib import Path
import argparse
import json
import os
import sys
import warnings
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
from scipy.stats import qmc

from VegasAfterglow import Scale

from grb.const import XRT_BAND
from grb.likelihood import log_likelihood, log_prior
from grb.params import dataset_cal_factor
from grb.plotting import compute_model_components, set_log_y_limits

import grid_search_common
from grid_search_common import (
    DEFAULT_RESULTS_ROOT, STATUS_SUCCESS, STATUS_PHYSICAL_INVALID, STATUS_NUMERICAL_FAILURE,
    STATUS_NAMES, PHYSICAL_VALIDITY_RULES, add_common_args, sampled_bounds, theta_to_params,
    check_physical_validity, make_progress, _param_defs_json, load_run_inputs, _init_worker,
)

# Same panel conventions as grb.plotting.plot_light_curves
LC_COLORS = {"i-band": "red", "Leavitt_Rc": "darkorange", "Leavitt_Ic": "darkred"}
LC_OFFSETS = {"i-band": 1.0, "Leavitt_Rc": 1.5, "Leavitt_Ic": 2.0}
CURVE_COMPONENTS = ("total", "core_fs", "core_rs", "flare", "wing")
N_OVERLAY = 20  # models overlaid in the summary figure


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Sobol parameter-space exploration for the final afterglow model")
    parser.add_argument("--nsamples", type=int, default=100000,
                        help="Physically valid Sobol samples to evaluate; rejected draws are "
                             "replaced and recorded (powers of 2 are optimal, e.g. 131072)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the scrambled Sobol sequence")
    add_common_args(parser, outdir_help=f"Run directory (default: {DEFAULT_RESULTS_ROOT}/"
                                          "mixing_<environment>_<timestamp>)")
    parser.add_argument("--n-best-plots", type=int, default=200,
                        help="Individual plots for the highest-log_prob models")
    parser.add_argument("--n-broad-plots", type=int, default=200,
                        help="Individual plots spread across the full quality range (rank quantiles)")
    return parser.parse_args(argv)


def generate_valid_samples(param_defs, nsamples, seed, max_draw_factor=20):
    """Scrambled Sobol samples over the prior box, topped up until nsamples
    pass the physical-validity rules.

    Draws are checked BEFORE any model evaluation; rejected ones are replaced
    by continuing the same Sobol sequence, so the full nsamples budget is
    spent on valid samples. Returns (thetas, reasons) in draw order — every
    generated combination is kept, with reason '' for valid rows.
    Deterministic for a given seed.
    """
    lowers, uppers = sampled_bounds(param_defs)
    sobol = qmc.Sobol(d=len(param_defs), scramble=True, seed=seed)
    theta_blocks, reason_blocks = [], []
    n_valid = n_drawn = 0
    with warnings.catch_warnings():
        # Harmless for exploration: Sobol balance only holds at powers of 2
        warnings.filterwarnings("ignore", message=".*balance properties.*")
        while n_valid < nsamples:
            block = qmc.scale(sobol.random(nsamples - n_valid), lowers, uppers)
            reasons = check_physical_validity(block, param_defs)
            theta_blocks.append(block)
            reason_blocks.append(reasons)
            n_valid += int(np.count_nonzero(reasons == ""))
            n_drawn += len(block)
            if n_drawn > max_draw_factor * nsamples:
                raise RuntimeError(
                    f"Rejected {n_drawn - n_valid}/{n_drawn} draws; the validity rules "
                    "exclude almost the whole prior box")
    return np.concatenate(theta_blocks), np.concatenate(reason_blocks)


def _evaluate(task):
    """Score one pre-validated sample. Returns (idx, status, reason, log_prior, log_likelihood)."""
    idx, theta = task
    param_defs = grid_search_common._CTX["param_defs"]

    lp = log_prior(theta, param_defs)
    if not np.isfinite(lp):  # cannot happen for in-box Sobol samples; safety net
        return idx, STATUS_PHYSICAL_INVALID, "outside prior bounds", np.nan, np.nan

    # Single expensive model evaluation per sample: log_likelihood builds the
    # VegasAfterglow models once internally (compute_model_flux_all_bands) and
    # already reuses them for the spectral-index term. No light curve is
    # computed here; plotted models are re-evaluated later on the display grid.
    ll = log_likelihood(theta, param_defs, grid_search_common._CTX["xrt_data"],
                        grid_search_common._CTX["optical_datasets"],
                        grid_search_common._CTX["include_flare"], grid_search_common._CTX["include_wing"],
                        grid_search_common._CTX["xrt_index_data"], grid_search_common._CTX["flare_shape"])
    if not np.isfinite(ll):
        return idx, STATUS_NUMERICAL_FAILURE, "log_likelihood=-inf (model exception or non-finite chi2)", lp, -np.inf
    return idx, STATUS_SUCCESS, "", lp, ll


def _compute_and_render(task):
    """Compute display-grid curves for one plotted sample and render its figures.

    task = (idx, theta, render_jobs), where render_jobs is a list of
    (outfile, title) pairs — one sample can appear in both plots/best/ and
    plots/broad/, rendered from a single curve computation.

    Returns (idx, xrt[5, nt], opt[n_bands, 5, nt]) with components ordered as
    CURVE_COMPONENTS; optical curves include the dataset cal factor.
    """
    idx, theta, render_jobs = task
    ctx = grid_search_common._CTX
    params = theta_to_params(theta, ctx["param_defs"])
    include_flare, include_wing = ctx["include_flare"], ctx["include_wing"]
    flare_shape = ctx["flare_shape"]
    t_grid = ctx["t_grid"]

    comp = compute_model_components(params, t_grid, None, XRT_BAND, include_flare, include_wing,
                                    flare_shape)
    xrt = np.array([comp[c] for c in CURVE_COMPONENTS], dtype=np.float32)

    opt = np.empty((len(ctx["lc_bands"]), len(CURVE_COMPONENTS), len(t_grid)), dtype=np.float32)
    for i, (name, nu) in enumerate(ctx["lc_bands"]):
        comp = compute_model_components(params, t_grid, nu, None, include_flare, include_wing,
                                        flare_shape)
        cal = dataset_cal_factor(params, name)
        opt[i] = np.array([comp[c] for c in CURVE_COMPONENTS]) * cal

    for outfile, title in render_jobs:
        render_sample_figure(outfile, title, t_grid, xrt, opt, ctx["lc_bands"],
                             ctx["xrt_data"], ctx["lc_data"], include_flare, include_wing)
    return idx, xrt, opt


def select_broad_ranks(n_success, n_broad):
    """Deterministic rank-quantile selection: best -> worst, uniform in rank."""
    if n_success == 0 or n_broad <= 0:
        return np.array([], dtype=int)
    return np.unique(np.round(np.linspace(0, n_success - 1, min(n_broad, n_success))).astype(int))


def render_sample_figure(outfile, title, t_grid, xrt_curves, opt_curves, lc_bands,
                         xrt_data, lc_data, include_flare, include_wing):
    """Two-panel quick-look figure (XRT + optical LC), following plot_light_curves conventions."""
    fig, (ax_xrt, ax_opt) = plt.subplots(1, 2, figsize=(11, 4.5))
    t_grid_hr = t_grid / 3600
    comp_idx = {c: i for i, c in enumerate(CURVE_COMPONENTS)}

    # XRT panel
    ax_xrt.errorbar(xrt_data["time"] / 3600, xrt_data["flux"], yerr=xrt_data["flux_error"],
                    fmt=".", color="black", alpha=0.7, markersize=6, label="XRT data")
    ax_xrt.plot(t_grid_hr, xrt_curves[comp_idx["total"]], "k-", lw=1.2, label="Total", zorder=3)
    ax_xrt.plot(t_grid_hr, xrt_curves[comp_idx["core_fs"]], "k:", lw=1.2, label="Core FS")
    ax_xrt.plot(t_grid_hr, xrt_curves[comp_idx["core_rs"]], "c-.", lw=1.2, label="Reverse Shock")
    if include_wing and np.any(xrt_curves[comp_idx["wing"]] > 0):
        ax_xrt.plot(t_grid_hr, xrt_curves[comp_idx["wing"]], "b--", lw=1.2, label="Wing")
    if include_flare and np.any(xrt_curves[comp_idx["flare"]] > 0):
        ax_xrt.plot(t_grid_hr, xrt_curves[comp_idx["flare"]], color="red", ls=(0, (5, 1)),
                    lw=1.2, alpha=0.7, label="Flare")
    ax_xrt.set_xlabel("Time since trigger [hr]", fontsize=11)
    ax_xrt.set_ylabel(r"Flux [erg cm$^{-2}$ s$^{-1}$]", fontsize=11)
    ax_xrt.set_xscale("log")
    ax_xrt.set_yscale("log")
    ax_xrt.legend(fontsize=8)
    ax_xrt.grid(which="both", alpha=0.25)
    ax_xrt.set_title("XRT 0.3-10 keV", fontsize=12, fontweight="bold")
    set_log_y_limits(ax_xrt, xrt_data["flux"] - xrt_data["flux_error"],
                     xrt_data["flux"] + xrt_data["flux_error"], xrt_curves[comp_idx["total"]])

    # Optical panel
    opt_data_values = []
    for dataset in lc_data:
        color = LC_COLORS.get(dataset["name"], "slategray")
        offset = LC_OFFSETS.get(dataset["name"], 1.0)
        label = dataset["name"] + (f" (×{offset:.1f})" if offset != 1.0 else "")
        ax_opt.errorbar(dataset["time"] / 3600, dataset["flux_mJy"] * offset,
                        yerr=dataset["flux_err"] * offset,
                        fmt=".", color=color, alpha=0.7, markersize=6, label=label)
        opt_data_values.extend((dataset["flux_mJy"] - dataset["flux_err"]) * offset)
        opt_data_values.extend((dataset["flux_mJy"] + dataset["flux_err"]) * offset)

    for i, (name, _nu) in enumerate(lc_bands):
        color = LC_COLORS.get(name, "slategray")
        offset = LC_OFFSETS.get(name, 1.0)
        curves = opt_curves[i]
        ax_opt.plot(t_grid_hr, curves[comp_idx["total"]] * offset, color=color, lw=1.2, zorder=3)
        ax_opt.plot(t_grid_hr, curves[comp_idx["core_fs"]] * offset, color=color, ls=":", lw=1.2, alpha=0.7)
        ax_opt.plot(t_grid_hr, curves[comp_idx["core_rs"]] * offset, color=color, ls="-.", lw=1.2, alpha=0.7)
        if include_wing and np.any(curves[comp_idx["wing"]] > 0):
            ax_opt.plot(t_grid_hr, curves[comp_idx["wing"]] * offset, color=color, ls="--", lw=1.2, alpha=0.7)
        if include_flare and np.any(curves[comp_idx["flare"]] > 0):
            ax_opt.plot(t_grid_hr, curves[comp_idx["flare"]] * offset, color=color,
                        ls=(0, (5, 1)), lw=1.2, alpha=0.7)

    ax_opt.set_xlabel("Time since trigger [hr]", fontsize=11)
    ax_opt.set_ylabel("Flux density [mJy]", fontsize=11)
    ax_opt.set_xscale("log")
    ax_opt.set_yscale("log")
    ax_opt.legend(fontsize=8, ncol=2)
    ax_opt.grid(which="both", alpha=0.25)
    ax_opt.set_title("Optical Light Curves", fontsize=12, fontweight="bold")
    if opt_data_values:
        set_log_y_limits(ax_opt, *opt_data_values)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_summary(outfile, t_grid, curve_store, ranked_indices, log_prob,
                         lc_bands, xrt_data, lc_data):
    """Overlay the top-N total model curves on the XRT and i-band data."""
    n_over = min(N_OVERLAY, len(ranked_indices))
    if n_over == 0:
        return
    try:
        iband_pos = [name for name, _ in lc_bands].index("i-band")
    except ValueError:
        iband_pos = 0

    fig, (ax_xrt, ax_opt) = plt.subplots(1, 2, figsize=(11, 4.5))
    t_grid_hr = t_grid / 3600
    cmap = plt.get_cmap("viridis")
    total_i = CURVE_COMPONENTS.index("total")

    for rank_pos in range(n_over):
        idx = ranked_indices[rank_pos]
        xrt_c, opt_c = curve_store[idx]
        color = cmap(rank_pos / max(n_over - 1, 1))
        label = f"rank {rank_pos + 1} (logP={log_prob[idx]:.1f})" if rank_pos in (0, n_over - 1) else None
        ax_xrt.plot(t_grid_hr, xrt_c[total_i], color=color, lw=1.0, alpha=0.8, label=label)
        ax_opt.plot(t_grid_hr, opt_c[iband_pos, total_i], color=color, lw=1.0, alpha=0.8, label=label)

    ax_xrt.errorbar(xrt_data["time"] / 3600, xrt_data["flux"], yerr=xrt_data["flux_error"],
                    fmt=".", color="black", alpha=0.8, markersize=6, zorder=5, label="XRT data")
    iband_name = lc_bands[iband_pos][0]
    for dataset in lc_data:
        if dataset["name"] == iband_name:
            ax_opt.errorbar(dataset["time"] / 3600, dataset["flux_mJy"], yerr=dataset["flux_err"],
                            fmt=".", color="black", alpha=0.8, markersize=6, zorder=5,
                            label=f"{iband_name} data")
    set_log_y_limits(ax_xrt, xrt_data["flux"] - xrt_data["flux_error"],
                     xrt_data["flux"] + xrt_data["flux_error"])

    for ax, title, ylabel in ((ax_xrt, "XRT 0.3-10 keV", r"Flux [erg cm$^{-2}$ s$^{-1}$]"),
                              (ax_opt, f"Optical ({iband_name})", "Flux density [mJy]")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time since trigger [hr]", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(which="both", alpha=0.25)
        ax.set_title(title, fontsize=12, fontweight="bold")

    fig.suptitle(f"Top {n_over} models by log probability", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_logprob_vs_rank(outfile, ranked_log_probs):
    """Model quality falloff: -log_prob (log scale) vs rank."""
    if len(ranked_log_probs) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(ranked_log_probs) + 1), -ranked_log_probs, "k-", lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank", fontsize=11)
    ax.set_ylabel(r"$-\log$ probability", fontsize=11)
    ax.grid(which="both", alpha=0.25)
    ax.set_title(f"Best logP = {ranked_log_probs[0]:.2f} ({len(ranked_log_probs)} successful models)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_run_config(outdir, args, param_defs, run_ts, n_workers, counts=None,
                     rejection_reasons=None):
    config = {
        "run_timestamp": run_ts,
        "command_line": " ".join(sys.argv),
        "mode": "mixing",
        "nsamples": args.nsamples,
        "sampling_method": ("scrambled Sobol sequence (scipy.stats.qmc.Sobol); "
                            "physically invalid draws rejected before evaluation and replaced"),
        "seed": args.seed,
        "environment": args.environment,
        "include_flare": args.include_flare,
        "flare_shape": args.flare_shape,
        "include_wing": args.include_wing,
        "include_cal": args.include_cal,
        "cal_uncert": args.cal_uncert,
        "use_spectral_index": args.use_spectral_index,
        "ncpus_requested": args.ncpus,
        "n_workers": n_workers,
        "n_best_plots": args.n_best_plots,
        "n_broad_plots": args.n_broad_plots,
        "broad_selection_rule": ("ranks = unique(round(linspace(0, n_success-1, n_broad_plots))) "
                                 "over successes sorted by descending log_prob"),
        "physical_validity_rules": PHYSICAL_VALIDITY_RULES,
        "parameters": _param_defs_json(param_defs),
        "counts": counts,
        "rejection_reasons": rejection_reasons,
    }
    (Path(outdir) / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")


def main(argv=None):
    args = parse_args(argv)
    ctx = load_run_inputs(args)
    xrt_data, optical_datasets = ctx.xrt_data, ctx.optical_datasets
    xrt_index_data = ctx.xrt_index_data
    param_defs, labels, param_names = ctx.param_defs, ctx.labels, ctx.param_names
    lc_data, lc_bands, t_grid = ctx.lc_data, ctx.lc_bands, ctx.t_grid
    ndim = len(param_defs)
    n_cpus, n_workers = ctx.n_cpus, ctx.n_workers

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = (Path(args.outdir) if args.outdir
              else DEFAULT_RESULTS_ROOT / f"mixing_{args.environment}_{run_ts}")
    results_dir = outdir / "results"
    plot_dirs = {name: outdir / "plots" / name for name in ("best", "broad", "summary")}
    results_dir.mkdir(parents=True, exist_ok=True)
    for d in plot_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nEnvironment: {args.environment}")
    print(f"Parameters: {ndim} ({', '.join(param_names)})")
    print(f"Samples: {args.nsamples} (scrambled Sobol, seed {args.seed})")
    print(f"Include flare: {args.include_flare}" +
          (f" ({args.flare_shape})" if args.include_flare else "") +
          f", wing: {args.include_wing}, cal: {args.include_cal}")
    print(f"XRT spectral index: {'used' if xrt_index_data is not None else 'not used'}")
    print(f"Workers: {n_workers} (out of {n_cpus} CPUs)")
    print(f"Output: {outdir}")

    write_run_config(outdir, args, param_defs, run_ts, n_workers)

    print("\nGenerating samples (validity checked before evaluation; rejected draws replaced)...")
    thetas, gen_reasons = generate_valid_samples(param_defs, args.nsamples, args.seed)

    n_total = len(thetas)
    valid_idx = np.flatnonzero(gen_reasons == "")
    n_eval = len(valid_idx)  # == args.nsamples
    status = np.full(n_total, STATUS_PHYSICAL_INVALID, dtype=np.int8)
    status[valid_idx] = -1
    reasons = list(gen_reasons)
    log_prior_arr = np.full(n_total, np.nan)
    log_like_arr = np.full(n_total, np.nan)
    if n_total > n_eval:
        print(f"Generated {n_total} draws to reach {n_eval} valid samples "
              f"({n_total - n_eval} rejected and recorded)")

    init_args = (param_defs, xrt_data, optical_datasets, args.include_flare, args.include_wing,
                 xrt_index_data, args.flare_shape, t_grid, lc_bands)

    print("\nEvaluating samples...")
    progress = make_progress(n_eval)
    tally = Counter()
    with Pool(n_workers, initializer=_init_worker, initargs=init_args) as pool:
        for i, (idx, st, reason, lp, ll) in enumerate(
                pool.imap_unordered(_evaluate,
                                    [(int(k), thetas[k]) for k in valid_idx], chunksize=32)):
            status[idx] = st
            reasons[idx] = reason
            log_prior_arr[idx] = lp
            log_like_arr[idx] = ll
            tally[st] += 1
            if progress is not None:
                progress.update(1)
                if (i + 1) % 500 == 0 or i + 1 == n_eval:
                    progress.set_postfix(ok=tally[STATUS_SUCCESS],
                                         fail=tally[STATUS_NUMERICAL_FAILURE])
            elif (i + 1) % 5000 == 0 or i + 1 == n_eval:
                print(f"  {i + 1}/{n_eval} done "
                      f"(ok={tally[STATUS_SUCCESS]}, failed={tally[STATUS_NUMERICAL_FAILURE]})")

        if progress is not None:
            progress.close()

        log_prob = log_prior_arr + log_like_arr
        success_mask = status == STATUS_SUCCESS
        success_idx = np.flatnonzero(success_mask)
        order = np.argsort(-log_prob[success_idx], kind="stable")
        ranked_indices = success_idx[order]  # sample indices, best first
        n_success = len(ranked_indices)

        n_invalid = int(np.count_nonzero(status == STATUS_PHYSICAL_INVALID))
        n_failed = int(np.count_nonzero(status == STATUS_NUMERICAL_FAILURE))
        print(f"\nTotal generated:      {n_total}")
        print(f"Physically valid:     {n_total - n_invalid}")
        print(f"Physically rejected:  {n_invalid}")
        print(f"Numerical failures:   {n_failed}")
        print(f"Successful:           {n_success}")
        rejection_reasons = Counter(r for r, s in zip(reasons, status)
                                    if s == STATUS_PHYSICAL_INVALID)
        for reason, count in rejection_reasons.most_common():
            print(f"  rejected by {reason}: {count}")

        # --- numerical results ---
        phys = np.empty_like(thetas)
        for j, p in enumerate(param_defs):
            phys[:, j] = 10 ** thetas[:, j] if p.scale is Scale.LOG else thetas[:, j]

        np.savez_compressed(
            results_dir / "samples.npz",
            theta=thetas, log_prior=log_prior_arr, log_likelihood=log_like_arr,
            log_prob=log_prob, status=status, labels=np.array(labels),
            param_names=np.array(param_names),
        )

        table = {"sample_index": np.arange(n_total),
                 "status": [STATUS_NAMES[s] for s in status],
                 "reason": reasons,
                 "log_prior": log_prior_arr,
                 "log_likelihood": log_like_arr,
                 "log_prob": log_prob}
        for j, name in enumerate(param_names):
            table[name] = phys[:, j]
        pd.DataFrame(table).to_csv(results_dir / "results.csv", index=False)

        ranked = {"rank": np.arange(1, n_success + 1),
                  "sample_index": ranked_indices,
                  "log_prob": log_prob[ranked_indices],
                  "log_likelihood": log_like_arr[ranked_indices],
                  "log_prior": log_prior_arr[ranked_indices]}
        for j, name in enumerate(param_names):
            ranked[name] = phys[ranked_indices, j]
        pd.DataFrame(ranked).to_csv(results_dir / "ranked_results.csv", index=False)
        print(f"\nNumerical results saved to {results_dir}")

        # --- plotted subset: top-N best + rank-quantile broad, curves via the pool ---
        n_best = min(args.n_best_plots, n_success)
        best_ranks = np.arange(n_best)
        broad_ranks = select_broad_ranks(n_success, args.n_broad_plots)
        overlay_ranks = np.arange(min(N_OVERLAY, n_success))
        sel_ranks = np.unique(np.concatenate([best_ranks, broad_ranks, overlay_ranks])).astype(int)
        sel_samples = ranked_indices[sel_ranks]

        # Each selected sample may be rendered under several names (best + broad)
        render_jobs = {int(i): [] for i in sel_samples}
        for rank_pos in best_ranks:
            idx = int(ranked_indices[rank_pos])
            render_jobs[idx].append((
                str(plot_dirs["best"] / f"rank_{rank_pos + 1:04d}_sample_{idx:06d}.png"),
                f"rank {rank_pos + 1} | sample {idx} | logP = {log_prob[idx]:.2f}"))
        for q, rank_pos in enumerate(broad_ranks):
            idx = int(ranked_indices[rank_pos])
            render_jobs[idx].append((
                str(plot_dirs["broad"] / f"broad_q{q:04d}_rank_{rank_pos + 1:06d}_sample_{idx:06d}.png"),
                f"quantile {q} | rank {rank_pos + 1}/{n_success} | sample {idx} | "
                f"logP = {log_prob[idx]:.2f}"))

        curve_store = {}
        if len(sel_samples):
            print(f"\nComputing curves and rendering {n_best} best + {len(broad_ranks)} broad "
                  f"plots ({len(sel_samples)} models, parallel)...")
            tasks = [(int(i), thetas[i], render_jobs[int(i)]) for i in sel_samples]
            progress = make_progress(len(tasks))
            for idx, xrt_c, opt_c in pool.imap_unordered(_compute_and_render, tasks, chunksize=1):
                curve_store[idx] = (xrt_c, opt_c)
                if progress is not None:
                    progress.update(1)
            if progress is not None:
                progress.close()

    # pool closed; summary figures below run in the parent
    if curve_store:
        sel_list = sorted(curve_store)
        np.savez_compressed(
            results_dir / "model_curves.npz",
            t_grid=t_grid,
            sample_indices=np.array(sel_list),
            xrt_curves=np.stack([curve_store[i][0] for i in sel_list]),
            opt_curves=np.stack([curve_store[i][1] for i in sel_list]),
            component_names=np.array(CURVE_COMPONENTS),
            band_names=np.array([name for name, _ in lc_bands]),
            band_frequencies=np.array([nu for _, nu in lc_bands]),
        )

        plot_overlay_summary(plot_dirs["summary"] / "overlay_best.png", t_grid,
                             curve_store, ranked_indices, log_prob,
                             lc_bands, xrt_data, lc_data)
    else:
        print("\nNo successful models - skipping plots.")

    plot_logprob_vs_rank(plot_dirs["summary"] / "logprob_vs_rank.png", log_prob[ranked_indices])

    counts = {
        "total_generated": n_total,
        "physically_valid": n_total - n_invalid,
        "physically_rejected": n_invalid,
        "numerical_failures": n_failed,
        "successful": n_success,
    }
    write_run_config(outdir, args, param_defs, run_ts, n_workers, counts=counts,
                     rejection_reasons=dict(rejection_reasons))

    print("\n" + "=" * 60)
    print(f"✓ Mixing-mode search done: {outdir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
