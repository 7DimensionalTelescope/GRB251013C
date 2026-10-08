#!/usr/bin/env python3
"""Diagnostic-only optimisations and profile scans of the early-phase posterior (no MCMC).

Mode optimisation / decomposition, tau / p / xi_e_r profiles, the eps_r prior-sensitivity and constrained
optimisations, and the E_iso upper-bound sensitivity test. Kept for reproducibility of the 2026-10-06/07
investigation; each needs the fixed parameter layout it was built for (checked at entry). All scoring goes
through early_phase_model.log_probability / posterior_decomposition.
"""
from datetime import datetime
from pathlib import Path

import numpy as np

from early_phase_model import (
    _init_early_phase_context,
    _log_probability_from_context,
    _neg_log_probability_from_context,
    _sampled_bounds,
    param_label,
    physical_param_value,
    posterior_decomposition,
)
from early_phase_optimize import (
    POLISH_BUDGET,
    PROFILE_BUDGET,
    _fixed_parameter_scan,
    _optimize_task,
    _profile_task,
    robust_minimize,
)
from early_phase_plotting import plot_p_profile, plot_tau_profile, plot_xi_profile


HIGH_MODE_LABELS = [
    "log10_E_iso", "log10_Gamma0", "log10_theta_c", "log10_A_star", "p", "log10_eps_e",
    "log10_eps_B", "log10_tau", "p_r", "log10_eps_e_r", "log10_eps_B_r", "log10_A_V",
]
HIGH_MODE_START = np.array([
    53.51618047, 1.41293979, -1.28564859, 0.61401606, 2.03718374, -1.96141243,
    -2.60643420, 2.56463493, 2.35476282, -1.03743966, -0.77058719, -0.38480586,
])


LOW_MODE_START = np.array([
    53.90769316, 1.46741698, -1.54401446, 0.61876206, 2.01569458, -2.28527625,
    -1.77066670, 1.35082087, 2.03987856, -0.31174235, -0.41545440, -0.58395182,
])


def optimize_mode(name, start, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling):
    """Local Powell optimisation of log_probability from a known mode (no MCMC); returns the optimum."""
    print(f"=== {name} ===")

    labels = [param_label(p) for p in param_defs]
    if labels != HIGH_MODE_LABELS:
        raise ValueError(
            "Parameter labels do not match the known mode start vector "
            "(needs --prior final --environment wind).\n"
            f"  expected: {HIGH_MODE_LABELS}\n  actual:   {labels}"
        )

    _init_early_phase_context((param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling))
    bounds = _sampled_bounds(param_defs)
    outside = [(l, v) for l, v, (lo, hi) in zip(labels, start, bounds) if not lo <= v <= hi]
    if outside:
        raise ValueError(f"Start vector lies outside the prior bounds for: {outside}")

    tau_i = labels.index("log10_tau")
    start_lp = _log_probability_from_context(start)
    print(f"tau prior range: {10 ** bounds[tau_i, 0]:.4g} - {10 ** bounds[tau_i, 1]:.4g} s")
    print(f"Start: log_prob = {start_lp:.6f}, tau = {10 ** start[tau_i]:.4f} s")

    x_opt, f_opt, n_evals, n_rounds, success, message = robust_minimize(
        _neg_log_probability_from_context, start, bounds, f_start=-start_lp
    )
    best_lp = -f_opt
    print(f"Optimized: log_prob = {best_lp:.6f}, tau = {10 ** x_opt[tau_i]:.4f} s")
    print(f"Delta log_prob (optimized - start) = {best_lp - start_lp:.6f}")
    print(f"Last Powell call: success = {success}, message = {message}")
    print(f"Function evaluations = {n_evals}, Powell/Nelder-Mead rounds = {n_rounds}")
    at_bound = [l for l, v, (lo, hi) in zip(labels, x_opt, bounds) if min(v - lo, hi - v) < 1e-6 * (hi - lo)]
    print(f"Parameters at a prior bound: {at_bound if at_bound else 'none'}")
    print(f"{'label':<16} {'start':>14} {'optimized':>14} {'physical':>14}")
    for param_def, label, v0, v1 in zip(param_defs, labels, start, x_opt):
        print(f"{label:<16} {v0:>14.8g} {v1:>14.8g} {physical_param_value(param_def, v1):>14.6g}")
    return x_opt


def optimize_high_mode(param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling):
    return optimize_mode("high-tau mode", HIGH_MODE_START, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)


def optimize_low_mode(param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling):
    return optimize_mode("low-tau mode", LOW_MODE_START, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)


def decompose_modes(param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling):
    """Optimise from the low- and high-tau modes and print the posterior decomposition of both optima."""
    optima = {
        "low-tau optimum": optimize_low_mode(param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling),
        "high-tau optimum": optimize_high_mode(param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling),
    }
    labels = [param_label(p) for p in param_defs]
    tau_i = labels.index("log10_tau")
    rows = {
        name: posterior_decomposition(x, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
        for name, x in optima.items()
    }
    names = list(rows)
    print("\n=== Posterior decomposition ===")
    print(f"{'':<26}" + "".join(f"{n:>20}" for n in names) + f"{'difference (high-low)':>24}")
    print(f"{'tau [s]':<26}" + "".join(f"{10 ** x[tau_i]:>20.4f}" for x in optima.values()))
    for key in rows[names[0]]:
        vals = [rows[n][key] for n in names]
        print(f"{key:<26}" + "".join(f"{v:>20.6f}" for v in vals) + f"{vals[1] - vals[0]:>24.6f}")
    n_xrt, n_i = len(xrt_data), len(i_data)
    n_si = len(xrt_index_data["time"]) if xrt_index_data is not None else 0
    print(f"data points: XRT flux {n_xrt}, i-band {n_i}, XRT spectral index {n_si}")



TAU_PROFILE_GRID = [
    10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300, 325, 350, 375, 400,
    410, 420, 430, 450, 475, 500, 550, 600, 700, 800, 1000, 1500, 2000, 3000,
]
# Optima found by optimize_mode from the two known modes (tau_max = 3000 s)
LOW_MODE_OPT = np.array([
    53.999999, 1.4951386, -1.646309, 0.56588684, 2.0100582, -2.089281,
    -1.7131403, 1.3768481, 2.0379043, -0.30103054, -0.30103052, -0.61169931,
])
HIGH_MODE_OPT = np.array([
    53.137766, 1.4209438, -1.4493202, 0.43972457, 2.015601, -1.5285896,
    -2.2212108, 2.6229423, 2.0408974, -0.30248063, -0.63690248, -0.34469906,
])
PROFILE_BOUND_FRACTION = 1e-3


def run_tau_profile(param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling, outdir, npool, seed):
    """Profile of the posterior over tau: max over the other parameters at fixed tau (no MCMC).

    Phase A optimises independently at every tau from the low/high optima (and perturbations of them).
    Phase B runs forward (low -> high tau) and reverse (high -> low tau) continuation scans, each seeded
    with the previous tau's solution plus the Phase A best. Independent, forward and reverse branches
    are all kept so hysteresis between local solutions can be seen.
    """
    import csv
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    labels = [param_label(p) for p in param_defs]
    if labels != HIGH_MODE_LABELS:
        raise ValueError(
            "The tau profile needs --prior final --environment wind.\n"
            f"  expected: {HIGH_MODE_LABELS}\n  actual:   {labels}"
        )
    tau_i = labels.index("log10_tau")
    bounds = _sampled_bounds(param_defs)
    taus = np.array(TAU_PROFILE_GRID, dtype=float)
    if taus.max() > 10 ** bounds[tau_i, 1] * (1 + 1e-9) or taus.min() < 10 ** bounds[tau_i, 0] * (1 - 1e-9):
        raise ValueError(f"tau grid {taus.min()}-{taus.max()} s exceeds the tau prior range "
                         f"{10 ** bounds[tau_i, 0]:.4g}-{10 ** bounds[tau_i, 1]:.4g} s; pass --tau-max.")
    log_taus = np.log10(taus)
    rng = np.random.default_rng(seed)
    budget = dict(PROFILE_BUDGET)
    context = (param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
    _init_early_phase_context(context)
    span = bounds[:, 1] - bounds[:, 0]
    n_taus = len(taus)

    def seed_at(x, k, perturb=0.0):
        y = np.clip(np.asarray(x, dtype=float), bounds[:, 0], bounds[:, 1])
        if perturb:
            y = np.clip(y + rng.normal(0.0, perturb * span), bounds[:, 0], bounds[:, 1])
        y[tau_i] = log_taus[k]
        return y

    def best_of(results):
        return min(results, key=lambda r: r["f"])

    print(f"tau profile: {n_taus} tau values, {taus.min():g}-{taus.max():g} s, {npool} workers, budget {budget}", flush=True)
    t_start = datetime.now()
    with ProcessPoolExecutor(max_workers=npool, initializer=_init_early_phase_context, initargs=(context,)) as pool:
        def submit(k, source, vec):
            return pool.submit(_profile_task, (log_taus[k], tau_i, source, vec, bounds, budget))

        # Phase A: independent optimisation at every tau
        futures = {k: [
            submit(k, "low-opt", seed_at(LOW_MODE_OPT, k)),
            submit(k, "high-opt", seed_at(HIGH_MODE_OPT, k)),
            submit(k, "low-opt-perturbed", seed_at(LOW_MODE_OPT, k, 0.02)),
            submit(k, "high-opt-perturbed", seed_at(HIGH_MODE_OPT, k, 0.02)),
        ] for k in range(n_taus)}
        indep = {}
        for k in range(n_taus):
            results = [f.result() for f in futures[k]]
            indep[k] = {**best_of(results), "nfev_total": sum(r["nfev"] for r in results)}
            print(f"  [A] tau={taus[k]:7.1f} s  log_post={-indep[k]['f']:10.3f}  ({indep[k]['source']})", flush=True)
        print(f"Phase A done after {datetime.now() - t_start}", flush=True)

        # Phase B: continuation scans
        def chain(order, name, first_seeds):
            out, prev = {}, None
            for k in order:
                seeds = [("indep-best", seed_at(indep[k]["x"], k))]
                if prev is None:
                    seeds += [(label, seed_at(vec, k)) for label, vec in first_seeds]
                else:
                    seeds += [(f"{name}-prev", seed_at(prev, k)), (f"{name}-prev-perturbed", seed_at(prev, k, 0.02))]
                results = [f.result() for f in [submit(k, label, vec) for label, vec in seeds]]
                out[k] = {**best_of(results), "nfev_total": sum(r["nfev"] for r in results)}
                prev = out[k]["x"]
                print(f"  [{name}] tau={taus[k]:7.1f} s  log_post={-out[k]['f']:10.3f}  ({out[k]['source']})", flush=True)
            return out

        with ThreadPoolExecutor(max_workers=2) as threads:
            fwd_future = threads.submit(chain, list(range(n_taus)), "forward", [("low-opt", LOW_MODE_OPT)])
            rev_future = threads.submit(chain, list(range(n_taus - 1, -1, -1)), "reverse", [("high-opt", HIGH_MODE_OPT)])
            forward, reverse = fwd_future.result(), rev_future.result()
    print(f"Phase B done after {datetime.now() - t_start}", flush=True)

    # Assemble rows (every branch kept; the overall best at each tau is flagged)
    branch_results = {"independent": indep, "forward": forward, "reverse": reverse}
    rows = []
    for k in range(n_taus):
        best_branch = min(branch_results, key=lambda b: branch_results[b][k]["f"])
        for branch, results in branch_results.items():
            r = results[k]
            dec = posterior_decomposition(r["x"], param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
            near = [l for l, v, (lo, hi) in zip(labels, r["x"], bounds)
                    if l != "log10_tau" and min(v - lo, hi - v) < PROFILE_BOUND_FRACTION * (hi - lo)]
            rows.append({
                "tau": taus[k], "branch": branch, "is_best": branch == best_branch, "source": r["source"],
                **dec, "nfev_best_run": r["nfev"], "nfev_total": r["nfev_total"],
                "near_prior_bound": ";".join(near),
                **{l: v for l, v in zip(labels, r["x"])},
            })

    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "tau_profile.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez(
        outdir / "tau_profile.npz", tau=taus, labels=np.array(labels),
        **{f"{b}_params": np.array([results[k]["x"] for k in range(n_taus)]) for b, results in branch_results.items()},
        **{f"{b}_log_posterior": np.array([-results[k]["f"] for k in range(n_taus)]) for b, results in branch_results.items()},
    )
    print(f"Saved: {outdir / 'tau_profile.csv'}, {outdir / 'tau_profile.npz'}")

    plot_tau_profile(rows, labels, param_defs, outdir)
    print_tau_profile_summary(rows, taus)


def print_tau_profile_summary(rows, taus):
    best = {r["tau"]: r for r in rows if r["is_best"]}
    by_branch = {b: {r["tau"]: r for r in rows if r["branch"] == b} for b in ("independent", "forward", "reverse")}
    lp = np.array([best[t]["log_posterior"] for t in taus])
    k_max = int(np.argmax(lp))
    print("\n=== tau profile summary ===")
    print(f"Profile maximum: tau = {taus[k_max]:g} s, log_posterior = {lp[k_max]:.3f} (branch {best[taus[k_max]]['branch']})")
    if 400 in taus:
        lp400 = lp[list(taus).index(400)]
        print(f"log_posterior at tau = 400 s: {lp400:.3f}; Delta (400 s - maximum) = {lp400 - lp[k_max]:.3f}")
    above = lp[k_max:]
    later_max = above[1:].max() if len(above) > 1 else float("nan")
    print(f"Above the maximum: log_posterior goes from {above[0]:.3f} to {above[-1]:.3f} at tau = {taus[-1]:g} s; "
          f"monotonic decrease = {bool(np.all(np.diff(above) <= 1e-6))}; largest later value = {later_max:.3f}")
    diff = np.array([abs(by_branch["forward"][t]["log_posterior"] - by_branch["reverse"][t]["log_posterior"]) for t in taus])
    spread = np.array([max(by_branch[b][t]["log_posterior"] for b in by_branch)
                       - min(by_branch[b][t]["log_posterior"] for b in by_branch) for t in taus])
    flagged = [f"{t:g}" for t, d in zip(taus, spread) if d > 0.5]
    print(f"Forward vs reverse: max |difference| = {diff.max():.3f} (at tau = {taus[int(np.argmax(diff))]:g} s); "
          f"branches (independent/forward/reverse) differing by > 0.5 at tau = {', '.join(flagged) if flagged else 'none'}")
    low = [t for t in taus if t <= 50]
    counts = {}
    for t in low:
        for name in filter(None, best[t]["near_prior_bound"].split(";")):
            counts[name] = counts.get(name, 0) + 1
    print(f"Parameters within {PROFILE_BOUND_FRACTION:g} of a prior bound in the best solutions at tau <= 50 s "
          f"({len(low)} points): {counts if counts else 'none'}")
    print(f"{'tau [s]':>9} {'best logP':>11} {'dlogP':>9} {'indep':>10} {'forward':>10} {'reverse':>10}  branch")
    for t in taus:
        print(f"{t:>9g} {best[t]['log_posterior']:>11.3f} {best[t]['log_posterior'] - lp[k_max]:>9.3f} "
              + "".join(f"{by_branch[b][t]['log_posterior']:>10.3f}" for b in by_branch) + f"  {best[t]['branch']}")


def optimize_from_profile(profile_dir, profile_tau, param_defs, xrt_data, i_data, xrt_index_data,
                          spectral_index_cooling, npool, seed):
    """Prior-sensitivity diagnostic: re-optimise from the best tau-profile solution under the given bounds.

    The start is the best row at `profile_tau` in <profile_dir>/tau_profile.csv. Two extra seeds (reverse-shock
    microphysics moved into the interior, and a small perturbation) check that the result is not an artefact
    of one optimiser path. No MCMC.
    """
    import csv
    from concurrent.futures import ProcessPoolExecutor

    labels = [param_label(p) for p in param_defs]
    if labels != HIGH_MODE_LABELS:
        raise ValueError(f"Needs --prior final --environment wind.\n  expected: {HIGH_MODE_LABELS}\n  actual:   {labels}")
    with open(Path(profile_dir) / "tau_profile.csv", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["is_best"] == "True" and abs(float(r["tau"]) - profile_tau) < 1e-6]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one best row at tau = {profile_tau} s in {profile_dir}, found {len(rows)}")
    previous = rows[0]
    start = np.array([float(previous[l]) for l in labels])

    bounds = _sampled_bounds(param_defs)
    context = (param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
    _init_early_phase_context(context)
    outside = [(l, v) for l, v, (lo, hi) in zip(labels, start, bounds) if not lo - 1e-9 <= v <= hi + 1e-9]
    if outside:
        raise ValueError(f"Start vector lies outside the prior bounds for: {outside}")
    start = np.clip(start, bounds[:, 0], bounds[:, 1])

    def decompose(x):
        return posterior_decomposition(x, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)

    start_dec = decompose(start)
    print(f"Prior bounds in use: " + ", ".join(
        f"{l}: [{10 ** lo:.4g}, {10 ** hi:.4g}]" if l.startswith("log10_") else f"{l}: [{lo:.4g}, {hi:.4g}]"
        for l, (lo, hi) in zip(labels, bounds) if l in ("log10_tau", "log10_eps_e_r", "log10_eps_B_r", "p")))
    print(f"Start (best tau-profile row at tau = {profile_tau:g} s, branch {previous['branch']}): "
          f"log_posterior = {start_dec['log_posterior']:.6f} (profile CSV: {float(previous['log_posterior']):.6f})", flush=True)

    rng = np.random.default_rng(seed)
    span = bounds[:, 1] - bounds[:, 0]
    interior = start.copy()
    for name in ("log10_eps_e_r", "log10_eps_B_r"):
        interior[labels.index(name)] = np.log10(0.8)
    perturbed = np.clip(start + rng.normal(0.0, 0.02 * span), bounds[:, 0], bounds[:, 1])
    seeds = [("profile-best", start), ("eps_r=0.8", np.clip(interior, bounds[:, 0], bounds[:, 1])), ("perturbed", perturbed)]

    with ProcessPoolExecutor(max_workers=min(npool, len(seeds)), initializer=_init_early_phase_context, initargs=(context,)) as pool:
        futures = [pool.submit(_optimize_task, (label, vec, bounds, {})) for label, vec in seeds]
        results = [f.result() for f in futures]
    for r in results:
        print(f"  seed {r['source']:<13} -> log_posterior = {-r['f']:.6f}, tau = {10 ** r['x'][labels.index('log10_tau')]:.3f} s, "
              f"{r['nfev']} evaluations, {r['rounds']} rounds")
    best = min(results, key=lambda r: r["f"])
    x = best["x"]
    dec = decompose(x)

    def near(x_vec, fraction):
        return [(l, v) for l, v, (lo, hi) in zip(labels, x_vec, bounds) if min(v - lo, hi - v) < fraction * (hi - lo)]

    tau_i, e_i, b_i = labels.index("log10_tau"), labels.index("log10_eps_e_r"), labels.index("log10_eps_B_r")
    print(f"\n=== Prior-sensitivity result (best seed: {best['source']}) ===")
    print(f"log_posterior: start {start_dec['log_posterior']:.6f} -> optimized {dec['log_posterior']:.6f} "
          f"(Delta = {dec['log_posterior'] - start_dec['log_posterior']:.6f})")
    print(f"tau: {10 ** start[tau_i]:.3f} s -> {10 ** x[tau_i]:.3f} s")
    print(f"eps_e_r: {10 ** start[e_i]:.5f} -> {10 ** x[e_i]:.5f} (upper bound {10 ** bounds[e_i, 1]:.4g})")
    print(f"eps_B_r: {10 ** start[b_i]:.5f} -> {10 ** x[b_i]:.5f} (upper bound {10 ** bounds[b_i, 1]:.4g})")
    print(f"{'':<26}{'previous (0.5/0.5)':>20}{'optimized':>14}{'difference':>14}")
    for key in ("chi2_XRT_flux", "chi2_i_band", "chi2_XRT_spectral_index", "chi2_total", "log_prior", "log_likelihood", "log_posterior"):
        print(f"{key:<26}{float(previous[key]):>20.4f}{dec[key]:>14.4f}{dec[key] - float(previous[key]):>14.4f}")
    print(f"{'label':<16} {'previous':>14} {'optimized':>14} {'physical':>14} {'lower':>12} {'upper':>12}")
    for param_def, label, v0, v1, (lo, hi) in zip(param_defs, labels, start, x, bounds):
        phys_lo, phys_hi = (10 ** lo, 10 ** hi) if label.startswith("log10_") else (lo, hi)
        print(f"{label:<16} {v0:>14.8g} {v1:>14.8g} {physical_param_value(param_def, v1):>14.6g} {phys_lo:>12.4g} {phys_hi:>12.4g}")
    at = near(x, 1e-6)
    close = near(x, 1e-3)
    print(f"At a prior bound (within 1e-6 of range): {[l for l, _ in at] if at else 'none'}")
    print(f"Within 1e-3 of range of a bound: {[l for l, _ in close] if close else 'none'}")
    for l in ("log10_eps_e_r", "log10_eps_B_r"):
        v, (lo, hi) = x[labels.index(l)], bounds[labels.index(l)]
        print(f"  {l}: {10 ** v:.5f}, {100 * (hi - v) / (hi - lo):.2f}% of the sampled range below the upper bound")
    return x


P_PROFILE_GRID = [2.001, 2.005, 2.01, 2.02, 2.03, 2.05, 2.10, 2.20]
# Optimum under eps_e_r / eps_B_r <= 1 (optimize_from_profile output, rounded to 8 digits)
HIGH_MODE_OPT_EPS_R_WIDE = np.array([
    53.097335, 1.4842638, -1.5106182, 0.13479393, 2.0110433, -1.4677059,
    -1.8289753, 2.6249877, 2.0318848, -0.13723442, -0.30188814, -0.35201006,
])


def run_p_profile(profile_dir, profile_tau, param_defs, xrt_data, i_data, xrt_index_data,
                  spectral_index_cooling, outdir, npool, seed):
    """Profile of the posterior over p: max over the other parameters with p held fixed (no MCMC).

    Same scheme as run_tau_profile (independent optimisations, then forward / reverse continuation, all
    branches kept) with more perturbed seeds per step so ~npool workers stay busy.
    """
    import csv
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    labels = [param_label(p) for p in param_defs]
    if labels != HIGH_MODE_LABELS:
        raise ValueError(f"Needs --prior final --environment wind.\n  expected: {HIGH_MODE_LABELS}\n  actual:   {labels}")
    p_i = labels.index("p")
    bounds = _sampled_bounds(param_defs)
    grid = np.array(P_PROFILE_GRID, dtype=float)
    if grid.min() < bounds[p_i, 0] - 1e-12 or grid.max() > bounds[p_i, 1] + 1e-12:
        raise ValueError(f"p grid {grid.min()}-{grid.max()} exceeds the p prior range "
                         f"{bounds[p_i, 0]}-{bounds[p_i, 1]}; pass --p-min.")
    with open(Path(profile_dir) / "tau_profile.csv", newline="") as fh:
        rows_csv = [r for r in csv.DictReader(fh) if r["is_best"] == "True" and abs(float(r["tau"]) - profile_tau) < 1e-6]
    if len(rows_csv) != 1:
        raise ValueError(f"Expected exactly one best row at tau = {profile_tau} s in {profile_dir}, found {len(rows_csv)}")
    csv_start = np.array([float(rows_csv[0][l]) for l in labels])

    rng = np.random.default_rng(seed)
    budget = dict(PROFILE_BUDGET)
    context = (param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
    _init_early_phase_context(context)
    span = bounds[:, 1] - bounds[:, 0]
    n_p = len(grid)
    base_opt = HIGH_MODE_OPT_EPS_R_WIDE

    def seed_at(x, k, perturb=0.0):
        y = np.clip(np.asarray(x, dtype=float), bounds[:, 0], bounds[:, 1])
        if perturb:
            y = np.clip(y + rng.normal(0.0, perturb * span), bounds[:, 0], bounds[:, 1])
        y[p_i] = grid[k]
        return y

    def best_of(results):
        return min(results, key=lambda r: r["f"])

    print(f"p profile: p = {grid.tolist()}, {npool} workers, budget {budget}", flush=True)
    t_start = datetime.now()
    with ProcessPoolExecutor(max_workers=npool, initializer=_init_early_phase_context, initargs=(context,)) as pool:
        def submit(k, source, vec):
            return pool.submit(_profile_task, (grid[k], p_i, source, vec, bounds, budget))

        # Phase A: independent optimisation at every p
        futures = {k: [
            submit(k, "eps_r-wide-opt", seed_at(base_opt, k)),
            submit(k, "tau-profile-420", seed_at(csv_start, k)),
            submit(k, "eps_r-wide-opt-perturbed-0.01", seed_at(base_opt, k, 0.01)),
            submit(k, "eps_r-wide-opt-perturbed-0.02", seed_at(base_opt, k, 0.02)),
            submit(k, "eps_r-wide-opt-perturbed-0.05", seed_at(base_opt, k, 0.05)),
            submit(k, "tau-profile-420-perturbed-0.02", seed_at(csv_start, k, 0.02)),
        ] for k in range(n_p)}
        indep = {}
        for k in range(n_p):
            results = [f.result() for f in futures[k]]
            indep[k] = {**best_of(results), "nfev_total": sum(r["nfev"] for r in results)}
            print(f"  [A] p={grid[k]:.3f}  log_post={-indep[k]['f']:12.3f}  ({indep[k]['source']})", flush=True)
        print(f"Phase A done after {datetime.now() - t_start}", flush=True)

        # Phase B: continuation scans
        def chain(order, name):
            out, prev = {}, None
            for k in order:
                seeds = [("indep-best", seed_at(indep[k]["x"], k)),
                         ("indep-best-perturbed-0.01", seed_at(indep[k]["x"], k, 0.01)),
                         ("eps_r-wide-opt", seed_at(base_opt, k))]
                if prev is not None:
                    seeds += [(f"{name}-prev", seed_at(prev, k)),
                              (f"{name}-prev-perturbed-0.01", seed_at(prev, k, 0.01)),
                              (f"{name}-prev-perturbed-0.03", seed_at(prev, k, 0.03)),
                              (f"{name}-prev-perturbed-0.05", seed_at(prev, k, 0.05))]
                results = [f.result() for f in [submit(k, label, vec) for label, vec in seeds]]
                out[k] = {**best_of(results), "nfev_total": sum(r["nfev"] for r in results)}
                prev = out[k]["x"]
                print(f"  [{name}] p={grid[k]:.3f}  log_post={-out[k]['f']:12.3f}  ({out[k]['source']})", flush=True)
            return out

        with ThreadPoolExecutor(max_workers=2) as threads:
            fwd_future = threads.submit(chain, list(range(n_p)), "forward")
            rev_future = threads.submit(chain, list(range(n_p - 1, -1, -1)), "reverse")
            forward, reverse = fwd_future.result(), rev_future.result()
    print(f"Phase B done after {datetime.now() - t_start}", flush=True)

    branch_results = {"independent": indep, "forward": forward, "reverse": reverse}
    best_lp = max(-r[k]["f"] for r in branch_results.values() for k in range(n_p))
    rows = []
    for k in range(n_p):
        best_branch = min(branch_results, key=lambda b: branch_results[b][k]["f"])
        k_lp = -branch_results[best_branch][k]["f"]
        for branch, results in branch_results.items():
            r = results[k]
            dec = posterior_decomposition(r["x"], param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
            near = [l for l, v, (lo, hi) in zip(labels, r["x"], bounds)
                    if l != "p" and min(v - lo, hi - v) < PROFILE_BOUND_FRACTION * (hi - lo)]
            rows.append({
                "p": grid[k], "branch": branch, "is_best": branch == best_branch, "source": r["source"],
                "delta_log_posterior": dec["log_posterior"] - best_lp, "tau": 10 ** r["x"][labels.index("log10_tau")],
                **dec, "nfev_best_run": r["nfev"], "nfev_total": r["nfev_total"],
                "near_prior_bound": ";".join(near),
                **{l: v for l, v in zip(labels, r["x"])},
            })
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "p_profile.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez(
        outdir / "p_profile.npz", p=grid, labels=np.array(labels),
        **{f"{b}_params": np.array([results[k]["x"] for k in range(n_p)]) for b, results in branch_results.items()},
        **{f"{b}_log_posterior": np.array([-results[k]["f"] for k in range(n_p)]) for b, results in branch_results.items()},
    )
    print(f"Saved: {outdir / 'p_profile.csv'}, {outdir / 'p_profile.npz'}")
    plot_p_profile(rows, outdir)
    print_p_profile_summary(rows, grid, bounds[p_i, 0])


def print_p_profile_summary(rows, grid, p_lower):
    best = {r["p"]: r for r in rows if r["is_best"]}
    by_branch = {b: {r["p"]: r for r in rows if r["branch"] == b} for b in ("independent", "forward", "reverse")}
    lp = np.array([best[p]["log_posterior"] for p in grid])
    k_max = int(np.argmax(lp))
    print("\n=== p profile summary ===")
    print(f"Best p = {grid[k_max]:g}, log_posterior = {lp[k_max]:.3f}")
    print(f"Delta log P (p = 2.01 - best tested p) = {lp[list(grid).index(2.01)] - lp[k_max]:.3f}")
    below = [p for p in grid if p < 2.01]
    print("Profile as p approaches 2 from above: " + ", ".join(f"p={p:g}: {best[p]['log_posterior']:.3f}" for p in grid[::-1] if p <= 2.1))
    diffs = np.diff(lp[np.argsort(grid)][::-1])
    print(f"Monotonic improvement towards lower p over all tested values = {bool(np.all(np.diff(lp) <= 1e-6))}")
    print(f"tau at the best solutions: " + ", ".join(f"p={p:g}: {best[p]['tau']:.1f} s" for p in grid))
    flagged = {p: best[p]["near_prior_bound"] for p in grid if best[p]["near_prior_bound"]}
    print(f"Parameters within {PROFILE_BOUND_FRACTION:g} of a prior bound (p itself excluded): {flagged if flagged else 'none'}")
    spread = {p: max(by_branch[b][p]["log_posterior"] for b in by_branch) - min(by_branch[b][p]["log_posterior"] for b in by_branch) for p in grid}
    print(f"Branch spread (max - min log_posterior over independent/forward/reverse): "
          + ", ".join(f"p={p:g}: {d:.2f}" for p, d in spread.items()))
    print(f"{'p':>7} {'best logP':>11} {'dlogP':>9} {'tau [s]':>9} {'chi2_XRT':>9} {'chi2_i':>9} {'chi2_beta':>9} {'indep':>10} {'forward':>10} {'reverse':>10}  branch")
    for p in grid:
        b = best[p]
        print(f"{p:>7g} {b['log_posterior']:>11.3f} {b['delta_log_posterior']:>9.3f} {b['tau']:>9.1f} {b['chi2_XRT_flux']:>9.2f} "
              f"{b['chi2_i_band']:>9.2f} {b['chi2_XRT_spectral_index']:>9.2f} "
              + "".join(f"{by_branch[x][p]['log_posterior']:>10.3f}" for x in by_branch) + f"  {b['branch']}")


def optimize_constrained(profile_dir, start_p, param_defs, xrt_data, i_data, xrt_index_data,
                         spectral_index_cooling, npool, seed):
    """Robust multi-start optimisation under eps_e + eps_B <= 1 (enforced by log_prior); no MCMC.

    Seeds: the best unconstrained row at p = start_p of <profile_dir>/p_profile.csv projected into the
    allowed region, eps_e_r / eps_B_r = (0.5, 0.5), (0.7, 0.3), (0.3, 0.7) on two sets of the other
    parameters, and small perturbations. The three best results are then polished with a longer budget.
    """
    import csv
    from concurrent.futures import ProcessPoolExecutor

    labels = [param_label(p) for p in param_defs]
    if labels != HIGH_MODE_LABELS:
        raise ValueError(f"Needs --prior final --environment wind.\n  expected: {HIGH_MODE_LABELS}\n  actual:   {labels}")
    with open(Path(profile_dir) / "p_profile.csv", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["is_best"] == "True" and abs(float(r["p"]) - start_p) < 1e-9]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one best row at p = {start_p} in {profile_dir}, found {len(rows)}")
    previous = rows[0]
    unconstrained = np.array([float(previous[l]) for l in labels])
    bounds = _sampled_bounds(param_defs)
    context = (param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
    _init_early_phase_context(context)
    span = bounds[:, 1] - bounds[:, 0]
    e_i, b_i = labels.index("log10_eps_e_r"), labels.index("log10_eps_B_r")
    rng = np.random.default_rng(seed)

    def project(x):
        """Clip to the box and, if eps_e_r + eps_B_r > 1, rescale the pair onto the constraint surface."""
        y = np.clip(np.asarray(x, dtype=float), bounds[:, 0], bounds[:, 1])
        e, b = 10 ** y[e_i], 10 ** y[b_i]
        if e + b > 1.0:
            scale = (1.0 - 1e-6) / (e + b)
            y[e_i], y[b_i] = np.log10(e * scale), np.log10(b * scale)
        return np.clip(y, bounds[:, 0], bounds[:, 1])

    def with_pair(x, e, b):
        y = np.array(x, dtype=float)
        y[e_i], y[b_i] = np.log10(e), np.log10(b)
        return project(y)

    def perturbed(x, scale):
        return project(np.asarray(x) + rng.normal(0.0, scale * span))

    other = np.clip(HIGH_MODE_OPT_EPS_R_WIDE, bounds[:, 0], bounds[:, 1])
    seeds = []
    for set_name, base in (("best-unconstrained", unconstrained), ("eps_r-wide-opt", other)):
        for pair_name, pair in (("projected", None), ("0.5/0.5", (0.5, 0.5)), ("0.7/0.3", (0.7, 0.3)), ("0.3/0.7", (0.3, 0.7))):
            vec = project(base) if pair is None else with_pair(base, *pair)
            seeds.append((f"{set_name}:{pair_name}", vec))
    for label, vec in list(seeds[:4]):                      # perturb the best-unconstrained seeds
        seeds += [(f"{label}+perturbed-0.01", perturbed(vec, 0.01)), (f"{label}+perturbed-0.03", perturbed(vec, 0.03))]
    for label, vec in list(seeds[4:8]):                     # and the alternative set once
        seeds.append((f"{label}+perturbed-0.02", perturbed(vec, 0.02)))
    for label, vec in seeds:
        assert np.isfinite(_log_probability_from_context(vec)), f"seed {label} is infeasible"
    print(f"Constrained optimisation: {len(seeds)} seeds on {min(npool, len(seeds))} workers; unconstrained reference "
          f"(p = {start_p:g}) log_posterior = {float(previous['log_posterior']):.6f}", flush=True)

    t_start = datetime.now()
    with ProcessPoolExecutor(max_workers=npool, initializer=_init_early_phase_context, initargs=(context,)) as pool:
        futures = [pool.submit(_optimize_task, (label, vec, bounds, dict(PROFILE_BUDGET))) for label, vec in seeds]
        results = [f.result() for f in futures]
        results.sort(key=lambda r: r["f"])
        for r in results:
            print(f"  [phase 1] {r['source']:<38} log_posterior = {-r['f']:10.4f}  (eps_e_r+eps_B_r = "
                  f"{10 ** r['x'][e_i] + 10 ** r['x'][b_i]:.4f}, {r['nfev']} evaluations)", flush=True)
        print(f"Phase 1 done after {datetime.now() - t_start}", flush=True)
        polish_budget = {"max_rounds": 15, "tol": 1e-3, "powell_maxfev": 20000, "nm_maxfev": 6000}
        polished = [pool.submit(_optimize_task, (f"polish-of-{r['source']}", r["x"], bounds, polish_budget)) for r in results[:3]]
        polished = [f.result() for f in polished]
        for r in polished:
            print(f"  [phase 2] {r['source']:<50} log_posterior = {-r['f']:10.4f}  ({r['nfev']} evaluations)", flush=True)
    print(f"Phase 2 done after {datetime.now() - t_start}", flush=True)

    best = min(results + polished, key=lambda r: r["f"])
    x = best["x"]
    dec = posterior_decomposition(x, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
    ref = {k: float(previous[k]) for k in dec}
    tau_i = labels.index("log10_tau")
    e_val, b_val = 10 ** x[e_i], 10 ** x[b_i]
    print(f"\n=== Constrained optimum (eps_e + eps_B <= 1; from seed {best['source']}) ===")
    print(f"log_posterior = {dec['log_posterior']:.6f}; Delta vs unconstrained best at p = {start_p:g} "
          f"({ref['log_posterior']:.6f}) = {dec['log_posterior'] - ref['log_posterior']:.6f}")
    print(f"tau = {10 ** x[tau_i]:.3f} s (unconstrained {float(previous['tau']):.3f} s)")
    print(f"eps_e_r = {e_val:.5f}, eps_B_r = {b_val:.5f}, sum = {e_val + b_val:.6f} "
          f"(on the constraint surface: {1.0 - (e_val + b_val) < 1e-3}; margin 1 - sum = {1.0 - (e_val + b_val):.2e})")
    print(f"eps_e + eps_B (forward shock) = {10 ** x[labels.index('log10_eps_e')] + 10 ** x[labels.index('log10_eps_B')]:.5f}")
    print(f"{'':<26}{'unconstrained':>16}{'constrained':>14}{'difference':>14}")
    for key in ("chi2_XRT_flux", "chi2_i_band", "chi2_XRT_spectral_index", "chi2_total", "log_prior", "log_likelihood", "log_posterior"):
        print(f"{key:<26}{ref[key]:>16.4f}{dec[key]:>14.4f}{dec[key] - ref[key]:>14.4f}")
    print(f"{'label':<16} {'unconstrained':>14} {'constrained':>14} {'physical':>14} {'lower':>12} {'upper':>12}")
    for param_def, label, v0, v1, (lo, hi) in zip(param_defs, labels, unconstrained, x, bounds):
        phys_lo, phys_hi = (10 ** lo, 10 ** hi) if label.startswith("log10_") else (lo, hi)
        print(f"{label:<16} {v0:>14.8g} {v1:>14.8g} {physical_param_value(param_def, v1):>14.6g} {phys_lo:>12.4g} {phys_hi:>12.4g}")
    near = [(l, "upper" if hi - v < v - lo else "lower") for l, v, (lo, hi) in zip(labels, x, bounds)
            if min(v - lo, hi - v) < PROFILE_BOUND_FRACTION * (hi - lo)]
    print(f"Within {PROFILE_BOUND_FRACTION:g} of a prior bound: {near if near else 'none'}")
    print(f"Physical boundary eps_e_r + eps_B_r = 1 reached: {1.0 - (e_val + b_val) < 1e-3}")
    return x


FREE_XI_LABELS = HIGH_MODE_LABELS[:-1] + ["xi_e_r", "log10_A_V"]
# Constrained (eps_e + eps_B <= 1) optimum with xi_e_r = 1 fixed, from optimize_constrained (8 digits)
CONSTRAINED_OPT = np.array([
    53.133663, 1.4523434, -1.4789543, 0.22465355, 2.0103095, -1.4636301,
    -1.8992912, 2.618179, 2.0420696, -0.21016096, -0.4160835, -0.34580831,
])
CONSTRAINED_REFERENCE = {
    "chi2_XRT_flux": 16.9946, "chi2_i_band": 143.6034, "chi2_XRT_spectral_index": 26.0158,
    "chi2_total": 186.6138, "log_prior": -0.6688, "log_likelihood": -93.3069, "log_posterior": -93.9757,
}


def optimize_free_xi_e_r(param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling, npool, seed):
    """Multi-start optimisation with xi_e_r newly free (0.1-1), under eps_e + eps_B <= 1; no MCMC."""
    from concurrent.futures import ProcessPoolExecutor

    labels = [param_label(p) for p in param_defs]
    print(f"Parameter order ({len(labels)}): {labels}")
    if labels != FREE_XI_LABELS:
        raise ValueError(f"Parameter labels differ from the expected order.\n  expected: {FREE_XI_LABELS}\n  actual:   {labels}")
    bounds = _sampled_bounds(param_defs)
    context = (param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
    _init_early_phase_context(context)
    span = bounds[:, 1] - bounds[:, 0]
    e_i, b_i, xi_i = labels.index("log10_eps_e_r"), labels.index("log10_eps_B_r"), labels.index("xi_e_r")
    tau_i = labels.index("log10_tau")
    rng = np.random.default_rng(seed)

    def project(x):
        y = np.clip(np.asarray(x, dtype=float), bounds[:, 0], bounds[:, 1])
        e, b = 10 ** y[e_i], 10 ** y[b_i]
        if e + b > 1.0:
            scale = (1.0 - 1e-6) / (e + b)
            y[e_i], y[b_i] = np.log10(e * scale), np.log10(b * scale)
        return np.clip(y, bounds[:, 0], bounds[:, 1])

    def embed(vec12, xi, pair=None):
        x = np.insert(np.asarray(vec12, dtype=float), xi_i, xi)
        if pair is not None:
            x[e_i], x[b_i] = np.log10(pair[0]), np.log10(pair[1])
        return project(x)

    def perturbed(x, scale):
        return project(np.asarray(x) + rng.normal(0.0, scale * span))

    seeds = [(f"xi_e_r={xi:g}", embed(CONSTRAINED_OPT, xi)) for xi in (1.0, 0.8, 0.5, 0.3, 0.1)]
    seeds += [("xi_e_r=0.65", embed(CONSTRAINED_OPT, 0.65)),
              ("eps_r-wide-opt:xi_e_r=1", embed(HIGH_MODE_OPT_EPS_R_WIDE, 1.0)),
              ("eps_r-wide-opt:xi_e_r=0.5", embed(HIGH_MODE_OPT_EPS_R_WIDE, 0.5)),
              ("xi_e_r=0.5,eps_r=(0.3,0.3)", embed(CONSTRAINED_OPT, 0.5, (0.3, 0.3))),
              ("xi_e_r=0.3,eps_r=(0.5,0.2)", embed(CONSTRAINED_OPT, 0.3, (0.5, 0.2)))]
    for label, vec in list(seeds[:5]):
        seeds += [(f"{label}+perturbed-0.02", perturbed(vec, 0.02)), (f"{label}+perturbed-0.05", perturbed(vec, 0.05))]
    for label, vec in seeds:
        assert np.isfinite(_log_probability_from_context(vec)), f"seed {label} is infeasible"
    print(f"Free xi_e_r optimisation: {len(seeds)} seeds on {min(npool, len(seeds))} workers; reference (xi_e_r = 1) "
          f"log_posterior = {CONSTRAINED_REFERENCE['log_posterior']:.4f}", flush=True)
    print(f"Seed log_posteriors at start: " + ", ".join(f"{l}: {_log_probability_from_context(v):.1f}" for l, v in seeds[:5]), flush=True)

    t_start = datetime.now()
    with ProcessPoolExecutor(max_workers=npool, initializer=_init_early_phase_context, initargs=(context,)) as pool:
        futures = [pool.submit(_optimize_task, (label, vec, bounds, dict(PROFILE_BUDGET))) for label, vec in seeds]
        results = sorted((f.result() for f in futures), key=lambda r: r["f"])
        for r in results:
            x = r["x"]
            print(f"  [phase 1] {r['source']:<42} log_post = {-r['f']:9.4f}  xi_e_r = {x[xi_i]:.3f}  tau = {10 ** x[tau_i]:6.1f} s  "
                  f"eps_r sum = {10 ** x[e_i] + 10 ** x[b_i]:.4f}  ({r['nfev']} evals)", flush=True)
        print(f"Phase 1 done after {datetime.now() - t_start}", flush=True)
        polish_budget = {"max_rounds": 15, "tol": 1e-3, "powell_maxfev": 20000, "nm_maxfev": 6000}
        polished = [pool.submit(_optimize_task, (f"polish-of-{r['source']}", r["x"], bounds, polish_budget)) for r in results[:6]]
        polished = sorted((f.result() for f in polished), key=lambda r: r["f"])
    print(f"Phase 2 done after {datetime.now() - t_start}", flush=True)

    def describe(x):
        dec = posterior_decomposition(x, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
        return dec, 10 ** x[e_i], 10 ** x[b_i]

    print("\n=== Best seeds after polishing ===")
    print(f"{'seed':<58}{'log_post':>10}{'xi_e_r':>8}{'tau[s]':>8}{'eps_e_r':>9}{'eps_B_r':>9}{'sum':>8}{'chi2_i':>8}")
    for r in polished:
        dec, e, b = describe(r["x"])
        print(f"{r['source']:<58}{dec['log_posterior']:>10.4f}{r['x'][xi_i]:>8.3f}{10 ** r['x'][tau_i]:>8.1f}{e:>9.4f}{b:>9.4f}{e + b:>8.4f}{dec['chi2_i_band']:>8.2f}")
    best = polished[0]
    x = best["x"]
    dec, e_val, b_val = describe(x)
    ref = CONSTRAINED_REFERENCE
    print(f"\n=== Free-xi_e_r optimum (from seed {best['source']}) ===")
    print(f"log_posterior = {dec['log_posterior']:.6f}; Delta vs xi_e_r = 1 constrained ({ref['log_posterior']:.4f}) = "
          f"{dec['log_posterior'] - ref['log_posterior']:.4f}")
    print(f"tau = {10 ** x[tau_i]:.3f} s; xi_e_r = {x[xi_i]:.5f} (bounds {bounds[xi_i, 0]:g}-{bounds[xi_i, 1]:g})")
    print(f"eps_e_r = {e_val:.5f}, eps_B_r = {b_val:.5f}, sum = {e_val + b_val:.6f}, on the physical boundary: {1.0 - (e_val + b_val) < 1e-3}")
    print(f"{'':<26}{'xi_e_r=1 constrained':>22}{'xi_e_r free':>14}{'difference':>14}")
    for key in ("chi2_XRT_flux", "chi2_i_band", "chi2_XRT_spectral_index", "chi2_total", "log_prior", "log_likelihood", "log_posterior"):
        print(f"{key:<26}{ref[key]:>22.4f}{dec[key]:>14.4f}{dec[key] - ref[key]:>14.4f}")
    print(f"{'label':<16} {'constrained xi=1':>16} {'free xi_e_r':>14} {'physical':>14} {'lower':>12} {'upper':>12}")
    previous = np.insert(CONSTRAINED_OPT, xi_i, 1.0)
    for param_def, label, v0, v1, (lo, hi) in zip(param_defs, labels, previous, x, bounds):
        phys_lo, phys_hi = (10 ** lo, 10 ** hi) if label.startswith("log10_") else (lo, hi)
        print(f"{label:<16} {v0:>16.8g} {v1:>14.8g} {physical_param_value(param_def, v1):>14.6g} {phys_lo:>12.4g} {phys_hi:>12.4g}")
    near = [(l, "upper" if hi - v < v - lo else "lower") for l, v, (lo, hi) in zip(labels, x, bounds)
            if min(v - lo, hi - v) < PROFILE_BOUND_FRACTION * (hi - lo)]
    print(f"Within {PROFILE_BOUND_FRACTION:g} of a prior bound: {near if near else 'none'}")
    print(f"xi_e_r distance from bounds: {100 * (x[xi_i] - bounds[xi_i, 0]) / span[xi_i]:.1f}% above the lower bound, "
          f"{100 * (bounds[xi_i, 1] - x[xi_i]) / span[xi_i]:.1f}% below the upper bound")
    return x


XI_PROFILE_GRID = [1.0, 0.7, 0.5, 0.3, 0.2, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01]
# Best free-xi_e_r optimum (optimize_free_xi_e_r output, 8 digits)
XI_FREE_OPT = np.array([
    53.910591, 1.5895197, -1.6173998, 0.39553194, 2.0100187, -2.236567, -1.4066663,
    2.6020679, 2.0048701, -0.15885476, -0.51379504, 0.10000015, -0.46573913,
])
XI_PROFILE_PHYSICAL = ("E_iso", "Gamma0", "theta_c", "A_star", "eps_e", "eps_B", "tau", "p_r", "eps_e_r", "eps_B_r")


def run_xi_profile(param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling, outdir, npool, seed):
    """Profile of the posterior over the reverse-shock xi_e_r: max over the other parameters at fixed xi_e_r (no MCMC)."""
    import csv

    labels = [param_label(p) for p in param_defs]
    print(f"Parameter order ({len(labels)}): {labels}")
    if labels != FREE_XI_LABELS:
        raise ValueError(f"Parameter labels differ from the expected order.\n  expected: {FREE_XI_LABELS}\n  actual:   {labels}")
    xi_i = labels.index("xi_e_r")
    e_i, b_i = labels.index("log10_eps_e_r"), labels.index("log10_eps_B_r")
    bounds = _sampled_bounds(param_defs)
    grid = np.array(XI_PROFILE_GRID, dtype=float)
    if grid.min() < bounds[xi_i, 0] - 1e-12 or grid.max() > bounds[xi_i, 1] + 1e-12:
        raise ValueError(f"xi_e_r grid {grid.min()}-{grid.max()} exceeds its prior range "
                         f"{bounds[xi_i, 0]}-{bounds[xi_i, 1]}; pass --xi-e-r-min.")
    context = (param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
    _init_early_phase_context(context)
    span = bounds[:, 1] - bounds[:, 0]
    rng = np.random.default_rng(seed)
    n = len(grid)

    def seed_at(x, k, perturb):
        y = np.clip(np.asarray(x, dtype=float), bounds[:, 0], bounds[:, 1])
        if perturb:
            y = np.clip(y + rng.normal(0.0, perturb * span), bounds[:, 0], bounds[:, 1])
        y[xi_i] = grid[k]
        e, b = 10 ** y[e_i], 10 ** y[b_i]
        if e + b > 1.0:                                   # project onto eps_e_r + eps_B_r <= 1
            scale = (1.0 - 1e-6) / (e + b)
            y[e_i], y[b_i] = np.log10(e * scale), np.log10(b * scale)
        return np.clip(y, bounds[:, 0], bounds[:, 1])

    xi_one_opt = np.insert(CONSTRAINED_OPT, xi_i, 1.0)
    independent_seeds = lambda k: [
        ("xi0.1-opt", seed_at(XI_FREE_OPT, k, 0.0)), ("xi1-constrained-opt", seed_at(xi_one_opt, k, 0.0)),
        ("xi0.1-opt-perturbed-0.01", seed_at(XI_FREE_OPT, k, 0.01)), ("xi0.1-opt-perturbed-0.03", seed_at(XI_FREE_OPT, k, 0.03)),
        ("xi1-constrained-opt-perturbed-0.02", seed_at(xi_one_opt, k, 0.02)),
    ]
    extra_seeds = lambda k: [("xi0.1-opt", seed_at(XI_FREE_OPT, k, 0.0)), ("xi1-constrained-opt", seed_at(xi_one_opt, k, 0.0))]
    budget = dict(PROFILE_BUDGET)
    print(f"xi_e_r profile: {grid.tolist()}, {npool} workers, budget {budget}", flush=True)
    indep, forward, reverse = _fixed_parameter_scan(xi_i, grid, bounds, context, npool, budget, seed_at,
                                                    independent_seeds, extra_seeds, "xi_e_r")

    branch_results = {"independent": indep, "forward": forward, "reverse": reverse}
    best_lp = max(-r[k]["f"] for r in branch_results.values() for k in range(n))
    rows = []
    for k in range(n):
        best_branch = min(branch_results, key=lambda b: branch_results[b][k]["f"])
        for branch, results in branch_results.items():
            r = results[k]
            dec = posterior_decomposition(r["x"], param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
            near = [l for l, v, (lo, hi) in zip(labels, r["x"], bounds)
                    if l != "xi_e_r" and min(v - lo, hi - v) < PROFILE_BOUND_FRACTION * (hi - lo)]
            eps_sum = 10 ** r["x"][e_i] + 10 ** r["x"][b_i]
            if 1.0 - eps_sum < PROFILE_BOUND_FRACTION:
                near.append("PHYSICAL:eps_e_r+eps_B_r=1")
            physical = {name: physical_param_value(param_defs[labels.index(l)], r["x"][labels.index(l)])
                        for name, l in ((n_, f"log10_{n_}" if f"log10_{n_}" in labels else n_) for n_ in XI_PROFILE_PHYSICAL)}
            rows.append({
                "xi_e_r": grid[k], "branch": branch, "is_best": branch == best_branch, "source": r["source"],
                "delta_log_posterior": dec["log_posterior"] - best_lp, **dec,
                "eps_e_r_plus_eps_B_r": eps_sum, **physical,
                "nfev_best_run": r["nfev"], "nfev_total": r["nfev_total"], "bound_flags": ";".join(near),
                **{l: v for l, v in zip(labels, r["x"])},
            })
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "xi_profile.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez(
        outdir / "xi_profile.npz", xi_e_r=grid, labels=np.array(labels),
        **{f"{b}_params": np.array([results[k]["x"] for k in range(n)]) for b, results in branch_results.items()},
        **{f"{b}_log_posterior": np.array([-results[k]["f"] for k in range(n)]) for b, results in branch_results.items()},
    )
    print(f"Saved: {outdir / 'xi_profile.csv'}, {outdir / 'xi_profile.npz'}")
    plot_xi_profile(rows, outdir)
    print_xi_profile_summary(rows, grid)


def print_xi_profile_summary(rows, grid):
    best = {r["xi_e_r"]: r for r in rows if r["is_best"]}
    by_branch = {b: {r["xi_e_r"]: r for r in rows if r["branch"] == b} for b in ("independent", "forward", "reverse")}
    order = sorted(grid, reverse=True)
    lp = np.array([best[x]["log_posterior"] for x in order])
    k_max = int(np.argmax(lp))
    print("\n=== xi_e_r profile summary ===")
    print(f"Best xi_e_r = {order[k_max]:g}, log_posterior = {lp[k_max]:.3f}"
          f"{' (interior of the grid)' if 0 < k_max < len(order) - 1 else ' (AT THE EDGE of the grid)'}")
    print(f"Delta log P (best - xi_e_r = 0.1) = {lp[k_max] - best[0.1]['log_posterior']:.3f}; "
          f"(best - xi_e_r = 1) = {lp[k_max] - best[1.0]['log_posterior']:.3f}; "
          f"(xi_e_r = 0.01 - best) = {best[0.01]['log_posterior'] - lp[k_max]:.3f}")
    print("tau [s]: " + ", ".join(f"{x:g}: {best[x]['tau']:.1f}" for x in order))
    print("eps_e_r + eps_B_r: " + ", ".join(f"{x:g}: {best[x]['eps_e_r_plus_eps_B_r']:.3f}" for x in order))
    spread = {x: max(by_branch[b][x]["log_posterior"] for b in by_branch) - min(by_branch[b][x]["log_posterior"] for b in by_branch) for x in order}
    print("Branch spread (max - min log_posterior): " + ", ".join(f"{x:g}: {d:.2f}" for x, d in spread.items()))
    flagged = {x: best[x]["bound_flags"] for x in order if best[x]["bound_flags"]}
    print(f"Bound flags (xi_e_r itself excluded): {flagged if flagged else 'none'}")
    # systematic parameter movement: Spearman rank correlation with log xi_e_r over the best rows
    from scipy.stats import spearmanr
    print("Rank correlation with log10(xi_e_r) and range (max/min) over the profile:")
    for name in XI_PROFILE_PHYSICAL:
        vals = np.array([best[x][name] for x in order])
        rho = spearmanr(np.log10(order), vals).correlation
        print(f"  {name:<8} rho = {rho:+.2f}   {vals.min():.4g} .. {vals.max():.4g}   (max/min = {vals.max() / vals.min():.2f})")
    low = [x for x in order if x <= 0.3]
    print(f"Delta log P range over xi_e_r <= 0.3: {max(best[x]['log_posterior'] for x in low) - min(best[x]['log_posterior'] for x in low):.2f}")
    print(f"{'xi_e_r':>7} {'best logP':>10} {'dlogP':>8} {'tau':>7} {'eps_e_r':>8} {'eps_B_r':>8} {'sum':>7} {'chi2_XRT':>9} {'chi2_i':>8} {'chi2_beta':>9} {'indep':>9} {'forward':>9} {'reverse':>9}  branch")
    for x in order:
        b = best[x]
        print(f"{x:>7g} {b['log_posterior']:>10.3f} {b['delta_log_posterior']:>8.3f} {b['tau']:>7.1f} {b['eps_e_r']:>8.4f} {b['eps_B_r']:>8.4f} "
              f"{b['eps_e_r_plus_eps_B_r']:>7.3f} {b['chi2_XRT_flux']:>9.2f} {b['chi2_i_band']:>8.2f} {b['chi2_XRT_spectral_index']:>9.2f} "
              + "".join(f"{by_branch[br][x]['log_posterior']:>9.3f}" for br in by_branch) + f"  {b['branch']}")


EISO_SENSITIVITY_XI = (0.10, 0.07, 0.05)


def optimize_eiso_sensitivity(profile_dir, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling, outdir, npool, seed):
    """E_iso upper-bound sensitivity: fixed xi_e_r = 0.10 / 0.07 / 0.05, all other parameters optimised (no MCMC).

    Starts come from the best xi-profile rows in <profile_dir>/xi_profile.csv (the point itself, its neighbours,
    perturbations, and E_iso up / eps_e down along the degeneracy direction), then the best two per xi_e_r are polished.
    Compares against the old-E_iso-cap values stored in the same CSV.
    """
    import csv
    from concurrent.futures import ProcessPoolExecutor

    labels = [param_label(p) for p in param_defs]
    print(f"Parameter order ({len(labels)}): {labels}")
    if labels != FREE_XI_LABELS:
        raise ValueError(f"Parameter labels differ from the expected order.\n  expected: {FREE_XI_LABELS}\n  actual:   {labels}")
    xi_i, ei_i, eps_i = labels.index("xi_e_r"), labels.index("log10_E_iso"), labels.index("log10_eps_e")
    e_i, b_i = labels.index("log10_eps_e_r"), labels.index("log10_eps_B_r")
    bounds = _sampled_bounds(param_defs)
    context = (param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
    _init_early_phase_context(context)
    span = bounds[:, 1] - bounds[:, 0]
    rng = np.random.default_rng(seed)
    print(f"E_iso prior range: {10 ** bounds[ei_i, 0]:.3g} - {10 ** bounds[ei_i, 1]:.3g} erg")

    with open(Path(profile_dir) / "xi_profile.csv", newline="") as fh:
        old = {}
        for r in csv.DictReader(fh):
            if r["is_best"] == "True" and any(abs(float(r["xi_e_r"]) - x) < 1e-9 for x in EISO_SENSITIVITY_XI):
                old[round(float(r["xi_e_r"]), 4)] = r
    if len(old) != len(EISO_SENSITIVITY_XI):
        raise ValueError(f"Best xi-profile rows for {EISO_SENSITIVITY_XI} not all found in {profile_dir}")
    old_vec = {x: np.array([float(old[round(x, 4)][l]) for l in labels]) for x in EISO_SENSITIVITY_XI}

    def project(x, xi):
        y = np.clip(np.asarray(x, dtype=float), bounds[:, 0], bounds[:, 1])
        y[xi_i] = xi
        e, b = 10 ** y[e_i], 10 ** y[b_i]
        if e + b > 1.0:
            scale = (1.0 - 1e-6) / (e + b)
            y[e_i], y[b_i] = np.log10(e * scale), np.log10(b * scale)
        return np.clip(y, bounds[:, 0], bounds[:, 1])

    def perturbed(x, xi, scale):
        return project(np.asarray(x) + rng.normal(0.0, scale * span), xi)

    def compensated(x, xi, dex):
        y = np.array(x, dtype=float)
        y[ei_i] += dex
        y[eps_i] -= dex
        return project(y, xi)

    neighbours = {0.10: 0.07, 0.07: 0.10, 0.05: 0.07}
    seeds = {}
    for xi in EISO_SENSITIVITY_XI:
        own = old_vec[xi]
        seeds[xi] = [
            ("old-profile-best", project(own, xi)),
            ("old-profile-best+perturbed-0.02", perturbed(own, xi, 0.02)),
            ("old-profile-best+perturbed-0.05", perturbed(own, xi, 0.05)),
            ("E_iso x2, eps_e /2", compensated(own, xi, np.log10(2))),
            ("E_iso x4, eps_e /4", compensated(own, xi, np.log10(4))),
            (f"neighbour xi={neighbours[xi]:g} best", project(old_vec[neighbours[xi]], xi)),
        ]
        for label, vec in seeds[xi]:
            assert np.isfinite(_log_probability_from_context(vec)), f"seed {label} at xi_e_r={xi} is infeasible"
    reference = {xi: float(old[round(xi, 4)]["log_posterior"]) for xi in EISO_SENSITIVITY_XI}
    print(f"Reference (E_iso max 1e54) log_posterior: {reference}; seeds per xi_e_r: {len(seeds[0.1])}", flush=True)

    t_start = datetime.now()
    phase1, final = {}, {}
    with ProcessPoolExecutor(max_workers=npool, initializer=_init_early_phase_context, initargs=(context,)) as pool:
        futures = {xi: [pool.submit(_profile_task, (xi, xi_i, label, vec, bounds, dict(PROFILE_BUDGET))) for label, vec in seeds[xi]]
                   for xi in EISO_SENSITIVITY_XI}
        for xi in EISO_SENSITIVITY_XI:
            phase1[xi] = sorted((f.result() for f in futures[xi]), key=lambda r: r["f"])
            for r in phase1[xi]:
                print(f"  [phase 1] xi_e_r={xi:g}  {r['source']:<34} log_post = {-r['f']:9.4f}  E_iso = {10 ** r['x'][ei_i]:.3e}  "
                      f"({r['nfev']} evals)", flush=True)
        print(f"Phase 1 done after {datetime.now() - t_start}", flush=True)
        polish_budget = dict(POLISH_BUDGET)
        polished = {xi: [pool.submit(_profile_task, (xi, xi_i, f"polish-of-{r['source']}", r["x"], bounds, polish_budget))
                         for r in phase1[xi][:2]] for xi in EISO_SENSITIVITY_XI}
        for xi in EISO_SENSITIVITY_XI:
            final[xi] = sorted([f.result() for f in polished[xi]] + phase1[xi][:2], key=lambda r: r["f"])
    print(f"Phase 2 done after {datetime.now() - t_start}", flush=True)

    rows = []
    for xi in EISO_SENSITIVITY_XI:
        best = final[xi][0]
        x = best["x"]
        dec = posterior_decomposition(x, param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
        ref_row = old[round(xi, 4)]
        e_val, b_val = 10 ** x[e_i], 10 ** x[b_i]
        near = [(l, "upper" if hi - v < v - lo else "lower") for l, v, (lo, hi) in zip(labels, x, bounds)
                if l != "xi_e_r" and min(v - lo, hi - v) < PROFILE_BOUND_FRACTION * (hi - lo)]
        print(f"\n=== xi_e_r = {xi:g} (fixed): E_iso upper bound {10 ** bounds[ei_i, 1]:.0e} ===")
        print(f"log_posterior = {dec['log_posterior']:.4f}; old (E_iso max 1e54) = {reference[xi]:.4f}; Delta = {dec['log_posterior'] - reference[xi]:+.4f}")
        print(f"E_iso = {10 ** x[ei_i]:.4e} erg (old {float(ref_row['E_iso']):.4e}); {100 * (bounds[ei_i, 1] - x[ei_i]) / span[ei_i]:.1f}% of its "
              f"sampled range below the new upper bound; near new bound: {any(l == 'log10_E_iso' for l, _ in near)}")
        print(f"tau = {10 ** x[labels.index('log10_tau')]:.1f} s (old {float(ref_row['tau']):.1f}); eps_e_r = {e_val:.4f}, eps_B_r = {b_val:.4f}, "
              f"sum = {e_val + b_val:.5f}; p = {x[labels.index('p')]:.4f}, p_r = {x[labels.index('p_r')]:.4f}")
        print(f"{'':<26}{'old (1e54)':>14}{'new (1e55)':>14}{'difference':>14}")
        for key in ("chi2_XRT_flux", "chi2_i_band", "chi2_XRT_spectral_index", "chi2_total", "log_prior", "log_likelihood", "log_posterior"):
            print(f"{key:<26}{float(ref_row[key]):>14.4f}{dec[key]:>14.4f}{dec[key] - float(ref_row[key]):>14.4f}")
        print(f"{'label':<16} {'old (1e54)':>14} {'new (1e55)':>14} {'physical':>14} {'lower':>12} {'upper':>12}")
        for param_def, label, v0, v1, (lo, hi) in zip(param_defs, labels, old_vec[xi], x, bounds):
            phys_lo, phys_hi = (10 ** lo, 10 ** hi) if label.startswith("log10_") else (lo, hi)
            print(f"{label:<16} {v0:>14.8g} {v1:>14.8g} {physical_param_value(param_def, v1):>14.6g} {phys_lo:>12.4g} {phys_hi:>12.4g}")
        print(f"Within {PROFILE_BOUND_FRACTION:g} of a prior bound (xi_e_r excluded): {near if near else 'none'}; "
              f"on eps_e_r + eps_B_r = 1: {1.0 - (e_val + b_val) < PROFILE_BOUND_FRACTION}")
        print(f"Best several seeds at xi_e_r = {xi:g}:")
        for r in final[xi][:4]:
            d2 = posterior_decomposition(r["x"], param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
            print(f"  {r['source']:<50} log_post = {d2['log_posterior']:9.4f}  E_iso = {10 ** r['x'][ei_i]:.3e}  tau = {10 ** r['x'][labels.index('log10_tau')]:6.1f}  "
                  f"eps_r sum = {10 ** r['x'][e_i] + 10 ** r['x'][b_i]:.4f}")
        for rank, r in enumerate(final[xi]):
            d2 = posterior_decomposition(r["x"], param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
            rows.append({"xi_e_r": xi, "rank": rank, "source": r["source"], "old_log_posterior": reference[xi], **d2,
                         "nfev": r["nfev"], **{l: v for l, v in zip(labels, r["x"])}})
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "eiso_sensitivity.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {outdir / 'eiso_sensitivity.csv'}")
