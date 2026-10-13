#!/usr/bin/env python3
"""Bounded multi-start search for distinct posterior modes of the FINAL early-phase model (diagnostic only).

Uses exactly the final production configuration (prior production, wind, xi both, resolution (0.3, 1, 30),
rtol 1e-6, smooth spectral-index term) and scores every point with early_phase_model.log_probability. Nothing
here modifies the final chains or results; everything is written to a new directory.

Stages (each reads the previous stages' files in OUTDIR, so the search can be run and costed step by step):
    screen   scrambled Sobol sweep of the prior box
    de       differential evolution: full box and tau-banded boxes (bands only restrict the DE search box)
    local    short multi-start Powell / Nelder-Mead from screen, DE, legacy and control seeds (full box)
    cluster  merge local optima that are connected (no barrier along the straight line, after removing the
             exact E_iso/A_star/eps/xi scaling ridge from the distance) and polish one representative per basin
    relax    relaxed-path (string) barrier test between selected line-separated optima
    pr_profile  p_r profile between families
    report   log P / log L / chi2 components, physical and derived parameters, volume indicators (exact scaling-ridge
             extent, parameters at prior bounds); no evidence estimate (local Gaussian / importance-sampling estimates
             proved unreliable for this curved posterior: IS ESS 1-12 of 4000 even with the converged chains' covariance)
    plots    FS / RS / total light curves with residuals for each distinct mode, and a comparison figure
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

import early_phase_model as epm
from early_phase_model import (
    _init_early_phase_context,
    _log_probability_from_context,
    _neg_log_probability_from_context,
    _sampled_bounds,
    param_label,
    posterior_decomposition,
)
from early_phase_optimize import PROFILE_BUDGET, robust_minimize

RESULT_ROOT = Path("/home/hongyp007/yp_folder/projects/GRB/result")
FINAL_DIR = RESULT_ROOT / "early_phase_FINAL_posterior"
NUMERICS = ((0.3, 1.0, 30.0), 1e-6)
SPECTRAL_INDEX_COOLING = "smooth"
TAU_BANDS = {"low": (5.0, 60.0), "intermediate": (60.0, 300.0), "high": (300.0, 700.0), "very_high": (700.0, 3000.0)}
SHORT_BUDGET = {"max_rounds": 3, "tol": 0.05, "powell_maxfev": 4000, "nm_maxfev": 2000}
SHORT_SECONDS, POLISH_SECONDS = 900, 2400   # wall-time caps per optimisation (a few regions evaluate at ~0.3 s)
RIDGE_SIGNS = {"log10_E_iso": -1, "log10_A_star": -1, "log10_eps_e": 1, "log10_eps_B": 1, "xi_e": 1,
               "log10_eps_e_r": 1, "log10_eps_B_r": 1, "xi_e_r": 1}
LINE_BARRIER_MERGE = 1.0   # optima are one basin if the path between them never drops > 1 below the lower end


def final_setup():
    epm.set_numerics(*NUMERICS)
    param_defs = epm.make_param_defs("production", "wind", free_xi_e=True, free_xi_e_r=True)
    xrt_data, i_data, xrt_index_data = epm.load_fit_data(include_spectral_index=True)
    context = (param_defs, xrt_data, i_data, xrt_index_data, SPECTRAL_INDEX_COOLING)
    return param_defs, [param_label(p) for p in param_defs], context


def _worker_init(context):
    epm.set_numerics(*NUMERICS)
    _init_early_phase_context(context)


def _eval(x):
    try:
        return float(_log_probability_from_context(np.asarray(x, dtype=float)))
    except Exception:
        return -np.inf


def _neg_eval(x):
    lp = _eval(x)
    return -lp if np.isfinite(lp) else 1e30


class _TimeUp(Exception):
    pass


def _local_task(task):
    """robust_minimize from one seed, stopped after `max_seconds` wall time (best point so far is kept)."""
    label, seed, bounds, budget, max_seconds = task
    t0 = time.time()
    start = np.clip(np.asarray(seed, dtype=float), bounds[:, 0], bounds[:, 1])
    best = {"x": start, "f": np.inf, "n": 0}

    def objective(x):
        if time.time() - t0 > max_seconds:
            raise _TimeUp
        f = _neg_log_probability_from_context(x)
        best["n"] += 1
        if f < best["f"]:
            best["x"], best["f"] = np.array(x, dtype=float), f
        return f

    try:
        _, _, _, rounds, _, _ = robust_minimize(objective, start, bounds, **budget)
        timed_out = False
    except _TimeUp:
        rounds, timed_out = -1, True
    return {"source": label, "x0": start, "x": best["x"], "lp": -best["f"], "nfev": best["n"], "rounds": rounds,
            "timed_out": timed_out, "seconds": time.time() - t0}


def _hyperplane_task(task):
    """Maximise log P on the hyperplane through x_mid orthogonal to direction u, both in coordinates scaled by
    `scale` (a relaxed-path image)."""
    x_mid, u, scale, bounds, maxfev = task
    basis = np.linalg.svd(np.eye(len(u)) - np.outer(u, u))[0][:, :len(u) - 1]

    def point(c):
        return x_mid + scale * (basis @ c)

    def neg(c):
        x = point(c)
        if np.any(x < bounds[:, 0]) or np.any(x > bounds[:, 1]):
            return 1e30
        return _neg_eval(x)

    from scipy.optimize import minimize
    best = {"f": neg(np.zeros(len(u) - 1)), "c": np.zeros(len(u) - 1)}

    def tracked(c):
        f = neg(c)
        if f < best["f"]:
            best["f"], best["c"] = f, np.array(c)
        return f

    minimize(tracked, best["c"], method="Powell", options={"maxfev": maxfev, "xtol": 1e-4, "ftol": 1e-6})
    return -best["f"], point(best["c"])


def make_pool(context, npool):
    from multiprocessing import get_context
    return get_context("fork").Pool(npool, initializer=_worker_init, initargs=(context,))


def log(outdir, text):
    line = f"[{datetime.now():%H:%M:%S}] {text}"
    print(line, flush=True)
    with open(Path(outdir) / "run.log", "a") as fh:
        fh.write(line + "\n")


# --------------------------------------------------------------------------------------------- geometry helpers
def ridge_direction(labels):
    d = np.array([RIDGE_SIGNS.get(l, 0) for l in labels], dtype=float)
    return d / np.linalg.norm(d)


def log_coords(x, labels):
    """xi_e / xi_e_r to log10 so that the scaling ridge is a straight line."""
    z = np.array(x, dtype=float)
    for name in ("xi_e", "xi_e_r"):
        z[..., labels.index(name)] = np.log10(z[..., labels.index(name)])
    return z


def ridge_split(x, labels):
    """(ridge coordinate, ridge-free coordinates) of x."""
    z, d = log_coords(x, labels), ridge_direction(labels)
    r = z @ d
    return r, z - np.multiply.outer(r, d)


def posterior_scales(labels, seed=0):
    """Posterior std of the ridge-free coordinates and the (shuffled) 60k thinned final posterior samples."""
    sub = np.random.default_rng(seed).permutation(np.load(FINAL_DIR / "samples.npy"))
    _, perp = ridge_split(sub, labels)
    return perp.std(axis=0), sub


def rms_distance(xa, xb, labels, scales):
    _, pa = ridge_split(xa, labels)
    _, pb = ridge_split(xb, labels)
    return float(np.sqrt(np.mean(((pa - pb) / scales) ** 2)))


def legacy_seeds(labels):
    """Previously found solutions mapped into the 14-parameter layout with xi_e = xi_e_r = 1 (the model they used)."""
    import csv
    from early_phase_diagnostics import HIGH_MODE_LABELS, HIGH_MODE_START, LOW_MODE_START

    def lift(values, old_labels):
        x = np.ones(len(labels))
        for l, v in zip(old_labels, values):
            x[labels.index(l)] = v
        return x

    seeds = [("legacy:LOW_MODE_START", lift(LOW_MODE_START, HIGH_MODE_LABELS)),
             ("legacy:HIGH_MODE_START", lift(HIGH_MODE_START, HIGH_MODE_LABELS))]
    profile = RESULT_ROOT / "early_phase_tauprofile_20261006_220735" / "tau_profile.csv"
    for row in csv.DictReader(open(profile)):
        if row["is_best"] == "True":
            seeds.append((f"legacy:tauprofile_tau{float(row['tau']):g}",
                          lift([float(row[l]) for l in HIGH_MODE_LABELS], HIGH_MODE_LABELS)))
    run = RESULT_ROOT / "early_phase_20261006_184042"
    old_labels = (run / "labels.txt").read_text().split()
    s = np.load(run / "samples.npy", mmap_mode="r")
    lp = np.load(run / "log_probs.npy")
    ti = old_labels.index("log10_tau")
    for name, (lo, hi) in {"low": (0, 2), "intermediate": (2, 2.5), "high": (2.5, 4)}.items():
        m = (s[:, ti] >= lo) & (s[:, ti] < hi) & np.isfinite(lp)
        if m.any():
            j = int(np.argmax(np.where(m, lp, -np.inf)))
            seeds.append((f"legacy:mcmc20261006_{name}tau", lift(np.asarray(s[j]), old_labels)))
    return seeds


def tau_band(x, labels):
    tau = 10 ** x[labels.index("log10_tau")]
    return next(b for b, (lo, hi) in TAU_BANDS.items() if lo <= tau <= hi)


# --------------------------------------------------------------------------------------------- stages
def stage_screen(outdir, context, labels, bounds, npool, n_log2):
    from scipy.stats import qmc
    u = qmc.Sobol(len(labels), scramble=True, seed=20261012).random_base2(n_log2)
    x = bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])
    t0 = time.time()
    with make_pool(context, npool) as pool:
        lp = np.array(pool.map(_eval, x, chunksize=32))
    dt = time.time() - t0
    np.savez(outdir / "screen.npz", x=x, lp=lp)
    log(outdir, f"screen: {len(x)} points in {dt:.0f} s ({dt / len(x) * npool:.4f} s per evaluation per worker); "
                f"finite {np.isfinite(lp).sum()}, best {np.nanmax(lp[np.isfinite(lp)]):.2f}")
    for band, (lo, hi) in TAU_BANDS.items():
        tau = 10 ** x[:, labels.index("log10_tau")]
        m = (tau >= lo) & (tau <= hi) & np.isfinite(lp)
        log(outdir, f"  band {band:13s} n={m.sum():6d} best log P {lp[m].max():9.2f}  top-10 median {np.sort(lp[m])[-10:].mean():9.2f}")


def stage_de(outdir, context, labels, bounds, npool, maxiter, runs):
    from scipy.optimize import differential_evolution
    ti = labels.index("log10_tau")
    out = {}
    with make_pool(context, npool) as pool:
        for name, band, seed in runs:
            b = bounds.copy()
            if band is not None:
                b[ti] = np.log10(TAU_BANDS[band])
            t0 = time.time()
            res = differential_evolution(_neg_eval, b, maxiter=maxiter, popsize=15, tol=1e-8, init="sobol",
                                         seed=seed, workers=pool.map, updating="deferred", polish=False)
            out[name] = {"x": res.x, "lp": -res.fun, "nfev": res.nfev, "population": res.population,
                         "population_lp": -res.population_energies}
            log(outdir, f"DE {name:16s}: best log P {-res.fun:9.2f}, tau {10 ** res.x[ti]:7.1f} s, "
                        f"{res.nfev} evaluations, {time.time() - t0:.0f} s")
            np.save(outdir / "de.npy", out, allow_pickle=True)


def diverse_top(x, lp, k, labels, scales, min_dist=1.0):
    picked = []
    for j in np.argsort(lp)[::-1]:
        if not np.isfinite(lp[j]):
            break
        if all(rms_distance(x[j], x[i], labels, scales) >= min_dist for i in picked):
            picked.append(j)
        if len(picked) == k:
            break
    return picked


def collect_seeds(outdir, labels, scales, post_sub, n_per_band):
    seeds = []
    scr = np.load(outdir / "screen.npz")
    tau = 10 ** scr["x"][:, labels.index("log10_tau")]
    for band, (lo, hi) in TAU_BANDS.items():
        m = np.where((tau >= lo) & (tau <= hi))[0]
        for j in diverse_top(scr["x"][m], scr["lp"][m], n_per_band, labels, scales):
            seeds.append((f"screen:{band}:{m[j]}", scr["x"][m[j]]))
    de = np.load(outdir / "de.npy", allow_pickle=True).item()
    for name, r in de.items():
        for j in diverse_top(r["population"], r["population_lp"], 3, labels, scales):
            seeds.append((f"de:{name}:{j}", r["population"][j]))
    seeds += legacy_seeds(labels)
    best = np.load(FINAL_DIR / "best_fit_params.npy")
    seeds += [("control:final_MAP", best), ("control:posterior_draw0", post_sub[0]), ("control:posterior_draw1", post_sub[1])]
    return seeds


def stage_local(outdir, context, labels, bounds, npool, seeds, budget, tag, max_seconds):
    tasks = [(label, x, bounds, budget, max_seconds) for label, x in seeds]
    results, t0 = [], time.time()
    with make_pool(context, npool) as pool:
        for r in pool.imap_unordered(_local_task, tasks):
            results.append(r)
            log(outdir, f"{tag} {len(results):3d}/{len(tasks)} {r['source']:40s} log P {r['lp']:10.3f} tau "
                        f"{10 ** r['x'][labels.index('log10_tau')]:7.1f} s  nfev {r['nfev']:6d}  {r['seconds']:.0f} s"
                        + ("  TIMED OUT" if r["timed_out"] else ""))
    log(outdir, f"{tag}: {len(tasks)} optimisations in {time.time() - t0:.0f} s; "
                f"nfev median {np.median([r['nfev'] for r in results]):.0f}, seconds median {np.median([r['seconds'] for r in results]):.0f}")
    np.save(outdir / f"{tag}.npy", results, allow_pickle=True)
    return results


def line_barriers(pool, pairs, n_inner=9):
    """For each (xa, lpa, xb, lpb): min log P along the straight segment minus min(lpa, lpb), and the worst point."""
    ts = np.linspace(0, 1, n_inner + 2)[1:-1]
    pts = [(1 - t) * xa + t * xb for xa, _, xb, _ in pairs for t in ts]
    lps = np.array(pool.map(_eval, pts, chunksize=4)).reshape(len(pairs), n_inner)
    return [(float(lps[k].min() - min(p[1], p[3])), lps[k]) for k, p in enumerate(pairs)]


def group_optima(pool, results, labels, scales, lp_floor, max_reps):
    """Union of optima connected by a barrier-free straight line (barrier > -LINE_BARRIER_MERGE). Returns groups,
    best member first, sorted by best log P."""
    res = sorted([r for r in results if np.isfinite(r["lp"]) and r["lp"] >= lp_floor], key=lambda r: -r["lp"])[:max_reps]
    n = len(res)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    barriers = line_barriers(pool, [(res[i]["x"], res[i]["lp"], res[j]["x"], res[j]["lp"]) for i, j in pairs])
    info = {}
    for (i, j), (b, _) in zip(pairs, barriers):
        info[(i, j)] = (b, rms_distance(res[i]["x"], res[j]["x"], labels, scales))
        if b > -LINE_BARRIER_MERGE:
            parent[find(j)] = find(i)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(res[i])
    return sorted(groups.values(), key=lambda g: -g[0]["lp"]), res, info


def stage_cluster(outdir, context, labels, bounds, npool, scales, lp_window, max_reps, polish_top):
    results = list(np.load(outdir / "local.npy", allow_pickle=True))
    lp_best = max(r["lp"] for r in results)
    with make_pool(context, npool) as pool:
        groups, _, _ = group_optima(pool, results, labels, scales, lp_best - lp_window, max_reps)
    log(outdir, f"cluster: {len(groups)} line-connected groups among local optima within {lp_window} of {lp_best:.2f}")
    for k, g in enumerate(groups):
        log(outdir, f"  group {k:2d}: n={len(g):3d} best log P {g[0]['lp']:9.3f} tau {10 ** g[0]['x'][labels.index('log10_tau')]:7.1f} s "
                    f"({g[0]['source']}); members' tau {sorted(round(10 ** r['x'][labels.index('log10_tau')]) for r in g)[:6]}")
    # polish the top-ranked groups plus the best groups of every tau band, so lower-ranked regions are not dropped
    per_band = set()
    for band in TAU_BANDS:
        per_band |= set([k for k, g in enumerate(groups) if tau_band(g[0]["x"], labels) == band][:3])
    chosen = set(per_band)
    for k in range(len(groups)):
        if len(chosen) >= polish_top:
            break
        chosen.add(k)
    log(outdir, f"polishing groups {sorted(chosen)} (best-3-per-tau-band groups: {sorted(per_band)})")
    seeds = [(f"polish:{groups[k][0]['source']}", groups[k][0]["x"]) for k in sorted(chosen)]
    polished = stage_local(outdir, context, labels, bounds, npool, seeds, PROFILE_BUDGET, "polish", POLISH_SECONDS)
    with make_pool(context, npool) as pool:
        groups, reps, info = group_optima(pool, polished, labels, scales, -np.inf, len(polished))
    modes = []
    for k, g in enumerate(groups):
        x = g[0]["x"]
        modes.append({"mode": k, "x": x, "lp": g[0]["lp"], "source": g[0]["source"], "n_merged": len(g),
                      "merged_sources": [r["source"] for r in g], "tau": 10 ** x[labels.index("log10_tau")]})
    np.save(outdir / "modes_line.npy", modes, allow_pickle=True)
    with open(outdir / "line_barriers_polished.json", "w") as fh:
        json.dump([{"a": reps[i]["source"], "b": reps[j]["source"], "lp_a": reps[i]["lp"], "lp_b": reps[j]["lp"],
                    "barrier": b, "rms_ridge_free_distance_sigma": d} for (i, j), (b, d) in info.items()], fh, indent=1)
    log(outdir, f"after polishing: {len(modes)} line-separated optima")
    for m in modes:
        log(outdir, f"  mode {m['mode']:2d}: log P {m['lp']:9.3f} tau {m['tau']:7.1f} s merged {m['n_merged']} ({m['source']})")


def stage_relax(outdir, context, bounds, npool, pairs, n_images=7, maxfev=2500):
    """Relaxed-path test between line-separated optima `pairs` [(a, b), ...]: each interior image is maximised on the
    hyperplane orthogonal to the a-b segment (coordinates scaled by the posterior widths). The pair is one basin if the
    relaxed path never drops > LINE_BARRIER_MERGE below the lower endpoint. Images may slide sideways, so the largest
    jump between consecutive images (in the same scaled units, vs the a-b distance) is recorded as a continuity check."""
    modes = list(np.load(outdir / "modes_line.npy", allow_pickle=True))
    s = np.load(FINAL_DIR / "samples.npy").std(axis=0)
    ts = np.linspace(0, 1, n_images + 2)[1:-1]
    tasks = []
    for a, b in pairs:
        u = (modes[b]["x"] - modes[a]["x"]) / s
        u /= np.linalg.norm(u)
        tasks += [((1 - t) * modes[a]["x"] + t * modes[b]["x"], u, s, bounds, maxfev) for t in ts]
    t0 = time.time()
    with make_pool(context, npool) as pool:
        res = pool.map(_hyperplane_task, tasks, chunksize=1)
    out = []
    for k, (a, b) in enumerate(pairs):
        chunk = res[k * n_images:(k + 1) * n_images]
        lps = np.array([r[0] for r in chunk])
        path = np.array([modes[a]["x"]] + [r[1] for r in chunk] + [modes[b]["x"]]) / s
        jump = float(np.max(np.linalg.norm(np.diff(path, axis=0), axis=1)) / np.linalg.norm(path[-1] - path[0]))
        barrier = float(lps.min() - min(modes[a]["lp"], modes[b]["lp"]))
        out.append({"a": a, "b": b, "lp_a": modes[a]["lp"], "lp_b": modes[b]["lp"], "barrier": barrier,
                    "path_lp": lps.tolist(), "max_jump_over_distance": jump,
                    "same_basin": barrier > -LINE_BARRIER_MERGE})
        log(outdir, f"relaxed path mode {a} ({modes[a]['lp']:.2f}) - mode {b} ({modes[b]['lp']:.2f}): barrier {barrier:8.2f}; "
                    f"max jump/dist {jump:.2f}; path {np.round(lps, 1).tolist()}")
    log(outdir, f"relax: {len(pairs)} pairs, {len(tasks)} images in {time.time() - t0:.0f} s")
    with open(outdir / "relaxed_paths.json", "w") as fh:
        json.dump(out, fh, indent=1)


def stage_pr_profile(outdir, context, labels, bounds, npool, seed_modes, grid):
    """p_r profile between the families: maximise log P at fixed p_r from each seed mode (one wave of tasks)."""
    from early_phase_optimize import _profile_task
    modes = list(np.load(outdir / "modes_line.npy", allow_pickle=True))
    pr_i = labels.index("p_r")
    budget = {"max_rounds": 3, "tol": 0.05, "powell_maxfev": 5000, "nm_maxfev": 2000}
    tasks = [(v, pr_i, f"mode{k}", modes[k]["x"], bounds, budget) for v in grid for k in seed_modes]
    t0 = time.time()
    with make_pool(context, npool) as pool:
        res = pool.map(_profile_task, tasks, chunksize=1)
    rows = []
    for (v, _, src, _, _, _), r in zip(tasks, res):
        rows.append({"p_r": v, "seed": src, "log_posterior": -r["f"], "tau": 10 ** r["x"][labels.index("log10_tau")],
                     "x": r["x"].tolist(), "nfev": r["nfev"]})
    for v in grid:
        line = "  ".join(f"{r['seed']}: {r['log_posterior']:9.2f} (tau {r['tau']:6.1f})" for r in rows if r["p_r"] == v)
        log(outdir, f"p_r = {v:5.3f}: {line}")
    log(outdir, f"p_r profile: {len(tasks)} optimisations in {time.time() - t0:.0f} s")
    with open(outdir / "p_r_profile.json", "w") as fh:
        json.dump(rows, fh, indent=1)


# --------------------------------------------------------------------------------------------- volume indicators
def feasible(xs, bounds, labels):
    """Inside the prior box and eps_e + eps_B <= 1 for both shocks (vectorised log_prior support)."""
    xs = np.atleast_2d(xs)
    ok = np.all((xs >= bounds[:, 0]) & (xs <= bounds[:, 1]), axis=1)
    v = lambda l: 10 ** xs[:, labels.index(l)]
    return ok & (v("log10_eps_e") + v("log10_eps_B") <= 1) & (v("log10_eps_e_r") + v("log10_eps_B_r") <= 1)


def ridge_extent(x, bounds, labels):
    """Allowed log10 f range of the exact scaling E_iso, A_star -> /f; eps_*, xi_* -> *f inside the box."""
    lo, hi = -np.inf, np.inf
    z_lo, z_hi = bounds[:, 0].copy(), bounds[:, 1].copy()
    z = log_coords(x, labels)
    for name in ("xi_e", "xi_e_r"):
        i = labels.index(name)
        z_lo[i], z_hi[i] = np.log10(z_lo[i]), np.log10(z_hi[i])
    for l, sign in RIDGE_SIGNS.items():
        i = labels.index(l)
        a, b = sign * (z_lo[i] - z[i]), sign * (z_hi[i] - z[i])
        lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    v = lambda l: 10 ** x[labels.index(l)]
    hi = min(hi, -np.log10(v("log10_eps_e") + v("log10_eps_B")), -np.log10(v("log10_eps_e_r") + v("log10_eps_B_r")))
    return float(lo), float(hi)


# --------------------------------------------------------------------------------------------- report + plots
def physical(x, param_defs):
    return {p.name: (10 ** v if p.scale is epm.Scale.LOG else float(v)) for p, v in zip(param_defs, x)}


def derived(phys):
    """Physically interpretable combinations (wind medium, A = A_star * 5e11 g/cm)."""
    from grb.const import REDSHIFT
    m_p, c = 1.6726e-24, 2.998e10
    A = phys["A_star"] * 5e11
    r_dec = phys["E_iso"] / (4 * np.pi * A * c ** 2 * phys["Gamma0"] ** 2)
    t_dec_obs = (1 + REDSHIFT) * r_dec / (2 * phys["Gamma0"] ** 2 * c)
    return {"E_jet = E_iso(1-cos theta_c)": phys["E_iso"] * (1 - np.cos(phys["theta_c"])),
            "Gamma0 theta_c": phys["Gamma0"] * phys["theta_c"],
            "t_dec,thin (obs) [s]": t_dec_obs, "(1+z) tau / t_dec,thin": (1 + REDSHIFT) * phys["tau"] / t_dec_obs,
            "R_B = eps_B_r / eps_B": phys["eps_B_r"] / phys["eps_B"],
            "eps_e_r / eps_e": phys["eps_e_r"] / phys["eps_e"],
            "E_iso xi_e_r": phys["E_iso"] * phys["xi_e_r"], "E_iso xi_e": phys["E_iso"] * phys["xi_e"]}


def stage_report(outdir, context, param_defs, labels, mode_ids):
    modes = list(np.load(outdir / "modes_line.npy", allow_pickle=True))
    bounds = _sampled_bounds(param_defs)
    post = np.load(FINAL_DIR / "samples.npy")
    ext = [np.subtract(*ridge_extent(x, bounds, labels)[::-1]) for x in post[::30]]
    log(outdir, f"final posterior: scaling-ridge extent median {np.median(ext):.2f} dex, 16-84% "
                f"[{np.percentile(ext, 16):.2f}, {np.percentile(ext, 84):.2f}]")
    rows = []
    for m in [modes[i] for i in mode_ids]:
        dec = posterior_decomposition(m["x"], *context)
        phys = physical(m["x"], param_defs)
        rows.append({"mode": m["mode"], "source": m["source"], "n_merged": m["n_merged"],
                     **{k: float(v) for k, v in dec.items()}, **phys, **{k: float(v) for k, v in derived(phys).items()},
                     "ridge_coordinate": float(ridge_split(m["x"], labels)[0]),
                     "scaling_ridge_extent_dex": float(np.subtract(*ridge_extent(m["x"], bounds, labels)[::-1])),
                     "at_bound": [l for l, v, (lo, hi) in zip(labels, m["x"], _sampled_bounds(param_defs))
                                  if min(v - lo, hi - v) < 1e-4 * (hi - lo)]})
    with open(outdir / "modes_report.json", "w") as fh:
        json.dump(rows, fh, indent=1, default=float)
    keys = [k for k in rows[0] if k not in ("source", "at_bound")]
    with open(outdir / "modes_report.txt", "w") as fh:
        fh.write(f"{'quantity':34s}" + "".join(f"{'mode ' + str(r['mode']):>14s}" for r in rows) + "\n")
        for k in keys:
            fh.write(f"{k:34s}" + "".join(f"{r[k]:14.5g}" for r in rows) + "\n")
        fh.write(f"{'at prior bound':34s}" + "  ".join(",".join(r["at_bound"]) or "-" for r in rows) + "\n")
        for r in rows:
            fh.write(f"mode {r['mode']}: {r['source']}\n")
    print((outdir / "modes_report.txt").read_text())


def stage_plots(outdir, context, param_defs, labels, mode_ids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from VegasAfterglow.units import Hz, mJy, sec
    from grb.utils import flux_error

    _, xrt_data, i_data, xrt_index_data, _ = context
    modes = list(np.load(outdir / "modes_line.npy", allow_pickle=True))
    sel = [modes[i] for i in mode_ids]
    xt, xf, xe = xrt_data["time"].to_numpy(float), xrt_data["flux"].to_numpy(float), flux_error(xrt_data)
    it, i_f, ie = i_data["time"].to_numpy(float), i_data["flux_mJy"].to_numpy(float), i_data["flux_mJy_error"].to_numpy(float)
    i_nu = float(i_data["frequency_Hz"].iloc[0])
    tg_x = np.geomspace(min(xt.min(), it.min()) * 0.5, xt.max() * 1.5, 300)
    tg_i = np.geomspace(it.min() * 0.5, it.max() * 1.5, 300)
    colors = plt.cm.tab10(np.arange(10))

    def curves(x):
        grid = epm.evaluate_components(x, param_defs, tg_x * sec, tg_i * sec, i_nu * Hz)
        at = epm.evaluate_components(x, param_defs, xt * sec, it * sec, i_nu * Hz)
        model, _ = epm.build_model_from_params(x, param_defs)
        beta = epm.model_spectral_index(model, xrt_index_data["time"])
        return grid, at, beta

    cache = {m["mode"]: curves(m["x"]) for m in sel}
    for m in sel:
        grid, at, beta = cache[m["mode"]]
        dec = posterior_decomposition(m["x"], *context)
        fig, ax = plt.subplots(2, 3, figsize=(17, 7.5), sharex="col", gridspec_kw={"height_ratios": [3, 1.2]})
        ax[0, 0].errorbar(xt, xf, yerr=xe, fmt="o", ms=3, color="k", label="XRT (pre-flare)")
        ax[0, 0].plot(tg_x, grid[0], "k-", label="total")
        ax[0, 0].plot(tg_x, grid[1], "b:", lw=1.8, label="FS")
        ax[0, 0].plot(tg_x, grid[2], "m-.", lw=1.5, label="RS")
        ax[1, 0].errorbar(xt, (xf - at[0]) / xe, yerr=1, fmt="o", ms=3, color="k")
        ax[0, 1].errorbar(it, i_f, yerr=ie, fmt="o", ms=3, color="r", label="i band")
        ax[0, 1].plot(tg_i, grid[3] / mJy, "r-", label="total")
        ax[0, 1].plot(tg_i, grid[4] / mJy, "b:", lw=1.8, label="FS")
        ax[0, 1].plot(tg_i, grid[5] / mJy, "m-.", lw=1.5, label="RS")
        ax[1, 1].errorbar(it, (i_f - at[3] / mJy) / ie, yerr=1, fmt="o", ms=3, color="r")
        bt, bo = xrt_index_data["time"], xrt_index_data["beta"]
        ax[0, 2].errorbar(bt, bo, yerr=[xrt_index_data["beta_err_low"], xrt_index_data["beta_err_high"]], fmt="o", ms=3, color="g", label="XRT beta")
        ax[0, 2].plot(bt, beta, "k-", marker=".", label="model")
        err = np.where(beta > bo, xrt_index_data["beta_err_high"], xrt_index_data["beta_err_low"])
        ax[1, 2].errorbar(bt, (bo - beta) / err, yerr=1, fmt="o", ms=3, color="g")
        for a in ax[0, :2]:
            a.set_xscale("log"); a.set_yscale("log"); a.legend(fontsize=8); a.grid(alpha=0.25, which="both")
        ax[0, 0].set_ylim(xf.min() / 5, xf.max() * 5)
        ax[0, 1].set_ylim(i_f.min() / 5, i_f.max() * 5)
        ax[0, 2].set_xscale("log"); ax[0, 2].legend(fontsize=8); ax[0, 2].grid(alpha=0.25)
        for a in ax[1]:
            a.axhline(0, color="gray", lw=0.8); a.set_xlabel("time since trigger [s]"); a.grid(alpha=0.25)
        ax[1, 0].set_ylabel("(data - model) / err")
        ax[0, 0].set_ylabel("flux [erg cm$^{-2}$ s$^{-1}$]"); ax[0, 1].set_ylabel("flux density [mJy]"); ax[0, 2].set_ylabel(r"$\beta$")
        ax[0, 0].set_title(f"XRT  chi2 = {dec['chi2_XRT_flux']:.1f} (n={len(xt)})")
        ax[0, 1].set_title(f"i band  chi2 = {dec['chi2_i_band']:.1f} (n={len(it)})")
        ax[0, 2].set_title(f"XRT spectral index  chi2 = {dec['chi2_XRT_spectral_index']:.1f} (n={len(bt)})")
        fig.suptitle(f"mode {m['mode']}: log P = {dec['log_posterior']:.2f}, log L = {dec['log_likelihood']:.2f}, "
                     f"tau = {m['tau']:.1f} s", fontsize=12)
        fig.tight_layout()
        fig.savefig(outdir / f"mode{m['mode']}_lightcurves.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(14, 8), sharex="col", gridspec_kw={"height_ratios": [3, 1.2]})
    ax[0, 0].errorbar(xt, xf, yerr=xe, fmt="o", ms=3, color="k")
    ax[0, 1].errorbar(it, i_f, yerr=ie, fmt="o", ms=3, color="k")
    for k, m in enumerate(sel):
        grid, at, _ = cache[m["mode"]]
        c = colors[k % 10]
        lab = f"mode {m['mode']} (tau {m['tau']:.0f} s, log P {m['lp']:.1f})"
        ax[0, 0].plot(tg_x, grid[0], "-", color=c, label=lab)
        ax[0, 0].plot(tg_x, grid[2], "-.", color=c, lw=1, alpha=0.7)
        ax[0, 1].plot(tg_i, grid[3] / mJy, "-", color=c, label=lab)
        ax[0, 1].plot(tg_i, grid[4] / mJy, ":", color=c, lw=1.3, alpha=0.8)
        ax[0, 1].plot(tg_i, grid[5] / mJy, "-.", color=c, lw=1, alpha=0.7)
        ax[1, 0].plot(xt, (xf - at[0]) / xe, "o-", ms=3, color=c, lw=0.6)
        ax[1, 1].plot(it, (i_f - at[3] / mJy) / ie, "o-", ms=3, color=c, lw=0.6)
    for a in ax[0]:
        a.set_xscale("log"); a.set_yscale("log"); a.grid(alpha=0.25, which="both"); a.legend(fontsize=8)
    ax[0, 0].set_ylim(xf.min() / 5, xf.max() * 5)
    ax[0, 1].set_ylim(i_f.min() / 5, i_f.max() * 5)
    for a in ax[1]:
        a.axhline(0, color="gray", lw=0.8); a.set_xlabel("time since trigger [s]"); a.grid(alpha=0.25)
    ax[0, 0].set_title("XRT: total (solid), RS (dash-dot)")
    ax[0, 1].set_title("i band: total (solid), FS (dotted), RS (dash-dot)")
    ax[1, 0].set_ylabel("(data - model) / err")
    fig.tight_layout()
    fig.savefig(outdir / "modes_comparison.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["screen", "de", "local", "cluster", "relax", "pr_profile", "report", "plots"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--npool", type=int, default=20)
    ap.add_argument("--screen-log2", type=int, default=16)
    ap.add_argument("--de-maxiter", type=int, default=150)
    ap.add_argument("--n-per-band", type=int, default=10)
    ap.add_argument("--pilot", type=int, default=0, help="local: run only the first N seeds (cost benchmark)")
    ap.add_argument("--lp-window", type=float, default=150.0)
    ap.add_argument("--max-reps", type=int, default=60)
    ap.add_argument("--polish-top", type=int, default=20)
    ap.add_argument("--modes", type=int, nargs="*", default=None)
    ap.add_argument("--pairs", nargs="*", default=[], help="relax: mode pairs as a-b")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    param_defs, labels, context = final_setup()
    bounds = _sampled_bounds(param_defs)
    log(outdir, f"stage {args.stage}: {vars(args)}; numerics {epm.MODEL_RESOLUTION} rtol {epm.MODEL_RTOL}; labels {labels}")
    if args.stage == "screen":
        stage_screen(outdir, context, labels, bounds, args.npool, args.screen_log2)
    elif args.stage == "de":
        runs = [("full_s1", None, 1), ("full_s2", None, 2), ("full_s3", None, 3)]
        runs += [(f"{band}_s{s}", band, 10 * s + k) for k, band in enumerate(("low", "intermediate", "high")) for s in (1, 2)]
        stage_de(outdir, context, labels, bounds, args.npool, args.de_maxiter, runs)
    elif args.stage == "local":
        scales, post_sub = posterior_scales(labels)
        seeds = collect_seeds(outdir, labels, scales, post_sub, args.n_per_band)
        log(outdir, f"local: {len(seeds)} seeds")
        with open(outdir / "local_seeds.json", "w") as fh:
            json.dump([{"source": l, "x": np.asarray(x).tolist(), "lp_seed": _seed_lp(context, x)} for l, x in seeds], fh, indent=1)
        if args.pilot:
            rng = np.random.default_rng(0)
            seeds = [seeds[i] for i in rng.choice(len(seeds), args.pilot, replace=False)]
        stage_local(outdir, context, labels, bounds, args.npool, seeds, SHORT_BUDGET, "local_pilot" if args.pilot else "local", SHORT_SECONDS)
    elif args.stage == "cluster":
        scales, _ = posterior_scales(labels)
        stage_cluster(outdir, context, labels, bounds, args.npool, scales, args.lp_window, args.max_reps, args.polish_top)
    elif args.stage == "relax":
        stage_relax(outdir, context, bounds, args.npool, [tuple(int(v) for v in p.split("-")) for p in args.pairs])
    elif args.stage == "pr_profile":
        stage_pr_profile(outdir, context, labels, bounds, args.npool, args.modes,
                         [2.1, 2.2, 2.35, 2.5, 2.7, 2.9, 3.1, 3.3, 3.45, 3.5])
    elif args.stage == "report":
        stage_report(outdir, context, param_defs, labels, args.modes)
    elif args.stage == "plots":
        stage_plots(outdir, context, param_defs, labels, args.modes)


def _seed_lp(context, x):
    _init_early_phase_context(context)
    lp = _eval(np.clip(x, *_sampled_bounds(context[0]).T))
    return lp if np.isfinite(lp) else None


if __name__ == "__main__":
    main()
