#!/usr/bin/env python3
"""Shared optimisation utilities for the early-phase fit (no physics: everything scores via early_phase_model).

robust_minimize          bounded Powell / Nelder-Mead restarts that keep the best point ever evaluated
_optimize_task           one free optimisation from a seed (pool worker)
_profile_task            one optimisation with a single parameter held fixed (pool worker)
_fixed_parameter_scan    independent + forward / reverse continuation scans over a fixed-parameter grid
find_start_point         differential-evolution seed for the legacy sampler
"""
from datetime import datetime

import numpy as np

from early_phase_model import _init_early_phase_context, _neg_log_probability_from_context


PROFILE_BUDGET = {"max_rounds": 6, "tol": 1e-2, "powell_maxfev": 8000, "nm_maxfev": 4000}
POLISH_BUDGET = {"max_rounds": 15, "tol": 1e-3, "powell_maxfev": 20000, "nm_maxfev": 6000}


def find_start_point(bounds, map_func, nopt, rng):
    """Global optimisation of log_probability to seed the walkers near the mode."""
    from scipy.optimize import differential_evolution

    if nopt <= 0:
        return None
    result = differential_evolution(
        _neg_log_probability_from_context,
        bounds,
        maxiter=nopt,
        popsize=10,
        tol=1e-6,
        seed=int(rng.integers(2**31)),
        workers=map_func if map_func is not None else 1,
        updating="deferred" if map_func is not None else "immediate",
        polish=False,
        disp=False,
    )
    print(f"Start-point optimisation: best log_prob = {-result.fun:.3f} ({result.nfev} evaluations)")
    return result.x if result.fun < 1e29 else None


def robust_minimize(objective, start, bounds, f_start=None, max_rounds=30, tol=1e-3,
                    powell_maxfev=200000, nm_maxfev=6000):
    """Minimise `objective` with bounded Powell / Nelder-Mead restarts, tracking the best point seen.

    scipy's bounded Powell can return a point worse than ones it evaluated and stalls on the
    curved high-tau ridge, so the best point over all evaluations is kept and both optimisers
    are restarted from it until the improvement in a round drops below `tol`.
    Returns (x, f, n_evaluations, n_rounds, last_powell_success, last_powell_message).
    """
    from scipy.optimize import minimize

    best = {"x": np.array(start, dtype=float), "f": objective(start) if f_start is None else f_start}
    n_calls = [0]

    def tracked(x):
        n_calls[0] += 1
        f = objective(x)
        if f < best["f"]:
            best["x"], best["f"] = np.array(x, dtype=float), f
        return f

    n_rounds, success, message = 0, False, ""
    for n_rounds in range(1, max_rounds + 1):
        f_before = best["f"]
        result = minimize(
            tracked, best["x"], method="Powell", bounds=bounds,
            options={"maxiter": 20000, "maxfev": powell_maxfev, "xtol": 1e-8, "ftol": 1e-10},
        )
        success, message = result.success, result.message
        minimize(
            tracked, best["x"], method="Nelder-Mead", bounds=bounds,
            options={"adaptive": True, "maxiter": nm_maxfev, "maxfev": nm_maxfev, "xatol": 1e-9, "fatol": 1e-9},
        )
        if f_before - best["f"] < tol:
            break
    return best["x"], best["f"], n_calls[0], n_rounds, success, message


def _profile_task(task):
    """Maximise log_probability over every parameter except tau, with tau held fixed (worker process)."""
    log_tau, tau_i, seed_label, seed_vec, bounds, budget = task
    free = [j for j in range(len(seed_vec)) if j != tau_i]

    def expand(z):
        x = np.empty(len(seed_vec))
        x[free] = z
        x[tau_i] = log_tau
        return x

    start = np.clip(np.asarray(seed_vec, dtype=float)[free], bounds[free, 0], bounds[free, 1])
    z, f, n_evals, n_rounds, _, _ = robust_minimize(
        lambda z: _neg_log_probability_from_context(expand(z)), start, bounds[free], **budget
    )
    return {"source": seed_label, "x": expand(z), "f": f, "nfev": n_evals, "rounds": n_rounds}


def _optimize_task(task):
    """Free-parameter robust optimisation from one seed (worker process)."""
    seed_label, seed_vec, bounds, budget = task
    start = np.clip(np.asarray(seed_vec, dtype=float), bounds[:, 0], bounds[:, 1])
    x, f, n_evals, n_rounds, success, message = robust_minimize(_neg_log_probability_from_context, start, bounds, **budget)
    return {"source": seed_label, "x": x, "f": f, "nfev": n_evals, "rounds": n_rounds}


def _fixed_parameter_scan(fixed_i, grid, bounds, context, npool, budget, seed_at, independent_seeds, extra_seeds, name):
    """Phase A (independent optimisations at every grid value) + forward / reverse continuation scans.

    `seed_at(x, k, perturb)` returns a feasible seed at grid index k; `independent_seeds(k)` and
    `extra_seeds(k)` give (label, vector) lists for Phase A and for every continuation step.
    Returns (independent, forward, reverse), each {k: result with 'f', 'x', 'source', 'nfev', 'nfev_total'}.
    """
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    n = len(grid)
    best_of = lambda results: min(results, key=lambda r: r["f"])
    t_start = datetime.now()
    with ProcessPoolExecutor(max_workers=npool, initializer=_init_early_phase_context, initargs=(context,)) as pool:
        def submit(k, source, vec):
            return pool.submit(_profile_task, (grid[k], fixed_i, source, vec, bounds, budget))

        futures = {k: [submit(k, label, vec) for label, vec in independent_seeds(k)] for k in range(n)}
        indep = {}
        for k in range(n):
            results = [f.result() for f in futures[k]]
            indep[k] = {**best_of(results), "nfev_total": sum(r["nfev"] for r in results)}
            print(f"  [A] {name}={grid[k]:g}  log_post={-indep[k]['f']:12.3f}  ({indep[k]['source']})", flush=True)
        print(f"Phase A done after {datetime.now() - t_start}", flush=True)

        def chain(order, tag):
            out, prev = {}, None
            for k in order:
                seeds = [("indep-best", seed_at(indep[k]["x"], k, 0.0)),
                         ("indep-best-perturbed-0.01", seed_at(indep[k]["x"], k, 0.01))] + extra_seeds(k)
                if prev is not None:
                    seeds += [(f"{tag}-prev", seed_at(prev, k, 0.0)),
                              (f"{tag}-prev-perturbed-0.01", seed_at(prev, k, 0.01)),
                              (f"{tag}-prev-perturbed-0.03", seed_at(prev, k, 0.03)),
                              (f"{tag}-prev-perturbed-0.05", seed_at(prev, k, 0.05))]
                results = [f.result() for f in [submit(k, label, vec) for label, vec in seeds]]
                out[k] = {**best_of(results), "nfev_total": sum(r["nfev"] for r in results)}
                prev = out[k]["x"]
                print(f"  [{tag}] {name}={grid[k]:g}  log_post={-out[k]['f']:12.3f}  ({out[k]['source']})", flush=True)
            return out

        with ThreadPoolExecutor(max_workers=2) as threads:
            fwd = threads.submit(chain, list(range(n)), "forward")
            rev = threads.submit(chain, list(range(n - 1, -1, -1)), "reverse")
            forward, reverse = fwd.result(), rev.result()
    print(f"Phase B done after {datetime.now() - t_start}", flush=True)
    return indep, forward, reverse
