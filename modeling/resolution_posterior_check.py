#!/usr/bin/env python3
"""Does the production model resolution reshape the posterior, or only offset log L? (no sampling)

    python resolution_posterior_check.py RUN1_DIR RUN2_DIR [--npool 20] [--outdir DIR]

Evaluates posterior points of the two production chains (random post-burn draws plus parameter-range tails) at the
production resolution and at a finer, converged resolution (and, for a subset, at an even finer one to check that
the "fine" setting is itself converged). The finer settings are applied only inside this script's worker processes.
Reports Delta log L = log L_fine - log L_production and its XRT / i-band / spectral-index parts, its correlations with
the parameters, and the importance-reweighted (w ~ exp(Delta log L)) 16/50/84% intervals of the random draws.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import emcee
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

import early_phase_model as m

PRODUCTION = ((0.1, 0.25, 10), 1e-5)
FINE = ((1, 4, 100), 1e-8)
FINER = ((2, 8, 200), 1e-9)
INTERMEDIATE = [((0.1, 0.25, 30), 1e-5), ((0.3, 1, 30), 1e-6), ((0.5, 2, 50), 1e-7)]
COOLING = "smooth"
_CTX = None


def _init(resolution, rtol):
    global _CTX
    m.MODEL_RESOLUTION, m.MODEL_RTOL = resolution, rtol         # this worker process only
    defs = m.make_param_defs("production", "wind", free_xi_e=True, free_xi_e_r=True)
    _CTX = (defs, *m.load_fit_data(include_spectral_index=True), COOLING)


def _evaluate(x):
    defs, xrt, i_data, xi_data, cooling = _CTX
    c = m.chi2_components(x, defs, xrt, i_data, xi_data, cooling)
    return [c[0], c[1], c[2], -0.5 * float(sum(c)), m.log_prior(x, defs)]


def run(points, setting, npool):
    with ProcessPoolExecutor(max_workers=npool, initializer=_init, initargs=setting) as pool:
        return np.array(list(pool.map(_evaluate, list(points), chunksize=4)))


def weighted_quantiles(v, w, qs):
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    cw = np.divide(np.subtract(cw, 0.5 * w[o]), cw[-1])
    return np.interp(qs, cw, v[o])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs=2)
    ap.add_argument("--npool", type=int, default=20)
    ap.add_argument("--ndraws", type=int, default=400, help="Random post-burn draws per run (default 400).")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    outdir = Path(args.outdir) if args.outdir else Path(args.runs[0]).parent / "early_phase_resolution_check"
    outdir.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(t=""):
        lines.append(t)
        print(t, flush=True)

    defs = m.make_param_defs("production", "wind", free_xi_e=True, free_xi_e_r=True)
    labels = [m.param_label(d) for d in defs]
    rng = np.random.default_rng(20261008)
    chains = [emcee.backends.HDFBackend(f"{r}/chain.h5", name="mcmc", read_only=True).get_chain()[50000:] for r in args.runs]
    draws = []
    for c in chains:                                  # distinct (step, walker) pairs, spread over the 50k-100k window
        flat_idx = rng.choice(c.shape[0] * c.shape[1], args.ndraws, replace=False)
        draws.append(c.reshape(-1, len(labels))[flat_idx])
    draws = np.concatenate(draws)
    pooled = np.concatenate([c[::20].reshape(-1, len(labels)) for c in chains])
    tails = []
    for j in range(len(labels)):                      # 3 lowest and 3 highest of every parameter
        o = np.argsort(pooled[:, j])
        tails += [pooled[k] for k in list(o[:3]) + list(o[-3:])]
    tails = np.array(tails)
    pts = np.concatenate([draws, tails])
    n_draws = len(draws)
    say(f"=== Model-resolution posterior check ({datetime.now():%Y-%m-%d %H:%M}) ===")
    say(f"production resolution {PRODUCTION[0]}, rtol {PRODUCTION[1]:g}; fine {FINE[0]}, rtol {FINE[1]:g}; "
        f"extra-fine check {FINER[0]}, rtol {FINER[1]:g} (diagnostic worker processes only)")
    say(f"{n_draws} random post-burn draws ({args.ndraws} per run, steps 50k-100k) + {len(tails)} tail points "
        f"(3 lowest / 3 highest of each parameter)")

    t0 = datetime.now()
    prod = run(pts, PRODUCTION, args.npool)
    t1 = datetime.now()
    fine = run(pts, FINE, args.npool)
    t2 = datetime.now()
    sub = rng.choice(n_draws, 100, replace=False)
    finer = run(pts[sub], FINER, args.npool)
    t3 = datetime.now()
    say(f"wall time: production {(t1 - t0).total_seconds():.0f} s, fine {(t2 - t1).total_seconds():.0f} s, "
        f"extra-fine (100 points) {(t3 - t2).total_seconds():.0f} s")
    assert np.all(np.isfinite(prod[:, 3])) and np.all(np.isfinite(fine[:, 3]))

    d_ll = np.subtract(fine[:, 3], prod[:, 3])
    d_parts = -0.5 * np.subtract(fine[:, :3], prod[:, :3])
    conv = np.subtract(finer[:, 3], fine[sub, 3])
    say(f"\nconvergence of the fine setting: log L(extra-fine) - log L(fine) on 100 draws: median {np.median(conv):+.4f}, "
        f"16-84% [{np.percentile(conv, 16):+.4f}, {np.percentile(conv, 84):+.4f}], max |.| {np.abs(conv).max():.4f}")

    def dist(name, v):
        q = np.percentile(v, [2.5, 16, 50, 84, 97.5])
        say(f"  {name:<34} median {q[2]:+.3f}  16-84% [{q[1]:+.3f}, {q[3]:+.3f}]  2.5-97.5% [{q[0]:+.3f}, {q[4]:+.3f}]  "
            f"min {v.min():+.3f}  max {v.max():+.3f}")

    for title, sel in (("random posterior draws", slice(0, n_draws)), ("tail points", slice(n_draws, None))):
        say(f"\n--- Delta log L = log L_fine - log L_production: {title} ---")
        dist("Delta log L", d_ll[sel])
        dist("Delta log L - median (random draws)", d_ll[sel] - np.median(d_ll[:n_draws]))
        for k, name in enumerate(("XRT flux", "i band", "XRT spectral index")):
            dist(f"  part from {name}", d_parts[sel, k])

    say("\n--- correlation of Delta log L with the parameters (random draws; sampled space) ---")
    corr = {}
    for j, l in enumerate(labels):
        pr, sr = pearsonr(draws[:, j], d_ll[:n_draws])[0], spearmanr(draws[:, j], d_ll[:n_draws])[0]
        corr[l] = (pr, sr)
        say(f"  {l:<15} Pearson {pr:+.3f}  Spearman {sr:+.3f}")
    pr, sr = pearsonr(prod[:n_draws, 3], d_ll[:n_draws])[0], spearmanr(prod[:n_draws, 3], d_ll[:n_draws])[0]
    say(f"  {'log L (prod.)':<15} Pearson {pr:+.3f}  Spearman {sr:+.3f}")
    X = np.column_stack([draws, np.ones(n_draws)])
    resid = np.subtract(d_ll[:n_draws], X @ np.linalg.lstsq(X, d_ll[:n_draws], rcond=None)[0])
    say(f"  linear model in all 14 parameters explains R^2 = {1 - resid.var() / d_ll[:n_draws].var():.3f} of the "
        f"Delta log L variance")

    # importance reweighting of the random draws
    lw = d_ll[:n_draws] - d_ll[:n_draws].max()
    w = np.exp(lw)
    w = w / w.sum()
    ess = 1.0 / np.sum(w ** 2)
    say(f"\n--- importance reweighting w ~ exp(Delta log L), {n_draws} random draws ---")
    say(f"  ESS {ess:.1f} of {n_draws} ({ess / n_draws:.1%}); max normalised weight {w.max():.4f} "
        f"(uniform {1 / n_draws:.4f}); weights > 5x uniform: {int((w > 5 / n_draws).sum())}")
    say(f"  {'parameter':<15} {'production 16/50/84':>34} {'resolution-corrected 16/50/84':>36} {'dMed/sig':>9} "
        f"{'boot sd':>8} {'d16/sig':>8} {'(sd)':>6} {'d84/sig':>8} {'(sd)':>6}")
    rows = []
    boot_rng = np.random.default_rng(1)
    for j, l in enumerate(labels):
        v = draws[:, j]
        q0 = np.percentile(v, [16, 50, 84])
        q1 = weighted_quantiles(v, w, [0.16, 0.5, 0.84])
        sig = 0.5 * (q0[2] - q0[0])
        shifts = []
        for _ in range(200):                           # paired bootstrap of the 16 / 50 / 84 % shifts
            b = boot_rng.integers(0, n_draws, n_draws)
            wb = w[b] / w[b].sum()
            shifts.append((weighted_quantiles(v[b], wb, [0.16, 0.5, 0.84]) - np.percentile(v[b], [16, 50, 84])) / sig)
        sd = np.std(shifts, axis=0)
        rows.append((l, q0, q1))
        say(f"  {l:<15} {q0[1]:>10.5g} [{q0[0]:.5g}, {q0[2]:.5g}]".ljust(52) + f"{q1[1]:>10.5g} [{q1[0]:.5g}, {q1[2]:.5g}]".ljust(38)
            + f"{(q1[1] - q0[1]) / sig:>+9.3f} {sd[1]:>8.3f} {(q1[0] - q0[0]) / sig:>+8.3f} {sd[0]:>6.3f} "
              f"{(q1[2] - q0[2]) / sig:>+8.3f} {sd[2]:>6.3f}")
    say("  (shifts in units of the production posterior half-width; sd: paired-bootstrap s.d. of each shift)")
    widths0 = {l: q0[2] - q0[0] for l, q0, _ in rows}
    say("  16-84% width ratio corrected / production: " + ", ".join(f"{l} {(q1[2] - q1[0]) / widths0[l]:.2f}" for l, _, q1 in rows))

    # cheaper settings: accuracy against the converged likelihood and single-process cost
    say("\n--- intermediate resolutions: error against the fine (converged) log L on the random draws ---")
    import time
    for setting in [PRODUCTION] + INTERMEDIATE + [FINE]:
        ll = fine[:n_draws, 3] if setting == FINE else (prod[:n_draws, 3] if setting == PRODUCTION else run(draws, setting, args.npool)[:, 3])
        err = np.subtract(ll, fine[:n_draws, 3])
        _init(*setting)
        t = time.perf_counter()
        for x in draws[:30]:
            _evaluate(x)
        cost = (time.perf_counter() - t) / 30
        lw2 = -err - (-err).max()
        w2 = np.exp(lw2) / np.exp(lw2).sum()
        say(f"  resolution {str(setting[0]):<16} rtol {setting[1]:<6g} error vs fine: 16-84% [{np.percentile(err, 16):+.3f}, "
            f"{np.percentile(err, 84):+.3f}], max |.| {np.abs(err).max():.3f}; IS ESS to fine {1 / np.sum(w2 ** 2) / n_draws:.0%}; "
            f"{1e3 * cost:.0f} ms per evaluation")

    np.savez(outdir / "resolution_check.npz", points=pts, n_draws=n_draws, production=prod, fine=fine,
             finer_subset=finer, subset=sub, labels=np.array(labels), delta_logL=d_ll, weights=w)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].hist(d_ll[:n_draws], bins=50, histtype="step", color="k", label="random draws")
    axes[0].hist(d_ll[n_draws:], bins=30, histtype="step", color="C3", label="tail points")
    axes[0].set_xlabel("log L_fine - log L_production")
    axes[0].legend()
    for k, name in enumerate(("XRT flux", "i band", "spectral index")):
        axes[1].hist(d_parts[:n_draws, k], bins=50, histtype="step", label=name)
    axes[1].set_xlabel("part of Delta log L")
    axes[1].legend()
    axes[2].scatter(draws[:, labels.index("p_r")], d_ll[:n_draws], s=5)
    axes[2].set_xlabel("p_r")
    axes[2].set_ylabel("Delta log L")
    fig.tight_layout()
    fig.savefig(outdir / "resolution_check.png", dpi=120)
    plt.close(fig)
    say(f"\nWritten to {outdir}")
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
