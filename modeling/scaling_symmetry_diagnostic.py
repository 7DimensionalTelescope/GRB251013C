#!/usr/bin/env python3
"""Is the common scaling E_iso, A_star -> /f; eps_e, eps_B, xi_e, eps_e_r, eps_B_r, xi_e_r -> *f a likelihood symmetry?

    python scaling_symmetry_diagnostic.py RUN1_DIR RUN2_DIR [--npool 20] [--outdir DIR]

No sampling. Uses posterior points of the two production chains (random post-burn points plus points chosen at the
important boundaries), the exact f interval allowed by every box bound and both eps-sum constraints, and the unchanged
likelihood (early_phase_model). Reports likelihood-component changes against the numerical noise floor (f = 1 +- 1e-9),
model-curve changes (total / forward / reverse shock, inside and beyond the data range), the invariant combinations,
and the implied conditional prior of the scale s = xi_e under the current production prior, checked against the
existing chains with a probability-integral transform.
"""
import argparse
import functools
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

import early_phase_model as m
from early_phase_mcmc import _autocorr

COOLING = "smooth"
UP = ("E_iso", "A_star")                                   # divided by f
DOWN = ("eps_e", "eps_B", "xi_e", "eps_e_r", "eps_B_r", "xi_e_r")    # multiplied by f
COMPONENTS = ("chi2_XRT_flux", "chi2_i_band", "chi2_XRT_spectral_index", "chi2_total", "log_likelihood")
INVARIANTS = ("E_g = E_iso xi_e", "A_g = A_star xi_e", "eps_e/xi_e", "eps_B/xi_e", "eps_e_r/xi_e", "eps_B_r/xi_e",
              "xi_e_r/xi_e")

_CTX = None


def _init(ctx, resolution=None, rtol=None, radiative_fireball=True):
    """Worker set-up. The overrides exist only for the numerical-vs-physical test below; they change module globals
    inside this diagnostic's worker processes, never the production code or defaults."""
    global _CTX
    if resolution is not None:
        m.MODEL_RESOLUTION, m.MODEL_RTOL = resolution, rtol
    if not radiative_fireball:
        m.Model = functools.partial(m.Model, radiative_fireball=False)
    _CTX = ctx
    m._init_early_phase_context(ctx)


def _logp(x):
    defs, xrt, i_data, xi_data, cooling = _CTX
    return m.log_probability(x, defs, xrt, i_data, xi_data, cooling)


def _evaluate(x):
    defs, xrt, i_data, xi_data, cooling = _CTX
    c = m.chi2_components(x, defs, xrt, i_data, xi_data, cooling)
    return [c[0], c[1], c[2], float(sum(c)), -0.5 * float(sum(c)), m.log_prior(x, defs)]


def _curves(x):
    defs, xrt, i_data, xi_data, cooling = _CTX
    t_x = np.geomspace(100.0, 2e4, 60)
    t_i = np.geomspace(30.0, 2e4, 60)
    out = m.evaluate_components(x, defs, t_x, t_i, float(i_data["frequency_Hz"].iloc[0]))
    model, _ = m.build_model_from_params(x, defs)
    beta = m.model_spectral_index(model, np.geomspace(300.0, 3000.0, 20))
    return np.concatenate([np.concatenate(out), beta])


def numpy_elision_bug_present():
    """True if `a - b` inside a function overwrites a large local array a (seen with Python 3.14 + NumPy 2.2.x:
    NumPy's temporary elision mistakes a borrowed local reference for a temporary). This script uses explicit
    np.add / np.subtract / np.divide on large locals, which never elide."""
    a = np.arange(200000, dtype=float)
    keep = a.copy()
    _ = a - np.ones_like(a)
    return not np.array_equal(a, keep)


def phys(x, labels):
    return {l.removeprefix("log10_"): (10 ** v if l.startswith("log10_") else v) for l, v in zip(labels, x)}


def scale(x, labels, f):
    y = np.array(x, dtype=float)
    for name in UP:
        y[labels.index(f"log10_{name}")] -= np.log10(f)
    for name in DOWN:
        lab = f"log10_{name}" if f"log10_{name}" in labels else name
        y[labels.index(lab)] = y[labels.index(lab)] + np.log10(f) if lab.startswith("log10_") else y[labels.index(lab)] * f
    return y


def f_interval(x, labels, defs):
    """[f_lo, f_hi] allowed by every box bound and both eps-sum constraints, plus which constraint sets each end."""
    p = phys(x, labels)
    b = {d.name: (d.lower, d.upper) for d in defs}
    lo, hi = [], []
    for name in UP:                                        # lower <= v / f <= upper
        lo.append((p[name] / b[name][1], f"{name} <= {b[name][1]:g}"))
        hi.append((p[name] / b[name][0], f"{name} >= {b[name][0]:g}"))
    for name in DOWN:                                      # lower <= v f <= upper
        lo.append((b[name][0] / p[name], f"{name} >= {b[name][0]:g}"))
        hi.append((b[name][1] / p[name], f"{name} <= {b[name][1]:g}"))
    for e, bb in m.EPS_SUM_PAIRS:
        hi.append((1.0 / (p[e] + p[bb]), f"{e} + {bb} <= 1"))
    f_lo, why_lo = max(lo)
    f_hi, why_hi = min(hi)
    return f_lo, f_hi, why_lo, why_hi


def invariants(x, labels):
    p = phys(x, labels)
    s = p["xi_e"]
    return np.array([p["E_iso"] * s, p["A_star"] * s, p["eps_e"] / s, p["eps_B"] / s, p["eps_e_r"] / s,
                     p["eps_B_r"] / s, p["xi_e_r"] / s])


def load_post(run, discard=50000):
    reader = emcee.backends.HDFBackend(f"{run}/chain.h5", name="mcmc", read_only=True)
    return reader.get_chain()[discard:], reader.get_log_prob()[discard:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs=2)
    ap.add_argument("--npool", type=int, default=20)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    outdir = Path(args.outdir) if args.outdir else Path(args.runs[0]).parent / f"early_phase_scaling_symmetry_{datetime.now():%Y%m%d_%H%M%S}"
    outdir.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(t=""):
        lines.append(t)
        print(t, flush=True)

    defs = m.make_param_defs("production", "wind", free_xi_e=True, free_xi_e_r=True)
    labels = [m.param_label(d) for d in defs]
    data = m.load_fit_data(include_spectral_index=True)
    ctx = (defs, *data, COOLING)
    rng = np.random.default_rng(20261008)
    chains = [load_post(r) for r in args.runs]

    # ---- representative posterior points: random + boundary-targeted ----
    pts, tags = [], []
    for k, (c, _) in enumerate(chains):
        for t, w in zip(rng.integers(0, len(c), 100), rng.integers(0, c.shape[1], 100)):
            pts.append(c[t, w])
            tags.append(f"run{k + 1}:random")
    pooled = np.concatenate([c[::50].reshape(-1, len(labels)) for c, _ in chains])
    P = {n: np.array([phys(x, labels)[n] for x in pooled]) for n in ("eps_e", "xi_e", "A_star", "E_iso", "xi_e_r",
                                                                        "eps_B", "p", "p_r")}
    epsr = np.array([phys(x, labels)["eps_e_r"] + phys(x, labels)["eps_B_r"] for x in pooled])
    targets = [("eps_e near 0.002", P["eps_e"], 1), ("xi_e near 1", P["xi_e"], -1), ("A_star near 10", P["A_star"], -1),
               ("E_iso near 1e55", P["E_iso"], -1), ("xi_e_r smallest", P["xi_e_r"], 1), ("eps_B near 0.05", P["eps_B"], -1),
               ("eps_r sum near 1", epsr, -1), ("p near 2.01", P["p"], 1), ("p_r near 2", P["p_r"], 1)]
    for name, v, sign in targets:
        for j in np.argsort(sign * v)[:5]:
            pts.append(pooled[j])
            tags.append(f"boundary:{name}")
    pts = np.array(pts)
    say(f"=== Common-scaling symmetry test ({datetime.now():%Y-%m-%d %H:%M}) ===")
    say("transformation: E_iso, A_star -> /f;  eps_e, eps_B, xi_e, eps_e_r, eps_B_r, xi_e_r -> *f")
    say(f"{len(pts)} posterior points: 100 random post-burn from each run + 5 at each of {len(targets)} boundaries")
    say(f"environment check: NumPy {np.__version__}; in-place temporary-elision bug on large local arrays present: "
        f"{numpy_elision_bug_present()} (large-array arithmetic below uses explicit ufunc calls)")

    # ---- allowed f intervals and the f grid ----
    intervals = [f_interval(x, labels, defs) for x in pts]
    from collections import Counter
    say("f interval [f_lo, f_hi]: median width log10(f_hi/f_lo) = "
        f"{np.median([np.log10(b / a) for a, b, _, _ in intervals]):.3f} dex, range "
        f"{min(np.log10(b / a) for a, b, _, _ in intervals):.3f}-{max(np.log10(b / a) for a, b, _, _ in intervals):.3f} dex")
    say("binding constraint at f_lo: " + ", ".join(f"{k} {v}" for k, v in Counter(w for _, _, w, _ in intervals).most_common()))
    say("binding constraint at f_hi: " + ", ".join(f"{k} {v}" for k, v in Counter(w for _, _, _, w in intervals).most_common()))
    jobs, meta = [], []
    for i, (x, (f_lo, f_hi, _, _)) in enumerate(zip(pts, intervals)):
        a, b = f_lo * (1 + 1e-9), f_hi * (1 - 1e-9)
        grid = [1.0, 1 + 1e-9, 1 - 1e-9] + list(np.geomspace(a, b, 7))
        for f in grid:
            jobs.append(scale(x, labels, f))
            meta.append((i, f, np.log10(f / a) / np.log10(b / a) if b > a else 0.0))
    say(f"{len(jobs)} evaluations (per point: f = 1, 1 +- 1e-9 noise probes, 7 values log-spaced over the full interval)")
    with ProcessPoolExecutor(max_workers=args.npool, initializer=_init, initargs=(ctx,)) as pool:
        res = np.array(list(pool.map(_evaluate, jobs, chunksize=4)))
        curve_jobs = [k for k, (i, f, _) in enumerate(meta) if i % 4 == 0 and (f == 1.0 or abs(f - 1) > 1e-6)]
        curves = np.array(list(pool.map(_curves, [jobs[k] for k in curve_jobs], chunksize=2)))
    meta = np.array(meta)
    idx = meta[:, 0].astype(int)
    f_all = meta[:, 1]
    base = {i: res[(idx == i) & (f_all == 1.0)][0] for i in range(len(pts))}
    delta = np.array([res[k] - base[idx[k]] for k in range(len(res))])
    is_noise = (np.abs(f_all - 1) > 0) & (np.abs(f_all - 1) < 1e-6)
    is_scan = np.abs(f_all - 1) > 1e-6
    say(f"all transformed points inside the prior box and constraints: {bool(np.all(np.isfinite(res[:, 5])))} "
        f"(log_prior finite for {np.isfinite(res[:, 5]).sum()}/{len(res)})")

    say("\n--- likelihood-component changes |transformed - original| ---")
    say(f"{'component':<26} {'noise probe f=1+-1e-9: max':>27} {'median':>10}   {'scan over interval: max':>24} {'median':>10} {'p95':>10}")
    for j, name in enumerate(COMPONENTS):
        nz, sc = np.abs(delta[is_noise, j]), np.abs(delta[is_scan, j])
        say(f"{name:<26} {nz.max():>27.2e} {np.median(nz):>10.2e}   {sc.max():>24.2e} {np.median(sc):>10.2e} "
            f"{np.percentile(sc, 95):>10.2e}")
    dll = delta[is_scan, 4]
    rel = np.abs(delta[is_scan, 3]) / np.array([base[i][3] for i in idx[is_scan]])
    say(f"max relative chi2_total change {rel.max():.2e}, median {np.median(rel):.2e}")
    say(f"|Delta log L| over the scan: fraction < 1e-3: {np.mean(np.abs(dll) < 1e-3):.3f}, < 0.01: "
        f"{np.mean(np.abs(dll) < 0.01):.3f}, < 0.1: {np.mean(np.abs(dll) < 0.1):.3f}, < 1: {np.mean(np.abs(dll) < 1):.3f}")
    # dependence on position in the interval and on the size of the step
    lf = np.abs(np.log10(f_all[is_scan]))
    for lo_, hi_ in ((0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 10)):
        sel = (lf >= lo_) & (lf < hi_)
        if sel.any():
            say(f"  |log10 f| in [{lo_}, {hi_}): {sel.sum():4d} evals, |dlogL| median {np.median(np.abs(dll[sel])):.2e}, "
                f"max {np.abs(dll[sel]).max():.2e}")
    worst = np.argsort(-np.abs(delta[:, 4]) * is_scan)[:8]
    say("largest |Delta log L| cases:")
    for k in worst:
        i = idx[k]
        p0 = phys(pts[i], labels)
        say(f"  point {i:3d} ({tags[i]}), f = {f_all[k]:.4g}: dlogL {delta[k, 4]:+.4f} (dchi2 XRT {delta[k, 0]:+.4f}, "
            f"i {delta[k, 1]:+.4f}, beta {delta[k, 2]:+.4f}); xi_e {p0['xi_e']:.3f}, xi_e_r {p0['xi_e_r']:.4f}, "
            f"p {p0['p']:.4f}, p_r {p0['p_r']:.4f}, eps_e {p0['eps_e']:.4f}")
    by_tag = {}
    for k in np.where(is_scan)[0]:
        by_tag.setdefault(tags[idx[k]].split(":")[0] if tags[idx[k]].startswith("run") else tags[idx[k]], []).append(abs(delta[k, 4]))
    say("max |Delta log L| by point group: " + "; ".join(f"{k} {max(v):.2e}" for k, v in by_tag.items()))

    # ---- model curves ----
    say("\n--- model curves (XRT 100 s - 20 ks, i band 30 s - 20 ks, XRT beta 300-3000 s; every 4th point) ---")
    cidx = idx[curve_jobs]
    cf = f_all[curve_jobs]
    names = ["XRT total", "XRT fwd", "XRT rvs", "i total", "i fwd", "i rvs"]
    ref = {i: curves[(cidx == i) & (cf == 1.0)][0] for i in set(cidx)}
    for s_, name in enumerate(names):
        sl = slice(60 * s_, 60 * (s_ + 1))
        d = [np.max(np.abs(curves[k, sl] / ref[cidx[k]][sl] - 1)) for k in range(len(curves)) if cf[k] != 1.0]
        say(f"  {name:<10} max relative change {np.max(d):.2e}, median {np.median(d):.2e}")
    d = [np.max(np.abs(curves[k, 360:] - ref[cidx[k]][360:])) for k in range(len(curves)) if cf[k] != 1.0]
    say(f"  {'XRT beta':<10} max absolute change {np.max(d):.2e}, median {np.median(d):.2e}")

    # ---- invariants ----
    say("\n--- invariant combinations under the transformation ---")
    inv_dev = np.zeros(7)
    for k in np.where(is_scan)[0]:
        inv_dev = np.maximum(inv_dev, np.abs(invariants(jobs[k], labels) / invariants(pts[idx[k]], labels) - 1))
    for name, v in zip(INVARIANTS, inv_dev):
        say(f"  {name:<18} max relative change {v:.2e}")
    say("  (the likelihood scan above is exactly a change of xi_e at fixed invariants: f = xi_e' / xi_e)")

    # ---- prior / bound analysis on the existing chains ----
    say("\n--- implied conditional prior of the scale s = xi_e at fixed invariants (current production prior) ---")
    say("prior density in (log10 E, log10 A, log10 eps x4, xi_e, xi_e_r) is uniform on the box x constraints;")
    say("(g = log10 invariants, rho = xi_e_r/xi_e, s = xi_e) has Jacobian |d old / d new| = s, so p(s | g, rho) ~ s on")
    say("[s_lo, s_hi] (the f interval times xi_e) and the marginal prior weight of (g, rho) is (s_hi^2 - s_lo^2) / 2.")
    pit, widths, lo_why, hi_why, s_vals, w_vals = [], [], Counter(), Counter(), [], []
    for c, _ in chains:
        for x in c[::25].reshape(-1, len(labels)):
            f_lo, f_hi, wl, wh = f_interval(x, labels, defs)
            s = phys(x, labels)["xi_e"]
            s_lo, s_hi = s * f_lo, s * f_hi
            pit.append((s ** 2 - s_lo ** 2) / (s_hi ** 2 - s_lo ** 2))
            widths.append(s_hi / s_lo)
            lo_why[wl] += 1
            hi_why[wh] += 1
            s_vals.append(s)
            w_vals.append(0.5 * (s_hi ** 2 - s_lo ** 2))
    pit = np.array(pit)
    from scipy.stats import kstest
    ks = kstest(pit, "uniform")
    say(f"PIT u = (s^2 - s_lo^2)/(s_hi^2 - s_lo^2) over {len(pit)} chain samples: mean {pit.mean():.3f} (uniform 0.5), "
        f"deciles {np.round(np.percentile(pit, np.arange(10, 100, 10)), 3).tolist()}, KS D {ks.statistic:.3f} "
        f"(samples are autocorrelated: D is descriptive)")
    say(f"s_hi / s_lo: median {np.median(widths):.2f}, 16-84% [{np.percentile(widths, 16):.2f}, {np.percentile(widths, 84):.2f}]")
    say("s_lo set by: " + ", ".join(f"{k} {100 * v / len(pit):.0f}%" for k, v in lo_why.most_common()))
    say("s_hi set by: " + ", ".join(f"{k} {100 * v / len(pit):.0f}%" for k, v in hi_why.most_common()))
    say(f"marginal prior weight (s_hi^2 - s_lo^2)/2: 16/50/84% = {np.round(np.percentile(w_vals, [16, 50, 84]), 4).tolist()} "
        f"(varies by x{np.percentile(w_vals, 84) / np.percentile(w_vals, 16):.1f} across the posterior)")

    # ---- mixing of the scale vs the invariants in the existing chains ----
    say("\n--- autocorrelation in the existing chains (25k windows): scale s = xi_e vs invariant coordinates ---")
    for k, (c, _) in enumerate(chains):
        for a, b in ((0, 25000), (25000, 50000)):
            w = c[a:b]
            pp = {n: w[..., labels.index(f"log10_{n}") if f"log10_{n}" in labels else labels.index(n)] for n in
                  ("E_iso", "A_star", "eps_e", "eps_B", "eps_e_r", "eps_B_r", "xi_e", "xi_e_r")}
            ls = np.log10(pp["xi_e"])
            inv = np.stack([np.add(pp["E_iso"], ls), np.add(pp["A_star"], ls), np.subtract(pp["eps_e"], ls),
                            np.subtract(pp["eps_B"], ls), np.subtract(pp["eps_e_r"], ls), np.subtract(pp["eps_B_r"], ls),
                            np.divide(pp["xi_e_r"], pp["xi_e"]), ls], axis=-1)
            rest = np.stack([w[..., labels.index(l)] for l in ("log10_Gamma0", "log10_theta_c", "p", "log10_tau", "p_r",
                                                               "log10_A_V")], axis=-1)
            t_inv, t_rest = _autocorr(inv), _autocorr(rest)
            say(f"  run{k + 1} steps {50000 + a}-{50000 + b}: tau(log10 s) {t_inv[-1]:.0f}; invariants "
                + ", ".join(f"{n} {v:.0f}" for n, v in zip(("logE_g", "logA_g", "log eps_e_g", "log eps_B_g", "log eps_e_r_g",
                                                            "log eps_B_r_g", "rho"), t_inv[:-1]))
                + "; others " + ", ".join(f"{n} {v:.0f}" for n, v in zip(("Gamma0", "theta_c", "p", "tau", "p_r", "A_V"), t_rest)))

    # ---- numerical or physical breaking? (diagnostic-only model settings in the worker processes) ----
    say("\n--- is the breaking numerical or physical? same scan with diagnostic-only model settings ---")
    for label, res_, rtol_, rad in (("production settings (resolution (0.1, 0.25, 10), rtol 1e-5, radiative)", None, None, True),
                                    ("fine resolution (1, 4, 100), rtol 1e-8, radiative", (1, 4, 100), 1e-8, True),
                                    ("production resolution, radiative_fireball=False", None, None, False),
                                    ("fine resolution, radiative_fireball=False", (1, 4, 100), 1e-8, False)):
        sel = [k for k in range(len(jobs)) if f_all[k] == 1.0 or abs(f_all[k] - 1) > 1e-6]
        with ProcessPoolExecutor(max_workers=args.npool, initializer=_init, initargs=(ctx, res_, rtol_, rad)) as pool:
            rr = np.array(list(pool.map(_evaluate, [jobs[k] for k in sel], chunksize=4)))
        b0 = {idx[k]: rr[n] for n, k in enumerate(sel) if f_all[k] == 1.0}
        dd = np.array([rr[n] - b0[idx[k]] for n, k in enumerate(sel) if f_all[k] != 1.0])
        say(f"  {label:<72} |dlogL| median {np.median(np.abs(dd[:, 4])):.2e}, p95 {np.percentile(np.abs(dd[:, 4]), 95):.2e}, "
            f"max {np.abs(dd[:, 4]).max():.2e}; max |dchi2| XRT {np.abs(dd[:, 0]).max():.2e}, i {np.abs(dd[:, 1]).max():.2e}, "
            f"beta {np.abs(dd[:, 2]).max():.2e}")
        if label.startswith("fine resolution (1"):
            fine_rad = np.array([rr[n][4] for n, k in enumerate(sel) if f_all[k] == 1.0])
        if label.startswith("fine resolution, rad"):
            fine_norad = np.array([rr[n][4] for n, k in enumerate(sel) if f_all[k] == 1.0])
    prod = np.array([base[i][4] for i in range(len(pts))])
    say(f"  log L(fine) - log L(production) at the original points: median {np.median(fine_rad - prod):+.3f}, 16-84% "
        f"[{np.percentile(fine_rad - prod, 16):+.3f}, {np.percentile(fine_rad - prod, 84):+.3f}], max |.| {np.abs(fine_rad - prod).max():.2f}")
    say(f"  log L(radiative_fireball=False) - log L(True), fine resolution: median {np.median(fine_norad - fine_rad):+.3f}, "
        f"16-84% [{np.percentile(fine_norad - fine_rad, 16):+.3f}, {np.percentile(fine_norad - fine_rad, 84):+.3f}], "
        f"max |.| {np.abs(fine_norad - fine_rad).max():.2f}")

    # ---- why are proposals rejected? replay emcee DEMove proposals from stored ensemble states ----
    say("\n--- DEMove proposal replay from stored ensemble states (no sampling: proposals are only evaluated) ---")
    rng_p = np.random.default_rng(7)
    g0 = 2.38 / np.sqrt(2 * len(labels))
    props, olds, old_lp = [], [], []
    for c, lp in chains:
        for t in rng_p.integers(0, len(c), 30):
            ens, ens_lp = c[t], lp[t]
            half = rng_p.permutation(len(ens))
            S, C = half[:len(ens) // 2], half[len(ens) // 2:]
            for w in S:
                i1, i2 = rng_p.choice(C, 2, replace=False)
                props.append(ens[w] + g0 * (1 + 1e-5 * rng_p.normal()) * (ens[i1] - ens[i2]))
                olds.append(ens[w])
                old_lp.append(ens_lp[w])
    props, olds, old_lp = np.array(props), np.array(olds), np.array(old_lp)
    bounds = m._sampled_bounds(defs)
    out_box = (props < bounds[:, 0]) | (props > bounds[:, 1])
    viol_eps = np.array([m.eps_sum_violated(x, defs) for x in props])
    in_prior = ~out_box.any(1) & ~viol_eps
    scale_names = {"log10_E_iso", "log10_A_star", "log10_eps_e", "log10_eps_B", "xi_e", "log10_eps_e_r", "log10_eps_B_r", "xi_e_r"}
    rescuable = []
    for x, ob, ve in zip(props, out_box, viol_eps):
        bad = {labels[j] for j in np.where(ob)[0]}
        if not (ob.any() or ve) or not bad <= scale_names:
            rescuable.append(False)
            continue
        f_lo, f_hi, _, _ = f_interval(x, labels, defs) if np.all(np.isfinite(x)) and x[labels.index("xi_e")] > 0 and x[labels.index("xi_e_r")] > 0 else (1, 0, "", "")
        rescuable.append(f_lo <= f_hi)
    rescuable = np.array(rescuable)
    with ProcessPoolExecutor(max_workers=args.npool, initializer=_init, initargs=(ctx,)) as pool:
        new_lp = np.full(len(props), -np.inf)
        new_lp[in_prior] = np.array(list(pool.map(_logp, list(props[in_prior]), chunksize=4)))
    acc_prob = np.minimum(1, np.exp(np.clip(new_lp - old_lp, -700, 50)))
    say(f"  {len(props)} proposals from 60 stored ensemble states: expected acceptance {acc_prob.mean():.3f} "
        f"(chains measured 0.020); outside the prior box or a constraint: {np.mean(~in_prior):.1%}")
    viol = Counter()
    for ob, ve in zip(out_box, viol_eps):
        for j in np.where(ob)[0]:
            viol[labels[j]] += 1
        if ve:
            viol["eps sum > 1"] += 1
    say("  bound / constraint violated (fraction of all proposals): " + ", ".join(f"{k} {v / len(props):.1%}" for k, v in viol.most_common()))
    say(f"  out-of-prior proposals whose invariants still admit a valid scale s (rescuable by marginalising s): "
        f"{np.mean(rescuable):.1%} of all proposals ({rescuable.sum()} of {np.sum(~in_prior)})")
    d_in = (new_lp - old_lp)[in_prior]
    say(f"  inside the prior: mean acceptance probability {acc_prob[in_prior].mean():.3f}; Delta log P median {np.median(d_in):.1f}, "
        f"fraction < -5: {np.mean(d_in < -5):.1%}, < -20: {np.mean(d_in < -20):.1%}")
    step = np.abs(props - olds)
    for name in ("p", "p_r", "log10_eps_e_r", "xi_e_r", "log10_tau"):
        j = labels.index(name)
        big = step[in_prior, j] > np.median(step[in_prior, j])
        say(f"  inside the prior, proposals with an above-median {name} step: acceptance {acc_prob[in_prior][big].mean():.3f} "
            f"vs {acc_prob[in_prior][~big].mean():.3f} below the median")

    # ---- p, p_r: conditional width vs linear predictability (curvature) ----
    say("\n--- p and p_r: conditional (1-D) width at fixed other parameters vs width left after linear regression ---")
    with ProcessPoolExecutor(max_workers=args.npool, initializer=_init, initargs=(ctx,)) as pool:
        sample_pts = pts[rng.choice(200, 16, replace=False)]
        hs = np.geomspace(1e-7, 3e-2, 40)
        cw = {}
        for name in ("p", "p_r", "log10_eps_e_r", "log10_tau"):
            j = labels.index(name)
            jobs2 = []
            for x in sample_pts:
                for h in hs:
                    for sgn in (-1, 1):
                        y = x.copy()
                        y[j] += sgn * h
                        jobs2.append(y)
            r2 = np.array(list(pool.map(_evaluate, jobs2, chunksize=8)))[:, 4]
            base_ll = np.array(list(pool.map(_evaluate, list(sample_pts))))[:, 4]
            r2 = r2.reshape(len(sample_pts), len(hs), 2)
            widths_ = []
            for q in range(len(sample_pts)):
                drop = base_ll[q] - r2[q].max(axis=1)          # the smaller drop of the two sides at each h
                ok = np.where(drop < 0.5)[0]
                widths_.append(hs[ok.max()] if len(ok) else hs[0])
            cw[name] = np.median(widths_)
    flat = np.concatenate([c[::10].reshape(-1, len(labels)) for c, _ in chains])
    for name in ("p", "p_r", "log10_eps_e_r", "log10_tau"):
        j = labels.index(name)
        others = np.delete(flat, j, axis=1)
        X = np.column_stack([others, np.ones(len(others))])
        res_lin = flat[:, j] - X @ np.linalg.lstsq(X, flat[:, j], rcond=None)[0]
        say(f"  {name:<14} marginal sigma {flat[:, j].std():.3g}; linear-regression residual sigma {res_lin.std():.3g}; "
            f"conditional half-width (dlogL = 0.5) {cw[name]:.3g}; residual / conditional = {res_lin.std() / cw[name]:.0f}")
    for name, off in (("p", 2.0), ("p_r", 2.0)):
        j = labels.index(name)
        v = np.log10(flat[:, j] - off)
        others = np.delete(flat, j, axis=1)
        X = np.column_stack([others, np.ones(len(others))])
        coef = np.linalg.lstsq(X, v, rcond=None)[0]
        r = np.subtract(v, X @ coef)
        Xs = np.column_stack([np.divide(np.subtract(others, others.mean(0)), others.std(0)), np.ones(len(others))])
        r_check = np.subtract(v, Xs @ np.linalg.lstsq(Xs, v, rcond=None)[0])     # same fit, standardised regressors
        say(f"  (check: R^2 with standardised regressors {1 - r_check.var() / v.var():.3f}; max |coef| {np.abs(coef).max():.3g})")
        say(f"  log10({name} - 2): marginal sigma {v.std():.3g}, linear-regression residual sigma {r.std():.3g} "
            f"(R^2 {1 - r.var() / v.var():.3f}; for {name} itself R^2 "
            f"{1 - (flat[:, j] - X @ np.linalg.lstsq(X, flat[:, j], rcond=None)[0]).var() / flat[:, j].var():.3f})")

    (outdir / "summary.txt").write_text("\n".join(lines) + "\n")
    np.savez(outdir / "scaling_scan.npz", points=pts, tags=np.array(tags), f=f_all, point_index=idx, results=res,
             delta=delta, components=np.array(list(COMPONENTS) + ["log_prior"]), pit=pit)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].scatter(np.log10(f_all[is_scan]), np.abs(dll) + 1e-12, s=4, alpha=0.5)
    axes[0].axhline(np.abs(delta[is_noise, 4]).max(), color="r", ls="--", label="noise probe max")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("log10 f")
    axes[0].set_ylabel("|Delta log L|")
    axes[0].legend()
    for j, name in enumerate(COMPONENTS[:3]):
        axes[1].scatter(np.log10(f_all[is_scan]), np.abs(delta[is_scan, j]) + 1e-12, s=4, alpha=0.5, label=name)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("log10 f")
    axes[1].set_ylabel("|Delta chi2| per component")
    axes[1].legend(fontsize=7)
    axes[2].hist(pit, bins=np.linspace(0, 1, 41), density=True, histtype="step", color="k")
    axes[2].axhline(1, color="r", ls="--")
    axes[2].set_xlabel("u = (s^2 - s_lo^2)/(s_hi^2 - s_lo^2)  [uniform if p(s|invariants) ~ s]")
    fig.tight_layout()
    fig.savefig(outdir / "scaling_symmetry.png", dpi=120)
    plt.close(fig)
    say(f"\nWritten to {outdir}: summary.txt, scaling_scan.npz, scaling_symmetry.png")
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
