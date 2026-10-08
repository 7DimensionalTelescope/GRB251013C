#!/usr/bin/env python3
"""Compare independent production MCMC runs of the early-phase fit (reproducibility check).

    python compare_early_phase_runs.py RUN1_DIR RUN2_DIR [--discard 50000] [--outdir DIR]

Reads each run's chain.h5 directly (not the thinned samples.npy) with the same discard, and compares
16/50/84% intervals, best log_prob, correlations, boundary pile-ups, acceptance, autocorrelation
estimates and window-to-window stability, with overlay plots of the key marginals and the
E_iso-xi_e_r ridge. Walkers of one emcee ensemble interact through the moves, so no walker-based
R-hat is computed: the runs themselves are the independent replicates. Differences are expressed
in units of the posterior width and, heuristically, of a Monte-Carlo error sigma * sqrt(tau_ac / (N * W))
that inherits the (unstable) autocorrelation estimates.
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import emcee
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

KEY = ("tau", "E_iso", "xi_e", "xi_e_r", "p_r", "eps_e_r")
ORDER = ("tau", "E_iso", "xi_e", "xi_e_r", "p", "p_r", "eps_e", "eps_B", "eps_e_r", "eps_B_r", "Gamma0", "theta_c",
         "A_star", "A_V")
LOG_AXIS = {"tau", "E_iso", "eps_e", "eps_B", "eps_e_r", "eps_B_r", "Gamma0", "theta_c", "A_star", "A_V"}
PAIRS = [("log10_E_iso", "xi_e_r"), ("log10_E_iso", "xi_e"), ("p_r", "log10_eps_e_r"), ("log10_tau", "log10_Gamma0"),
         ("xi_e", "xi_e_r"), ("log10_E_iso", "log10_eps_e_r"), ("xi_e_r", "log10_eps_e_r"), ("log10_tau", "log10_eps_e_r")]
# physically / prior-motivated edge checks (physical units) on the post-discard samples
EDGES = [("E_iso > 9e54 (cap 1e55)", "E_iso", lambda v: v > 9e54), ("eps_e < 0.0022 (min 0.002)", "eps_e", lambda v: v < 0.0022),
         ("eps_B > 0.045 (max 0.05)", "eps_B", lambda v: v > 0.045), ("A_star > 9 (max 10)", "A_star", lambda v: v > 9),
         ("xi_e > 0.95 (max 1)", "xi_e", lambda v: v > 0.95), ("xi_e_r < 0.02 (min 0.01)", "xi_e_r", lambda v: v < 0.02),
         ("p < 2.0115 (min 2.01)", "p", lambda v: v < 2.0115), ("p_r < 2.003 (min 2.0)", "p_r", lambda v: v < 2.003)]


def load(run_dir, discard):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "run_config.json").read_text())
    reader = emcee.backends.HDFBackend(str(run_dir / "chain.h5"), name="mcmc", read_only=True)
    chain, lp = reader.get_chain(), reader.get_log_prob()
    labels = config["labels"]
    bounds = np.array([[np.log10(b["lower"]), np.log10(b["upper"])] if b["scale"] == "log10" else [b["lower"], b["upper"]]
                       for b in config["bounds"]])
    return {"dir": run_dir, "config": config, "chain": chain, "lp": lp, "labels": labels, "bounds": bounds,
            "acc": reader.accepted / reader.iteration, "n": reader.iteration, "post": chain[discard:], "post_lp": lp[discard:]}


def phys(x, label):
    return 10 ** x if label.startswith("log10_") else x


def col(run, name, part="post"):
    label = f"log10_{name}" if f"log10_{name}" in run["labels"] else name
    return phys(run[part][..., run["labels"].index(label)], label)


def hpd_levels(h, fractions=(0.68, 0.95)):
    flat = np.sort(h.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    return sorted(flat[np.searchsorted(cum, f)] for f in fractions)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs=2)
    ap.add_argument("--discard", type=int, default=50000)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    runs = [load(r, args.discard) for r in args.runs]
    names = ["run1", "run2"]
    assert runs[0]["labels"] == runs[1]["labels"], "parameter layouts differ"
    assert runs[0]["config"]["param_kwargs"] == runs[1]["config"]["param_kwargs"], "priors differ"
    labels = runs[0]["labels"]
    outdir = Path(args.outdir) if args.outdir else runs[0]["dir"].parent / f"early_phase_production_comparison_{datetime.now():%Y%m%d_%H%M%S}"
    outdir.mkdir(parents=True, exist_ok=True)
    lines, out = [], {"runs": [str(r["dir"]) for r in runs], "discard": args.discard}

    def say(t=""):
        lines.append(t)
        print(t, flush=True)

    say(f"=== Early-phase production MCMC: run 1 vs run 2 ({datetime.now():%Y-%m-%d %H:%M}) ===")
    for nm, r in zip(names, runs):
        c = r["config"]
        say(f"{nm}: {r['dir']}  seed {c['seed']}, {r['n']} steps, {c['nwalkers']} walkers, {c['npool']} workers, "
            f"{c['total_wall_seconds'] / 3600:.2f} h sampling")
    init_shared = int((np.abs(np.load(runs[0]["dir"] / "initial_walkers.npy")[:, None] -
                              np.load(runs[1]["dir"] / "initial_walkers.npy")[None]).max(-1) < 1e-12).sum())
    say(f"identical model/prior configuration: True; initial walker coordinates shared between runs: {init_shared}")
    say(f"comparison window: steps {args.discard}-{runs[0]['n']} for both runs ({runs[0]['n'] - args.discard} steps x 64 walkers)")

    # autocorrelation on the comparison window (needed for the heuristic Monte-Carlo errors)
    taus = [emcee.autocorr.integrated_time(r["post"], c=5, tol=0, quiet=True) for r in runs]
    nw = [r["post"].shape[0] * r["post"].shape[1] for r in runs]

    say("\n--- 16 / 50 / 84 % (physical units), comparison window ---")
    say(f"{'param':<8} {'run1 median [16, 84]':>34} {'run2 median [16, 84]':>34} {'dMed/sig':>9} {'z_MC':>6} {'width2/1':>9} {'KS D':>6}")
    table = {}
    for name in ORDER:
        label = f"log10_{name}" if f"log10_{name}" in labels else name
        j = labels.index(label)
        q = [np.percentile(col(r, name), [16, 50, 84]) for r in runs]
        s = [r["post"][..., j].ravel() for r in runs]                            # sampled space for the distances
        sig = 0.5 * (np.diff(np.percentile(s[0], [16, 84]))[0] + np.diff(np.percentile(s[1], [16, 84]))[0]) / 2
        dmed = (np.median(s[1]) - np.median(s[0])) / sig
        mcse = [1.2533 * np.std(x) * np.sqrt(t[j] / n) for x, t, n in zip(s, taus, nw)]
        z = (np.median(s[1]) - np.median(s[0])) / np.hypot(*mcse)
        w = np.diff(np.percentile(s[1], [16, 84]))[0] / np.diff(np.percentile(s[0], [16, 84]))[0]
        rng = np.random.default_rng(0)
        a, b = (np.sort(rng.choice(x, 200000, replace=False)) for x in s)
        grid = np.concatenate([a, b])
        ks = np.max(np.abs(np.searchsorted(a, grid, "right") / len(a) - np.searchsorted(b, grid, "right") / len(b)))
        fmt = (lambda v: f"{v:.4g}")
        say(f"{name:<8} {fmt(q[0][1]):>11} [{fmt(q[0][0])}, {fmt(q[0][2])}]".ljust(44)
            + f"{fmt(q[1][1]):>11} [{fmt(q[1][0])}, {fmt(q[1][2])}]".ljust(35)
            + f"{dmed:>+9.2f} {z:>+6.1f} {w:>9.2f} {ks:>6.3f}")
        table[name] = {"run1": q[0].tolist(), "run2": q[1].tolist(), "dmedian_over_sigma": dmed, "z_mc": z,
                       "width_ratio": w, "ks_distance": ks}
    out["intervals"] = table
    say("(dMed/sig: median difference / posterior 1-sigma half-width, sampled space; z_MC: same over the heuristic\n"
        " Monte-Carlo error 1.25 sigma sqrt(tau_ac/(N W)); KS D: max CDF difference, descriptive only - samples are correlated)")

    say("\n--- best log_prob ---")
    for nm, r in zip(names, runs):
        k = np.unravel_index(np.argmax(r["lp"]), r["lp"].shape)
        say(f"{nm}: overall {r['lp'].max():.4f} (step {k[0]}), comparison window {r['post_lp'].max():.4f}; "
            f"window log_prob median {np.median(r['post_lp']):.3f} [16-84: {np.percentile(r['post_lp'], 16):.3f}, "
            f"{np.percentile(r['post_lp'], 84):.3f}]")
        out.setdefault("best_log_prob", {})[nm] = float(r["lp"].max())

    say("\n--- correlations (sampled space, comparison window) ---")
    for a, b in PAIRS:
        cs = [np.corrcoef(r["post"][..., labels.index(a)].ravel(), r["post"][..., labels.index(b)].ravel())[0, 1] for r in runs]
        say(f"{a + ' - ' + b:<32} run1 {cs[0]:+.3f}   run2 {cs[1]:+.3f}")
        out.setdefault("correlations", {})[f"{a}-{b}"] = cs
    ridge = [np.log10(col(r, "E_iso") * col(r, "xi_e_r")) for r in runs]
    say("log10(E_iso * xi_e_r): " + "   ".join(f"{nm} median {np.median(x):.3f}, 16-84 width {np.diff(np.percentile(x, [16, 84]))[0]:.3f} dex"
                                             for nm, x in zip(names, ridge)))

    say("\n--- boundary pile-ups (fraction of comparison-window samples) ---")
    for desc, name, test in EDGES:
        fr = [float(test(col(r, name)).mean()) for r in runs]
        say(f"{desc:<30} run1 {100 * fr[0]:5.1f}%   run2 {100 * fr[1]:5.1f}%")
        out.setdefault("edges", {})[desc] = fr
    fr = [float(((col(r, "eps_e_r") + col(r, "eps_B_r")) > 0.99).mean()) for r in runs]
    say(f"{'eps_e_r + eps_B_r > 0.99':<30} run1 {100 * fr[0]:5.1f}%   run2 {100 * fr[1]:5.1f}%")
    for nm, r in zip(names, runs):
        span = r["bounds"][:, 1] - r["bounds"][:, 0]
        flat = r["post"].reshape(-1, len(labels))
        edge = np.minimum(0.01 * span, 0.5 * flat.std(0))
        lo, hi = (flat - r["bounds"][:, 0] < edge).mean(0), (r["bounds"][:, 1] - flat < edge).mean(0)
        flagged = [f"{l} {s} {100 * v:.1f}%" for l, a_, b_ in zip(labels, lo, hi) for s, v in (("lower", a_), ("upper", b_)) if v > 0.05]
        say(f"{nm} edge-zone (min(1% range, 0.5 sigma)) pile-ups > 5%: {', '.join(flagged) or 'none'}")

    say("\n--- acceptance ---")
    for nm, r in zip(names, runs):
        a = r["acc"]
        moved = np.mean(np.any(r["post"][1:] != r["post"][:-1], axis=2), axis=0)
        say(f"{nm}: mean {a.mean():.4f}, median {np.median(a):.4f}, min {a.min():.4f} (walker {a.argmin()}), max {a.max():.4f}; "
            f"window move fraction min {moved.min():.4f}; walkers < 0.01 in window: {np.where(moved < 0.01)[0].tolist() or 'none'}")
        out.setdefault("acceptance", {})[nm] = {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}

    say("\n--- autocorrelation time on the comparison window (steps; estimates are NOT converged, N/tau ~ 3) ---")
    say(f"{'label':<15} {'run1':>8} {'run2':>8}")
    for j, l in enumerate(labels):
        say(f"{l:<15} {taus[0][j]:>8.0f} {taus[1][j]:>8.0f}")
    say(f"{'max / median':<15} {taus[0].max():>8.0f} {taus[1].max():>8.0f}  /  {np.median(taus[0]):.0f} {np.median(taus[1]):.0f}")
    out["tau_ac"] = {nm: dict(zip(labels, t.tolist())) for nm, t in zip(names, taus)}

    say("\n--- window-to-window stability (median [16, 84]) ---")
    windows = [(25000, 50000), (50000, 75000), (75000, 100000)]
    stab = {}
    for name in ORDER:
        row = []
        for r in runs:
            for a, b in windows:
                v = col(r, name, "chain")[a:b].ravel()
                row.append(np.percentile(v, [16, 50, 84]))
        stab[name] = [x.tolist() for x in row]
        meds = np.array([x[1] for x in row])
        say(f"{name:<8} " + "  ".join(f"{m:.4g}" for m in meds[:3]) + "  |  " + "  ".join(f"{m:.4g}" for m in meds[3:])
            + f"   (run1 25-50/50-75/75-100k | run2 ...; spread of the 6 medians / mean 1-sigma = "
            f"{(np.ptp(np.log10(meds)) if name in LOG_AXIS else np.ptp(meds)) / np.mean([((np.log10(x[2]) - np.log10(x[0])) if name in LOG_AXIS else (x[2] - x[0])) / 2 for x in row]):.2f})")
    out["windows"] = stab

    say("\n--- alternative-mode checks (whole chains after 5k steps) ---")
    for nm, r in zip(names, runs):
        tau_all = col(r, "tau", "chain")[5000:]
        walker_tau = np.median(col(r, "tau"), axis=0)
        walker_lp = np.median(r["post_lp"], axis=0)
        say(f"{nm}: tau range {tau_all.min():.1f}-{tau_all.max():.1f} s; fraction tau < 300 s {np.mean(tau_all < 300):.2e}, "
            f"> 600 s {np.mean(tau_all > 600):.2e}; per-walker median tau {walker_tau.min():.1f}-{walker_tau.max():.1f} s; "
            f"per-walker median log_prob {walker_lp.min():.2f} to {walker_lp.max():.2f}")
    (outdir / "comparison_summary.txt").write_text("\n".join(lines) + "\n")
    (outdir / "comparison.json").write_text(json.dumps(out, indent=2, default=float) + "\n")

    # ---- plots ----
    colors = ("C0", "C3")
    fig, axes = plt.subplots(4, 4, figsize=(16, 13))
    for ax, name in zip(axes.flat, ORDER):
        vals = [col(r, name).ravel() for r in runs]
        if name in LOG_AXIS:
            vals = [np.log10(v) for v in vals]
        lo, hi = np.percentile(np.concatenate(vals), [0.05, 99.95])
        bins = np.linspace(lo, hi, 70)
        for v, c, nm in zip(vals, colors, names):
            ax.hist(v, bins=bins, density=True, histtype="step", color=c, lw=1.4, label=nm)
            for qv in np.percentile(v, [16, 50, 84]):
                ax.axvline(qv, color=c, ls=":" if qv != np.median(v) else "--", lw=0.8)
        ax.set_xlabel(f"log10 {name}" if name in LOG_AXIS else name)
        ax.set_yticks([])
    axes.flat[0].legend()
    for ax in list(axes.flat)[len(ORDER):]:
        ax.axis("off")
    fig.suptitle(f"Marginals, steps {args.discard}-{runs[0]['n']} (dashed: median, dotted: 16/84%)")
    fig.tight_layout()
    fig.savefig(outdir / "overlay_all_marginals.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, name in zip(axes.flat, KEY):
        vals = [col(r, name).ravel() for r in runs]
        if name in LOG_AXIS:
            vals = [np.log10(v) for v in vals]
        lo, hi = np.percentile(np.concatenate(vals), [0.05, 99.95])
        bins = np.linspace(lo, hi, 70)
        for v, c, nm in zip(vals, colors, names):
            ax.hist(v, bins=bins, density=True, histtype="step", color=c, lw=1.6, label=nm)
        ax.set_xlabel(f"log10 {name}" if name in LOG_AXIS else name)
        ax.set_yticks([])
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(outdir / "overlay_key_marginals.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    xs = [np.log10(col(r, "E_iso")).ravel() for r in runs]
    ys = [col(r, "xi_e_r").ravel() for r in runs]
    xb = np.linspace(53.3, 55.0, 60)
    yb = np.linspace(0.0, 0.3, 60)
    for k, (x, y, c, nm) in enumerate(zip(xs, ys, colors, names)):
        pick = np.random.default_rng(k).choice(len(x), 15000, replace=False)
        axes[k].plot(x[pick], y[pick], ",", color=c, alpha=0.5)
        axes[k].set_title(f"{nm}: E_iso - xi_e_r ridge")
        h = gaussian_filter(np.histogram2d(x, y, bins=[xb, yb])[0], 1.2)   # light smoothing for readable contours
        axes[2].contour(0.5 * (xb[1:] + xb[:-1]), 0.5 * (yb[1:] + yb[:-1]), h.T, levels=hpd_levels(h), colors=c,
                        linewidths=[1.6, 1.0])
    axes[2].plot([], [], color=colors[0], label="run1 68/95%")
    axes[2].plot([], [], color=colors[1], label="run2 68/95%")
    e = np.linspace(53.3, 55.0, 100)
    for prod in (52.8, 53.0, 53.2):
        axes[2].plot(e, 10 ** (prod - e), "k:", lw=0.7)
    axes[2].text(54.85, 10 ** (53.0 - 54.85) + 0.01, "E_iso xi_e_r = 1e53", fontsize=7)
    axes[2].legend(fontsize=8)
    axes[2].set_title("overlay (dotted: E_iso xi_e_r = const)")
    for ax in axes:
        ax.set_xlabel("log10 E_iso")
        ax.set_ylabel("xi_e_r")
        ax.set_xlim(53.3, 55.0)
        ax.set_ylim(0, 0.3)
    fig.tight_layout()
    fig.savefig(outdir / "ridge_E_iso_xi_e_r.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for ax, name in zip(axes.flat, ORDER):
        for k, (c, nm) in enumerate(zip(colors, names)):
            for w, (a, b) in enumerate(windows):
                lo_, med, hi_ = stab[name][3 * k + w]
                xpos = w + 0.15 * (k - 0.5)
                ax.errorbar(xpos, med, yerr=[[med - lo_], [hi_ - med]], fmt="o", color=c, label=nm if w == 0 else None)
        ax.set_xticks(range(3), ["25-50k", "50-75k", "75-100k"], fontsize=7)
        ax.set_title(name, fontsize=9)
        if name in LOG_AXIS:
            ax.set_yscale("log")
    axes.flat[0].legend(fontsize=7)
    for ax in list(axes.flat)[len(ORDER):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(outdir / "window_stability.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for r, c, nm in zip(runs, colors, names):
        ax.hist(r["post_lp"].ravel(), bins=np.linspace(-100, -85, 80), density=True, histtype="step", color=c, label=nm)
    ax.set_xlabel("log_probability (comparison window)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "overlay_log_prob.png", dpi=110)
    plt.close(fig)
    say(f"\nWritten to {outdir}: comparison_summary.txt, comparison.json, overlay_key_marginals.png, "
        f"overlay_all_marginals.png, ridge_E_iso_xi_e_r.png, window_stability.png, overlay_log_prob.png")
    (outdir / "comparison_summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
