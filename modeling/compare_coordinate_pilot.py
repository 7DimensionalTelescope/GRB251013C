#!/usr/bin/env python3
"""Sampling-coordinate pilot versus the two physical-coordinate production runs on equal-length windows.

    python compare_coordinate_pilot.py PILOT_DIR RUN1_DIR RUN2_DIR [--pilot-discard 5000] [--outdir DIR]

Windows: pilot steps 5k-30k (started from posterior draws) against run 1 / run 2 steps 50k-75k and 75k-100k.
Mixing: acceptance (fraction of steps a walker moved), autocorrelation times (early_phase_mcmc._autocorr) of every
physical coordinate and of u_r = log10(p_r - 2) and q_E_xir for all runs, autocorrelation of principal components in
ONE common basis (standardised pooled run-1 + run-2 posterior, in physical and in u_r coordinates), the (u_r,
log10 eps_e_r) principal axes, walker mixing (rank persistence of per-walker means between window halves), traces and
window stability. Posterior: physical marginals of the pilot against both runs. log P is the physical log posterior
(the pilot's stored target minus its coordinate log-Jacobian).
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import emcee
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from early_phase_mcmc import _autocorr
from early_phase_model import from_sampling_coords, sampling_labels, sampling_log_jacobian

PHYS = ("tau", "E_iso", "xi_e", "xi_e_r", "Gamma0", "theta_c", "A_star", "p", "p_r", "eps_e", "eps_B", "eps_e_r",
        "eps_B_r", "A_V")
LOG_AXIS = {"tau", "E_iso", "Gamma0", "theta_c", "A_star", "eps_e", "eps_B", "eps_e_r", "eps_B_r", "A_V"}


def load(run_dir):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "run_config.json").read_text())
    reader = emcee.backends.HDFBackend(str(run_dir / "chain.h5"), name="mcmc", read_only=True)
    coords = config.get("sampling_coords", "physical")
    raw = reader.get_chain()
    labels = config["labels"]
    phys = from_sampling_coords(raw, labels, coords)
    lp = reader.get_log_prob()
    if coords != "physical":
        lp = np.subtract(lp, sampling_log_jacobian(raw, labels, coords))
    pr = labels.index("p_r")
    e_i, xi_i = labels.index("log10_E_iso"), labels.index("xi_e_r")
    extra = np.stack([np.log10(np.subtract(phys[..., pr], 2.0)),                       # u_r
                      np.add(phys[..., e_i], np.log10(phys[..., xi_i]))], axis=-1)       # q_E_xir
    return {"dir": run_dir, "config": config, "coords": coords, "labels": labels, "raw": raw, "phys": phys,
            "lp": lp, "extra": extra, "n": reader.iteration}


def ur_coords(run, a, b):
    """Window in u_r coordinates (p_r replaced by log10(p_r - 2)), physical otherwise."""
    x = np.array(run["phys"][a:b])
    x[..., run["labels"].index("p_r")] = run["extra"][a:b, :, 0]
    return x


def column(run, name, a, b):
    L = run["labels"]
    if name == "u_r":
        return run["extra"][a:b, :, 0]
    if name == "q_E_xir":
        return run["extra"][a:b, :, 1]
    label = f"log10_{name}" if f"log10_{name}" in L else name
    v = run["phys"][a:b, :, L.index(label)]
    return np.power(10.0, v) if label.startswith("log10_") else v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pilot")
    ap.add_argument("run1")
    ap.add_argument("run2")
    ap.add_argument("--pilot-discard", type=int, default=5000)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    pilot, r1, r2 = load(args.pilot), load(args.run1), load(args.run2)
    labels = pilot["labels"]
    assert r1["labels"] == labels == r2["labels"]
    assert pilot["config"]["param_kwargs"] == r1["config"]["param_kwargs"] == r2["config"]["param_kwargs"], "priors differ"
    outdir = Path(args.outdir) if args.outdir else pilot["dir"] / "comparison_vs_run1_run2"
    outdir.mkdir(parents=True, exist_ok=True)
    lines, out = [], {}

    def say(t=""):
        lines.append(t)
        print(t, flush=True)

    wp = (args.pilot_discard, pilot["n"])
    windows = [("pilot", pilot, *wp), ("run1 50-75k", r1, 50000, 75000), ("run1 75-100k", r1, 75000, 100000),
               ("run2 50-75k", r2, 50000, 75000), ("run2 75-100k", r2, 75000, 100000)]
    names = [w[0] for w in windows]
    say(f"=== Sampling-coordinate pilot vs physical-coordinate runs ({pilot['dir'].name}) ===")
    say(f"pilot: sampling coordinates {pilot['coords']} ({sampling_labels(labels, pilot['coords'])[labels.index('p_r')]} in the "
        f"p_r slot), seed {pilot['config']['seed']}, {pilot['n']} steps, {pilot['config']['total_wall_seconds'] / 3600:.2f} h")
    say(f"windows of {wp[1] - wp[0]} steps: " + ", ".join(names))

    say("\n--- acceptance (fraction of steps a walker moved, per window) ---")
    acc = {}
    for name, run, a, b in windows:
        x = run["raw"][a:b]
        moved = np.mean(np.any(x[1:] != x[:-1], axis=2), axis=0)
        acc[name] = moved
        say(f"{name:<14} mean {moved.mean():.4f}  median {np.median(moved):.4f}  min {moved.min():.4f}  max {moved.max():.4f}"
            f"  walkers < 0.01: {int((moved < 0.01).sum())}")

    say("\n--- integrated autocorrelation time [steps], same estimator and window length for every run ---")
    taus = {}
    for name, run, a, b in windows:
        t_phys = _autocorr(run["phys"][a:b])
        t_extra = _autocorr(run["extra"][a:b])
        taus[name] = (t_phys, t_extra)
    say(f"{'coordinate':<16}" + "".join(f"{n:>14}" for n in names))
    for j, l in enumerate(labels):
        say(f"{l:<16}" + "".join(f"{taus[n][0][j]:>14.0f}" for n in names))
    for k, l in enumerate(("u_r", "q_E_xir")):
        say(f"{l + ' (derived)':<16}" + "".join(f"{taus[n][1][k]:>14.0f}" for n in names))
    say(f"{'max (physical)':<16}" + "".join(f"{np.nanmax(taus[n][0]):>14.0f}" for n in names))
    say(f"{'median (phys.)':<16}" + "".join(f"{np.nanmedian(taus[n][0]):>14.0f}" for n in names))
    say(f"{'ESS min':<16}" + "".join(f"{(b - a) * 64 / np.nanmax(taus[n][0]):>14.0f}" for n, run, a, b in windows))
    say(f"{'ESS median':<16}" + "".join(f"{(b - a) * 64 / np.nanmedian(taus[n][0]):>14.0f}" for n, run, a, b in windows))
    out["tau"] = {n: {"physical": dict(zip(labels, t[0].tolist())), "u_r": float(t[1][0]), "q_E_xir": float(t[1][1])}
                  for n, t in taus.items()}

    say("\n--- common principal-component basis (pooled run 1 + run 2, steps 50k-100k, standardised) ---")
    for system, getter in (("physical coordinates", lambda run, a, b: run["phys"][a:b]), ("u_r coordinates", ur_coords)):
        pooled = np.concatenate([getter(r, 50000, 100000)[::5].reshape(-1, len(labels)) for r in (r1, r2)])
        mu, sd = pooled.mean(0), pooled.std(0)
        evals, evecs = np.linalg.eigh(np.cov(np.divide(np.subtract(pooled, mu), sd).T))
        pc_tau = {}
        for name, run, a, b in windows:
            z = np.divide(np.subtract(getter(run, a, b), mu), sd)
            pc_tau[name] = _autocorr(np.einsum("twd,dk->twk", z, evecs))
        old = np.mean([pc_tau[n] for n in names[1:]], axis=0)
        order = np.argsort(old)[::-1]
        say(f"{system}: PCs ordered by the old runs' mean tau (rows), tau per window (columns)")
        say(f"{'PC (variance)':<16}" + "".join(f"{n:>14}" for n in names) + "   main loadings")
        for k in order[:5]:
            top = np.argsort(np.abs(evecs[:, k]))[::-1][:4]
            lab = sampling_labels(labels, "u_r") if system.startswith("u_r") else labels
            say(f"PC{k:<2d} ({evals[k]:6.3f})  " + "".join(f"{pc_tau[n][k]:>14.0f}" for n in names) + "   "
                + ", ".join(f"{lab[i]} {evecs[i, k]:+.2f}" for i in top))
        say(f"{'slowest PC tau':<16}" + "".join(f"{pc_tau[n].max():>14.0f}" for n in names))
        out[f"pc_tau_{system}"] = {n: v.tolist() for n, v in pc_tau.items()}

    say("\n--- (u_r, log10 eps_e_r) plane: principal axes of the pooled posterior and their autocorrelation ---")
    er = labels.index("log10_eps_e_r")
    pooled2 = np.concatenate([np.stack([r["extra"][50000:100000:5, :, 0], r["phys"][50000:100000:5, :, er]], -1).reshape(-1, 2)
                              for r in (r1, r2)])
    mu2, sd2 = pooled2.mean(0), pooled2.std(0)
    c2 = np.corrcoef(pooled2.T)[0, 1]
    ev2, evec2 = np.linalg.eigh(np.cov(np.divide(np.subtract(pooled2, mu2), sd2).T))
    say(f"corr(u_r, log10 eps_e_r) = {c2:+.3f}; standardised axes: narrow {evec2[:, 0].round(3).tolist()} (variance "
        f"{ev2[0]:.3f}), long {evec2[:, 1].round(3).tolist()} (variance {ev2[1]:.3f})")
    for name, run, a, b in windows:
        pair = np.stack([run["extra"][a:b, :, 0], run["phys"][a:b, :, er]], -1)
        z = np.einsum("twd,dk->twk", np.divide(np.subtract(pair, mu2), sd2), evec2)
        t = _autocorr(z)
        say(f"  {name:<14} tau(narrow axis) {t[0]:6.0f}   tau(long axis) {t[1]:6.0f}")

    say("\n--- walker mixing: Spearman correlation of per-walker means between the two halves of the window ---")
    say("(near 0 = walkers move through the whole ensemble within half a window; near 1 = walkers keep their place)")
    for name, run, a, b in windows:
        h = (a + b) // 2
        vals = []
        for getter in (lambda s, e: run["extra"][s:e, :, 0], lambda s, e: run["phys"][s:e, :, er], lambda s, e: run["lp"][s:e]):
            vals.append(spearmanr(getter(a, h).mean(0), getter(h, b).mean(0))[0])
        say(f"  {name:<14} u_r {vals[0]:+.2f}   log10 eps_e_r {vals[1]:+.2f}   log P {vals[2]:+.2f}")

    say("\n--- window stability: medians in the two halves of each window (shift in units of the half-width) ---")
    for pname in ("p_r", "eps_e_r", "xi_e_r", "E_iso", "tau", "q_E_xir"):
        row = []
        for name, run, a, b in windows:
            h = (a + b) // 2
            v1, v2 = column(run, pname, a, h).ravel(), column(run, pname, h, b).ravel()
            tr = np.log10 if pname in LOG_AXIS else (lambda v: v)
            sig = 0.5 * np.diff(np.percentile(tr(column(run, pname, a, b).ravel()), [16, 84]))[0]
            row.append((np.median(tr(v2)) - np.median(tr(v1))) / sig)
        say(f"  {pname:<8} " + "  ".join(f"{n} {v:+.2f}" for n, v in zip(names, row)))

    say("\n--- physical marginals: pilot vs run1 / run2 (50k-100k), median [16, 84] ---")
    say(f"{'param':<8} {'pilot':>32} {'run1':>32} {'run2':>32} {'dMed/sig vs pooled':>20} {'KS D':>7}")
    rng = np.random.default_rng(0)
    stats = {}
    for pname in ("q_E_xir",) + PHYS:
        p = column(pilot, pname, *wp).ravel()
        o1, o2 = column(r1, pname, 50000, 100000).ravel(), column(r2, pname, 50000, 100000).ravel()
        tr = np.log10 if pname in LOG_AXIS else (lambda v: v)
        pooled = np.concatenate([o1, o2])
        sig = 0.5 * np.diff(np.percentile(tr(pooled), [16, 84]))[0]
        dmed = (np.median(tr(p)) - np.median(tr(pooled))) / sig
        a_ = np.sort(rng.choice(tr(p), 200000, replace=False))
        b_ = np.sort(rng.choice(tr(pooled), 200000, replace=False))
        g = np.concatenate([a_, b_])
        ks = np.max(np.abs(np.searchsorted(a_, g, "right") / len(a_) - np.searchsorted(b_, g, "right") / len(b_)))
        f = lambda v: "{:.4g} [{:.4g}, {:.4g}]".format(*np.percentile(v, [50, 16, 84]))
        say(f"{pname:<8} {f(p):>32} {f(o1):>32} {f(o2):>32} {dmed:>+20.2f} {ks:>7.3f}")
        stats[pname] = {"dmed_sigma": float(dmed), "ks": float(ks)}
    out["marginals"] = stats
    say("(reference: run 1 vs run 2 differed by up to 0.26 sigma, KS D up to 0.11; the q pilot by up to 0.20 sigma)")
    best = [float(np.max(run["lp"])) for run in (pilot, r1, r2)]
    say(f"best physical log P: pilot {best[0]:.4f}, run1 {best[1]:.4f}, run2 {best[2]:.4f}")
    (outdir / "comparison_summary.txt").write_text("\n".join(lines) + "\n")
    (outdir / "comparison.json").write_text(json.dumps(out, indent=1) + "\n")

    # ---- plots ----
    colors = {"pilot": "k", "run1": "C0", "run2": "C3"}
    fig, axes = plt.subplots(3, 3, figsize=(17, 10), sharey="row")
    stride = 10
    for col, (title, run, a, b) in enumerate((("pilot u_r 5-30k", pilot, *wp), ("run1 75-100k", r1, 75000, 100000),
                                               ("run2 75-100k", r2, 75000, 100000))):
        steps = np.arange(a, b)[::stride]
        axes[0, col].plot(steps, run["phys"][a:b:stride, :, labels.index("p_r")], lw=0.3, alpha=0.4)
        axes[1, col].plot(steps, run["phys"][a:b:stride, :, er], lw=0.3, alpha=0.4)
        axes[2, col].plot(steps, run["lp"][a:b:stride], lw=0.3, alpha=0.4)
        axes[2, col].plot(steps, np.median(run["lp"][a:b:stride], axis=1), "k", lw=1)
        axes[0, col].set_title(title)
        axes[2, col].set_xlabel("step")
    axes[0, 0].set_ylabel("p_r")
    axes[1, 0].set_ylabel("log10 eps_e_r")
    axes[2, 0].set_ylabel("physical log P")
    axes[2, 0].set_ylim(np.percentile(pilot["lp"][wp[0]:], 0.5) - 2, np.max(pilot["lp"]) + 1)
    fig.tight_layout()
    fig.savefig(outdir / "traces_p_r_eps_e_r_logP.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(17, 13))
    for ax, pname in zip(axes.flat, ("u_r",) + PHYS):
        tr = np.log10 if pname in LOG_AXIS else (lambda v: v)
        vals = {"pilot": tr(column(pilot, pname, *wp).ravel()), "run1": tr(column(r1, pname, 50000, 100000).ravel()),
                "run2": tr(column(r2, pname, 50000, 100000).ravel())}
        lo, hi = np.percentile(np.concatenate(list(vals.values())), [0.05, 99.95])
        bins = np.linspace(lo, hi, 60)
        for k, v in vals.items():
            ax.hist(v, bins=bins, density=True, histtype="step", color=colors[k], lw=1.6 if k == "pilot" else 1.1, label=k)
        ax.set_xlabel(f"log10 {pname}" if pname in LOG_AXIS else pname)
        ax.set_yticks([])
    axes.flat[0].legend()
    for ax in list(axes.flat)[len(PHYS) + 1:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(outdir / "overlay_marginals.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    xb, yb = np.linspace(-3.0, -1.2, 60), np.linspace(-1.8, 0.05, 60)
    for key, run, a, b in (("run1", r1, 50000, 100000), ("run2", r2, 50000, 100000), ("pilot", pilot, *wp)):
        h = gaussian_filter(np.histogram2d(run["extra"][a:b, :, 0].ravel(), run["phys"][a:b, :, er].ravel(), bins=[xb, yb])[0], 1.2)
        fl = np.sort(h.ravel())[::-1]
        cum = np.cumsum(fl) / fl.sum()
        ax.contour(0.5 * (xb[1:] + xb[:-1]), 0.5 * (yb[1:] + yb[:-1]), h.T,
                   levels=sorted(fl[np.searchsorted(cum, q)] for q in (0.68, 0.95)), colors=colors[key], linewidths=[1.8, 1.0])
        ax.plot([], [], color=colors[key], label=f"{key} 68/95%")
    ax.set_xlabel("u_r = log10(p_r - 2)")
    ax.set_ylabel("log10 eps_e_r")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "u_r_vs_eps_e_r.png", dpi=120)
    plt.close(fig)
    say(f"\nWritten to {outdir}")
    (outdir / "comparison_summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
