#!/usr/bin/env python3
"""q_E_xir pilot versus the two physical-coordinate production runs: mixing efficiency and physical posterior.

    python compare_q_pilot.py PILOT_DIR RUN1_DIR RUN2_DIR [--outdir DIR]

Mixing is compared on windows of EQUAL length (unconverged autocorrelation estimates grow with the window):
pilot steps 5k-30k (started from posterior draws, so only a short discard) against run 1 / run 2 steps 50k-75k and
75k-100k. Autocorrelation times are computed in both coordinate systems for every run (physical: log10_E_iso;
q: q_E_xir = log10(E_iso xi_e_r)), with the same estimator as the checkpoint diagnostics
(early_phase_mcmc._autocorr). Acceptance in a window is the fraction of steps in which a walker moved. The slowest
direction is identified from autocorrelation times of the principal components of the standardised posterior.
Physical marginals: pilot 5k-30k against run 1 / run 2 50k-100k.
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

from early_phase_mcmc import _autocorr
from early_phase_model import from_sampling_coords, sampling_labels, to_sampling_coords

PHYS = ("tau", "E_iso", "xi_e", "xi_e_r", "Gamma0", "theta_c", "A_star", "p", "p_r", "eps_e", "eps_B", "eps_e_r",
        "eps_B_r", "A_V")
LOG_AXIS = {"tau", "E_iso", "Gamma0", "theta_c", "A_star", "eps_e", "eps_B", "eps_e_r", "eps_B_r", "A_V"}
KEY = ("tau", "E_iso", "q_E_xir", "xi_e", "xi_e_r", "p_r", "eps_e_r", "Gamma0")


def load(run_dir):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "run_config.json").read_text())
    reader = emcee.backends.HDFBackend(str(run_dir / "chain.h5"), name="mcmc", read_only=True)
    coords = config.get("sampling_coords", "physical")
    raw = reader.get_chain()
    labels = config["labels"]
    phys = from_sampling_coords(raw, labels, coords)
    return {"dir": run_dir, "config": config, "coords": coords, "labels": labels, "phys": phys,
            "q": to_sampling_coords(phys, labels, "q_E_xir") if coords == "physical" else raw,
            "lp": reader.get_log_prob(), "n": reader.iteration}


def column(run, name, a, b):
    L = run["labels"]
    if name == "q_E_xir":
        return run["q"][a:b, :, L.index("log10_E_iso")]
    label = f"log10_{name}" if f"log10_{name}" in L else name
    v = run["phys"][a:b, :, L.index(label)]
    return 10 ** v if label.startswith("log10_") else v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pilot")
    ap.add_argument("run1")
    ap.add_argument("run2")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--pilot-discard", type=int, default=5000, help="Pilot steps discarded (default 5000).")
    args = ap.parse_args()
    pilot, r1, r2 = load(args.pilot), load(args.run1), load(args.run2)
    labels = pilot["labels"]
    assert r1["labels"] == labels == r2["labels"]
    assert pilot["config"]["param_kwargs"] == r1["config"]["param_kwargs"] == r2["config"]["param_kwargs"], "priors differ"
    qlabels = sampling_labels(labels, "q_E_xir")
    outdir = Path(args.outdir) if args.outdir else pilot["dir"] / "comparison_vs_run1_run2"
    outdir.mkdir(parents=True, exist_ok=True)
    lines, out = [], {}

    def say(t=""):
        lines.append(t)
        print(t, flush=True)

    # windows of equal length
    wp = (args.pilot_discard, pilot["n"])
    windows = [("pilot q", pilot, *wp), ("run1 50-75k", r1, 50000, 75000), ("run1 75-100k", r1, 75000, 100000),
               ("run2 50-75k", r2, 50000, 75000), ("run2 75-100k", r2, 75000, 100000)]
    say(f"=== q_E_xir pilot vs physical-coordinate runs ({pilot['dir'].name}) ===")
    say(f"pilot: seed {pilot['config']['seed']}, {pilot['n']} steps, sampling coordinates {pilot['coords']}, "
        f"{pilot['config']['total_wall_seconds'] / 3600:.2f} h; init: {pilot['config']['initialisation'][:80]}...")
    say(f"mixing windows ({wp[1] - wp[0]} steps each): " + ", ".join(n for n, *_ in windows))

    say("\n--- acceptance (fraction of steps in which a walker moved, per window) ---")
    acc = {}
    for name, run, a, b in windows:
        x = run["phys"][a:b]
        moved = np.mean(np.any(x[1:] != x[:-1], axis=2), axis=0)
        acc[name] = moved
        say(f"{name:<14} mean {moved.mean():.4f}  median {np.median(moved):.4f}  min {moved.min():.4f} (walker "
            f"{moved.argmin()})  max {moved.max():.4f}  walkers < 0.01: {int((moved < 0.01).sum())}")
    say(f"pilot whole-chain acceptance (emcee counter): mean "
        f"{np.mean(emcee.backends.HDFBackend(str(pilot['dir'] / 'chain.h5'), name='mcmc', read_only=True).accepted / pilot['n']):.4f}")
    out["acceptance"] = {k: v.tolist() for k, v in acc.items()}

    say("\n--- integrated autocorrelation time [steps] on equal-length windows ---")
    say("(q-coordinate table: the coordinates the pilot samples; for runs 1/2 q_E_xir is derived from their chains)")
    taus = {}
    for system, key, labs in (("q coordinates", "q", qlabels), ("physical coordinates", "phys", labels)):
        say(f"{system:<16}" + "".join(f"{n:>14}" for n, *_ in windows))
        t = {n: _autocorr(run[key][a:b]) for n, run, a, b in windows}
        taus[system] = t
        for j, l in enumerate(labs):
            say(f"{l:<16}" + "".join(f"{t[n][j]:>14.0f}" for n, *_ in windows))
        say(f"{'max':<16}" + "".join(f"{np.nanmax(t[n]):>14.0f}" for n, *_ in windows))
        say(f"{'median':<16}" + "".join(f"{np.nanmedian(t[n]):>14.0f}" for n, *_ in windows))
        say(f"{'ESS (min)':<16}" + "".join(f"{(b - a) * 64 / np.nanmax(t[n]):>14.0f}" for n, run, a, b in windows))
        say("")
    out["tau"] = {s: {n: v.tolist() for n, v in t.items()} for s, t in taus.items()}

    say("--- slowest directions: autocorrelation time of principal components (standardised q coordinates) ---")
    for name, run, a, b in windows:
        x = run["q"][a:b]
        flat = x.reshape(-1, x.shape[-1])
        mu, sd = flat.mean(0), flat.std(0)
        z = (x - mu) / sd
        evals, evecs = np.linalg.eigh(np.cov(((flat - mu) / sd).T))
        pcs = np.einsum("twd,dk->twk", z, evecs)
        t = _autocorr(pcs)
        order = np.argsort(t)[::-1][:2]
        for rank, k in enumerate(order):
            loading = evecs[:, k]
            top = np.argsort(np.abs(loading))[::-1][:5]
            say(f"{name:<14} slow #{rank + 1}: tau {t[k]:7.0f}, PC variance {evals[k]:.3f} (standardised); loadings "
                + ", ".join(f"{qlabels[i]} {loading[i]:+.2f}" for i in top))
    say("")

    say("--- physical marginals: pilot 5k-30k vs run1 / run2 50k-100k (median [16, 84]) ---")
    say(f"{'param':<8} {'pilot':>32} {'run1':>32} {'run2':>32} {'dMed/sig vs pooled old':>24} {'KS D vs pooled':>15}")
    stats = {}
    rng = np.random.default_rng(0)
    for name in ("q_E_xir",) + PHYS:
        p = column(pilot, name, *wp).ravel()
        o1, o2 = column(r1, name, 50000, 100000).ravel(), column(r2, name, 50000, 100000).ravel()
        tr = (lambda v: np.log10(v)) if name in LOG_AXIS else (lambda v: v)
        pooled = np.concatenate([o1, o2])
        sig = 0.5 * np.diff(np.percentile(tr(pooled), [16, 84]))[0]
        dmed = (np.median(tr(p)) - np.median(tr(pooled))) / sig
        a_, b_ = np.sort(rng.choice(tr(p), 200000, replace=False)), np.sort(rng.choice(tr(pooled), 200000, replace=False))
        g = np.concatenate([a_, b_])
        ks = np.max(np.abs(np.searchsorted(a_, g, "right") / len(a_) - np.searchsorted(b_, g, "right") / len(b_)))
        f = lambda v: "{:.4g} [{:.4g}, {:.4g}]".format(*np.percentile(v, [50, 16, 84]))
        say(f"{name:<8} {f(p):>32} {f(o1):>32} {f(o2):>32} {dmed:>+24.2f} {ks:>15.3f}")
        stats[name] = {"pilot": np.percentile(p, [16, 50, 84]).tolist(), "run1": np.percentile(o1, [16, 50, 84]).tolist(),
                       "run2": np.percentile(o2, [16, 50, 84]).tolist(), "dmed_sigma": dmed, "ks": ks}
    out["marginals"] = stats
    say("(old runs differed from each other by up to 0.26 sigma and KS D 0.11 along the ridge)")

    say("\n--- correlations (sampled log10 / linear space) ---")
    pairs = [("log10_E_iso", "xi_e_r"), ("log10_E_iso", "xi_e"), ("p_r", "log10_eps_e_r"), ("log10_tau", "log10_Gamma0")]
    for a, b in pairs:
        cs = [np.corrcoef(run["phys"][s:e, :, labels.index(a)].ravel(), run["phys"][s:e, :, labels.index(b)].ravel())[0, 1]
              for run, s, e in ((pilot, *wp), (r1, 50000, 100000), (r2, 50000, 100000))]
        say(f"{a + ' - ' + b:<30} pilot {cs[0]:+.3f}  run1 {cs[1]:+.3f}  run2 {cs[2]:+.3f}")
    cq = np.corrcoef(pilot["q"][wp[0]:, :, 0].ravel(), pilot["q"][wp[0]:, :, labels.index("xi_e_r")].ravel())[0, 1]
    say(f"{'q_E_xir - xi_e_r (pilot sampling coordinates)':<46} {cq:+.3f}")

    say("\n--- pilot window stability (median [16, 84]) ---")
    halves = [(wp[0], (wp[0] + wp[1]) // 2), ((wp[0] + wp[1]) // 2, wp[1])]
    for name in ("q_E_xir", "E_iso", "xi_e_r", "xi_e", "tau", "p_r", "eps_e_r", "Gamma0"):
        r = [np.percentile(column(pilot, name, a, b), [16, 50, 84]) for a, b in halves]
        tr = (lambda v: np.log10(v)) if name in LOG_AXIS else (lambda v: v)
        shift = (tr(r[1][1]) - tr(r[0][1])) / (0.5 * (tr(r[0][2]) - tr(r[0][0])))
        say(f"{name:<8} 1st half {r[0][1]:.4g} [{r[0][0]:.4g}, {r[0][2]:.4g}]   2nd half {r[1][1]:.4g} [{r[1][0]:.4g}, "
            f"{r[1][2]:.4g}]   median shift {shift:+.2f} sigma")
    best = [float(run["lp"].max()) for run in (pilot, r1, r2)]
    say(f"\nbest log_prob: pilot {best[0]:.4f}, run1 {best[1]:.4f}, run2 {best[2]:.4f}")
    (outdir / "comparison_summary.txt").write_text("\n".join(lines) + "\n")
    (outdir / "comparison.json").write_text(json.dumps(out, indent=1, default=float) + "\n")

    # plots
    colors = {"pilot": "k", "run1": "C0", "run2": "C3"}
    fig, axes = plt.subplots(4, 4, figsize=(17, 13))
    for ax, name in zip(axes.flat, ("q_E_xir",) + PHYS):
        tr = (lambda v: np.log10(v)) if name in LOG_AXIS else (lambda v: v)
        vals = {"pilot": tr(column(pilot, name, *wp).ravel()), "run1": tr(column(r1, name, 50000, 100000).ravel()),
                "run2": tr(column(r2, name, 50000, 100000).ravel())}
        lo, hi = np.percentile(np.concatenate(list(vals.values())), [0.05, 99.95])
        bins = np.linspace(lo, hi, 60)
        for k, v in vals.items():
            ax.hist(v, bins=bins, density=True, histtype="step", color=colors[k], lw=1.6 if k == "pilot" else 1.1, label=k)
        ax.set_xlabel(f"log10 {name}" if name in LOG_AXIS else name)
        ax.set_yticks([])
    axes.flat[0].legend()
    for ax in list(axes.flat)[len(PHYS) + 1:]:
        ax.axis("off")
    fig.suptitle("Physical marginals: q-coordinate pilot (5k-30k) vs physical-coordinate runs 1 / 2 (50k-100k)")
    fig.tight_layout()
    fig.savefig(outdir / "overlay_marginals.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    xb, yb = np.linspace(53.3, 55.0, 60), np.linspace(0.0, 0.3, 60)
    for key, run, a, b in (("run1", r1, 50000, 100000), ("run2", r2, 50000, 100000), ("pilot", pilot, *wp)):
        h = gaussian_filter(np.histogram2d(np.log10(column(run, "E_iso", a, b)).ravel(), column(run, "xi_e_r", a, b).ravel(),
                                           bins=[xb, yb])[0], 1.2)
        flat = np.sort(h.ravel())[::-1]
        cum = np.cumsum(flat) / flat.sum()
        levels = sorted(flat[np.searchsorted(cum, f)] for f in (0.68, 0.95))
        ax.contour(0.5 * (xb[1:] + xb[:-1]), 0.5 * (yb[1:] + yb[:-1]), h.T, levels=levels, colors=colors[key],
                   linewidths=[1.8, 1.0])
        ax.plot([], [], color=colors[key], label=f"{key} 68/95%")
    ax.set_xlabel("log10 E_iso")
    ax.set_ylabel("xi_e_r")
    ax.legend()
    ax.set_title("E_iso - xi_e_r ridge")
    fig.tight_layout()
    fig.savefig(outdir / "ridge_E_iso_xi_e_r.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    steps = np.arange(pilot["n"])[::10]
    for ax, (name, lab) in zip(axes, (("q_E_xir", "q_E_xir"), ("E_iso", "log10 E_iso"), ("xi_e_r", "xi_e_r"), ("tau", "log10 tau"))):
        v = column(pilot, name, 0, pilot["n"])[::10]
        ax.plot(steps, np.log10(v) if name in ("E_iso", "tau") else v, lw=0.3, alpha=0.4)
        ax.set_ylabel(lab)
    axes[-1].set_xlabel("pilot step")
    fig.tight_layout()
    fig.savefig(outdir / "pilot_traces_q_E_xir.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    names = [n for n, *_ in windows]
    x = np.arange(len(qlabels))
    for k, n in enumerate(names):
        ax.bar(x + (k - 2) * 0.16, taus["q coordinates"][n], width=0.16, label=n)
    ax.set_xticks(x, qlabels, rotation=60, fontsize=8)
    ax.set_ylabel("tau_ac [steps], 25k-step windows")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "autocorr_comparison.png", dpi=110)
    plt.close(fig)
    say(f"\nWritten to {outdir}")
    (outdir / "comparison_summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
