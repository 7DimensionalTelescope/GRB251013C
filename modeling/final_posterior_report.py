#!/usr/bin/env python3
"""Final posterior from two independent production chains: burn-in, per-run diagnostics, run comparison, combination.

    python final_posterior_report.py RUN1_DIR RUN2_DIR [--discard1 N] [--discard2 N] [--npool 20] [--outdir DIR]

Both runs must share prior, model resolution / rtol and sampling coordinates (checked). Burn-in per run defaults to the
rule of the checkpoint diagnostics (5 tau_ac,max of the second half, clamped to [n/5, n/2]) and is checked against the
log-P trace and late-window quantile stability. Walkers inside an ensemble interact, so no walker-based R-hat is used:
each run is one chain, and the two runs are compared through their medians / quantiles with Monte-Carlo errors from
each run's ensemble autocorrelation time (MCSE of the median ~ 1.25 sigma sqrt(tau / (N W))). The runs are combined
only if they agree to within CONSISTENCY_* in units of the posterior width; otherwise the report stops at the comparison.
"""
import argparse
import json
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
from scipy.ndimage import gaussian_filter

import early_phase_model as m
from early_phase_mcmc import _autocorr

CONSISTENCY_MEDIAN = 0.25          # |median_1 - median_2| / sigma for every parameter
CONSISTENCY_QUANTILE = 0.30        # |q16 / q84 differences| / sigma
EDGE_FRACTION = 0.05               # prior-bound flag: > 5% of samples within min(1% range, 0.5 sigma) of a bound
PHYS = ("tau", "E_iso", "Gamma0", "theta_c", "A_star", "p", "p_r", "eps_e", "eps_B", "xi_e", "eps_e_r", "eps_B_r",
        "xi_e_r", "A_V")
LOG_AXIS = {"tau", "E_iso", "Gamma0", "theta_c", "A_star", "eps_e", "eps_B", "eps_e_r", "eps_B_r", "A_V"}
_CTX = None


def _init(numerics):
    global _CTX
    m.set_numerics(*numerics)
    defs = m.make_param_defs("production", "wind", free_xi_e=True, free_xi_e_r=True)
    _CTX = (defs, *m.load_fit_data(include_spectral_index=True), "smooth")


def _curves(x):
    defs, xrt, i_data, si, cooling = _CTX
    t_x = np.geomspace(xrt["time"].min() * 0.8, xrt["time"].max() * 1.2, 120)
    t_i = np.geomspace(i_data["time"].min() * 0.8, i_data["time"].max() * 1.2, 120)
    out = m.evaluate_components(x, defs, t_x, t_i, float(i_data["frequency_Hz"].iloc[0]))
    model, _ = m.build_model_from_params(x, defs)
    t_b = np.geomspace(si["time"].min() * 0.8, si["time"].max() * 1.2, 60)
    return np.concatenate([*out, m.model_spectral_index(model, t_b)])


def load(run_dir, discard=None):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "run_config.json").read_text())
    reader = emcee.backends.HDFBackend(str(run_dir / "chain.h5"), name="mcmc", read_only=True)
    coords = config.get("sampling_coords", "physical")
    raw = reader.get_chain()
    labels = config["labels"]
    phys = m.from_sampling_coords(raw, labels, coords)
    lp = reader.get_log_prob()
    if coords != "physical":
        lp = np.subtract(lp, m.sampling_log_jacobian(raw, labels, coords))
    n = reader.iteration
    if discard is None:
        tau_half = _autocorr(raw[n // 2:])
        discard = int(min(n // 2, max(n // 5, 5 * np.nanmax(tau_half))))
    return {"dir": run_dir, "config": config, "coords": coords, "labels": labels, "raw": raw, "phys": phys, "lp": lp,
            "n": n, "discard": discard, "acc": reader.accepted / n, "post": phys[discard:], "post_raw": raw[discard:],
            "post_lp": lp[discard:]}


def col(run, name, key="post"):
    L = run["labels"]
    label = f"log10_{name}" if f"log10_{name}" in L else name
    v = run[key][..., L.index(label)]
    return np.power(10.0, v) if label.startswith("log10_") else v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run1")
    ap.add_argument("run2")
    ap.add_argument("--discard1", type=int, default=None)
    ap.add_argument("--discard2", type=int, default=None)
    ap.add_argument("--npool", type=int, default=20)
    ap.add_argument("--ndraws", type=int, default=300, help="Posterior draws for the light-curve bands (default 300).")
    ap.add_argument("--allow-mismatch", action="store_true", help="Testing only: do not require identical run settings.")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    runs = [load(args.run1, args.discard1), load(args.run2, args.discard2)]
    names = ["run1", "run2"]
    labels = runs[0]["labels"]
    c1, c2 = runs[0]["config"], runs[1]["config"]
    same = {key: c1.get(key) == c2.get(key) for key in ("labels", "param_kwargs", "model_resolution", "model_rtol",
                                                         "sampling_coords", "nwalkers", "moves")}
    same["git_commit"] = c1.get("provenance", {}).get("git_commit") == c2.get("provenance", {}).get("git_commit")
    if not all(same.values()) and not args.allow_mismatch:
        raise SystemExit(f"runs differ in configuration: {[k for k, v in same.items() if not v]}")
    m.set_numerics(c1["model_resolution"], c1["model_rtol"])
    defs = m.make_param_defs(**c1["param_kwargs"])
    bounds = m._sampled_bounds(defs)
    span = bounds[:, 1] - bounds[:, 0]
    outdir = Path(args.outdir) if args.outdir else runs[0]["dir"].parent / f"early_phase_FINAL_posterior_{datetime.now():%Y%m%d_%H%M%S}"
    outdir.mkdir(parents=True, exist_ok=True)
    lines, out = [], {"runs": [str(r["dir"]) for r in runs]}

    def say(t=""):
        lines.append(t)
        print(t, flush=True)

    say(f"=== FINAL early-phase posterior report ({datetime.now():%Y-%m-%d %H:%M}) ===")
    for nm, r in zip(names, runs):
        c, p = r["config"], r["config"].get("provenance", {})
        say(f"{nm}: {r['dir']}")
        say(f"  seed {c['seed']}, {r['n']} steps, {c['nwalkers']} walkers, {c['npool']} workers, sampling coordinates "
            f"{r['coords']}, wall-clock sampling {c.get('total_wall_seconds', 0) / 3600:.2f} h over {len(c['segments'])} segment(s)")
        say(f"  code {p.get('git_commit', '?')} ({p.get('git_branch', '?')}); Python {p.get('python', '?')}, NumPy "
            f"{p.get('numpy', '?')}, SciPy {p.get('scipy', '?')}, emcee {p.get('emcee', '?')}, VegasAfterglow "
            f"{p.get('VegasAfterglow', '?')}; modified tracked files {p.get('git_modified_tracked_files', '?')}")
        say(f"  model resolution {c['model_resolution']}, rtol {c['model_rtol']}; "
            f"{c.get('numerical_settings', {}).get('radiative_fireball', 'radiative_fireball not recorded')}; "
            f"initialisation: {c['initialisation']}")
    say(f"identical configuration (labels, prior, resolution, rtol, coordinates, walkers, moves, commit): {same}")
    init_shared = int((np.abs(np.load(runs[0]["dir"] / "initial_walkers.npy")[:, None]
                              - np.load(runs[1]["dir"] / "initial_walkers.npy")[None]).max(-1) == 0).sum())
    say(f"initial walker coordinates shared between the runs: {init_shared}")

    # ---- burn-in ----
    say("\n--- burn-in (5 tau_ac,max of the second half, clamped to [n/5, n/2], unless given) and its check ---")
    for nm, r in zip(names, runs):
        lp_med = np.median(r["lp"], axis=1)
        k = max(1, (r["n"] - r["discard"]) // 4)
        first, rest = np.median(r["post_lp"][:k]), np.median(r["post_lp"][k:])
        pre = np.median(r["lp"][: max(1, r["discard"] // 4)])
        say(f"{nm}: discard {r['discard']} of {r['n']} steps; median log P over walkers: first quarter of the chain "
            f"{pre:.2f}, first post-burn quarter {first:.2f}, rest {rest:.2f} (difference {first - rest:+.2f}); "
            f"running median log P at the cut {np.median(lp_med[max(0, r['discard'] - 500):r['discard'] + 500]):.2f}")

    # ---- per-run diagnostics ----
    taus = {}
    say("\n--- per-run sampling diagnostics on the post-burn chains ---")
    for nm, r in zip(names, runs):
        a = r["acc"]
        moved = np.mean(np.any(r["post_raw"][1:] != r["post_raw"][:-1], axis=2), axis=0)
        t_s, t_p = _autocorr(r["post_raw"]), _autocorr(r["post"])
        taus[nm] = t_p
        npost = r["n"] - r["discard"]
        say(f"{nm}: acceptance mean {a.mean():.4f}, median {np.median(a):.4f}, min {a.min():.4f}, max {a.max():.4f}; "
            f"post-burn move fraction min {moved.min():.4f}, walkers < 0.01: {int((moved < 0.01).sum())}")
        say(f"  tau_ac (sampling coordinates) max {np.nanmax(t_s):.0f}, median {np.nanmedian(t_s):.0f}; physical max "
            f"{np.nanmax(t_p):.0f} ({labels[int(np.nanargmax(t_p))]}), median {np.nanmedian(t_p):.0f}; N_post / tau_max = "
            f"{npost / np.nanmax(t_s):.1f}; ESS min {npost * 64 / np.nanmax(t_p):.0f}, median {npost * 64 / np.nanmedian(t_p):.0f}")
        say("  tau_ac physical: " + ", ".join(f"{l} {v:.0f}" for l, v in zip(labels, t_p)))
        edges = np.linspace(0, npost, 5).astype(int)
        dev = []
        for j in range(len(labels)):
            allq = np.percentile(r["post"][..., j], [16, 50, 84])
            sig = 0.5 * (allq[2] - allq[0])
            wq = np.array([np.percentile(r["post"][a_:b_, :, j], [16, 50, 84]) for a_, b_ in zip(edges[:-1], edges[1:])])
            dev.append(np.abs(np.subtract(wq, allq)).max(0) / sig)
        dev = np.array(dev)
        say(f"  late-window stability (4 windows): worst |median shift| {dev[:, 1].max():.2f} sigma ({labels[int(dev[:, 1].argmax())]}), "
            f"worst 16% {dev[:, 0].max():.2f}, worst 84% {dev[:, 2].max():.2f}")
        pr = labels.index("p_r")
        say("  p_r 84% / 97.5% per window: " + ", ".join(
            f"{np.percentile(r['post'][a_:b_, :, pr], 84):.4f}/{np.percentile(r['post'][a_:b_, :, pr], 97.5):.4f}"
            for a_, b_ in zip(edges[:-1], edges[1:])))

    # ---- run comparison ----
    say("\n--- run 1 vs run 2 (physical; median [16, 84]); sigma = pooled half-width; z_MC uses each run's tau_ac ---")
    say(f"{'param':<8} {'run1':>32} {'run2':>32} {'dMed/sig':>9} {'d16/sig':>8} {'d84/sig':>8} {'z_MC':>6} {'KS D':>6}")
    worst_med, worst_q = 0.0, 0.0
    comp = {}
    rng = np.random.default_rng(0)
    for name in ("q_E_xir",) + PHYS:
        if name == "q_E_xir":
            v = [np.add(r["post"][..., labels.index("log10_E_iso")], np.log10(r["post"][..., labels.index("xi_e_r")])) for r in runs]
            tr = lambda x: x
            j_tau = labels.index("log10_E_iso")
        else:
            v = [col(r, name) for r in runs]
            tr = np.log10 if name in LOG_AXIS else (lambda x: x)
            j_tau = labels.index(f"log10_{name}" if f"log10_{name}" in labels else name)
        s = [tr(x).ravel() for x in v]
        q = [np.percentile(x, [16, 50, 84]) for x in s]
        sig = 0.5 * (np.percentile(np.concatenate(s), 84) - np.percentile(np.concatenate(s), 16))
        d = (q[1] - q[0]) / sig
        mcse = [1.2533 * np.std(x) * np.sqrt(taus[nm][j_tau] / len(x)) for x, nm in zip(s, names)]
        z = (q[1][1] - q[0][1]) / np.hypot(*mcse)
        a_, b_ = (np.sort(rng.choice(x, min(len(x), 200000), replace=False)) for x in s)
        g = np.concatenate([a_, b_])
        ks = np.max(np.abs(np.searchsorted(a_, g, "right") / len(a_) - np.searchsorted(b_, g, "right") / len(b_)))
        f = lambda x: "{:.4g} [{:.4g}, {:.4g}]".format(*np.percentile(x, [50, 16, 84]))
        raw = [x.ravel() for x in v]
        say(f"{name:<8} {f(raw[0]):>32} {f(raw[1]):>32} {d[1]:>+9.3f} {d[0]:>+8.3f} {d[2]:>+8.3f} {z:>+6.1f} {ks:>6.3f}")
        worst_med = max(worst_med, abs(d[1]))
        worst_q = max(worst_q, abs(d[0]), abs(d[2]))
        comp[name] = {"run1": np.percentile(raw[0], [16, 50, 84]).tolist(), "run2": np.percentile(raw[1], [16, 50, 84]).tolist(),
                      "d_sigma": d.tolist(), "z_mc": float(z), "ks": float(ks)}
    consistent = worst_med < CONSISTENCY_MEDIAN and worst_q < CONSISTENCY_QUANTILE
    say(f"worst |median difference| {worst_med:.3f} sigma (limit {CONSISTENCY_MEDIAN}), worst 16/84% difference {worst_q:.3f} "
        f"sigma (limit {CONSISTENCY_QUANTILE}) -> runs {'CONSISTENT' if consistent else 'NOT CONSISTENT'}")
    say("(z_MC: median difference over the combined Monte-Carlo error; informative only - tau_ac estimates carry their own "
        "uncertainty, and z grows with chain length for practically irrelevant offsets)")
    for nm, r in zip(names, runs):
        pairs = {"E_iso-xi_e_r": (r["post"][..., labels.index("log10_E_iso")], r["post"][..., labels.index("xi_e_r")]),
                 "p_r-eps_e_r": (r["post"][..., labels.index("p_r")], r["post"][..., labels.index("log10_eps_e_r")]),
                 "tau-Gamma0": (r["post"][..., labels.index("log10_tau")], r["post"][..., labels.index("log10_Gamma0")]),
                 "E_iso-xi_e": (r["post"][..., labels.index("log10_E_iso")], r["post"][..., labels.index("xi_e")])}
        say(f"{nm} correlations: " + ", ".join(f"{k} {np.corrcoef(a_.ravel(), b_.ravel())[0, 1]:+.3f}" for k, (a_, b_) in pairs.items()))
    out["comparison"] = comp
    out["consistent"] = bool(consistent)

    # ---- best fit ----
    say("\n--- best log posterior / log likelihood ---")
    data = m.load_fit_data(include_spectral_index=True)
    best = None
    for nm, r in zip(names, runs):
        k = np.unravel_index(np.argmax(r["lp"]), r["lp"].shape)
        x = r["phys"][k]
        lpost = m.log_probability(x, defs, *data, "smooth")
        lprior = m.log_prior(x, defs)
        ch = m.chi2_components(x, defs, *data, "smooth")
        say(f"{nm}: max log posterior {lpost:.4f} (chain value {r['lp'][k]:.4f}; step {k[0]}) = log likelihood {lpost - lprior:.4f} "
            f"+ log prior {lprior:.4f}; chi2 XRT {ch[0]:.3f}, i {ch[1]:.3f}, beta {ch[2]:.3f}, total {sum(ch):.3f}")
        if best is None or lpost > best[0]:
            best = (lpost, x, nm)
    np.save(outdir / "best_fit_params.npy", best[1])
    (outdir / "labels.txt").write_text("\n".join(labels) + "\n")

    if not consistent:
        say("\nThe two final runs are NOT consistent at the stated tolerances: no combined posterior is produced.")
        (outdir / "FINAL_REPORT.txt").write_text("\n".join(lines) + "\n")
        (outdir / "report.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
        _plots(runs, labels, outdir, combined=None)
        return

    # ---- combined posterior ----
    comb = np.concatenate([r["post"].reshape(-1, len(labels)) for r in runs])
    comb_lp = np.concatenate([r["post_lp"].ravel() for r in runs])
    say(f"\n--- COMBINED posterior: run 1 + run 2 post-burn, {len(comb):,} samples (median [16, 84]; 2.5-97.5%) ---")
    final = {}
    for name in ("q_E_xir",) + PHYS:
        if name == "q_E_xir":
            v = np.add(comb[:, labels.index("log10_E_iso")], np.log10(comb[:, labels.index("xi_e_r")]))
        else:
            label = f"log10_{name}" if f"log10_{name}" in labels else name
            v = np.power(10.0, comb[:, labels.index(label)]) if label.startswith("log10_") else comb[:, labels.index(label)]
        q = np.percentile(v, [2.5, 16, 50, 84, 97.5])
        say(f"  {name:<8} {q[2]:.5g} [{q[1]:.5g}, {q[3]:.5g}]   (95%: {q[0]:.5g} - {q[4]:.5g})")
        final[name] = q.tolist()
    out["combined_quantiles_2.5_16_50_84_97.5"] = final
    corr = np.corrcoef(comb.T)
    np.savetxt(outdir / "correlation_matrix_combined.csv", corr, delimiter=",", header=",".join(labels), fmt="%.4f")
    say("correlation matrix (sampled space): correlation_matrix_combined.csv; largest |r|: " + ", ".join(
        f"{labels[i]}-{labels[j]} {corr[i, j]:+.2f}" for i, j in sorted(((i, j) for i in range(len(labels)) for j in range(i)),
                                                                       key=lambda ij: -abs(corr[ij]))[:8]))
    lq = np.add(comb[:, labels.index("log10_E_iso")], np.log10(comb[:, labels.index("xi_e_r")]))
    say(f"log10(E_iso xi_e_r) = {np.median(lq):.3f} [{np.percentile(lq, 16):.3f}, {np.percentile(lq, 84):.3f}] (width "
        f"{np.diff(np.percentile(lq, [16, 84]))[0]:.3f} dex vs log10 E_iso {np.diff(np.percentile(comb[:, 0], [16, 84]))[0]:.3f} dex)")

    say("\n--- prior-bound check (combined): fraction within min(1% range, 0.5 sigma) of each bound ---")
    edge = np.minimum(0.01 * span, 0.5 * comb.std(0))
    lo = (np.subtract(comb, bounds[:, 0]) < edge).mean(0)
    hi = (np.subtract(bounds[:, 1], comb) < edge).mean(0)
    flagged = []
    for j, l in enumerate(labels):
        hist, e = np.histogram(comb[:, j], bins=40)
        peak_at = "lower" if hist.argmax() == 0 else "upper" if hist.argmax() == len(hist) - 1 else ""
        flag = lo[j] > EDGE_FRACTION or hi[j] > EDGE_FRACTION or peak_at
        say(f"  {l:<15} lower {100 * lo[j]:5.1f}%  upper {100 * hi[j]:5.1f}%  marginal peak in the edge bin: {peak_at or 'no'}"
            f"{'   <-- PRIOR-BOUND' if flag else ''}")
        if flag:
            flagged.append(l)
    phys = {n_: np.power(10.0, comb[:, labels.index(f'log10_{n_}')]) for n_ in ("eps_e", "eps_B", "eps_e_r", "eps_B_r")}
    for a_, b_ in m.EPS_SUM_PAIRS:
        say(f"  {a_} + {b_} > 0.99: {100 * np.mean(np.add(phys[a_], phys[b_]) > 0.99):.2f}%")
    say(f"posterior significantly prior-bound for: {flagged or 'none'}")
    out["prior_bound"] = flagged
    say(f"best fit (max log posterior over both runs): {best[2]}, log posterior {best[0]:.4f}")

    # ---- light-curve / spectral-index posterior bands ----
    rng = np.random.default_rng(1)
    pick = comb[rng.choice(len(comb), args.ndraws, replace=False)]
    numerics = (c1["model_resolution"], c1["model_rtol"])
    with ProcessPoolExecutor(max_workers=args.npool, initializer=_init, initargs=(numerics,)) as pool:
        curves = np.array(list(pool.map(_curves, list(pick) + [best[1]], chunksize=2)))
    np.save(outdir / "posterior_curves.npy", curves)
    _lc_plot(curves, data, outdir)
    (outdir / "FINAL_REPORT.txt").write_text("\n".join(lines) + "\n")
    (outdir / "report.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    _plots(runs, labels, outdir, combined=comb)
    from grb.plotting import plot_corner
    sub = rng.choice(len(comb), min(len(comb), 60000), replace=False)
    np.save(outdir / "samples.npy", comb[sub])
    np.save(outdir / "log_probs.npy", comb_lp[sub])
    plot_corner(outdir, max_samples=60000)
    say(f"\nWritten to {outdir}")
    (outdir / "FINAL_REPORT.txt").write_text("\n".join(lines) + "\n")


def _lc_plot(curves, data, outdir):
    from VegasAfterglow.units import mJy
    from grb.utils import flux_error
    xrt, i_data, si = data
    t_x = np.geomspace(xrt["time"].min() * 0.8, xrt["time"].max() * 1.2, 120)
    t_i = np.geomspace(i_data["time"].min() * 0.8, i_data["time"].max() * 1.2, 120)
    t_b = np.geomspace(si["time"].min() * 0.8, si["time"].max() * 1.2, 60)
    draws, best = curves[:-1], curves[-1]
    seg = lambda c, k: c[..., 120 * k:120 * (k + 1)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, k_tot, k_fs, k_rs, t, scale, title in ((axes[0], 0, 1, 2, t_x, 1.0, "XRT 0.3-10 keV"),
                                                   (axes[1], 3, 4, 5, t_i, 1.0 / mJy, "i band")):
        tot = seg(draws, k_tot) * scale
        for lo, hi, a in ((2.5, 97.5, 0.15), (16, 84, 0.3)):
            ax.fill_between(t / 3600, np.percentile(tot, lo, axis=0), np.percentile(tot, hi, axis=0), color="C0", alpha=a, lw=0)
        ax.plot(t / 3600, np.median(tot, axis=0), "C0", label="posterior median (68 / 95% bands)")
        ax.plot(t / 3600, seg(best, k_tot) * scale, "k", lw=1, label="best fit")
        ax.plot(t / 3600, seg(best, k_fs) * scale, "k:", lw=1, label="best fit FS")
        ax.plot(t / 3600, seg(best, k_rs) * scale, "k-.", lw=1, label="best fit RS")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("time since trigger [hr]")
        ax.set_title(title)
    axes[0].errorbar(xrt["time"] / 3600, xrt["flux"], yerr=flux_error(xrt), fmt="o", color="k", ms=3)
    axes[0].set_ylabel("flux [erg cm$^{-2}$ s$^{-1}$]")
    axes[1].errorbar(i_data["time"] / 3600, i_data["flux_mJy"], yerr=i_data["flux_mJy_error"], fmt="o", color="r", ms=3)
    axes[1].set_ylabel("flux density [mJy]")
    beta = draws[:, 720:]
    axes[2].fill_between(t_b / 3600, np.percentile(beta, 16, axis=0), np.percentile(beta, 84, axis=0), color="C0", alpha=0.3)
    axes[2].plot(t_b / 3600, np.median(beta, axis=0), "C0")
    axes[2].plot(t_b / 3600, best[720:], "k", lw=1)
    axes[2].errorbar(si["time"] / 3600, si["beta"], yerr=[si["beta_err_low"], si["beta_err_high"]], fmt="o", color="k", ms=3)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("time since trigger [hr]")
    axes[2].set_ylabel("XRT spectral slope beta")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "posterior_light_curves.png", dpi=150)
    plt.close(fig)


def _plots(runs, labels, outdir, combined):
    colors = ("C0", "C3")
    ndim = len(labels)
    fig, axes = plt.subplots(ndim + 1, 2, figsize=(16, 1.5 * (ndim + 1)), sharex="col")
    for c, r in enumerate(runs):
        stride = max(1, r["n"] // 3000)
        steps = np.arange(0, r["n"], stride)
        for j in range(ndim):
            axes[j, c].plot(steps, r["phys"][::stride, :, j], lw=0.3, alpha=0.3)
            axes[j, c].axvline(r["discard"], color="k", ls="--", lw=0.8)
            axes[j, c].set_ylabel(labels[j], fontsize=7)
        lp = r["lp"][::stride]
        axes[ndim, c].plot(steps, lp, lw=0.3, alpha=0.3)
        axes[ndim, c].plot(steps, np.median(lp, axis=1), "k", lw=1)
        axes[ndim, c].set_ylim(np.percentile(lp, 1) - 2, lp.max() + 1)
        axes[ndim, c].set_ylabel("log P", fontsize=7)
        axes[ndim, c].axvline(r["discard"], color="k", ls="--", lw=0.8)
        axes[0, c].set_title(f"run {c + 1} (dashed: burn-in cut)")
    fig.tight_layout()
    fig.savefig(outdir / "traces_both_runs.png", dpi=90)
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(17, 13))
    for ax, name in zip(axes.flat, ("q_E_xir",) + PHYS):
        vals = []
        for r in runs:
            if name == "q_E_xir":
                vals.append(np.add(r["post"][..., labels.index("log10_E_iso")], np.log10(r["post"][..., labels.index("xi_e_r")])).ravel())
            else:
                v = col(r, name).ravel()
                vals.append(np.log10(v) if name in LOG_AXIS else v)
        lo, hi = np.percentile(np.concatenate(vals), [0.05, 99.95])
        for v, c, nm in zip(vals, colors, ("run 1", "run 2")):
            ax.hist(v, bins=np.linspace(lo, hi, 60), density=True, histtype="step", color=c, lw=1.4, label=nm)
        ax.set_xlabel("log10 E_iso xi_e_r" if name == "q_E_xir" else f"log10 {name}" if name in LOG_AXIS else name)
        ax.set_yticks([])
    axes.flat[0].legend()
    for ax in list(axes.flat)[len(PHYS) + 1:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(outdir / "overlay_marginals_run1_run2.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    specs = (("log10_E_iso", "xi_e_r", (53.3, 55.0), (0.0, 0.3)), ("p_r", "log10_eps_e_r", None, None),
             ("log10_tau", "log10_Gamma0", None, None))
    for ax, (a_, b_, xr, yr) in zip(axes, specs):
        allx = np.concatenate([r["post"][..., labels.index(a_)].ravel() for r in runs])
        ally = np.concatenate([r["post"][..., labels.index(b_)].ravel() for r in runs])
        xb = np.linspace(*(xr or np.percentile(allx, [0.1, 99.9])), 60)
        yb = np.linspace(*(yr or np.percentile(ally, [0.1, 99.9])), 60)
        for r, c, nm in zip(runs, colors, ("run 1", "run 2")):
            h = gaussian_filter(np.histogram2d(r["post"][..., labels.index(a_)].ravel(), r["post"][..., labels.index(b_)].ravel(),
                                               bins=[xb, yb])[0], 1.2)
            fl = np.sort(h.ravel())[::-1]
            cum = np.cumsum(fl) / fl.sum()
            ax.contour(0.5 * (xb[1:] + xb[:-1]), 0.5 * (yb[1:] + yb[:-1]), h.T, colors=c, linewidths=[1.6, 1.0],
                       levels=sorted(fl[np.searchsorted(cum, q)] for q in (0.68, 0.95)))
            ax.plot([], [], color=c, label=f"{nm} 68/95%")
        ax.set_xlabel(a_)
        ax.set_ylabel(b_)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "ridges_E_iso_xi_e_r__p_r_eps_e_r__tau_Gamma0.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
