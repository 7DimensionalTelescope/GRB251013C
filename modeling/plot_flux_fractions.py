#!/usr/bin/env python3
"""Posterior of the forward- / reverse-shock fractions of the i-band flux as a function of time.

    python plot_flux_fractions.py POSTERIOR_DIR RUN_DIR [--ndraws 1500] [--npool 20]

POSTERIOR_DIR: output of final_posterior_report.py (samples.npy / labels.txt: combined post-burn samples).
RUN_DIR: one of the final runs, for the prior / model resolution / rtol in its run_config.json.
Fractions are F_FS / F_total and F_RS / F_total of the host-extinction-corrected model (the extinction factor cancels).
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import early_phase_model as m

T_GRID = np.geomspace(30.0, 2e4, 150)
_CTX = None


def _init(param_kwargs, resolution, rtol):
    global _CTX
    m.set_numerics(resolution, rtol)
    _CTX = (m.make_param_defs(**param_kwargs), m.load_fit_data(include_spectral_index=False)[1])


def _components(x):
    defs, i_data = _CTX
    out = m.evaluate_components(x, defs, np.array([1000.0]), T_GRID, float(i_data["frequency_Hz"].iloc[0]))
    return np.stack(out[3:])                          # i-band total, FS, RS (cgs, host extinction applied)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("posterior_dir")
    ap.add_argument("run_dir")
    ap.add_argument("--ndraws", type=int, default=1500)
    ap.add_argument("--npool", type=int, default=20)
    args = ap.parse_args()
    post = Path(args.posterior_dir)
    config = json.loads((Path(args.run_dir) / "run_config.json").read_text())
    labels = (post / "labels.txt").read_text().split()
    assert labels == config["labels"]
    samples = np.load(post / "samples.npy")
    draws = samples[np.random.default_rng(20261009).choice(len(samples), args.ndraws, replace=False)]
    with ProcessPoolExecutor(max_workers=args.npool, initializer=_init,
                             initargs=(config["param_kwargs"], config["model_resolution"], config["model_rtol"])) as pool:
        comp = np.array(list(pool.map(_components, list(draws), chunksize=8)))       # (draws, 3, time)
    from VegasAfterglow.units import mJy
    tot, fs, rs = comp[:, 0], comp[:, 1], comp[:, 2]
    f_fs, f_rs = np.divide(fs, tot), np.divide(rs, tot)
    print(f"{len(draws)} posterior draws; max |FS + RS - total| / total = {np.max(np.abs((fs + rs) / tot - 1)):.1e}")

    # time at which the RS stops dominating (first crossing of f_RS = 0.5 going down), per draw
    cross = []
    for fr in f_rs:
        idx = np.where((fr[:-1] >= 0.5) & (fr[1:] < 0.5))[0]
        if len(idx):
            k = idx[0]
            cross.append(np.exp(np.interp(0.5, [fr[k + 1], fr[k]], [np.log(T_GRID[k + 1]), np.log(T_GRID[k])])))
        else:
            cross.append(np.nan)
    cross = np.array(cross)
    i_data = m.load_fit_data(include_spectral_index=False)[1]
    t_lo, t_hi = i_data["time"].min(), i_data["time"].max()

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), gridspec_kw={"width_ratios": [1.25, 1.25, 0.8]})
    ax = axes[0]
    for frac, color, name in ((f_rs, "C3", "reverse shock"), (f_fs, "C0", "forward shock")):
        ax.fill_between(T_GRID, *np.percentile(frac, [2.5, 97.5], axis=0), color=color, alpha=0.15, lw=0)
        ax.fill_between(T_GRID, *np.percentile(frac, [16, 84], axis=0), color=color, alpha=0.35, lw=0)
        ax.plot(T_GRID, np.median(frac, axis=0), color=color, lw=1.8, label=f"{name} (median, 68 / 95%)")
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.axvspan(t_lo, t_hi, color="0.85", zorder=0, label="i-band data span")
    for t in i_data["time"]:
        ax.axvline(t, ymin=0, ymax=0.03, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("time since trigger [s]")
    ax.set_ylabel("fraction of the i-band flux")
    ax.set_title("i band: F_FS / F_total and F_RS / F_total")
    ax.legend(fontsize=8, loc="center right")

    ax = axes[1]
    for flux, color, name in ((rs, "C3", "RS"), (fs, "C0", "FS"), (tot, "k", "total")):
        y = flux / mJy
        ax.fill_between(T_GRID, *np.percentile(y, [16, 84], axis=0), color=color, alpha=0.3, lw=0)
        ax.plot(T_GRID, np.median(y, axis=0), color=color, lw=1.5, label=f"{name} (median, 68%)")
    ax.errorbar(i_data["time"], i_data["flux_mJy"], yerr=i_data["flux_mJy_error"], fmt="o", ms=3, color="k", label="i data")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time since trigger [s]")
    ax.set_ylabel("i-band flux density [mJy] (host-extinguished)")
    ax.set_title("i band: forward- and reverse-shock components")
    ax.legend(fontsize=8)

    ax = axes[2]
    ok = np.isfinite(cross)
    if ok.any():
        ax.hist(np.log10(cross[ok]), bins=40, color="C3", alpha=0.7)
        q = np.percentile(cross[ok], [16, 50, 84])
        ax.axvline(np.log10(q[1]), color="k")
        ax.set_title(f"RS -> FS dominance: t = {q[1]:.0f} [{q[0]:.0f}, {q[2]:.0f}] s\n"
                     f"({100 * ok.mean():.0f}% of draws cross within {T_GRID[0]:.0f} s - {T_GRID[-1] / 1e3:.0f} ks)", fontsize=9)
    ax.set_xlabel("log10 t(F_RS = F_FS) [s]")
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(post / "i_band_FS_RS_fractions.png", dpi=150)
    plt.close(fig)

    rows = [f"i-band FS / RS fractions from {len(draws)} posterior draws (median [16, 84])", "t [s]      FS fraction              RS fraction"]
    for t in (50, 100, 200, 300, 500, 700, 1000, 2000, 5000, 1e4):
        k = np.argmin(np.abs(T_GRID - t))
        a, b = np.percentile(f_fs[:, k], [16, 50, 84]), np.percentile(f_rs[:, k], [16, 50, 84])
        rows.append(f"{T_GRID[k]:7.0f}   {a[1]:.3f} [{a[0]:.3f}, {a[2]:.3f}]    {b[1]:.3f} [{b[0]:.3f}, {b[2]:.3f}]")
    if ok.any():
        rows.append(f"t(F_RS = F_FS) = {q[1]:.0f} [{q[0]:.0f}, {q[2]:.0f}] s ({100 * ok.mean():.0f}% of draws cross in the grid)")
    (post / "i_band_FS_RS_fractions.txt").write_text("\n".join(rows) + "\n")
    np.savez(post / "i_band_FS_RS_fractions.npz", t=T_GRID, f_fs=f_fs, f_rs=f_rs, total=tot, fs=fs, rs=rs, draws=draws,
             t_cross=cross)
    print("\n".join(rows))
    print(f"Written: {post / 'i_band_FS_RS_fractions.png'}, .txt, .npz")


if __name__ == "__main__":
    main()
