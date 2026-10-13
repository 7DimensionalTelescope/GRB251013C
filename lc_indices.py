"""Temporal decay indices (alpha, F ∝ t^-alpha) and mean XRT photon index per segment,
with a check figure overlaying every fitted segment on the light curves.

Segments are fixed by eye from the light curve (see SEGMENTS_*); the XRT flare
block (5.2-5.6 ks) and the optical hump (1.5-5 ks) are excluded from decay fits.
Run from the repo root:
    /home/dtak/miniconda3/envs/grb251013c/bin/python lc_indices.py
Writes figures/lc_indices.txt and figures/lc_indices_check.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from grb.modeling import load_all_optical_data
from grb.spectral_index import load_xrt_spectral_index

FIGDIR = Path(__file__).parent / "figures"

SEGMENTS_X = {  # name: time range(s) in seconds
    "X1  first orbit (pre-flare)":        (400, 2000),
    "X2  late, after flare (2 pts)":      (1e5, 5e5),
    "X1+X2 single PL, flare excluded":    ((400, 2000), (1e5, 5e5)),
}
SEGMENTS_OPT = {
    "O1  initial decay":                  (90, 300),
    "O2  plateau":                        (300, 700),
    "O3  pre-hump decay":                 (700, 1500),
    "O4  post-hump decay (i only)":       (5000, 11000),
}
JOINT_POST = (5000, 25000)   # i + Leavitt Rc/Ic + 7DT r,i, common alpha, free normalisations
EXCLUDED_X, EXCLUDED_OPT = (5000, 6000), (1500, 5000)

COLORS = {"i-band": "#e34948", "Leavitt_Rc": "#eda100", "Leavitt_Ic": "#1baf7a", "7DT_r": "#2a78d6", "7DT_i": "#2a78d6"}
MARKERS = {"i-band": "o", "Leavitt_Rc": "s", "Leavitt_Ic": "D", "7DT_r": "*", "7DT_i": "*"}


def _sel(t, rng):
    if isinstance(rng[0], tuple):
        return np.any([(t >= a) & (t <= b) for a, b in rng], axis=0)
    return (t >= rng[0]) & (t <= rng[1])


def fit_alpha(t, f, e):
    """Weighted power-law fit in log space: log F = b - alpha log t."""
    x, y, s = np.log10(t), np.log10(f), e / (f * np.log(10))
    (b, a), cov = curve_fit(lambda x, b, a: b - a * x, x, y, sigma=s, absolute_sigma=True)
    chi2 = np.sum(((y - (b - a * x)) / s) ** 2)
    return dict(alpha=a, err=np.sqrt(cov[1, 1]), chi2=chi2, dof=len(t) - 2, n=len(t), b=b,
                trange=(t.min(), t.max()))


def fit_alpha_joint(sets):
    """Common alpha, one free log-normalisation per dataset."""
    x = np.concatenate([np.log10(d[0]) for d in sets]); y = np.concatenate([np.log10(d[1]) for d in sets])
    s = np.concatenate([d[2] / (d[1] * np.log(10)) for d in sets])
    idx = np.concatenate([np.full(len(d[0]), i) for i, d in enumerate(sets)])
    def model(x, a, *norms):
        return np.asarray(norms)[idx] - a * x
    p, cov = curve_fit(model, x, y, p0=[1.0] + [0.0] * len(sets), sigma=s, absolute_sigma=True)
    chi2 = np.sum(((y - model(x, *p)) / s) ** 2)
    return dict(alpha=p[0], err=np.sqrt(cov[0, 0]), chi2=chi2, dof=len(x) - len(p), n=len(x), norms=p[1:])


def mean_gamma(si, rng):
    m = _sel(si["time"], rng)
    g = 1.0 - si["beta"][m]
    err = 0.5 * (si["beta_err_low"][m] + si["beta_err_high"][m])
    w = 1 / err ** 2
    gm = np.sum(w * g) / np.sum(w)
    chi2 = np.sum(((g - gm) / err) ** 2)
    return gm, 1 / np.sqrt(np.sum(w)), g.std(ddof=1) if m.sum() > 1 else np.nan, chi2, m.sum() - 1


def pl(t, b, a):
    return 10 ** b * t ** (-a)


def draw_fit(ax, r, color="black", ls="-", label=None, extend=1.0, ty=1.6):
    t0, t1 = r["trange"]
    tt = np.geomspace(t0 / extend, t1 * extend, 50)
    ax.plot(tt, pl(tt, r["b"], r["alpha"]), color=color, ls=ls, lw=1.8, zorder=6)
    if label:
        tm = np.sqrt(t0 * t1)
        ax.annotate(rf"{label}: $\alpha$={r['alpha']:.2f}±{r['err']:.2f}", (tm, pl(tm, r["b"], r["alpha"]) * ty),
                    ha="center", va="bottom", fontsize=9.5, color=color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8), zorder=7)


def main():
    xrt, opt = load_all_optical_data()
    si = load_xrt_spectral_index()
    ds = {d["name"]: d for d in opt}
    lines = []
    P = lambda s="": (print(s), lines.append(s))
    fx, fo = {}, {}

    P("=== XRT photon index Gamma (weighted mean) ===")
    P(f"{'segment':42s} {'Gamma':>14s} {'scatter':>8s} {'chi2/dof':>10s}  n")
    for name, rng in {"X1  first orbit 0.45-1.9 ks (pre-flare)": (400, 2000),
                      "    flare block 5.2-5.6 ks (excluded)": (5000, 6000),
                      "X2  late 1.4e5-4.1e5 s": (1e5, 5e5),
                      "X1+X2 all non-flare": ((400, 2000), (1e5, 5e5))}.items():
        gm, ge, sd, chi2, dof = mean_gamma(si, rng)
        P(f"{name:42s} {gm:6.2f} +/- {ge:4.2f} {sd:8.2f} {chi2:6.1f}/{dof:<3d}  {dof+1}")
    P("   (beta = Gamma - 1 ; F_nu ∝ nu^-beta)")
    P()

    P("=== X-ray temporal index alpha (F ∝ t^-alpha) ===")
    P(f"{'segment':42s} {'alpha':>14s} {'chi2/dof':>10s}  n")
    for name, rng in SEGMENTS_X.items():
        m = _sel(xrt["time"], rng)
        r = fx[name] = fit_alpha(xrt["time"][m], xrt["flux"][m], xrt["flux_error"][m])
        P(f"{name:42s} {r['alpha']:6.2f} +/- {r['err']:4.2f} {r['chi2']:6.1f}/{r['dof']:<3d}  {r['n']}")
    P()

    P("=== Optical temporal index alpha (NUTTelA-TAO i-band) ===")
    P(f"{'segment':42s} {'alpha':>14s} {'chi2/dof':>10s}  n")
    d = ds["i-band"]
    for name, rng in SEGMENTS_OPT.items():
        m = _sel(d["time"], rng)
        r = fo[name] = fit_alpha(d["time"][m], d["flux_mJy"][m], d["flux_err"][m])
        P(f"{name:42s} {r['alpha']:6.2f} +/- {r['err']:4.2f} {r['chi2']:6.1f}/{r['dof']:<3d}  {r['n']}")
    for key in ("Leavitt_Rc", "Leavitt_Ic"):
        dd = ds[key]
        r = fo[key] = fit_alpha(dd["time"], dd["flux_mJy"], dd["flux_err"])
        P(f"{'O4  post-hump, ' + key + ' alone':42s} {r['alpha']:6.2f} +/- {r['err']:4.2f} {r['chi2']:6.1f}/{r['dof']:<3d}  {r['n']}")
    joint_keys, sets = [], []
    for key in ("i-band", "Leavitt_Rc", "Leavitt_Ic", "7DT_r", "7DT_i"):
        dd = ds[key]; m = _sel(dd["time"], JOINT_POST)
        if m.any():
            joint_keys.append(key); sets.append((dd["time"][m], dd["flux_mJy"][m], dd["flux_err"][m]))
    rj = fit_alpha_joint(sets)
    P(f"{'O4  joint 5-25 ks: i+Rc+Ic+7DT(r,i), free norms':42s} {rj['alpha']:6.2f} +/- {rj['err']:4.2f} {rj['chi2']:6.1f}/{rj['dof']:<3d}  {rj['n']}")
    P()
    P("Excluded from decay fits: XRT flare block 5.2-5.6 ks; optical hump 1.5-5 ks.")
    P("Errors are 1-sigma from weighted least squares (absolute errors); chi2/dof >> 1 means the segment is not a clean power law.")
    FIGDIR.mkdir(exist_ok=True)
    (FIGDIR / "lc_indices.txt").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ check figure
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12})
    fig, (ax_x, ax_o, ax_j) = plt.subplots(3, 1, figsize=(11, 14))

    # X-ray
    ax_x.errorbar(xrt["time"], xrt["flux"], yerr=xrt["flux_error"], fmt="o", ms=4, color="black", elinewidth=0.7, capsize=0)
    ax_x.axvspan(*EXCLUDED_X, color="gray", alpha=0.15, lw=0)
    ax_x.text(np.sqrt(np.prod(EXCLUDED_X)), 2e-13, "flare\n(excluded)", ha="center", fontsize=9, color="gray")
    draw_fit(ax_x, fx["X1  first orbit (pre-flare)"], color="#e34948", label="X1", ty=2.5)
    draw_fit(ax_x, fx["X2  late, after flare (2 pts)"], color="#2a78d6", label="X2", ty=3)
    draw_fit(ax_x, fx["X1+X2 single PL, flare excluded"], color="#1baf7a", ls="--", label="X1+X2", ty=0.08)
    ax_x.set_ylabel(r"XRT 0.3–10 keV flux [erg cm$^{-2}$ s$^{-1}$]")
    ax_x.set_title("X-ray: segment power-law fits (solid = fit range)", fontsize=12)

    # Optical i-band segments
    ax_o.errorbar(d["time"], d["flux_mJy"], yerr=d["flux_err"], fmt="o", ms=4, color=COLORS["i-band"], mec="black",
                  mew=0.3, elinewidth=0.7, capsize=0, label="NUTTelA-TAO i")
    ax_o.axvspan(*EXCLUDED_OPT, color="gray", alpha=0.15, lw=0)
    ax_o.text(np.sqrt(np.prod(EXCLUDED_OPT)), 0.7, "hump\n(excluded)", ha="center", fontsize=9, color="gray")
    for (name, r), ty in zip(((k, v) for k, v in fo.items() if k.startswith("O")), (1.8, 1.6, 1.8, 1.8)):
        draw_fit(ax_o, r, color="black", label=name[:2], ty=ty)
    ax_o.set_ylabel("Flux density [mJy]")
    ax_o.set_title("Optical i-band: segment fits O1–O4", fontsize=12)
    ax_o.legend(frameon=False, loc="lower left")

    # Post-hump: individual + joint
    for key in ("i-band", "Leavitt_Rc", "Leavitt_Ic", "7DT_r", "7DT_i"):
        dd = ds[key]; m = dd["time"] >= 4000
        ax_j.errorbar(dd["time"][m], dd["flux_mJy"][m], yerr=dd["flux_err"][m], fmt=MARKERS[key], ms=7 if key.startswith("7DT") else 4.5,
                      color=COLORS[key], mec="black", mew=0.3, elinewidth=0.7, capsize=0, label=key.replace("_", " "))
    draw_fit(ax_j, fo["O4  post-hump decay (i only)"], color=COLORS["i-band"], label="i only", ty=1.7)
    draw_fit(ax_j, fo["Leavitt_Rc"], color=COLORS["Leavitt_Rc"], label="Rc alone", ty=0.45)
    draw_fit(ax_j, fo["Leavitt_Ic"], color=COLORS["Leavitt_Ic"], label="Ic alone", ty=1.8)
    tt = np.geomspace(*JOINT_POST, 50)
    for key, b in zip(joint_keys, rj["norms"]):
        ax_j.plot(tt, pl(tt, b, rj["alpha"]), color=COLORS[key], ls="--", lw=1.4, alpha=0.9, zorder=5)
    ax_j.plot([], [], color="gray", ls="--", label=rf"joint fit  $\alpha$={rj['alpha']:.2f}±{rj['err']:.2f}  ($\chi^2$/dof={rj['chi2']:.0f}/{rj['dof']})")
    ax_j.set_ylabel("Flux density [mJy]")
    ax_j.set_xlabel("Time since trigger [s]")
    ax_j.set_title("Post-hump 5–25 ks: individual fits (solid) and joint fit with free normalisations (dashed)", fontsize=12)
    ax_j.legend(frameon=False, loc="lower left", fontsize=9.5)
    ax_j.set_xlim(4000, 3e4)

    for ax in (ax_x, ax_o, ax_j):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(which="both", alpha=0.15, lw=0.5)
    ax_o.set_xlabel("Time since trigger [s]")
    fig.tight_layout()
    fig.savefig(FIGDIR / "lc_indices_check.png", dpi=170, bbox_inches="tight")
    print("wrote", FIGDIR / "lc_indices_check.png")


if __name__ == "__main__":
    main()
