"""Multi-telescope, multi-wavelength light curve of GRB 251013C (paper style).

Single log-log panel of energy flux vs time (XRT/Chandra band flux, optical
nu*F_nu per band, radio nu*F_nu), colour = band, marker = telescope, with the
XRT photon-index panel below. Uses every published point in data/ plus the
Chandra/radio points archived under analysis/.

Run from the repo root:
    /home/dtak/miniconda3/envs/grb251013c/bin/python plot_mwl_lc.py            # everything published
    /home/dtak/miniconda3/envs/grb251013c/bin/python plot_mwl_lc.py --fit-only # only what the fit uses
Writes figures/lc_mwl.{png,pdf} (or lc_mwl_fit.* with --fit-only).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import re

from grb.const import FILTER_INFO, FIT_RESULTS_DIR, HOST_AV_LOG10_MEAN, REDSHIFT
from grb.extinction import correct_galactic_extinction, host_extinction_attenuation
from grb.io import read_data
from grb.modeling import load_all_optical_data
from grb.spectral_index import load_xrt_spectral_index
from grb.utils import seconds_from_trigger
from lc_indices import fit_alpha, mean_gamma, _sel

XRAY_NONFLARE = ((400, 2000), (1e5, 5e5))   # XRT segments used for alpha and <Gamma> (flare block excluded)

ROOT = Path(__file__).parent
OUTDIR = ROOT / "figures"
EXTRA = ROOT / "analysis/2026-08_core_flare_team/data/grbexpert_newpoints.csv"

# Band groups: nearest central wavelength decides the group of any filter.
BANDS = {  # name: (centre nm, colour)  -- validated categorical palette, wavelength order
    "u": (361, "#e87ba4"), "g": (467, "#2a78d6"), "V": (550, "#1baf7a"),
    "r/R": (614, "#eda100"), "i/I": (762, "#e34948"), "z": (892, "#6250d6"), "J": (1235, "#008300"),
}
CLEAR = "#8a8a8a"  # unfiltered / clear (assumed R-like, 641 nm)
AB_ZP = 3631.0

# Filters without a wavelength in FILTER_INFO: assumed centre (nm), zero point (Jy)
ASSUMED = {
    "clear": (641, 3080), "Clear": (641, 3080), "white": (641, 3080), "w": (641, 3080),
    "L": (641, 3080),              # GOTO L is a wide optical filter, not NIR L
    "Blue": (525, AB_ZP), "Red": (825, AB_ZP),   # SVOM/VT channels
}

MARKERS = {  # facility -> marker; everything else -> "other GCN telescopes"
    "NUTTelA-TAO": "o", "Leavitt": "s", "7DT": "*", "Calapai": "D", "BassanoBrescianoObservatory": "^",
    "MASTER-SAAO": "v", "LCO": "p", "INO340": "h", "SVOM/COLIBRÍ": "P", "LiverpoolTelescope": "X",
    "SAO/Zeiss-1000": "<", "Mondy/AZT-33IK": ">", "SAORAS": "d", "OHP/T193": "8",
    "OsservatorioAstronomicoNastroVerde": "H", "ISON-Castelgrande": "1", "BOOTES-6/DPRT": "2",
}
PRETTY = {"BassanoBrescianoObservatory": "Bassano Bresciano", "OsservatorioAstronomicoNastroVerde": "Nastro Verde",
          "LiverpoolTelescope": "Liverpool", "SAO/Zeiss-1000": "SAO/Zeiss-1000", "SVOM/COLIBRÍ": "SVOM/COLIBRÍ"}
OTHER = "other GCN telescopes"

SCALES = {"XRT": 10, "i-band": 1, "Leavitt_Rc": 0.1, "Leavitt_Ic": 0.01, "7DT": 1}   # fit-only view offsets

FIT_SERIES = {  # dataset name -> (label, colour, marker); validated 4-slot palette
    "i-band":     ("NUTTelA-TAO  $i$", "#e34948", "o"),
    "Leavitt_Rc": ("Leavitt  $R_c$", "#eda100", "s"),
    "Leavitt_Ic": ("Leavitt  $I_c$", "#1baf7a", "D"),
    "7DT":        ("7DT  (median of 23 filters)", "#2a78d6", "*"),
}

plt.rcParams.update({
    "font.size": 15, "axes.labelsize": 19, "legend.fontsize": 14,
    "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.title_fontsize": 15,
})


def band_of(wl_nm):
    return min(BANDS, key=lambda b: abs(BANDS[b][0] - wl_nm))


def facility_key(name):
    name = str(name).strip()
    if name.startswith("SVOM/COLIBR"):
        return "SVOM/COLIBRÍ"
    return name if name in MARKERS else OTHER


def circular_points():
    """GCN-circular photometry with proper Vega/AB zero points and galactic correction."""
    df = read_data("circular")
    df = df[~df["facility"].isin(["7DT", "NUTTelA-TAO"])].copy()   # better reductions used instead
    wl, zp = [], []
    for f in df["filter"]:
        info = FILTER_INFO.get(f)
        if info and info["central_wavelength_nm"] > 0:
            wl.append(info["central_wavelength_nm"])
            zp.append(info["vega_zero_point_jy"] if info["system"] == "Vega" else AB_ZP)
        else:
            w, z = ASSUMED[f]
            wl.append(w); zp.append(z)
    df["wavelength"] = np.array(wl) * 10.0       # AA, as correct_galactic_extinction expects
    df = correct_galactic_extinction(df)
    zp = np.array(zp)
    flux = zp * 10 ** (-0.4 * df["magnitude"].to_numpy(float)) * 1e3
    err = flux * 0.4 * np.log(10) * df["mag_error"].to_numpy(float)
    return pd.DataFrame({
        "time": df["time"].to_numpy(float), "nu": 2.99792458e17 / (df["wavelength"].to_numpy(float) / 10.0),
        "flux_mJy": flux, "err_mJy": err, "ul": df["upper_limit"].astype(bool).to_numpy(),
        "band": [band_of(w) if f not in ASSUMED else "clear" for w, f in zip(wl, df["filter"])],
        "facility": [facility_key(x) for x in df["facility"]],
    })


def iband_points():
    d = read_data("i_data", correct_galactic_extinction=True, add_converted_flux=True)
    return pd.DataFrame({
        "time": d["time"].to_numpy(float), "nu": d["frequency_Hz"].to_numpy(float),
        "flux_mJy": d["flux_mJy"].to_numpy(float), "err_mJy": d["flux_mJy_error"].to_numpy(float),
        "ul": False, "band": "i/I", "facility": "NUTTelA-TAO",
    })


def sdt_points():
    d = read_data("sdt", correct_galactic_extinction=True, add_converted_flux=True)
    wl_nm = d["wavelength"].to_numpy(float) / 10.0
    return pd.DataFrame({
        "time": [seconds_from_trigger(x) for x in d["date_obs"]], "nu": d["frequency_Hz"].to_numpy(float),
        "flux_mJy": d["flux_mJy"].to_numpy(float), "err_mJy": d["flux_mJy_error"].to_numpy(float),
        "ul": d["is_upper_limit"].astype(bool).to_numpy(), "band": [band_of(w) for w in wl_nm],
        "facility": "7DT",
    })


def fit_points():
    """Exactly the optical datasets load_all_optical_data() feeds the fit (same values)."""
    _, datasets = load_all_optical_data()
    rows = []
    for d in datasets:
        series = "7DT" if d["name"].startswith("7DT_") else d["name"]
        fac = "NUTTelA-TAO" if d["name"] == "i-band" else d["name"].split("_")[0]
        band = band_of(2.99792458e17 / d["frequency"])
        for t, f, e in zip(d["time"], d["flux_mJy"], d["flux_err"]):
            rows.append(dict(time=t, nu=d["frequency"], flux_mJy=f, err_mJy=e, ul=False, band=band,
                             facility=fac, series=series))
    return pd.DataFrame(rows)


def _alpha_fit(t, f, e, mask=None):
    """Power-law fit on the masked points; returns (legend tag, fit dict)."""
    m = np.ones(len(t), bool) if mask is None else mask
    r = fit_alpha(np.asarray(t)[m], np.asarray(f)[m], np.asarray(e)[m])
    return rf"  ($\alpha$ = {r['alpha']:.2f} ± {r['err']:.2f})", r


def _draw_pl(ax, r, color, scale=1.0, extend=1.3, lw=2.2, zorder=5):
    """Fitted power law: solid over the fit range, dashed a little beyond it."""
    t0, t1 = r["trange"]
    f = lambda t: scale * 10 ** r["b"] * t ** (-r["alpha"])
    tt = np.geomspace(t0, t1, 50)
    ax.plot(tt, f(tt), color=color, lw=lw, zorder=zorder)
    for a, b in ((t0 / extend, t0), (t1, t1 * extend)):
        tt = np.geomspace(a, b, 10)
        ax.plot(tt, f(tt), color=color, lw=lw * 0.7, ls="--", zorder=zorder)


def _scale_tag(k):
    n = int(round(np.log10(k)))
    return "" if n == 0 else (r"  ($\times 10$)" if n == 1 else rf"  ($\times 10^{{{n}}}$)")


def plot_fit_series(ax, opt, xrt):
    """Fit-only view: one colour/marker per telescope+band; 7DT collapsed to its median."""
    xtag, rx = _alpha_fit(xrt["time"], xrt["flux"], xrt["flux_error"], _sel(xrt["time"], XRAY_NONFLARE))
    _draw_pl(ax, rx, "black", scale=SCALES["XRT"], extend=1.0, lw=1.6, zorder=2)
    handles = [Line2D([], [], ls="", marker="o", color="black", ms=5,
                      label="Swift/XRT  0.3–10 keV" + xtag + _scale_tag(SCALES["XRT"]))]
    for key, (label, color, mk) in FIT_SERIES.items():
        g = opt[opt["series"] == key].copy()
        k = SCALES[key]
        g["nuFnu"] *= k; g["nuFnu_err"] *= k
        if key.startswith("Leavitt"):
            tag, r = _alpha_fit(g["time"], g["flux_mJy"], g["err_mJy"])
            label = label + tag
            _draw_pl(ax, r, color, scale=k * 1e-26 * float(g["nu"].iloc[0]))
        label = label + _scale_tag(k)
        if key == "7DT":
            t = np.median(g["time"]); y = np.median(g["nuFnu"])
            lo, hi = np.percentile(g["nuFnu"], [16, 84])
            ax.errorbar(t, y, yerr=[[y - lo], [hi - y]], fmt=mk, ms=14, color=color, mec="black", mew=0.5,
                        elinewidth=1.2, capsize=3, zorder=6)
            ax.axvline(t, color=color, ls="--", lw=1.2, alpha=0.8, zorder=1)
            ax.text(t * 1.08, 0.03, "7DT obs.", color=color, ha="left", va="bottom", fontsize=15, rotation=90,
                    transform=mtransforms.blended_transform_factory(ax.transData, ax.transAxes))
        else:
            ax.errorbar(g["time"], g["nuFnu"], yerr=g["nuFnu_err"], fmt=mk, ms=6, color=color, mec="black",
                        mew=0.3, elinewidth=0.7, capsize=0, alpha=0.9, zorder=3)
        handles.append(Line2D([], [], ls="", marker=mk, color=color, mec="black", mew=0.3,
                              ms=12 if mk == "*" else 6, label=label))
    ax.legend(handles=handles, loc="lower left", frameon=False)


def host_av():
    """Best-fit host A_V from the latest final_* MCMC run (falls back to the prior mean)."""
    runs = sorted(Path(FIT_RESULTS_DIR).glob("final_*/bestfit_params.txt"), key=lambda f: f.stat().st_mtime)
    for f in reversed(runs):   # newest run by modification time
        m = re.search(r"^log10_A_V\s+(\S+)", f.read_text(), re.M)
        if m:
            return 10 ** float(m.group(1)), f.parent.name
    return 10 ** HOST_AV_LOG10_MEAN, "prior mean"


def optical_photon_index(sdt, a_v=0.0):
    """Optical photon index from the 7DT SED (power-law fit, detections only).

    a_v > 0 divides the fluxes by the same SMC host attenuation the MCMC model applies."""
    d = sdt[~sdt["ul"]]
    flux = d["flux_mJy"] / host_extinction_attenuation(d["nu"].to_numpy(float), a_v, REDSHIFT)
    beta, _ = np.polyfit(np.log10(d["nu"]), np.log10(flux), 1)
    resid = np.log10(flux) - np.polyval([beta, _], np.log10(d["nu"]))
    n = len(d); x = np.log10(d["nu"]) - np.log10(d["nu"]).mean()
    beta_err = np.sqrt(resid.var(ddof=2) / (x ** 2).sum())
    return float(d["time"].mean()), 1.0 - beta, beta_err


def main(fit_only=False):
    xrt, _ = load_all_optical_data()
    extra = pd.read_csv(EXTRA, comment="#")
    if fit_only:
        opt = fit_points()
    else:
        opt = pd.concat([iband_points(), sdt_points(), circular_points()], ignore_index=True)
    opt["nuFnu"] = opt["flux_mJy"] * 1e-26 * opt["nu"]
    opt["nuFnu_err"] = opt["err_mJy"] * 1e-26 * opt["nu"]

    fig, (ax, axg) = plt.subplots(2, 1, figsize=(13, 10.5), sharex=True,
                                  gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05})

    # --- X-ray ---
    xs = SCALES["XRT"] if fit_only else 1.0
    ax.errorbar(xrt["time"], xrt["flux"] * xs, yerr=xrt["flux_error"] * xs, fmt="o", ms=4, color="black",
                elinewidth=0.8, capsize=0, zorder=5)
    if not fit_only:
        ch = extra[extra.instrument == "Chandra"].iloc[0]
        ax.errorbar(ch.t_s, ch.value, yerr=ch.error, fmt="*", ms=13, color="black", mfc="white", mew=1.2,
                    elinewidth=0.8, capsize=0, zorder=5)

    # --- Optical: colour = band, marker = facility ---
    for (band, fac), g in ([] if fit_only else opt.groupby(["band", "facility"])):
        color = CLEAR if band == "clear" else BANDS[band][1]
        mk = MARKERS.get(fac, ".")
        det, ul = g[~g["ul"]], g[g["ul"]]
        if len(det):
            ax.errorbar(det["time"], det["nuFnu"], yerr=det["nuFnu_err"], fmt=mk, ms=6 if mk != "*" else 9,
                        color=color, mec="black", mew=0.3, elinewidth=0.7, capsize=0, alpha=0.9, zorder=3)
        if len(ul):
            ax.plot(ul["time"], ul["nuFnu"], ls="", marker="v", ms=6, mfc="none", mec=color, mew=1.0,
                    alpha=0.7, zorder=2)

    # --- Radio (nu F_nu, unscaled) ---
    radio_style = {"ALMA": ("D", "97.5 GHz"), "AMI-LA": ("s", "15.5 GHz"), "VLA": ("o", "10 GHz")}
    radio_handles = []
    for name, (mk, lab) in ({} if fit_only else radio_style).items():
        r = extra[extra.instrument == name].iloc[0]
        y, ye = r.value * 1e-26 * r.nu_lo_Hz, r.error * 1e-26 * r.nu_lo_Hz
        ax.errorbar(r.t_s, y, yerr=ye, fmt=mk, ms=7, color="#5b3a1e", mfc="none", mew=1.3,
                    elinewidth=0.8, capsize=0, zorder=4)
        radio_handles.append(Line2D([], [], ls="", marker=mk, color="#5b3a1e", mfc="none", mew=1.3, ms=7,
                                    label=f"{name} {lab}"))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(60, 8e5 if fit_only else 4e6)
    ax.set_ylabel(r"Flux  [erg cm$^{-2}$ s$^{-1}$]")
    ax.grid(which="both", alpha=0.15, lw=0.5)
    ax.tick_params(labelbottom=False)
    sec = ax.secondary_xaxis("top", functions=(lambda s: s / 86400, lambda d: d * 86400))
    sec.set_xlabel("Time since trigger [days]", fontsize=19)
    sec.tick_params(labelsize=16)
    sec.set_xticks([0.01, 0.1, 1, 10])
    sec.set_xticklabels(["0.01", "0.1", "1", "10"])

    if fit_only:
        plot_fit_series(ax, opt, xrt)
    # Legend 1: wavelength / instrument (colour)
    h1 = [Line2D([], [], ls="", marker="o", color="black", ms=5, label="Swift/XRT  0.3–10 keV")]
    if not fit_only:
        h1.append(Line2D([], [], ls="", marker="*", color="black", mfc="white", mew=1.2, ms=12, label="Chandra  0.3–10 keV"))
    h1 += [Line2D([], [], ls="", marker="o", color=c, mec="black", mew=0.3, ms=6, label=rf"{b}  ($\nu F_\nu$)")
           for b, (_, c) in BANDS.items() if (opt["band"] == b).any()]
    if (opt["band"] == "clear").any():
        h1.append(Line2D([], [], ls="", marker="o", color=CLEAR, mec="black", mew=0.3, ms=6, label=r"unfiltered  ($\nu F_\nu$)"))
    if opt["ul"].any():
        h1.append(Line2D([], [], ls="", marker="v", color=CLEAR, mfc="none", mew=1.0, ms=6, label="upper limit"))
    h1 += radio_handles
    if not fit_only:
        leg1 = ax.legend(handles=h1, loc="lower left", frameon=False, title="Band", ncol=2)
        ax.add_artist(leg1)

    # Legend 2: telescope (marker shape)
    facs = [f for f in list(MARKERS) + [OTHER] if (opt["facility"] == f).any()]
    h2 = [Line2D([], [], ls="", marker=MARKERS.get(f, "."), color="#555555", mec="black", mew=0.3,
                 ms=6 if MARKERS.get(f) != "*" else 9, label=PRETTY.get(f, f)) for f in facs]
    if not fit_only:
        ax.legend(handles=h2, loc="upper right", frameon=False, title="Optical telescope", ncol=2)

    # --- Photon index panel ---
    si = load_xrt_spectral_index()
    gamma = 1.0 - si["beta"]
    gm, ge = mean_gamma(si, XRAY_NONFLARE)[:2]
    xlabel = "Swift/XRT" + (rf"  ($\langle\Gamma\rangle$ = {gm:.2f} ± {ge:.2f}, flare excl.)" if fit_only else "")
    axg.errorbar(si["time"], gamma, yerr=[si["beta_err_low"], si["beta_err_high"]],
                 xerr=[si["time_low"], si["time_high"]],
                 fmt="o", ms=3.5, color="black", elinewidth=0.7, capsize=0, label=xlabel)
    sdt7 = opt[opt["facility"] == "7DT"]
    t7, g7, g7e = optical_photon_index(sdt7)
    av, av_src = host_av()
    _, g7h, g7he = optical_photon_index(sdt7, a_v=av)
    c7 = FIT_SERIES["7DT"][1] if fit_only else "#2a78d6"
    if fit_only:   # host-corrected only, index in the legend like alpha above
        axg.errorbar(t7, g7h, yerr=g7he, fmt="*", ms=13, color=c7, mec="black", mew=0.4, elinewidth=1.0, capsize=0,
                     label=rf"7DT  ($\Gamma$ = {g7h:.2f} ± {g7he:.2f}, host $A_V$ = {av:.2f} corrected)")
    else:
        axg.errorbar(t7, g7, yerr=g7e, fmt="*", ms=12, color=c7, mec="black", mew=0.4,
                     elinewidth=1.0, capsize=0, label="7DT optical (observed)")
    if fit_only:
        axg.axvline(t7, color=c7, ls="--", lw=1.2, alpha=0.8, zorder=1)
        axg.axhspan(gm - ge, gm + ge, color="black", alpha=0.12, lw=0, zorder=0)
        axg.axhline(gm, color="black", ls=":", lw=1.3, zorder=1)
    axg.set_ylabel(r"Photon index $\Gamma$")
    axg.set_xlabel("Time since trigger [s]")
    axg.set_ylim(0.8, 3.2)
    axg.grid(which="both", alpha=0.15, lw=0.5)
    axg.legend(frameon=False, loc="upper left", ncol=1, fontsize=13)
    axg.set_ylim(0.8, 3.6)

    OUTDIR.mkdir(exist_ok=True)
    stem = "lc_mwl_fit" if fit_only else "lc_mwl"
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", OUTDIR / f"{stem}.png")
    print(f"optical points: {len(opt)} ({opt['ul'].sum()} UL); 7DT Gamma_opt = {g7:.2f} +/- {g7e:.2f} (observed), "
          f"{g7h:.2f} +/- {g7he:.2f} (host A_V = {av:.3f}, {av_src}) at {t7:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-only", action="store_true", help="only the datasets the MCMC fit uses")
    main(fit_only=ap.parse_args().fit_only)
