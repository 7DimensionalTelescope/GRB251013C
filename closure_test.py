"""Closure-relation test for GRB 251013C using the measured alpha/beta.

Inputs are computed live: alpha from lc_indices.fit_alpha (XRT non-flare single
power law; optical post-hump segments), beta_X from the XRT photon index
(flare excluded), beta_opt from the 7DT SED (host SMC A_V from the latest fit).
Relations: F_nu ∝ t^-alpha nu^-beta, standard synchrotron closure relations
(Zhang & Meszaros 2004; Racusin et al. 2009, Table 1; Gao et al. 2013).

Run from the repo root:
    /home/dtak/miniconda3/envs/grb251013c/bin/python closure_test.py [--av 0.27]
Writes figures/closure_test.txt.
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np

from grb.modeling import load_all_optical_data
from grb.spectral_index import load_xrt_spectral_index
from lc_indices import fit_alpha, mean_gamma, _sel, POST_HUMP_I

ROOT = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("plot_mwl_lc", ROOT / "plot_mwl_lc.py")
pm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(pm)

XRAY_NONFLARE = ((400, 2000), (1e5, 5e5))
OPT_POST = (5000, 25000)

# name: (alpha(beta), dalpha/dbeta, p(beta), label)  -- all for p > 2
RELATIONS = {
    "ISM  slow  nu_m<nu<nu_c":          (lambda b: 3 * b / 2,        1.5, lambda b: 2 * b + 1),
    "ISM  slow/fast  nu>nu_c":          (lambda b: (3 * b - 1) / 2,  1.5, lambda b: 2 * b),
    "ISM  fast  nu_c<nu<nu_m (b=1/2)":  (lambda b: 0.25,             0.0, lambda b: np.nan),
    "Wind slow  nu_m<nu<nu_c":          (lambda b: (3 * b + 1) / 2,  1.5, lambda b: 2 * b + 1),
    "Wind slow/fast  nu>nu_c":          (lambda b: (3 * b - 1) / 2,  1.5, lambda b: 2 * b),
    "Wind fast  nu_c<nu<nu_m (b=1/2)":  (lambda b: -0.25,            0.0, lambda b: np.nan),
    "Jet (spreading)  nu_m<nu<nu_c":    (lambda b: 2 * b + 1,        2.0, lambda b: 2 * b + 1),
    "Jet (spreading)  nu>nu_c":         (lambda b: 2 * b,            2.0, lambda b: 2 * b),
    "Jet ISM no-spread  nu_m<nu<nu_c":  (lambda b: 3 * b / 2 + 0.75, 1.5, lambda b: 2 * b + 1),
    "Jet ISM no-spread  nu>nu_c":       (lambda b: (3 * b - 1) / 2 + 0.75, 1.5, lambda b: 2 * b),
}


def test(label, a, ae, b, be, lines):
    lines.append(f"\n--- {label}:  alpha = {a:.2f} ± {ae:.2f},  beta = {b:.2f} ± {be:.2f} ---")
    lines.append(f"{'relation':36s} {'alpha_pred':>12s} {'p(beta)':>8s} {'tension':>9s}")
    for name, (fa, dadb, fp) in RELATIONS.items():
        ap, ape = fa(b), abs(dadb) * be
        sig = (a - ap) / np.hypot(ae, ape)
        flag = "  <-- OK" if abs(sig) < 2 else ("  ~" if abs(sig) < 3 else "")
        lines.append(f"{name:36s} {ap:6.2f} ± {ape:4.2f} {fp(b):8.2f} {sig:+8.1f}σ{flag}")


def main(av_override=None):
    xrt, opt = load_all_optical_data()
    ds = {d["name"]: d for d in opt}
    si = load_xrt_spectral_index()
    lines = ["Closure-relation test, GRB 251013C (F ∝ t^-alpha nu^-beta; tension = (alpha_obs - alpha_pred)/σ)"]

    # X-ray
    m = _sel(xrt["time"], XRAY_NONFLARE)
    rx = fit_alpha(xrt["time"][m], xrt["flux"][m], xrt["flux_error"][m])
    gm, ge = mean_gamma(si, XRAY_NONFLARE)[:2]
    bx, bxe = gm - 1, ge
    test("X-ray 0.45 ks–4.7 d, flare excluded", rx["alpha"], rx["err"], bx, bxe, lines)

    # Optical beta from 7DT (host corrected)
    sdt = pm.sdt_points(); sdt["nuFnu"] = sdt["flux_mJy"] * 1e-26 * sdt["nu"]
    av, src = (av_override, "override") if av_override is not None else pm.host_av()
    _, g_obs, _ = pm.optical_photon_index(sdt)
    _, g7, g7e = pm.optical_photon_index(sdt, a_v=av)
    bo, boe = g7 - 1, g7e
    lines.append(f"\n7DT beta_opt: observed {g_obs-1:.2f}; host-corrected (SMC, A_V = {av:.2f}, {src}) {bo:.2f} ± {boe:.2f}")

    # Optical alpha (post-hump)
    d = ds["i-band"]
    mm = _sel(d["time"], POST_HUMP_I)
    r = fit_alpha(d["time"][mm], d["flux_mJy"][mm], d["flux_err"][mm])
    test("Optical NUTTelA-TAO i, post-hump 6.7–10.8 ks", r["alpha"], r["err"] * max(1.0, np.sqrt(r["chi2"] / r["dof"])), bo, boe, lines)
    ri = r
    for key, lab in (("Leavitt_Rc", "Leavitt Rc"), ("Leavitt_Ic", "Leavitt Ic")):
        d = ds[key]; mm = _sel(d["time"], OPT_POST)
        r = fit_alpha(d["time"][mm], d["flux_mJy"][mm], d["flux_err"][mm])
        err = r["err"] * max(1.0, np.sqrt(r["chi2"] / r["dof"]))   # inflate by sqrt(chi2/dof)
        test(f"Optical {lab} (alpha err × sqrt(chi2/dof))", r["alpha"], err, bo, boe, lines)

    # Cross-band differences
    ric = fit_alpha(ds["Leavitt_Ic"]["time"], ds["Leavitt_Ic"]["flux_mJy"], ds["Leavitt_Ic"]["flux_err"])
    lines.append("\n--- Cross-band (X minus optical) ---")
    lines.append(f"beta_X - beta_opt  = {bx - bo:+.2f} ± {np.hypot(bxe, boe):.2f}   "
                 "(same segment: 0 ; nu_c between: +0.5)")
    for lab, r in (("i", ri), ("Ic", ric)):
        lines.append(f"alpha_X - alpha_{lab:2s} = {rx['alpha'] - r['alpha']:+.2f} ± {np.hypot(rx['err'], r['err']):.2f}   "
                     "(same segment: 0 ; nu_c between, ISM: +0.25 ; wind: -0.25)")
    # A_V needed for the two simplest pictures
    from scipy.optimize import brentq
    d7 = sdt[~sdt["ul"]]
    def beta_at(a):
        return pm.optical_photon_index(sdt, a_v=a)[1] - 1
    lines.append(f"A_V (SMC) needed for beta_opt = beta_X        : {brentq(lambda a: beta_at(a) - bx, 0, 5):.2f}")
    lines.append(f"A_V (SMC) needed for beta_opt = beta_X - 0.5  : {brentq(lambda a: beta_at(a) - (bx - 0.5), 0, 5):.2f}")

    txt = "\n".join(lines); print(txt)
    (ROOT / "figures" / "closure_test.txt").write_text(txt + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--av", type=float, default=None, help="host A_V to use instead of the latest best fit")
    main(ap.parse_args().av)
