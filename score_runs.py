"""Score fit_results run directories on the SAME extended data (xray_extra + late_optical)
with per-dataset chi2, so runs fitted to different data can be compared.

    /home/dtak/miniconda3/envs/grb251013c/bin/python score_runs.py modeling/fit_results/final_* ...
"""
import sys, numpy as np
sys.path.insert(0, "/data/dtak/research/grb/GRB251013C")
from grb.modeling import load_all_optical_data
from grb.likelihood import compute_model_flux_all_bands
from grb.results import load_best_fit_params
from grb.spectral_index import load_xrt_spectral_index
xrt, opt = load_all_optical_data(xray_extra=True, late_optical=True)
si = load_xrt_spectral_index()
groups = {"XRT (44+2)": lambda n: n == "xrt", "i-band": lambda n: n == "i-band", "Leavitt": lambda n: n.startswith("Leavitt"),
          "7DT": lambda n: n.startswith("7DT_"), "late optical": lambda n: n.startswith("late_")}
print(f"{'run':40s} " + " ".join(f"{g:>13s}" for g in groups) + f" {'UL pen':>8s} {'SI chi2':>8s} {'total':>9s}")
for d in sys.argv[1:]:
    params, fl, wg = load_best_fit_params(d)
    xm, om, s = compute_model_flux_all_bands(params, xrt, opt, fl, wg, si)
    n = len(xrt["time"]); chi = {"xrt": np.sum(((xrt["flux"] - xm[:n]) / xrt["flux_error"])**2)}
    ul = np.sum((np.maximum(0, xm[n:] - xrt["ul_flux"]) / (xrt["ul_flux"] / 3))**2)
    for ds, m in zip(opt, om): chi[ds["name"]] = np.sum(((ds["flux_mJy"] - m) / ds["flux_err"])**2)
    row = [sum(v for k, v in chi.items() if f(k)) for f in groups.values()]
    tot = sum(chi.values()) + ul + s
    print(f"{d.split('/')[-1]:40s} " + " ".join(f"{v:13.1f}" for v in row) + f" {ul:8.1f} {s:8.1f} {tot:9.1f}")
