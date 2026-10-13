"""Closure-relation (alpha-beta plane) figure for GRB 251013C.

Uses cr_plot() from the user's grbpy package (/home/dtak/script/grbpy/external/
efs_cr_test.py) for the relation lines, and the same live-measured alpha/beta
as closure_test.py for the points.

Run from the repo root:
    /home/dtak/miniconda3/envs/grb251013c/bin/python plot_cr.py [--av 0.27]
Writes figures/closure_relation.png / .pdf.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/dtak/script/grbpy/external")
from efs_cr_test import cr_plot  # noqa: E402

from grb.modeling import load_all_optical_data  # noqa: E402
from grb.spectral_index import load_xrt_spectral_index  # noqa: E402
from lc_indices import fit_alpha, mean_gamma, _sel  # noqa: E402

ROOT = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("plot_mwl_lc", ROOT / "plot_mwl_lc.py")
pm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(pm)

XRAY_NONFLARE = ((400, 2000), (1e5, 5e5))
OPT_POST = (5000, 25000)
STYLE = {  # same colours/markers as the light-curve figure
    "X-ray": ("black", "o"), "i-band": ("#e34948", "o"), "Leavitt_Rc": ("#eda100", "s"), "Leavitt_Ic": ("#1baf7a", "D"),
}
LABEL = {"i-band": "NUTTelA-TAO $i$", "Leavitt_Rc": "Leavitt $R_c$", "Leavitt_Ic": "Leavitt $I_c$"}


def measurements(av_override=None):
    xrt, opt = load_all_optical_data()
    ds = {d["name"]: d for d in opt}
    si = load_xrt_spectral_index()
    m = _sel(xrt["time"], XRAY_NONFLARE)
    rx = fit_alpha(xrt["time"][m], xrt["flux"][m], xrt["flux_error"][m])
    gm, ge = mean_gamma(si, XRAY_NONFLARE)[:2]
    pts = {"X-ray": dict(beta=gm - 1, beta_err=ge, alpha=rx["alpha"], alpha_err=rx["err"])}

    sdt = pm.sdt_points()
    av, src = (av_override, "override") if av_override is not None else pm.host_av()
    _, g7, g7e = pm.optical_photon_index(sdt, a_v=av)
    for key in ("i-band", "Leavitt_Rc", "Leavitt_Ic"):
        d = ds[key]; mm = _sel(d["time"], OPT_POST)
        r = fit_alpha(d["time"][mm], d["flux_mJy"][mm], d["flux_err"][mm])
        err = r["err"] * max(1.0, np.sqrt(r["chi2"] / r["dof"]))
        pts[key] = dict(beta=g7 - 1, beta_err=g7e, alpha=r["alpha"], alpha_err=err)
    return pts, av, src


def main(av_override=None):
    plt.rcParams.update({"font.size": 13, "legend.fontsize": 11})
    pts, av, src = measurements(av_override)
    ax = None   # cr_plot(ax=None) creates the figure and draws the relation lines; reuse it afterwards
    for key, v in pts.items():
        color, mk = STYLE[key]
        lab = "Swift/XRT (flare excl.)" if key == "X-ray" else f"{LABEL[key]} (post-hump)"
        cr_plot(**v, ax=ax, marker=mk, color=color, ms=9 if key == "X-ray" else 8,
                mec="black", mew=0.5, capsize=3, elinewidth=1.2, zorder=10, label=lab,
                show_cr_legend=ax is None, plot_equal_p=True)
        if ax is None:
            ax = plt.gca()
    fig = ax.figure
    fig.set_size_inches(9, 6.5)
    ax.set_xlabel(r"$\beta$   ($F_\nu \propto \nu^{-\beta}$)", fontsize=15)
    ax.set_ylabel(r"$\alpha$   ($F_\nu \propto t^{-\alpha}$)", fontsize=15)
    ax.set_xlim(0, 2.2); ax.set_ylim(0, 3.0)
    ax.set_xticks([0, 0.5, 1, 1.5, 2]); ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    # two legends: relation lines (outside right) and data points (inside)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    h, l = ax.get_legend_handles_labels()
    is_data = [("Swift" in t or "post-hump" in t) for t in l]
    ax.legend([hh for hh, d in zip(h, is_data) if d], [ll for ll, d in zip(l, is_data) if d],
              loc="upper left", frameon=True, framealpha=0.9,
              title=rf"optical $\beta$: 7DT, host $A_V$={av:.2f} (SMC)", title_fontsize=10)
    data_leg = ax.get_legend()
    ax.add_artist(data_leg)
    ax.legend([hh for hh, d in zip(h, is_data) if not d], [ll for ll, d in zip(l, is_data) if not d],
              loc="lower right", frameon=True, framealpha=0.9, title="closure relations",
              title_fontsize=12, fontsize=11)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.11)
    ax.set_title("GRB 251013C closure relations", fontsize=15)
    (ROOT / "figures").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"closure_relation.{ext}", dpi=200, bbox_inches="tight")
    print("wrote figures/closure_relation.png  (A_V =", f"{av:.3f}, {src})")
    for k, v in pts.items():
        print(f"  {k:12s} beta = {v['beta']:.2f} ± {v['beta_err']:.2f}   alpha = {v['alpha']:.2f} ± {v['alpha_err']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--av", type=float, default=None)
    main(ap.parse_args().av)
