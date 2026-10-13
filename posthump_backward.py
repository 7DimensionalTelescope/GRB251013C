"""Post-hump i-band decay fitted from the LAST point backwards.

Starting from the latest NUTTelA-TAO i point (after the 3.4 ks hump peak),
add earlier points one at a time and refit a power law; the alpha / chi2
profile shows where a single power law stops holding. Leavitt Rc/Ic and the
7DT r,i points are drawn for comparison.

Run from the repo root:
    /home/dtak/miniconda3/envs/grb251013c/bin/python posthump_backward.py
Writes figures/posthump_backward.png.
"""
import numpy as np
import matplotlib.pyplot as plt

from grb.modeling import load_all_optical_data
from lc_indices import fit_alpha

HUMP_PEAK = 3400   # s; only i points after this enter the backward fit


def main():
    _, opt = load_all_optical_data(); ds = {d["name"]: d for d in opt}
    i = ds["i-band"]; m = i["time"] > HUMP_PEAK
    t, f, e = i["time"][m], i["flux_mJy"][m], i["flux_err"][m]
    o = np.argsort(t)[::-1]; t, f, e = t[o], f[o], e[o]
    prof = np.array([(t[n-1],) + tuple(fit_alpha(t[:n], f[:n], e[:n])[k] for k in ("alpha", "err", "chi2", "dof"))
                     for n in range(4, len(t) + 1)])
    print(f"{'t_start':>8s} {'n':>3s} {'alpha':>12s} {'chi2/dof':>10s}")
    for n, row in zip(range(4, len(t) + 1), prof):
        print(f"{row[0]:8.0f} {n:3d} {row[1]:6.2f} ± {row[2]:4.2f} {row[3]:6.1f}/{int(row[4])}")
    # last clean segment = longest backward run with chi2/dof < 1.5
    ok = prof[:, 3] / prof[:, 4] < 1.5
    n_clean = 4 + int(np.max(np.where(ok)[0])) if ok.any() else 4
    seg = fit_alpha(t[:n_clean], f[:n_clean], e[:n_clean])
    allr = fit_alpha(t, f, e)
    rc = ds["Leavitt_Rc"]; rrc = fit_alpha(rc["time"], rc["flux_mJy"], rc["flux_err"])
    print(f"\nlast clean segment: {n_clean} pts, {t[n_clean-1]:.0f}-{t[0]:.0f} s, alpha = {seg['alpha']:.2f} ± {seg['err']:.2f}")

    fig, (ax, axp) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True)
    ax.errorbar(i["time"], i["flux_mJy"], i["flux_err"], fmt="o", ms=5, color="#e34948", label="NUTTelA-TAO i")
    for k, c, mk in (("Leavitt_Rc", "#eda100", "s"), ("Leavitt_Ic", "#1baf7a", "D")):
        d = ds[k]; ax.errorbar(d["time"], d["flux_mJy"], d["flux_err"], fmt=mk, ms=5, color=c, mec="black", mew=0.3, label=k.replace("_", " "))
    for k in ("7DT_r", "7DT_i"):
        d = ds[k]; ax.errorbar(d["time"], d["flux_mJy"], d["flux_err"], fmt="*", ms=13, color="#2a78d6", mec="black", mew=0.4, label=k.replace("_", " "))
    tt = np.geomspace(t[n_clean-1] * 0.95, 25000, 50)
    ax.plot(tt, 10**seg["b"] * tt**(-seg["alpha"]), "k-", lw=2,
            label=rf"i, last {n_clean} pts {t[n_clean-1]/1e3:.1f}–{t[0]/1e3:.1f} ks: $\alpha$={seg['alpha']:.2f}±{seg['err']:.2f}")
    tt2 = np.geomspace(4500, 25000, 50)
    ax.plot(tt2, 10**allr["b"] * tt2**(-allr["alpha"]), "k--", lw=1.2, label=rf"i, all {t[-1]/1e3:.1f}–{t[0]/1e3:.1f} ks: $\alpha$={allr['alpha']:.2f}")
    tt3 = np.geomspace(5000, 25000, 50)
    ax.plot(tt3, 10**rrc["b"] * tt3**(-rrc["alpha"]), color="#eda100", lw=1.5, label=rf"Leavitt Rc: $\alpha$={rrc['alpha']:.2f}")
    ax.axvspan(HUMP_PEAK, t[n_clean-1], color="gray", alpha=0.12, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(2500, 3e4); ax.set_ylim(0.2, 12)
    ax.set_ylabel("Flux density [mJy]"); ax.grid(alpha=.2, which="both"); ax.legend(fontsize=9, loc="lower left")
    ax.set_title("Post-hump i-band: power law fitted from the LAST point backwards")
    axp.errorbar(prof[:, 0], prof[:, 1], prof[:, 2], fmt="o-", color="k", ms=4, label=r"$\alpha$ (all points from t_start to the last)")
    axp2 = axp.twinx(); axp2.plot(prof[:, 0], prof[:, 3] / prof[:, 4], "s--", color="#e34948", ms=4, label=r"$\chi^2$/dof")
    axp2.set_ylabel(r"$\chi^2$/dof", color="#e34948"); axp2.axhline(1, color="#e34948", lw=0.6, ls=":")
    axp.set_xlabel("t_start of the backward-cumulative fit [s]"); axp.set_ylabel(r"$\alpha$"); axp.grid(alpha=.2, which="both")
    axp.legend(loc="upper right", fontsize=9); axp2.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig("figures/posthump_backward.png", dpi=150); print("wrote figures/posthump_backward.png")


if __name__ == "__main__":
    main()
