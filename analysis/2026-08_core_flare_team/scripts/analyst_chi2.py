"""Decompose the Leavitt chi2 against the FLARE-X model, and test what the
zero-point correction and error inflation each do to it."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np
WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0, WT); os.chdir(WT)
from VegasAfterglow import Scale
from grb.modeling import load_all_optical_data
from grb.params import make_param_defs
from grb.likelihood import compute_model_flux_all_bands

theta = np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_flare_best.npy")
defs = make_param_defs(True, False)
print("ndim", len(theta), "defs", len(defs))
params = {}
for d, v in zip(defs, theta):
    params[d.name] = 10 ** v if d.scale is Scale.LOG else v
for k, v in params.items():
    print(f"  {k:18s} {v:.6g}")

xrt, opt = load_all_optical_data()
xrt_f, opt_f, si = compute_model_flux_all_bands(params, xrt, opt, True, False, None)

print(f"\nXRT chi2 = {np.sum(((xrt['flux']-xrt_f)/xrt['flux_error'])**2):.1f} ({len(xrt['time'])} pts)")

ZP_AB, ZP_RC, ZP_IC = 3631.0, 3080.0, 2550.0
VEGA_SCALE = {"Leavitt_Rc": ZP_RC / ZP_AB, "Leavitt_Ic": ZP_IC / ZP_AB}

print(f"\n{'dataset':14s} {'n':>3} {'chi2':>9} {'chi2/pt':>8} "
      f"{'mean resid (mag)':>17} {'scatter(mag)':>13}")
tot = 0.0
for d, mf in zip(opt, opt_f):
    r = (d['flux_mJy'] - mf) / d['flux_err']
    c = np.sum(r**2); tot += c
    dm = -2.5 * np.log10(d['flux_mJy'] / mf)   # data minus model in mag (negative = data brighter)
    print(f"{d['name']:14s} {len(r):3d} {c:9.1f} {c/len(r):8.2f} "
          f"{dm.mean():+17.3f} {dm.std(ddof=1) if len(dm)>1 else 0:13.3f}")
print(f"{'TOTAL optical':14s} {sum(len(d['time']) for d in opt):3d} {tot:9.1f}")

print("\n" + "=" * 78)
print("Leavitt chi2 decomposition and the effect of the two corrections")
print("=" * 78)
for d, mf in zip(opt, opt_f):
    if not d['name'].startswith("Leavitt"):
        continue
    n = len(d['time'])
    f, e = d['flux_mJy'], d['flux_err']
    s = VEGA_SCALE[d['name']]
    print(f"\n--- {d['name']} (n={n}, Vega/AB flux scale = {s:.3f}) ---")
    rows = [
        ("as fitted now                    ", f, e),
        ("Vega ZP fix only                 ", f * s, e * s),
        ("error inflation only (see below) ", f, None),
        ("Vega ZP fix + error inflation    ", f * s, None),
    ]
    # inflation: add the measured extra scatter in quadrature, in magnitudes
    extra_mag = {"Leavitt_Rc": 0.0481, "Leavitt_Ic": 0.0632}[d['name']]
    for label, ff, ee in rows:
        if ee is None:
            rel = ee_rel = np.sqrt((e / f) ** 2 + (extra_mag * np.log(10) * 0.4) ** 2)
            ee = ff * rel
        c = np.sum(((ff - mf) / ee) ** 2)
        dm = -2.5 * np.log10(ff / mf)
        print(f"  {label} chi2={c:8.1f}  chi2/pt={c/n:6.2f}  "
              f"mean(data-model)={dm.mean():+.3f} mag  scatter={dm.std(ddof=1):.3f}")

# how much of the chi2 is a pure offset vs scatter?
print("\n" + "=" * 78)
print("offset vs scatter split (current data/errors, FLARE-X model)")
print("=" * 78)
for d, mf in zip(opt, opt_f):
    if not d['name'].startswith("Leavitt") and d['name'] != 'i-band':
        continue
    r = (d['flux_mJy'] - mf) / d['flux_err']
    n = len(r)
    c = np.sum(r**2)
    # best single multiplicative rescale of the model
    k = np.sum(d['flux_mJy'] * mf / d['flux_err']**2) / np.sum(mf**2 / d['flux_err']**2)
    c_after = np.sum(((d['flux_mJy'] - k * mf) / d['flux_err'])**2)
    print(f"{d['name']:14s} n={n:3d} chi2={c:8.1f} -> after free normalisation "
          f"k={k:.3f} ({-2.5*np.log10(k):+.3f} mag): chi2={c_after:7.1f} "
          f"({100*(c-c_after)/c:.0f}% of the chi2 was a pure offset)")
