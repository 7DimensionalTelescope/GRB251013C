"""Statistical weight carried by each optical dataset, before and after the
proposed error floor -- and the resulting chi2/pt under the FLARE-X model."""
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
params = {d.name: (10 ** v if d.scale is Scale.LOG else v) for d, v in zip(defs, theta)}
xrt, opt = load_all_optical_data()
_, opt_f, _ = compute_model_flux_all_bands(params, xrt, opt, True, False, None)

LN = np.log(10) * 0.4
FLOOR = 0.05          # mag, added in quadrature to every optical dataset
ZP = {"Leavitt_Rc": 3080 / 3631, "Leavitt_Ic": 2550 / 3631}


def group(name):
    if name.startswith("7DT"):
        return "7DT (22 x 1pt)"
    return name


print(f"{'dataset':16s} {'n':>3} {'weight now':>11} {'share':>7} | "
      f"{'weight w/floor':>14} {'share':>7} | {'chi2/pt now':>11} {'floor':>7} {'ZP+floor':>9}")
agg = {}
for d, mf in zip(opt, opt_f):
    g = group(d['name'])
    rel = d['flux_err'] / d['flux_mJy']
    mag_err = rel / LN
    w_now = np.sum(1 / mag_err**2)
    mag_new = np.sqrt(mag_err**2 + FLOOR**2)
    w_new = np.sum(1 / mag_new**2)
    c_now = np.sum(((d['flux_mJy'] - mf) / d['flux_err'])**2)
    e_new = d['flux_mJy'] * mag_new * LN
    c_floor = np.sum(((d['flux_mJy'] - mf) / e_new)**2)
    s = ZP.get(d['name'], 1.0)
    c_zp = np.sum(((d['flux_mJy'] * s - mf) / (e_new * s))**2)
    a = agg.setdefault(g, [0, 0, 0, 0, 0, 0])
    a[0] += len(d['time']); a[1] += w_now; a[2] += w_new
    a[3] += c_now; a[4] += c_floor; a[5] += c_zp

W_now = sum(a[1] for a in agg.values())
W_new = sum(a[2] for a in agg.values())
for g, a in agg.items():
    n = a[0]
    print(f"{g:16s} {n:3d} {a[1]:11.0f} {100*a[1]/W_now:6.1f}% | {a[2]:14.0f} "
          f"{100*a[2]/W_new:6.1f}% | {a[3]/n:11.2f} {a[4]/n:7.2f} {a[5]/n:9.2f}")
print(f"\n{'TOTAL':16s} {sum(a[0] for a in agg.values()):3d} {W_now:11.0f}        | {W_new:14.0f}")
print(f"total optical chi2: now {sum(a[3] for a in agg.values()):.0f}, "
      f"with 0.05 mag floor {sum(a[4] for a in agg.values()):.0f}, "
      f"with ZP fix + floor {sum(a[5] for a in agg.values()):.0f}")
print("\nNOTE: the 'ZP+floor' column is NOT a fair test of the zero-point fix --")
print("FLARE-X was fitted to the uncorrected fluxes, so it necessarily looks")
print("worse against corrected data until it is refitted.  The zero-point")
print("evidence is data-vs-data (analyst_system2.py), not chi2-vs-model.")
