"""Is the incumbent wing's late nosedive an ARCHITECTURAL property or a p_wing RAIL artifact?
Scan p_wing x E_iso_wing around the incumbent; report fitted-data chi2 AND predicted r-band
mag at 2e5 / 5e5 s vs the parsed compilation (SN-safe window, t < 7e5 s)."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np
WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0, WT); os.chdir(WT)
from VegasAfterglow import Scale
from grb.params import make_param_defs
from grb.modeling import load_all_optical_data
from grb.likelihood import compute_model_flux_all_bands
from grb.spectral_index import load_xrt_spectral_index
from grb.plotting import compute_model_components
from grb.extinction import galactic_extinction

xrt_data, optical_datasets = load_all_optical_data()
xrt_index_data = load_xrt_spectral_index()
pdefs = make_param_defs(True, True)
names = [p.name for p in pdefs]
FR = "/data/dtak/research/grb/GRB251013C/modeling/fit_results"
th0 = np.load(f"{FR}/final_flare_wing_20260802_131026/top_k_params.npy")[0].copy()

C_AA = 2.99792458e18
WL_R = 6215.0
NU_R = C_AA/WL_R
A_R = float(galactic_extinction(np.array([WL_R]))[0])

# compilation r-band reference points (parsed, plotted mag == apparent since r offset = 0)
d = np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/sample_parsed.npz")
rt, rm = d["r_t"], d["r_m"]
def obs_r(t0, frac=0.25):
    m = (rt > t0*(1-frac)) & (rt < t0*(1+frac))
    return (np.median(rm[m]), m.sum()) if m.sum() else (np.nan, 0)
REF = [(1.0e5, *obs_r(1.0e5)), (2.0e5, *obs_r(2.0e5)), (5.0e5, *obs_r(5.0e5))]
print("compilation r-band reference (median plotted AB mag, SN-safe t<7e5 s):")
for t, mm, nn in REF:
    print(f"  t={t:.0e}  r={mm:.2f}  (n={nn})")

def score(th):
    params = {p.name: (10**v if p.scale is Scale.LOG else v) for p, v in zip(pdefs, th)}
    xm, om, si = compute_model_flux_all_bands(params, xrt_data, optical_datasets, True, True, xrt_index_data)
    xc = np.sum(((xrt_data['flux']-xm)/xrt_data['flux_error'])**2)
    oc = sum(np.sum(((dd['flux_mJy']-m)/dd['flux_err'])**2) for dd, m in zip(optical_datasets, om))
    tg = np.array([t for t, _, _ in REF])
    comp = compute_model_components(params, tg, NU_R, None, True, True)
    mags = -2.5*np.log10(comp['total']) + 16.4 + A_R
    return xc, oc, si, mags

ip, iE = names.index("p_wing"), names.index("E_iso_wing")
print(f"\nincumbent: p_wing={th0[ip]:.3f} (box top 3.3), log10 E_iso_wing={th0[iE]:.3f}")
xc, oc, si, mg = score(th0)
print(f"  baseline  XRT={xc:.1f} opt={oc:.1f} SI={si:.1f}  "
      f"r(1e5)={mg[0]:.2f} r(2e5)={mg[1]:.2f} r(5e5)={mg[2]:.2f}")

print(f"\nscan (everything else frozen at incumbent):")
print(f"{'p_wing':>7}{'dlogE_w':>9}{'XRTchi2':>9}{'optchi2':>9}{'SIchi2':>8}"
      f"{'r(1e5)':>9}{'r(2e5)':>9}{'r(5e5)':>9}{'totchi2':>10}")
rows = []
for pw in (2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.285):
    for dlE in (-0.3, 0.0, 0.3, 0.6):
        th = th0.copy(); th[ip] = pw; th[iE] = th0[iE] + dlE
        if not (pdefs[iE].lower <= 10**th[iE] <= pdefs[iE].upper):
            continue
        xc, oc, si, mg = score(th)
        rows.append((pw, dlE, xc, oc, si, mg, xc+oc+si))
        print(f"{pw:>7.2f}{dlE:>9.2f}{xc:>9.1f}{oc:>9.1f}{si:>8.1f}"
              f"{mg[0]:>9.2f}{mg[1]:>9.2f}{mg[2]:>9.2f}{xc+oc+si:>10.1f}")

print("\nobserved (compilation): " + "  ".join(f"r({t:.0e})={m:.2f}" for t, m, _ in REF))
best = min(rows, key=lambda r: r[6])
print(f"\nbest in scan: p_wing={best[0]:.2f} dlogE={best[1]:+.2f} totchi2={best[6]:.1f} "
      f"(incumbent totchi2={sum(score(th0)[:3]):.1f})")
ok = [r for r in rows if all(abs(r[5][i]-REF[i][1]) < 0.5 for i in range(3) if np.isfinite(REF[i][1]))]
print(f"\nwing configs matching ALL compilation refs to <0.5 mag: {len(ok)}")
for r in ok:
    print(f"  p_wing={r[0]:.2f} dlogE={r[1]:+.2f}  totchi2(fitted)={r[6]:.1f}  "
          f"r={r[5][0]:.2f}/{r[5][1]:.2f}/{r[5][2]:.2f}")
