"""Rebuild the parsed late-time datasets from consistency_parsed_fixed.npz
(the corrected parse: the old z trail 2-6e4 s was anti-alias halo from the black
y markers).  Regenerates the main files, the SN sidecars, the t<2.5e5 fit cuts,
and re-validates against real circular photometry.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np, pandas as pd
WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0, WT); os.chdir(WT)
from grb.extinction import galactic_extinction
from grb.utils import filter_to_wavelength

OUT = "/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data"
C_AA, ZP_AB, SN_CUT, FIT_CUT = 2.99792458e18, 3631.0, 7e5, 2.5e5
OFFSET = {"r": 0.0, "i": -1.0, "z": -2.0}
PARSE_SIGMA = 0.22          # empirical accuracy, re-measured below

npz = np.load(f"{OUT}/consistency_parsed_fixed.npz")
old = np.load(f"{OUT}/sample_parsed.npz")

print("=" * 76)
print("corrected parse vs old parse")
print("=" * 76)
for b in ("r", "i", "z"):
    print(f"  {b}: old n={len(old[f'{b}_t']):5d} "
          f"m={old[f'{b}_m'].min():.2f}-{old[f'{b}_m'].max():.2f}   "
          f"new n={len(npz[f'{b}_t']):5d} "
          f"m={npz[f'{b}_m'].min():.2f}-{npz[f'{b}_m'].max():.2f}")

for band in ("r", "i", "z"):
    t, mplot = npz[f"{band}_t"], npz[f"{band}_m"]
    m_app = mplot - OFFSET[band]
    wl = filter_to_wavelength(band)
    a_gal = float(galactic_extinction(np.array([wl]))[0])
    nu = C_AA / wl
    lt = np.log10(t)
    recs = []
    for lo in np.arange(np.floor(lt.min() / 0.05) * 0.05, lt.max() + 0.05, 0.05):
        s = (lt >= lo) & (lt < lo + 0.05)
        if s.sum() < 3:
            continue
        med = np.median(m_app[s])
        scat = 1.4826 * np.median(np.abs(m_app[s] - med))
        err_mag = max(0.15, scat)
        mcorr = med - a_gal
        f = ZP_AB * 10 ** (-0.4 * mcorr) * 1e3
        recs.append((10 ** (lo + 0.025), f, f * np.log(10) * 0.4 * err_mag, nu, wl,
                     mcorr, err_mag, int(s.sum()),
                     f * np.log(10) * 0.4 * np.sqrt(err_mag**2 + PARSE_SIGMA**2)))
    d = pd.DataFrame(recs, columns=["time_s", "flux_mJy", "flux_err_mJy", "frequency_Hz",
                                    "wavelength_AA", "mag_AB_galcorr", "mag_err", "n_pix",
                                    "flux_err_mJy_conservative"])
    d[d.time_s <= SN_CUT].to_csv(f"{OUT}/analyst_late_{band}.csv", index=False)
    if (d.time_s > SN_CUT).any():
        d[d.time_s > SN_CUT].to_csv(f"{OUT}/analyst_late_{band}_SNcontaminated.csv", index=False)
    d[d.time_s < FIT_CUT].to_csv(f"{OUT}/analyst_late_{band}_fit.csv", index=False)
    print(f"\n  {band}: lam={wl:.0f} A A_gal={a_gal:.3f}  "
          f"{(d.time_s<=SN_CUT).sum()} kept / {(d.time_s>SN_CUT).sum()} SN / "
          f"{(d.time_s<FIT_CUT).sum()} in fit cut  "
          f"(t {d.time_s.min():.3g}-{d.time_s.max():.3g} s)")

# ---- re-validate against real photometry ----
print("\n" + "=" * 76)
print("re-validation: corrected parse vs real circular photometry")
print("=" * 76)
parsed = {b: pd.read_csv(f"{OUT}/analyst_late_{b}.csv") for b in ("r", "i", "z")}
cl = pd.read_csv(f"{OUT}/analyst_late_circular.csv")
d = []
for _, r in cl.iterrows():
    band = {"r": "r", "r'": "r", "i": "i", "z": "z"}.get(str(r["filter"]))
    if band is None:
        continue
    p = parsed[band]
    if not (p.time_s.min() <= r.time_s <= p.time_s.max()):
        continue
    mp = np.interp(np.log10(r.time_s), np.log10(p.time_s), p.mag_AB_galcorr)
    d.append(r.mag_AB_galcorr - mp)
    print(f"  {band:2s} t={r.time_s:8.0f} {r.facility[:22]:22s} real={r.mag_AB_galcorr:6.2f} "
          f"parsed={mp:6.2f} diff={r.mag_AB_galcorr-mp:+.2f}")
d = np.array(d)
print(f"\n  mean(real - parsed) = {d.mean():+.3f} mag, scatter {d.std(ddof=1):.3f}, n={len(d)}")
print(f"  (old parse gave +0.014 +- 0.219)")
