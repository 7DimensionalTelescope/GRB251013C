"""Validate the pixel-parsed compilation against real photometry, and finish
the i_data vs compilation-i cross-check."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np, pandas as pd
WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0, WT); os.chdir(WT)
from grb.io import read_data
OUT = "/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data"

parsed = {b: pd.read_csv(f"{OUT}/analyst_late_{b}.csv") for b in ("r", "i", "z")}
circ_late = pd.read_csv(f"{OUT}/analyst_late_circular.csv")

print("=" * 74)
print("A) parsed compilation vs real circular photometry, same band & epoch")
print("=" * 74)
print(f"{'band':5s} {'t':>9} {'facility':22s} {'m_real':>7} {'m_parsed':>9} {'diff':>7}")
d = []
for _, r in circ_late.iterrows():
    band = {"r": "r", "r'": "r", "i": "i", "z": "z"}.get(str(r["filter"]))
    if band is None:
        continue
    p = parsed[band]
    if not (p.time_s.min() <= r.time_s <= p.time_s.max()):
        continue
    mp = np.interp(np.log10(r.time_s), np.log10(p.time_s), p.mag_AB_galcorr)
    diff = r.mag_AB_galcorr - mp
    d.append(diff)
    print(f"{band:5s} {r.time_s:9.0f} {r.facility[:22]:22s} {r.mag_AB_galcorr:7.2f} "
          f"{mp:9.2f} {diff:+7.2f}")
d = np.array(d)
print(f"\n  mean(real - parsed) = {d.mean():+.3f} mag, scatter {d.std(ddof=1):.3f}, n={len(d)}")
print("  -> this is the accuracy of the pixel extraction + offset assumptions")

print("\n" + "=" * 74)
print("B) i_data vs the parsed compilation i-trail (full overlap, 94-10725 s)")
print("=" * 74)
idat = read_data("i_data", correct_galactic_extinction=True, add_converted_flux=True)
pi = parsed["i"]
ti, mi = idat["time"].to_numpy(float), idat["magnitude"].to_numpy(float)
ov = (ti >= pi.time_s.min()) & (ti <= pi.time_s.max())
mp = np.interp(np.log10(ti[ov]), np.log10(pi.time_s), pi.mag_AB_galcorr)
diff = mi[ov] - mp
print(f"n={ov.sum()} i_data points inside the parsed i-trail")
print(f"  mean(i_data - compilation_i) = {diff.mean():+.3f} mag "
      f"+- {diff.std(ddof=1)/np.sqrt(ov.sum()):.3f} (scatter {diff.std(ddof=1):.3f})")
for lo, hi in ((90, 500), (500, 2000), (2000, 6000), (6000, 11000)):
    s = (ti[ov] >= lo) & (ti[ov] < hi)
    if s.sum():
        print(f"    {lo:6d}-{hi:6d} s: n={s.sum():2d}  mean {diff[s].mean():+.3f} mag")

print("\n" + "=" * 74)
print("C) Leavitt Rc/Ic vs the parsed compilation r-trail")
print("=" * 74)
circ = read_data("circular", correct_galactic_extinction=True, add_converted_flux=True)
circ = circ[~circ["upper_limit"].astype(bool)]
pr = parsed["r"]
for filt, abcorr in (("Rc", 0.179), ("Ic", 0.384)):
    lea = circ[(circ.facility == "Leavitt") & (circ["filter"] == filt)]
    t, m = lea["time"].to_numpy(float), lea["magnitude"].to_numpy(float)
    s = (t >= pr.time_s.min()) & (t <= pr.time_s.max())
    if s.sum() == 0:
        continue
    mp = np.interp(np.log10(t[s]), np.log10(pr.time_s), pr.mag_AB_galcorr)
    print(f"  Leavitt {filt} vs compilation r ({s.sum()} pts in overlap):")
    print(f"    as-is (treated AB): mean(Leavitt - comp_r) = {(m[s]-mp).mean():+.3f} mag")
    print(f"    with Vega->AB +{abcorr:.3f}:  = {(m[s]+abcorr-mp).mean():+.3f} mag")

print("\n" + "=" * 74)
print("D) the late-time rebrightening, from real photometry only")
print("=" * 74)
rb = circ_late[circ_late["filter"].isin(["r", "r'", "Rc", "R"])].sort_values("time_s")
print(rb[["time_s", "facility", "filter", "mag_AB_galcorr", "mag_err", "flux_mJy"]].to_string(index=False))
