"""Apply the logic audit's t < 2.5e5 s cut for fitting use, and check where the
achromatic bump actually starts."""
import os, sys, numpy as np, pandas as pd
OUT = "/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data"
CUT = 2.5e5

for b in ("r", "i", "z"):
    d = pd.read_csv(f"{OUT}/analyst_late_{b}.csv")
    d[d.time_s < CUT].to_csv(f"{OUT}/analyst_late_{b}_fit.csv", index=False)
    print(f"analyst_late_{b}_fit.csv: {(d.time_s < CUT).sum()} of {len(d)} pts kept "
          f"(t < {CUT:.1e} s)")

c = pd.read_csv(f"{OUT}/analyst_late_circular.csv")
c[c.time_s < CUT].to_csv(f"{OUT}/analyst_late_circular_fit.csv", index=False)
print(f"analyst_late_circular_fit.csv: {(c.time_s < CUT).sum()} of {len(c)} pts kept")

print("\nWhere does the bump start?  r/r'/Rc/R, AB, galactic-corrected:")
rb = c[c["filter"].isin(["r", "r'", "Rc", "R"])].sort_values("time_s")
print(rb[["time_s", "facility", "filter", "mag_AB_galcorr", "mag_err"]].to_string(index=False))
# decline rate between consecutive points
t = rb.time_s.to_numpy(float); m = rb.mag_AB_galcorr.to_numpy(float)
print("\nlocal decay index alpha (F ~ t^-alpha) between consecutive r-like points:")
for i in range(len(t) - 1):
    al = (m[i+1] - m[i]) / (2.5 * np.log10(t[i+1] / t[i]))
    flag = "  <-- RISING" if al < 0 else ""
    print(f"  {t[i]:8.0f} -> {t[i+1]:8.0f} s : alpha = {al:+.2f}{flag}")
