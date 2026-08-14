"""Global cross-calibration: put every circular measurement (and i_data, 7DT)
on a common scale using the densely-sampled Leavitt Rc curve as the reference
light curve, and a colour term with assumed spectral slope beta.

offset = m_obs - [ Rc_ref(t) - 2.5*beta*log10(lam_obs/lam_Rc) ]
(redder band is brighter for F_nu ~ nu^-beta, hence the minus sign.)

All magnitudes are GALACTIC-EXTINCTION-CORRECTED before comparison.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np, pandas as pd
WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0, WT); os.chdir(WT)
from grb.io import read_data
from grb.extinction import galactic_extinction

WL_RC = 6410.0
circ = read_data("circular", correct_galactic_extinction=True, add_converted_flux=True)
circ = circ[~circ["upper_limit"].astype(bool)]

lea_rc = circ[(circ["facility"] == "Leavitt") & (circ["filter"] == "Rc")].sort_values("time")
trc = lea_rc["time"].to_numpy(float)
mrc = lea_rc["magnitude"].to_numpy(float)   # galactic-corrected
TMIN, TMAX = trc.min(), trc.max()

# i_data on the same footing
idat = read_data("i_data", correct_galactic_extinction=True, add_converted_flux=True)
rows = []
for _, r in circ.iterrows():
    wl = float(r["wavelength"])
    if not np.isfinite(wl) or wl <= 0:
        continue
    if not (TMIN <= r["time"] <= TMAX):
        continue
    rows.append((r["facility"], str(r["filter"]), float(r["time"]),
                 float(r["magnitude"]), float(r["mag_error"]), wl))
for _, r in idat.iterrows():
    if TMIN <= r["time"] <= TMAX:
        rows.append(("i_data", "i", float(r["time"]), float(r["magnitude"]),
                     float(r["mag_error"]), float(r["wavelength"])))

df = pd.DataFrame(rows, columns=["facility", "filter", "time", "mag", "err", "wl"])
print(f"{len(df)} measurements with a defined wavelength inside the Leavitt Rc "
      f"window {TMIN:.0f}-{TMAX:.0f} s\n")

for beta in (0.8, 1.2, 1.53):
    ref = np.interp(np.log10(df["time"]), np.log10(trc), mrc)
    pred = ref - 2.5 * beta * np.log10(df["wl"] / WL_RC)
    df[f"off{beta}"] = df["mag"] - pred

g = df.groupby(["facility", "filter"])
print(f"{'facility':38s} {'filt':5s} {'n':>3}  " +
      "  ".join([f"off(b={b})" for b in (0.8, 1.2, 1.53)]) + "   scatter")
for (fac, filt), sub in g:
    line = f"{fac[:38]:38s} {filt:5s} {len(sub):3d}  "
    for b in (0.8, 1.2, 1.53):
        line += f"{sub[f'off{b}'].mean():+9.3f} "
    line += f"  {sub['off0.8'].std(ddof=1) if len(sub) > 1 else 0:.3f}"
    print(line)

print("\n" + "=" * 78)
print("SAME-BAND cross-facility checks (no colour term needed)")
print("=" * 78)


def match(fac_a, filt_a, fac_b, filt_b, tol_dex=0.02):
    a = df[(df.facility == fac_a) & (df["filter"] == filt_a)].sort_values("time")
    b = df[(df.facility == fac_b) & (df["filter"] == filt_b)].sort_values("time")
    if len(a) == 0 or len(b) == 0:
        return
    out = []
    for _, r in b.iterrows():
        if not (a.time.min() <= r.time <= a.time.max()):
            continue
        ma = np.interp(np.log10(r.time), np.log10(a.time), a.mag)
        out.append(r.mag - ma)
    if out:
        out = np.array(out)
        print(f"  {fac_b}/{filt_b} minus {fac_a}/{filt_a}: n={len(out)} "
              f"mean {out.mean():+.3f} +- {out.std(ddof=1)/np.sqrt(len(out)) if len(out)>1 else 0:.3f} mag")


for fac in ("SAORAS", "OsservatorioAstronomicoNastroVerde", "Kilonova-Catcher", "MarSEC"):
    match("Leavitt", "Rc", fac, "Rc")
match("Leavitt", "Ic", "i_data", "i")
match("Leavitt", "Ic", "INO340", "i")
match("i_data", "i", "INO340", "i")
match("Leavitt", "Rc", "NOT", "r")
match("Leavitt", "Rc", "INO340", "r")

print("\n" + "=" * 78)
print("The Ic-vs-i question in detail (nearly the same band: 7980 vs 7625 A)")
print("=" * 78)
ic = df[(df.facility == "Leavitt") & (df["filter"] == "Ic")].sort_values("time")
ib = df[(df.facility == "i_data")].sort_values("time")
ov = ib[(ib.time >= ic.time.min()) & (ib.time <= ic.time.max())]
print(f"i_data points inside the Leavitt Ic window: {len(ov)}")
for _, r in ov.iterrows():
    mi = np.interp(np.log10(r.time), np.log10(ic.time), ic.mag)
    print(f"  t={r.time:8.0f}  i_data={r.mag:.3f}  Leavitt_Ic(interp)={mi:.3f}  "
          f"diff(i-Ic)={r.mag-mi:+.3f}")
