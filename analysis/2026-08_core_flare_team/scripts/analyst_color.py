"""Is the Leavitt Rc/Ic flux conversion using the wrong zero point?

grb/utils._mag_to_flux_mJy hardcodes zero_point=3631 Jy (AB) for every band,
but const.FILTER_INFO declares Rc and Ic as system='Vega' with zero points
3080 and 2550 Jy.  Test: compare the Rc->Ic and Rc->i colours implied by the
Leavitt fluxes with the colours measured by the 7DT medium-band SED (a single
instrument, genuinely AB, 22 bands) and by the i_data series.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np, pandas as pd
WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0, WT); os.chdir(WT)
from grb.io import read_data

C_AA = 2.99792458e18
ZP_AB, ZP_RC, ZP_IC = 3631.0, 3080.0, 2550.0
WL_RC, WL_IC, WL_I = 6410.0, 7980.0, 7625.0

# ---- 7DT SED at ~23 ks: measure the observed optical slope ----
sdt = read_data("sdt", correct_galactic_extinction=True, add_converted_flux=True)
sdt = sdt[~sdt["is_upper_limit"].astype(bool)]
med = sdt[sdt["filter_name"].astype(str).str.startswith("m")].copy()
med["wl"] = med["wavelength"].astype(float)
med = med.sort_values("wl")
w = med["wl"].to_numpy(float)
f = med["flux_mJy"].to_numpy(float)
fe = med["flux_mJy_error"].to_numpy(float)
print("7DT medium-band SED at ~23 ks (galactic-corrected, AB):")
for a, b, c in zip(w, f, fe):
    print(f"  {a:6.0f} A   {b:.4f} +- {c:.4f} mJy")

# power-law fit F ~ nu^-beta over the red half (5500-8000 A), weighted
sel = (w >= 5500) & (w <= 8000)
x = np.log10(C_AA / w[sel])
y = np.log10(f[sel])
ye = fe[sel] / f[sel] / np.log(10)
Wt = 1 / ye**2
A = np.vstack([x, np.ones_like(x)]).T
cov = np.linalg.inv(A.T @ (A * Wt[:, None]))
beta_fit = cov @ (A.T @ (Wt * y))
beta = -beta_fit[0]
sig_beta = np.sqrt(cov[0, 0])
print(f"\nObserved (host-extincted) optical slope 5500-8000 A: "
      f"F_nu ~ nu^-beta, beta = {beta:.2f} +- {sig_beta:.2f}  (n={sel.sum()})")


def pred_ratio(wl_num, wl_den, b):
    """F(wl_num)/F(wl_den) for F_nu ~ nu^-b."""
    return (wl_num / wl_den) ** b


for name, b in (("7DT-measured", beta), ("beta=0.8", 0.8), ("beta=1.2", 1.2)):
    print(f"  {name:14s}: expected F(Ic)/F(Rc) = {pred_ratio(WL_IC, WL_RC, b):.3f}, "
          f"F(i)/F(Rc) = {pred_ratio(WL_I, WL_RC, b):.3f}")

# also read the ratio straight off the 7DT SED by interpolation
lw = np.log10(w); lf = np.log10(f)
def sed_at(wl):
    return 10 ** np.interp(np.log10(wl), lw, lf)
print(f"  direct 7DT interpolation: F(Ic)/F(Rc) = {sed_at(WL_IC)/sed_at(WL_RC):.3f}, "
      f"F(i)/F(Rc) = {sed_at(WL_I)/sed_at(WL_RC):.3f}")

# ---- Leavitt colours, AB vs Vega ----
print("\n" + "=" * 78)
print("Leavitt Rc->Ic colour under the two zero-point choices")
print("=" * 78)
circ = read_data("circular", correct_galactic_extinction=True, add_converted_flux=True)
lea = circ[circ["facility"] == "Leavitt"]
rc = lea[lea["filter"] == "Rc"].sort_values("time")
ic = lea[lea["filter"] == "Ic"].sort_values("time")
# gal-corrected magnitudes
trc, mrc = rc["time"].to_numpy(float), rc["magnitude"].to_numpy(float)
tic, mic = ic["time"].to_numpy(float), ic["magnitude"].to_numpy(float)
# interpolate Rc onto the Ic epochs (Ic all lie inside the Rc range except the last two)
inside = (tic >= trc.min()) & (tic <= trc.max())
mrc_i = np.interp(np.log10(tic[inside]), np.log10(trc), mrc)
color = mrc_i - mic[inside]           # Rc - Ic in the ORIGINAL magnitude system
print(f"n={inside.sum()} matched epochs, mean (Rc-Ic)_instrumental = "
      f"{color.mean():+.3f} +- {color.std(ddof=1)/np.sqrt(len(color)):.3f} mag")
ratio_ab = 10 ** (0.4 * color.mean())                       # both AB -> flux ratio Ic/Rc
ratio_vega = 10 ** (0.4 * color.mean()) * (ZP_IC / ZP_RC)   # Vega ZPs
print(f"  if BOTH treated as AB (what the code does): F(Ic)/F(Rc) = {ratio_ab:.3f}")
print(f"  if BOTH treated as Vega (what FILTER_INFO says): F(Ic)/F(Rc) = {ratio_vega:.3f}")
b_ab = np.log(ratio_ab) / np.log(WL_IC / WL_RC)
b_vg = np.log(ratio_vega) / np.log(WL_IC / WL_RC)
print(f"  implied beta: AB -> {b_ab:.2f}   Vega -> {b_vg:.2f}   "
      f"(7DT measures {beta:.2f} +- {sig_beta:.2f})")

# ---- Leavitt Rc vs i_data at matched epochs ----
print("\n" + "=" * 78)
print("Leavitt Rc vs i_data (AB, independent) in the overlap window")
print("=" * 78)
idat = read_data("i_data", correct_galactic_extinction=True, add_converted_flux=True)
ti, fi = idat["time"].to_numpy(float), idat["flux_mJy"].to_numpy(float)
ov = (ti >= trc.min()) & (ti <= trc.max())
print(f"overlap: {ov.sum()} i-band points between {ti[ov].min():.0f} and {ti[ov].max():.0f} s")
mrc_at_i = np.interp(np.log10(ti[ov]), np.log10(trc), mrc)
f_rc_ab = ZP_AB * 10 ** (-0.4 * mrc_at_i)
f_rc_vg = ZP_RC * 10 ** (-0.4 * mrc_at_i)
r_ab = fi[ov] / f_rc_ab
r_vg = fi[ov] / f_rc_vg
print(f"  observed F(i)/F(Rc):  AB ZP -> {np.mean(r_ab):.3f} +- {np.std(r_ab, ddof=1)/np.sqrt(ov.sum()):.3f}")
print(f"                        Vega ZP -> {np.mean(r_vg):.3f} +- {np.std(r_vg, ddof=1)/np.sqrt(ov.sum()):.3f}")
print(f"  expected from 7DT SED: {sed_at(WL_I)/sed_at(WL_RC):.3f}")
print(f"  implied beta: AB -> {np.log(np.mean(r_ab))/np.log(WL_I/WL_RC):.2f}   "
      f"Vega -> {np.log(np.mean(r_vg))/np.log(WL_I/WL_RC):.2f}")
