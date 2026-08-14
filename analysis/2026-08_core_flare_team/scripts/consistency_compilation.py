"""Claim 4: band-by-band residuals of FLARE-X vs incumbent against the parsed compilation."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np
WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0, WT); os.chdir(WT)

from VegasAfterglow import Scale
from grb.params import make_param_defs
from grb.plotting import compute_model_components
from grb.extinction import galactic_extinction

C_AA = 2.99792458e18
WAVE = {"VT/B": 4450, "green": 4770, "r": 6215, "R": 6580, "VT/R": 6580,
        "i": 7545, "I": 7545, "z": 8700, "y": 9620}
OFFSET = {"VT/B": +2, "r": 0, "R": 0, "VT/R": 0, "i": -1, "I": -1, "z": -2, "y": -3, "green": +1}

# --- model parameter dicts, each with ITS OWN fitted A_V ---
def to_params(theta, pdefs):
    return {p.name: (10**v if p.scale is Scale.LOG else v) for p, v in zip(pdefs, theta)}

bx = np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_flare_best.npy")
FLAREX = to_params(bx, make_param_defs(True, False))
RD = "/data/dtak/research/grb/GRB251013C/modeling/fit_results/final_flare_wing_20260802_131026"
inc = np.load(os.path.join(RD, "top_k_params.npy"))[0]
INCUMB = to_params(inc, make_param_defs(True, True))
print(f"FLARE-X A_V = {FLAREX['A_V']:.4f}   INCUMBENT A_V = {INCUMB['A_V']:.4f}")
print(f"FLARE-X has E_iso_wing: {'E_iso_wing' in FLAREX}   "
      f"INCUMBENT has E_iso_wing: {'E_iso_wing' in INCUMB}")

MODELS = [("FLARE-X", FLAREX, False), ("INCUMBENT", INCUMB, True)]

tgrid = np.geomspace(1e4, 1.2e6, 200)
A_gal = {}
model_mag = {m[0]: {} for m in MODELS}
for band, wl in WAVE.items():
    nu = C_AA / wl
    A_gal[band] = float(galactic_extinction(np.array([wl]))[0])
    for tag, prm, iw in MODELS:
        comp = compute_model_components(prm, tgrid, nu, None, True, iw)
        F = comp['total']                                  # mJy, host-extincted
        mag = -2.5 * np.log10(F) + 16.4 + A_gal[band]      # APPARENT AB
        model_mag[tag][band] = mag + OFFSET[band]          # plotted convention
print("\ngalactic extinction used:", {k: round(v, 4) for k, v in A_gal.items()})
print("(brief quoted A_r=0.135 A_i=0.100 A_z=0.079)")

# --- parsed trails ---
npz = np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/sample_parsed.npz")
def binned(t, mag, w=0.04):
    lt = np.log10(t); out = []
    for c in np.arange(2.0, 6.4, w):
        s = (lt >= c) & (lt < c + w)
        if s.sum() >= 3: out.append((10**(c + w / 2), np.median(mag[s])))
    return np.array(out).T if out else np.zeros((2, 0))

WINDOWS = [(2e4, 6e4), (1e5, 4e5), (4e5, 7e5)]
BANDS = ["r", "VT/R", "R", "i", "z", "green", "VT/B", "y"]

print("\n=== residual median (model - parsed), mag; POSITIVE = model too FAINT ===")
hdr = f"{'band':7s}" + "".join(f"{'FLARE-X':>10s}{'INCUMB':>10s}{'nbin':>6s}" for _ in WINDOWS)
print(f"{'':7s}" + "".join(f"{f'[{a:.0e},{b:.0e}]':^26s}" for a, b in WINDOWS))
print(hdr)
summary = {}
for band in BANDS:
    bt, bm = binned(npz[f"{band}_t"], npz[f"{band}_m"])
    row = f"{band:7s}"
    for a, b in WINDOWS:
        s = (bt >= a) & (bt < b)
        if s.sum() == 0:
            row += f"{'-':>10s}{'-':>10s}{0:>6d}"; continue
        vals = {}
        for tag, _, _ in MODELS:
            mm = np.interp(np.log10(bt[s]), np.log10(tgrid), model_mag[tag][band])
            vals[tag] = float(np.median(mm - bm[s]))
        summary[(band, a)] = (vals["FLARE-X"], vals["INCUMBENT"], int(s.sum()))
        row += f"{vals['FLARE-X']:>10.2f}{vals['INCUMBENT']:>10.2f}{s.sum():>6d}"
    print(row)

print("\n=== which model is CLOSER (|residual|)?  per band x window ===")
for a, b in WINDOWS:
    fx_win = inc_win = 0
    diffs = []
    for band in BANDS:
        if (band, a) not in summary: continue
        f_, i_, n = summary[(band, a)]
        diffs.append((band, abs(f_), abs(i_)))
        if abs(f_) < abs(i_): fx_win += 1
        else: inc_win += 1
    print(f"  window [{a:.0e},{b:.0e}]: FLARE-X closer in {fx_win} bands, INCUMBENT in {inc_win}")
    for band, af, ai in diffs:
        print(f"      {band:6s} |FLARE-X|={af:5.2f}  |INCUMB|={ai:5.2f}  -> "
              f"{'FLARE-X' if af < ai else 'INCUMBENT'}")

# --- aggregate RMS over the r/i/z trails, t in [2e4, 7e5] ---
print("\n=== aggregate over r,VT/R,R,i,z bins with 2e4 < t < 7e5 ===")
allres = {tag: [] for tag, _, _ in MODELS}
for band in ("r", "VT/R", "R", "i", "z"):
    bt, bm = binned(npz[f"{band}_t"], npz[f"{band}_m"])
    s = (bt >= 2e4) & (bt < 7e5)
    for tag, _, _ in MODELS:
        mm = np.interp(np.log10(bt[s]), np.log10(tgrid), model_mag[tag][band])
        allres[tag].extend((mm - bm[s]).tolist())
for tag in allres:
    a = np.array(allres[tag])
    print(f"  {tag:10s} n={len(a):4d} median={np.median(a):+6.2f} "
          f"mean={a.mean():+6.2f} rms={np.sqrt((a**2).mean()):5.2f} "
          f"median|res|={np.median(np.abs(a)):5.2f}")

np.savez("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/consistency_compilation.npz",
         tgrid=tgrid, **{f"{tag}_{b.replace('/','')}": model_mag[tag][b]
                         for tag, _, _ in MODELS for b in WAVE})
