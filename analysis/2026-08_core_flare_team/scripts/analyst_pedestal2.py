"""Follow-ups: (a) Monte-Carlo null for the lag-1 ACF of 3-point residuals,
(b) can ONE common factor fit both Leavitt bands, as-is vs zero-point-corrected,
(c) two-band consistency of the fitted pedestals."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np, pandas as pd
WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0, WT); os.chdir(WT)
from VegasAfterglow import Scale
from grb.modeling import load_all_optical_data
from grb.params import make_param_defs
from grb.likelihood import compute_model_flux_all_bands

rng = np.random.default_rng(7)
circ = pd.read_excel("data/circular.xlsx").sort_values("time")
circ = circ[~circ['upper_limit'].astype(bool)]


def three_point(t, m, e):
    lt = np.log10(t); res = []
    for i in range(1, len(t) - 1):
        w = (lt[i] - lt[i-1]) / (lt[i+1] - lt[i-1])
        res.append(m[i] - ((1 - w) * m[i-1] + w * m[i+1]))
    return np.array(res)


def lag1(x):
    x = x - x.mean()
    return float(np.sum(x[1:] * x[:-1]) / np.sum(x**2))


print("=" * 78)
print("(a) Monte-Carlo null: lag-1 ACF of 3-point residuals from WHITE noise,")
print("    using each dataset's actual time sampling and quoted errors")
print("=" * 78)
print(f"{'dataset':16s} {'observed':>9} {'null mean':>10} {'null 5-95%':>18} {'verdict':>28}")
for name, sel in (("Leavitt_Rc", ("Leavitt", "Rc")), ("Leavitt_Ic", ("Leavitt", "Ic"))):
    d = circ[(circ.facility == sel[0]) & (circ["filter"] == sel[1])].sort_values("time")
    t, m, e = d.time.to_numpy(float), d.magnitude.to_numpy(float), d.mag_error.to_numpy(float)
    obs = lag1(three_point(t, m, e))
    # smooth trend + white noise scaled to the OBSERVED excess scatter
    trend = np.poly1d(np.polyfit(np.log10(t), m, 3))(np.log10(t))
    sig = np.std(m - trend, ddof=4)
    null = np.array([lag1(three_point(t, trend + rng.normal(0, sig, len(t)), e))
                     for _ in range(4000)])
    lo, hi = np.percentile(null, [5, 95])
    ok = lo <= obs <= hi
    print(f"{name:16s} {obs:9.2f} {null.mean():10.2f} {f'[{lo:.2f}, {hi:.2f}]':>18} "
          f"{'consistent with white' if ok else 'CORRELATED':>28}")

print("\n" + "=" * 78)
print("(b) can ONE common factor fit BOTH Leavitt bands?")
print("=" * 78)
theta = np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_flare_best.npy")
defs = make_param_defs(True, False)
params = {d.name: (10 ** v if d.scale is Scale.LOG else v) for d, v in zip(defs, theta)}
xrt, opt = load_all_optical_data()
_, opt_f, _ = compute_model_flux_all_bands(params, xrt, opt, True, False, None)
ZP = {"Leavitt_Rc": 3080 / 3631, "Leavitt_Ic": 2550 / 3631}

for tag, apply_zp in (("data AS-IS (AB zero point)", False),
                      ("data with Vega->AB correction", True)):
    F, M, E = [], [], []
    per = {}
    for d, mf in zip(opt, opt_f):
        if not d['name'].startswith("Leavitt"):
            continue
        s = ZP[d['name']] if apply_zp else 1.0
        f, e = d['flux_mJy'] * s, d['flux_err'] * s
        F.append(f); M.append(mf); E.append(e)
        w = 1 / e**2
        per[d['name']] = np.sum(w * f * mf) / np.sum(w * mf**2)
    f = np.concatenate(F); mf = np.concatenate(M); e = np.concatenate(E)
    w = 1 / e**2
    k = np.sum(w * f * mf) / np.sum(w * mf**2)
    c_common = np.sum(w * (f - k * mf)**2)
    c_sep = 0.0
    for d, mfi in zip([d for d in opt if d['name'].startswith("Leavitt")],
                      [m for d, m in zip(opt, opt_f) if d['name'].startswith("Leavitt")]):
        s = ZP[d['name']] if apply_zp else 1.0
        ff, ee = d['flux_mJy'] * s, d['flux_err'] * s
        ww = 1 / ee**2
        c_sep += np.sum(ww * (ff - per[d['name']] * mfi)**2)
    print(f"\n{tag}:")
    print(f"  per-band factors: Rc k={per['Leavitt_Rc']:.3f} ({-2.5*np.log10(per['Leavitt_Rc']):+.3f} mag), "
          f"Ic k={per['Leavitt_Ic']:.3f} ({-2.5*np.log10(per['Leavitt_Ic']):+.3f} mag)")
    print(f"  band-to-band disagreement: "
          f"{abs(-2.5*np.log10(per['Leavitt_Rc']) + 2.5*np.log10(per['Leavitt_Ic'])):.3f} mag")
    print(f"  single common factor k={k:.3f}: chi2={c_common:.1f}; "
          f"two separate factors: chi2={c_sep:.1f}; cost of forcing one factor = {c_common-c_sep:.1f}")

print("\n" + "=" * 78)
print("(c) two-band consistency of the fitted additive pedestals")
print("=" * 78)
for d, mf in zip(opt, opt_f):
    if not d['name'].startswith("Leavitt"):
        continue
    f, e = d['flux_mJy'], d['flux_err']
    w = 1 / e**2
    C = np.sum(w * (f - mf)) / np.sum(w)
    sC = 1 / np.sqrt(np.sum(w))
    print(f"  {d['name']:12s} pedestal C = {C:+.4f} +- {sC:.4f} mJy   "
          f"({'POSITIVE' if C > 0 else 'NEGATIVE - unphysical for a host/blend'})")
print("\n  A host galaxy or blended star in the aperture contributes POSITIVE flux in")
print("  every band.  Rc and Ic were observed with the same telescope and aperture.")
