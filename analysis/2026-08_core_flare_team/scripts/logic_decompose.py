"""LOGIC audit: per-dataset chi2 decomposition of FLARE-X vs incumbent, under
IDENTICAL current data + current likelihood. Also error-inflation sensitivity."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np
WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0, WT); os.chdir(WT)
from VegasAfterglow import Scale
from grb.params import make_param_defs
from grb.modeling import load_all_optical_data
from grb.likelihood import log_probability, compute_model_flux_all_bands
from grb.spectral_index import load_xrt_spectral_index

xrt_data, optical_datasets = load_all_optical_data()
xrt_index_data = load_xrt_spectral_index()
FR = "/data/dtak/research/grb/GRB251013C/modeling/fit_results"

CAND = [
    ("FLARE-X (no wing)", np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_flare_best.npy"), False),
    ("incumbent (wing)",  np.load(f"{FR}/final_flare_wing_20260802_131026/top_k_params.npy")[0], True),
]

print("dataset inventory:")
for d in optical_datasets:
    print(f"  {d['name']:<16} n={len(d['time']):>4}  t=[{d['time'].min():.3g},{d['time'].max():.3g}] s")
print(f"  {'XRT':<16} n={len(xrt_data['time']):>4}  t=[{xrt_data['time'].min():.3g},{xrt_data['time'].max():.3g}] s")
print(f"  {'XRT photon idx':<16} n={len(xrt_index_data['time']):>4}")
print()

store = {}
for tag, th, wing in CAND:
    pdefs = make_param_defs(True, wing)
    assert len(th) == len(pdefs), (tag, len(th), len(pdefs))
    lp = log_probability(th, pdefs, xrt_data, optical_datasets, True, wing, xrt_index_data)
    params = {p.name: (10**v if p.scale is Scale.LOG else v) for p, v in zip(pdefs, th)}
    xm, om, si = compute_model_flux_all_bands(params, xrt_data, optical_datasets, True, wing, xrt_index_data)
    xc = np.sum(((xrt_data['flux'] - xm)/xrt_data['flux_error'])**2)
    per = {}
    for d, m in zip(optical_datasets, om):
        r = (d['flux_mJy'] - m)/d['flux_err']
        per[d['name']] = (np.sum(r**2), len(r), r, d['time'], m, d['flux_mJy'], d['flux_err'])
    store[tag] = dict(lp=lp, xc=xc, si=si, per=per, ndim=len(th))
    print(f"{tag:<20} ndim={len(th)}  logP={lp:>9.1f}  XRTchi2={xc:>7.1f}  SIchi2={si:>6.1f}  "
          f"opt_chi2={sum(v[0] for v in per.values()):>8.1f}")

print("\nper-optical-dataset chi2 (chi2, n, chi2/pt):")
labs = list(store[CAND[0][0]]['per'].keys())
hdr = f"{'dataset':<16}" + "".join(f"{t:>26}" for t, _, _ in CAND) + f"{'delta(FX-inc)':>16}"
print(hdr)
tot = {t: 0.0 for t, _, _ in CAND}
for L in labs:
    row = f"{L:<16}"
    vals = []
    for t, _, _ in CAND:
        c, n, *_ = store[t]['per'][L]
        vals.append(c); tot[t] += c
        row += f"{c:>12.1f} (n={n:>3}, {c/n:>5.1f})"
    row += f"{vals[0]-vals[1]:>16.1f}"
    print(row)
print(f"{'OPTICAL TOTAL':<16}" + "".join(f"{tot[t]:>26.1f}" for t, _, _ in CAND)
      + f"{tot[CAND[0][0]]-tot[CAND[1][0]]:>16.1f}")

fx, inc = CAND[0][0], CAND[1][0]
d_opt = tot[fx] - tot[inc]
d_xrt = store[fx]['xc'] - store[inc]['xc']
d_si  = store[fx]['si'] - store[inc]['si']
print(f"\nchi2 budget (FLARE-X minus incumbent; positive = FLARE-X worse):")
print(f"  optical {d_opt:+.1f}   XRT {d_xrt:+.1f}   SI {d_si:+.1f}   TOTAL {d_opt+d_xrt+d_si:+.1f}")
print(f"  -> Delta logP = {-(d_opt+d_xrt+d_si)/2:+.1f} (plus prior differences)")
print(f"  actual logP diff (FX - inc) = {store[fx]['lp']-store[inc]['lp']:+.1f}")

# --- error-inflation sensitivity: how much must OPTICAL errors be inflated for the
#     wing preference to fall to Delta chi2 = 4 (2 sigma-ish)?
print("\nerror-inflation sensitivity (uniform optical error scale f):")
for f in (1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
    print(f"  f={f:>4.1f}  optical Delta chi2 -> {d_opt/f**2:>8.1f}   "
          f"total Delta chi2 -> {d_opt/f**2 + d_xrt + d_si:>8.1f}  "
          f"(FLARE-X preferred if negative)")
fcrit = np.sqrt(d_opt/max(1e-9, -(d_xrt+d_si))) if (d_xrt+d_si) < 0 else np.nan
print(f"  break-even f (where XRT+SI gain offsets optical loss) = {fcrit:.2f}")

# --- residual structure: run test + lag-1 autocorrelation per dataset
print("\nresidual structure per dataset (is the misfit SCATTER or STRUCTURE?):")
print(f"{'dataset':<16}{'model':<20}{'rms_resid':>10}{'lag1_corr':>11}{'n_runs':>8}{'E[runs]':>9}{'z_runs':>8}{'mean_resid':>11}")
for L in labs:
    for t, _, _ in CAND:
        c, n, r, tt, m, fl, fe = store[t]['per'][L]
        o = np.argsort(tt); rr = r[o]
        s = np.sign(rr - 0.0)
        nr = 1 + np.sum(s[1:] != s[:-1])
        n1 = np.sum(s > 0); n2 = np.sum(s < 0)
        if n1 and n2:
            mu = 2*n1*n2/(n1+n2) + 1
            var = (mu-1)*(mu-2)/max(1, (n1+n2-1))
            z = (nr-mu)/np.sqrt(var) if var > 0 else np.nan
        else:
            mu, z = np.nan, np.nan
        lag1 = np.corrcoef(rr[:-1], rr[1:])[0, 1] if len(rr) > 3 else np.nan
        print(f"{L:<16}{t:<20}{np.sqrt(np.mean(rr**2)):>10.2f}{lag1:>11.2f}{nr:>8d}{mu:>9.1f}{z:>8.1f}{np.mean(rr):>11.2f}")

# --- how much of the optical delta comes from t in [8e3, 3e4] s (the "hump")?
print("\noptical Delta chi2 localized in time (FLARE-X minus incumbent):")
edges = [0, 2e3, 8e3, 3e4, 1e5]
for lo, hi in zip(edges[:-1], edges[1:]):
    dsum = 0.0; ntot = 0
    for L in labs:
        c1, n1, r1, t1, *_ = store[fx]['per'][L]
        c2, n2, r2, t2, *_ = store[inc]['per'][L]
        msk = (t1 >= lo) & (t1 < hi)
        dsum += np.sum(r1[msk]**2) - np.sum(r2[msk]**2); ntot += msk.sum()
    print(f"  t=[{lo:.0e},{hi:.0e}) n={ntot:>4}  Delta chi2 = {dsum:+9.1f}")

# --- can a per-dataset multiplicative calibration nuisance absorb the difference?
print("\nbest-fit per-dataset flux rescale (analytic) and residual chi2 after rescale:")
print(f"{'dataset':<16}{'model':<20}{'scale':>8}{'mag_off':>9}{'chi2_before':>13}{'chi2_after':>12}")
after = {t: 0.0 for t, _, _ in CAND}
for L in labs:
    for t, _, _ in CAND:
        c, n, r, tt, m, fl, fe = store[t]['per'][L]
        w = 1.0/fe**2
        a = np.sum(w*fl*m)/np.sum(w*m*m)          # scale model by a
        c2 = np.sum(w*(fl - a*m)**2)
        after[t] += c2
        print(f"{L:<16}{t:<20}{a:>8.3f}{-2.5*np.log10(a):>9.3f}{c:>13.1f}{c2:>12.1f}")
print(f"{'TOTAL after cal':<36}" + "".join(f"{after[t]:>12.1f}" for t, _, _ in CAND)
      + f"   Delta = {after[fx]-after[inc]:+.1f}")
