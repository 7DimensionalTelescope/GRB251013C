"""Claim 1+2: independently re-score FLARE-X and the incumbent, with per-term split."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np
WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0, WT); os.chdir(WT)

from VegasAfterglow import Scale
from grb.params import make_param_defs, ParamDefWithPrior
from grb.modeling import load_all_optical_data
from grb.likelihood import log_probability, log_prior, log_likelihood, compute_model_flux_all_bands
from grb.spectral_index import load_xrt_spectral_index

xrt_data, optical_datasets = load_all_optical_data()
xrt_index_data = load_xrt_spectral_index()

print("=== dataset inventory ===")
print(f"XRT pts: {len(xrt_data['time'])}  t=[{xrt_data['time'].min():.1f},{xrt_data['time'].max():.1f}]")
tot = 0
for i, d in enumerate(optical_datasets):
    tot += len(d['time'])
    print(f"  opt[{i:2d}] n={len(d['time']):3d} nu={d['frequency']:.3e} "
          f"t=[{d['time'].min():.0f},{d['time'].max():.0f}] label={d.get('label', d.get('name','?'))}")
print(f"optical total pts: {tot}")
print(f"SI pts: {len(xrt_index_data['time'])}")


def widened_pdefs():
    pd = make_param_defs(True, False)
    def widen(name, lo, hi):
        for i, p in enumerate(pd):
            if p.name == name:
                pd[i] = ParamDefWithPrior(name, lo, hi, p.scale, gaussian_prior=p.gaussian_prior)
    for k, (lo, hi) in {
        "E_iso_core": (1e51, 1e55), "Gamma0_core": (100, 2000), "theta_c_core": (0.02, 0.8),
        "n_ism": (0.01, 3000), "p": (2.01, 3.2), "eps_e": (0.001, 0.5), "eps_B": (1e-7, 0.3),
        "xi": (0.1, 1.0), "tau": (5, 100), "p_r": (2.0, 3.5), "eps_e_r": (0.005, 0.5),
        "eps_B_r": (1e-4, 0.6), "xi_r": (0.3, 1.0), "t_start_flare": (500, 2e4),
        "tau_rise_flare": (5, 5000), "tau_decay_flare": (500, 1e5),
        "A_flare": (1e-11, 2e-8), "flare_beta": (0.0, 2.0),
    }.items():
        widen(k, lo, hi)
    return pd


def split(theta, pdefs, include_wing):
    params = {}
    for pd, v in zip(pdefs, theta):
        params[pd.name] = 10**v if pd.scale is Scale.LOG else v
    xrt_model, opt_models, si_chi2 = compute_model_flux_all_bands(
        params, xrt_data, optical_datasets, True, include_wing, xrt_index_data)
    c_xrt = float(np.sum(((xrt_data['flux'] - xrt_model) / xrt_data['flux_error'])**2))
    c_opt = 0.0
    per = []
    for d, m in zip(optical_datasets, opt_models):
        c = float(np.sum(((d['flux_mJy'] - m) / d['flux_err'])**2))
        per.append((d.get('label', d.get('name', '?')), len(d['time']), c))
        c_opt += c
    lp = log_prior(theta, pdefs)
    ll = log_likelihood(theta, pdefs, xrt_data, optical_datasets, True, include_wing, xrt_index_data)
    return dict(xrt=c_xrt, si=float(si_chi2), opt=c_opt, per=per, prior=lp, ll=ll,
                logP=lp + ll if np.isfinite(lp) else -np.inf, params=params)


def report(tag, theta, pdefs, include_wing):
    s = split(theta, pdefs, include_wing)
    print(f"\n=== {tag} (include_wing={include_wing}) ===")
    print(f"  chi2 XRT      = {s['xrt']:.2f}")
    print(f"  chi2 SI       = {s['si']:.2f}")
    print(f"  chi2 optical  = {s['opt']:.2f}")
    print(f"  sum chi2      = {s['xrt']+s['si']+s['opt']:.2f}")
    print(f"  log_prior     = {s['prior']}")
    print(f"  log_like      = {s['ll']:.3f}")
    print(f"  logP          = {s['logP']}")
    for lab, n, c in s['per']:
        print(f"     opt {lab:<22} n={n:<4d} chi2={c:9.2f}  chi2/pt={c/max(n,1):7.2f}")
    return s


# ---------- FLARE-X ----------
bx = np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_flare_best.npy")
pd_def = make_param_defs(True, False)
pd_wid = widened_pdefs()
labels = [f"log10_{p.name}" if p.scale is Scale.LOG else p.name for p in pd_def]
print("\n=== FLARE-X vector vs DEFAULT make_param_defs(True,False) box ===")
nviol = 0
for p, v, lab in zip(pd_def, bx, labels):
    lo = np.log10(p.lower) if p.scale is Scale.LOG else p.lower
    hi = np.log10(p.upper) if p.scale is Scale.LOG else p.upper
    lin = 10**v if p.scale is Scale.LOG else v
    ok = lo <= v <= hi
    nviol += (not ok)
    print(f"  {lab:<24} {lin:>12.5g}   box[{(10**lo if p.scale is Scale.LOG else lo):.4g},"
          f"{(10**hi if p.scale is Scale.LOG else hi):.4g}]  {'OK' if ok else '*** OUTSIDE ***'}")
print(f"  --> {nviol} parameters outside the default box")

print(f"\nlog_probability(FLARE-X, DEFAULT pdefs) = "
      f"{log_probability(bx, pd_def, xrt_data, optical_datasets, True, False, xrt_index_data)}")
report("FLARE-X / WIDENED pdefs", bx, pd_wid, False)

# ---------- incumbent ----------
RD = "modeling/fit_results/final_flare_wing_20260802_131026"
inc = np.load(os.path.join(RD, "top_k_params.npy"))[0]
inc_labels = [l.strip() for l in open(os.path.join(RD, "labels.txt"))]
print(f"\nincumbent vector len={len(inc)}  labels={len(inc_labels)}")
pd_full = make_param_defs(True, True)
full_labels = [f"log10_{p.name}" if p.scale is Scale.LOG else p.name for p in pd_full]
print("labels match make_param_defs(True,True):", inc_labels == full_labels)
if inc_labels != full_labels:
    print(" file :", inc_labels)
    print(" code :", full_labels)
s_inc = report("INCUMBENT top_k[0]", inc, pd_full, True)
for lab, v in zip(inc_labels, inc):
    print(f"  {lab:<24} {v:>10.4f}")

np.save("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/consistency_incumbent.npy", inc)
