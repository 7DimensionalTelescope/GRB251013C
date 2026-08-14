"""Claim 2: re-score the incumbent (final_flare_wing_20260802_131026 top_k[0])."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np
WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0, WT); os.chdir(WT)

from VegasAfterglow import Scale
from grb.params import make_param_defs
from grb.modeling import load_all_optical_data
from grb.likelihood import log_prior, log_likelihood, compute_model_flux_all_bands
from grb.spectral_index import load_xrt_spectral_index

xrt_data, optical_datasets = load_all_optical_data()
xrt_index_data = load_xrt_spectral_index()

RD = "/data/dtak/research/grb/GRB251013C/modeling/fit_results/final_flare_wing_20260802_131026"
inc = np.load(os.path.join(RD, "top_k_params.npy"))[0]
inc_lp_stored = np.load(os.path.join(RD, "top_k_log_probs.npy"))
inc_labels = [l.strip() for l in open(os.path.join(RD, "labels.txt")) if l.strip()]
pd_full = make_param_defs(True, True)
full_labels = [f"log10_{p.name}" if p.scale is Scale.LOG else p.name for p in pd_full]
print("stored top_k logprobs:", inc_lp_stored)
print("labels match make_param_defs(True,True):", inc_labels == full_labels)


def split(theta, pdefs, include_wing):
    params = {pd.name: (10**v if pd.scale is Scale.LOG else v) for pd, v in zip(pdefs, theta)}
    xrt_model, opt_models, si_chi2 = compute_model_flux_all_bands(
        params, xrt_data, optical_datasets, True, include_wing, xrt_index_data)
    c_xrt = float(np.sum(((xrt_data['flux'] - xrt_model) / xrt_data['flux_error'])**2))
    per, c_opt = [], 0.0
    for d, m in zip(optical_datasets, opt_models):
        c = float(np.sum(((d['flux_mJy'] - m) / d['flux_err'])**2))
        per.append((d['name'], len(d['time']), c)); c_opt += c
    lp = log_prior(theta, pdefs)
    ll = log_likelihood(theta, pdefs, xrt_data, optical_datasets, True, include_wing, xrt_index_data)
    print(f"  chi2 XRT={c_xrt:.2f}  SI={si_chi2:.2f}  optical={c_opt:.2f}  sum={c_xrt+si_chi2+c_opt:.2f}")
    print(f"  log_prior={lp}  log_like={ll:.3f}  logP={lp+ll if np.isfinite(lp) else -np.inf}")
    for n, k, c in per:
        if k > 1 or c > 5:
            print(f"     {n:<20} n={k:<4d} chi2={c:9.2f}")
    return c_xrt, si_chi2, c_opt, lp + ll


print("\n=== INCUMBENT, include_wing=True (as intended) ===")
split(inc, pd_full, True)
print("\n=== INCUMBENT, include_wing=False (wing silently dropped) ===")
split(inc, pd_full, False)

# late-XRT residual sigmas for both models
def late_xrt(theta, pdefs, include_wing, tag):
    params = {pd.name: (10**v if pd.scale is Scale.LOG else v) for pd, v in zip(pdefs, theta)}
    xm, _, _ = compute_model_flux_all_bands(params, xrt_data, optical_datasets, True,
                                            include_wing, xrt_index_data)
    r = (xrt_data['flux'] - xm) / xrt_data['flux_error']
    idx = np.argsort(xrt_data['time'])[-4:]
    print(f"\n{tag} late XRT residual sigmas (data-model)/err:")
    for i in idx:
        print(f"   t={xrt_data['time'][i]:10.0f}s ({xrt_data['time'][i]/3600:6.2f} hr)  "
              f"F={xrt_data['flux'][i]:.3e}  model={xm[i]:.3e}  resid={r[i]:+.2f} sigma")

late_xrt(inc, pd_full, True, "INCUMBENT")
bx = np.load("/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/data/nowing_flare_best.npy")
late_xrt(bx, make_param_defs(True, False), False, "FLARE-X")
