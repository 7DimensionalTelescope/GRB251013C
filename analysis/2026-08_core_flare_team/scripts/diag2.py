import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import sys, numpy as np
WT="/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0,WT); os.chdir(WT)

from VegasAfterglow import Scale
from grb.params import make_param_defs
from grb.modeling import load_all_optical_data, make_core_model, make_wing_model
from grb.likelihood import (log_probability, log_likelihood, log_prior,
                            compute_model_flux_all_bands, spectral_index_model)
from grb.spectral_index import load_xrt_spectral_index
from grb.const import XRT_BAND, XRT_NU_LO, XRT_NU_HI
from grb.utils import model_array
from grb.functions import norris_flare

RUN="/data/dtak/research/grb/GRB251013C/modeling/fit_results/final_flare_wing_20260730_171914"
theta=np.load(os.path.join(RUN,"top_k_params.npy"))[0]
labels=open(os.path.join(RUN,"labels.txt")).read().split()

xrt_data, optical_datasets = load_all_optical_data()
xrt_index_data = load_xrt_spectral_index()
pdefs = make_param_defs(True, True)
assert [f"log10_{p.name}" if p.scale is Scale.LOG else p.name for p in pdefs]==labels

params={p.name:(10**v if p.scale is Scale.LOG else v) for p,v in zip(pdefs,theta)}

lp  = log_probability(theta,pdefs,xrt_data,optical_datasets,True,True,xrt_index_data)
print(f"log_probability = {lp:.2f}")

xrt_model, opt_models, si_chi2 = compute_model_flux_all_bands(
    params, xrt_data, optical_datasets, True, True, xrt_index_data)

xrt_chi2=np.sum(((xrt_data['flux']-xrt_model)/xrt_data['flux_error'])**2)
print(f"\nXRT chi2 = {xrt_chi2:.1f} over {len(xrt_model)} pts")
print(f"SI  chi2 = {si_chi2:.1f} over {len(xrt_index_data['time'])} pts (kept subset)")
opt_tot=0
print(f"{'dataset':<12}{'n':>4}{'chi2':>9}")
for d,m in zip(optical_datasets,opt_models):
    c=np.sum(((d['flux_mJy']-m)/d['flux_err'])**2); opt_tot+=c
    if c>10 or len(d['time'])>2: print(f"{d['name']:<12}{len(d['time']):>4}{c:>9.1f}")
print(f"{'ALL OPTICAL':<12}{sum(len(d['time']) for d in optical_datasets):>4}{opt_tot:>9.1f}")
print(f"TOTAL chi2 = {xrt_chi2+opt_tot+si_chi2:.1f}")

# worst XRT points
r=(xrt_data['flux']-xrt_model)/xrt_data['flux_error']
idx=np.argsort(-np.abs(r))[:8]
print("\nWorst XRT points (t[hr], data, model, resid_sigma):")
for i in sorted(idx):
    print(f"  {xrt_data['time'][i]/3600:8.2f} {xrt_data['flux'][i]:.3e} {xrt_model[i]:.3e} {r[i]:+6.2f}")

# late-time wing vs core decomposition + photon index
core=make_core_model(params); wing=make_wing_model(params)
t_late=np.array([38.0,115.0])*3600
fc=model_array(core.flux(t_late,XRT_BAND[0],XRT_BAND[1],10).total).copy()
fw=model_array(wing.flux(t_late,XRT_BAND[0],XRT_BAND[1],10).total)
print("\nLate XRT decomposition (t[hr], core, wing, wing_frac):")
for t,a,b in zip(t_late,fc,fw):
    print(f"  {t/3600:6.1f} {a:.3e} {b:.3e} {b/(a+b):.2f}")

beta_model, keep = spectral_index_model(core, wing, params, xrt_index_data['time'], True)
gamma_model = 1 - beta_model
gamma_obs = 1 - xrt_index_data['beta']
print("\nSI points (t[hr], Gamma_obs, Gamma_model(total core+wing), kept, resid_sig):")
err=np.where(beta_model>xrt_index_data['beta'],xrt_index_data['beta_err_high'],xrt_index_data['beta_err_low'])
res=(xrt_index_data['beta']-beta_model)/err
for i in range(len(gamma_obs)):
    t=xrt_index_data['time'][i]/3600
    if t>1.2 or i%6==0:
        print(f"  {t:8.2f} {gamma_obs[i]:6.3f} {gamma_model[i]:6.3f} {bool(keep[i])} {res[i]:+6.2f}")
print(f"\nSI chi2 kept-points check: {np.sum(res[keep]**2):.1f}")
