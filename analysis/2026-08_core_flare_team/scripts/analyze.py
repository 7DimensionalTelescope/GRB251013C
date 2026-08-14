import numpy as np, os, sys
base="/data/dtak/research/grb/GRB251013C/modeling/fit_results"
runs=["final_flare_wing_20260723_091033","final_flare_wing_20260723_141819"]


# bounds from final_model.py (sampled space)
B={
"log10_E_iso_core":(np.log10(5e51),np.log10(1e53)),
"log10_Gamma0_core":(np.log10(300),np.log10(1100)),
"log10_theta_c_core":(np.log10(0.001),np.log10(0.04)),
"log10_n_ism":(np.log10(5),np.log10(150)),
"p":(2.01,2.3),
"log10_eps_e":(np.log10(0.02),np.log10(0.1)),
"log10_eps_B":(np.log10(0.005),np.log10(0.05)),
"xi":(0.8,1.0),
"log10_tau":(np.log10(5),np.log10(30)),
"p_r":(2.0,3.0),
"log10_eps_e_r":(np.log10(0.02),np.log10(0.1)),
"log10_eps_B_r":(np.log10(0.005),np.log10(0.3)),
"xi_r":(0.7,1.0),
"log10_A_V":(np.log10(0.001),np.log10(2.0)),
"log10_t_start_flare":(np.log10(1000),np.log10(5000)),
"log10_tau_rise_flare":(np.log10(30),np.log10(2000)),
"log10_tau_decay_flare":(np.log10(1000),np.log10(10000)),
"log10_A_flare":(np.log10(1e-10),np.log10(5e-9)),
"flare_beta":(0.5,1.2),
"log10_E_iso_wing":(np.log10(1e52),np.log10(1e53)),
"log10_Gamma0_wing":(np.log10(10),np.log10(100)),
"log10_theta_c_wing":(np.log10(0.2),np.log10(0.5)),
"p_wing":(2.2,2.9),
"log10_eps_e_wing":(np.log10(0.3),np.log10(1.0)),
"log10_eps_B_wing":(np.log10(0.001),np.log10(0.02)),
"xi_wing":(0.6,1.0),
}
for r in runs:
    d=os.path.join(base,r)
    if not os.path.exists(os.path.join(d,"samples.npy")): continue
    s=np.load(os.path.join(d,"samples.npy"))
    lp=np.load(os.path.join(d,"log_probs.npy"))
    labels=open(os.path.join(d,"labels.txt")).read().split()
    print("="*100)
    print(f"{r}  samples shape={s.shape} logp shape={lp.shape}  best logp={lp.max():.1f}")
    # burn-in: last half
    n=s.shape[0]
    fl=s[n//2:]
    lpf=lp[n//2:]
    best=s[np.argmax(lp)]
    print(f"{'param':<24}{'lo':>9}{'hi':>9}{'best':>10}{'med':>10}{'p16':>10}{'p84':>10}  {'pos_in_range':>12}  {'%<5pct':>7}{'%>95pct':>8}")
    for i,lab in enumerate(labels):
        lo,hi=B[lab]
        v=fl[:,i]
        med=np.median(v); p16,p84=np.percentile(v,[16,84])
        frac=(med-lo)/(hi-lo)
        e_lo=100*np.mean(v<lo+0.05*(hi-lo)); e_hi=100*np.mean(v>hi-0.05*(hi-lo))
        flag=""
        if frac<0.10 or frac>0.90: flag=" <== RAIL"
        elif e_lo>25 or e_hi>25: flag=" <== edge"
        print(f"{lab:<24}{lo:>9.3f}{hi:>9.3f}{best[i]:>10.3f}{med:>10.3f}{p16:>10.3f}{p84:>10.3f}  {frac:>12.2f}  {e_lo:>7.1f}{e_hi:>8.1f}{flag}")
