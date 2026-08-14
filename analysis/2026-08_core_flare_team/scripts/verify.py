"""Score a parameter vector under the current likelihood: total + per-dataset chi2."""
import os,sys,json
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
sys.path.insert(0,"/data/dtak/research/grb/GRB251013C")
sys.path.insert(0,"/data/dtak/research/grb/GRB251013C/modeling")
os.chdir("/data/dtak/research/grb/GRB251013C/modeling")
import numpy as np
import final_model as fm
from utils import load_xrt_spectral_index, HOST_AV_LOG10_MEAN, HOST_AV_LOG10_SIGMA

xrt_data,optical_datasets=fm.load_all_optical_data()
sid=load_xrt_spectral_index()
pdefs=fm.make_param_defs(True,True)
labels=[f"log10_{p.name}" if p.scale is fm.Scale.LOG else p.name for p in pdefs]
LOGP=[p.scale is fm.Scale.LOG for p in pdefs]
iav=labels.index("log10_A_V")

def breakdown(th,tag):
    params={}
    for pd,v in zip(pdefs,th): params[pd.name]=10**v if pd.scale is fm.Scale.LOG else v
    xm,om,si=fm.compute_model_flux_all_bands(params,xrt_data,optical_datasets,True,True,sid)
    rows=[]
    c_x=float(np.sum(((xrt_data['flux']-xm)/xrt_data['flux_error'])**2))
    rows.append(("XRT",len(xm),c_x))
    tot=c_x
    for ds,mf in zip(optical_datasets,om):
        c=float(np.sum(((ds['flux_mJy']-mf)/ds['flux_err'])**2)); tot+=c
        rows.append((ds.get('label','opt'),len(mf),c))
    tot+=si; rows.append(("XRT_spec_index",len(sid['time']),float(si)))
    ll=fm.log_likelihood(th,pdefs,xrt_data,optical_datasets,True,True,sid)
    avp=-0.5*((th[iav]-HOST_AV_LOG10_MEAN)/HOST_AV_LOG10_SIGMA)**2
    n=sum(r[1] for r in rows)
    print(f"\n===== {tag} =====")
    print(f"{'dataset':<20}{'N':>5}{'chi2':>12}{'chi2/N':>9}")
    for nm,k,c in rows:
        print(f"{nm:<20}{k:>5}{c:>12.1f}{c/max(k,1):>9.2f}")
    print(f"{'TOTAL':<20}{n:>5}{tot:>12.1f}{tot/n:>9.2f}")
    print(f"logL={ll:.2f}  logL+AVprior={ll+avp:.2f}")
    return tot,ll+avp

if __name__=="__main__":
    cands=json.load(open(sys.argv[1]))  # {"tag":[theta,...]}
    res={}
    for tag,th in cands.items():
        res[tag]=breakdown(np.asarray(th,float),tag)
    print("\n===== SUMMARY (higher logL+prior is better) =====")
    for tag,(t,l) in sorted(res.items(),key=lambda x:-x[1][1]):
        print(f"{tag:<28} chi2={t:>10.1f}  logL+prior={l:>10.2f}")
