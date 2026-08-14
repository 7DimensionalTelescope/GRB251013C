import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
import sys, multiprocessing as mp
sys.path.insert(0, "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor")
os.chdir("/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor")
import numpy as np

TH0 = np.array([52.03341845,2.1319984,-0.11918524,2.1262565,2.12092578,-1.471047,-1.52301058,
                0.31155838,1.66720046,2.77048111,-0.96510305,-0.27741778,0.88184081,-1.44551679,
                3.31945804,1.9939548,3.87196744,-9.32615794,0.68324872])

def _work(vec, q):
    try:
        from grb.modeling import load_all_optical_data
        from grb.spectral_index import load_xrt_spectral_index
        from grb.likelihood import log_likelihood
        from grb.params import make_param_defs
        xrt, opt = load_all_optical_data(); si = load_xrt_spectral_index()
        q.put(float(log_likelihood(np.asarray(vec), make_param_defs(True, False), xrt, opt, True, False, si)))
    except BaseException as e:
        q.put(float("nan"))

def LL(vec):
    q = mp.Queue(); p = mp.Process(target=_work, args=(list(vec), q)); p.start(); p.join(180)
    if p.exitcode != 0 or q.empty():
        p.terminate(); return float("nan")
    return q.get()

if __name__ == "__main__":
    from grb.params import make_param_defs
    defs = make_param_defs(True, False); names = [d.name for d in defs]
    base = LL(TH0); print("base logL = %.2f" % base, flush=True)
    scans = {"eps_B_r":[0.05,0.1,0.2,0.3,0.4,0.528,0.6,0.8],
             "Gamma0_core":[60,90,136,200,300,500],
             "n_ism":[10,30,60,134,300,600],
             "xi":[0.15,0.2,0.31,0.5,0.8,1.0],
             "eps_e":[0.02,0.034,0.06,0.1],
             "eps_B":[0.005,0.01,0.03,0.05],
             "p":[1.9,2.0,2.121,2.25],
             "tau":[10,20,46.5,80,150]}
    for nm, vals in scans.items():
        i = names.index(nm); islog = "LOG" in str(defs[i].scale).upper()
        out = []
        for v in vals:
            t = TH0.copy(); t[i] = np.log10(v) if islog else v
            d = LL(t) - base
            out.append("%g:%s" % (v, "CRASH" if not np.isfinite(d) else "%+.1f" % d))
        print("%-12s " % nm + "   ".join(out), flush=True)
