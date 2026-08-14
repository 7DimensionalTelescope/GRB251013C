import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[k]="1"
import sys
sys.path.insert(0,"/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor")
import numpy as np
from grb.modeling import make_core_model

BASE = dict(E_iso_core=1.08e52, Gamma0_core=136.0, theta_c_core=0.760, n_ism=134.0,
            p=2.121, eps_e=0.034, eps_B=0.030, xi=0.31155838, tau=46.5,
            p_r=2.7705, eps_e_r=0.1084, eps_B_r=0.528, xi_r=0.88184)
m = make_core_model(BASE)
d = m.details(10, 1e6)
print("details attrs:", [a for a in dir(d) if not a.startswith('_')])
for a in [a for a in dir(d) if not a.startswith('_')]:
    v = getattr(d,a)
    try:
        arr=np.asarray(v)
        print(f"  {a:20s} shape={arr.shape} dtype={arr.dtype}")
    except Exception as e:
        print(f"  {a:20s} {type(v)}")
