import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import sys, time
WT="/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0,WT); os.chdir(WT)
import matplotlib
matplotlib.use("Agg")
from grb.plotting import plot_light_curves, plot_spectral_index_comparison

RUN="/data/dtak/research/grb/GRB251013C/modeling/fit_results/final_flare_wing_20260730_171914"
t0=time.time()
plot_light_curves(RUN, band_draws=100)
print(f"light curves done in {time.time()-t0:.0f}s", flush=True)
plot_spectral_index_comparison(RUN)
print("spectral index done", flush=True)
