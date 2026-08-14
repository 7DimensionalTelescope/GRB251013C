import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np, pandas as pd
WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0, WT); os.chdir(WT)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 500)

raw = pd.read_excel("data/circular.xlsx").sort_values("time")
print("total rows:", len(raw))
print("\n=== ALL rows, t > 3e4 s ===")
late = raw[raw['time'] > 3e4]
print(late[['time','facility','filter','magnitude','mag_error','Circular','upper_limit']].to_string())

print("\n=== counts by filter (non-UL) ===")
ok = raw[~raw['upper_limit'].astype(bool)]
print(ok.groupby('filter').agg(n=('time','size'), tmin=('time','min'), tmax=('time','max')).to_string())

print("\n=== rows in 1e4 < t < 3e4 (the 'hump' window), all facilities ===")
mid = raw[(raw['time'] > 1e4) & (raw['time'] <= 3e4)]
print(mid[['time','facility','filter','magnitude','mag_error','Circular','upper_limit']].to_string())

print("\n=== i / i' / r / r' rows t < 3e4 for cross-check with i_data ===")
ib = raw[(raw['filter'].isin(['i',"i'",'r',"r'",'z'])) & (raw['time'] < 3e4)]
print(ib[['time','facility','filter','magnitude','mag_error','Circular','upper_limit']].to_string())
