import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np, pandas as pd
WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0, WT); os.chdir(WT)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 400); pd.set_option("display.max_colwidth", 300)

raw = pd.read_excel("data/circular.xlsx")
print("RAW columns:", list(raw.columns))
lea = raw[raw['facility'] == 'Leavitt']
print("\nLeavitt circular ids:", lea['Circular'].unique())
print("\nAll unique Circular entries mentioning Leavitt-like text:")
for c in sorted(raw['Circular'].astype(str).unique())[:80]:
    print("   ", c)

print("\n--- rows with filter Rc or Ic (any facility) ---")
sub = raw[raw['filter'].isin(['Rc','Ic','R','I'])]
print(sub[['time','facility','filter','magnitude','mag_error','Circular','upper_limit']].to_string())
