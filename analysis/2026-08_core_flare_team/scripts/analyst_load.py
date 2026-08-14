"""Step 1: dump the fitted datasets + raw circular/i_data frames for inspection."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys
import numpy as np
import pandas as pd

WT = "/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor"
sys.path.insert(0, WT); os.chdir(WT)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)

from grb.modeling import load_all_optical_data
from grb.io import read_data, filter_data

xrt, opt = load_all_optical_data()
print("XRT:", len(xrt['time']), "pts", xrt['time'].min(), xrt['time'].max())
for d in opt:
    print(f"{d['name']:14s} n={len(d['time']):3d} nu={d['frequency']:.4e} "
          f"t=[{d['time'].min():.1f},{d['time'].max():.1f}] "
          f"relerr med={np.median(d['flux_err']/d['flux_mJy']):.4f}")

print("\n=== raw circular (Leavitt rows) ===")
circ = read_data("circular", correct_galactic_extinction=True, add_converted_flux=True)
print("columns:", list(circ.columns))
print("facilities:", circ['facility'].unique())
print("filters:", circ['filter'].unique())
lea = circ[circ['facility'] == 'Leavitt']
print(lea[['time','filter','magnitude','mag_error','wavelength','gal_extinction',
           'flux_mJy','flux_mJy_error','upper_limit']].to_string())

print("\n=== i_data ===")
idat = read_data("i_data", correct_galactic_extinction=True, add_converted_flux=True)
print(idat.to_string())

print("\n=== sdt ===")
sdt = read_data("sdt", correct_galactic_extinction=True, add_converted_flux=True)
print("columns:", list(sdt.columns))
print(sdt.head(30).to_string())
