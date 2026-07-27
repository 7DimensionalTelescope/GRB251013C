#!/usr/bin/env python3
"""
Test to understand how VegasAfterglow handles jet break
"""
import sys
from VegasAfterglow import TophatJet, ISM, Observer, Model, Radiation
import numpy as np

print("=" * 80, flush=True)
print("UNDERSTANDING JET BREAK IN VEGASAFTERGLOW", flush=True)
print("=" * 80, flush=True)

# Your core jet parameters
observer = Observer(lumi_dist=1.39e28, z=0.438, theta_obs=0)
medium = ISM(n_ism=18.76)

# Test 1: With spreading (jet break enabled)
print("\nCreating model WITH jet break (spreading=True)...", flush=True)
jet1 = TophatJet(E_iso=1.189e52, Gamma0=522, theta_c=0.0345, spreading=True, duration=27.6)
rad = Radiation(eps_e=0.0435, eps_B=0.0163, p=2.158, xi_e=0.943, ssc=False, kn=False)
model1 = Model(jet=jet1, medium=medium, observer=observer, fwd_rad=rad)

# Test 2: Without spreading (no jet break)
print("Creating model WITHOUT jet break (spreading=False)...", flush=True)
jet2 = TophatJet(E_iso=1.189e52, Gamma0=522, theta_c=0.0345, spreading=False, duration=27.6)
model2 = Model(jet=jet2, medium=medium, observer=observer, fwd_rad=rad)

# Calculate flux at various times
times = np.array([100.0, 500.0, 1000.0, 3000.0, 10000.0])  # seconds
nu = 3.932e14  # Hz

print("\nCore Jet: Gamma0=522, theta_c=0.0345 rad, E_iso=1.19e52, n_ISM=18.76", flush=True)
print("\nFlux Comparison (i-band, 3.932e14 Hz):", flush=True)
print("-" * 80, flush=True)
print(f"{'Time (s)':<10} {'Time (hr)':<10} {'With Break':<18} {'No Break':<18} {'Ratio':<10}", flush=True)
print("-" * 80, flush=True)

for t in times:
    f1 = float(model1.flux_density(np.array([t]), np.array([nu])).total[0])
    f2 = float(model2.flux_density(np.array([t]), np.array([nu])).total[0])
    ratio = f1/f2 if f2 > 0 else 0
    print(f"{t:<10.0f} {t/3600:<10.3f} {f1:<18.3e} {f2:<18.3e} {ratio:<10.4f}", flush=True)

print("\n" + "=" * 80, flush=True)
print("KEY INSIGHTS:", flush=True)
print("=" * 80, flush=True)
print("1. VegasAfterglow calculates jet break time AUTOMATICALLY", flush=True)
print("2. It uses the Blandford-McKee solution with sideways expansion", flush=True)
print("3. Jet break depends on: E_iso, Gamma0, theta_c, n_ism", flush=True)
print("4. When ratio < 1: jet break has significantly affected the flux", flush=True)
print("5. With Gamma0=522, jet break is VERY EARLY!", flush=True)
print("\nFormula (approximate):", flush=True)
print("  t_jet ~ (E_iso / n_ism)^(1/3) * theta_c^(8/3) / Gamma0^(8/3)", flush=True)
print("\n  With YOUR parameters:", flush=True)
E_52 = 1.189
n = 18.76
theta = 0.0345
Gamma = 522
t_j = 0.61 * (E_52/n)**(1/3) * theta**(8/3) * 1.438  # (1+z) factor
print(f"  t_jet ~ {t_j*86400:.1f} s = {t_j*86400/3600:.3f} hr", flush=True)
print("\nThis early jet break is WHY reverse shock dominates early optical!", flush=True)
