#!/usr/bin/env python3

from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

import early_phase as ep
from grb.prior import ParamDefWithPrior


# ============================================================
# Configuration
# ============================================================

RUN_DIR = Path(
    "/home/hongyp007/yp_folder/projects/GRB/result/"
    "early_phase_20261006_184042"
)

PRIOR = "final"
ENVIRONMENT = "wind"
SPECTRAL_INDEX_COOLING = "smooth"

# Test the current prior and a widened tau prior.
TAU_MAX_VALUES = [None, 3000.0]


# ============================================================
# Helpers
# ============================================================

def replace_tau_upper(param_defs, tau_max):
    """
    Return a new parameter-definition list with only the upper
    bound of tau changed.
    """
    new_defs = []

    for p in param_defs:
        if p.name != "tau":
            new_defs.append(p)
            continue

        gaussian_prior = None
        if isinstance(p, ParamDefWithPrior) and p.has_gaussian_prior():
            gaussian_prior = p.get_prior_mean_sigma()

        new_defs.append(
            ParamDefWithPrior(
                p.name,
                p.lower,
                tau_max,
                p.scale,
                gaussian_prior=gaussian_prior,
            )
        )

    return new_defs


def objective(x, param_defs, xrt_data, i_data, xrt_index_data):
    """
    Negative log posterior for scipy.optimize.minimize.
    """
    lp = ep.log_probability(
        x,
        param_defs,
        xrt_data,
        i_data,
        xrt_index_data,
        SPECTRAL_INDEX_COOLING,
    )

    if not np.isfinite(lp):
        return 1e30

    return -lp


def print_solution(title, result, param_defs):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)

    print(f"success  = {result.success}")
    print(f"message  = {result.message}")
    print(f"nfev     = {result.nfev}")
    print(f"log_prob = {-result.fun:.8f}")

    print()

    for p, value in zip(param_defs, result.x):
        if p.scale is ep.Scale.LOG:
            physical = 10.0 ** value
            print(
                f"{'log10_' + p.name:20s} "
                f"= {value: .8f}   "
                f"(physical = {physical:.8g})"
            )
        else:
            print(f"{p.name:20s} = {value: .8f}")


# ============================================================
# Load original result
# ============================================================

xrt_data, i_data, xrt_index_data = ep.load_fit_data(
    include_spectral_index=True
)

base_param_defs = ep.make_param_defs(
    PRIOR,
    ENVIRONMENT,
)

expected_labels = [
    ep.param_label(p) for p in base_param_defs
]

saved_labels = [
    line.strip()
    for line in (RUN_DIR / "labels.txt").read_text().splitlines()
    if line.strip()
]

if expected_labels != saved_labels:
    raise RuntimeError(
        "Saved labels do not match current parameter definitions.\n"
        f"Expected: {expected_labels}\n"
        f"Saved:    {saved_labels}"
    )

# Use the actual best sample from the existing run.
top_params = np.load(RUN_DIR / "top_k_params.npy")
top_log_probs = np.load(RUN_DIR / "top_k_log_probs.npy")

x0 = np.array(top_params[0], dtype=float)

print("=" * 75)
print("SAVED HIGH-MODE STARTING POINT")
print("=" * 75)

print(f"saved log_prob = {top_log_probs[0]:.8f}")

for p, value in zip(base_param_defs, x0):
    if p.name == "tau":
        print(
            f"log10_tau      = {value:.8f}\n"
            f"tau            = {10.0**value:.6f} s\n"
            f"current bounds = {p.lower:.6f} -- {p.upper:.6f} s"
        )


# ============================================================
# Run local optimizations
# ============================================================

for tau_max in TAU_MAX_VALUES:

    if tau_max is None:
        param_defs = base_param_defs
        label = "CURRENT TAU PRIOR"
    else:
        param_defs = replace_tau_upper(
            base_param_defs,
            tau_max,
        )
        label = f"WIDENED TAU PRIOR: tau_max = {tau_max:g} s"

    bounds = ep._sampled_bounds(param_defs)

    # Verify the saved point under this prior.
    start_lp = ep.log_probability(
        x0,
        param_defs,
        xrt_data,
        i_data,
        xrt_index_data,
        SPECTRAL_INDEX_COOLING,
    )

    tau_def = next(p for p in param_defs if p.name == "tau")

    print()
    print("=" * 75)
    print(label)
    print("=" * 75)

    print(
        f"tau bounds        = "
        f"{tau_def.lower:g} -- {tau_def.upper:g} s"
    )
    print(f"starting log_prob = {start_lp:.8f}")
    print(f"starting tau      = {10.0**x0[saved_labels.index('log10_tau')]:.6f} s")

    t0 = time.time()

    result = minimize(
        objective,
        x0,
        args=(
            param_defs,
            xrt_data,
            i_data,
            xrt_index_data,
        ),
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": 300,
            "xtol": 1e-5,
            "ftol": 1e-5,
            "disp": True,
        },
    )

    elapsed = time.time() - t0

    print_solution(
        f"OPTIMIZED RESULT — {label}",
        result,
        param_defs,
    )

    tau_idx = saved_labels.index("log10_tau")

    print()
    print(f"optimized tau = {10.0**result.x[tau_idx]:.6f} s")
    print(f"elapsed       = {elapsed:.1f} s")