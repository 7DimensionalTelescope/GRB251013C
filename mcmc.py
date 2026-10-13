import numpy as np
from pathlib import Path

RESULT_DIR = Path(
    "/home/hongyp007/yp_folder/projects/GRB/result/"
    "early_phase_20261006_184042"
)

samples = np.load(RESULT_DIR / "samples.npy")
logp = np.load(RESULT_DIR / "log_probs.npy")

with open(RESULT_DIR / "labels.txt") as f:
    labels = [x.strip() for x in f if x.strip()]

tau_idx = labels.index("log10_tau")
tau = 10.0 ** samples[:, tau_idx]


def print_point(name, idx):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"log_prob = {logp[idx]:.6f}")
    print(f"tau      = {tau[idx]:.6f} s")
    print()

    for label, value in zip(labels, samples[idx]):
        if label == "log10_tau":
            print(
                f"{label:20s} = {value: .8f} "
                f"(tau = {10**value:.3f} s)"
            )
        else:
            print(f"{label:20s} = {value: .8f}")


# ------------------------------------------------------------
# Global best point
# ------------------------------------------------------------

i_global = np.nanargmax(logp)
print_point("GLOBAL BEST", i_global)


# ------------------------------------------------------------
# Low-tau mode
# tau < 150 s
# ------------------------------------------------------------

low_mask = tau < 150.0
low_indices = np.where(low_mask)[0]

i_low = low_indices[
    np.nanargmax(logp[low_mask])
]

print_point("BEST LOW-TAU MODE", i_low)


# ------------------------------------------------------------
# High-tau mode
# tau > 300 s
# ------------------------------------------------------------

high_mask = tau > 300.0
high_indices = np.where(high_mask)[0]

i_high = high_indices[
    np.nanargmax(logp[high_mask])
]

print_point("BEST HIGH-TAU MODE", i_high)


# ------------------------------------------------------------
# Difference
# ------------------------------------------------------------

print()
print("=" * 70)
print("MODE COMPARISON")
print("=" * 70)

print(
    f"Low mode : tau = {tau[i_low]:.3f} s, "
    f"logP = {logp[i_low]:.3f}"
)

print(
    f"High mode: tau = {tau[i_high]:.3f} s, "
    f"logP = {logp[i_high]:.3f}"
)

print(
    f"Delta logP (high - low) = "
    f"{logp[i_high] - logp[i_low]:.3f}"
)