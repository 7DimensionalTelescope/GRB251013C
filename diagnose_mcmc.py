from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Settings
# ============================================================

RESULT_DIR = Path(
    "/home/hongyp007/yp_folder/projects/GRB/result/"
    "early_phase_20261006_184042"
)

N_STEPS = 63000
N_BURN = 2500
THIN = 1

# Define the high-probability mode as samples within this
# many log-probability units of the global maximum.
DELTA_LOGP = 3.0


# ============================================================
# Load data
# ============================================================

samples = np.load(RESULT_DIR / "samples.npy")
log_probs = np.load(RESULT_DIR / "log_probs.npy")

labels_path = RESULT_DIR / "labels.txt"
if labels_path.exists():
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f if line.strip()]
else:
    labels = [f"param_{i}" for i in range(samples.shape[-1])]


print("=" * 70)
print("RAW FILE INFORMATION")
print("=" * 70)
print(f"samples.shape   = {samples.shape}")
print(f"log_probs.shape = {log_probs.shape}")
print(f"Number of labels = {len(labels)}")
print()

for i, label in enumerate(labels):
    print(f"{i:2d}: {label}")

print()


# ============================================================
# Check whether samples were saved flattened
# ============================================================

if samples.ndim == 3:
    # Already stored as (step, walker, parameter)
    chain = samples
    lp_chain = log_probs

    n_post_steps, n_walkers, n_dim = chain.shape

    print("\nChain is already 3-dimensional.")
    print(f"post-burn steps = {n_post_steps}")
    print(f"walkers         = {n_walkers}")
    print(f"dimensions      = {n_dim}")

elif samples.ndim == 2:
    # Probably flattened with shape:
    # (n_post_steps * n_walkers, n_dim)

    n_flat, n_dim = samples.shape

    expected_post_steps = len(
        range(N_BURN, N_STEPS, THIN)
    )

    if n_flat % expected_post_steps != 0:
        raise RuntimeError(
            "\nCannot infer the number of walkers.\n"
            f"Number of flattened samples = {n_flat}\n"
            f"Expected saved steps         = {expected_post_steps}\n\n"
            "This probably means that N_STEPS, N_BURN, or THIN "
            "is different from the values entered above."
        )

    n_walkers = n_flat // expected_post_steps

    print("=" * 70)
    print("RECONSTRUCTING WALKER HISTORY")
    print("=" * 70)

    print(f"Flattened samples : {n_flat}")
    print(f"Saved steps       : {expected_post_steps}")
    print(f"Inferred walkers  : {n_walkers}")

    chain = samples.reshape(
        expected_post_steps,
        n_walkers,
        n_dim
    )

    lp_chain = log_probs.reshape(
        expected_post_steps,
        n_walkers
    )

    n_post_steps = expected_post_steps

else:
    raise RuntimeError(
        f"Unexpected samples.npy shape: {samples.shape}"
    )


# ============================================================
# Basic log-probability diagnostics
# ============================================================

max_lp = np.nanmax(lp_chain)
threshold = max_lp - DELTA_LOGP

best_index = np.unravel_index(
    np.nanargmax(lp_chain),
    lp_chain.shape
)

best_post_step, best_walker = best_index

# Convert saved-chain index back to original MCMC step.
best_original_step = (
    N_BURN + best_post_step * THIN
)

high_mask = lp_chain >= threshold

walkers_found = np.where(
    np.any(high_mask, axis=0)
)[0]


print()
print("=" * 70)
print("HIGH-PROBABILITY MODE")
print("=" * 70)

print(f"Global maximum log_prob = {max_lp:.6f}")
print(
    f"High-mode threshold     = {threshold:.6f} "
    f"(max - {DELTA_LOGP})"
)

print(
    f"Global maximum found by walker {best_walker} "
    f"at original step {best_original_step}"
)

print(
    f"Walkers reaching high mode = "
    f"{len(walkers_found)} / {n_walkers}"
)

n_high_samples = np.sum(high_mask)
n_total_samples = lp_chain.size

print(
    f"Samples in high mode = {n_high_samples:,} / "
    f"{n_total_samples:,} "
    f"({100 * n_high_samples / n_total_samples:.4f}%)"
)

print()


# ============================================================
# Per-walker information
# ============================================================

rows = []

print("=" * 70)
print("WALKERS THAT REACHED THE HIGH-PROBABILITY MODE")
print("=" * 70)

if len(walkers_found) == 0:
    print("No walker reached the selected threshold.")

else:
    header = (
        f"{'walker':>7} "
        f"{'first step':>12} "
        f"{'best step':>12} "
        f"{'best logP':>12} "
        f"{'last step':>12} "
        f"{'N high':>10}"
    )

    print(header)
    print("-" * len(header))

    for walker in walkers_found:

        hit_indices = np.where(
            high_mask[:, walker]
        )[0]

        first_post = hit_indices[0]
        last_post = hit_indices[-1]

        first_step = (
            N_BURN + first_post * THIN
        )
        last_step = (
            N_BURN + last_post * THIN
        )

        walker_lp = lp_chain[:, walker]

        best_post = np.nanargmax(walker_lp)
        best_step = (
            N_BURN + best_post * THIN
        )
        best_lp_walker = walker_lp[best_post]

        n_high = len(hit_indices)

        rows.append(
            [
                walker,
                first_step,
                best_step,
                best_lp_walker,
                last_step,
                n_high,
            ]
        )

        print(
            f"{walker:7d} "
            f"{first_step:12d} "
            f"{best_step:12d} "
            f"{best_lp_walker:12.3f} "
            f"{last_step:12d} "
            f"{n_high:10d}"
        )


# ============================================================
# Save walker summary
# ============================================================

summary_file = RESULT_DIR / "high_mode_walkers.txt"

with open(summary_file, "w") as f:

    f.write(f"Maximum log_prob: {max_lp:.8f}\n")
    f.write(
        f"Threshold: {threshold:.8f} "
        f"(max - {DELTA_LOGP})\n"
    )
    f.write(
        f"Walkers reaching threshold: "
        f"{len(walkers_found)} / {n_walkers}\n"
    )
    f.write(
        f"High samples: {n_high_samples} / "
        f"{n_total_samples}\n\n"
    )

    f.write(
        "walker first_step best_step "
        "best_log_prob last_step n_high\n"
    )

    for row in rows:
        f.write(
            f"{row[0]} "
            f"{row[1]} "
            f"{row[2]} "
            f"{row[3]:.8f} "
            f"{row[4]} "
            f"{row[5]}\n"
        )

print()
print(f"Saved summary to: {summary_file}")


# ============================================================
# Plot 1: log-probability trace of every walker
# ============================================================

step_axis = (
    N_BURN
    + np.arange(n_post_steps) * THIN
)

fig, ax = plt.subplots(figsize=(12, 7))

for walker in range(n_walkers):

    # Make walkers that found the high mode more prominent.
    if walker in walkers_found:
        ax.plot(
            step_axis,
            lp_chain[:, walker],
            linewidth=1.0,
            alpha=0.9,
            label=f"walker {walker}",
        )
    else:
        ax.plot(
            step_axis,
            lp_chain[:, walker],
            linewidth=0.5,
            alpha=0.25,
        )

ax.axhline(
    threshold,
    linestyle="--",
    linewidth=1.2,
    label=f"max logP - {DELTA_LOGP}",
)

ax.set_xlabel("MCMC step")
ax.set_ylabel("log probability")
ax.set_title("Log-probability trace by walker")

# Focus on the scientifically relevant part while still showing
# the local mode around ~ -190.
lower_limit = max(
    np.nanpercentile(lp_chain, 0.5),
    -350,
)

ax.set_ylim(lower_limit, max_lp + 5)

if len(walkers_found) <= 10:
    ax.legend()

fig.tight_layout()

trace_path = RESULT_DIR / "log_prob_walker_trace.png"
fig.savefig(trace_path, dpi=200)
plt.close(fig)

print(f"Saved: {trace_path}")


# ============================================================
# Find tau parameter
# ============================================================

def normalize_label(label):
    return (
        label.lower()
        .replace("$", "")
        .replace("\\", "")
        .replace("{", "")
        .replace("}", "")
        .replace("_", "")
        .replace(" ", "")
    )


tau_idx = None

for i, label in enumerate(labels):
    norm = normalize_label(label)

    if "tau" in norm:
        tau_idx = i
        break


if tau_idx is None:
    print()
    print("WARNING: Could not automatically find tau in labels.txt.")

else:

    tau_label = labels[tau_idx]

    print()
    print("=" * 70)
    print("TAU PARAMETER")
    print("=" * 70)

    print(f"tau index = {tau_idx}")
    print(f"tau label = {tau_label}")

    tau_chain_raw = chain[:, :, tau_idx]

    # Your corner plot suggests the fitted parameter is log10_tau.
    # Convert it to seconds if appropriate.
    normalized_tau_label = normalize_label(tau_label)

    if "log10tau" in normalized_tau_label:
        tau_chain = 10.0 ** tau_chain_raw
        tau_ylabel = "tau [s]"
    else:
        tau_chain = tau_chain_raw
        tau_ylabel = tau_label


    # ========================================================
    # Plot 2: tau trace for walkers that reached high mode
    # ========================================================

    fig, ax = plt.subplots(figsize=(12, 7))

    if len(walkers_found) > 0:

        for walker in walkers_found:

            ax.plot(
                step_axis,
                tau_chain[:, walker],
                linewidth=0.9,
                alpha=0.8,
                label=f"walker {walker}",
            )

    else:

        for walker in range(n_walkers):

            ax.plot(
                step_axis,
                tau_chain[:, walker],
                linewidth=0.5,
                alpha=0.3,
            )

    ax.set_xlabel("MCMC step")
    ax.set_ylabel(tau_ylabel)
    ax.set_title(
        "Tau trace for walkers reaching high-probability mode"
    )

    if 0 < len(walkers_found) <= 10:
        ax.legend()

    fig.tight_layout()

    tau_trace_path = RESULT_DIR / "tau_high_mode_trace.png"
    fig.savefig(tau_trace_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {tau_trace_path}")


    # ========================================================
    # Print tau value when each walker first reaches high mode
    # ========================================================

    if len(walkers_found) > 0:

        print()
        print(
            "tau when each walker first enters the high mode:"
        )

        for walker in walkers_found:

            first_post = np.where(
                high_mask[:, walker]
            )[0][0]

            first_step = (
                N_BURN + first_post * THIN
            )

            tau_value = tau_chain[
                first_post,
                walker
            ]

            print(
                f"walker {walker:3d}: "
                f"step={first_step:6d}, "
                f"tau={tau_value:.3f} s"
            )


    # ========================================================
    # Plot 3: tau versus log probability
    # ========================================================

    tau_flat = tau_chain.reshape(-1)
    lp_flat = lp_chain.reshape(-1)

    # Plotting ~2 million points is unnecessarily heavy.
    # Downsample only for visualization.
    max_plot_points = 200000

    if len(lp_flat) > max_plot_points:
        plot_indices = np.linspace(
            0,
            len(lp_flat) - 1,
            max_plot_points,
            dtype=int,
        )
    else:
        plot_indices = np.arange(len(lp_flat))

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(
        tau_flat[plot_indices],
        lp_flat[plot_indices],
        s=2,
        alpha=0.15,
    )

    high_tau = tau_flat[
        high_mask.reshape(-1)
    ]
    high_lp = lp_flat[
        high_mask.reshape(-1)
    ]

    ax.scatter(
        high_tau,
        high_lp,
        s=5,
        alpha=0.7,
    )

    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(tau_ylabel)
    ax.set_ylabel("log probability")
    ax.set_title("Tau vs log probability")

    fig.tight_layout()

    tau_lp_path = RESULT_DIR / "tau_vs_log_prob.png"
    fig.savefig(tau_lp_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {tau_lp_path}")


# ============================================================
# Threshold sensitivity
# ============================================================

print()
print("=" * 70)
print("THRESHOLD SENSITIVITY")
print("=" * 70)

for delta in [1, 3, 5, 10, 20]:

    mask = lp_chain >= max_lp - delta

    n_walk = np.sum(
        np.any(mask, axis=0)
    )

    n_samp = np.sum(mask)

    print(
        f"max - {delta:2d}: "
        f"{n_walk:2d}/{n_walkers} walkers, "
        f"{n_samp:8d} samples "
        f"({100*n_samp/lp_chain.size:.4f}%)"
    )


print()
print("Done.")