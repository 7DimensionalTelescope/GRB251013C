"""Reading and writing MCMC run artifacts.

Each run writes a directory containing samples.npy, log_probs.npy, labels.txt,
top_k_params.npy, top_k_log_probs.npy and bestfit_params.txt.
"""
from pathlib import Path

import numpy as np

from VegasAfterglow import Scale


def top_k_samples(samples, log_probs, top_k):
    order = np.argsort(log_probs)[::-1]
    samples_sorted = samples[order]
    log_probs_sorted = log_probs[order]
    keep = []
    seen = set()
    for idx, sample in enumerate(np.round(samples_sorted, 12)):
        key = tuple(sample)
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)
        if len(keep) >= top_k:
            break
    return samples_sorted[keep], log_probs_sorted[keep]


def read_labels(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def latest_result_dir(base_dir, prefix):
    candidates = []
    for path in Path(base_dir).glob(f"{prefix}*"):
        if (path / "top_k_params.npy").exists() and (path / "labels.txt").exists():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No usable {prefix}* result directory found in {base_dir}")
    return sorted(candidates)[-1]


def load_best_fit_params(outdir):
    """Load best-fit parameters from saved results"""
    outdir = Path(outdir)

    # Load samples and labels
    samples = np.load(outdir / "samples.npy")
    log_probs = np.load(outdir / "log_probs.npy")
    labels = [l.strip() for l in (outdir / "labels.txt").read_text().strip().split("\n") if l.strip()]

    # Get best-fit sample
    best_idx = np.argmax(log_probs)
    best_sample = samples[best_idx]

    # Determine model configuration
    include_flare = any("t_start_flare" in label for label in labels)
    include_wing = any("E_iso_wing" in label for label in labels)

    # Convert to physical parameters
    params = {}
    for label, value in zip(labels, best_sample):
        if label.startswith("log10_"):
            param_name = label.replace("log10_", "")
            params[param_name] = 10 ** value
        else:
            param_name = label
            params[param_name] = value

    return params, include_flare, include_wing


def save_run_arrays(outdir, samples, log_probs, labels, top_k=10):
    """Write samples/log_probs/labels and the top-k subset.

    Returns (top_params, top_log_probs).
    """
    outdir = Path(outdir)
    np.save(outdir / "samples.npy", samples)
    np.save(outdir / "log_probs.npy", log_probs)
    (outdir / "labels.txt").write_text("\n".join(labels))

    top_params, top_log_probs = top_k_samples(samples, log_probs, top_k)
    np.save(outdir / "top_k_params.npy", top_params)
    np.save(outdir / "top_k_log_probs.npy", top_log_probs)
    return top_params, top_log_probs


def save_bestfit_params(outdir, labels, param_defs, top_params, top_log_probs,
                        xrt_data, optical_datasets):
    """Write the human-readable bestfit_params.txt summary."""
    outdir = Path(outdir)
    lines = [
        "=== Fit Configuration ===",
        f"Model: Core+RS + Norris flare + Wing jet",
        f"XRT data: {len(xrt_data['time'])} points",
    ]
    for dataset in optical_datasets:
        lines.append(f"{dataset['name']}: {len(dataset['time'])} points")
    lines.append(f"Best log probability: {top_log_probs[0]:.6g}")
    lines.append("")
    lines.append("=== Best-fit Parameters ===")
    lines.append(f"{'label':<20} {'sampled':>14} {'physical':>14}")
    lines.append("-" * 50)

    for label, param_def, sampled in zip(labels, param_defs, top_params[0]):
        if param_def.scale is Scale.LOG:
            physical = 10 ** sampled
        else:
            physical = sampled
        lines.append(f"{label:<20} {sampled:>14.6g} {physical:>14.6g}")

    (outdir / "bestfit_params.txt").write_text("\n".join(lines) + "\n")
