"""Reading and writing MCMC run artifacts.

Each run writes a directory containing samples.npy, log_probs.npy, labels.txt,
top_k_params.npy, top_k_log_probs.npy and bestfit_params.txt.
"""
from pathlib import Path

import numpy as np

from VegasAfterglow import Scale

from .prior import ParamDefWithPrior


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
    include_flare = any("t_start_flare" in label or "t_peak_flare" in label
                        or "t_break_flare" in label for label in labels)
    include_wing = any("E_iso_wing" in label for label in labels)

    # Flare shape: labels alone can't disambiguate shapes that share parameter
    # names (e.g. norris and any future shape reusing t_start_flare), so it's
    # persisted separately by save_run_arrays; default "norris" for older runs
    # saved before flare_shape.txt existed.
    flare_shape_file = outdir / "flare_shape.txt"
    flare_shape = flare_shape_file.read_text().strip() if flare_shape_file.exists() else "norris"
    if not flare_shape:
        flare_shape = "norris"

    # Convert to physical parameters
    params = {}
    for label, value in zip(labels, best_sample):
        if label.startswith("log10_"):
            param_name = label.replace("log10_", "")
            params[param_name] = 10 ** value
        else:
            param_name = label
            params[param_name] = value

    return params, include_flare, include_wing, flare_shape


def save_run_arrays(outdir, samples, log_probs, labels, top_k=10, flare_shape=None):
    """Write samples/log_probs/labels and the top-k subset.

    flare_shape (see grb.functions.FLARE_SHAPES) is persisted to flare_shape.txt
    so load_best_fit_params can pick the right profile back up later -- labels
    alone can't disambiguate shapes that share parameter names. Omit (or None)
    when the run has no flare.

    Returns (top_params, top_log_probs).
    """
    outdir = Path(outdir)
    np.save(outdir / "samples.npy", samples)
    np.save(outdir / "log_probs.npy", log_probs)
    (outdir / "labels.txt").write_text("\n".join(labels))
    if flare_shape:
        (outdir / "flare_shape.txt").write_text(flare_shape)

    top_params, top_log_probs = top_k_samples(samples, log_probs, top_k)
    np.save(outdir / "top_k_params.npy", top_params)
    np.save(outdir / "top_k_log_probs.npy", top_log_probs)
    return top_params, top_log_probs


_FLARE_SHAPE_PARAM_ORDER = {
    "norris": ["t_start_flare", "tau_rise_flare", "tau_decay_flare", "A_flare", "flare_beta"],
    "sbpl": ["t_break_flare", "alpha_rise_flare", "alpha_decay_flare", "smooth_flare",
             "A_flare", "flare_beta"],
    "gaussian": ["t_peak_flare", "sigma_flare", "A_flare", "flare_beta"],
}

_WING_PARAM_ORDER = ["E_iso_wing", "Gamma0_wing", "theta_c_wing", "p_wing",
                     "eps_e_wing", "eps_B_wing"]


def _format_overrides_block(environment, physical_values, include_flare, flare_shape,
                            include_wing, initial_guess):
    """Build the `overrides = {...}` dict lines for manual_model.ipynb.

    Mirrors the overrides cell in manual_model.ipynb: every parameter name is
    listed, the ones active in this fit are live (uncommented) with their
    best-fit value, and the ones for other medium/flare-shape/wing choices are
    commented out with their fit_final_model.py INITIAL_GUESS seed so they're
    ready to uncomment if the config is switched.
    """
    def fmt(name, value, commented=False, note=""):
        prefix = "    # " if commented else "    "
        return f'{prefix}"{name}": {value:.6g},{note}'

    lines = ["overrides = {"]

    lines.append("    # --- Core jet ---")
    for name in ("E_iso_core", "Gamma0_core", "theta_c_core"):
        lines.append(fmt(name, physical_values[name]))
    lines.append("")

    lines.append(f'    # --- Circumburst medium (matches environment="{environment}" above) ---')
    medium_notes = {
        "n_ism": "",
        "A_star": '   # uncomment for environment="wind" or "stratified"',
        "n0": '        # uncomment for environment="stratified" only',
    }
    for name, note in medium_notes.items():
        if name in physical_values:
            lines.append(fmt(name, physical_values[name], commented=False, note=note))
        elif name in initial_guess:
            lines.append(fmt(name, initial_guess[name], commented=True, note=note))
    lines.append("")

    lines.append("    # --- Forward shock microphysics ---")
    for name in ("p", "eps_e", "eps_B", "tau"):
        lines.append(fmt(name, physical_values[name]))
    lines.append("")

    lines.append("    # --- Reverse shock ---")
    for name in ("p_r", "eps_e_r", "eps_B_r"):
        lines.append(fmt(name, physical_values[name]))
    lines.append("")

    lines.append("    # --- Host extinction ---")
    lines.append(fmt("A_V", physical_values["A_V"]))
    lines.append("")

    for shape, shape_names in _FLARE_SHAPE_PARAM_ORDER.items():
        is_active = include_flare and shape == flare_shape
        if is_active:
            lines.append(f'    # --- Flare (matches include_flare=True, flare_shape="{shape}" above) ---')
            for name in shape_names:
                lines.append(fmt(name, physical_values[name]))
        else:
            lines.append(f'    # --- Flare (flare_shape="{shape}"; not used in this fit -- these are just'
                          " the fit_final_model.py INITIAL_GUESS seed) ---")
            for name in shape_names:
                if name in initial_guess:
                    lines.append(fmt(name, initial_guess[name], commented=True))
        lines.append("")

    if include_wing:
        lines.append("    # --- Wing jet ---")
        for name in _WING_PARAM_ORDER:
            lines.append(fmt(name, physical_values[name]))
    else:
        lines.append("    # --- Wing jet (uncomment all 6 for include_wing=True; this best fit didn't")
        lines.append("    # include a wing, so these are still the INITIAL_GUESS placeholder values) ---")
        for name in _WING_PARAM_ORDER:
            if name in initial_guess:
                lines.append(fmt(name, initial_guess[name], commented=True))

    lines.append("}")
    return lines


def save_bestfit_params(outdir, labels, param_defs, top_params, top_log_probs,
                        xrt_data, optical_datasets, include_flare=True, include_wing=True,
                        flare_shape="norris", initial_guess=None, environment="ism"):
    """Write the human-readable bestfit_params.txt summary.

    initial_guess: optional {param_name: physical_value} dict (e.g. INITIAL_GUESS
    from fit_final_model.py) used to seed the walkers. When given, three extra
    sections are appended below the original output: the initial guess and prior
    used for every fitted parameter, the best fit re-written in the same
    INITIAL_GUESS dict format for copy-paste back into the driver script, and
    the best fit re-written as an `overrides` dict for copy-paste into
    manual_model.ipynb.
    """
    outdir = Path(outdir)
    model_desc = "Core+RS"
    if include_flare:
        model_desc += f" + {flare_shape} flare"
    if include_wing:
        model_desc += " + Wing jet"
    lines = [
        "=== Fit Configuration ===",
        f"Model: {model_desc}",
        f"XRT data: {len(xrt_data['time'])} points",
    ]
    for dataset in optical_datasets:
        lines.append(f"{dataset['name']}: {len(dataset['time'])} points")
    lines.append(f"Best log probability: {top_log_probs[0]:.6g}")
    lines.append("")
    lines.append("=== Best-fit Parameters ===")
    lines.append(f"{'label':<20} {'sampled':>14} {'physical':>14}")
    lines.append("-" * 50)

    physical_values = {}
    for label, param_def, sampled in zip(labels, param_defs, top_params[0]):
        if param_def.scale is Scale.LOG:
            physical = 10 ** sampled
        else:
            physical = sampled
        physical_values[param_def.name] = physical
        lines.append(f"{label:<20} {sampled:>14.6g} {physical:>14.6g}")

    if initial_guess is not None:
        lines.append("")
        lines.append("=== Initial Guesses & Priors ===")
        header = f"{'name':<20} {'initial_guess':>14} {'scale':>7} {'bounds':>22}  {'prior'}"
        lines.append(header)
        lines.append("-" * len(header))
        for param_def in param_defs:
            name = param_def.name
            scale_str = "LOG" if param_def.scale is Scale.LOG else "LINEAR"
            bounds_str = f"[{param_def.lower:.6g}, {param_def.upper:.6g}]"
            if isinstance(param_def, ParamDefWithPrior) and param_def.has_gaussian_prior():
                mean, sigma = param_def.get_prior_mean_sigma()
                space = "log10" if param_def.scale is Scale.LOG else "linear"
                prior_str = f"gaussian({space} mean={mean:.6g}, sigma={sigma:.6g}) on top of the box"
            else:
                prior_str = "uniform (box only)"
            guess_str = f"{initial_guess[name]:.6g}" if name in initial_guess else "n/a"
            lines.append(f"{name:<20} {guess_str:>14} {scale_str:>7} {bounds_str:>22}  {prior_str}")

        lines.append("")
        lines.append("=== Best-fit Parameters (INITIAL_GUESS format) ===")
        lines.append("INITIAL_GUESS = {")
        for param_def in param_defs:
            name = param_def.name
            lines.append(f'    "{name}": {physical_values[name]:.6g},')
        lines.append("}")

        lines.append("")
        lines.append("=== Best-fit Parameters (overrides format for manual_model.ipynb) ===")
        lines.extend(_format_overrides_block(
            environment, physical_values, include_flare, flare_shape, include_wing, initial_guess,
        ))

    (outdir / "bestfit_params.txt").write_text("\n".join(lines) + "\n")
