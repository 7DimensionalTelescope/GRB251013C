#!/usr/bin/env python3
"""Early-phase MCMC: run I/O, the legacy sampler, and the production sampler with its checkpoint diagnostics.

legacy (final / wide priors): run_custom_emcee - StretchMove, Gaussian ball around a differential-evolution
    optimum; unchanged except for the optional --nwalkers.
production: run_production_mcmc - DE / DE-snooker move mixture, ridge-aware initial ensemble over the known
    high-tau family, emcee HDFBackend (<outdir>/chain.h5) that survives disconnection and is continued in place
    with --resume; check_chain - autocorrelation / acceptance / stationarity / boundary diagnostics at a checkpoint
    and the autonomous continue / stop decision used by run_early_phase_mcmc_overnight.sh.

Every likelihood evaluation goes through early_phase_model.log_probability.
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path

import emcee
import numpy as np
from VegasAfterglow import Scale

os.sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from grb.const import HOST_AV_LOG10_MEAN, HOST_AV_LOG10_SIGMA, XRT_FLARE_START_TIME
import early_phase_model as epm
from early_phase_model import (
    EPS_SUM_PAIRS,
    I_DATA_ROWS,
    _init_early_phase_context,
    _log_probability_from_context,
    _sampled_bounds,
    from_sampling_coords,
    make_param_defs,
    param_label,
    physical_param_value,
    sampling_labels,
    sampling_log_jacobian,
    to_sampling_coords,
)
from early_phase_optimize import find_start_point


def best_params_from_result(result):
    if result.top_k_params is not None and len(result.top_k_params) > 0:
        return result.top_k_params[0]
    return result.samples[np.nanargmax(result.log_probs)]


def save_result_arrays(result, outdir):
    np.save(outdir / "samples.npy", result.samples)
    np.save(outdir / "log_probs.npy", result.log_probs)
    if result.top_k_params is not None:
        np.save(outdir / "top_k_params.npy", result.top_k_params)
    if result.top_k_log_probs is not None:
        np.save(outdir / "top_k_log_probs.npy", result.top_k_log_probs)
    (outdir / "labels.txt").write_text("\n".join(result.labels) + "\n")


def save_bestfit_params(outdir, param_defs, xrt_data, i_data, environment="ism"):
    labels = [line.strip() for line in (outdir / "labels.txt").read_text().splitlines() if line.strip()]
    top_params = np.load(outdir / "top_k_params.npy")
    top_log_probs = np.load(outdir / "top_k_log_probs.npy")
    best_params = top_params[0]
    best_log_prob = top_log_probs[0]

    lines = [
        "=== Fit Configuration ===",
        f"Model: {environment} tophat forward shock + reverse shock",
        f"VEGASafterglow: Fitter(jet='tophat', medium='{'ism' if environment == 'ism' else 'wind'}', rvs_shock=True)",
        f"XRT data: {len(xrt_data)} points, band = 0.3-10 keV",
        f"i-band data: {len(i_data)} points, first {I_DATA_ROWS} rows after sorting",
        f"XRT selection: time < {XRT_FLARE_START_TIME:.0f} s (pre-flare only)",
        f"Host A_V prior: log10(A_V) ~ Normal({HOST_AV_LOG10_MEAN}, {HOST_AV_LOG10_SIGMA})",
        f"Best log probability: {best_log_prob:.6g}",
        "",
        "=== Best-fit Parameters ===",
        f"{'label':<18} {'sampled':>14} {'physical':>14} {'prior_min':>14} {'prior_max':>14}",
        "-" * 80,
    ]

    for label, param_def, sampled in zip(labels, param_defs, best_params):
        physical = physical_param_value(param_def, sampled)
        lines.append(
            f"{label:<18} {sampled:>14.6g} {physical:>14.6g} "
            f"{param_def.lower:>14.6g} {param_def.upper:>14.6g}"
        )

    samples_path = outdir / "samples.npy"
    if samples_path.exists():
        samples = np.load(samples_path)
        lines.extend([
            "",
            f"=== Posterior Summary (median, 16-84%; {len(samples)} samples; sampled-space values) ===",
            f"{'label':<18} {'median':>14} {'-1sigma':>14} {'+1sigma':>14}",
            "-" * 64,
        ])
        for label, column in zip(labels, samples.T):
            lo, med, hi = np.percentile(column, [16, 50, 84])
            lines.append(f"{label:<18} {med:>14.6g} {med - lo:>14.6g} {hi - med:>14.6g}")

    lines.extend([
        "",
        "=== Top-k Log Probabilities ===",
    ])
    for idx, log_prob in enumerate(top_log_probs, start=1):
        lines.append(f"{idx:02d}: {log_prob:.6g}")

    (outdir / "bestfit_params.txt").write_text("\n".join(lines) + "\n")


def run_custom_emcee(param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling, npool=4, nsteps=10000, nburn=2000, nopt=100, seed=None, nwalkers=None):
    """Run emcee with custom likelihood including spectral index."""
    from concurrent.futures import ProcessPoolExecutor

    rng = np.random.default_rng(seed)
    ndim = len(param_defs)
    nwalkers = resolve_nwalkers(nwalkers, ndim)
    bounds = _sampled_bounds(param_defs)
    lower, upper = bounds[:, 0], bounds[:, 1]

    # Main-process context (used for seeding checks); workers get it via the pool initializer
    context = (param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
    _init_early_phase_context(context)

    from contextlib import nullcontext

    pool_ctx = (
        ProcessPoolExecutor(max_workers=npool, initializer=_init_early_phase_context, initargs=(context,))
        if npool > 1 else nullcontext()
    )
    with pool_ctx as executor:
        start = find_start_point(bounds, executor.map if executor is not None else None, nopt, rng)
        if start is None:
            start = 0.5 * (lower + upper)
            scatter = 0.1 * (upper - lower)
        else:
            scatter = 1e-3 * (upper - lower)

        # Starting positions: Gaussian ball around the start point, redrawn until log_prob is finite
        p0 = []
        n_tries = 0
        while len(p0) < nwalkers:
            pos = np.clip(rng.normal(start, scatter), lower, upper)
            n_tries += 1
            if np.isfinite(_log_probability_from_context(pos)):
                p0.append(pos)
            elif n_tries > 1000 * nwalkers:
                raise RuntimeError("Could not find finite-probability starting positions for the walkers.")
        p0 = np.array(p0)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, _log_probability_from_context, pool=executor)
        sampler.run_mcmc(p0, nsteps, progress=True)

    tau = sampler.get_autocorr_time(tol=0)
    print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    print(f"Autocorrelation time: max {np.max(tau):.0f} steps, mean {np.mean(tau):.0f} steps")
    if nsteps < 50 * np.max(tau):
        print(f"WARNING: chain is shorter than 50 autocorrelation times (nsteps={nsteps}, 50*tau_max={50 * np.max(tau):.0f}); "
              "the fit is probably not converged. Increase --nsteps.")
    if nburn < 2 * np.max(tau):
        print(f"WARNING: nburn={nburn} is below 2*tau_max={2 * np.max(tau):.0f}.")

    # Extract results
    samples = sampler.get_chain(discard=nburn, flat=True)
    log_probs = sampler.get_log_prob(discard=nburn, flat=True)
    
    # Get labels
    labels = []
    for param_def in param_defs:
        if param_def.scale is Scale.LOG:
            labels.append(f"log10_{param_def.name}")
        else:
            labels.append(param_def.name)
    
    # Create result object
    class Result:
        pass
    
    result = Result()
    result.samples = samples
    result.log_probs = log_probs
    result.labels = labels
    
    # Get top-k samples
    from grb.results import top_k_samples
    result.top_k_params, result.top_k_log_probs = top_k_samples(samples, log_probs, top_k=10)
    
    return result


def resolve_nwalkers(nwalkers, ndim):
    """--nwalkers: None keeps the automatic max(2 * ndim, 32); an explicit value must be even and >= 2 * ndim."""
    if nwalkers is None:
        return max(2 * ndim, 32)
    if nwalkers < 2 * ndim or nwalkers % 2:
        raise ValueError(f"--nwalkers must be an even number >= 2 * ndim = {2 * ndim} for emcee, got {nwalkers}")
    return nwalkers


# ---------------------------------------------------------------------------------------------------------------
# Production sampler: DE move mixture, ridge-aware start, HDF backend with resume
# ---------------------------------------------------------------------------------------------------------------

# Differential-evolution proposals jump along the difference of two other walkers, so their geometry follows the
# ensemble's own covariance instead of the single stretch direction of StretchMove (the old default, which mixed
# badly here). DEMove (80%) handles the long, nearly linear E_iso / A_star / eps / xi ridge; DESnookerMove (20%)
# handles its curvature (xi is sampled linearly, the other ridge parameters in log10) and non-Gaussian tails.
# The 0.8 / 0.2 weights are the mixture recommended in the emcee documentation.
MOVES_DESCRIPTION = "DEMove(sigma=1e-5, gamma0=2.38/sqrt(2*ndim)) weight 0.8 + DESnookerMove(gammas=1.7) weight 0.2"


def production_moves():
    return [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]


PRODUCTION_LABELS = [
    "log10_E_iso", "log10_Gamma0", "log10_theta_c", "log10_A_star", "p", "log10_eps_e", "log10_eps_B", "xi_e",
    "log10_tau", "p_r", "log10_eps_e_r", "log10_eps_B_r", "xi_e_r", "log10_A_V",
]
# Known good high-tau solutions (xi_e = 1, xi_e_r fixed during each optimisation; 9 digits) from
# early_phase_eisosens_20261007_220624 (E_iso <= 1e55) and early_phase_xiprofile_20261007_180936 (E_iso <= 1e54, not
# at the cap for these rows). Layout: PRODUCTION_LABELS without xi_e. (name, log_posterior, vector)
RIDGE_ANCHORS = [
    ("eisosens xi_e_r=0.05 rank0", -86.3337, [54.2550437, 1.65468898, -1.77806885, 0.425797558, 2.01, -2.36895667, -1.30565595, 2.60838799, 2.00289313, -0.178415317, -0.47251015, 0.05, -0.496685381]),
    ("eisosens xi_e_r=0.05 rank1", -86.4876, [54.2416039, 1.62034539, -1.74525078, 0.539981665, 2.01002245, -2.33907456, -1.44671519, 2.60581651, 2.00292829, -0.167034676, -0.624292464, 0.05, -0.502240696]),
    ("eisosens xi_e_r=0.07 rank0", -86.6402, [54.0363515, 1.62284989, -1.651382, 0.427977489, 2.0107093, -2.33887536, -1.37846488, 2.60291349, 2.00320809, -0.155448131, -0.52160717, 0.07, -0.465639512]),
    ("eisosens xi_e_r=0.1 rank0", -87.0569, [53.9141943, 1.60466919, -1.6331926, 0.378478501, 2.01000001, -2.21300302, -1.40413104, 2.60530692, 2.00461526, -0.17648797, -0.476328925, 0.1, -0.457847318]),
    ("xiprofile xi_e_r=0.15", -87.9052, [53.7703128, 1.54779483, -1.5762596, 0.477308818, 2.01176938, -2.14250485, -1.61833485, 2.60837587, 2.00560801, -0.120224438, -0.616657819, 0.15, -0.446031574]),
    ("xiprofile xi_e_r=0.2", -88.7693, [53.6917014, 1.55482289, -1.58331776, 0.436037348, 2.01000293, -1.99132527, -1.6438985, 2.61687437, 2.00671537, -0.163588774, -0.546398403, 0.2, -0.426807219]),
]
REFERENCE_LOG_POSTERIOR = -86.3337      # best known optimum (first anchor); used to check the chain reached the region
# Joint rescaling E_iso, A_star -> /f; eps_e, eps_B, eps_e_r, eps_B_r, xi_e, xi_e_r -> *f is a near-exact
# degeneracy of this model (checked 2026-10-08 on the three eisosens optima: |dlogP| < 0.1 for f = 0.5-1, while
# rescaling only the forward-shock part costs dlogP ~ -200 already at f = 0.9). Walkers are spread along it.
RIDGE_UP = ("log10_E_iso", "log10_A_star")
RIDGE_DOWN = ("log10_eps_e", "log10_eps_B", "log10_eps_e_r", "log10_eps_B_r")
RIDGE_XI = ("xi_e", "xi_e_r")
INIT_LOGP_WINDOW = 10.0       # accept starts within this of the best anchor (typical set of a 14-D posterior: ~max - 7)
INIT_WIDTH_DROP = 1.0         # jitter sigma per parameter = the one-parameter step that lowers log_prob by this
INIT_TAU_WIDTH_FACTOR = 3.0   # log10(tau) jitter = this many conditional widths, so tau is explored around ~400 s
INIT_BOUND_MARGIN = 0.01      # ridge moves stop this fraction of the range inside the box bounds
INIT_SUMMARY_PARAMS = ("E_iso", "Gamma0", "A_star", "xi_e", "xi_e_r", "eps_e", "eps_B", "tau", "eps_e_r", "eps_B_r")


def ridge_move(x, labels, f):
    """Move x by factor f along the E_iso / eps / xi degeneracy ridge."""
    y = np.array(x, dtype=float)
    for name in RIDGE_UP:
        y[labels.index(name)] -= np.log10(f)
    for name in RIDGE_DOWN:
        y[labels.index(name)] += np.log10(f)
    for name in RIDGE_XI:
        y[labels.index(name)] *= f
    return y


def ridge_f_min(x, labels, bounds):
    """Smallest f <= 1 that keeps ridge_move(x, f) inside the box (minus INIT_BOUND_MARGIN)."""
    margin = INIT_BOUND_MARGIN * (bounds[:, 1] - bounds[:, 0])
    log_f = []
    for name in RIDGE_UP:
        i = labels.index(name)
        log_f.append(x[i] - (bounds[i, 1] - margin[i]))
    for name in RIDGE_DOWN:
        i = labels.index(name)
        log_f.append((bounds[i, 0] + margin[i]) - x[i])
    for name in RIDGE_XI:
        i = labels.index(name)
        log_f.append(np.log10((bounds[i, 0] + margin[i]) / x[i]))
    return 10 ** min(0.0, max(log_f))


def project_physical(x, labels, bounds):
    """Clip to the box and rescale any eps_e + eps_B > 1 pair just inside the constraint."""
    y = np.clip(np.asarray(x, dtype=float), bounds[:, 0], bounds[:, 1])
    for electrons, magnetic in EPS_SUM_PAIRS:
        e_i, b_i = labels.index(f"log10_{electrons}"), labels.index(f"log10_{magnetic}")
        total = 10 ** y[e_i] + 10 ** y[b_i]
        if total > 1.0:
            y[e_i] -= np.log10(total / (1.0 - 1e-4))
            y[b_i] -= np.log10(total / (1.0 - 1e-4))
    return y


def conditional_widths(x, lp_x, bounds, map_func):
    """Per parameter, the largest one-parameter step (either side, smaller of the two) lowering log_prob by
    <= INIT_WIDTH_DROP. Some directions are extremely narrow (e.g. p, p_r just above 2), so one global jitter
    fraction does not work; these conditional widths are lower limits of the marginal widths."""
    ndim = len(x)
    span = bounds[:, 1] - bounds[:, 0]
    fractions = np.geomspace(1e-6, 0.3, 28)
    trials = []
    for j in range(ndim):
        for sign in (-1, 1):
            for frac in fractions:
                y = np.array(x, dtype=float)
                y[j] += sign * frac * span[j]
                if bounds[j, 0] <= y[j] <= bounds[j, 1]:
                    trials.append((j, sign, frac, y))
    lps = list(map_func(_log_probability_from_context, [t[3] for t in trials]))
    widths = np.full(ndim, np.inf)
    for j in range(ndim):
        for sign in (-1, 1):
            side = [(frac, lp) for (jj, ss, frac, _), lp in zip(trials, lps) if jj == j and ss == sign]
            if not side or not np.isfinite(side[0][1]):  # box bound or physical constraint on this side
                continue
            ok = 0.0
            for frac, lp in side:                          # increasing step; stop at the first larger drop
                if not (np.isfinite(lp) and lp_x - lp <= INIT_WIDTH_DROP):
                    break
                ok = frac
            widths[j] = min(widths[j], max(ok, fractions[0]) * span[j])
    return widths


def physical_columns(samples, labels):
    """{name: physical values} for sampled-space columns (10 ** for log10_ labels)."""
    return {l.removeprefix("log10_"): (10 ** samples[:, j] if l.startswith("log10_") else samples[:, j])
            for j, l in enumerate(labels)}


def ridge_aware_initial_state(param_defs, nwalkers, rng, map_func, outdir):
    """Walkers spread over the known high-tau family instead of a tiny ball around one optimum.

    Each candidate = one of RIDGE_ANCHORS (xi_e = 1 inserted; round-robin, so every anchor and its xi_e_r value is
    represented), moved along the exact degeneracy ridge by f drawn log-uniformly between the box limit and 1
    (this spreads xi_e, xi_e_r, E_iso, A_star and the four eps together), plus small Gaussian jitter and extra
    log10(tau) jitter, projected inside the box and the eps_e + eps_B <= 1 constraints. Candidates are kept only if
    log_probability is finite and within INIT_LOGP_WINDOW of the best anchor, so no catastrophic region is seeded.
    """
    import csv

    labels = [param_label(p) for p in param_defs]
    if labels != PRODUCTION_LABELS:
        raise ValueError(f"Ridge-aware start needs the production wind / xi both layout.\n"
                         f"  expected: {PRODUCTION_LABELS}\n  actual:   {labels}")
    bounds = _sampled_bounds(param_defs)
    xi_e_i, tau_i = labels.index("xi_e"), labels.index("log10_tau")
    anchors = [(name, np.insert(np.array(vec, dtype=float), xi_e_i, 1.0)) for name, _, vec in RIDGE_ANCHORS]
    anchor_lp = list(map_func(_log_probability_from_context, [vec for _, vec in anchors]))
    print("Ridge anchors (xi_e = 1):")
    for (name, vec), lp, (_, ref, _) in zip(anchors, anchor_lp, RIDGE_ANCHORS):
        print(f"  {name:<28} log_prob = {lp:10.4f} (recorded {ref:.4f})  tau = {10 ** vec[tau_i]:6.1f} s  "
              f"ridge f range = [{ridge_f_min(vec, labels, bounds):.3f}, 1]")
    if not all(np.isfinite(anchor_lp)):
        raise RuntimeError("A ridge anchor has non-finite log_probability under the production prior.")
    threshold = max(anchor_lp) - INIT_LOGP_WINDOW
    best_k = int(np.argmax(anchor_lp))
    widths = conditional_widths(anchors[best_k][1], anchor_lp[best_k], bounds, map_func)
    widths[tau_i] *= INIT_TAU_WIDTH_FACTOR
    print(f"Jitter sigma per parameter (conditional width at dlogP = {INIT_WIDTH_DROP:g} around {anchors[best_k][0]}; "
          f"tau x{INIT_TAU_WIDTH_FACTOR:g}):")
    print("  " + ", ".join(f"{l}: {w:.3g}" for l, w in zip(labels, widths)))
    quota = int(np.ceil(nwalkers / len(anchors)))

    accepted, rows, n_tried = [], [], 0
    per_anchor = [0] * len(anchors)
    for _ in range(50):
        if len(accepted) >= nwalkers:
            break
        cands, meta = [], []
        for k, (name, vec) in enumerate(anchors):
            if per_anchor[k] >= quota:
                continue
            for _ in range(2 * (quota - per_anchor[k])):
                f = 10 ** rng.uniform(np.log10(ridge_f_min(vec, labels, bounds)), 0.0)
                x = ridge_move(vec, labels, f) + rng.normal(0.0, widths)
                cands.append(project_physical(x, labels, bounds))
                meta.append((k, name, f))
        if not cands:                                      # every anchor at quota but still short: lift the quota
            quota += 1
            continue
        lps = list(map_func(_log_probability_from_context, cands))
        n_tried += len(cands)
        for x, lp, (k, name, f) in zip(cands, lps, meta):
            if np.isfinite(lp) and lp >= threshold and per_anchor[k] < quota and len(accepted) < nwalkers:
                accepted.append((x, lp))
                per_anchor[k] += 1
                rows.append({"walker": len(accepted) - 1, "anchor": name, "ridge_f": f, "log_prob": lp,
                             **{l: v for l, v in zip(labels, x)}})
    if len(accepted) < nwalkers:
        raise RuntimeError(f"Only {len(accepted)} of {nwalkers} starting points within {INIT_LOGP_WINDOW} of the best "
                           f"anchor after {n_tried} candidates.")
    p0 = np.array([x for x, _ in accepted])
    lp0 = np.array([lp for _, lp in accepted])
    assert np.all(np.isfinite(lp0))

    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "initial_walkers.npy", p0)
    np.save(outdir / "initial_log_prob.npy", lp0)
    with open(outdir / "initial_walkers.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Initial ensemble: {nwalkers} walkers from {n_tried} candidates; per anchor "
          + ", ".join(f"{name.split()[-2] if 'rank' in name else name.split()[-1]}: {n}" for (name, _), n in zip(anchors, per_anchor)))
    print_initial_ranges(p0, lp0, labels)
    print(f"Saved: {outdir / 'initial_walkers.npy'}, initial_log_prob.npy, initial_walkers.csv")
    return p0, lp0


def posterior_initial_state(param_defs, nwalkers, rng, map_func, outdir, source_dirs, discard=50000, end=None):
    """Walkers drawn from the posterior already established by earlier production chains with the same prior.

    The ridge-aware strategy adapted to a known posterior: instead of hand-picked optima spread along the
    degeneracy, take nwalkers points from the post-discard part of each source chain (split evenly between the
    sources; distinct walkers within a source, random steps), converted to physical coordinates. Every point is
    re-evaluated and must have finite log_probability. The draws come from steps [discard, end) of every source.
    """
    import csv

    labels = [param_label(p) for p in param_defs]
    per = [nwalkers // len(source_dirs) + (k < nwalkers % len(source_dirs)) for k in range(len(source_dirs))]
    points, rows = [], []
    for source, k in zip(source_dirs, per):
        source = Path(source)
        config = json.loads((source / RUN_CONFIG_FILE).read_text())
        if config["labels"] != labels:
            raise ValueError(f"{source} has a different parameter layout: {config['labels']}")
        def prior_key(defs):
            return [(p.name, p.lower, p.upper, p.scale, p.gaussian_prior) for p in defs]
        if prior_key(make_param_defs(**config["param_kwargs"])) != prior_key(param_defs):
            raise ValueError(f"{source} was sampled with a different prior")
        reader = emcee.backends.HDFBackend(str(source / CHAIN_FILE), name="mcmc", read_only=True)
        chain = from_sampling_coords(reader.get_chain()[discard:end], labels, config.get("sampling_coords", "physical"))
        walkers = rng.permutation(chain.shape[1])[:k]
        steps = rng.integers(0, chain.shape[0], k)
        for w, t in zip(walkers, steps):
            points.append(chain[t, w])
            rows.append({"source": str(source), "source_step": int(discard + t), "source_walker": int(w)})
    p0 = np.array(points)
    lp0 = np.array(list(map_func(_log_probability_from_context, list(p0))))
    if not np.all(np.isfinite(lp0)):
        raise RuntimeError(f"{int((~np.isfinite(lp0)).sum())} posterior draws have non-finite log_probability")
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "initial_walkers.npy", p0)
    np.save(outdir / "initial_log_prob.npy", lp0)
    with open(outdir / "initial_walkers.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["walker", *rows[0], "log_prob", *labels])
        writer.writeheader()
        writer.writerows({"walker": i, **r, "log_prob": lp, **dict(zip(labels, x))}
                         for i, (r, lp, x) in enumerate(zip(rows, lp0, p0)))
    print(f"Initial ensemble: {nwalkers} posterior draws (steps {discard}-{end or 'end'}) from " + ", ".join(
        f"{Path(d).name}: {k}" for d, k in zip(source_dirs, per)))
    print_initial_ranges(p0, lp0, labels)
    q = np.log10(physical_columns(p0, labels)["E_iso"] * p0[:, labels.index("xi_e_r")])
    print(f"{'q_E_xir':<10} {q.min():>12.5g} {np.median(q):>12.5g} {q.max():>12.5g}   (log10 E_iso xi_e_r)")
    print(f"Saved: {outdir / 'initial_walkers.npy'} (physical coordinates), initial_log_prob.npy, initial_walkers.csv")
    return p0, lp0


def print_initial_ranges(p0, lp0, labels):
    phys = physical_columns(p0, labels)
    print(f"{'parameter':<10} {'min':>12} {'median':>12} {'max':>12}   (physical units)")
    for name in INIT_SUMMARY_PARAMS:
        v = phys[name]
        print(f"{name:<10} {v.min():>12.5g} {np.median(v):>12.5g} {v.max():>12.5g}")
    print(f"{'log_prob':<10} {lp0.min():>12.4f} {np.median(lp0):>12.4f} {lp0.max():>12.4f}")


def print_parameter_table(param_defs):
    """Final parameter table: order, label, bounds, prior shape, and the active physical constraints."""
    print(f"{'#':>2} {'parameter':<9} {'sampled as':<15} {'lower':>10} {'upper':>10}  prior")
    for j, p in enumerate(param_defs):
        if p.has_gaussian_prior():
            mean, sigma = p.get_prior_mean_sigma()
            shape = f"Gaussian in log10: N({mean:.4g}, {sigma:.4g}), truncated to the box"
        else:
            shape = "uniform in log10 (log-uniform)" if p.scale is Scale.LOG else "uniform (LINEAR)"
        print(f"{j:>2} {p.name:<9} {param_label(p):<15} {p.lower:>10.4g} {p.upper:>10.4g}  {shape}")
    print("Physical constraints (log_prior = -inf if violated): "
          + ", ".join(f"{e} + {b} <= 1" for e, b in EPS_SUM_PAIRS))


# Worker-side wrapper: a model exception at an extreme point becomes -inf (and is logged) instead of killing an
# overnight run; the checkpoint diagnostics count the logged failures. The likelihood itself is untouched.
_WORKER_ERROR_LOG = None
_WORKER_COORDS = (None, "physical")         # (physical labels, sampling coordinates)


def _init_mcmc_worker(context, error_log, labels=None, coords="physical", numerics=None):
    """Pool-worker set-up: likelihood context, error log, sampling coordinates and (explicitly, independent of the
    process start method) the model resolution / rtol of the run."""
    global _WORKER_ERROR_LOG, _WORKER_COORDS
    if numerics is not None:
        epm.set_numerics(*numerics)
    _init_early_phase_context(context)
    _WORKER_ERROR_LOG = error_log
    _WORKER_COORDS = (labels, coords)


def _mcmc_log_probability(params_array):
    """Sampler coordinates -> physical sampled space (identity for "physical") -> the unchanged log_probability,
    plus the coordinate change's log-Jacobian (0 except for u_r), so the physical prior is unchanged."""
    labels, coords = _WORKER_COORDS
    try:
        if coords == "physical":
            return _log_probability_from_context(params_array)
        lp = _log_probability_from_context(from_sampling_coords(params_array, labels, coords))
        return lp + sampling_log_jacobian(params_array, labels, coords) if np.isfinite(lp) else -np.inf
    except Exception as exc:
        if _WORKER_ERROR_LOG is not None:
            with open(_WORKER_ERROR_LOG, "a") as fh:
                fh.write(f"{datetime.now().isoformat()} pid={os.getpid()} {type(exc).__name__}: {exc} "
                         f"params={np.array2string(np.asarray(params_array), precision=10, max_line_width=10**6)}\n")
        return -np.inf


CHAIN_FILE = "chain.h5"
RUN_CONFIG_FILE = "run_config.json"


def provenance():
    """Code and environment identification recorded in run_config.json (git commit, package versions, host)."""
    import platform
    import subprocess
    import sys

    import h5py
    import scipy
    import VegasAfterglow
    repo = Path(__file__).resolve().parent.parent

    def git(*cmd):
        try:
            return subprocess.run(["git", *cmd], cwd=repo, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as exc:                       # git missing: record why instead of failing the run
            return f"unavailable ({exc})"
    return {
        "git_commit": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_modified_tracked_files": git("status", "--porcelain", "--untracked-files=no").splitlines(),
        "python": sys.version.split()[0], "python_executable": sys.executable, "numpy": np.__version__,
        "scipy": scipy.__version__, "emcee": emcee.__version__, "h5py": h5py.__version__,
        "VegasAfterglow": getattr(VegasAfterglow, "__version__", "unknown"), "VegasAfterglow_path": VegasAfterglow.__file__,
        "platform": platform.platform(), "host": platform.node(), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }


def _write_json(path, payload):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(path)


def run_production_mcmc(param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling, outdir, nsteps,
                        nwalkers=None, npool=20, seed=None, resume=False, run_config=None, log_every=500,
                        sampling_coords="physical", init_from=None, init_window=(50000, None)):
    """emcee with the DE move mixture and an HDF backend; runs (or continues) the SAME chain to `nsteps` total steps.

    New run: ridge-aware start (or posterior draws from init_from chains), backend reset, run_config.json written.
    resume=True: reopens <outdir>/chain.h5 and appends from its last state (emcee restores the stored random state),
    never restarting the chain. sampling_coords (early_phase_model.SAMPLING_COORDS) chooses the sampler's coordinates;
    the HDF chain stores those coordinates, run_config.json records them, and the likelihood is always evaluated on
    the physical vector.
    """
    from concurrent.futures import ProcessPoolExecutor

    labels = [param_label(p) for p in param_defs]
    ndim = len(param_defs)
    outdir = Path(outdir)
    chain_path = outdir / CHAIN_FILE
    config_path = outdir / RUN_CONFIG_FILE
    context = (param_defs, xrt_data, i_data, xrt_index_data, spectral_index_cooling)
    _init_early_phase_context(context)

    if resume:
        config = json.loads(config_path.read_text())
        if config["labels"] != labels:
            raise ValueError(f"Labels of {config_path} differ from the requested configuration:\n"
                             f"  saved: {config['labels']}\n  now:   {labels}")
        nwalkers = config["nwalkers"]
        sampling_coords = config.get("sampling_coords", "physical")
        saved = (tuple(float(v) for v in config["model_resolution"]), float(config["model_rtol"]))
        if (tuple(epm.MODEL_RESOLUTION), epm.MODEL_RTOL) != saved:
            raise ValueError(f"Model resolution/rtol now {(epm.MODEL_RESOLUTION, epm.MODEL_RTOL)} differ from the chain's {saved}")
        if npool != config["npool"]:
            print(f"Note: resuming with {npool} workers (chain started with {config['npool']}).")
    else:
        if chain_path.exists():
            raise FileExistsError(f"{chain_path} exists; use --resume {outdir} to continue it.")
        nwalkers = resolve_nwalkers(nwalkers, ndim)
        seed = int(np.random.SeedSequence().entropy % 2**31) if seed is None else seed
        config = {
            **(run_config or {}), "labels": labels, "ndim": ndim, "nwalkers": nwalkers, "seed": seed, "npool": npool,
            "moves": MOVES_DESCRIPTION,
            "initialisation": (f"posterior draws from {[str(d) for d in init_from]}, steps {init_window[0]}-"
                               f"{init_window[1] or 'end'}" if init_from
                               else "ridge-aware (RIDGE_ANCHORS + degeneracy ridge + jitter)"),
            "sampling_coords": sampling_coords, "sampling_labels": sampling_labels(labels, sampling_coords),
            "bounds": [{"name": p.name, "label": param_label(p), "lower": p.lower, "upper": p.upper,
                        "scale": "log10" if p.scale is Scale.LOG else "linear",
                        "gaussian_prior": p.get_prior_mean_sigma() if p.has_gaussian_prior() else None}
                       for p in param_defs],
            "physical_constraints": [f"{e} + {b} <= 1" for e, b in EPS_SUM_PAIRS],
            "model_resolution": list(epm.MODEL_RESOLUTION), "model_rtol": epm.MODEL_RTOL,
            "numerical_settings": epm.numerical_settings(), "provenance": provenance(),
            "spectral_index_cooling": spectral_index_cooling,
            "emcee_version": emcee.__version__, "backend": str(chain_path), "created": datetime.now().isoformat(),
            "segments": [], "total_steps": 0,
        }

    error_log = str(outdir / "worker_errors.log")
    with ProcessPoolExecutor(max_workers=npool, initializer=_init_mcmc_worker,
                             initargs=(context, error_log, labels, sampling_coords,
                                       (epm.MODEL_RESOLUTION, epm.MODEL_RTOL))) as pool:
        if resume:
            backend = emcee.backends.HDFBackend(str(chain_path), name="mcmc")
            if backend.shape != (nwalkers, ndim):
                raise ValueError(f"{chain_path} holds shape {backend.shape}, expected {(nwalkers, ndim)}")
            initial = backend.get_last_sample()
            print(f"Resuming {chain_path} at step {backend.iteration} (target {nsteps} total steps).")
        else:
            rng = np.random.default_rng(seed)
            if init_from:
                initial, _ = posterior_initial_state(param_defs, nwalkers, rng, pool.map, outdir, init_from,
                                                     discard=init_window[0], end=init_window[1])
            else:
                initial, _ = ridge_aware_initial_state(param_defs, nwalkers, rng, pool.map, outdir)
            initial = to_sampling_coords(initial, labels, sampling_coords)
            lp_check = np.array(list(pool.map(_mcmc_log_probability, list(initial))))
            if not np.all(np.isfinite(lp_check)):
                raise RuntimeError("Initial walkers have non-finite log_probability in sampling coordinates.")
            print(f"Sampling coordinates: {sampling_coords} ({sampling_labels(labels, sampling_coords)}); all "
                  f"{nwalkers} initial walkers finite through the sampler's log_probability "
                  f"(range {lp_check.min():.3f} to {lp_check.max():.3f})", flush=True)
            backend = emcee.backends.HDFBackend(str(chain_path), name="mcmc")
            backend.reset(nwalkers, ndim)
            _write_json(config_path, config)
        done = backend.iteration
        remaining = nsteps - done
        if remaining <= 0:
            print(f"Chain already has {done} >= {nsteps} steps; nothing to run.")
            return config

        sampler = emcee.EnsembleSampler(nwalkers, ndim, _mcmc_log_probability, pool=pool,
                                        moves=production_moves(), backend=backend)
        if not resume:
            sampler.random_state = np.random.RandomState(seed).get_state()
        print(f"Sampling: {nwalkers} walkers x {ndim} parameters, {npool} workers, steps {done} -> {nsteps}; "
              f"moves: {MOVES_DESCRIPTION}", flush=True)
        segment = {"start_step": int(done), "target_step": nsteps, "npool": npool, "started": datetime.now().isoformat(),
                   "provenance": provenance(), "model_resolution": list(epm.MODEL_RESOLUTION), "model_rtol": epm.MODEL_RTOL}
        t0 = time.time()
        last_acc = backend.accepted.copy()
        for k, state in enumerate(sampler.sample(initial, iterations=remaining, store=True, progress=False), start=1):
            if k % log_every and k != remaining:
                continue
            now = time.time()
            accepted = backend.accepted
            block = log_every if k % log_every == 0 else k % log_every
            block_acc = (accepted - last_acc) / block
            lp = state.log_prob
            rate = (now - t0) / k
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] step {done + k:>7d}/{nsteps}  {rate:6.3f} s/step  "
                  f"ETA {(remaining - k) * rate / 3600:5.2f} h  acc(block) mean {block_acc.mean():.3f} "
                  f"min {block_acc.min():.3f}  acc(total) {np.mean(accepted / (done + k)):.3f}  "
                  f"log_prob max {lp.max():.3f} median {np.median(lp):.3f} min {lp.min():.3f}", flush=True)
            last_acc = accepted.copy()
        wall = time.time() - t0

    segment.update(end_step=int(backend.iteration), finished=datetime.now().isoformat(), wall_seconds=wall,
                   seconds_per_step=wall / remaining)
    config["segments"].append(segment)
    config["total_steps"] = int(backend.iteration)
    config["total_wall_seconds"] = sum(s["wall_seconds"] for s in config["segments"])
    _write_json(config_path, config)
    print(f"Segment done: {remaining} steps in {wall / 3600:.2f} h ({wall / remaining:.3f} s/step); "
          f"chain length now {backend.iteration}.", flush=True)
    return config


# ---------------------------------------------------------------------------------------------------------------
# Checkpoint diagnostics and the autonomous continue / stop decision
# ---------------------------------------------------------------------------------------------------------------

EXIT_CONVERGED, EXIT_CONTINUE, EXIT_FAILURE = 0, 10, 20
# Serious-failure thresholds (any one stops the overnight run)
FAIL_ACCEPTANCE = (0.02, 0.8)        # pathological mean acceptance fraction
FAIL_WALKER_FRACTION = 0.10          # > 10% of walkers frozen or isolated
FROZEN_ACCEPTANCE = 0.01             # a walker accepting < 1% of proposals (over the chain or the last quarter)
ISOLATED_LOGP_GAP = 15.0             # walker median log_prob in the last quarter this far below the ensemble median
FAIL_REACH_LOGP = 5.0                # best log_prob must come within this of REFERENCE_LOG_POSTERIOR
TAU_REGION = (200.0, 1000.0)         # posterior median tau [s] must sit in the known high-tau region
FAIL_LOGP_DRIFT = 5.0                # post-burn log_prob median still rising this much (from 60k on: serious)
FAIL_ERROR_FRACTION = 1e-3           # logged model exceptions per likelihood evaluation
# Convergence (all required)
CONVERGED_TAU_MULTIPLE = 50          # N_postburn >= 50 tau_ac,max ...
CONVERGED_TAU_STABILITY = 0.2        # ... with tau_ac,max changing < 20% between 75% and 100% of the post-burn chain
CONVERGED_MEAN_SHIFT = 0.25          # every parameter's mean differs < 0.25 sigma between post-burn halves
CONVERGED_LOGP_DRIFT = 1.0
CONVERGED_WINDOW_SHIFT = 0.25        # every median within 0.25 sigma of the overall in each late window
WINDOWS = 4                          # late windows for quantile / tail / ridge stability
BOUND_FRACTION = 0.01                # "near a bound" = within min(1% of the sampled range, 0.5 sigma of the marginal),
BOUND_SIGMA = 0.5                    # so a narrow interior peak close to a bound (p_r ~ 2.005) is not called a pile-up
PILEUP_FRACTION = 0.05               # flagged when more than 5% of the samples sit in that edge zone
KEY_PAIRS = [("log10_E_iso", "xi_e"), ("log10_E_iso", "xi_e_r"), ("xi_e", "xi_e_r"), ("log10_E_iso", "log10_A_star"),
             ("log10_E_iso", "log10_eps_e"), ("log10_E_iso", "log10_eps_e_r"), ("log10_eps_e_r", "log10_eps_B_r"),
             ("xi_e_r", "log10_eps_e_r"), ("xi_e_r", "log10_eps_B_r"), ("log10_tau", "log10_eps_e_r"),
             ("log10_tau", "log10_Gamma0"), ("p_r", "log10_eps_e_r")]


def _autocorr(chain):
    """emcee's integrated autocorrelation time (walker-averaged ACF, auto window c = 5), except that walkers whose
    series is constant in this window are left out of the average (their 0/0 ACF would turn the result into NaN).
    Identical to emcee.autocorr.integrated_time(chain, c=5, tol=0) when every walker moves."""
    n, nwalkers, ndim = chain.shape
    tau = np.full(ndim, np.nan)
    for d in range(ndim):
        moving = [k for k in range(nwalkers) if np.ptp(chain[:, k, d]) > 0]
        if not moving:
            continue
        f = np.mean([emcee.autocorr.function_1d(chain[:, k, d]) for k in moving], axis=0)
        taus = 2.0 * np.cumsum(f) - 1.0
        tau[d] = taus[emcee.autocorr.auto_window(taus, 5)]
    return tau


def provisional_discard(chain):
    """Provisional burn-in: 5 tau_ac,max estimated on the second half (at least n/5, at most n/2). Not final."""
    n = len(chain)
    tau_half = _autocorr(chain[n // 2:])
    return int(min(n // 2, max(n // 5, 5 * np.nanmax(tau_half))))


def check_chain(outdir, final=False):
    """Diagnostics of <outdir>/chain.h5 at its current length; writes a checkpoint directory and returns an exit code.

    EXIT_CONVERGED (0): convergence criteria met. EXIT_CONTINUE (10): healthy but too short. EXIT_FAILURE (20):
    serious sampling failure. Also refreshes samples.npy / log_probs.npy / labels.txt / top_k_*.npy /
    bestfit_params.txt in <outdir> (post provisional burn-in, thinned) for the existing plotting tools.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    config = json.loads((outdir / RUN_CONFIG_FILE).read_text())
    labels = config["labels"]
    param_defs = make_param_defs(**config["param_kwargs"])
    epm.set_numerics(config.get("model_resolution"), config.get("model_rtol"))   # plots use the chain's own settings
    if [param_label(p) for p in param_defs] != labels:
        raise ValueError("Parameter layout rebuilt from run_config.json differs from the saved labels.")
    bounds = _sampled_bounds(param_defs)
    span = bounds[:, 1] - bounds[:, 0]
    coords = config.get("sampling_coords", "physical")      # old runs: the chain is in physical coordinates
    slabels = sampling_labels(labels, coords)
    reader = emcee.backends.HDFBackend(str(outdir / CHAIN_FILE), name="mcmc", read_only=True)
    chain_s = reader.get_chain()                               # sampler coordinates: mixing diagnostics
    chain = from_sampling_coords(chain_s, labels, coords)      # physical coordinates: posterior summaries
    log_prob = reader.get_log_prob()
    if coords != "physical":                                   # stored target includes the log-Jacobian: remove it
        with np.errstate(invalid="ignore"):
            log_prob = np.subtract(log_prob, sampling_log_jacobian(chain_s, labels, coords))
    n, nwalkers, ndim = chain.shape
    ckdir = outdir / "checkpoints" / f"step_{n:06d}"
    ckdir.mkdir(parents=True, exist_ok=True)
    acc = reader.accepted / n
    stage_serious_drift = n >= 60000

    lines, serious, warnings = [], [], []

    def say(text=""):
        lines.append(text)
        print(text, flush=True)

    say(f"=== Chain diagnostics at step {n} ({datetime.now():%Y-%m-%d %H:%M:%S}) ===")
    say(f"chain: {outdir / CHAIN_FILE}; ndim {ndim}, nwalkers {nwalkers}, workers {config['npool']}, seed {config['seed']}")
    say(f"moves: {config['moves']}; prior {config['prior']}, environment {config['environment']}, xi {config['xi']}; "
        f"sampling coordinates {coords}")
    wall = sum(s["wall_seconds"] for s in config["segments"])
    say(f"wall-clock sampling time so far: {wall / 3600:.2f} h over {len(config['segments'])} segment(s)")
    prov = config.get("provenance", {})
    say(f"model resolution {tuple(epm.MODEL_RESOLUTION)}, rtol {epm.MODEL_RTOL:g}; code {prov.get('git_commit', 'not recorded')}; "
        f"Python {prov.get('python', '?')}, NumPy {prov.get('numpy', '?')}")

    # numerical health
    n_nonfinite = int((~np.isfinite(chain_s)).sum())
    n_bad_lp = int(np.isnan(log_prob).sum() + np.isposinf(log_prob).sum())
    n_neg_inf_now = int(np.isneginf(log_prob[-1]).sum())
    err_path = outdir / "worker_errors.log"
    n_errors = sum(1 for _ in open(err_path)) if err_path.exists() else 0
    n_evals = n * nwalkers
    say(f"numerical: non-finite coordinates {n_nonfinite}, NaN/+inf log_prob {n_bad_lp}, walkers at -inf now "
        f"{n_neg_inf_now}, logged model exceptions {n_errors} ({n_errors / n_evals:.2e} per evaluation)")
    if n_nonfinite or n_bad_lp or n_neg_inf_now or n_errors > FAIL_ERROR_FRACTION * n_evals:
        serious.append("numerical failures in the chain")
    elif n_errors:
        warnings.append(f"{n_errors} model exceptions logged (treated as -inf)")

    # acceptance
    med_acc = float(np.median(acc))
    q = max(1, n // 4)
    moved_recent = np.mean(np.any(chain_s[-q:][1:] != chain_s[-q:][:-1], axis=2), axis=0)
    frozen = np.where((acc < FROZEN_ACCEPTANCE) | (moved_recent < FROZEN_ACCEPTANCE))[0]
    say(f"acceptance: mean {acc.mean():.3f}, median {med_acc:.3f}, min {acc.min():.3f} (walker {acc.argmin()}), "
        f"max {acc.max():.3f}; last-quarter move fraction mean {moved_recent.mean():.3f} min {moved_recent.min():.3f}")
    say(f"frozen walkers (acceptance < {FROZEN_ACCEPTANCE} overall or in the last quarter): {frozen.tolist() or 'none'}")
    if not FAIL_ACCEPTANCE[0] <= acc.mean() <= FAIL_ACCEPTANCE[1]:
        serious.append(f"pathological mean acceptance {acc.mean():.3f}")
    if len(frozen) > FAIL_WALKER_FRACTION * nwalkers:
        serious.append(f"{len(frozen)} frozen walkers")
    elif len(frozen):
        warnings.append(f"{len(frozen)} frozen walker(s): {frozen.tolist()}")

    # isolated walkers
    walker_lp = np.median(log_prob[-q:], axis=0)
    ens_lp = float(np.median(walker_lp))
    isolated = np.where(walker_lp < ens_lp - ISOLATED_LOGP_GAP)[0]
    say(f"last-quarter walker median log_prob: ensemble median {ens_lp:.3f}, lowest {walker_lp.min():.3f}; "
        f"isolated (> {ISOLATED_LOGP_GAP:g} below): {isolated.tolist() or 'none'}")
    if len(isolated) > FAIL_WALKER_FRACTION * nwalkers:
        serious.append(f"{len(isolated)} isolated walkers (log_prob far below the ensemble)")
    elif len(isolated):
        warnings.append(f"{len(isolated)} isolated walker(s): {isolated.tolist()}")

    # burn-in, autocorrelation, stationarity
    discard = provisional_discard(chain_s)
    post_s = chain_s[discard:]
    post = chain[discard:]
    post_lp = log_prob[discard:]
    n_post = len(post)
    tau = _autocorr(post_s)
    tau_75 = _autocorr(post_s[: int(0.75 * n_post)])
    tau_50 = _autocorr(post_s[: n_post // 2])
    tau_max, tau_med = float(np.nanmax(tau)), float(np.nanmedian(tau))
    tau_change = abs(tau_max - np.nanmax(tau_75)) / tau_max
    ess = n_post * nwalkers / tau
    say(f"provisional discard: {discard} steps (5 tau_ac,max of the second half, clamped to [n/5, n/2]); "
        f"post-burn length {n_post}")
    say(f"autocorrelation time (sampling coordinates): max {tau_max:.0f} ({slabels[int(np.nanargmax(tau))]}), median {tau_med:.0f}; "
        f"N_post / tau_max = {n_post / tau_max:.1f} (target >= {CONVERGED_TAU_MULTIPLE}); tau_max at 50/75/100% of "
        f"post-burn: {np.nanmax(tau_50):.0f} / {np.nanmax(tau_75):.0f} / {tau_max:.0f} (change {100 * tau_change:.0f}%)")
    halves = post_s[: n_post // 2].reshape(-1, ndim), post_s[n_post // 2:].reshape(-1, ndim)
    shift = np.abs(halves[1].mean(0) - halves[0].mean(0)) / post_s.reshape(-1, ndim).std(0)
    q4 = max(1, n_post // 4)
    lp_drift = float(np.median(post_lp[-q4:]) - np.median(post_lp[:q4]))
    say(f"stationarity: largest mean shift between post-burn halves {shift.max():.2f} sigma ({slabels[int(shift.argmax())]}); "
        f"post-burn log_prob median drift (last - first quarter) {lp_drift:+.2f}")
    if lp_drift > FAIL_LOGP_DRIFT:
        (serious if stage_serious_drift else warnings).append(f"log_prob still rising by {lp_drift:.1f} after burn-in")
    # split-R-hat over walkers (ensemble walkers are not independent; used only as a mixing indicator)
    halves_w = np.concatenate([post_s[: n_post // 2], post_s[n_post // 2: 2 * (n_post // 2)]], axis=1)
    m_chain = halves_w.mean(0)
    w_var = halves_w.var(0, ddof=1).mean(0)
    b_var = (n_post // 2) * m_chain.var(0, ddof=1)
    rhat = np.sqrt(((n_post // 2 - 1) / (n_post // 2) * w_var + b_var / (n_post // 2)) / w_var)

    # high-tau region reached
    flat = post.reshape(-1, ndim)
    flat_lp = post_lp.reshape(-1)
    phys = physical_columns(flat, labels)
    best_lp = float(log_prob.max())
    best_post_lp = float(post_lp.max())
    tau_med_s = float(np.median(phys["tau"]))
    say(f"best log_prob {best_lp:.4f} overall, {best_post_lp:.4f} after the discard (reference optimum "
        f"{REFERENCE_LOG_POSTERIOR}); posterior median tau {tau_med_s:.1f} s (known high-tau region "
        f"{TAU_REGION[0]:g}-{TAU_REGION[1]:g} s)")
    if best_post_lp < REFERENCE_LOG_POSTERIOR - FAIL_REACH_LOGP or not TAU_REGION[0] <= tau_med_s <= TAU_REGION[1]:
        serious.append("chain is not in the known high-probability high-tau region")

    # per-parameter table
    say(f"{'coordinate':<15} {'tau_ac':>8} {'ESS':>8} {'R-hat':>7} {'shift':>6} | {'parameter':<15} {'median':>11} {'-1sig':>10} "
        f"{'+1sig':>10} {'@lower':>7} {'@upper':>7}   (tau_ac..shift: sampling coordinate; median..: physical sampled space;"
        f" @: % of samples within min(1% range, 0.5 sigma) of the bound)")
    pileups = []
    edge = np.minimum(BOUND_FRACTION * span, BOUND_SIGMA * flat.std(0))
    near_lo = (flat - bounds[:, 0] < edge).mean(0)
    near_hi = (bounds[:, 1] - flat < edge).mean(0)
    for j, l in enumerate(labels):
        lo, med, hi = np.percentile(flat[:, j], [16, 50, 84])
        say(f"{slabels[j]:<15} {tau[j]:>8.0f} {ess[j]:>8.0f} {rhat[j]:>7.3f} {shift[j]:>6.2f} | {l:<15} {med:>11.5g} "
            f"{med - lo:>10.4g} {hi - med:>10.4g} {100 * near_lo[j]:>6.1f}% {100 * near_hi[j]:>6.1f}%")
        for side, frac in (("lower", near_lo[j]), ("upper", near_hi[j])):
            if frac > PILEUP_FRACTION:
                pileups.append(f"{l} {side} ({100 * frac:.1f}% of samples in the edge zone)")
    phys_sum = {f"{e} + {b}": phys[e] + phys[b] for e, b in EPS_SUM_PAIRS}
    for name, total in phys_sum.items():
        frac = float((total > 0.99).mean())
        say(f"physical boundary {name} > 0.99: {100 * frac:.1f}% of samples (max {total.max():.5f})")
        if frac > PILEUP_FRACTION:
            pileups.append(f"{name} = 1 ({100 * frac:.1f}%)")
    say(f"boundary pile-ups: {'; '.join(pileups) if pileups else 'none'}")

    # correlations
    corr = np.corrcoef(flat.T)
    say("correlations (sampled space): " + ", ".join(
        f"{a}-{b} {corr[labels.index(a), labels.index(b)]:+.2f}" for a, b in KEY_PAIRS if a in labels and b in labels))
    np.savetxt(ckdir / "correlation_matrix.csv", corr, delimiter=",", header=",".join(labels), fmt="%.4f")
    ridge_product = np.log10(phys["E_iso"] * phys["xi_e_r"])
    say(f"log10(E_iso * xi_e_r): median {np.median(ridge_product):.3f}, 16-84% width "
        f"{np.diff(np.percentile(ridge_product, [16, 84]))[0]:.3f} dex (log10 E_iso width "
        f"{np.diff(np.percentile(np.log10(phys['E_iso']), [16, 84]))[0]:.3f} dex)")

    # physical-coordinate autocorrelation (differs from the sampled coordinates only where they are transformed)
    if coords != "physical":
        tau_phys = _autocorr(post)
        say(f"autocorrelation time (physical coordinates): max {np.nanmax(tau_phys):.0f} ({labels[int(np.nanargmax(tau_phys))]}), "
            f"median {np.nanmedian(tau_phys):.0f}; " + ", ".join(
                f"{l} {tau_phys[labels.index(l)]:.0f}" for l in labels if sampling_labels(labels, coords)[labels.index(l)] != l))
    else:
        tau_phys = tau
    # late-window stability: post-burn chain split into WINDOWS equal windows
    win_edges = np.linspace(0, n_post, WINDOWS + 1).astype(int)
    windows = list(zip(win_edges[:-1], win_edges[1:]))
    win_dev = np.zeros((ndim, 3))
    for j in range(ndim):
        allq = np.percentile(post[..., j], [16, 50, 84])
        sig = 0.5 * (allq[2] - allq[0])
        wq = np.array([np.percentile(post[a:b, :, j], [16, 50, 84]) for a, b in windows])
        win_dev[j] = np.abs(np.subtract(wq, allq)).max(0) / sig
    say(f"late-window stability ({WINDOWS} post-burn windows of {n_post // WINDOWS} steps): max |window - overall| "
        f"16/50/84% in units of the half-width; worst median {win_dev[:, 1].max():.2f} ({labels[int(win_dev[:, 1].argmax())]}), "
        f"worst 16% {win_dev[:, 0].max():.2f}, worst 84% {win_dev[:, 2].max():.2f}")
    say("  per parameter (16/50/84): " + ", ".join(f"{l} {d[0]:.2f}/{d[1]:.2f}/{d[2]:.2f}" for l, d in zip(labels, win_dev)))
    pr_i = labels.index("p_r")
    tail = [np.percentile(post[a:b, :, pr_i], [84, 97.5]) for a, b in windows]
    say("p_r upper tail per window (84% / 97.5%): " + ", ".join(f"{t[0]:.4f}/{t[1]:.4f}" for t in tail))
    e_i, xr_i = labels.index("log10_E_iso"), labels.index("xi_e_r")
    ridge_w = []
    for a, b in windows:
        lq = np.add(post[a:b, :, e_i], np.log10(post[a:b, :, xr_i])).ravel()
        ridge_w.append((np.median(lq), np.diff(np.percentile(lq, [16, 84]))[0],
                        np.corrcoef(post[a:b, :, e_i].ravel(), post[a:b, :, xr_i].ravel())[0, 1]))
    say("E_iso-xi_e_r ridge per window (median / 16-84 width of log10(E_iso xi_e_r), corr(log E_iso, xi_e_r)): "
        + ", ".join(f"{r[0]:.3f}/{r[1]:.3f}/{r[2]:+.2f}" for r in ridge_w))

    # decision
    converged = (not serious and n_post >= CONVERGED_TAU_MULTIPLE * tau_max and tau_change < CONVERGED_TAU_STABILITY
                 and shift.max() < CONVERGED_MEAN_SHIFT and abs(lp_drift) < CONVERGED_LOGP_DRIFT
                 and win_dev[:, 1].max() < CONVERGED_WINDOW_SHIFT and not len(frozen) and not len(isolated))
    code = EXIT_FAILURE if serious else EXIT_CONVERGED if converged else EXIT_CONTINUE
    say(f"serious problems: {serious or 'none'}")
    say(f"warnings: {warnings or 'none'}")
    verdict = {EXIT_FAILURE: "SERIOUS SAMPLING FAILURE - stop and inspect",
               EXIT_CONVERGED: "CONVERGENCE CRITERIA MET",
               EXIT_CONTINUE: "healthy so far but not converged - continue the same chain"}[code]
    say(f"DECISION at step {n}: {verdict} (exit code {code})")
    if final:
        rec = ("C - sampler has mixing problems; do NOT simply extend" if serious else
               "A - converged / probably sufficient (consider D: an independent seed as a cross-check)" if converged else
               "B - continue the same chain somewhat longer" if n_post / tau_max >= 20 and tau_change < 0.5 else
               "B/C - continue, but autocorrelation estimates are not yet stable; inspect the traces first")
        say(f"AUTOMATIC RECOMMENDATION: {rec}")

    # exports for the existing tools (thinned post-burn samples) and plots
    thin = max(1, int(0.5 * np.nanmin(tau)))
    samples = from_sampling_coords(reader.get_chain(discard=discard, thin=thin, flat=True), labels, coords)
    sample_lp = reader.get_log_prob(discard=discard, thin=thin, flat=True)
    from grb.results import top_k_samples
    top_params, top_lp = top_k_samples(flat, flat_lp, top_k=10)
    for target in (ckdir, outdir):
        np.save(target / "samples.npy", samples)
        np.save(target / "log_probs.npy", sample_lp)
        np.save(target / "top_k_params.npy", top_params)
        np.save(target / "top_k_log_probs.npy", top_lp)
        (target / "labels.txt").write_text("\n".join(labels) + "\n")
    say(f"exported {len(samples)} samples (discard {discard}, thin {thin}) to samples.npy / log_probs.npy")

    diagnostics = {
        "step": n, "discard_provisional": discard, "thin": thin, "n_post": n_post, "nwalkers": nwalkers, "ndim": ndim,
        "acceptance_mean": float(acc.mean()), "acceptance_per_walker": acc.tolist(),
        "sampling_coords": coords, "tau_ac": dict(zip(slabels, tau.tolist())), "tau_ac_max": tau_max, "tau_ac_median": tau_med,
        "tau_ac_max_at_50_75_100": [float(np.nanmax(tau_50)), float(np.nanmax(tau_75)), tau_max],
        "ess": dict(zip(slabels, ess.tolist())), "rhat_walkers": dict(zip(slabels, rhat.tolist())),
        "mean_shift_sigma": dict(zip(slabels, shift.tolist())), "log_prob_drift": lp_drift,
        "tau_ac_physical": dict(zip(labels, tau_phys.tolist())),
        "window_quantile_deviation_sigma": {l: d.tolist() for l, d in zip(labels, win_dev)},
        "p_r_tail_per_window": [t.tolist() for t in tail], "ridge_per_window": [list(map(float, r)) for r in ridge_w],
        "model_resolution": list(epm.MODEL_RESOLUTION), "model_rtol": epm.MODEL_RTOL,
        "best_log_prob": best_lp, "best_log_prob_post_burn": best_post_lp, "median_tau_s": tau_med_s, "frozen_walkers": frozen.tolist(),
        "isolated_walkers": isolated.tolist(), "boundary_pileups": pileups, "serious": serious, "warnings": warnings,
        "decision": verdict, "exit_code": code, "wall_seconds": wall,
    }
    _write_json(ckdir / "diagnostics.json", diagnostics)
    (ckdir / "summary.txt").write_text("\n".join(lines) + "\n")
    # plots last: a plotting failure is reported but never changes or blocks the decision above
    for name, plot in (("chain diagnostics", lambda: _plot_chain_diagnostics(plt, chain_s, log_prob, acc, slabels, discard,
                                                                             tau, ckdir, flat, phys, labels)),
                       ("best fit / corner", lambda: _plot_fit_and_corner(param_defs, config, ckdir, outdir))):
        try:
            plot()
        except Exception as exc:
            plt.close("all")
            say(f"WARNING: {name} plots failed ({type(exc).__name__}: {exc}); decision unaffected")
            (ckdir / "summary.txt").write_text("\n".join(lines) + "\n")
    return code, ckdir


def _plot_fit_and_corner(param_defs, config, ckdir, outdir):
    """Best-fit light curve, spectral-index comparison, bestfit_params.txt and corner plot (existing plotting code)."""
    from grb.plotting import plot_corner
    from early_phase_model import XRT_BAND, HOST_AV_PARAM, _evaluate_fn, build_model_from_params, load_fit_data, \
        make_fitter, model_spectral_index
    from early_phase_plotting import plot_best_fit, plot_spectral_index_comparison
    try:
        from VegasAfterglow.fitting.utils import _build_transformer
    except ImportError:
        from VegasAfterglow.runner import _build_transformer

    xrt_data, i_data, xrt_index_data = load_fit_data(include_spectral_index=True)
    fitter = make_fitter(xrt_data, i_data, config["environment"])
    fitter._to_params = _build_transformer([p.to_param_def() for p in param_defs])
    best = np.load(ckdir / "top_k_params.npy")[0]
    labels = config["labels"]
    for target in (ckdir, outdir):
        plot_best_fit(fitter, best, xrt_data, i_data, target, XRT_BAND, HOST_AV_PARAM,
                      evaluate_fn=_evaluate_fn(best, param_defs))
        save_bestfit_params(target, param_defs, xrt_data, i_data, config["environment"])
    if xrt_index_data is not None:
        plot_spectral_index_comparison(
            best, labels, ckdir, xrt_index_data,
            beta_model_fn=(lambda times: model_spectral_index(build_model_from_params(best, param_defs)[0], times))
            if config["environment"] != "ism" else None,
        )
    plot_corner(ckdir, max_samples=20000)


def _plot_chain_diagnostics(plt, chain, log_prob, acc, labels, discard, tau, ckdir, flat, phys, flat_labels=None):
    """Traces and autocorrelation use `chain` / `labels` (sampling coordinates); pair plots and marginals use the
    physical `flat` / `flat_labels` (default: the same labels)."""
    flat_labels = labels if flat_labels is None else flat_labels
    n, nwalkers, ndim = chain.shape
    stride = max(1, n // 3000)
    steps = np.arange(0, n, stride)

    fig, axes = plt.subplots(ndim, 1, figsize=(11, 1.6 * ndim), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(steps, chain[::stride, :, j], lw=0.3, alpha=0.35)
        ax.axvline(discard, color="k", ls="--", lw=0.8)
        ax.set_ylabel(labels[j], fontsize=7)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("step (dashed: provisional discard)")
    fig.tight_layout()
    fig.savefig(ckdir / "trace_params.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    lp = log_prob[::stride]
    ax.plot(steps, lp, lw=0.3, alpha=0.35)
    ax.plot(steps, np.median(lp, axis=1), "k", lw=1.2, label="median over walkers")
    top = np.nanmax(log_prob)
    ax.set_ylim(top - 60, top + 2)
    ax.axvline(discard, color="k", ls="--", lw=0.8)
    ax.axhline(REFERENCE_LOG_POSTERIOR, color="r", ls=":", lw=1, label="reference optimum")
    ax.set_xlabel("step")
    ax.set_ylabel("log_probability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ckdir / "trace_log_prob.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].bar(np.arange(nwalkers), acc)
    axes[0].set_xlabel("walker")
    axes[0].set_ylabel("acceptance fraction")
    lengths = np.unique(np.linspace(max(1000, n // 10), n - discard, 8).astype(int))
    lengths = lengths[lengths > 100]
    if len(lengths):
        est = np.array([_autocorr(chain[discard:discard + m]) for m in lengths])
        axes[1].plot(lengths, est, alpha=0.6)
        axes[1].plot(lengths, lengths / 50.0, "k--", label="N / 50")
        axes[1].set_xlabel("post-burn steps used")
        axes[1].set_ylabel("tau_ac estimate")
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ckdir / "acceptance_autocorr.png", dpi=110)
    plt.close(fig)

    pairs = [(a, b) for a, b in KEY_PAIRS if a in flat_labels and b in flat_labels][:12]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(flat), min(len(flat), 20000), replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    for ax, (a, b) in zip(axes.flat, pairs):
        ax.plot(flat[pick, flat_labels.index(a)], flat[pick, flat_labels.index(b)], ",", alpha=0.4)
        ax.set_xlabel(a, fontsize=8)
        ax.set_ylabel(b, fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(ckdir / "key_correlations.png", dpi=110)
    plt.close(fig)

    names = ("tau", "xi_e", "xi_e_r", "E_iso", "eps_e_r", "eps_B_r", "q_E_xir", "p_r")
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    for ax, name in zip(axes.flat, names):
        if name == "q_E_xir":
            v = np.log10(phys["E_iso"] * phys["xi_e_r"])
        else:
            v = np.log10(phys[name]) if name in ("tau", "E_iso") else phys[name]
        lo, hi = float(v.min()), float(v.max())
        ax.hist(v, bins=np.linspace(lo, hi if hi > lo else lo + 1e-9, 61), histtype="step", color="k")
        ax.set_xlabel("q_E_xir = log10(E_iso xi_e_r)" if name == "q_E_xir" else
                      f"log10 {name}" if name in ("tau", "E_iso") else name)
    fig.tight_layout()
    fig.savefig(ckdir / "key_marginals.png", dpi=110)
    plt.close(fig)
