#!/usr/bin/env python3
"""Representative, MAP and random samples of the FINAL early-phase posterior, written as manual_model.ipynb overrides.

    python extract_representative_manual_model.py [--run1 DIR] [--run2 DIR] [--discard1 N] [--discard2 N] [--outdir DIR]

Reads the two FINAL production chains (read-only), drops the burn-in used by final_posterior_report.py (30000 steps
each), pools the post-burn samples and selects

    representative  the actual sample minimising sum(((x - median) / std)^2) over the 14 sampled-space parameters
                    (log10 for log-scaled parameters, linear for p, p_r, xi_e, xi_e_r; physical p_r, not u_r)
    map             the sample with the highest physical log posterior (sampling-coordinate Jacobian removed)
    random          one uniformly drawn sample (fixed seed)

No synthetic vector is ever used as a parameter set: every output is one stored chain state, so the E_iso - xi_e_r and
p_r - eps_e_r correlations are those of the posterior. The marginal-median vector is evaluated only as a diagnostic.

Each sample's log posterior is recomputed with the run's model resolution / rtol and compared with the chain value;
the representative sample is then rebuilt through grb.modeling.make_core_model (the manual_model.ipynb path) and its
fluxes compared with early_phase_model.evaluate_components at a few times.

Writes to OUTDIR: representative_sample.json (all three samples + diagnostics), representative_overrides.py,
map_overrides.py, random_overrides.py.
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import numpy as np

import early_phase_model as m
from final_posterior_report import load

RESULT = Path("/home/hongyp007/yp_folder/projects/GRB/result")
RUN1 = RESULT / "early_phase_FINAL_run1_20261008_220945"
RUN2 = RESULT / "early_phase_FINAL_run2_20261008_221517"
DISCARD = 30000                    # burn-in of both runs in early_phase_FINAL_posterior/FINAL_REPORT.txt
RANDOM_SEED = 20261009
# early_phase_model name -> grb.params / manual_model.ipynb name (all others are identical)
MANUAL_NAME = {"E_iso": "E_iso_core", "Gamma0": "Gamma0_core", "theta_c": "theta_c_core"}
OVERRIDE_BLOCKS = (
    ("Core jet", ("E_iso", "Gamma0", "theta_c")),
    ("Wind environment", ("A_star",)),
    ("Forward shock", ("p", "eps_e", "eps_B", "xi_e")),
    ("Early-afterglow jet / ejecta duration (TophatJet duration, NOT a flare time)", ("tau",)),
    ("Reverse shock", ("p_r", "eps_e_r", "eps_B_r", "xi_e_r")),
    ("Host extinction", ("A_V",)),
)
PAIRS = (("log10_E_iso", "xi_e_r"), ("p_r", "log10_eps_e_r"))


def physical(x, labels):
    return {(l[6:] if l.startswith("log10_") else l): float(10.0 ** v if l.startswith("log10_") else v)
            for l, v in zip(labels, x)}


def manual_overrides(phys):
    return {MANUAL_NAME.get(k, k): v for _, names in OVERRIDE_BLOCKS for k in names for v in [phys[k]]}


def evaluate(x, defs, data):
    lpost = m.log_probability(x, defs, *data, "smooth")
    lprior = m.log_prior(x, defs)
    chi2 = m.chi2_components(x, defs, *data, "smooth")
    return {"log_posterior": float(lpost), "log_prior": float(lprior), "log_likelihood": float(lpost - lprior),
            "chi2_xrt": float(chi2[0]), "chi2_i": float(chi2[1]), "chi2_beta": float(chi2[2]),
            "chi2_total": float(sum(chi2))}


def pair_fit(a, b):
    """Least-squares line b ~ slope * a + intercept over the posterior, with the residual standard deviation."""
    slope, icpt = np.polyfit(a, b, 1)
    return float(slope), float(icpt), float(np.std(np.subtract(b, np.add(np.multiply(slope, a), icpt))))


def ridge_position(x, labels, fits, q_comb):
    """Where x sits relative to each correlated pair: residual from the posterior's regression line in units of the
    residual scatter, and the posterior percentile of log10(E_iso xi_e_r)."""
    out = {}
    for (a, b), (slope, icpt, sd) in zip(PAIRS, fits):
        r = x[labels.index(b)] - (slope * x[labels.index(a)] + icpt)
        out[f"{a}|{b}"] = {"regression_residual_sigma": float(r / sd)}
    q = x[labels.index("log10_E_iso")] + np.log10(x[labels.index("xi_e_r")])
    out["log10_E_iso_xi_e_r"] = {"value": float(q), "posterior_percentile": float(100 * np.mean(q_comb < q))}
    return out


def manual_model_check(overrides, defs, x, resolution):
    """Rebuild the sample through grb.modeling.make_core_model exactly as manual_model.ipynb does (grb.params wind
    defs + xi_e / xi_e_r, flare and wing off) and compare its FS+RS fluxes with early_phase_model at a few times."""
    import grb.modeling as gm
    from grb.const import XRT_BAND
    from grb.extinction import host_extinction_attenuation
    from grb.params import make_param_defs as make_manual_defs
    from grb.utils import model_array

    names = {p.name for p in make_manual_defs(include_flare=False, include_wing=False, environment="wind")}
    missing = names - set(overrides)
    if missing:
        raise RuntimeError(f"overrides do not cover the manual-model parameters {sorted(missing)}")
    params = dict(overrides)
    t_x = np.array([150.0, 300.0, 600.0, 1200.0, 2500.0])
    t_i = np.array([60.0, 200.0, 500.0, 1500.0, 5000.0])
    nu_i = float(m.read_data("i_data", correct_galactic_extinction=True, add_converted_flux=True)["frequency_Hz"].iloc[0])
    ref = m.evaluate_components(x, defs, t_x, t_i, nu_i)
    out = {"t_xrt_s": t_x.tolist(), "t_i_s": t_i.tolist(), "early_phase_model": {
        "xrt_total": ref[0].tolist(), "i_total_mJy": (ref[3] * 1e26).tolist()}}
    saved = gm.MODEL_RESOLUTIONS
    try:
        for tag, res in (("mcmc_resolution", tuple(resolution)), ("grb_default_resolution", saved)):
            gm.MODEL_RESOLUTIONS = res
            model = gm.make_core_model(params)
            xrt = model_array(model.flux(t_x, XRT_BAND[0], XRT_BAND[1], 10).total)
            i_tot = model_array(model.flux_density(t_i, np.full_like(t_i, nu_i)).total) * host_extinction_attenuation(
                np.full_like(t_i, nu_i), params["A_V"], m.REDSHIFT)
            ok = bool(np.all(np.isfinite(xrt)) and np.all(np.isfinite(i_tot)) and np.all(xrt > 0) and np.all(i_tot > 0))
            out[tag] = {"resolution": list(res), "finite_positive": ok, "xrt_total": xrt.tolist(),
                        "i_total_mJy": (i_tot * 1e26).tolist(),
                        "max_rel_diff_xrt": float(np.max(np.abs(xrt / ref[0] - 1))),
                        "max_rel_diff_i": float(np.max(np.abs(i_tot / ref[3] - 1)))}
    finally:
        gm.MODEL_RESOLUTIONS = saved
    return out


def write_overrides(path, name, phys, info):
    lines = [f'"""{name} sample of the FINAL early-phase MCMC posterior, in manual_model.ipynb parameter names.',
             "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M} by code/fitting/modeling/extract_representative_manual_model.py from",
             f"  {RUN1}",
             f"  {RUN2}",
             f"(post-burn, {info['n_samples']:,} pooled samples). Selected: {info['run']} step {info['step']} walker {info['walker']};",
             f"log posterior {info['log_posterior']:.4f}, distance from the posterior median {info['distance']:.4f}.",
             "",
             "Early afterglow (forward + reverse shock) only. One actual chain state - keep all values together;",
             "do not mix them with marginal medians or with values from another sample.",
             "xi_e and xi_e_r were free in the fit and must be passed (the model would otherwise use xi = 1).",
             "tau is the TophatJet ejecta duration (s), unrelated to the flare tau_rise_flare / tau_decay_flare.",
             '"""', "", "early_afterglow = {"]
    for title, keys in OVERRIDE_BLOCKS:
        lines.append(f"    # {title}")
        lines += [f'    "{MANUAL_NAME.get(k, k)}": {phys[k]!r},' for k in keys]
        lines.append("")
    lines[-1] = "}"
    lines += ["", "overrides = dict(early_afterglow)", ""]
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run1", default=str(RUN1))
    ap.add_argument("--run2", default=str(RUN2))
    ap.add_argument("--discard1", type=int, default=DISCARD)
    ap.add_argument("--discard2", type=int, default=DISCARD)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--outdir", default=str(RESULT / "early_phase_FINAL_posterior" / "manual_model_overrides"))
    args = ap.parse_args()

    runs = [load(args.run1, args.discard1), load(args.run2, args.discard2)]
    c1, c2 = runs[0]["config"], runs[1]["config"]
    for key in ("labels", "param_kwargs", "model_resolution", "model_rtol", "sampling_coords"):
        if c1.get(key) != c2.get(key):
            raise SystemExit(f"runs differ in {key}")
    labels = runs[0]["labels"]
    m.set_numerics(c1["model_resolution"], c1["model_rtol"])
    defs = m.make_param_defs(**c1["param_kwargs"])
    data = m.load_fit_data(include_spectral_index=True)

    # pooled post-burn samples; flat index k of run r <-> step discard + k // nwalkers, walker k % nwalkers
    comb = np.concatenate([r["post"].reshape(-1, len(labels)) for r in runs])
    comb_lp = np.concatenate([r["post_lp"].ravel() for r in runs])
    sizes = [r["post"].shape[0] * r["post"].shape[1] for r in runs]
    del runs[0]["raw"], runs[0]["phys"], runs[1]["raw"], runs[1]["phys"]

    def origin(k):
        ri = 0 if k < sizes[0] else 1
        kk = k - (0 if ri == 0 else sizes[0])
        nw = runs[ri]["post"].shape[1]
        return {"run": f"run{ri + 1}", "run_dir": str(runs[ri]["dir"]), "flat_index": int(k),
                "step": int(runs[ri]["discard"] + kk // nw), "walker": int(kk % nw)}

    med = np.median(comb, axis=0)
    std = comb.std(axis=0)
    z = np.divide(np.subtract(comb, med), std)          # explicit ufuncs: no NumPy temporary elision on big arrays
    dist = np.einsum("ij,ij->i", z, z)
    del z
    picks = {"representative": int(np.argmin(dist)), "map": int(np.argmax(comb_lp)),
             "random": int(np.random.default_rng(args.seed).integers(len(comb)))}

    fits = [pair_fit(comb[:, labels.index(a)], comb[:, labels.index(b)]) for a, b in PAIRS]
    q_comb = np.add(comb[:, labels.index("log10_E_iso")], np.log10(comb[:, labels.index("xi_e_r")]))
    corr = {f"{a}|{b}": float(np.corrcoef(comb[:, labels.index(a)], comb[:, labels.index(b)])[0, 1]) for a, b in PAIRS}

    out = {"created": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "runs": [args.run1, args.run2],
           "discard": [args.discard1, args.discard2], "n_samples": int(len(comb)), "labels": labels,
           "model_resolution": c1["model_resolution"], "model_rtol": c1["model_rtol"],
           "method": "representative = argmin over actual post-burn samples of sum(((x - median) / std)^2) in the "
                     "sampled physical space (log10 for log-scaled parameters, p_r linear); map = argmax physical log "
                     "posterior; random = uniform draw with the given seed",
           "random_seed": args.seed, "posterior_median_sampled_space": dict(zip(labels, med.tolist())),
           "posterior_std_sampled_space": dict(zip(labels, std.tolist())),
           "distance_quantiles_5_50_95": np.percentile(dist, [5, 50, 95]).tolist(),
           "posterior_correlations_sampled_space": corr,
           "pair_regressions_slope_intercept_residual_sd": {f"{a}|{b}": list(f) for (a, b), f in zip(PAIRS, fits)}}

    for name, k in picks.items():
        x = comb[k].copy()
        phys = physical(x, labels)
        info = {**origin(k), "distance": float(dist[k]), "log_posterior_chain": float(comb_lp[k]),
                **evaluate(x, defs, data), "sampled_space": dict(zip(labels, x.tolist())), "physical": phys,
                "ridge_position": ridge_position(x, labels, fits, q_comb),
                "manual_model_overrides": manual_overrides(phys)}
        info["log_posterior_recomputed_minus_chain"] = info["log_posterior"] - info["log_posterior_chain"]
        out[name] = info

    # diagnostic only - the synthetic marginal-median vector is never written as an override set
    out["marginal_median_vector_diagnostic"] = {**evaluate(med, defs, data),
                                                "ridge_position": ridge_position(med, labels, fits, q_comb)}
    rep = out["representative"]
    rep["manual_model_check"] = manual_model_check(rep["manual_model_overrides"], defs, comb[picks["representative"]],
                                                   c1["model_resolution"])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "representative_sample.json").write_text(json.dumps(out, indent=1) + "\n")
    for name in picks:
        write_overrides(outdir / f"{name}_overrides.py", name.capitalize(), out[name]["physical"],
                        {**out[name], "n_samples": len(comb)})

    print(f"{len(comb):,} pooled post-burn samples; distance 5/50/95%: "
          + " / ".join(f"{v:.2f}" for v in out["distance_quantiles_5_50_95"]))
    print(f"posterior correlations: " + ", ".join(f"{k} {v:+.3f}" for k, v in corr.items()))
    head = f"{'parameter':<14}" + "".join(f"{n:>16}" for n in ("median", *picks))
    print(head)
    for l in labels:
        name = l[6:] if l.startswith("log10_") else l
        vals = [10 ** med[labels.index(l)] if l.startswith("log10_") else med[labels.index(l)]]
        vals += [out[n]["physical"][name] for n in picks]
        print(f"{MANUAL_NAME.get(name, name):<14}" + "".join(f"{v:>16.6g}" for v in vals))
    for n in (*picks, "marginal_median_vector_diagnostic"):
        o = out[n]
        rp = o["ridge_position"]
        extra = (f"{o['run']} step {o['step']} walker {o['walker']}, distance {o['distance']:.3f}, chain log P "
                 f"{o['log_posterior_chain']:.4f}, " if "run" in o else "")
        print(f"{n}: {extra}log P {o['log_posterior']:.4f} (log L {o['log_likelihood']:.4f}, chi2 {o['chi2_total']:.2f}); "
              f"ridge residuals " + ", ".join(f"{k} {v['regression_residual_sigma']:+.2f} sig" for k, v in rp.items()
                                              if "regression_residual_sigma" in v)
              + f"; log10(E_iso xi_e_r) percentile {rp['log10_E_iso_xi_e_r']['posterior_percentile']:.1f}")
    chk = rep["manual_model_check"]
    for tag in ("mcmc_resolution", "grb_default_resolution"):
        print(f"manual-model path ({tag} {chk[tag]['resolution']}): finite/positive {chk[tag]['finite_positive']}, "
              f"max |rel diff| vs early_phase_model XRT {chk[tag]['max_rel_diff_xrt']:.2e}, i {chk[tag]['max_rel_diff_i']:.2e}")
    print(f"written to {outdir}")


if __name__ == "__main__":
    main()
