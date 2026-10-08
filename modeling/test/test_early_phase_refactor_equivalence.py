#!/usr/bin/env python3
"""Numerical equivalence of the refactored early-phase model against the frozen pre-refactor code.

The frozen copy (test/reference/early_phase_pre_refactor.py, sha256 3f077114...) is the 2043-line
early_phase.py as of 2026-10-08 before the split into early_phase_model / _optimize / _diagnostics / _mcmc.

    python test/test_early_phase_refactor_equivalence.py --record   # write the baseline with the frozen code
    python test/test_early_phase_refactor_equivalence.py            # compare frozen (live), baseline and refactored

For every test point and every parameter layout it compares chi2_XRT_flux, chi2_i_band,
chi2_XRT_spectral_index, chi2_total, log_likelihood, log_prior and log_probability, and checks
that both eps_e + eps_B > 1 and eps_e_r + eps_B_r > 1 give log_probability = -inf.
Run from the modeling/ directory.
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = Path(__file__).resolve().parent
MODELING = HERE.parent
RESULTS = Path("/home/hongyp007/yp_folder/projects/GRB/result")
BASELINE = HERE / "reference" / "early_phase_equivalence_baseline.npz"
COOLING = "smooth"
RTOL = 1e-12
QUANTITIES = ("chi2_XRT_flux", "chi2_i_band", "chi2_XRT_spectral_index", "chi2_total",
              "log_likelihood", "log_prior", "log_probability")

sys.path.insert(0, str(MODELING))
sys.path.insert(0, str(MODELING.parent))

# make_param_defs keyword sets (identical in the frozen and the refactored module)
CONFIGS = {
    "wide_ism": dict(prior="wide", environment="ism"),
    "final_ism_none": dict(prior="final", environment="ism"),
    "final_wind_none": dict(prior="final", environment="wind"),
    "final_wind_none_tau3000": dict(prior="final", environment="wind", tau_max=3000),
    "final_wind_none_tau3000_epsr1": dict(prior="final", environment="wind", tau_max=3000, eps_e_r_max=1.0, eps_B_r_max=1.0),
    "final_wind_reverse_eiso1e54": dict(prior="final", environment="wind", tau_max=3000, eps_e_r_max=1.0, eps_B_r_max=1.0,
                                        free_xi_e_r=True),
    "final_wind_reverse_eiso1e55": dict(prior="final", environment="wind", tau_max=3000, eps_e_r_max=1.0, eps_B_r_max=1.0,
                                        free_xi_e_r=True, e_iso_max=1e55),
    "final_wind_both_eiso1e55": dict(prior="final", environment="wind", tau_max=3000, eps_e_r_max=1.0, eps_B_r_max=1.0,
                                     free_xi_e_r=True, free_xi_e=True, e_iso_max=1e55),
}


def load_frozen():
    """Import the frozen pre-refactor module (and its frozen plotting companion) under private names."""
    for name, filename in (("early_phase_plotting", "early_phase_plotting_pre_refactor.py"),
                           ("early_phase_pre_refactor", "early_phase_pre_refactor.py")):
        spec = importlib.util.spec_from_file_location(name, HERE / "reference" / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    frozen = sys.modules["early_phase_pre_refactor"]
    del sys.modules["early_phase_plotting"]          # the refactored code imports the live plotting module
    return frozen


def label_index(labels, name):
    return labels.index(name)


def build_points(ref):
    """(config, tag, vector) test points: saved optima, profile rows, old MCMC samples, random interior points, violations."""
    rng = np.random.default_rng(20261008)
    points = []
    labels = {c: [ref.param_label(p) for p in ref.make_param_defs(**kw)] for c, kw in CONFIGS.items()}
    bounds = {c: ref._sampled_bounds(ref.make_param_defs(**kw)) for c, kw in CONFIGS.items()}

    def add(config, tag, vec):
        vec = np.array(vec, dtype=float)          # copy: callers edit their vector after adding it
        assert len(vec) == len(labels[config]), (config, tag, len(vec))
        points.append((config, tag, vec))

    # 12-parameter (xi none) layout: known modes and optima
    for name in ("HIGH_MODE_START", "LOW_MODE_START", "LOW_MODE_OPT", "HIGH_MODE_OPT", "HIGH_MODE_OPT_EPS_R_WIDE", "CONSTRAINED_OPT"):
        for config in ("final_wind_none", "final_wind_none_tau3000", "final_wind_none_tau3000_epsr1"):
            add(config, name, getattr(ref, name))
    tau = np.load(RESULTS / "early_phase_tauprofile_20261006_220735" / "tau_profile.npz")
    for k in range(0, len(tau["tau"]), 3):           # high- and low-tau profile solutions, every third grid value
        add("final_wind_none_tau3000", f"tau_profile_independent_tau{tau['tau'][k]:g}", tau["independent_params"][k])
    # old saved MCMC runs (the layouts of the three earlier early_phase runs)
    for config, run in (("wide_ism", "early_phase_20261006_170102"), ("final_ism_none", "early_phase_20261006_172911"),
                        ("final_wind_none", "early_phase_20261006_184042")):
        samples = np.load(RESULTS / run / "samples.npy", mmap_mode="r")
        add(config, f"{run}:top1", np.load(RESULTS / run / "top_k_params.npy")[0])
        for j in rng.choice(len(samples), 5, replace=False):
            add(config, f"{run}:sample{j}", samples[j])

    # 13-parameter (xi reverse) layout
    xi_i = label_index(labels["final_wind_reverse_eiso1e54"], "xi_e_r")
    for config in ("final_wind_reverse_eiso1e54", "final_wind_reverse_eiso1e55"):
        add(config, "XI_FREE_OPT", ref.XI_FREE_OPT)
        add(config, "CONSTRAINED_OPT+xi_e_r=1", np.insert(ref.CONSTRAINED_OPT, xi_i, 1.0))
    xi = np.load(RESULTS / "early_phase_xiprofile_20261007_180936" / "xi_profile.npz")
    for k in range(len(xi["xi_e_r"])):
        add("final_wind_reverse_eiso1e54", f"xi_profile_independent_xi{xi['xi_e_r'][k]:g}", xi["independent_params"][k])
    import csv
    with open(RESULTS / "early_phase_eisosens_20261007_220624" / "eiso_sensitivity.csv", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["rank"] in ("0", "1")]
    for r in rows:
        add("final_wind_reverse_eiso1e55", f"eisosens_xi{r['xi_e_r']}_rank{r['rank']}",
            [float(r[l]) for l in labels["final_wind_reverse_eiso1e55"]])

    # 14-parameter (xi both) layout: the 13-parameter solutions with xi_e inserted at several values
    both = labels["final_wind_both_eiso1e55"]
    xe_i = label_index(both, "xi_e")
    for config, tag, vec in [p for p in points if p[0] == "final_wind_reverse_eiso1e55"]:
        for xi_e in (1.0, 0.5, 0.2):
            add("final_wind_both_eiso1e55", f"{tag}+xi_e={xi_e:g}", np.insert(vec, xe_i, xi_e))

    # random interior points of every layout (ordinary finite-probability points where the model allows)
    for config in CONFIGS:
        lo, hi = bounds[config][:, 0], bounds[config][:, 1]
        for j in range(4):
            add(config, f"uniform{j}", lo + rng.random(len(lo)) * (hi - lo))
    # small perturbations of the best 14-parameter point (typical sampler positions)
    base = [p for p in points if p[0] == "final_wind_both_eiso1e55" and p[1].startswith("eisosens_xi0.05_rank0+xi_e=1")][0][2]
    span = bounds["final_wind_both_eiso1e55"][:, 1] - bounds["final_wind_both_eiso1e55"][:, 0]
    for j in range(6):
        vec = np.clip(base + rng.normal(0, 0.01 * span), *bounds["final_wind_both_eiso1e55"].T)
        add("final_wind_both_eiso1e55", f"eisosens_best_perturbed{j}", vec)

    # physical-constraint violations
    w = labels["wide_ism"]
    v = np.load(RESULTS / "early_phase_20261006_170102" / "top_k_params.npy")[0].copy()
    v[w.index("log10_eps_e")], v[w.index("log10_eps_B")] = np.log10(0.45), np.log10(0.70)
    add("wide_ism", "VIOLATION:eps_e+eps_B=1.15", v)
    v[w.index("log10_eps_e")], v[w.index("log10_eps_B")] = np.log10(0.45), np.log10(0.549)
    add("wide_ism", "allowed:eps_e+eps_B=0.999", v)
    v = np.load(RESULTS / "early_phase_20261006_170102" / "top_k_params.npy")[0].copy()
    v[w.index("log10_eps_e_r")], v[w.index("log10_eps_B_r")] = np.log10(0.7), np.log10(0.5)
    add("wide_ism", "VIOLATION:eps_e_r+eps_B_r=1.2", v)
    for config in ("final_wind_none_tau3000_epsr1", "final_wind_reverse_eiso1e55", "final_wind_both_eiso1e55"):
        src = [p for p in points if p[0] == config][0][2].copy()
        L = labels[config]
        src[L.index("log10_eps_e_r")], src[L.index("log10_eps_B_r")] = np.log10(0.7), np.log10(0.5)
        add(config, "VIOLATION:eps_e_r+eps_B_r=1.2", src)
        src[L.index("log10_eps_e_r")], src[L.index("log10_eps_B_r")] = np.log10(0.6), np.log10(0.399)
        add(config, "allowed:eps_e_r+eps_B_r=0.999", src)
    return points


def evaluate(module, points):
    """Every quantity for every point; NaN marks a quantity whose call raised (recorded in `errors`)."""
    xrt_data, i_data, xrt_index_data = module.load_fit_data(include_spectral_index=True)
    defs = {c: module.make_param_defs(**kw) for c, kw in CONFIGS.items()}
    values = np.full((len(points), len(QUANTITIES)), np.nan)
    errors = []
    for n, (config, tag, vec) in enumerate(points):
        pd_ = defs[config]
        args = (vec, pd_, xrt_data, i_data, xrt_index_data, COOLING)
        out = {}
        try:
            c = module.chi2_components(*args)
            out.update(chi2_XRT_flux=c[0], chi2_i_band=c[1], chi2_XRT_spectral_index=c[2], chi2_total=float(sum(c)))
            out["log_likelihood"] = module.log_likelihood(*args)
        except Exception as e:                       # e.g. model failure far out in the box
            errors.append(f"{config}/{tag}: {type(e).__name__}: {e}")
        out["log_prior"] = module.log_prior(vec, pd_)
        out["log_probability"] = module.log_probability(*args)
        values[n] = [out.get(q, np.nan) for q in QUANTITIES]
    return values, errors


def compare(name_a, a, name_b, b, points):
    """Max relative difference per quantity; inf / -inf / NaN must match exactly."""
    ok = True
    for q, qa, qb in zip(QUANTITIES, a.T, b.T):
        same_special = np.array_equal(np.isnan(qa), np.isnan(qb)) and np.array_equal(np.isinf(qa), np.isinf(qb)) \
            and np.array_equal(qa[np.isinf(qa)], qb[np.isinf(qb)])
        finite = np.isfinite(qa) & np.isfinite(qb)
        rel = np.abs(qa[finite] - qb[finite]) / np.maximum(np.abs(qa[finite]), 1e-300)
        worst = rel.max() if rel.size else 0.0
        status = "OK" if same_special and worst <= RTOL else "FAIL"
        ok &= status == "OK"
        print(f"  {name_a} vs {name_b}: {q:<24} finite {finite.sum():3d}/{len(qa)}  max rel diff {worst:.3e}  "
              f"special values match {same_special}  {status}")
        if status == "FAIL":
            for n in np.where(finite)[0][rel > RTOL][:5]:
                print(f"      {points[n][0]}/{points[n][1]}: {qa[n]!r} vs {qb[n]!r}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", action="store_true", help="Write the baseline from the frozen pre-refactor code.")
    args = parser.parse_args()
    os.chdir(MODELING)
    frozen = load_frozen()
    points = build_points(frozen)
    print(f"{len(points)} test points over {len(CONFIGS)} layouts: "
          + ", ".join(f"{c} ({sum(p[0] == c for p in points)})" for c in CONFIGS))

    if args.record:
        values, errors = evaluate(frozen, points)
        np.savez(BASELINE, values=values, configs=np.array([p[0] for p in points]), tags=np.array([p[1] for p in points]),
                 vectors=np.array([np.pad(p[2], (0, 14 - len(p[2])), constant_values=np.nan) for p in points]),
                 quantities=np.array(QUANTITIES))
        print(f"Baseline written: {BASELINE}; {len(errors)} points raised in chi2_components: {errors[:5]}")
        print(f"finite log_probability: {np.isfinite(values[:, -1]).sum()}/{len(points)}")
        return

    import early_phase_model as new
    baseline = np.load(BASELINE)
    assert list(baseline["tags"]) == [p[1] for p in points], "test point set changed since the baseline was recorded"
    frozen_values, frozen_errors = evaluate(frozen, points)
    new_values, new_errors = evaluate(new, points)
    ok = compare("frozen(live)", frozen_values, "baseline", baseline["values"], points)
    ok &= compare("frozen(live)", frozen_values, "refactored", new_values, points)
    ok &= frozen_errors == new_errors
    print(f"  exceptions identical: {frozen_errors == new_errors} ({len(new_errors)} points raised in chi2_components)")
    bitwise = np.array_equal(np.nan_to_num(frozen_values, nan=1.2345e300), np.nan_to_num(new_values, nan=1.2345e300))
    print(f"  refactored values bit-identical to frozen: {bitwise}")

    # physical constraints: every VIOLATION point must be -inf, every 'allowed' point finite
    lp = new_values[:, QUANTITIES.index("log_probability")]
    for n, (config, tag, _) in enumerate(points):
        if tag.startswith("VIOLATION") or tag.startswith("allowed"):
            want_inf = tag.startswith("VIOLATION")
            good = (lp[n] == -np.inf) if want_inf else np.isfinite(lp[n])
            ok &= bool(good)
            print(f"  constraint {config}/{tag}: log_probability = {lp[n]:.6g}  {'OK' if good else 'FAIL'}")
    print("EQUIVALENCE PASSED" if ok else "EQUIVALENCE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
