#!/usr/bin/env python3
"""Equivalence harness for the grb-module refactor.

Asserts that the `grb` package reproduces the pre-refactor standalone scripts
(modeling/final_model.py, final_model_plotting.py, utils.py) exactly. The
baseline sources are materialised from git at commit BASELINE rather than kept
as frozen duplicates in the tree.

Equivalence here means bit-identical arrays and scalars -- every comparison uses
np.testing.assert_array_equal or ==, never assert_allclose. MCMC chains are NOT
compared: the driver seeds walkers with unseeded np.random, so two runs of the
*same* code already differ.

Run directly (there is no pytest harness in this project):

    /home/dtak/miniconda3/envs/grb251013c/bin/python modeling/test/test_refactor_equivalence.py

Add --png to also render both sides' figures and compare bytes. That is slow
(~300 model builds per side, several minutes) so it is opt-in.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASELINE = "a40dd204e0b2bc7e5286387548fe14f75328e9f8"

# Reference log-probability at INITIAL_GUESS (clipped to bounds), measured on the
# baseline before any code moved. A change here means the fit itself changed.
REF_LOG_PROB = -40459.052164414396
REF_LOG_PRIOR = -1.787277712341638
REF_SI_CHI2 = 37.10252733400446

INITIAL_GUESS = {
    "E_iso_core": 1.189e52, "Gamma0_core": 522, "theta_c_core": 0.02, "n_ism": 18.76,
    "p": 2.158, "eps_e": 0.0435, "eps_B": 0.0163, "xi": 0.943, "tau": 15.0,
    "p_r": 3.329, "eps_e_r": 0.0422, "eps_B_r": 0.20, "xi_r": 0.849, "A_V": 0.0254,
    "t_start_flare": 3000, "tau_rise_flare": 300, "tau_decay_flare": 2000,
    "A_flare": 3e-10, "flare_beta": 0.8, "E_iso_wing": 3e51, "Gamma0_wing": 30,
    "theta_c_wing": 0.3, "p_wing": 2.3, "eps_e_wing": 0.9, "eps_B_wing": 0.005,
    "xi_wing": 0.8,
}

# The bounds retune (commit 76aeca4) intentionally diverges from the baseline
# parameter space. These are the ONLY allowed (lower, upper) differences;
# check_params asserts the current values match this table exactly and that
# every other parameter still equals the baseline bit-for-bit.
RETUNED_BOUNDS = {
    "theta_c_core": (0.001, 0.08),      # was (0.001, 0.04)
    "n_ism": (5, 400),                  # was (5, 150)
    "eps_B": (0.002, 0.05),             # was (0.005, 0.05)
    "tau_rise_flare": (10, 2000),       # was (30, 2000)
    "E_iso_wing": (1e51, 1e53),         # was (1e52, 1e53)
    "theta_c_wing": (0.2, 0.7),         # was (0.2, 0.5)
    "p_wing": (2.2, 3.3),               # was (2.2, 2.9)
    "eps_e_wing": (0.1, 1.0),           # was (0.3, 1.0)
}

# The driver's INITIAL_GUESS after the retune: the re-optimized parameter
# vector (log_probability = -550.28 under the current data). Unlike the
# baseline guess above, every value must lie INSIDE its own bounds.
RETUNED_INITIAL_GUESS = {
    "E_iso_core": 1.124e52, "Gamma0_core": 551, "theta_c_core": 0.0391, "n_ism": 146.9,
    "p": 2.164, "eps_e": 0.0416, "eps_B": 0.00563, "xi": 0.897, "tau": 12.8,
    "p_r": 2.30, "eps_e_r": 0.0511, "eps_B_r": 0.162, "xi_r": 0.852, "A_V": 0.238,
    "t_start_flare": 2553, "tau_rise_flare": 25.5, "tau_decay_flare": 2391,
    "A_flare": 9.62e-10, "flare_beta": 0.638, "E_iso_wing": 1.011e52, "Gamma0_wing": 19.2,
    "theta_c_wing": 0.492, "p_wing": 3.06, "eps_e_wing": 0.303, "eps_B_wing": 0.0121,
    "xi_wing": 0.98,
}


def stage_baseline():
    """Materialise the pre-refactor modeling/ scripts from git and import them.

    The temp tree mirrors the real layout (<base>/modeling/*.py, <base>/data)
    because baseline utils.load_xrt_spectral_index resolves data/ two levels up
    from its own __file__.
    """
    base = tempfile.mkdtemp(prefix="grb_baseline_")
    modeling = os.path.join(base, "modeling")
    os.mkdir(modeling)
    os.symlink(os.path.join(ROOT, "data"), os.path.join(base, "data"))
    for name in ("utils.py", "final_model.py", "final_model_plotting.py"):
        blob = subprocess.check_output(["git", "-C", ROOT, "show", f"{BASELINE}:modeling/{name}"])
        open(os.path.join(modeling, name), "wb").write(blob)
    sys.path.insert(0, modeling)
    import utils as baseline_utils
    import final_model as baseline_model
    import final_model_plotting as baseline_plotting
    return baseline_utils, baseline_model, baseline_plotting


BU, BFM, BFMP = stage_baseline()


def bounds(param_def):
    from VegasAfterglow import Scale
    lo = np.log10(param_def.lower) if param_def.scale is Scale.LOG else param_def.lower
    hi = np.log10(param_def.upper) if param_def.scale is Scale.LOG else param_def.upper
    return lo, hi


def reference_theta(param_defs):
    """INITIAL_GUESS in sampled space, clipped to bounds (as the driver does)."""
    from VegasAfterglow import Scale
    theta = []
    for p in param_defs:
        v = np.log10(INITIAL_GUESS[p.name]) if p.scale is Scale.LOG else INITIAL_GUESS[p.name]
        theta.append(np.clip(v, *bounds(p)))
    return np.array(theta)


# --------------------------------------------------------------------------
# constants and small helpers (grb.const, grb.utils)
# --------------------------------------------------------------------------
def check_constants():
    from VegasAfterglow.units import keV
    from grb import const
    from grb.utils import model_array, seconds_from_trigger

    v = seconds_from_trigger("2025-10-14T00:04:00")
    expected = (datetime.strptime("2025-10-14T00:04:00", "%Y-%m-%dT%H:%M:%S")
                - const.TRIGGER_TIME).total_seconds()
    assert v == expected, (v, expected)
    assert isinstance(seconds_from_trigger("2025-10-14T00:04:00.500"), float)
    try:
        seconds_from_trigger("not-a-date")
        raise AssertionError("bad date should raise ValueError")
    except ValueError:
        pass

    assert const.XRT_NU_LO == 7.25e16
    assert const.XRT_NU_HI == 2.42e18
    assert const.SI_FLARE_FRAC_MAX == 0.5
    assert const.HOST_AV_LOG10_MEAN == -0.82
    assert const.HOST_AV_LOG10_SIGMA == 0.41
    assert const.C_CM_PER_S == 2.99792458e10
    assert const.LN10_OVER_2P5 == 0.4 * np.log(10.0)
    assert const.XRT_EXCLUDE_TIME_RANGE == (3e3, 1e4)
    assert const.XRT_FLARE_START_TIME == 3e3
    assert const.XRT_FLARE_END_TIME == 1e4
    assert const.MODEL_RESOLUTIONS == (0.1, 0.25, 10) == BFM.MODEL_RESOLUTIONS
    assert const.XRT_BAND == (0.3 * keV, 10.0 * keV) == BFM.XRT_BAND
    # BFM.FIT_RESULTS_DIR is __file__-relative, so under staging it points into the
    # temp tree; assert grb's copy resolves to the same place relative to the repo.
    assert const.FIT_RESULTS_DIR == Path(ROOT) / "modeling" / "fit_results"
    assert BFM.FIT_RESULTS_DIR.parent.name == "modeling"
    assert BFM.FIT_RESULTS_DIR.name == const.FIT_RESULTS_DIR.name == "fit_results"
    assert const.XRT_NU_LO == BFM.XRT_NU_LO and const.XRT_NU_HI == BFM.XRT_NU_HI

    class _SS:
        sync = np.array([1.0, 2.0])
        ssc = np.array([0.5, 0.5])

    class _T:
        total = _SS()

    for arg, want in ((_T(), np.array([1.5, 2.5])), (np.array([[3.0, 4.0]]), np.array([3.0, 4.0]))):
        np.testing.assert_array_equal(model_array(arg), want)
        np.testing.assert_array_equal(model_array(arg), BU.model_array(arg))
    print("  constants match final_model.py; seconds_from_trigger and model_array OK")


# --------------------------------------------------------------------------
# grb.extinction.host_extinction_attenuation
# --------------------------------------------------------------------------
def check_extinction():
    from VegasAfterglow.extinction import BUILTIN_LAWS
    from grb.const import C_CM_PER_S, LN10_OVER_2P5
    from grb.extinction import host_extinction_attenuation

    nu = np.array([3.93e14, 5.0e14])
    np.testing.assert_array_equal(host_extinction_attenuation(nu, 0, 1.0), np.ones(2))

    a_v, z = 0.0254, 1.0
    lam = C_CM_PER_S / nu / (1.0 + z)
    expected = np.exp(-a_v * LN10_OVER_2P5 * BUILTIN_LAWS["smc"](lam))
    np.testing.assert_array_equal(host_extinction_attenuation(nu, a_v, z), expected)
    np.testing.assert_array_equal(host_extinction_attenuation(nu, a_v, z),
                                  BU.host_extinction_attenuation(nu, a_v, z))
    print("  host_extinction_attenuation identical (SMC law, rest-frame)")


# --------------------------------------------------------------------------
# grb.spectral_index
# --------------------------------------------------------------------------
def check_spectral_index():
    from grb import spectral_index as si

    old = BU.load_xrt_spectral_index()
    new = si.load_xrt_spectral_index()
    assert set(old) == set(new), (set(old), set(new))
    for k in old:
        np.testing.assert_array_equal(new[k], old[k], err_msg=f"load_xrt_spectral_index[{k}]")

    p = {"E_iso": 1.189e52, "n_ism": 18.76, "eps_e": 0.0435, "eps_B": 0.0163, "p": 2.158}
    t = np.geomspace(100, 5e5, 200)
    o, n = BU.compute_break_frequencies(p, 1.0, t), si.compute_break_frequencies(p, 1.0, t)
    for k in ("nu_m", "nu_c"):
        np.testing.assert_array_equal(n[k], o[k], err_msg=f"compute_break_frequencies[{k}]")

    for regime in ("slow", "fast", "both"):
        assert (si.compute_p_prior_from_spectral_index(new, regime)
                == BU.compute_p_prior_from_spectral_index(old, regime)), regime
    print(f"  spectral index identical ({len(new['time'])} points), breaks and p prior identical")


# --------------------------------------------------------------------------
# grb.params
# --------------------------------------------------------------------------
def check_params():
    from grb.params import default_nwalkers, make_param_defs

    retuned_seen = set()
    for flare in (True, False):
        for wing in (True, False):
            old, new = BFM.make_param_defs(flare, wing), make_param_defs(flare, wing)
            assert len(old) == len(new), (flare, wing, len(old), len(new))
            for o, n in zip(old, new):
                assert o.name == n.name, (o.name, n.name)
                if n.name in RETUNED_BOUNDS:
                    lo, hi = RETUNED_BOUNDS[n.name]
                    assert (n.lower, n.upper) == (lo, hi), (n.name, n.lower, n.upper)
                    # the overlay must be a real divergence, not stale bookkeeping
                    assert (o.lower, o.upper) != (lo, hi), (n.name, "baseline already matches")
                    retuned_seen.add(n.name)
                else:
                    assert o.lower == n.lower, (o.name, o.lower, n.lower)
                    assert o.upper == n.upper, (o.name, o.upper, n.upper)
                assert o.scale == n.scale, (o.name,)
                assert o.has_gaussian_prior() == n.has_gaussian_prior(), (o.name,)
                assert o.get_prior_mean_sigma() == n.get_prior_mean_sigma(), (o.name,)
    assert retuned_seen == set(RETUNED_BOUNDS), retuned_seen

    assert default_nwalkers(26) == max(4 * 26, 32) == 104
    assert default_nwalkers(2) == 32
    print(f"  make_param_defs match baseline except the {len(RETUNED_BOUNDS)} documented "
          f"retuned bounds, for all 4 flare/wing combinations")


# --------------------------------------------------------------------------
# grb.modeling
# --------------------------------------------------------------------------
def check_modeling():
    from grb.modeling import load_all_optical_data, make_core_model, make_wing_model
    from grb.utils import model_array

    o_xrt, o_opt = BFM.load_all_optical_data()
    n_xrt, n_opt = load_all_optical_data()

    for k in ("time", "flux", "flux_error"):
        np.testing.assert_array_equal(n_xrt[k], o_xrt[k], err_msg=f"xrt[{k}]")

    assert len(n_opt) == len(o_opt) == 25, (len(n_opt), len(o_opt))
    for a, b in zip(o_opt, n_opt):
        assert a["name"] == b["name"], (a["name"], b["name"])
        assert a["frequency"] == b["frequency"], a["name"]
        for k in ("time", "flux_mJy", "flux_err"):
            np.testing.assert_array_equal(b[k], a[k], err_msg=f"{a['name']}[{k}]")

    params = {k: v for k, v in INITIAL_GUESS.items()}
    params["p_r"] = 3.0          # keep inside bounds; model build is bounds-agnostic
    params["E_iso_wing"] = 1e52
    t = np.geomspace(100, 1e5, 20)
    for label, mk_new, mk_old in (("core", make_core_model, BFM.make_core_model),
                                  ("wing", make_wing_model, BFM.make_wing_model)):
        a = model_array(mk_old(params).flux_density(t, 3.93e14 * np.ones_like(t)).total)
        b = model_array(mk_new(params).flux_density(t, 3.93e14 * np.ones_like(t)).total)
        np.testing.assert_array_equal(b, a, err_msg=f"{label} flux")

    print(f"  data identical: {len(n_xrt['time'])} XRT + {len(n_opt)} optical datasets "
          f"({sum(len(d['time']) for d in n_opt)} points); core and wing model flux identical")
    return n_xrt, n_opt


# --------------------------------------------------------------------------
# grb.likelihood  -- the decisive check
# --------------------------------------------------------------------------
def check_likelihood(xrt, opt, n_random=50):
    from grb import likelihood as L
    from grb.params import make_param_defs
    from grb.spectral_index import load_xrt_spectral_index

    pds = make_param_defs(True, True)          # grb implementation
    bpds = BFM.make_param_defs(True, True)     # baseline implementation
    # Each side must use its OWN ParamDefWithPrior class: baseline log_prior does an
    # isinstance() check against the class it imported, so feeding it grb-built defs
    # would silently skip the Gaussian prior. check_params proved the sets identical.
    idx = load_xrt_spectral_index()

    # Draw every theta inside the BASELINE box: the retuned bounds strictly
    # contain it, so these points are valid under both parameter spaces and the
    # frozen REF_* values keep their meaning. Thetas in the widened region
    # would get -inf from the baseline log_prior by construction.
    thetas = [reference_theta(bpds)]
    rng = np.random.default_rng(20260727)
    for _ in range(n_random):
        thetas.append(np.array([rng.uniform(*bounds(p)) for p in bpds]))

    n_finite = 0
    for i, th in enumerate(thetas):
        o_pri, n_pri = BFM.log_prior(th, bpds), L.log_prior(th, pds)
        assert o_pri == n_pri, (i, o_pri, n_pri)
        o_lp = BFM.log_probability(th, bpds, xrt, opt, True, True, idx)
        n_lp = L.log_probability(th, pds, xrt, opt, True, True, idx)
        assert o_lp == n_lp, (i, o_lp, n_lp)
        if np.isfinite(n_lp):
            n_finite += 1

    # Regression guard against the values measured before any code moved
    ref = thetas[0]
    assert L.log_prior(ref, pds) == REF_LOG_PRIOR, L.log_prior(ref, pds)
    lp = L.log_probability(ref, pds, xrt, opt, True, True, idx)
    assert lp == REF_LOG_PROB, lp

    from VegasAfterglow import Scale
    params = {p.name: (10 ** v if p.scale is Scale.LOG else v) for p, v in zip(pds, ref)}
    o_x, o_o, o_s = BFM.compute_model_flux_all_bands(params, xrt, opt, True, True, idx)
    n_x, n_o, n_s = L.compute_model_flux_all_bands(params, xrt, opt, True, True, idx)
    np.testing.assert_array_equal(n_x, o_x, err_msg="xrt model flux")
    assert n_s == o_s == REF_SI_CHI2, (n_s, o_s)
    assert len(n_o) == len(o_o) == 25
    for j, (a, b) in enumerate(zip(o_o, n_o)):
        np.testing.assert_array_equal(b, a, err_msg=f"optical[{j}] {opt[j]['name']}")

    print(f"  log_probability identical over {len(thetas)} theta ({n_finite} finite); "
          f"reference log_prob={lp!r}")


# --------------------------------------------------------------------------
# grb.results
# --------------------------------------------------------------------------
def check_results():
    from grb import results as R
    from grb.const import FIT_RESULTS_DIR

    rng = np.random.default_rng(7)
    s, lp = rng.normal(size=(500, 4)), rng.normal(size=500)
    s[10] = s[3]
    lp[10] = lp[3]                              # force a duplicate to exercise dedup
    o_s, o_l = BU.top_k_samples(s, lp, 10)
    n_s, n_l = R.top_k_samples(s, lp, 10)
    np.testing.assert_array_equal(n_s, o_s)
    np.testing.assert_array_equal(n_l, o_l)

    tmp = Path(tempfile.mkdtemp())
    f = tmp / "labels.txt"
    f.write_text("a\n\nb\n c \n")
    assert R.read_labels(f) == BU.read_labels(f) == ["a", "b", "c"]

    if not FIT_RESULTS_DIR.exists():
        print("  top_k_samples/read_labels identical; SKIP result-dir checks (no fit_results/)")
        return None
    o_d = BU.latest_result_dir(FIT_RESULTS_DIR, "final_")
    n_d = R.latest_result_dir(FIT_RESULTS_DIR, "final_")
    assert o_d == n_d, (o_d, n_d)
    o_p, o_f, o_w = BFMP.load_best_fit_params(n_d)
    n_p, n_f, n_w = R.load_best_fit_params(n_d)
    assert (o_f, o_w) == (n_f, n_w)
    assert set(o_p) == set(n_p)
    for k in o_p:
        assert o_p[k] == n_p[k], (k, o_p[k], n_p[k])
    print(f"  top_k_samples/read_labels/latest_result_dir/load_best_fit_params identical "
          f"({n_d.name}, {len(n_p)} params)")
    return n_d


# --------------------------------------------------------------------------
# grb.plotting
# --------------------------------------------------------------------------
def check_plotting(result_dir, compare_png=False):
    from grb import plotting as P
    from grb.const import XRT_BAND
    from grb.results import load_best_fit_params

    vals = [np.array([1e-3, -5.0, np.nan, 2.0]), np.array([np.inf, 7.0])]
    lims = []
    for mod in (BFMP, P):
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        mod.set_log_y_limits(ax, *vals)
        lims.append(ax.get_ylim())
        plt.close(fig)
    assert lims[0] == lims[1], lims

    if result_dir is None:
        print("  set_log_y_limits identical; SKIP component checks (no fit_results/)")
        return

    params, include_flare, include_wing = load_best_fit_params(result_dir)
    t = np.geomspace(1e3, 3e5, 12)
    for label, freq, band in (("XRT", None, XRT_BAND), ("optical", 3.9e14, None)):
        o = BFMP.compute_model_components(params, t, freq, band, include_flare, include_wing)
        n = P.compute_model_components(params, t, freq, band, include_flare, include_wing)
        assert set(o) == set(n)
        for k in o:
            np.testing.assert_array_equal(np.asarray(n[k]), np.asarray(o[k]),
                                          err_msg=f"{label} components[{k}]")
    print("  set_log_y_limits and compute_model_components identical (XRT and optical branches)")

    if not compare_png:
        print("  (skipping figure byte comparison; pass --png to run it)")
        return

    def stage(tag):
        d = Path(tempfile.mkdtemp(prefix=f"lc_{tag}_"))
        for name in ("samples.npy", "log_probs.npy", "labels.txt"):
            os.symlink(result_dir / name, d / name)
        return d

    d_old, d_new = stage("old"), stage("new")
    BFMP.plot_light_curves(d_old)
    P.plot_light_curves(d_new)
    BFMP.plot_spectral_index_comparison(d_old)
    P.plot_spectral_index_comparison(d_new)
    for name in ("bestfit_lc.png", "spectral_index_comparison.png"):
        a, b = (d_old / name).read_bytes(), (d_new / name).read_bytes()
        assert a == b, f"{name} differs ({len(a)} vs {len(b)} bytes)"
        print(f"  {name} byte-identical")


# --------------------------------------------------------------------------
# the driver's own wiring
# --------------------------------------------------------------------------
def check_driver():
    from VegasAfterglow import Scale
    import fit_final_model as driver

    assert driver.INITIAL_GUESS == RETUNED_INITIAL_GUESS

    # The regression that motivated the retune: the old guess had p_r and
    # E_iso_wing OUTSIDE their own bounds, silently clipping every walker onto
    # the boundary. The guess must lie strictly inside the box.
    for p in driver.make_param_defs(True, True):
        v = driver.INITIAL_GUESS[p.name]
        assert p.lower <= v <= p.upper, (p.name, v, p.lower, p.upper)

    args = driver.parse_args([])
    assert (args.include_flare, args.include_wing, args.use_spectral_index) == (True, True, True)
    assert (args.nsteps, args.nwalkers, args.ncpus, args.outdir) == (3000, None, 64, None)
    assert driver.parse_args(["--no-include-wing"]).include_wing is False

    param_defs = driver.make_param_defs(include_flare=True, include_wing=True)
    labels = [f"log10_{p.name}" if p.scale is Scale.LOG else p.name for p in param_defs]
    baseline_labels = [f"log10_{p.name}" if p.scale is Scale.LOG else p.name
                       for p in BFM.make_param_defs(True, True)]
    assert labels == baseline_labels, (labels, baseline_labels)
    assert len(labels) == 26

    pos0 = driver.initial_positions(param_defs, 54)
    assert pos0.shape == (54, 26), pos0.shape
    for i, p in enumerate(param_defs):
        lo, hi = bounds(p)
        assert pos0[:, i].min() >= lo and pos0[:, i].max() <= hi, p.name
    print(f"  driver CLI defaults, {len(labels)} labels and walker seeding match baseline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", action="store_true",
                        help="also render and byte-compare figures (slow)")
    parser.add_argument("--n-random", type=int, default=50,
                        help="random theta drawn for the likelihood check")
    args = parser.parse_args()

    print(f"Baseline: {BASELINE[:8]}")
    print("constants + utils");        check_constants()
    print("extinction");               check_extinction()
    print("spectral index");           check_spectral_index()
    print("params");                   check_params()
    print("modeling");                 xrt, opt = check_modeling()
    print("likelihood");               check_likelihood(xrt, opt, args.n_random)
    print("results");                  result_dir = check_results()
    print("plotting");                 check_plotting(result_dir, compare_png=args.png)
    print("driver");                   check_driver()
    print("\nAll equivalence checks PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
