#!/usr/bin/env python3
"""Exact-equivalence check of the q_E_xir sampling coordinates against the physical production coordinates.

For points taken from the run-1 and run-2 production chains (post-burn and burn-in), compares the old evaluation
log_probability(x) with the new path y = to_sampling_coords(x) -> from_sampling_coords(y) -> same functions, and
with the MCMC worker's own entry point _mcmc_log_probability(y). Also checks prior rejection at every E_iso and
xi_e_r boundary, and that the coordinate map has unit Jacobian determinant. Run from modeling/.
"""
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import emcee
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import early_phase_mcmc as mc
import early_phase_model as m

RUNS = ["/home/hongyp007/yp_folder/projects/GRB/result/early_phase_production_20261008_015315",
        "/home/hongyp007/yp_folder/projects/GRB/result/early_phase_production_seed2_20261008_144242"]
COORDS = m.Q_E_XIR
COOLING = "smooth"
QUANTITIES = ("chi2_XRT_flux", "chi2_i_band", "chi2_XRT_spectral_index", "chi2_total", "log_likelihood", "log_prior",
              "log_probability")

ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}")


def quantities(x, defs, data):
    c = m.chi2_components(x, defs, *data, COOLING)
    return [c[0], c[1], c[2], float(sum(c)), m.log_likelihood(x, defs, *data, COOLING), m.log_prior(x, defs),
            m.log_probability(x, defs, *data, COOLING)]


def main():
    defs = m.make_param_defs("production", "wind", free_xi_e=True, free_xi_e_r=True)
    labels = [m.param_label(p) for p in defs]
    data = m.load_fit_data(include_spectral_index=True)
    e_i, xi_i = labels.index("log10_E_iso"), labels.index("xi_e_r")
    print(f"physical labels: {labels}\nsampling labels: {m.sampling_labels(labels, COORDS)}")

    rng = np.random.default_rng(20261008)
    points, tags = [], []
    for run in RUNS:
        chain = emcee.backends.HDFBackend(f"{run}/chain.h5", name="mcmc", read_only=True).get_chain()
        for lo, hi, n, tag in ((50000, 100000, 250, "post-burn"), (0, 50000, 50, "burn-in")):
            steps, walkers = rng.integers(lo, hi, n), rng.integers(0, chain.shape[1], n)
            points += list(chain[steps, walkers])
            tags += [f"{run[-15:]}:{tag}"] * n
    points = np.array(points)
    print(f"{len(points)} chain points (run 1 and run 2; 250 post-burn + 50 burn-in each)")

    # 1. coordinate round trip
    y = m.to_sampling_coords(points, labels, COORDS)
    x_back = m.from_sampling_coords(y, labels, COORDS)
    check("q_E_xir = log10_E_iso + log10(xi_e_r) for every point",
          np.allclose(y[:, e_i], points[:, e_i] + np.log10(points[:, xi_i]), rtol=0, atol=1e-13))
    check("all coordinates other than log10_E_iso untouched by the transform (bit-identical)",
          np.array_equal(np.delete(y, e_i, 1), np.delete(points, e_i, 1)))
    d_e = np.abs(x_back[:, e_i] - points[:, e_i])
    print(f"     round trip log10_E_iso: max |diff| {d_e.max():.2e} dex, bit-identical {np.mean(d_e == 0):.1%} of points")
    phys_old = np.array([[m.physical_param_value(p, v) for p, v in zip(defs, x)] for x in points])
    phys_new = np.array([[m.physical_param_value(p, v) for p, v in zip(defs, x)] for x in x_back])
    rel_phys = np.max(np.abs(phys_new - phys_old) / np.abs(phys_old))
    check(f"reconstructed physical parameters agree (max relative difference {rel_phys:.2e})", rel_phys < 1e-13)

    # 2. every quantity, old (x) versus new (reconstructed from y).
    # Floating-point addition is not exactly invertible, so x -> q -> x can move log10_E_iso by one ulp (<= 7e-15 dex)
    # at a few points; the likelihood itself (VegasAfterglow adaptive integration, rtol 1e-5) reacts to a one-ulp input
    # change at the ~1e-9..1e-8 relative level. Hence three requirements:
    #   (i)   points whose reconstructed vector is bit-identical: every quantity bit-identical;
    #   (ii)  all points: the new path is bit-identical to the OLD functions evaluated on the reconstructed vector;
    #   (iii) points moved by the round trip: |old(x) - new(y)| no larger than the old code's own one-ulp sensitivity.
    old = np.array([quantities(x, defs, data) for x in points])
    new = np.array([quantities(x, defs, data) for x in x_back])
    same_input = np.all(x_back == points, axis=1)
    moved = ~same_input
    print(f"     {same_input.sum()} points reconstruct bit-identically, {moved.sum()} differ by <= 1 ulp in log10_E_iso")
    check("(i) bit-identical reconstructed vector -> all 7 quantities bit-identical",
          np.array_equal(old[same_input], new[same_input]))
    old_on_reconstructed = new                          # quantities() is the unchanged old code, applied to x(y)
    for k, q in enumerate(QUANTITIES):
        a, b = old[:, k], new[:, k]
        fin = np.isfinite(a) & np.isfinite(b)
        rel = np.abs(a[fin] - b[fin]) / np.maximum(np.abs(a[fin]), 1e-300)
        print(f"     {q:<24} finite {fin.sum()}/{len(a)}, max rel diff old(x) vs new(y) {rel.max():.2e}, "
              f"bit-identical {np.mean(a == b):.1%}")
    sens = []
    for x in points[moved]:
        x2 = x.copy()
        x2[e_i] = np.nextafter(x[e_i], np.inf)            # old code, no transform, one ulp up
        sens.append(np.abs(np.array(quantities(x2, defs, data)) - np.array(quantities(x, defs, data))))
    sens = np.array(sens)
    diff_moved = np.abs(new[moved] - old[moved])
    print("     old code's own one-ulp sensitivity on those points (max over points): "
          + ", ".join(f"{q} {v:.1e}" for q, v in zip(QUANTITIES, sens.max(0))))
    print("     round-trip differences on those points (max over points):           "
          + ", ".join(f"{q} {v:.1e}" for q, v in zip(QUANTITIES, diff_moved.max(0))))
    check("(iii) round-trip differences within the old likelihood's one-ulp noise floor (x 10 margin, per quantity)",
          np.all(diff_moved.max(0) <= 10 * np.maximum(sens.max(0), 1e-300)))
    # 3. the MCMC worker's own entry point in q coordinates
    context = (defs, *data, COOLING)
    mc._init_mcmc_worker(context, None, labels, COORDS)
    worker = np.array([mc._mcmc_log_probability(v) for v in y])
    check("(ii) _mcmc_log_probability(q coordinates) bit-identical to the old log_probability of the reconstructed "
          "vector, all points", np.array_equal(worker, old_on_reconstructed[:, -1]))
    rel = np.abs(worker - old[:, -1]) / np.abs(old[:, -1])
    check(f"_mcmc_log_probability(q coordinates) vs log_probability(original x): bit-identical at "
          f"{np.mean(worker == old[:, -1]):.1%}, max rel diff {rel.max():.2e} (moved points only)",
          np.array_equal(worker[same_input], old[same_input, -1]))
    mc._init_mcmc_worker(context, None, labels, "physical")
    check("_mcmc_log_probability in physical coordinates is bit-identical to log_probability(x)",
          np.array_equal(np.array([mc._mcmc_log_probability(v) for v in points[:50]]), old[:50, -1]))

    # 4. unit Jacobian of y -> x (numerical, central differences)
    x0 = points[0]
    y0 = m.to_sampling_coords(x0, labels, COORDS)
    h = 1e-6
    jac = np.array([(m.from_sampling_coords(y0 + h * np.eye(len(y0))[j], labels, COORDS)
                     - m.from_sampling_coords(y0 - h * np.eye(len(y0))[j], labels, COORDS)) / (2 * h) for j in range(len(y0))]).T
    check(f"Jacobian determinant of (q, xi_e_r) -> (log10_E_iso, xi_e_r) = {np.linalg.det(jac):.12f} (exact value 1)",
          abs(np.linalg.det(jac) - 1) < 1e-8)

    # 5. prior rejection at the E_iso / xi_e_r boundaries (q range coupled to xi_e_r)
    bounds = m._sampled_bounds(defs)
    (e_lo, e_hi), (xi_lo, xi_hi) = bounds[e_i], bounds[xi_i]
    base = points[0].copy()
    eps = 1e-9

    def case(log_e, xi):
        x = base.copy()
        x[e_i], x[xi_i] = log_e, xi
        return x

    def y_of(log_e, xi):                       # q built directly from the intended physical values
        y = base.copy()
        y[xi_i] = xi
        with np.errstate(divide="ignore", invalid="ignore"):
            y[e_i] = log_e + np.log10(xi)
        return y

    mc._init_mcmc_worker(context, None, labels, COORDS)
    cases = [
        ("E_iso just below its upper bound 1e55", e_hi - eps, 0.05, True),
        ("E_iso just above its upper bound 1e55", e_hi + eps, 0.05, False),
        ("E_iso just above its lower bound 5e51", e_lo + eps, 0.05, True),
        ("E_iso just below its lower bound 5e51", e_lo - eps, 0.05, False),
        ("xi_e_r = 0.01 (lower bound, inclusive)", 54.0, xi_lo, True),
        ("xi_e_r just below 0.01", 54.0, xi_lo - eps, False),
        ("xi_e_r = 1 (upper bound, inclusive)", 54.0, xi_hi, True),
        ("xi_e_r just above 1", 54.0, xi_hi + eps, False),
        ("xi_e_r = 0", 54.0, 0.0, False),
        ("xi_e_r < 0", 54.0, -0.1, False),
        ("same q, xi_e_r = 0.02 -> E_iso inside", 54.9, 0.02, True),
        ("same q, xi_e_r = 0.01 -> E_iso = 10^55.2 outside", 54.9 + np.log10(2), 0.01, False),
        ("E_iso at the cap and xi_e_r at its minimum (corner)", e_hi - eps, xi_lo, True),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # no RuntimeWarning (log of non-positive xi_e_r) may escape
        for name, log_e, xi, allowed in cases:
            lp_old_prior = m.log_prior(case(log_e, xi), defs)
            yv = y_of(log_e, xi)
            x_new = m.from_sampling_coords(yv, labels, COORDS)
            lp_new_prior = m.log_prior(x_new, defs)
            lp_worker = mc._mcmc_log_probability(yv)
            good = (np.isfinite(lp_new_prior) == allowed and np.isfinite(lp_old_prior) == allowed
                    and (lp_new_prior == lp_old_prior or not allowed) and (np.isfinite(lp_worker) or not allowed)
                    and (lp_worker == -np.inf if not allowed else True))
            check(f"{name:<52} prior old {lp_old_prior:>9.4g}  new {lp_new_prior:>9.4g}  worker log_prob {lp_worker:>10.5g}", good)
    print("Q REPARAMETERIZATION VERIFIED" if ok else "Q REPARAMETERIZATION CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
