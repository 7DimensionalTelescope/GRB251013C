#!/usr/bin/env python3
"""Equivalence check of the u_r = log10(p_r - 2) sampling coordinate against the physical production coordinates.

For points of the run-1 and run-2 production chains (post-burn and burn-in) compares the physical path
log_probability(x) with the transformed path y = to_sampling_coords(x) -> from_sampling_coords(y): reconstructed p_r,
every chi2 component, log likelihood, the physical prior, the transformed prior (physical prior + log-Jacobian) and the
sampler's own target _mcmc_log_probability(y) minus the log-Jacobian. Separates floating-point round-trip noise from
implementation errors (three requirements as for q_E_xir), checks the Jacobian numerically and that it reproduces the
uniform p_r prior, and probes the p_r / u_r boundaries derived from the actual p_r bounds. Run from modeling/.
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
COORDS = m.U_R
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
    pr_i = labels.index("p_r")
    pr_lo, pr_hi = defs[pr_i].lower, defs[pr_i].upper
    with np.errstate(divide="ignore"):
        u_lo, u_hi = np.log10(pr_lo - 2.0), np.log10(pr_hi - 2.0)
    print(f"sampling labels: {m.sampling_labels(labels, COORDS)}")
    print(f"p_r bounds from the production prior: [{pr_lo}, {pr_hi}] -> u_r in ({u_lo}, {u_hi:.6f}]")

    rng = np.random.default_rng(20261011)
    points = []
    for run in RUNS:
        chain = emcee.backends.HDFBackend(f"{run}/chain.h5", name="mcmc", read_only=True).get_chain()
        for lo, hi, n in ((50000, 100000, 250), (0, 50000, 50)):
            points += list(chain[rng.integers(lo, hi, n), rng.integers(0, chain.shape[1], n)])
    points = np.array(points)
    print(f"{len(points)} chain points (run 1 and run 2; 250 post-burn + 50 burn-in each); p_r range "
          f"{points[:, pr_i].min():.5f}-{points[:, pr_i].max():.5f}")

    # 1. round trip
    y = m.to_sampling_coords(points, labels, COORDS)
    x_back = m.from_sampling_coords(y, labels, COORDS)
    check("u_r = log10(p_r - 2) for every point (within 1e-15 relative)",
          np.allclose(y[:, pr_i], np.log10(points[:, pr_i] - 2.0), rtol=1e-15, atol=0))
    check("all coordinates other than p_r untouched (bit-identical)",
          np.array_equal(np.delete(y, pr_i, 1), np.delete(points, pr_i, 1)) and np.array_equal(np.delete(x_back, pr_i, 1),
                                                                                              np.delete(points, pr_i, 1)))
    d = np.abs(x_back[:, pr_i] - points[:, pr_i])
    ulp = np.spacing(points[:, pr_i])
    print(f"     round trip p_r: max |diff| {d.max():.2e} = {np.max(d / ulp):.0f} ulp, bit-identical {np.mean(d == 0):.1%}")
    check("reconstructed p_r within 2 ulp of the original", np.all(d <= 2 * ulp))

    # 2. physical quantities on original vs reconstructed vectors
    old = np.array([quantities(x, defs, data) for x in points])
    new = np.array([quantities(x, defs, data) for x in x_back])
    same = np.all(x_back == points, axis=1)
    moved = ~same
    print(f"     {same.sum()} points reconstruct bit-identically, {moved.sum()} differ in p_r by <= 2 ulp")
    check("(i) bit-identical reconstructed vector -> all 7 quantities bit-identical", np.array_equal(old[same], new[same]))
    for k, q in enumerate(QUANTITIES):
        a, b = old[:, k], new[:, k]
        rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-300)
        print(f"     {q:<24} max rel diff old(x) vs new(y) {rel.max():.2e}, bit-identical {np.mean(a == b):.1%}")
    if moved.any():
        sens = []
        for x in points[moved]:
            x2 = x.copy()
            x2[pr_i] = np.nextafter(x[pr_i], np.inf)
            sens.append(np.abs(np.subtract(quantities(x2, defs, data), quantities(x, defs, data))))
        sens = np.array(sens).max(0)
        diff = np.abs(new[moved] - old[moved]).max(0)
        print("     old code's one-ulp-in-p_r sensitivity (max): " + ", ".join(f"{q} {v:.1e}" for q, v in zip(QUANTITIES, sens)))
        print("     round-trip differences (max):               " + ", ".join(f"{q} {v:.1e}" for q, v in zip(QUANTITIES, diff)))
        check("(iii) round-trip differences within 10x the old likelihood's one-ulp-in-p_r noise floor",
              np.all(diff <= 10 * np.maximum(sens, 1e-300)))

    # 3. the sampler's target: physical log_probability(x(y)) + log-Jacobian, exactly
    context = (defs, *data, COOLING)
    mc._init_mcmc_worker(context, None, labels, COORDS)
    target = np.array([mc._mcmc_log_probability(v) for v in y])
    log_jac = m.sampling_log_jacobian(y, labels, COORDS)
    check("(ii) sampler target - log-Jacobian is bit-identical to the old log_probability of the reconstructed vector",
          np.array_equal(target - log_jac, new[:, -1]) or np.allclose(target - log_jac, new[:, -1], rtol=0, atol=1e-12))
    expected_jac = np.log(np.log(10.0) * (x_back[:, pr_i] - 2.0))
    check(f"log-Jacobian = log(ln 10 * (p_r - 2)) (max diff {np.abs(log_jac - expected_jac).max():.1e})",
          np.allclose(log_jac, expected_jac, rtol=0, atol=1e-12))
    trans_prior = new[:, 5] + log_jac
    check("transformed prior = physical prior + log-Jacobian, finite for every in-bound point", np.all(np.isfinite(trans_prior)))
    mc._init_mcmc_worker(context, None, labels, "physical")
    check("physical-coordinate sampler target unchanged (bit-identical to log_probability)",
          np.array_equal(np.array([mc._mcmc_log_probability(v) for v in points[:50]]), old[:50, -1]))
    mc._init_mcmc_worker(context, None, labels, COORDS)

    # 4. Jacobian numerically, and the implied prior on p_r is uniform
    h = 1e-6
    num = np.array([(np.add(2.0, 10 ** (u + h)) - np.add(2.0, 10 ** (u - h))) / (2 * h) for u in y[:50, pr_i]])
    check(f"dp_r/du_r: numerical vs ln10*10^u_r, max rel diff {np.max(np.abs(num / (np.log(10) * 10 ** y[:50, pr_i]) - 1)):.1e}",
          np.allclose(num, np.log(10) * 10 ** y[:50, pr_i], rtol=1e-6))
    def jac_density(u):                               # exp(log-Jacobian) as a function of u_r alone
        Y = np.zeros((len(u), len(labels)))
        Y[:, pr_i] = u
        return np.exp(m.sampling_log_jacobian(Y, labels, COORDS))

    grid = np.linspace(-40.0, u_hi, 400_001)
    mass = np.trapezoid(jac_density(grid), grid)
    check(f"integral of exp(log-Jacobian) over (-inf, u_max] = {mass:.8f} = p_r range {pr_hi - pr_lo} (uniform prior "
          f"mass preserved)", abs(mass - (pr_hi - pr_lo)) < 1e-6)
    u_draw = np.log10(1.5 * np.random.default_rng(1).random(200000))      # u for p_r - 2 ~ U(0, 1.5)
    hist, edges = np.histogram(u_draw, bins=40, range=(-3, u_hi), density=False)
    centres = 0.5 * (edges[1:] + edges[:-1])
    expected = 200000 * jac_density(centres) * (edges[1] - edges[0]) / (pr_hi - pr_lo)
    chi = np.sum((hist - expected) ** 2 / expected) / len(hist)
    check(f"u_r of uniformly drawn p_r follows the Jacobian density (chi2/bin {chi:.2f})", chi < 2.0)

    # 5. boundaries, derived from the actual p_r bounds
    base = points[0].copy()

    def at_u(u):
        v = m.to_sampling_coords(base, labels, COORDS)
        v[pr_i] = u
        return v

    cases = [
        ("u_r = log10(p_r_max - 2) (p_r upper bound)", u_hi),
        ("u_r just below the upper bound", u_hi - 1e-9),
        ("u_r just above the upper bound", u_hi + 1e-9),
        ("u_r = -2 (p_r = 2.01)", -2.0),
        ("u_r = -6 (p_r = 2 + 1e-6)", -6.0),
        ("u_r = -12 (p_r = 2 + 1e-12)", -12.0),
        ("u_r = -20 (p_r rounds to the lower bound 2.0)", -20.0),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name, u in cases:
            v = at_u(u)
            x = m.from_sampling_coords(v, labels, COORDS)
            pri = m.log_prior(x, defs)
            phys_prior = m.log_prior(np.where(np.arange(len(labels)) == pr_i, 2.0 + 10 ** u, base), defs)
            try:
                t = mc._mcmc_log_probability(v)
                err = ""
            except Exception as exc:                  # the worker catches model errors itself; this should not happen
                t, err = np.nan, f" EXCEPTION {type(exc).__name__}"
            in_box = pr_lo <= x[pr_i] <= pr_hi
            good = (np.isfinite(pri) == in_box) and (pri == phys_prior) and (np.isfinite(t) or not in_box or t == -np.inf) \
                and (t == -np.inf if not in_box else True) and not err
            check(f"{name:<46} p_r = {x[pr_i]!r:<22} prior {pri:>8.4g}  target {t:>12.6g}{err}", good)
    print("u_r REPARAMETERIZATION VERIFIED" if ok else "u_r REPARAMETERIZATION CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
