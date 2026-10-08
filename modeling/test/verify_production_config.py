#!/usr/bin/env python3
"""Checks of the production configuration (--prior production --environment wind --xi both). Run from modeling/."""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import early_phase_model as m
from early_phase_mcmc import PRODUCTION_LABELS, print_parameter_table

ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}")


prod = m.make_param_defs("production", "wind", free_xi_e=True, free_xi_e_r=True)
final = m.make_param_defs("final", "wind", free_xi_e=True, free_xi_e_r=True)
labels = [m.param_label(p) for p in prod]
print_parameter_table(prod)
check("ndim = 14", len(prod) == 14)
check(f"parameter order = {PRODUCTION_LABELS}", labels == PRODUCTION_LABELS)
check("xi_e and xi_e_r present", "xi_e" in labels and "xi_e_r" in labels)
b = {p.name: p for p in prod}
expected = {"xi_e": (0.01, 1.0, "linear"), "xi_e_r": (0.01, 1.0, "linear")}
for name, (lo, hi, scale) in expected.items():
    check(f"{name} = {lo}-{hi}, {scale}", (b[name].lower, b[name].upper) == (lo, hi) and b[name].scale is m.Scale.LINEAR)
check("E_iso max = 1e55 erg", b["E_iso"].upper == 1e55)
check("tau max = 3000 s", b["tau"].upper == 3000.0)
check("eps_e_r max = 1.0", b["eps_e_r"].upper == 1.0)
check("eps_B_r max = 1.0", b["eps_B_r"].upper == 1.0)
check("p min = 2.01", b["p"].lower == 2.01)
changed = {"E_iso": "upper", "tau": "upper", "eps_e_r": "upper", "eps_B_r": "upper", "p": "lower"}
for p, q in zip(prod, final):
    same = (p.name, p.lower, p.upper, p.scale, p.gaussian_prior) == (q.name, q.lower, q.upper, q.scale, q.gaussian_prior)
    if p.name in changed:
        other = "lower" if changed[p.name] == "upper" else "upper"
        print(f"     {p.name:<8} final [{q.lower:.4g}, {q.upper:.4g}] -> production [{p.lower:.4g}, {p.upper:.4g}]")
        check(f"{p.name}: only the {changed[p.name]} bound may differ from final", getattr(p, other) == getattr(q, other)
              and p.scale is q.scale and p.gaussian_prior == q.gaussian_prior)
    else:
        check(f"{p.name}: identical to the historical final prior", same)
hist = m.make_param_defs("final", "wind")
check("historical 'final' (xi none) unchanged: tau <= 400, E_iso <= 1e54, eps_e_r/eps_B_r <= 0.5",
      [(p.name, p.upper) for p in hist if p.name in ("tau", "E_iso", "eps_e_r", "eps_B_r")]
      == [("E_iso", 1e54), ("tau", 400), ("eps_e_r", 0.5), ("eps_B_r", 0.5)])
explicit = m.make_param_defs("final", "wind", tau_max=3000, eps_e_r_max=1.0, eps_B_r_max=1.0, p_min=2.01, e_iso_max=1e55,
                             free_xi_e=True, free_xi_e_r=True)
check("production == final + the explicit historical overrides (same objects field by field)",
      [(p.name, p.lower, p.upper, p.scale, p.gaussian_prior) for p in prod]
      == [(p.name, p.lower, p.upper, p.scale, p.gaussian_prior) for p in explicit])
check("eps_e / eps_B pair present for the forward-shock constraint", {"eps_e", "eps_B"} <= set(b))
check("constraint pairs active in log_prior: " + str(m.EPS_SUM_PAIRS), m.EPS_SUM_PAIRS == (("eps_e", "eps_B"), ("eps_e_r", "eps_B_r")))

# xi_e -> forward-shock Radiation, xi_e_r -> reverse-shock Radiation: record the Radiation constructor calls
calls = []
real_radiation = m.Radiation
m.Radiation = lambda **kw: (calls.append(kw), real_radiation(**kw))[1]
x = np.array([54.0, 1.6, -1.6, 0.4, 2.02, -2.3, -1.4, 0.37, 2.6, 2.003, -0.17, -0.5, 0.083, -0.47])
model, a_v = m.build_model_from_params(x, prod)
m.Radiation = real_radiation
check(f"two Radiation objects built: forward then reverse ({len(calls)})", len(calls) == 2)
check(f"forward-shock Radiation gets xi_e = 0.37, eps_e = 10^-2.3, p = 2.02 (got xi_e={calls[0]['xi_e']})",
      calls[0]["xi_e"] == 0.37 and np.isclose(calls[0]["eps_e"], 10 ** -2.3) and calls[0]["p"] == 2.02)
check(f"reverse-shock Radiation gets xi_e_r = 0.083, eps_e_r = 10^-0.17, p_r = 2.003 (got xi_e={calls[1]['xi_e']})",
      calls[1]["xi_e"] == 0.083 and np.isclose(calls[1]["eps_e"], 10 ** -0.17) and calls[1]["p"] == 2.003)
src = open(m.__file__).read()
check("Model(..., fwd_rad=fwd_radiation, rvs_rad=rvs_radiation) in build_model_from_params",
      "fwd_rad=fwd_radiation" in src and "rvs_rad=rvs_radiation" in src)
check("MODEL_RESOLUTION = (0.1, 0.25, 10), MODEL_RTOL = 1e-5 (unchanged)",
      m.MODEL_RESOLUTION == (0.1, 0.25, 10) and m.MODEL_RTOL == 1e-5)
xrt, i, si = m.load_fit_data(include_spectral_index=True)
lp1 = m.log_probability(x, prod, xrt, i, si, "smooth")
x2 = x.copy(); x2[labels.index("xi_e")] = 0.5
x3 = x.copy(); x3[labels.index("xi_e_r")] = 0.2
lp2, lp3 = (m.log_probability(v, prod, xrt, i, si, "smooth") for v in (x2, x3))
check(f"log_probability responds to xi_e ({lp1:.3f} -> {lp2:.3f}) and to xi_e_r (-> {lp3:.3f})", lp1 != lp2 and lp1 != lp3)
v = x.copy(); v[labels.index("log10_eps_e_r")], v[labels.index("log10_eps_B_r")] = np.log10(0.7), np.log10(0.4)
check("eps_e_r + eps_B_r = 1.1 -> log_probability = -inf", m.log_probability(v, prod, xrt, i, si, "smooth") == -np.inf)
check("eps_sum_violated detects eps_e + eps_B > 1 (direct call; the production box caps eps_e <= 0.1, eps_B <= 0.05)",
      m.eps_sum_violated(np.r_[x[:5], np.log10(0.6), np.log10(0.5), x[7:]], prod))
print("PRODUCTION CONFIGURATION VERIFIED" if ok else "PRODUCTION CONFIGURATION CHECK FAILED")
sys.exit(0 if ok else 1)
