#!/usr/bin/env python3
"""Cross-environment check of the GRB early-phase stack (e.g. Python 3.14 env "yp" vs Python 3.13 env "yp313").

    <python of env A> verify_cross_environment.py --dump A.npz
    <python of env B> verify_cross_environment.py --dump B.npz
    any python        verify_cross_environment.py --compare A.npz B.npz

--dump evaluates, in the running environment: the processed data arrays (XRT, i band, spectral index after
galactic-extinction correction), every likelihood quantity for 200 production-posterior vectors (100 from each
production chain, fixed seed) plus 28 parameter-range tail points, and chain-analysis results on a fixed HDF slice
of both chains (autocorrelation times, quantiles, correlation matrix, provisional discard, q-coordinate transform).
Large-array arithmetic uses explicit ufuncs so the result does not depend on the NumPy elision bug.
--compare reports the largest differences and judges model quantities against the likelihood's own one-ulp
sensitivity (measured in the dump) and analysis quantities against floating-point round-off.
Run from modeling/.
"""
import argparse
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import numpy as np

RUNS = ["/home/hongyp007/yp_folder/projects/GRB/result/early_phase_production_20261008_015315",
        "/home/hongyp007/yp_folder/projects/GRB/result/early_phase_production_seed2_20261008_144242"]
MODEL_KEYS = ("chi2_XRT_flux", "chi2_i_band", "chi2_XRT_spectral_index", "chi2_total", "log_likelihood",
              "log_prior", "log_probability")


def dump(path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import emcee
    import scipy
    import VegasAfterglow
    import early_phase_model as m
    from early_phase_mcmc import _autocorr, provisional_discard

    defs = m.make_param_defs("production", "wind", free_xi_e=True, free_xi_e_r=True)
    labels = [m.param_label(d) for d in defs]
    xrt, i_data, si = m.load_fit_data(include_spectral_index=True)
    rng = np.random.default_rng(20261008)
    chains = [emcee.backends.HDFBackend(f"{r}/chain.h5", name="mcmc", read_only=True) for r in RUNS]
    pts = []
    for b in chains:
        c = b.get_chain()[50000:]
        flat_idx = rng.choice(c.shape[0] * c.shape[1], 100, replace=False)
        pts.append(c.reshape(-1, len(labels))[flat_idx])
        pooled = c[::50].reshape(-1, len(labels))
        for j in range(len(labels)):                  # lowest value of every parameter in this chain
            pts.append(pooled[np.argmin(pooled[:, j])][None])
    pts = np.concatenate(pts)
    vals = []
    sens = []
    e_i = labels.index("log10_E_iso")
    for k, x in enumerate(pts):
        c = m.chi2_components(x, defs, xrt, i_data, si, "smooth")
        vals.append([c[0], c[1], c[2], float(sum(c)), m.log_likelihood(x, defs, xrt, i_data, si, "smooth"),
                     m.log_prior(x, defs), m.log_probability(x, defs, xrt, i_data, si, "smooth")])
        if k < 60:                                    # one-ulp sensitivity of the likelihood (noise floor)
            y = x.copy()
            y[e_i] = np.nextafter(y[e_i], np.inf)
            c2 = m.chi2_components(y, defs, xrt, i_data, si, "smooth")
            sens.append(np.abs(np.subtract([c2[0], c2[1], c2[2], float(sum(c2))], [c[0], c[1], c[2], float(sum(c))])))
    analysis = {}
    for k, b in enumerate(chains):
        sl = b.get_chain()[60000:65000]
        flat = sl.reshape(-1, len(labels))
        analysis[f"run{k + 1}_tau"] = _autocorr(sl)
        analysis[f"run{k + 1}_discard"] = np.array([provisional_discard(sl)])
        analysis[f"run{k + 1}_quantiles"] = np.percentile(flat, [2.5, 16, 50, 84, 97.5], axis=0)
        analysis[f"run{k + 1}_mean"] = flat.mean(0)
        analysis[f"run{k + 1}_std"] = flat.std(0)
        analysis[f"run{k + 1}_corr"] = np.corrcoef(flat.T)
        analysis[f"run{k + 1}_q"] = m.to_sampling_coords(flat[:2000], labels, "q_E_xir")
        analysis[f"run{k + 1}_logprob_mean"] = np.array([b.get_log_prob()[60000:65000].mean()])
    np.savez(path, points=pts, model=np.array(vals), sensitivity=np.array(sens),
             xrt_flux=xrt["flux"].to_numpy(float), xrt_time=xrt["time"].to_numpy(float),
             i_flux=i_data["flux_mJy"].to_numpy(float), i_err=i_data["flux_mJy_error"].to_numpy(float),
             si_beta=si["beta"], labels=np.array(labels),
             env=np.array([sys.version.split()[0], np.__version__, scipy.__version__, VegasAfterglow.__version__,
                           emcee.__version__, platform.platform()]), **analysis)
    print(f"dumped {len(pts)} model points and chain analysis to {path} "
          f"(Python {sys.version.split()[0]}, NumPy {np.__version__})")


def compare(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    print(f"A: Python {a['env'][0]}, NumPy {a['env'][1]}, SciPy {a['env'][2]}, VegasAfterglow {a['env'][3]}, emcee {a['env'][4]}")
    print(f"B: Python {b['env'][0]}, NumPy {b['env'][1]}, SciPy {b['env'][2]}, VegasAfterglow {b['env'][3]}, emcee {b['env'][4]}")
    assert np.array_equal(a["points"], b["points"]), "different parameter vectors"
    ok = True

    def rel(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        fin = np.isfinite(x) & np.isfinite(y)
        same_special = np.array_equal(np.isfinite(x), np.isfinite(y))
        r = np.abs(x[fin] - y[fin]) / np.maximum(np.abs(x[fin]), 1e-300)
        return (r.max() if r.size else 0.0), same_special, np.mean(x == y)

    print("\ndata arrays:")
    for key in ("xrt_time", "xrt_flux", "i_flux", "i_err", "si_beta"):
        r, s, eq = rel(a[key], b[key])
        good = s and r <= 1e-14
        ok &= good
        print(f"  [{'OK' if good else 'FAIL'}] {key:<10} max rel diff {r:.2e}, bit-identical {eq:.0%}")

    print(f"\nmodel quantities ({len(a['points'])} posterior / tail vectors):")
    floor = np.maximum(a["sensitivity"].max(0), b["sensitivity"].max(0))      # one-ulp sensitivity, per chi2 part
    for k, key in enumerate(MODEL_KEYS):
        x, y = a["model"][:, k], b["model"][:, k]
        r, s, eq = rel(x, y)
        absdiff = np.abs(x - y)[np.isfinite(x) & np.isfinite(y)].max()
        limit = (floor[min(k, 3)] if k < 4 else 0.5 * floor[3]) if key != "log_prior" else 0.0
        good = s and (absdiff <= max(limit, 1e-12 * np.abs(x).max()))
        ok &= good
        print(f"  [{'OK' if good else 'FAIL'}] {key:<24} max |diff| {absdiff:.2e} (rel {r:.2e}), bit-identical {eq:.1%}; "
              f"one-ulp noise floor {limit:.1e}")

    print("\nchain analysis on steps 60000-65000 of both chains:")
    for key in sorted(k for k in a.files if k.startswith("run")):
        r, s, eq = rel(a[key], b[key])
        tol = 1e-10
        good = s and r <= tol
        ok &= good
        print(f"  [{'OK' if good else 'FAIL'}] {key:<20} max rel diff {r:.2e}, bit-identical {eq:.0%}")
    print("\nCROSS-ENVIRONMENT CONSISTENT" if ok else "\nCROSS-ENVIRONMENT CHECK FAILED")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", metavar="OUT.npz")
    ap.add_argument("--compare", nargs=2, metavar=("A.npz", "B.npz"))
    args = ap.parse_args()
    if args.dump:
        dump(args.dump)
    elif args.compare:
        compare(*args.compare)
    else:
        ap.error("use --dump or --compare")


if __name__ == "__main__":
    main()
