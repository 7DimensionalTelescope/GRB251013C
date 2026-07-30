#!/usr/bin/env python3
"""Run the final-model MCMC both ways and compare the resulting chains.

Baseline   : modeling/final_model.py, materialised from git at BASELINE
Refactored : fit_final_model.py, driving the grb package

test_refactor_equivalence.py proves the *pieces* agree. This proves the whole
sampler agrees end to end: same walkers, same proposals, same chain.

Why a seed is needed
--------------------
Both sides seed walker positions with unseeded np.random, so two runs of the
SAME code already differ. emcee 3.x seeds its own RandomState from np.random's
global state at sampler construction, so seeding np.random once at the top pins
both the start positions and every proposal thereafter. Because
fit_final_model.initial_positions draws in exactly the same order as the
baseline's inline loop, both sides then consume the global stream identically.

Bitwise agreement on several independent seeds is a complete result: if the
chains match for any seed, they match for the unseeded production case too.

Plotting is stubbed out on both sides -- it costs ~300 model builds per run, and
the figures are already proven byte-identical by test_refactor_equivalence.py.

Usage
-----
    # run one side (used internally, but callable directly)
    python compare_mcmc_runs.py run {baseline|refactored} OUTDIR SEED NSTEPS NWALKERS NCPUS

    # run both sides across seeds and diff each pair (the normal entry point)
    python compare_mcmc_runs.py compare --seeds 1 7 42 --nsteps 30 --nwalkers 54 --ncpus 27

Note: nwalkers must be >= 2 * ndim (52 for the 26-parameter model) or emcee
rejects the initial state.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BASELINE = "a40dd204e0b2bc7e5286387548fe14f75328e9f8"

ARTIFACTS = ("samples.npy", "log_probs.npy", "top_k_params.npy", "top_k_log_probs.npy")
TEXT_ARTIFACTS = ("labels.txt", "bestfit_params.txt")


def _stub_plotting(mod):
    for name in ("plot_light_curves", "plot_spectral_index_comparison", "plot_corner"):
        if hasattr(mod, name):
            setattr(mod, name, lambda *a, **k: None)


def run_one(side, outdir, seed, nsteps, nwalkers, ncpus):
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np

    sys.path.insert(0, str(ROOT))

    if side == "baseline":
        base = tempfile.mkdtemp(prefix="mcmc_baseline_")
        mdir = os.path.join(base, "modeling")
        os.mkdir(mdir)
        os.symlink(ROOT / "data", os.path.join(base, "data"))
        for name in ("utils.py", "final_model.py", "final_model_plotting.py"):
            blob = subprocess.check_output(
                ["git", "-C", str(ROOT), "show", f"{BASELINE}:modeling/{name}"])
            open(os.path.join(mdir, name), "wb").write(blob)
        sys.path.insert(0, mdir)
        import final_model as target
    else:
        import fit_final_model as target

    _stub_plotting(target)
    sys.argv = ["run", "--nsteps", str(nsteps), "--nwalkers", str(nwalkers),
                "--ncpus", str(ncpus), "--outdir", str(outdir)]
    np.random.seed(int(seed))
    target.main()


def diff(a, b):
    import numpy as np
    a, b = Path(a), Path(b)
    ok = True
    for name in ARTIFACTS:
        x, y = np.load(a / name), np.load(b / name)
        same = x.shape == y.shape and np.array_equal(x, y)
        ok &= same
        extra = ""
        if not same and x.shape == y.shape:
            d = np.abs(x - y)
            extra = f"  max|diff|={np.nanmax(d):.6g}  ndiff={int(np.sum(d != 0))}/{x.size}"
        print(f"    [{'EQUAL' if same else 'DIFFER'}] {name:22s} shape={x.shape}{extra}")
    for name in TEXT_ARTIFACTS:
        same = (a / name).read_text() == (b / name).read_text()
        ok &= same
        print(f"    [{'EQUAL' if same else 'DIFFER'}] {name}")
    lp = np.load(a / "log_probs.npy")
    print(f"    chain: {lp.size} samples, best log-prob {np.max(lp):.10f}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("side", choices=("baseline", "refactored"))
    r.add_argument("outdir")
    r.add_argument("seed", type=int)
    r.add_argument("nsteps", type=int)
    r.add_argument("nwalkers", type=int)
    r.add_argument("ncpus", type=int)

    c = sub.add_parser("compare")
    c.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 42])
    c.add_argument("--nsteps", type=int, default=30)
    c.add_argument("--nwalkers", type=int, default=54)
    c.add_argument("--ncpus", type=int, default=27)
    c.add_argument("--workdir", default=None)

    args = parser.parse_args()

    if args.cmd == "run":
        run_one(args.side, args.outdir, args.seed, args.nsteps, args.nwalkers, args.ncpus)
        return 0

    work = Path(args.workdir or tempfile.mkdtemp(prefix="mcmc_compare_"))
    work.mkdir(parents=True, exist_ok=True)
    ok = True
    for seed in args.seeds:
        print(f"\n=== seed {seed} (nsteps={args.nsteps} nwalkers={args.nwalkers}) ===")
        dirs = {}
        for side in ("baseline", "refactored"):
            out = work / f"{side}_{seed}"
            log = work / f"{side}_{seed}.log"
            with open(log, "wb") as fh:
                p = subprocess.run(
                    [sys.executable, str(HERE / "compare_mcmc_runs.py"), "run", side, str(out),
                     str(seed), str(args.nsteps), str(args.nwalkers), str(args.ncpus)],
                    stdout=fh, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                print(f"    {side} FAILED -- see {log}")
                ok = False
            dirs[side] = out
        if all(d.exists() for d in dirs.values()):
            ok &= diff(dirs["baseline"], dirs["refactored"])

    print("\n" + ("CHAINS BITWISE IDENTICAL on every seed" if ok else "CHAINS DIFFER"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
