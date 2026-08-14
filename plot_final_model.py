#!/usr/bin/env python3
"""Regenerate plots for a saved final-model MCMC run.

Thin driver over `grb.plotting`, analogous to the former
`modeling/final_model_plotting.py`. Defaults to the latest `final_*`
directory under `modeling/fit_results/`.

    python plot_final_model.py
    python plot_final_model.py modeling/fit_results/final_flare_wing_20260730_171914
"""
from pathlib import Path
import argparse

from grb.const import FIT_RESULTS_DIR
from grb.plotting import plot_light_curves, plot_spectral_index_comparison
from grb.results import latest_result_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot fit_final_model.py results")
    parser.add_argument(
        "result_dir",
        nargs="?",
        default=None,
        help="Result directory (default: latest final_* directory)",
    )
    parser.add_argument(
        "--band-draws",
        type=int,
        default=300,
        help="Posterior draws for 3-sigma (0.15-99.85%%) LC envelope (0 disables; default: 300)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.result_dir:
        result_dir = Path(args.result_dir)
    else:
        result_dir = latest_result_dir(FIT_RESULTS_DIR, "final_")

    if not result_dir.exists():
        print(f"Error: Result directory not found: {result_dir}")
        return 1

    print("=" * 60)
    print("Plotting fit_final_model.py results")
    print("=" * 60)
    print(f"\nResults directory: {result_dir}\n")

    print(f"Plotting light curves for: {result_dir.name}")
    print(f"  Model: Core+RS + Flare + Wing")
    if args.band_draws:
        print(f"  Posterior band: {args.band_draws} draws (3σ / 0.15-99.85%)")
    plot_light_curves(result_dir, band_draws=args.band_draws)

    print(f"\nPlotting spectral index comparison...")
    plot_spectral_index_comparison(result_dir)

    print("\n" + "=" * 60)
    print("✓ All plots generated successfully!")
    print("=" * 60)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
