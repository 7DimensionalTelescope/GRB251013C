#!/usr/bin/env python3
"""Generate a single afterglow light-curve model for manually-specified parameter
values, overlaid on all observed data.

Unlike `sweep_search.py` (which scans one parameter across its full prior
range around a baseline), this evaluates exactly one hand-picked parameter set --
useful for "what if eps_e=0.05 and n_ism=200" checks. Any parameter not given on
the command line falls back to INITIAL_GUESS from fit_final_model.py, or the prior
bounds midpoint if that parameter isn't in INITIAL_GUESS (e.g. A_star for
--environment wind).

    python manual_model.py --eps_e 0.08 --n_ism 300 --include-wing
"""
from datetime import datetime
from pathlib import Path
import argparse
import json

import numpy as np

from VegasAfterglow import Scale

from fit_final_model import INITIAL_GUESS
from grb.functions import FLARE_SHAPES
from grb.params import ENVIRONMENTS, make_param_defs
from grb.plotting import render_light_curve_figure

DEFAULT_RESULTS_ROOT = Path("/home/hongyp007/hongyp/projects/GRB/results")


def parse_args(argv=None):
    # Which --<param> flags are valid depends on --environment/--include-flare/
    # --include-wing, so those three are resolved first via parse_known_args.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--environment", choices=ENVIRONMENTS, default="ism")
    pre.add_argument("--include-flare", default=True, action=argparse.BooleanOptionalAction)
    pre.add_argument("--flare-function", dest="flare_shape", choices=FLARE_SHAPES, default="norris")
    pre.add_argument("--include-wing", default=False, action=argparse.BooleanOptionalAction)
    pre_args, _ = pre.parse_known_args(argv)

    param_defs = make_param_defs(
        include_flare=pre_args.include_flare,
        include_wing=pre_args.include_wing,
        environment=pre_args.environment,
        flare_shape=pre_args.flare_shape,
    )

    parser = argparse.ArgumentParser(
        parents=[pre],
        description="Generate a single model light curve for manually-set parameter "
                     "values, overlaid on the observed data.",
    )
    parser.add_argument("--mark-7dt-time", default=True, action=argparse.BooleanOptionalAction,
                        help="Draw a vertical line at the 7DT spectrum time in the "
                             "Optical Light Curves panel (default: on)")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: manual_<environment>_<timestamp> "
                             f"under {DEFAULT_RESULTS_ROOT})")
    for p in param_defs:
        scale_desc = "log" if p.scale is Scale.LOG else "linear"
        parser.add_argument(f"--{p.name}", type=float, default=None,
                            help=f"physical value, prior box [{p.lower:g}, {p.upper:g}] "
                                 f"({scale_desc}-scaled in the fit); default: baseline")
    return parser.parse_args(argv), param_defs


def build_baseline(param_defs):
    """Baseline value per param_def: INITIAL_GUESS if present (clipped to bounds,
    with a note), else the bounds midpoint. Same convention as sweep_search.py's
    build_sweep_baseline, reimplemented here to avoid a script-to-script import."""
    baseline, notes = {}, []
    for p in param_defs:
        if p.name in INITIAL_GUESS:
            value = float(INITIAL_GUESS[p.name])
            if not (p.lower <= value <= p.upper):
                clipped = min(max(value, p.lower), p.upper)
                notes.append(f"{p.name}: INITIAL_GUESS value {value:g} outside "
                             f"[{p.lower:g}, {p.upper:g}], clipped to {clipped:g}")
                value = clipped
        else:
            value = (float(np.sqrt(p.lower * p.upper)) if p.scale is Scale.LOG
                     else 0.5 * (p.lower + p.upper))
            notes.append(f"{p.name}: not in INITIAL_GUESS, set to bounds midpoint {value:g}")
        baseline[p.name] = value
    return baseline, notes


def check_physical_validity(params):
    """Non-blocking check of the coupled validity rules fit_final_model.py's MCMC
    and mixing_search.py/sweep_search.py enforce. Returns a list of violation strings (empty if none);
    manual exploration is never blocked by this, only warned about."""
    violations = []
    for suffix in ("", "_r", "_wing"):
        eps_e, eps_b = params.get(f"eps_e{suffix}"), params.get(f"eps_B{suffix}")
        if eps_e is not None and eps_b is not None and eps_e + eps_b > 1.0:
            violations.append(f"eps_e{suffix} + eps_B{suffix} = {eps_e + eps_b:g} > 1")
    if "theta_c_wing" in params and params["theta_c_wing"] <= params["theta_c_core"]:
        violations.append("theta_c_wing <= theta_c_core")
    if "tau_decay_flare" in params and params["tau_decay_flare"] <= params["tau_rise_flare"]:
        violations.append("tau_decay_flare <= tau_rise_flare")
    return violations


def main(argv=None):
    args, param_defs = parse_args(argv)

    baseline, notes = build_baseline(param_defs)
    for note in notes:
        print(f"  Note: {note}")

    params, sources = {}, {}
    for p in param_defs:
        cli_value = getattr(args, p.name)
        if cli_value is not None:
            if not (p.lower <= cli_value <= p.upper):
                print(f"  Warning: --{p.name}={cli_value:g} is outside the prior box "
                      f"[{p.lower:g}, {p.upper:g}] (allowed for manual exploration)")
            params[p.name] = cli_value
            sources[p.name] = "cli"
        else:
            params[p.name] = baseline[p.name]
            sources[p.name] = "baseline"

    flare_desc = f"on/{args.flare_shape}" if args.include_flare else "off"
    print(f"\nParameters ({args.environment} medium, "
          f"flare={flare_desc}, "
          f"wing={'on' if args.include_wing else 'off'}):")
    print(f"  {'name':<20} {'value':>14}   source")
    for p in param_defs:
        print(f"  {p.name:<20} {params[p.name]:>14.6g}   {sources[p.name]}")

    validity_warnings = check_physical_validity(params)
    for warning in validity_warnings:
        print(f"  Warning: physically invalid combination ({warning}) -- plotting anyway")

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = DEFAULT_RESULTS_ROOT / f"manual_{args.environment}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    outfile = render_light_curve_figure(
        params, args.include_flare, args.include_wing, outdir / "manual_lc.png",
        mark_7dt_time=args.mark_7dt_time, flare_shape=args.flare_shape,
    )

    config = {
        "environment": args.environment,
        "include_flare": args.include_flare,
        "flare_shape": args.flare_shape,
        "include_wing": args.include_wing,
        "params": params,
        "sources": sources,
        "validity_warnings": validity_warnings,
    }
    (outdir / "params_used.json").write_text(json.dumps(config, indent=2))

    print(f"\nSaved: {outfile}")
    print(f"Saved: {outdir / 'params_used.json'}")


if __name__ == "__main__":
    main()
