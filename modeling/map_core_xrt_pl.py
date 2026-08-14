#!/usr/bin/env python3
"""Map core-FS parameter region for single-power-law XRT (no wing).

No MCMC. Parallel Latin-hypercube / branch grids over core FS params against
XRT flux + spectral index only. Flare / RS / A_V / tau are frozen.

Branches (analytic):
  HP  hard p<2, high eps_B   (Gamma ~ p/2+1, above nu_c)
  SC  slow cooling, low eps_B (Gamma ~ (p+1)/2, nu_c above XRT)
  MID p~2.2, mid eps_B
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from dotenv import load_dotenv
from VegasAfterglow import Scale

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from grb.const import FIT_RESULTS_DIR
from grb.likelihood import compute_model_flux_all_bands, log_likelihood
from grb.modeling import load_all_optical_data
from grb.params import make_param_defs
from grb.spectral_index import load_xrt_spectral_index

SEED_PATH = Path(__file__).resolve().parent / "corexrt_seeds.json"

FROZEN = {
    "tau": 20.0651,
    "p_r": 2.995,
    "eps_e_r": 0.0362407,
    "eps_B_r": 0.266559,
    "xi_r": 0.999,
    "A_V": 0.247939,
}

# Core params scanned (physical units for grids; converted to theta as needed).
CORE_SCAN = [
    "E_iso_core", "Gamma0_core", "theta_c_core", "n_ism",
    "p", "eps_e", "eps_B", "xi",
]

# Flare frozen at HP_narrow seed (best known XRT+SI point).
FLARE_FREEZE_FROM = "HP_narrow"

_CTX = {}


def _label(p):
    return f"log10_{p.name}" if p.scale is Scale.LOG else p.name


def build_base_theta(pdefs, flare_vals):
    defaults = {
        "E_iso_core": 6.6226e51,
        "Gamma0_core": 402.145,
        "theta_c_core": 0.42,
        "n_ism": 200.0,
        "p": 1.7,
        "eps_e": 0.08,
        "eps_B": 0.09,
        "xi": 0.5,
        **FROZEN,
        **flare_vals,
    }
    th = []
    for p in pdefs:
        v = defaults[p.name]
        th.append(np.log10(v) if p.scale is Scale.LOG else float(v))
    return np.asarray(th, dtype=float)


def core_dict_to_free_slice(core, pdefs, base_theta, core_ix):
    """Return full theta with core physical values written into base."""
    th = base_theta.copy()
    for name, val in core.items():
        i = core_ix[name]
        p = pdefs[i]
        th[i] = np.log10(val) if p.scale is Scale.LOG else float(val)
    return th


def eval_one(core):
    """Worker: evaluate one core-param dict. Returns (core_list, logL)."""
    th = core_dict_to_free_slice(core, _CTX["pdefs"], _CTX["base_theta"], _CTX["core_ix"])
    ll = log_likelihood(
        th, _CTX["pdefs"], _CTX["xrt_data"], [], True, False, _CTX["xrt_index_data"]
    )
    vec = [float(core[k]) for k in CORE_SCAN]
    return vec, float(ll) if np.isfinite(ll) else -np.inf


def lhs_unit(n, d, rng):
    """Simple Latin-hypercube in [0,1]^d."""
    u = np.empty((n, d))
    for j in range(d):
        cut = np.linspace(0.0, 1.0, n + 1)
        u[:, j] = rng.uniform(cut[:-1], cut[1:])
        rng.shuffle(u[:, j])
    return u


def map_unit_to_core(u, box):
    """Map unit hypercube row to physical core params. Log-space for LOG keys."""
    core = {}
    for j, name in enumerate(CORE_SCAN):
        lo, hi, is_log = box[name]
        if is_log:
            core[name] = 10 ** (np.log10(lo) + u[j] * (np.log10(hi) - np.log10(lo)))
        else:
            core[name] = lo + u[j] * (hi - lo)
    return core


# Branch boxes: (lo, hi, logscale)
BOX_HP = {
    "E_iso_core": (1e51, 1e54, True),
    "Gamma0_core": (150, 1500, True),
    "theta_c_core": (0.15, 0.7, True),
    "n_ism": (1.0, 2000, True),
    "p": (1.55, 1.95, False),
    "eps_e": (0.01, 0.4, True),
    "eps_B": (0.01, 0.3, True),
    "xi": (0.1, 1.0, False),
}
BOX_SC = {
    "E_iso_core": (1e51, 3e54, True),
    "Gamma0_core": (100, 1000, True),
    "theta_c_core": (0.15, 0.7, True),
    "n_ism": (0.1, 1000, True),
    "p": (2.35, 2.9, False),
    "eps_e": (0.01, 0.4, True),
    "eps_B": (1e-7, 3e-4, True),
    "xi": (0.1, 1.0, False),
}
BOX_MID = {
    "E_iso_core": (1e51, 1e54, True),
    "Gamma0_core": (150, 1200, True),
    "theta_c_core": (0.1, 0.7, True),
    "n_ism": (0.5, 1500, True),
    "p": (2.0, 2.4, False),
    "eps_e": (0.01, 0.3, True),
    "eps_B": (3e-4, 0.02, True),
    "xi": (0.1, 1.0, False),
}


def structured_grid(box, n_p, n_eb, n_th, n_en, rng):
    """Coarse product grid on (p, eps_B, theta_c) x LHS on (E, n, Gamma, eps_e, xi)."""
    p_lo, p_hi, _ = box["p"]
    eb_lo, eb_hi, eb_log = box["eps_B"]
    th_lo, th_hi, th_log = box["theta_c_core"]

    p_vals = np.linspace(p_lo, p_hi, n_p)
    if eb_log:
        eb_vals = np.geomspace(eb_lo, eb_hi, n_eb)
    else:
        eb_vals = np.linspace(eb_lo, eb_hi, n_eb)
    if th_log:
        th_vals = np.geomspace(th_lo, th_hi, n_th)
    else:
        th_vals = np.linspace(th_lo, th_hi, n_th)

    # Remaining dims via small LHS
    rest_names = ["E_iso_core", "Gamma0_core", "n_ism", "eps_e", "xi"]
    u_rest = lhs_unit(n_en, len(rest_names), rng)
    rest_pts = []
    for row in u_rest:
        r = {}
        for j, name in enumerate(rest_names):
            lo, hi, is_log = box[name]
            if is_log:
                r[name] = 10 ** (np.log10(lo) + row[j] * (np.log10(hi) - np.log10(lo)))
            else:
                r[name] = lo + row[j] * (hi - lo)
        rest_pts.append(r)

    cores = []
    for p in p_vals:
        for eb in eb_vals:
            for th in th_vals:
                for r in rest_pts:
                    cores.append({
                        "p": float(p),
                        "eps_B": float(eb),
                        "theta_c_core": float(th),
                        **r,
                    })
    return cores


def rng_str_arr(x, logscale=False):
    lo, med, hi = np.percentile(x, [5, 50, 95])
    if logscale:
        return {
            "p05": float(lo), "p50": float(med), "p95": float(hi),
            "str": f"{lo:.3g} .. {hi:.3g} (med {med:.3g})",
        }
    return {
        "p05": float(lo), "p50": float(med), "p95": float(hi),
        "str": f"{lo:.2f} .. {hi:.2f} (med {med:.2f})",
    }


def optical_chi2(th, pdefs, xrt_data, optical_datasets, xrt_index_data):
    params = {}
    for p, v in zip(pdefs, th):
        params[p.name] = 10 ** v if p.scale is Scale.LOG else v
    try:
        _, optical_models, _ = compute_model_flux_all_bands(
            params, xrt_data, optical_datasets, True, False, xrt_index_data
        )
    except Exception:
        return np.inf
    chi2 = 0.0
    for dataset, model_flux in zip(optical_datasets, optical_models):
        chi2 += float(np.sum(((dataset["flux_mJy"] - model_flux) / dataset["flux_err"]) ** 2))
    return chi2


def late_xrt_resid(th, pdefs, xrt_data, xrt_index_data, t_min=1e4):
    params = {}
    for p, v in zip(pdefs, th):
        params[p.name] = 10 ** v if p.scale is Scale.LOG else v
    xrt_model, _, _ = compute_model_flux_all_bands(
        params, xrt_data, [], True, False, xrt_index_data
    )
    t = xrt_data["time"]
    m = t >= t_min
    if not np.any(m):
        return []
    resid = (xrt_data["flux"][m] - xrt_model[m]) / xrt_data["flux_error"][m]
    return [
        {"t_hr": float(t[m][i] / 3600), "resid_sigma": float(resid[i])}
        for i in range(int(np.sum(m)))
    ]


def flare_vals_from_seed(seeds_raw, name):
    free = seeds_raw["free"]
    by_name = {r[0]: r[2] for r in seeds_raw["results"]}
    v = by_name[name]
    mapping = {
        "log10_t_start_flare": "t_start_flare",
        "log10_tau_rise_flare": "tau_rise_flare",
        "log10_tau_decay_flare": "tau_decay_flare",
        "log10_A_flare": "A_flare",
        "flare_beta": "flare_beta",
    }
    out = {}
    for lab, key in mapping.items():
        i = free.index(lab)
        out[key] = 10 ** v[i] if lab.startswith("log10_") else v[i]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-lhs", type=int, default=400,
                        help="LHS points per branch (in addition to structured grid)")
    parser.add_argument("--n-p", type=int, default=5)
    parser.add_argument("--n-epsB", type=int, default=5)
    parser.add_argument("--n-theta", type=int, default=5)
    parser.add_argument("--n-en", type=int, default=4,
                        help="LHS draws for (E, Gamma, n, eps_e, xi) per grid cell")
    parser.add_argument("--ncpus", type=int, default=12)
    parser.add_argument("--delta-logL", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    print("Loading XRT + spectral index...")
    xrt_data, optical_datasets = load_all_optical_data()
    xrt_index_data = load_xrt_spectral_index()
    print(f"  XRT points: {len(xrt_data['time'])}")
    print(f"  SI points:  {len(xrt_index_data['time'])}")

    seeds_raw = json.loads(SEED_PATH.read_text())
    flare_vals = flare_vals_from_seed(seeds_raw, FLARE_FREEZE_FROM)
    print(f"  Flare frozen from {FLARE_FREEZE_FROM}: {flare_vals}")

    pdefs = make_param_defs(include_flare=True, include_wing=False)
    labels = [_label(p) for p in pdefs]
    core_ix = {p.name: i for i, p in enumerate(pdefs)}
    base_theta = build_base_theta(pdefs, flare_vals)

    _CTX.clear()
    _CTX.update({
        "pdefs": pdefs,
        "base_theta": base_theta,
        "core_ix": core_ix,
        "xrt_data": xrt_data,
        "xrt_index_data": xrt_index_data,
    })

    rng = np.random.default_rng(args.seed)
    branches_box = [("HP", BOX_HP), ("SC", BOX_SC), ("MID", BOX_MID)]

    all_cores = []
    branch_of = []
    for bname, box in branches_box:
        grid = structured_grid(box, args.n_p, args.n_epsB, args.n_theta, args.n_en, rng)
        u = lhs_unit(args.n_lhs, len(CORE_SCAN), rng)
        lhs = [map_unit_to_core(row, box) for row in u]
        # Also include known seed cores for that branch if present
        pts = grid + lhs
        print(f"  {bname}: {len(pts)} points (grid={len(grid)}, lhs={len(lhs)})")
        all_cores.extend(pts)
        branch_of.extend([bname] * len(pts))

    # Add exact seed cores
    by_name = {r[0]: r[2] for r in seeds_raw["results"]}
    free = seeds_raw["free"]
    for sname, vec in by_name.items():
        core = {}
        for lab, val in zip(free, vec):
            if lab.startswith("log10_") and lab.replace("log10_", "") in CORE_SCAN:
                core[lab.replace("log10_", "")] = 10 ** val
            elif lab in CORE_SCAN:
                core[lab] = val
        if len(core) == len(CORE_SCAN):
            all_cores.append(core)
            # classify
            if core["eps_B"] < 10 ** (-3.5):
                branch_of.append("SC")
            elif core["p"] < 2.0:
                branch_of.append("HP")
            else:
                branch_of.append("MID")

    print(f"\nEvaluating {len(all_cores)} models on {args.ncpus} workers...")
    with Pool(args.ncpus) as pool:
        results = pool.map(eval_one, all_cores, chunksize=4)

    vecs = np.array([r[0] for r in results], dtype=float)
    lps = np.array([r[1] for r in results], dtype=float)
    branch_of = np.array(branch_of)

    finite = np.isfinite(lps) & (lps > -1e10)
    best = float(np.max(lps[finite])) if np.any(finite) else -np.inf
    keep = finite & (lps > best - args.delta_logL)
    print(f"best logL(XRT+SI) = {best:.2f}; {int(keep.sum())} / {len(lps)} within {args.delta_logL}")

    summary_lines = [
        f"best_logL_XRT_SI = {best:.3f}",
        f"n_within_{args.delta_logL:g} = {int(keep.sum())} / {len(lps)}",
        "method = structured grid + LHS (no MCMC)",
        "include_wing = False",
        f"flare_frozen_from = {FLARE_FREEZE_FROM}",
        "",
    ]
    branches_out = {}

    for bname in ("HP", "SC", "MID"):
        # Use eps_B / p split on kept samples (independent of generation label)
        good = vecs[keep]
        glp = lps[keep]
        if len(good) == 0:
            continue
        eB = good[:, CORE_SCAN.index("eps_B")]
        pv = good[:, CORE_SCAN.index("p")]
        if bname == "SC":
            mask = eB < 10 ** (-3.5)
        elif bname == "HP":
            mask = (eB >= 10 ** (-3.5)) & (pv < 2.0)
        else:
            mask = (eB >= 10 ** (-3.5)) & (pv >= 2.0)

        label = {
            "HP": "HP (hard p<2)",
            "SC": "SC (low eps_B, slow cooling)",
            "MID": "MID (p>2, mid eps_B)",
        }[bname]

        if mask.sum() < 3:
            msg = f"{label}: only {int(mask.sum())} samples"
            print(f"\n{msg}")
            summary_lines.append(msg)
            branches_out[bname] = {"n": int(mask.sum()), "params": {}}
            continue

        best_i = int(np.argmax(glp[mask]))
        best_core = {k: float(good[mask][best_i, j]) for j, k in enumerate(CORE_SCAN)}
        best_th = core_dict_to_free_slice(best_core, pdefs, base_theta, core_ix)
        opt_c2 = optical_chi2(best_th, pdefs, xrt_data, optical_datasets, xrt_index_data)
        late = late_xrt_resid(best_th, pdefs, xrt_data, xrt_index_data)

        print(f"\n{label}: {int(mask.sum())} samples, best {glp[mask].max():.1f}")
        print(f"  optical_chi2 (frozen A_V, no wing): {opt_c2:.1f}")
        print(f"  late XRT resid (t>1e4 s): {late}")
        summary_lines.append(
            f"{label}: n={int(mask.sum())}, best_logL={float(glp[mask].max()):.2f}, "
            f"optical_chi2={opt_c2:.1f}"
        )
        params_out = {}
        for j, name in enumerate(CORE_SCAN):
            is_log = name not in ("p", "xi")
            info = rng_str_arr(good[mask][:, j], logscale=is_log)
            print(f"  {name:<14}{info['str']}")
            summary_lines.append(f"  {name:<14}{info['str']}")
            params_out[name] = {k: info[k] for k in ("p05", "p50", "p95")}
        branches_out[bname] = {
            "n": int(mask.sum()),
            "best_logL": float(glp[mask].max()),
            "optical_chi2": float(opt_c2),
            "late_xrt_resid": late,
            "params": params_out,
            "best_core": best_core,
        }
        summary_lines.append("")

    # Per-generation success rates (diagnostic)
    summary_lines.append("generation hit rates (any finite logL):")
    for bname in ("HP", "SC", "MID"):
        m = branch_of == bname
        n_ok = int(np.sum(finite & m))
        summary_lines.append(f"  {bname}: {n_ok}/{int(m.sum())} finite")
        print(f"generation {bname}: {n_ok}/{int(m.sum())} finite")

    outdir = FIT_RESULTS_DIR / f"core_xrt_pl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "all_cores.npy", vecs)
    np.save(outdir / "all_logprob.npy", lps)
    np.save(outdir / "keep_mask.npy", keep)
    np.save(outdir / "branch_gen.npy", branch_of)
    (outdir / "core_labels.txt").write_text("\n".join(CORE_SCAN) + "\n")
    payload = {
        "best_logL": best,
        "n_within_delta": int(keep.sum()),
        "n_total": int(len(lps)),
        "delta_logL": args.delta_logL,
        "method": "grid+LHS",
        "include_wing": False,
        "flare_frozen": flare_vals,
        "branches": branches_out,
        "core_labels": CORE_SCAN,
    }
    (outdir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (outdir / "summary.txt").write_text("\n".join(summary_lines) + "\n")
    print(f"\nSaved to {outdir}")


if __name__ == "__main__":
    main()
