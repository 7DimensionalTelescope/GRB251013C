# Refactoring the final model into the `grb` package

**Date:** 2026-07-27
**Baseline commit:** `a40dd204e0b2bc7e5286387548fe14f75328e9f8`
**Branch:** `worktree-grb-module-refactor`

## Problem

`modeling/final_model.py` is the repository's primary scientific artifact — the
combined core-jet + reverse-shock + Norris-flare + wing-jet fit against all XRT and
optical data. It is a 659-line standalone script. Together with its companion
`modeling/final_model_plotting.py` (505 lines) and the shared `modeling/utils.py`
(284 lines), it holds physics, data loading, likelihood, sampling, result I/O, and
plotting in three files that only run as scripts.

None of it is reusable. A notebook cannot import `make_core_model` without also
importing an argparse CLI and an emcee driver.

Worse, a previous attempt to move this code into `grb/` was left half-finished.
`grb/` already contains near-verbatim copies of several of these functions, and that
copy is both **dead** and **broken**:

| Symbol in `grb/` | Copy of | Similarity |
|---|---|---|
| `grb/modeling.py:make_wing_model` | `final_model.py:make_wing_model` | 1.000 |
| `grb/modeling.py:make_core_model` | `final_model.py:make_core_model` | 0.998 |
| `grb/modeling.py:load_all_data` | `final_model.py:load_all_optical_data` | 0.997 |
| `grb/functions.py:norris_flare` | `final_model.py:norris_flare` | 1.000 |
| `grb/utils.py:flux_error` | `modeling/utils.py:xrt_flux_error` | 0.986 |
| `grb/const.py:MODEL_RESOLUTIONS` | `final_model.py:MODEL_RESOLUTIONS` | identical |

Nothing imports those `grb/modeling.py` functions. The only reference anywhere is
`modeling.ipynb` importing `add_observation`, which is commented out in the source,
so that notebook cell fails too. And `grb/utils.seconds_from_trigger` raises
`AttributeError: module 'datetime' has no attribute 'strptime'` — it does
`import datetime` where it needs `from datetime import datetime` — which means
`grb.modeling.load_all_data()` has never successfully run.

## Goals

1. Move `final_model.py`, `final_model_plotting.py`, and `modeling/utils.py` into
   `grb/` at locations their names predict.
2. Replace the monolith with a thin driver script producing **bitwise-identical**
   results.
3. Remove the duplication that this refactor makes redundant, and fix the bugs that
   block it.

## Non-goals

- Changing any fit numerics. Every number the fit produces must be unchanged.
- Reconciling the two host-extinction implementations (see Deferred).
- Moving `modeling/spectral_index_interpolator.py` or the other phase scripts
  (`early_phase`, `partial_data`, `late_phase`). Their imports get updated; their
  code does not move.
- Adding a pytest harness. The repo has none, and the equivalence check follows the
  existing standalone-script convention.

## Architecture

`grb/` is a flat package of eleven single-purpose modules. The refactor keeps that
idiom: absorb each symbol into the module its name already implies, and add new
flat modules only for concerns with no existing home.

### Absorbed into existing modules

| Symbol | Destination | Note |
|---|---|---|
| `norris_flare` | `grb/functions.py` | already present and identical — delete the copy |
| `make_core_model`, `make_wing_model` | `grb/modeling.py` | already present — de-duplicate |
| `load_all_optical_data` | `grb/modeling.py` | present as `load_all_data`; rename to the canonical name |
| `xrt_flux_error` | `grb/utils.py` as `flux_error` | already present — drop the alias |
| `model_array` | `grb/utils.py` | new |
| `host_extinction_attenuation` | `grb/extinction.py` | new; sits beside the tabulated implementation |
| `XRT_BAND`, `XRT_NU_LO`, `XRT_NU_HI`, `SI_FLARE_FRAC_MAX`, `HOST_AV_LOG10_MEAN`, `HOST_AV_LOG10_SIGMA`, `FIT_RESULTS_DIR` | `grb/const.py` | `MODEL_RESOLUTIONS` already present |

### New modules

- **`grb/params.py`** — `ParamDefWithPrior`, `make_param_defs`, `default_nwalkers`.
  What the sampler samples.
- **`grb/likelihood.py`** — `compute_model_flux_all_bands`, `spectral_index_model`,
  `spectral_index_chi2`, `log_likelihood`, `log_prior`, `log_probability`.
  How well a parameter vector fits.
- **`grb/spectral_index.py`** — `load_xrt_spectral_index`,
  `compute_break_frequencies`, `compute_p_prior_from_spectral_index`.
  Granot & Sari 2001 tooling for the XRT photon index.
- **`grb/results.py`** — `top_k_samples`, `read_labels`, `latest_result_dir`,
  `load_best_fit_params`, and the save logic currently inlined in `main()`.
  Run artifacts on disk.
- **`grb/plotting.py`** — `plot_corner`, `set_log_y_limits`,
  `compute_model_components`, `plot_light_curves`,
  `plot_spectral_index_comparison`.

### Driver

`fit_final_model.py` at the repository root: argparse, data loading, walker
initialisation, the emcee loop, and calls into `grb`. Target ~120 lines against the
current 659.

`FIT_RESULTS_DIR` resolves to `<repo>/modeling/fit_results/` as it does today, so
output lands in the same place and existing result directories stay discoverable.

### Import direction

`grb/` must not import from `modeling/`. The dependency runs one way, as it does
today (the knowledge graph confirms zero reverse edges). `grb/likelihood.py` imports
from `grb/modeling.py`, `grb/params.py`, `grb/spectral_index.py`, `grb/functions.py`,
`grb/extinction.py`, `grb/utils.py`, and `grb/const.py`. Nothing in `grb/` imports
`grb/plotting.py`.

The current circular pair — `final_model.py` imports `final_model_plotting`, which
lazily imports `final_model` inside its functions — disappears. `grb/plotting.py`
imports from `grb/likelihood.py` and `grb/modeling.py` at module top; nothing
imports back.

## Callers to migrate

`modeling/utils.py` is imported by nine files. All move to the new `grb.*` paths in
the same change:

- `modeling/early_phase.py`, `modeling/early_phase_plotting.py`
- `modeling/partial_data.py`, `modeling/partial_data_plotting.py`
- `modeling/late_phase.py`, `modeling/late_phase_plotting.py`
- `modeling/test/test_jet_break_spreading.py`, `modeling/test/test_spreading.py`,
  `modeling/test/test_narrow_jet_final.py`

`modeling/utils.py`, `modeling/final_model.py`, and `modeling/final_model_plotting.py`
are deleted. No compatibility shim — a shim would preserve exactly the duplication
this work removes.

Each migrated file must still import cleanly. That is verified mechanically
(`python -c "import <module>"` for each, with the repo root on `sys.path`), not by
inspection.

## Equivalence verification

**Definition.** For any parameter vector θ, the refactored code returns a bitwise
identical `log_probability`, from bitwise identical input data.

Bit-identical MCMC chains are not achievable and are not the target: `final_model.py`
seeds `pos0` with unseeded `np.random`, so two runs of the *current* script already
differ. Adding a seed was considered and rejected as an unrequested behaviour change.

**Mechanism.** `modeling/test/test_refactor_equivalence.py`, a standalone script
matching the existing `modeling/test/` convention. It materialises the three
baseline modules out of git at commit `a40dd204` into a temporary directory, imports
them alongside the refactored `grb` package, and asserts equality.

Reading the baseline from git rather than committing a frozen copy avoids adding
1,448 lines of duplicated source to the tree.

**Assertions**, all via `np.testing.assert_array_equal` (exact, not `allclose`):

1. `xrt_data['time' | 'flux' | 'flux_error']` identical.
2. All 25 optical datasets identical — `name`, `frequency`, `time`, `flux_mJy`,
   `flux_err`.
3. `load_xrt_spectral_index()` identical across all six returned arrays.
4. `make_param_defs()` yields the same names, bounds, scales, and Gaussian priors in
   the same order.
5. For θ at the clipped initial guess plus 50 seeded random in-bounds draws:
   `log_prior`, `log_likelihood`, and `log_probability` identical.
6. For the same θ set: `compute_model_flux_all_bands` returns an identical XRT array,
   25 identical optical arrays, and an identical `si_chi2`.

**Reference values** measured against the baseline at the clipped initial guess:

```
log_probability = -40459.052164414396
log_prior       = -1.787277712341638
log_likelihood  = -40457.264886702054
si_chi2         = 37.10252733400446
xrt_flux[:3]    = [1.14919904e-10, 1.06607771e-10, 9.66980288e-11]

45 XRT points | 25 optical datasets / 142 points | 45 spectral-index points | ndim 26
```

One model evaluation costs ~1.09 s with the spectral-index term enabled, so 51 θ
values across two implementations runs in roughly two minutes.

**Plotting** is verified separately by regenerating figures from an existing result
directory under both implementations and comparing PNG bytes. Matplotlib output is
reproducible for identical input given a fixed backend (`Agg`) and no timestamps in
the figures. If byte comparison proves flaky, the fallback is comparing the plotted
arrays — `compute_model_components` output — which is the part that carries meaning.

## Bugs fixed

Both block the refactor and neither changes fit numerics:

1. **`grb/utils.seconds_from_trigger`** — `import datetime` → `from datetime import
   datetime`. Currently raises `AttributeError`. Nothing depends on the broken
   behaviour because nothing successfully calls it.
2. **`grb/extinction.py:19`** — `if "gal_corrected" in df.columns and
   df["corrected"].all()` reads a `"corrected"` column that never exists, so the
   already-corrected branch would `KeyError`. Corrected to `df["gal_corrected"]`.
   Unreachable in current usage; fixed while the file is open.

## Deferred — reported, not changed

These are real and worth deciding on, but changing any of them alters fit numerics
and so breaks the equivalence guarantee by construction.

1. **Two disagreeing host-extinction implementations.**
   `grb/extinction.py:host_galaxy_extinction` interpolates the tabulated
   `data/host_galaxy_extinction.csv` curve and defaults to **MW**.
   `modeling/utils.py:host_extinction_attenuation` uses
   `VegasAfterglow.extinction.BUILTIN_LAWS` and defaults to **SMC**. The fit uses the
   second. Both survive the refactor, side by side in `grb/extinction.py`, with
   docstrings stating which is which and which the fit uses.

2. **Two readers for `data/xrt_index.csv`.** `grb/io.py:read_data("xrt_index")` names
   the columns `index / index_high / index_low`; `load_xrt_spectral_index` bypasses
   `read_data` entirely and re-reads the file with pandas as
   `gamma / gamma_err_high / gamma_err_low`, converting Γ → β = 1 − Γ. Only the
   second is used. The first appears to be dead.

3. **The host `A_V` prior is duplicated and mislabelled.**
   `grb/prior.py:host_galaxy_extinction_prior` declares `"Av": (-0.82, 0.41)` in its
   `default_param_bounds` dict, but those are the log10 mean and sigma — the same
   numbers as `HOST_AV_LOG10_MEAN` and `HOST_AV_LOG10_SIGMA`. The `"log_norm"` prior
   type reads them as (mean, std), so it works by accident while reading as bounds.

4. **Two initial guesses fall outside their own declared bounds.** In
   `final_model.py:main()`, `p_r = 3.329` against bounds `(2.0, 3.0)`, and
   `E_iso_wing = 3e51` against bounds `(1e52, 1e53)`. `log_prior` at the raw guess is
   `-inf`. The clipping at lines 573–579 masks this, but every walker starts pinned
   exactly at the boundary for those two parameters, which is unlikely to be
   intended. Preserved verbatim — changing it would change the fit.

5. **Dead imports of `default_nwalkers`.** `late_phase.py` and `partial_data.py` both
   import it and never call it, inlining `max(int(2.5 * ndim), 32)` instead, where
   the helper returns `max(4 * ndim, 32)`. The imports are dropped as part of caller
   migration; the inlined expressions are left alone.

6. **XRT band edges triplicated.** `7.25e16` and `2.42e18` appear as named constants
   in `final_model.py` and as bare inline literals twice more — in
   `compute_model_flux_all_bands` and in `final_model_plotting.py`. All three sites
   come to reference `grb.const`. This is a pure de-duplication: the values are
   identical, so numerics are unaffected.

## Risks

- **Silent numeric drift.** Mitigated by exact-equality assertions, not tolerances.
- **A migrated phase script breaks.** Mitigated by mechanically importing all nine.
  These scripts take hours to run, so import-level verification is the practical
  bound; the equivalence guarantee covers `final_model` only.
- **Notebooks go stale.** `run.ipynb`, `spectrum.ipynb`, and `modeling.ipynb` import
  `grb` modules. They are not updated by this work. `modeling.ipynb` is already
  broken (`add_observation`). Called out, not fixed.

## Success criteria

1. `modeling/test/test_refactor_equivalence.py` passes every assertion in
   Equivalence verification.
2. All nine migrated callers import cleanly.
3. For identical CLI arguments, the driver's setup phase derives the same `labels`
   list, the same `ndim`, and the same `nwalkers` as the baseline. Asserted in the
   equivalence test, not by eyeballing stdout. (`final_model.py` has no `--dry-run`
   flag despite CLAUDE.md listing one as common across phase scripts; none is added
   here.)
4. `modeling/final_model.py`, `modeling/final_model_plotting.py`, and
   `modeling/utils.py` no longer exist.
5. No symbol from the duplication table is defined in more than one place.
6. `grb/` contains no import from `modeling/`.
