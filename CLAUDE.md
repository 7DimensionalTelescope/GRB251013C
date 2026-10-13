# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```

1. [Step] → verify: [check]

2. [Step] → verify: [check]

3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

This is a research project fitting the multi-wavelength afterglow of GRB 251013C (XRT + optical) with structured-jet synchrotron models via MCMC.

## 5. Environment & running

- **Interpreter:** the `grb251013c` conda env. Use its Python explicitly:
  `/home/dtak/miniconda3/envs/grb251013c/bin/python`. It has `VegasAfterglow` 2.0.6
  installed (note: `requirements.txt` pins `1.1.0`, which is stale — trust the installed version).
- **Config comes from a `.env` in the *parent* directory** (`/home/dtak/research/grb/.env`),
  not this folder. `grb/const.py` calls `load_dotenv()`, which walks up the tree to find it,
  then reads `ra`, `dec`, `redshift`, `trigger_time`, `AV`. `D_L` (luminosity distance) is
  derived from `redshift` via `astropy` `Planck18`. Nothing runs without these vars.
- **The final model runs from the repo root.** `fit_final_model.py` is a thin driver over
  the `grb` package. Example (full run, parallel):
  ```bash
  /home/dtak/miniconda3/envs/grb251013c/bin/python fit_final_model.py --ncpus 64
  ```

  Its flags: `--nsteps`, `--nwalkers`, `--ncpus`, `--outdir`,
  `--include-flare/--no-include-flare`, `--include-wing/--no-include-wing`,
  `--use-spectral-index/--no-use-spectral-index`. Note `--nwalkers` must be at least
  `2 * ndim` (52 for the 26-parameter model) or emcee rejects the initial state.
- **Parameter-space exploration lives in the `grid_search/` directory** (like `modeling/`
  below): `mixing_search.py` and `sweep_search.py` share helpers via `grid_search_common.py`
  in the same folder, and each does `os.sys.path.append(os.path.join(os.path.dirname(__file__),
  ".."))` so it can `import grb` (same trick `modeling/late_phase.py` uses). Run them from
  within that directory:
  ```bash
  cd grid_search
  /home/dtak/miniconda3/envs/grb251013c/bin/python mixing_search.py --ncpus 20
  /home/dtak/miniconda3/envs/grb251013c/bin/python sweep_search.py --ncpus 20
  ```
- **Parameter-space exploration before MCMC ("mixing mode"):** `mixing_search.py` draws a
  scrambled Sobol sequence over the existing prior box, scores every sample with the existing
  `log_prior + log_likelihood` (one model evaluation per sample), and writes ranked results
  plus a small plotted subset to
  `/home/hongyp007/hongyp/projects/GRB/results/mixing_<timestamp>/`
  (`results/samples.npz`, `results.csv`, `ranked_results.csv`, `plots/{best,broad,summary}/`,
  `run_config.json`). Flags: `--nsamples --seed --ncpus --n-best-plots --n-broad-plots --outdir`
  plus the same model flags as `fit_final_model.py`, and `--environment ism|wind|stratified`
  (one medium per run; the run directory is `mixing_<environment>_<timestamp>`; media are
  defined by `grb.params.medium_param_defs`). Unlike the MCMC, it rejects samples that
  violate three coupled validity rules (approved 2026-08-23): `eps_e + eps_B <= 1` per shock
  region, `theta_c_wing > theta_c_core`, `tau_decay_flare > tau_rise_flare`. Rejection happens
  at generation time and rejected draws are replaced (and still recorded), so `--nsamples`
  counts physically valid evaluated samples. Not a sampler and not an optimizer — the ranking
  only locates starting regions for MCMC.
- **Parameter sweeps (how the model changes as each parameter changes):** `sweep_search.py
  [PARAM ...]` (no names = all parameters; `--sweep-nvalues`, default 15) holds every
  parameter at a baseline (filtered to the current param defs, out-of-bounds values clipped
  and missing ones set to the bounds midpoint — both recorded as notes) while one parameter
  is varied across its full prior range. Baseline source is `INITIAL_GUESS` from
  `fit_final_model.py` by default, or `--baseline-dir DIR` to use the best-fit sample from a
  saved run directory instead (any dir with `samples.npy`/`log_probs.npy`/`labels.txt`, e.g. a
  `fit_final_model.py` output dir under `projects/GRB/results/GRB251013C/` — loaded via
  `grb.results.load_best_fit_params`, the same loader `grb/plotting.py` uses; full-precision,
  unlike the 6-sig-fig `bestfit_params.txt`).
  Writes `sweep_<environment>_<timestamp>/` with one
  figure per parameter (`plots/sweep_<param>.png`: XRT + i-band light-curve families, plus a
  7DT-spectrum-vs-wavelength panel at the median 7DT observation time, all three colored by
  parameter value with the baseline in black; the spectrum panel is blank if there is no 7DT
  data). Every swept value AND the baseline are further broken into forward-shock (dotted) and
  reverse-shock (dash-dot) components in all three panels — same line-style convention as
  `grb.plotting.render_light_curve_figure` (`'k:'` core FS, `'c-.'` reverse shock), but colored
  by parameter value like the total curves so the FS/RS trend is visible too, not just the
  total. Row-aligned `results/sweep_results.csv` + `results/sweep_curves.npz` (`{xrt,opt,sed}_
  {total,fs,rs}` and `baseline_{xrt,opt,sed}_{,fs,rs}` arrays, plus `wavelength_grid` when 7DT
  data exists), and `run_config.json`.
  `log_likelihood`/`delta_logL` per point are still recorded in the CSV/npz (and the baseline
  logL is shown in each figure's title) even though the ΔlogL panel itself was replaced by the
  spectrum panel. Sweep points violating the coupled validity rules are marked
  `physical_invalid` and skipped in the curve panels. Evaluation runs on a
  `multiprocessing.Pool` (`--ncpus`).
- **One hand-picked parameter set (not a scan):** `manual_model.py [--PARAM VALUE ...]`
  evaluates a single model at manually-specified parameter values and plots it against all
  data with the same 3-panel/component-breakdown style as `plot_light_curves`. Same baseline
  convention as the sweep (`INITIAL_GUESS`, missing keys to the bounds midpoint), but CLI
  overrides are **not** clipped to the prior box (only warned about) since the point is
  unconstrained manual exploration; also warns (non-blocking) on the three coupled validity
  rules. `--environment ism|wind|stratified`, `--include-flare/--include-wing` pick which
  `--PARAM` flags exist (one per name in `grb.params.make_param_defs`). Writes
  `manual_<environment>_<timestamp>/manual_lc.png` + `params_used.json` (values, sources,
  validity warnings) under the same results root as the sweep, or `--outdir`. For interactive
  work, `manual_model.ipynb` (same directory) imports `build_baseline`/`check_physical_validity`
  from the script and exposes an `overrides = {...}` dict cell to edit values by hand and
  re-render inline — run its kernel from this directory so `import grb`/`manual_model` resolve.
- **Run the *other* phase scripts from the `modeling/` directory.** Each does
  `os.sys.path.append("..")` so it can `import grb`, and resolves plotting modules as
  same-dir imports:
  ```bash
  cd modeling
  /home/dtak/miniconda3/envs/grb251013c/bin/python late_phase.py --ncpus 64
  ```

  Common flags across those: `--nsteps`, `--nwalkers`, `--ncpus`, `--outdir`, `--dry-run`,
  `--skip-corner`. Plot-only reruns from saved results use `--plot-from` / `--result-dir`.
- **No formal test suite.** `modeling/test/*.py` are standalone physics-comparison scripts
  that render PNGs (jet break, spreading, narrow jet, smooth spectral index); run them
  directly, there is no pytest harness. The one exception is
  `modeling/test/test_refactor_equivalence.py`, which asserts the `grb` package still
  reproduces the pre-refactor scripts bit-for-bit — run it after touching anything under
  `grb/` or `fit_final_model.py`:
  ```bash
  /home/dtak/miniconda3/envs/grb251013c/bin/python modeling/test/test_refactor_equivalence.py
  ```

## 6. Architecture

Two layers, kept separate:

### `grb/` — reusable data pipeline (import as a package)

- **`grb/io.py`** is the entry point: `read_data(keyword, correct_galactic_extinction=, add_converted_flux=)` dispatches by keyword (`"xrt"`, `"i_data"`, `"circular"`, `"sdt"`,
  `"xrt_index"`) to the right file in `data/` with the right column schema, sorts by time,
  and optionally applies galactic-extinction correction and mag→flux conversion. `filter_data()`
  slices a frame by filter/facility/time-range and can drop upper limits.
- **`grb/const.py`** — env-derived constants (`REDSHIFT`, `D_L`, `TRIGGER_TIME`, data paths)
  plus `FILTER_INFO` (per-band photometric system, zero point, central wavelength, width),
  and the fit constants: `XRT_BAND`, `XRT_NU_LO`/`XRT_NU_HI`, `MODEL_RESOLUTIONS`,
  `HOST_AV_LOG10_MEAN`/`SIGMA`, `SI_FLARE_FRAC_MAX`, `FIT_RESULTS_DIR`.
- **`grb/extinction.py`** — Milky Way (galactic) extinction correction using SFD dustmaps,
  applied on data load. Also holds `host_extinction_attenuation(nu, A_V, z)`, the
  **host-galaxy** SMC-law rest-frame multiplier the fit actually uses. The two are distinct;
  see the caveat in Conventions below.
- **`grb/utils.py`** — photometric conversions (`mag_to_flux_mJy`, `mJy_to_erg_cm2_s*`),
  filter→wavelength/width lookups, `flux_error()`, `seconds_from_trigger()`, and
  `model_array()` (normalizes VegasAfterglow output objects, `.total`/`.sync`+`.ssc`, to arrays).
- **`grb/modeling.py`** — `load_all_optical_data()` (XRT + every optical dataset the fit uses)
  and the model builders `make_core_model()` / `make_wing_model()`.
- **`grb/params.py`** — `ParamDefWithPrior` (wraps VegasAfterglow's `ParamDef` to add a
  Gaussian prior on top of the box bounds, used for host `A_V`), `make_param_defs()` (takes
  `environment="ism"|"wind"|"stratified"`, default `"ism"` = the historical set), and
  `medium_param_defs()` (per-environment density parameters: `n_ism` / `A_star` /
  `A_star`+`n_ism`+`n0` with `k_m` fixed at 2).
- **`grb/likelihood.py`** — `log_probability` = `log_prior` + `log_likelihood`, plus
  `compute_model_flux_all_bands()` and the XRT spectral-index terms.
- **`grb/spectral_index.py`** — Granot & Sari 2001 tooling (`load_xrt_spectral_index`,
  `compute_p_prior_from_spectral_index`, `compute_break_frequencies`) to derive a prior on
  the electron index `p` from the XRT photon index.
- **`grb/results.py`** — run-artifact I/O: `save_run_arrays`, `save_bestfit_params`,
  `top_k_samples`, `latest_result_dir`, `load_best_fit_params`, `read_labels`.
- **`grb/plotting.py`** — `plot_light_curves` (loads a saved MCMC run dir, then delegates to
  `render_light_curve_figure`), `render_light_curve_figure(params, include_flare, include_wing,
  outfile, ...)` (the actual 3-panel/component-breakdown renderer, params-first so callers
  without a saved run dir — e.g. `manual_model.py` — can drive it directly),
  `plot_spectral_index_comparison`, `plot_corner`, `compute_model_components`.

### `fit_final_model.py` — the primary fit (repo root)

The combined model: core jet + reverse shock **+** Norris flare **+** wing jet, fit against
*all* data (XRT + i-band + Leavitt Rc/Ic + per-filter 7DT/SDT). A thin driver — argument
parsing, walker seeding, the emcee loop and the run directory. All physics lives in `grb/`.

**Physics model composition:** total flux = forward-shock core jet (`TophatJet` +
`Radiation`, with an optional reverse-shock `Radiation`) + an optional `TophatJet` "wing"
for late times + an optional Norris-function temporal flare in the XRT band (spectrally
extrapolated to optical). Each component is a `VegasAfterglow.Model` (`ISM` medium, on-axis
`Observer`).

### `modeling/` — earlier phase fits (standalone CLI scripts)

The fit was built up in phases, each a self-contained script with a companion `*_plotting.py`.
These import from `grb/` but still define their own model builders and likelihoods:

- `early_phase.py` — core jet + reverse shock (early emission). Split 2026-10-08: `early_phase.py` is CLI/dispatch
  only; `early_phase_model.py` is the single model / prior / physical-constraint / likelihood implementation
  (`--prior final|wide|production`, `--xi both|reverse|none`); `early_phase_optimize.py` (robust optimisers),
  `early_phase_diagnostics.py` (tau/p/xi/E_iso profiles), `early_phase_mcmc.py` (legacy sampler + production
  sampler: DE moves, ridge-aware start, HDF `chain.h5` with `--resume DIR`, `--check-chain DIR`).
  `run_early_phase_mcmc_overnight.sh OUTDIR` drives 30k→60k→≤100k steps autonomously. After touching the model run
  `modeling/test/test_early_phase_refactor_equivalence.py` (bit-identity vs the frozen pre-split file) and
  `modeling/test/verify_production_config.py`. On this machine the working interpreter is
  `/home/hongyp007/anaconda3/envs/yp/bin/python` (the `grb251013c` path above does not exist here).
- `partial_data.py` — core + Norris flare + wing on XRT + i-band.
- `late_phase.py` — wing jet for late-time emission.

**MCMC pattern** (consistent across scripts): `log_probability = log_prior + log_likelihood`,
where the likelihood is a χ² over XRT plus each optical dataset separately. LOG-scale
parameters are sampled in log10 space (labels prefixed `log10_`). Parallelized with
`emcee` + `multiprocessing.Pool`. **Non-obvious constraint:** emcee evaluates walkers in two
half-batches, so useful parallelism is capped at `nwalkers/2` — to use N cores you need
~`2N` walkers. Keep BLAS single-threaded per worker (`OMP_NUM_THREADS=1` etc., set before
`import numpy`) so pool workers don't oversubscribe. One model evaluation is ~100 ms, so full
fits are hours-long; run with a large `--ncpus`.

**Results layout:** each run writes `modeling/fit_results/<phase>_<YYYYMMDD_HHMMSS>/`
containing `samples.npy`, `log_probs.npy`, `labels.txt`, `top_k_params.npy`,
`bestfit_params.txt`, and PNG light-curve/corner plots.

## 7. Conventions

- Flux densities are in **mJy**; internal VegasAfterglow flux is converted (`* 1e26`).
- Two independent extinction stages: galactic (Milky Way, applied at data load in `grb/`) and
  host-galaxy (SMC, applied inside the model via the fitted `A_V`). Don't conflate them.
  **Caveat:** `grb/extinction.py` holds *two* host implementations that disagree —
  `host_galaxy_extinction` (interpolates `data/host_galaxy_extinction.csv`, defaults to MW)
  and `host_extinction_attenuation` (`VegasAfterglow` `BUILTIN_LAWS`, defaults to SMC).
  The fit uses the second. See the fragmentation audit below.
- Time is seconds from `TRIGGER_TIME`; `date_obs` strings are converted with the trigger.
- Known issues that alter fit numerics are catalogued, not fixed, in
  `docs/superpowers/reports/2026-07-27-fragmentation-audit.md` — read it before changing
  extinction, the `A_V` prior, or `INITIAL_GUESS`.
