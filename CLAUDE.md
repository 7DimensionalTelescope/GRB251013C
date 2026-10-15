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
  `2 * ndim` or emcee rejects the initial state.
  **Added 2026-10-13:** `--second-component` (slow second jet component with opened wing
  bounds, meant for the ~2 ks optical hump), `--flare-xray-only` (Norris component acts in
  the XRT band only, no optical extrapolation), `--medium ism|wind` (wind replaces `n_ism`
  by `A_star`), `--xray-extra` (EP-FXT, Chandra, XRT 3σ upper limit from
  `data/xray_extra.csv`), `--late-optical` (`data/late_optical.csv`, t ≤ `LATE_OPTICAL_TMAX`),
  `--preoptimize N` (LHS + Powell walker seeding for models with a rough `INITIAL_GUESS`).
  The run directory records its options in `fit_config.json`, which the plotter reads back.
  The best configuration so far is the hybrid
  `--include-flare --flare-xray-only --second-component --medium wind --xray-extra --late-optical --preoptimize 4000`
  (run `final_xflare_second_wind_20261013_195546`, total χ² 2072 vs 3808 for the old flare
  model on the same data; see `score_runs.py` for the apples-to-apples scorer).
- **`score_runs.py`** re-scores any saved runs on the identical extended dataset (per-dataset
  χ²); use it instead of comparing `Best log probability` across runs fitted to different data.
- **Plot-only drivers at the repo root:** `plot_final_model.py [result_dir]` regenerates
  the light-curve/spectral-index figures for a saved `final_*` run (defaults to the latest).
  `plot_mwl_lc.py` draws the paper-style multi-wavelength light curve with an XRT
  photon-index panel: `--fit-only` shows exactly the datasets `load_all_optical_data()`
  feeds the fit (with per-dataset scale offsets and fitted power laws), without it every
  GCN photometry point + Chandra + radio (colour = band, marker = telescope).
  `lc_indices.py` fits segment power laws (alpha) and the mean XRT photon index and renders
  a check figure; `closure_test.py` compares those alpha/beta with the standard closure
  relations; `plot_cr.py` draws the alpha-beta plane using `cr_plot()` from the user's
  `~/script/grbpy` package (outside this repo). All write to `figures/`.
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
  reproduces the pre-refactor scripts bit-for-bit (it materialises the baseline from git
  commit `a40dd20`). **It has been failing since the bounds retune (`theta_c_core` upper
  bound 0.08 → 1.0) and, since the 2026-10-13 data corrections, the data-loading comparisons
  differ by design** — treat it as a historical harness, not a gate:
  ```bash
  /home/dtak/miniconda3/envs/grb251013c/bin/python modeling/test/test_refactor_equivalence.py
  ```

## 6. Architecture

Two layers, kept separate:

### `grb/` — reusable data pipeline (import as a package)

- **`update_time.py`** (repo root) regenerates `data/circular.xlsx` from the hand-maintained
  `data/raw_data.xlsx` (GCN photometry): parses the mixed date formats, normalises facility
  names, splits upper limits, and converts HJD-quoted facilities (`HJD_FACILITIES`) to UTC.
  Re-run it after editing `raw_data.xlsx`; never edit `circular.xlsx` by hand.
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
- **`grb/prior.py`** — `ParamDefWithPrior` (wraps VegasAfterglow's `ParamDef` to add a
  Gaussian prior on top of the box bounds, used for host `A_V`) plus default bounds/priors
  for the empirical curve-fit models. **`grb/params.py`** imports it and defines
  `make_param_defs()` — the MCMC parameter box lives there.
- **`grb/vegas_compat.py`** — import-for-side-effect patch that makes VegasAfterglow 2.0.6's
  `AfterglowLikelihood` thread-safe for bilby samplers with `npool > 1`. Only the
  notebooks' bilby path needs it; the emcee drivers don't.
- **Notebook-side empirical fitting** (not used by the MCMC drivers): `grb/fit.py`
  (`Fitter` base with scipy/bilby backends) → `grb/analysis.py` (`Analyzer`, masks data and
  fits light curves or SEDs), with shapes in `grb/functions.py` (power laws, broken power
  laws, `norris_flare` — the latter *is* shared with the final model's flare), and plotting
  helpers in `grb/lightcurve.py` / `grb/sed.py`. Consumers are `run.ipynb` and
  `spectrum.ipynb`.
- **`grb/likelihood.py`** — `log_probability` = `log_prior` + `log_likelihood`, plus
  `compute_model_flux_all_bands()` and the XRT spectral-index terms.
- **`grb/spectral_index.py`** — Granot & Sari 2001 tooling (`load_xrt_spectral_index`,
  `compute_p_prior_from_spectral_index`, `compute_break_frequencies`) to derive a prior on
  the electron index `p` from the XRT photon index.
- **`grb/results.py`** — run-artifact I/O: `save_run_arrays`, `save_bestfit_params`,
  `top_k_samples`, `latest_result_dir`, `load_best_fit_params`, `read_labels`.
- **`grb/plotting.py`** — `plot_light_curves`, `plot_spectral_index_comparison`,
  `plot_corner`, `compute_model_components`.

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

- `early_phase.py` — core jet + reverse shock (early emission).
- `partial_data.py` — core + Norris flare + wing on XRT + i-band.
- `late_phase.py` — wing jet for late-time emission.
- `map_core_xrt_pl.py` — no-MCMC Latin-hypercube/branch grid of the core forward-shock
  parameters against XRT flux + spectral index only (seeds in `corexrt_seeds.json`).

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
  extinction, the `A_V` prior, or `INITIAL_GUESS`. The refactor design/plan behind the
  `grb/` package is in `docs/superpowers/specs/` and `docs/superpowers/plans/`.
- **`analysis/2026-08_core_flare_team/`** is the archived core+flare (no-wing) team
  exploration — start at its `README.md`, then `reports/team_synthesis.md` (Section 4 is the
  next-generation fit spec). It also holds fit-ready external data (Chandra, radio, late GCN
  optical). Its `reports/leavitt_zp_fix.patch` (Vega/AB zero-point fix) **was applied on
  2026-10-13** together with a `MAG_ERROR_FLOOR = 0.05` mag calibration floor
  (`grb/const.py`, applied in `mag_to_flux_mJy`) and a Leavitt HJD→UTC time correction
  (`HJD_FACILITIES` in `update_time.py`, −413 s; GCN 42333 quotes HJD). Fit results saved
  before that date used the old, brighter Leavitt fluxes. The archive's scripts point at the
  `.claude/worktrees/retune-on-refactor` checkout for `grb`.
