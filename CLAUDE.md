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
- **Run fit scripts from the `modeling/` directory.** Each script does
  `os.sys.path.append("..")` so it can `import grb`, and resolves plotting modules as
  same-dir imports. Example (full run, parallel):
  ```bash
  cd modeling
  /home/dtak/miniconda3/envs/grb251013c/bin/python final_model.py --ncpus 64
  ```

  Common flags across phase scripts: `--nsteps`, `--nwalkers`, `--ncpus`, `--outdir`,
  `--include-flare/--no-include-flare`, `--include-wing/--no-include-wing`, `--dry-run`,
  `--skip-corner`. Plot-only reruns from saved results use `--plot-from` / `--result-dir`.
- **No formal test suite.** `modeling/test/*.py` are standalone physics-comparison scripts
  that render PNGs (jet break, spreading, narrow jet, smooth spectral index); run them
  directly, there is no pytest harness.

## 6. Architecture

Two layers, kept separate:

### `grb/` — reusable data pipeline (import as a package)

- **`grb/io.py`** is the entry point: `read_data(keyword, correct_galactic_extinction=, add_converted_flux=)` dispatches by keyword (`"xrt"`, `"i_data"`, `"circular"`, `"sdt"`,
  `"xrt_index"`) to the right file in `data/` with the right column schema, sorts by time,
  and optionally applies galactic-extinction correction and mag→flux conversion. `filter_data()`
  slices a frame by filter/facility/time-range and can drop upper limits.
- **`grb/const.py`** — env-derived constants (`REDSHIFT`, `D_L`, `TRIGGER_TIME`, data paths)
  plus `FILTER_INFO` (per-band photometric system, zero point, central wavelength, width).
- **`grb/extinction.py`** — Milky Way (galactic) extinction correction using SFD dustmaps.
  Applied on data load. Distinct from *host-galaxy* extinction (see below).
- **`grb/utils.py`** — photometric conversions (`mag_to_flux_mJy`, `mJy_to_erg_cm2_s*`),
  filter→wavelength/width lookups.

### `modeling/` — phase-based afterglow fitting (standalone CLI scripts)

The fit is built up in phases, each a self-contained script with a companion `*_plotting.py`:

- `early_phase.py` — core jet + reverse shock (early emission).
- `partial_data.py` — core + Norris flare + wing on XRT + i-band.
- `late_phase.py` — wing jet for late-time emission.
- **`final_model.py`** — the combined model: core jet + reverse shock **+** Norris flare
  **+** wing jet, fit against *all* data (XRT + i-band + Leavitt Rc/Ic + per-filter 7DT/SDT).
  This is the primary, most complete fit.

**Physics model composition** (see `final_model.py`): total flux = forward-shock core jet
(`TophatJet` + `Radiation`, with an optional reverse-shock `Radiation`) + an optional
`TophatJet` "wing" for late times + an optional Norris-function temporal flare in the XRT
band (spectrally extrapolated to optical). Each component is a `VegasAfterglow.Model`
(`ISM` medium, on-axis `Observer`).

**Shared fitting utilities — `modeling/utils.py`:**

- `ParamDefWithPrior` wraps VegasAfterglow's `ParamDef` to add a Gaussian prior on top of
  the box bounds (used e.g. for host `A_V`, `HOST_AV_LOG10_MEAN/SIGMA`).
- `host_extinction_attenuation(nu, A_V, z)` applies **host-galaxy** extinction (SMC law,
  rest-frame) as a model multiplier — separate from the galactic correction in `grb/`.
- `model_array()` normalizes VegasAfterglow output objects (`.total`/`.sync`+`.ssc`) to arrays.
- `top_k_samples()`, `plot_corner()`, `latest_result_dir()` for post-processing.
- Granot & Sari 2001 tooling (`load_xrt_spectral_index`, `compute_p_prior_from_spectral_index`,
  `compute_break_frequencies`) to derive a prior on the electron index `p` from the XRT photon index.

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
- Time is seconds from `TRIGGER_TIME`; `date_obs` strings are converted with the trigger.
