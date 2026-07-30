# Fragmentation audit — GRB251013C

Date: 2026-07-27
Scope: whole repository, prompted by "review the entire code base and check any
fragmented stuffs" alongside the `grb`-module refactor.
Baseline: `a40dd204`

This report separates what the refactor **fixed** from what it **found and left
alone**. Everything in the second half is a real observation with a
recommendation, not a change.

---

## 1. What the refactor resolved

### 1.1 An abandoned half-finished refactor

The single most important finding. `grb/` already contained near-verbatim copies
of code in `modeling/final_model.py`, and that copy was both **dead** (nothing
imported it) and **broken** (`grb/utils.py:seconds_from_trigger` did
`import datetime` then called `datetime.strptime`, raising `AttributeError`, so
`grb.modeling.load_all_data()` could never have run successfully).

Measured similarity of the duplicated definitions (AST-level):

| Symbol in `grb/` | Copy of | Similarity |
|---|---|---|
| `grb/modeling.py:make_wing_model` | `final_model.py:make_wing_model` | 1.000 |
| `grb/modeling.py:make_core_model` | `final_model.py:make_core_model` | 0.998 |
| `grb/modeling.py:load_all_data` | `final_model.py:load_all_optical_data` | 0.997 |
| `grb/functions.py:norris_flare` | `final_model.py:norris_flare` | 1.000 |
| `grb/utils.py:flux_error` | `modeling/utils.py:xrt_flux_error` | 0.986 |
| `grb/const.py:MODEL_RESOLUTIONS` | `final_model.py:MODEL_RESOLUTIONS` | identical |

Resolved: `grb/` is now the single home for all of it, `load_all_data` is renamed
to the canonical `load_all_optical_data`, and the three `modeling/` modules are
deleted.

### 1.2 A circular import held together by lazy imports

`final_model.py` imported `final_model_plotting` at module level, and
`final_model_plotting` imported back from `final_model` **inside three function
bodies** (`compute_model_components`, `plot_light_curves`,
`plot_spectral_index_comparison`) purely to defer the cycle. The package layout
removes the cycle; all three lazy imports are gone.

### 1.3 XRT band edges triplicated

`7.25e16` / `2.42e18` existed as named constants in `final_model.py` and as bare
inline literals twice more. All sites now reference `grb.const.XRT_NU_LO` /
`XRT_NU_HI`. Values identical, so numerics unaffected.

### 1.4 Two blocking bugs

- `grb/utils.py:seconds_from_trigger` — `import datetime` vs `from datetime import
  datetime` (see 1.1). Fixed.
- `grb/extinction.py` — checked `df["corrected"]` where the column is
  `df["gal_corrected"]`, raising `KeyError`. Fixed.

### 1.5 Dead code removed

- 44 lines of commented-out `add_observation` in `grb/modeling.py`.
- Dead `default_nwalkers` imports in `late_phase.py` and `partial_data.py`
  (imported, never called — both inline `max(int(2.5 * ndim), 32)` instead, where
  the helper returns `max(4 * ndim, 32)`). Imports dropped; the inlined
  expressions left alone.
- `grb/__pycache__/*.pyc` were tracked in git. Untracked.

---

## 2. Found and deliberately left alone

These change fit numerics or fall outside the agreed scope ("consolidate + fix,
no behaviour change"). Each needs a decision from the project owner.

### 2.1 Two disagreeing host-extinction implementations — **highest priority**

`grb/extinction.py` now holds both, side by side:

| | Curve source | Default profile |
|---|---|---|
| `host_galaxy_extinction` | interpolates `data/host_galaxy_extinction.csv` | **MW** |
| `host_extinction_attenuation` | `VegasAfterglow.extinction.BUILTIN_LAWS` | **SMC** |

The fit uses the second. They disagree. The docstring on the second states this
explicitly. **Recommendation:** decide which is authoritative and delete or
clearly quarantine the other — two extinction curves with different defaults is
a live foot-gun.

### 2.2 Two readers for `data/xrt_index.csv`

`grb/io.py:read_data("xrt_index")` names the columns `index / index_high /
index_low`. `load_xrt_spectral_index` bypasses `read_data` entirely and re-reads
the same file with pandas as `gamma / gamma_err_high / gamma_err_low`, converting
Γ → β = 1 − Γ. Only the second is used; the first appears dead.
**Recommendation:** fold the Γ→β conversion into `read_data` and drop the
duplicate reader.

### 2.3 The host `A_V` prior is duplicated and mislabelled

`grb/prior.py:host_galaxy_extinction_prior` declares `"Av": (-0.82, 0.41)` inside
its `default_param_bounds` dict. Those are not bounds — they are the log10 mean
and sigma, the same numbers as `HOST_AV_LOG10_MEAN` / `HOST_AV_LOG10_SIGMA`. The
`"log_norm"` prior type happens to read them as (mean, std), so it works by
accident while reading as bounds. **Recommendation:** rename the dict key or move
these into `grb/const.py` as the single source.

### 2.4 Two initial guesses fall outside their own declared bounds

In the driver's `INITIAL_GUESS`:

| Parameter | Guess | Declared bounds |
|---|---|---|
| `p_r` | 3.329 | (2.0, 3.0) |
| `E_iso_wing` | 3e51 | (1e52, 1e53) |

`log_prior` at the raw guess is `-inf`. The existing clip masks this, but **every
walker starts pinned exactly at the boundary** for those two parameters, which is
unlikely to be intended. Preserved verbatim and flagged in a comment — changing
it would move where the chains start. **Recommendation:** decide whether the
bounds or the guesses are wrong.

### 2.5 Residual duplication across the phase scripts

`early_phase.py`, `late_phase.py` and `partial_data.py` were explicitly out of
scope (their imports were updated; their code did not move). They still define
their own same-named functions. Similarity against the new `grb/` versions:

| Symbol | Also defined in | Similarity |
|---|---|---|
| `make_core_model` | `modeling/partial_data.py` | 0.997 |
| `log_prior` | `modeling/early_phase.py` | 0.992 |
| `make_wing_model` | `modeling/partial_data.py` | 0.973 |
| `log_probability` | `modeling/partial_data.py` | 0.954 |
| `load_best_fit_params` | `modeling/partial_data_plotting.py` | 0.949 |
| `norris_flare` | `modeling/partial_data.py` | 0.933 |
| `log_prior` | `modeling/partial_data.py` | 0.840 |
| `make_param_defs` | `modeling/partial_data.py` | 0.805 |
| `log_probability` | `modeling/early_phase.py` | 0.723 |
| `log_likelihood` | `modeling/partial_data.py` | 0.495 |
| `make_param_defs` | `modeling/early_phase.py` | 0.466 |
| `log_probability` | `modeling/late_phase.py` | 0.435 |
| *(11 further pairs below 0.30 — genuinely different phase physics)* | | |

Read this as two groups. Above ~0.90 (`partial_data`'s model builders,
`norris_flare`, `load_best_fit_params`, `early_phase`'s `log_prior`) these are
copies that drifted, and `partial_data.py` in particular could adopt `grb/`
wholesale with little risk. Below ~0.50 they are genuinely different models for
different phases, and sharing a name is the only thing they share — those should
stay separate but would benefit from distinct names.

**Recommendation:** migrate `partial_data.py` next; it is the closest to the
final model and the cheapest win. Leave `late_phase.py` alone.

### 2.6 Smaller observations

- `modeling/test/*.py` are standalone PNG-rendering physics comparisons, not
  tests. They previously put only `modeling/` on `sys.path`, which satisfied the
  old local `utils` import but would not satisfy `import grb`; they now add the
  repo root. There is still no pytest harness (out of scope by agreement).
- `requirements.txt` pins `VegasAfterglow==1.1.0`; the installed and required
  version is `2.0.6`. Stale — already noted in `CLAUDE.md` but worth fixing at
  source.
- `template.py` at the repo root defines its own `log_prior` / `log_likelihood` /
  `log_probability`. It is a scaffold, unreferenced by anything; harmless, but
  it is a fourth copy of those names.

---

## 3. Equivalence guarantee

The refactor is backed by `modeling/test/test_refactor_equivalence.py`, which
materialises the pre-refactor scripts from git at `a40dd204` and asserts
**bit-identical** results — `np.testing.assert_array_equal` and `==` throughout,
never `assert_allclose`.

Verified identical: constants, `host_extinction_attenuation`, spectral-index
loading / break frequencies / p-prior, `make_param_defs` across all four
flare/wing combinations, all loaded data (45 XRT + 25 optical datasets / 142
points), core and wing model flux, `log_probability` over 51 θ, the results
helpers, `compute_model_components` on both branches, and both rendered figures
(`bestfit_lc.png`, `spectral_index_comparison.png`) **byte-for-byte**.

Reference values pinned in the harness as a regression guard:

```
log_probability = -40459.052164414396
log_prior       = -1.787277712341638
si_chi2         =  37.10252733400446
```

MCMC chains are deliberately **not** compared: the driver seeds walkers with
unseeded `np.random`, so two runs of the *same* code already differ. This was
agreed up front.
