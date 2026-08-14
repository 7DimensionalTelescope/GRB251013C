## Summary

Two logical changes:

1. **Refactor** (`d636023`…`2dcb601`): `modeling/final_model.py` is decomposed into the `grb` package (`grb/params.py`, `grb/likelihood.py`, `grb/modeling.py`, `grb/results.py`, `grb/plotting.py`, `grb/spectral_index.py`) with a thin `fit_final_model.py` driver at the repo root. Verified numerically identical to the standalone script (equivalence harness + end-to-end MCMC chain comparison in `docs/`). The branch is self-contained: `grb/io.py` with the `i_data` loader and `data/i_data.csv` are committed.

2. **Retune** (`76aeca4`, supersedes #3): bounds and initial guess updated from a verified 28-start joint re-optimization — **logL −577.6 → −548.0** (total χ² 1154.9 → 1095.8 over 232 points).
   - Required widenings (optimum sat outside the old box): `tau_rise_flare` 30→10 s, `p_wing` 2.9→3.3
   - Wall-release widenings: `theta_c_core` ≤0.08, `n_ism` ≤400, `eps_B` ≥0.002, `E_iso_wing` ≥1e51, `theta_c_wing` ≤0.7, `eps_e_wing` ≥0.1
   - Deliberately NOT widened: `p`, low `eps_B` — the low-`eps_B`/high-`p` branch that would match the observed XRT photon index (~1.88 vs model floor ~2.04) was tested and loses ΔlogL ≤ −620; that spectral tension is a model limitation, not a bounds artifact
   - `INITIAL_GUESS` replaced with the re-optimized vector; fixes `p_r`=3.329 and `E_iso_wing`=3e51 which sat **outside their own bounds** and silently clipped every walker onto the walls
   - Walker scatter 0.3→0.1 dex: the center is now a converged optimum

## Verification

- `INITIAL_GUESS` scores `log_probability = −550.28` **identically** through the refactored `grb.*` path and the old standalone script — one number that end-to-end checks both the refactor equivalence and the retune port.
- 0 initial values outside bounds (was 2).
- End-to-end smoke (`fit_final_model.py --nsteps 20 --ncpus 12`): best −560.9, light-curve/spectral-index/corner plots all produced; light curve tracks XRT (incl. the two late points), all optical bands, and the 7DT spectrum.

## After merging

The main checkout has uncommitted `grb/` edits that this branch strictly supersedes (it adds constants/`model_array`/docstrings on top of them; nothing is lost — verified by diff). To pull cleanly there:

```bash
git checkout -- grb/          # discard uncommitted grb/ edits (all contained in this PR)
mkdir -p modeling/superseded && mv modeling/*.py modeling/superseded/  # untracked copies would block checkout
git pull
```

`modeling/final_model.py`, `modeling/final_model_plotting.py`, and `modeling/utils.py` are fully replaced by `grb/*` + `fit_final_model.py`; the other phase scripts remain (migrated to `grb.*` imports).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_0112PZc7fej1EwGrwBiFn62R
