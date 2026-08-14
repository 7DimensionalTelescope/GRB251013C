Two commits:

1. **`a97d56c` — equivalence-test fix for the first retune** (unchanged from the original PR): encodes the 2026-07-30 bounds retune as a documented exception table so the harness keeps guarding everything else bit-for-bit.

2. **`23fcf37` — second retune + plot upgrades**, responding to the `final_flare_wing_20260730_171914` run:

## Why

- The two late XRT points (37.8 hr, 112.8 hr) are underpredicted by ~20–30× (+4.3σ, +2.7σ); the wing supplies ~83% of late XRT flux but `p_wing`=3.14 makes it too faint and too soft.
- The model photon index softens to ~2.5 at late times while the data stay at 1.82±0.21 — opposite evolution to the data.
- `n_ism` railed at the 400 bound; `p`=2.23 sat near its 2.3 edge.

## What changed

- **`p` 2.01→1.6, `p_wing` 2.2→1.8** (lower edges): VegasAfterglow supports hard (p<2) electron spectra — verified Γ = p/2+1, so p≈1.76 could in principle match the observed Γ=1.88. A 600-step emcee probe with these bounds *and walkers seeded at p≈1.8* migrates back to p≈2.15, p_wing≈3.2 (the ~700 optical points and the XRT decay slope dominate), so the fit is expected to stay above 2 — but the bounds no longer forbid the hard-spectrum branch.
- **`n_ism` 400→1000**: releasing the rail moves the optimum to ≈530 and improves logP −474.8 → −435.9. All other bounds unchanged.
- **`INITIAL_GUESS`**: emcee probe + monotone Powell polish; scores **logP = −436.65** (run best was −474.78; total χ² 949.4 → 873.0). Full precision on purpose — 3-sig-fig rounding costs ~65 logL. Rail-sitting optima (`xi`, `xi_r`, `xi_wing`, `p_r`, `p_wing`) nudged strictly inside the box (cost: 0.8 logL).
- **`plot_light_curves`**: optional 16–84% (1σ) posterior envelope from thinned post-burn-in draws (`band_draws=100` default; `0` disables). Already regenerated for the 20260730_171914 run directory.
- **`plot_spectral_index_comparison`**: adds the fitted **total (core+wing)** photon-index curve — the quantity the likelihood actually constrains; makes the late-time core→wing softening visible.
- **Equivalence test**: `p`/`p_wing`/`n_ism` updated in `RETUNED_BOUNDS`, new `RETUNED_INITIAL_GUESS`, and the `--png` check now asserts the light-curve figure is byte-identical with the band off while the spectral-index figure is a *documented* divergence.

## Still unfixable by bounds

The late-XRT excess + hard index survive every variant tested (Δχ² available there ≤ ~45 vs hundreds lost in optical). Matching them needs a new model ingredient (e.g. late energy injection or SSC in the XRT band), not wider boxes.

## Verification

- `modeling/test/test_refactor_equivalence.py` passes in default **and** `--png` modes.
- Powell caveat: scipy's Powell **with `bounds=`** is non-monotone here (a control run walked −474.8 → −576.9); all polish numbers come from unbounded Powell with the box enforced as a penalty.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_0112PZc7fej1EwGrwBiFn62R
