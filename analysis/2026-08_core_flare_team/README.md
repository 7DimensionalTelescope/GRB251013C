# 2026-08 core+flare team exploration — archived working files

Moved from the ephemeral Claude job scratch (`~/.claude/jobs/8177b5c4/tmp`) on
2026-08-04 so the material survives job cleanup and future sessions/agents can
find it inside the project. All internal path references have been rewritten to
this folder; scripts remain runnable as-is (they point at the worktree
`retune-on-refactor` for the `grb` package).

## Start here

- **`reports/team_synthesis.md`** — the main deliverable of the five-role team
  (physicist / data-analyst / consistency / logic / GRB expert). Section 4 is
  the concrete next-generation fit specification; Section 3 lists method traps.
- **`reports/leavitt_zp_fix.patch`** — the Leavitt Vega/AB zero-point bug fix
  for `grb/utils.py` (Rc/Ic fitted 0.179/0.384 mag too bright). **NOT applied**
  — coordinate with the in-flight uncommitted edits in the main checkout.
- **`reports/analyst_late_README.md`** — provenance of the late-time photometry
  CSVs. `reports/team_brief.md` is the original team charter.

## Fit-ready data (`data/`)

- `grbexpert_newpoints.csv` — Chandra 27.4-d X-ray point + ALMA/AMI/VLA radio
  (AMI weakest: 11″ positional offset).
- `analyst_late_circular_fit.csv` — 12 real GCN late-time optical points,
  cut at 2.0e5 s (Koshka 6.09e5 s outlier dropped).
- `analyst_late_{r,i,z}_fit.csv` — parsed-figure photometry (±0.22 mag
  systematic; backup only). `*_SNcontaminated.csv` variants keep t>7e5 s.
- `consistency_parsed_fixed.npz` — corrected parse of `sample.png`
  (**supersedes** `sample_parsed.npz`, whose z-band was a y-marker halo).
- Best-fit vectors (19-dim no-wing unless noted):
  - `nowing_flare_best.npy` — "FLARE-X" core+RS+flare, logP −793.3 (wide box).
  - `nowing_matched_best.npy` — matched-box no-wing control, −757.3.
  - `widewing_best.npy` — 26-dim wide-box wing control, −419.0.
  - `nowing_best.npy` (−766.7), `polished_best.npy` / `joint_best.npy`
    (26-dim polished incumbent-family optimum, −436.65).
- `coremap_good.npy` / `coremap_lp.npy` — single-power-law-XRT core FS
  parameter-space map.

## Scripts (`scripts/`) and figures (`figures/`)

Scripts are the audit trail: `nowing_matched.py` / `widewing.py` are the two
control MCMC runs that closed the wing-vs-no-wing arbitration; `*_` prefixes
(`physicist_`, `analyst_`, `consistency_`, `logic_`, `grbexpert_`) are the
team agents' analyses; earlier phases: `opt2/polish` (retune), `corexrt*`
(single-PL core search), `threecase/showfit*/flarex_figs` (comparison figures).
Key figures: `flarex_lc.png`, `flarex_vs_sample.png`, `threecase_lc.png`,
`runs_vs_sample.png`. Raw run logs in `logs/`.

## Headline conclusions (details in team_synthesis.md)

1. Wing beats no-wing on the *fitted* dataset by ≥321 logP in a matched box —
   not a prior-width artifact — but the fitted dataset is defective in
   wing-favoring ways (Leavitt ZP bug, stale i-band errors, no cal floor), and
   the wing fails all external data (Chandra ×28, radio ×6–29).
2. Published Chandra point ⇒ jet break at 6–10 d ⇒ θ_j ≈ 0.43–0.60; radio
   rejects n≈134; low-n branch fixes photon-index floor + chromaticity.
3. The decisive experiment is the Section-4 refit (ZP fix + new i_data +
   0.05 mag floors + Chandra/radio + low-n box + Magnetar injection), judged
   against three pre-registered predictions recorded in team_synthesis.md.
