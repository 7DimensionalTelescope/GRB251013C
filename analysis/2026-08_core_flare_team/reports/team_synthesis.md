# GRB 251013C — five-role team synthesis (2026-08-03)

Team: physicist, data-analyst, consistency checker, logic auditor, GRB expert.
Question: explore the core+flare(+RS), no-wing model ("FLARE-X") further.

## 1. Verdict on the architecture question

The core+flare direction is **vindicated in shape but wrong in three specifics**, and
the week's wing-vs-no-wing framing is partly a false dichotomy:

- **Wide core: yes, but broken.** The fitted theta_c=0.76 was a flat direction
  (likelihood insensitive above ~0.5 rad; only theta_c > 0.3 is a measurement).
  A published Chandra point at 27.4 d (GCN 42642) requires a **jet break at 6-10 d**
  -> theta_j ~ 0.43-0.60 rad. Wide (25-34 deg), precedented (030329-like), on the
  canonical Frail energy at theta~0.5. FLARE-X without a break over-predicts
  Chandra 3.5x; the incumbent wing model under-predicts it **28x**.
- **Density: n~134 is rejected by radio.** ALMA 0.30 mJy @ 5.5 hr is over-predicted
  14x (nu_a parked on the band). The low-n branch (n ~ 0.5-30) also naturally
  places nu_c between optical and X-ray, resolving BOTH the photon-index floor
  (obs 1.82 vs model 2.06) and the observed chromatic decay (delta-alpha = 0.46+-0.13).
  The n>=5 box floor was silently forcing the wrong cooling regime.
- **The "flare" is two things.** A genuinely distinct HARD X-ray component exists
  (photon index hardens to 1.58+-0.15 during 2.1-10 ks — real). But as fitted it is
  not credible internal dissipation: width/t = 2.3 (population: 0.1-0.5), 28% of the
  blast energy, and its optical extrapolation is rejected 2.7-sigma by the 7DT SED
  (measured beta=-1.10 vs blended model -0.90). The achromatic optical bump at
  2.5-3 ks + three later rebrightenings (2.5e5 s [flux x2, 3 teams], day 9,
  day 12-13; GCN 42736) form a refreshed-shock / late-engine SEQUENCE.
  First mechanism to fit: **Magnetar energy injection** (time-dependent; the only
  such mechanism in VegasAfterglow). StepPowerLawJet would re-introduce the wing.
- **RS cannot carry the early optical** (post-crossing decay must be ~t^-2.3;
  no physical config sustains alpha~0.6 to 2 ks). It isn't needed: the FS nu_m-break
  curvature reproduces the early shallow optical; RS tops up normalization <200 s.
  Cap eps_B_r <= 0.2 — the fit was climbing into sigma>1 where real reverse shocks
  are suppressed (no code feedback there).

## 2. Data-side findings (these move numbers more than any physics choice)

1. **Leavitt zero-point BUG (fix first).** grb/utils.py `_mag_to_flux_mJy` uses the
   AB ZP (3631 Jy) for all bands; Rc/Ic are Vega (FILTER_INFO declares it; GCN 42333
   -> Lupton 2005 -> Vega by construction). Fitted Leavitt fluxes are 0.179 (Rc) /
   0.384 (Ic) mag too bright. Patch: analysis/2026-08_core_flare_team/reports/leavitt_zp_fix.patch (NOT applied — main
   has in-flight edits to the same modules). After the fix, the residual band-to-band
   disagreement is 0.05 mag — inside a ±10% cal prior.
2. **Error model.** Leavitt errors are real S/N-only errors missing calibration
   systematics: add a **0.05 mag floor in quadrature to all optical datasets**
   (Rc chi2/pt 24.2 -> 2.7). Additive pedestal REFUTED (wrong signs/shape/amplitude).
   Main's new i_data.csv is a genuine re-reduction with honest errors (chi_rms 0.97;
   the old 6.5-6.7 ks dip was spurious and is gone); under it FLARE-X's optical
   chi2 = 2414 with i-band 1199 — model-shape error, no data escape hatch.
3. **i-band absolute cross-cal is undetermined at ±0.15 mag** — no fixed scale;
   ±10% include_cal prior, treat result as uninformative. (Earlier "x1.11" retracted.)
4. **New external data, fit-ready** (analysis/2026-08_core_flare_team/data/grbexpert_newpoints.csv): Chandra 27.4-d
   X-ray point + ALMA/AMI/VLA. AMI is the weakest (position 11 arcsec off, 30 arcsec
   beam). Chandra/ALMA errors are estimated (30%/15%), values carry the argument.
5. **Real late-time GCN photometry** (analysis/2026-08_core_flare_team/data/analyst_late_circular_fit.csv, 12 pts,
   cut at 2.0e5 s — the bump already rises by 2.4e5 s; hold later points out-of-sample;
   drop the Koshka 6.09e5 s outlier). Parsed-figure datasets exist as backup
   (±0.22 mag systematic; z-band from the CORRECTED parse only).
6. **SN and host: neither needs modeling.** The "SN" is tentative photometric
   (GCN 42736), 1.6 mag brighter and ~10 d earlier than 98bw at z=0.572, colors
   get bluer not redder, still declining through the expected peak; the day 9 /
   12-13 bumps are more rebrightenings. Host r >~ 24 (DESI DR10 empty at the radio
   position): <=0.5% at all fitted epochs. Just exclude optical t > 7e5 s.

## 3. Method traps documented this session

- FLARE-X lies OUTSIDE the committed grb/params.py box on six parameters; it is
  reachable only in the widened box. Any comparison must hold the box fixed
  (the wide-box wing control run is doing exactly this; result pending).
- Likelihood has a POLE at exactly p=2 (barrier ~7000): p<2 and p>2 are
  dynamically disconnected; never conclude across p=2 from a single chain.
- Fix xi=1 (exact Eichler-Waxman degeneracy: fitted xi=0.31 means the "physical"
  numbers are E=3.5e52, n=430, eps_e=0.011, eps_B=0.009 in xi=1 units).
- Tie tau to the prompt T90 (~19 s) instead of fitting it (thick-shell degeneracy
  with Gamma0; Gamma_crit=83 is what is actually measured).
- scipy bounded Powell non-monotone (older finding, still applies); flat-error
  i-band under-weights the band 5.6x vs the re-reduction.
- Sign prediction on record for the pending control run: it runs on the data
  version most favorable to no-wing; a wing win there is final, a no-wing win
  must survive re-scoring under the new i-band errors.

## 4. Next-generation fit specification (concrete)

Data: XRT(45) + spectral index + Chandra point + ALMA/VLA (AMI down-weighted)
+ new i_data.csv + Leavitt (ZP-fixed, 0.05 mag floor) + 7DT SED
+ analyst_late_circular_fit.csv (12 pts, t<2e5 s); optical t>7e5 s excluded;
0.05 mag floor everywhere; include_cal ±10%.

Model: single TophatJet core, theta_c in [0.15, 0.8] (expect ~0.4-0.5, now
constrained by Chandra), Gamma0 [100, 2000] (or fix via Gamma_crit), n in
[0.05, 100] (floor LOWERED — decisive), p [2.02, 2.6], eps_e [0.005, 0.3],
eps_B [1e-5, 0.1], xi=1 FIXED, tau=19 s FIXED, RS with eps_B_r capped 0.2,
+ Magnetar injection (L0, t_sd) targeted at the 2.5-3 ks episode,
+ retain a Norris flare ONLY as the hard X-ray spectral component if the
injection alone fails the 2-10 ks X-ray hardness (falsifier: stacked XRT
photon index at 1-5e4 s: flare predicts 1.70, injection predicts ~2.06).
Late rebrightenings (2.5e5 s+) NOT fit yet — out-of-sample validation targets.

## 5. Open items

- Wide-box wing control run finishing (~step 300/2400 at synthesis time;
  best -432.2, theta_c already migrating to ~0.65).
- Leavitt ZP patch to be applied (coordinate with in-flight main edits).
- Tabular compilation photometry from the collaboration remains the biggest
  single data upgrade for the 1e5-7e5 s regime.
