# GRB 251013C — core+flare(+RS) model exploration: team brief

## Object & data
- GRB 251013C, z=0.572, D_L=1.059e28 cm, trigger 2025-10-13 17:39:42. Host A_V prior: log10 A_V ~ N(-0.82, 0.41).
- Repo (post-refactor, use this code): `/data/dtak/research/grb/GRB251013C/.claude/worktrees/retune-on-refactor`
  (package `grb/`, driver `fit_final_model.py`). Run python as
  `/home/dtak/miniconda3/envs/grb251013c/bin/python` with cwd = that worktree and
  `OMP_NUM_THREADS=1` etc. set BEFORE importing numpy. VegasAfterglow 2.0.6.
- Fitted datasets (via `grb.modeling.load_all_optical_data`): XRT 45 pts (0.13–113 hr,
  incl. 2 late pts at 37.8/112.8 hr), i-band 67, Leavitt_Rc 35, Leavitt_Ic 18,
  7DT SED 22×1pt at 6.47 hr. Plus XRT photon-index series (45 pts, `load_xrt_spectral_index`).
- External: `/data/dtak/research/grb/GRB251013C/sample.png` = preliminary SVOM/VT+FM-GFT+LCO+GRANDMA
  multiband LC compilation, 1e2–2e6 s. Parsed points (pixel-extracted):
  `/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/sample_parsed.npz` (keys `<band>_t`,`<band>_m`;
  PLOTTED mags: offsets VT/B+2, r+0, R+0, VT/R+0, i-1, I-1, z-2, y-3; green=g?+1 guess;
  apparent AB, NOT galactic-extinction-corrected).
- USER-PROVIDED FACTS (take as given): (1) at ~1e5 s the compilation favors core+flare
  over core+wing; (2) at ~1e6 s there is a SUPERNOVA observation — the latest optical
  bump is SN light, not afterglow. (z=0.572 → SN Ic-BL peak ~2e6 s obs; rise from ~7e5 s.)

## Model architecture under study (user-selected): core + flare + RS, NO wing
- Core = TophatJet FS (spreading, duration tau) + optional RS; flare = Norris pulse in
  XRT band, spectrally extrapolated to optical with slope flare_beta; host SMC extinction.
- Best current no-wing fit "FLARE-X" (emcee probe, logP=-793.3 vs all fitted data;
  vector: `/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/nowing_flare_best.npy`, 19 labels =
  make_param_defs(True,False) order):
  E_iso=1.08e52, Gamma0=136, theta_c=0.760 (AT box edge 0.8), n=134, p=2.121,
  eps_e=0.034, eps_B=0.030, xi=0.31, tau=46.5, p_r=2.77, eps_e_r=0.108, eps_B_r=0.528,
  xi_r=0.88, A_V=0.036, t_start=2087, tau_rise=98.6, tau_decay=7447, A=4.72e-10, beta=0.683.
- Per-term: XRT chi2=47.8 (45 pts), SI chi2=38.2, optical chi2=1498 (142 pts),
  late XRT resid +1.9/+1.1 sigma.
- Incumbent 2-component model (core+RS+flare+WING, production MCMC
  `modeling/fit_results/final_flare_wing_20260802_131026`, logP=-430.18): XRT 65,
  SI 60, optical 748, late XRT +4.3/+2.7 sigma. WORSE than FLARE-X on every X-ray
  metric and on the compilation at t>2e4 s (wing nosedives late), BETTER on the
  fitted optical (mostly the Leavitt 1-2e4 s hump; note Leavitt chi2/pt ~ 10 in EVERY
  model ever fit — errors may be underestimated).
- Established negative results (do not re-litigate): p<2 branch loses in joint fits and
  user rules it unphysical; low-eps_B/high-p core branch loses (dlogL<=-620); wing-based
  fits can't fix late XRT photon index (obs Gamma=1.82±0.21 at 37.8/112.8 hr; wing p~3.2
  predicts ~2.5); scipy Powell WITH bounds= is non-monotone on this problem (use penalty box).
- Key figures: `/home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/figures/flarex_lc.png` (components),
  `flarex_vs_sample.png` (FLARE-X vs incumbent vs compilation), `threecase_lc.png`.

## Open problems for the team
1. FLARE-X params at/near rails: theta_c=0.76 (box 0.8), eps_B_r=0.53 (box 0.6). Physical?
2. The Leavitt/i-band 1-2e4 s optical hump: FLARE-X underfits it (optical chi2 1498 vs 748).
   Flare posterior refuses to move there. What supplies it? (RS? flare spectral shape?
   data calibration? note user's uncommitted `include_cal` feature adds ±10% cal scales.)
3. The 2-3e5 s optical bump in the compilation (~1 mag above FLARE-X in r/i) — pre-SN.
   Energy injection? density bump? Or SN rising earlier than assumed?
4. RS role: user wants RS to carry the EARLY optical (t<2e3 s, i-band 5-17 mJy,
   shallow decay ~0.6-0.8). Current FLARE-X RS: tau=46 s, eps_B_r=0.53, decays fast.
   Can a physical RS do this? What parameter region?
5. SN at ~1e6 s: how to handle in fits (exclude t>7e5 s optical, or add SN template?).
6. Machine courtesy: shared 72-core box, other jobs running. Cap any computation at
   ~8 workers. Do NOT modify repo files; write only under /home/dtak/research/grb/GRB251013C/analysis/2026-08_core_flare_team/<yourrole>_*.
