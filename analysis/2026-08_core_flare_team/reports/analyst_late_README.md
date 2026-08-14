# Provisional late-time optical datasets — GRB 251013C

Built 2026-08-03 by the data-analyst role. Nothing in the repository was modified.
Builder script: `analyst_build_late_v2.py` (rerun it to regenerate).

## Data versions used (rebuild 2, 2026-08-03)

- **Parse:** `consistency_parsed_fixed.npz`. The earlier `sample_parsed.npz` is
  **superseded** — its z trail between 3e4 and 5e4 s was anti-alias halo from the
  black y markers and was **1.1 mag too bright** (a size cap had discarded the
  real 1141-px z blob). The r and i trails are unchanged between the two parses;
  only z moved. Any z number from the first delivery should be discarded.
- **i-band:** the main checkout's re-reduced `data/i_data.csv`
  (md5 `301689c1…`, schema `mjd,mag,magerr,filter,instrument_name`, real
  per-point errors 0.028–0.080, median 0.053). The worktree's older file
  (flat 0.10 mag errors) is stale.
- **Everything else** (`circular.xlsx`, `sdt.csv`, `xrt*.csv`) is byte-identical
  between the main checkout and the worktree, so all Leavitt, 7DT and XRT
  numbers are version-independent.

## Files

| file | source | n | time span |
|---|---|---|---|
| `analyst_late_r.csv` | parsed `sample.png` | 26 | 6.7e3 – 6.7e5 s |
| `analyst_late_i.csv` | parsed `sample.png` | 55 | 1.1e2 – 3.4e5 s |
| `analyst_late_z.csv` | parsed `sample.png` | 19 | 3.0e4 – 6.7e5 s |
| `analyst_late_<band>_SNcontaminated.csv` | same, t > 7e5 s | 5 / 0 / 9 | excluded from the main files |
| `analyst_late_circular.csv` | **real photometry**, `data/circular.xlsx` | 22 | 3.1e4 – 6.9e5 s |
| `analyst_late_circular_SNcontaminated.csv` | same, t > 7e5 s | 8 | excluded |
| `analyst_late_<band>_fit.csv`, `analyst_late_circular_fit.csv` | **the ones to fit**, cut at t < 2.5e5 s | 18 / 51 / 6 / 12 | see below |

### Which file to fit

Use the `*_fit.csv` files. Per the logic audit, anything at t > 2.5e5 s is inside
the achromatic rebrightening and should be held out as out-of-sample validation
rather than fitted.

**Caveat on the cut:** the bump has already begun before 2.5e5 s. The local decay
index between consecutive r-like points goes to alpha = −3.3 (rising) between
1.87e5 and 2.44e5 s, and the JinShan r point at 2.44e5 s is already 0.25 mag
*brighter* than the 1.2e5 s point. **A cut at 2.0e5 s is the defensible one**;
2.5e5 s leaves that single rising point in. The `*_fit.csv` files use 2.5e5 s as
specified — drop the 2.44e5 s row if you want the clean version.

**Use `analyst_late_circular.csv` in preference to the parsed files wherever they
overlap.** It is real GCN photometry with quoted errors that is already in the
repository and is currently used by no fit. The parsed files add coverage in `i`
and `z` where the circulars have almost nothing.

## Columns

`time_s` (s since trigger), `flux_mJy`, `flux_err_mJy`, `frequency_Hz`,
`wavelength_AA`, `mag_AB_galcorr`, `mag_err`, and

- parsed files: `n_pix` (pixels in the bin), `flux_err_mJy_conservative`
- circular file: `facility`, `filter`, `ab_corr`, `Circular` (GCN number)

## Provenance and processing

**Parsed files.** Pixel trails from `sample_parsed.npz` (extracted from
`/data/dtak/research/grb/GRB251013C/sample.png`). Legend offsets undone
(r +0, i −1, z −2) to recover apparent AB magnitude, binned by median in
0.05-dex time bins requiring ≥3 pixels, galactic extinction removed with
`grb.extinction.galactic_extinction` at the `FILTER_INFO` effective wavelengths
(A_gal = 0.137 / 0.098 / 0.075 mag for r / i / z), then converted to mJy with the
AB zero point. `mag_err` is `max(0.15 mag, 1.4826·MAD of the bin)`, as specified.

**Circular file.** Non-upper-limit rows of `data/circular.xlsx` with t > 3e4 s.
Vega bands (R, Rc, V) were converted to AB using the `FILTER_INFO` zero points
before the flux conversion — see the caveat below, this is *not* what the
current fitting code does. Galactic extinction removed the same way. A 0.05 mag
error floor was applied.

## Caveats

1. **Parse accuracy is ±0.22 mag, not ±0.15.** Checked against the 8 real
   circular measurements that fall inside the parsed trails: mean(real − parsed)
   = +0.014 mag with 0.219 mag scatter. So there is no systematic bias, but the
   0.15 mag floor required by the spec is optimistic. Use
   `flux_err_mJy_conservative` (0.22 mag added in quadrature) for fitting.

2. **The parsed `i` trail is not independent of the fitted `i_data`.** Against the
   old i-band reduction they agreed to +0.012 ± 0.007 mag over 63 overlapping
   points — far too good for independent datasets. `i_data` is the NUTTelA-TAO
   i-band series (8 of its early epochs also appear verbatim in
   `circular.xlsx`). Do not fit the parsed `i` points at t < 1.1e4 s alongside
   `i_data`, and note that a wholesale ingest of `circular.xlsx` would duplicate
   the 8 NUTTelA-TAO rows at 96–635 s.

3. **Vega/AB — now confirmed from the source circular.** The circular file
   applies a Vega→AB correction (+0.179 mag for R/Rc, +0.384 for Ic, −0.003 for
   V) that the repository's `grb.utils._mag_to_flux_mJy` does **not** apply — it
   converts every band with the AB zero point 3631 Jy.

   GCN Circular 42333 (Leavitt Observatory, Manciano, Italy; 250mm f/8
   Ritchey-Chrétien) states the magnitudes were calibrated against Pan-STARRS
   catalog stars and **"converted using Lupton (2005) equations"**. The Lupton
   (2005) transformations were derived by matching SDSS DR4 photometry to
   Stetson's published standard-star photometry — i.e. the Landolt
   Johnson-Cousins system, which is Vega-based. So Leavitt Rc/Ic are Vega
   magnitudes by construction, and the AB conversion in the repository is wrong.
   The circular also states the magnitudes are **not** corrected for Galactic
   extinction, which is what the loader assumes.

   Independent statistical confirmation is in `analyst_system2.py` (measured
   offsets +0.144 Rc, +0.333 Ic against predicted +0.179, +0.384, with the V
   control at −0.012 against predicted −0.003).

   If you load these CSVs alongside data loaded through `grb.io.read_data`, the
   two are on different scales.

4. **t > 7e5 s is supernova-contaminated** and has been split into the
   `_SNcontaminated` sidecars, per the brief. Those points flatten near
   0.0072 mJy at 1.0–1.1e6 s, consistent with a SN plateau rather than afterglow.

5. **Koshka/Ziess-1000 R at 6.09e5 s is an outlier** (20.75 AB-corrected, versus
   21.66 from SAORAS and SAO/Zeiss-1000 at 6.13e5 s — a factor 2.3 in flux).
   Consider dropping it.

6. **The additive-pedestal hypothesis is refuted** (`analyst_pedestal.py`,
   `analyst_pedestal2.py`). Fitting F_obs = F_model + C per Leavitt dataset gives
   C = −0.0250 ± 0.0045 mJy in Rc but +0.1772 ± 0.0122 mJy in Ic — same
   telescope, same aperture, opposite signs. A host or blend contributes positive
   flux in every band. Independently, the Leavitt offset is flat as the source
   fades (−0.119 mag/dex versus −1.0 predicted for a pedestal), and the afterglow
   is detected at 0.0046 mJy at 6.1e5 s, bounding any common field pedestal to
   <0.8% of the faintest Leavitt point.

7. The `green` trail in `sample_parsed.npz` has a guessed band and offset (g, +1)
   because the legend is cut off in the figure; it is not used here. `y` and the
   `VT/*` trails are also not used.

## The late-time rebrightening (real photometry, r/r'/Rc/R, AB, galactic-corrected)

    1.20e5 s  21.15   0.0126 mJy   SVOM/COLIBRI
    1.87e5 s  21.85   0.0066       INO340        (+-0.25, noisy)
    2.44e5 s  20.90   0.0158       JinShan
    3.43e5 s  20.46   0.0238       SAO/Zeiss-1000
    3.51e5 s  20.60   0.0208       OHP/T193
    4.21e5 s  20.87   0.0163       Mondy/AZT-33IK
    5.00e5 s  21.63   0.0081       Mondy/AZT-33IK
    6.13e5 s  21.66   0.0079       SAORAS and SAO/Zeiss-1000 (independent, agree)

The flux roughly doubles between 1.2e5 and 3.4e5 s and then falls by a factor 3
by 6.1e5 s. The peak is confirmed by two independent facilities in two filters.
At z = 0.572 this is ~2.5 days rest-frame, far too early for the supernova, so it
is an afterglow feature (energy injection / refreshed shock / density structure),
distinct from the SN plateau after 7e5 s.
