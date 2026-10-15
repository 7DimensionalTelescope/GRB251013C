Swift/XRT light-curve products for GRB 251013C from the UK Swift Science Data Centre
(https://www.swift.ac.uk/xrt_curves/03000137/, ObsID 00021862008 v4, 2025-10-25),
downloaded 2026-10-13. T0 = 2025-10-13 17:39:42 UT. PC mode only.

- flux.qdp         0.3-10 keV observed flux, time-averaged ECF 3.70e-11 erg/cm^2 per count
- flux_incbad.qdp  same, including the 129788-141478 s ToO snapshots that UKSSDC flags
                   as unreliable (no centroid); this is where data/xrt.csv's 136158 s
                   point comes from
- PCUL.qdp         3-sigma upper limit at 836199 s (< 6.09e-14 erg/cm^2/s)
- curve.qdp        count-rate curve

data/xrt.csv keeps a spectrum-dependent conversion (10-40% above flux.qdp in the
hard 5.3 ks snapshot); only the 136158 s row is excluded from fits
(XRT_UNRELIABLE_TIMES in grb/const.py).
