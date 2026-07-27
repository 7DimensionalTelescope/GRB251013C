import os
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from astropy.cosmology import Planck18 as cosmo
from VegasAfterglow.units import keV

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.abspath(__file__)

DATA_DIR = os.path.join(BASE_DIR, "data")
CIRCULAR_DATA_FILENAME = os.path.join(DATA_DIR, "circular.xlsx")
I_DATA_FILENAME = os.path.join(DATA_DIR, "i_data.csv")
XRT_DATA_FILENAME = os.path.join(DATA_DIR, "xrt.csv")
XRT_INDEX_DATA_FILENAME = os.path.join(DATA_DIR, "xrt_index.csv")
SDT_DATA_FILENAME = os.path.join(DATA_DIR, "sdt.csv")

RA = float(os.getenv("ra"))
DEC = float(os.getenv("dec"))
REDSHIFT = float(os.getenv("redshift"))

AV = float(os.getenv("AV", None))

D_L = cosmo.luminosity_distance(REDSHIFT).to("cm").value

TRIGGER_TIME_STR = str(os.getenv("trigger_time"))
TRIGGER_TIME = datetime.strptime(TRIGGER_TIME_STR, "%Y-%m-%dT%H:%M:%S UTC")

MODEL_RESOLUTIONS = (0.1, 0.25, 10)

FIT_RESULTS_DIR = Path(BASE_DIR) / "modeling" / "fit_results"

# XRT 0.3-10 keV band, and the two frequencies (Hz) between which the local
# synchrotron slope is measured for the spectral-index constraint.
XRT_BAND = (0.3 * keV, 10.0 * keV)
XRT_NU_LO = 7.25e16
XRT_NU_HI = 2.42e18

# The spectral-index constraint is applied only where the phenomenological
# flare contributes less than this fraction of the XRT flux.
SI_FLARE_FRAC_MAX = 0.5

XRT_EXCLUDE_TIME_RANGE = (3e3, 1e4)
XRT_FLARE_START_TIME = 3e3
XRT_FLARE_END_TIME = 1e4

# Log10 Gaussian prior on host-galaxy A_V.
HOST_AV_LOG10_MEAN = -0.82
HOST_AV_LOG10_SIGMA = 0.41

C_CM_PER_S = 2.99792458e10
LN10_OVER_2P5 = 0.4 * np.log(10.0)

FILTER_INFO = {
    # Johnson–Cousins (Vega)
    "B": {
        "system": "Vega",
        "vega_zero_point_jy": 4260,
        "central_wavelength_nm": 440,
        "bandwidth_nm": 98,
    },
    "V": {
        "system": "Vega",
        "vega_zero_point_jy": 3640,
        "central_wavelength_nm": 550,
        "bandwidth_nm": 89,
    },
    "R": {
        "system": "Vega",
        "vega_zero_point_jy": 3080,
        "central_wavelength_nm": 640,
        "bandwidth_nm": 150,
    },
    "Rc": {
        "system": "Vega",
        "vega_zero_point_jy": 3080,
        "central_wavelength_nm": 641,
        "bandwidth_nm": 158,
    },
    "Ic": {
        "system": "Vega",
        "vega_zero_point_jy": 2550,
        "central_wavelength_nm": 798,
        "bandwidth_nm": 154,
    },

    # Near-IR (Vega)
    "J": {
        "system": "Vega",
        "vega_zero_point_jy": 1594,
        "central_wavelength_nm": 1235,
        "bandwidth_nm": 162,
    },
    "L": {
        "system": "Vega",
        "vega_zero_point_jy": 280,
        "central_wavelength_nm": 3450,
        "bandwidth_nm": 570,
    },
    "u": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 360.8,
        "bandwidth_nm": 56.4,
    },
    "g": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 467.1,
        "bandwidth_nm": 106.5,
    },
    "g'": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 472.3,
        "bandwidth_nm": 126.5,
    },
    "r": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 614.1,
        "bandwidth_nm": 105.5,
    },
    "r'": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 620.2,
        "bandwidth_nm": 125.4,
    },
    "i": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 762.5,
        "bandwidth_nm": 110.3,
    },
    "z": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 892.3,
        "bandwidth_nm": 116.4,
    },
    # Unfiltered / instrumental
    "Clear": {
        "system": "instrumental",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 0,
        "bandwidth_nm": 0,
    },
    "clear": {
        "system": "Vega",
        "vega_zero_point_jy": 3064,
        "central_wavelength_nm": 0,
        "bandwidth_nm": 0,
    },
    "white": {
        "system": "instrumental",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 0,
        "bandwidth_nm": 0,
    },
    "w": {
        "system": "instrumental",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 0,
        "bandwidth_nm": 0,
    },
    "Blue": {  
        "system": "instrumental",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 0,
        "bandwidth_nm": 0,
    },
    "Red": {
        "system": "instrumental",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 0,
        "bandwidth_nm": 0,
    },
}
