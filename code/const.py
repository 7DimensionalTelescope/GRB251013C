import os
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.abspath(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
CIRCULAR_DATA_FILENAME = os.path.join(DATA_DIR, "circular.xlsx")
XRT_DATA_FILENAME = os.path.join(DATA_DIR, "xrt.csv")

TRIGGER_TIME_STR = "2025-10-13T17:39:42 UTC"
TRIGGER_TIME = datetime.strptime(TRIGGER_TIME_STR, "%Y-%m-%dT%H:%M:%S %Z")

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

    # SDSS / Sloan (AB)
    "u": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 355,
        "bandwidth_nm": 57,
    },
    "g": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 475,
        "bandwidth_nm": 138,
    },
    "g'": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 477,
        "bandwidth_nm": 137,
    },
    "r": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 622,
        "bandwidth_nm": 138,
    },
    "r'": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 623,
        "bandwidth_nm": 137,
    },
    "i": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 763,
        "bandwidth_nm": 153,
    },
    "z": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 905,
        "bandwidth_nm": 137,
    },

    # Unfiltered / instrumental
    "Clear": {
        "system": "instrumental",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 600,
        "bandwidth_nm": None,
    },
    "clear": {
        "system": "instrumental",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 600,
        "bandwidth_nm": None,
    },
    "white": {
        "system": "instrumental",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 600,
        "bandwidth_nm": None,
    },
    "w": {
        "system": "instrumental",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 600,
        "bandwidth_nm": None,
    },
}
