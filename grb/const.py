import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.abspath(__file__)

DATA_DIR = os.path.join(BASE_DIR, "data")
CIRCULAR_DATA_FILENAME = os.path.join(DATA_DIR, "circular.xlsx")
XRT_DATA_FILENAME = os.path.join(DATA_DIR, "xrt.csv")
XRT_INDEX_DATA_FILENAME = os.path.join(DATA_DIR, "xrt_index.csv")
SDT_DATA_FILENAME = os.path.join(DATA_DIR, "sdt.csv")

RA = float(os.getenv("ra"))
DEC = float(os.getenv("dec"))
REDSHIFT = float(os.getenv("redshift"))
TRIGGER_TIME_STR = str(os.getenv("trigger_time"))
TRIGGER_TIME = datetime.strptime(TRIGGER_TIME_STR, "%Y-%m-%dT%H:%M:%S UTC")

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
        "central_wavelength_nm": 745.8,
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
}
