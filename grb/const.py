"""
Constants for GRB fitting
"""
import os
from datetime import datetime
from dotenv import load_dotenv
from astropy.cosmology import Planck18 as cosmo

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.abspath(__file__)

DATA_DIR = os.path.join(BASE_DIR, "data")
CIRCULAR_DATA_FILENAME = os.path.join(DATA_DIR, "circular.xlsx")
XRT_DATA_FILENAME = os.path.join(DATA_DIR, "xrt.csv")
XRT_INDEX_DATA_FILENAME = os.path.join(DATA_DIR, "xrt_index.csv")
XRT_UNABSORB_DATA_FILENAME = os.path.join(DATA_DIR, "xrt_unabsorb_Jy_10keV.csv")
SDT_DATA_FILENAME = os.path.join(DATA_DIR, "sdt.csv")
SDT_PIVOT_DATA_FILENAME = os.path.join(DATA_DIR, "sdt_pivot.csv")
SDT_TRACER_DATA_FILENAME = os.path.join(DATA_DIR, "sdt_tractor.csv")
CIRCULAR_WAVELENGTH_DATA_FILENAME = os.path.join(DATA_DIR, "circular_wavelength.csv")

RA = float(os.getenv("ra"))
DEC = float(os.getenv("dec"))
REDSHIFT = float(os.getenv("redshift"))

AV = float(os.getenv("AV", None))

D_L = cosmo.luminosity_distance(REDSHIFT).to("cm").value

TRIGGER_TIME_STR = str(os.getenv("trigger_time"))
TRIGGER_TIME = datetime.strptime(TRIGGER_TIME_STR, "%Y-%m-%dT%H:%M:%S UTC")
AV = float(os.getenv("av", 0))

FILTER_INFO = {
    #################### Vega System ####################
    
    # ===========================================================================
    # Johnson–Cousins (Vega)
    # ===========================================================================
    # Vega zero point (Jy) & Pivot wavelength (nm) & Effective width (nm)
    #       Reference: SVO Filter Profile Service - OHP/Cam120 filter
    #       Sensor: Andor Ikon L 936 / https://ohp.osupytheas.fr/le-telescope-de-120cm/
    "U": {
        "system": "Vega",
        "vega_zero_point_jy": 2082.42,
        "central_wavelength_nm": 370.376,
        "bandwidth_nm": 45.714,
    },
    "B": {
        "system": "Vega",
        "vega_zero_point_jy": 4042.25,
        "central_wavelength_nm": 440.964,
        "bandwidth_nm": 94.922,
    },
    "V": {
        "system": "Vega",
        "vega_zero_point_jy": 3583.14,
        "central_wavelength_nm": 553.812,
        "bandwidth_nm": 116.480,
    },
    "R": {
        "system": "Vega",
        "vega_zero_point_jy": 3026.23,
        "central_wavelength_nm": 653.395,
        "bandwidth_nm": 167.668,
    },
    "Rc": {
        "system": "Vega",
        "vega_zero_point_jy": 3026.23,
        "central_wavelength_nm": 653.395,
        "bandwidth_nm": 167.668,
    },
    "Ic": {
        "system": "Vega",
        "vega_zero_point_jy": 2415.99,
        "central_wavelength_nm": 798.940,
        "bandwidth_nm": 133.700,
    },
    # ===========================================================================
    # Near-IR (Vega)
    # ===========================================================================
    # Vega zero point (Jy) & Pivot wavelength (nm) & Effective width (nm)
    #       Reference: SVO Filter Profile Service - 2MASS filter
    "J": {
        "system": "Vega",
        "vega_zero_point_jy": 1594,
        "central_wavelength_nm": 1235.000,
        "bandwidth_nm": 152.026,
    },
    "H": {
        "system": "Vega",
        "vega_zero_point_jy": 1024,
        "central_wavelength_nm": 1662.000,
        "bandwidth_nm": 241.018,
    },
    "K": {
        "system": "Vega",
        "vega_zero_point_jy": 666.8,
        "central_wavelength_nm": 2159.000,
        "bandwidth_nm": 250.619,
    },
    "Ks": {
        "system": "Vega",
        "vega_zero_point_jy": 666.8,
        "central_wavelength_nm": 2159.000,
        "bandwidth_nm": 250.619,
    },
    
    #################### AB System ####################
    
    # ===========================================================================
    # GOTO (AB System)
    # ===========================================================================
    # Pivot wavelength(nm) & Effective width(nm)
    #       Reference: SVO Filter Profile Service - GOTO filter
    "B_GOTO": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 4622.88,
        "bandwidth_nm": 735.14,
    },
    "G_GOTO": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 5374.54,
        "bandwidth_nm": 857.73,
    },
    "L": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 3450,
        "bandwidth_nm": 570,
    },
    "L_GOTO": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 3450,
        "bandwidth_nm": 570,
    },
    "R_GOTO": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 6426.84,
        "bandwidth_nm": 946.75,
    },
    # ===========================================================================
    # SDSS (AB System)
    # ===========================================================================
    # Pivot wavelength(nm) & Effective width(nm)
    #       Reference: SVO Filter Profile Service - OAN-SPM/OPTICAM filter (+CCD)
    #       Sensor: Andor Zyla 4.2-Plus / https://www.southampton.ac.uk/opticam/project/instrument.page
    "u": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 356.544,
        "bandwidth_nm": 37.512,
    },
    "g": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 480.636,
        "bandwidth_nm": 107.148,
    },
    "g'": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 480.636,
        "bandwidth_nm": 107.148,
    },
    "r": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 625.752,
        "bandwidth_nm": 105.5,
    },
    "r'": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 625.752,
        "bandwidth_nm": 126.457,
    },
    "i": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 758.091,
        "bandwidth_nm": 128.867,
    },
    "z": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 887.417,
        "bandwidth_nm": 101.013,
    },
    # ===========================================================================
    # Unfiltered / instrumental (AB System)
    # ===========================================================================
    # Pivot wavelength(nm) & Effective width(nm)
    #       Reference: Calapai - Only use sensor specification
    #       Sensor: Sony ICS285AL
    # Calibration: ATLAS-REFCAT2 - AB System
    "Clear": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 562.212,
        "bandwidth_nm": 187.690,
    },
    "clear": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 562.212,
        "bandwidth_nm": 187.690,
    },
    # White band
    # Pivot wavelength(nm) & Effective width(nm)
    #       Reference: SVO Filter Profile Service - PAN-STARRS PS1 w filter
    # Calibration: USNO-B1 and Pan-STARRS PS1 DR2 - AB System
    "w": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 628.591,
        "bandwidth_nm": 256.173,
    },
    # ===========================================================================
    ## SVOM VT ##
    # ===========================================================================
    # Central wavelength(nm) & Width(nm)
    #       Reference: SVOM VT
    #       Sensor: Sony ICS285AL
    # Calibration: ? - AB System
    "Blue": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 5250.0,
        "bandwidth_nm": 2500.0,
    },
    "Red": {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 8250.0,
        "bandwidth_nm": 3000.0,
    },
    # ===========================================================================
    ## Swift UVOT ##
    # ===========================================================================
    # Pivot wavelength(nm) & Effective width(nm)
    #       Reference: SVO Filter Profile Service - Swift UVOT filter full transmission
    # Zero point flux (Jy) from Swift UVOT calibration
    #       Zero point flux = Flux Factor * 10^(0.4*ZP_mag) * (effective_wavelength^2 / c)
    #       Reference: Poole et al. (2008), Breeveld et al. (2011)
    "v_swift": {
        "system": "UVOT",
        "central_wavelength_nm": 540.2,
        "bandwidth_nm": 76.9,
        "uvot_zero_point_jy": 3644.4 
    },
    "b_swift": {
        "system": "UVOT",
        "central_wavelength_nm": 432.9,
        "bandwidth_nm": 97.5,
        "uvot_zero_point_jy": 4053.7
    },
    "u_swift": {
        "system": "UVOT",
        "central_wavelength_nm": 350.1,
        "bandwidth_nm": 78.5,
        "uvot_zero_point_jy": 1444.6
    },
    "uvw1": {
        "system": "UVOT",
        "central_wavelength_nm": 263.4,
        "bandwidth_nm": 69.3,
        "uvot_zero_point_jy": 917.1
    },
    "uvm2": {
        "system": "UVOT",
        "central_wavelength_nm": 223.1,
        "bandwidth_nm": 49.8,
        "uvot_zero_point_jy": 754.3
    },
    "uvw2": {
        "system": "UVOT",
        "central_wavelength_nm": 203.0,
        "bandwidth_nm": 65.7,
        "uvot_zero_point_jy": 742.0
    },
    "white": {
        "system": "UVOT",
        "central_wavelength_nm": 347.1,
        "bandwidth_nm": 640.0,
        "uvot_zero_point_jy": 1941.9
    },
    # ===========================================================================
    ## 7DT ##
    # ===========================================================================
    # Pivot wavelength(nm) & Effective width(nm)
    #       Reference: Hyeonho Choi
    'm400': {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 4013,
        "bandwidth_nm": 250,
    },
    'm425': {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 4255,
        "bandwidth_nm": 250,
    },
    'm450': {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 4508,
        "bandwidth_nm": 250,
    },
    'm475': {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 4753,
        "bandwidth_nm": 250,
    },
    'm500': {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 5003,
        "bandwidth_nm": 250,
    },
    'm525': {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 5248,
        "bandwidth_nm": 250,
    },
    'm550': {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 5501,
        "bandwidth_nm": 250,
    },
    'm575': {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 5749,
        "bandwidth_nm": 250,
    },
    'm600': {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 6001,
        "bandwidth_nm": 250,
    },
    'm625': {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 6248,
        "bandwidth_nm": 250,
    },
    'm650': {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 6501,
        "bandwidth_nm": 250,
    },
    'm675': {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 6745,
        "bandwidth_nm": 250,
    },
    'm700': {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 6999,
        "bandwidth_nm": 250,
    },
    'm725': {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 7246,
        "bandwidth_nm": 250,
    },
    'm750': {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 7489,
        "bandwidth_nm": 250,
    },
    'm775': {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 7752,
        "bandwidth_nm": 250,
    },
    'm800': {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 7992,
        "bandwidth_nm": 250,
    },
    'm825': {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 8240,
        "bandwidth_nm": 250,
    },
    'm850': {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 8483,
        "bandwidth_nm": 250,
    },
    'm875': {
        "system": "AB",
        "vega_zero_point_jy": None,
        "central_wavelength_nm": 8729,
        "bandwidth_nm": 250,
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

# FILTER_INFO = {
#     # Johnson–Cousins (Vega)
#     "B": {
#         "system": "Vega",
#         "vega_zero_point_jy": 4260,
#         "central_wavelength_nm": 440,
#         "bandwidth_nm": 98,
#     },
#     "V": {
#         "system": "Vega",
#         "vega_zero_point_jy": 3640,
#         "central_wavelength_nm": 550,
#         "bandwidth_nm": 89,
#     },
#     "R": {
#         "system": "Vega",
#         "vega_zero_point_jy": 3080,
#         "central_wavelength_nm": 640,
#         "bandwidth_nm": 150,
#     },
#     "Rc": {
#         "system": "Vega",
#         "vega_zero_point_jy": 3080,
#         "central_wavelength_nm": 641,
#         "bandwidth_nm": 158,
#     },
#     "Ic": {
#         "system": "Vega",
#         "vega_zero_point_jy": 2550,
#         "central_wavelength_nm": 798,
#         "bandwidth_nm": 154,
#     },

#     # Near-IR (Vega)
#     "J": {
#         "system": "Vega",
#         "vega_zero_point_jy": 1594,
#         "central_wavelength_nm": 1235,
#         "bandwidth_nm": 162,
#     },
#     "L": {
#         "system": "Vega",
#         "vega_zero_point_jy": 280,
#         "central_wavelength_nm": 3450,
#         "bandwidth_nm": 570,
#     },
#     "u": {
#         "system": "AB",
#         "vega_zero_point_jy": None,
#         "central_wavelength_nm": 360.8,
#         "bandwidth_nm": 56.4,
#     },
#     "g": {
#         "system": "AB",
#         "vega_zero_point_jy": None,
#         "central_wavelength_nm": 467.1,
#         "bandwidth_nm": 106.5,
#     },
#     "g'": {
#         "system": "AB",
#         "vega_zero_point_jy": None,
#         "central_wavelength_nm": 472.3,
#         "bandwidth_nm": 126.5,
#     },
#     "r": {
#         "system": "AB",
#         "vega_zero_point_jy": None,
#         "central_wavelength_nm": 614.1,
#         "bandwidth_nm": 105.5,
#     },
#     "r'": {
#         "system": "AB",
#         "vega_zero_point_jy": None,
#         "central_wavelength_nm": 620.2,
#         "bandwidth_nm": 125.4,
#     },
#     "i": {
#         "system": "AB",
#         "vega_zero_point_jy": None,
#         "central_wavelength_nm": 745.8,
#         "bandwidth_nm": 110.3,
#     },
#     "z": {
#         "system": "AB",
#         "vega_zero_point_jy": None,
#         "central_wavelength_nm": 892.3,
#         "bandwidth_nm": 116.4,
#     },
#     # Unfiltered / instrumental
#     "Clear": {
#         "system": "instrumental",
#         "vega_zero_point_jy": None,
#         "central_wavelength_nm": 0,
#         "bandwidth_nm": 0,
#     },
#     "clear": {
#         "system": "Vega",
#         "vega_zero_point_jy": 3064,
#         "central_wavelength_nm": 0,
#         "bandwidth_nm": 0,
#     },
#     "white": {
#         "system": "instrumental",
#         "vega_zero_point_jy": None,
#         "central_wavelength_nm": 0,
#         "bandwidth_nm": 0,
#     },
#     "w": {
#         "system": "instrumental",
#         "vega_zero_point_jy": None,
#         "central_wavelength_nm": 0,
#         "bandwidth_nm": 0,
#     },
# }
