import copy
import pandas as pd
import warnings
import numpy as np

warnings.simplefilter("ignore", RuntimeWarning)

def read_data(filename, correct_galactic_extinction=False, add_converted_flux=False, **kwargs):
    normalize_i_data = False
    if filename == "circular":
        from .const import CIRCULAR_DATA_FILENAME
        filename = CIRCULAR_DATA_FILENAME
        kwargs = {}
    elif filename == "i_data":
        # New schema: mjd, mag, magerr, filter, instrument_name
        from .const import I_DATA_FILENAME
        filename = I_DATA_FILENAME
        normalize_i_data = True
        kwargs = {}
    elif filename == "xrt":
        from .const import XRT_DATA_FILENAME
        filename = XRT_DATA_FILENAME
        kwargs = {
            "sep": "\t",
            "header": None,
            "names": ["time", "time_high", "time_low", "flux", "flux_high", "flux_low"]
        }
    elif filename == "xrt_index":
        from .const import XRT_INDEX_DATA_FILENAME
        filename = XRT_INDEX_DATA_FILENAME
        kwargs = {
            "sep": ",",
            "header": None,
            "names": ["time", "time_high", "time_low", "index", "index_high", "index_low"]
        }
    elif filename == "sdt":
        from .const import SDT_DATA_FILENAME
        filename = SDT_DATA_FILENAME
        kwargs = {
            "sep": ",",
        }
        
    if filename.endswith(".csv"):
        df = pd.read_csv(filename, **kwargs)
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(filename, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {filename}")

    if normalize_i_data:
        from .utils import mjd_to_seconds_from_trigger, normalize_filter_name
        df = df.rename(columns={
            "mag": "magnitude",
            "magerr": "mag_error",
            "instrument_name": "facility",
        })
        if "facility" in df.columns:
            df["facility"] = df["facility"].astype(str).str.strip()
        df["filter"] = df["filter"].map(normalize_filter_name)
        df["time"] = mjd_to_seconds_from_trigger(df["mjd"].to_numpy(float))

    if "filter" in df.columns and "wavelength" not in df.columns:
        from .utils import filter_to_wavelength, filter_width
        df["wavelength"] = filter_to_wavelength(df["filter"])
        df["filter_width"] = filter_width(df["filter"])

    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)

    if correct_galactic_extinction:
        from .extinction import correct_galactic_extinction
        df = correct_galactic_extinction(df)

    if add_converted_flux:
        from .utils import mag_to_flux_mJy
        df = mag_to_flux_mJy(df)
    
    return df

def filter_data(df, filter_name=None, facility_name = None, exclude_time_range=None, remove_upper_limits=False):
    filtered_df = copy.deepcopy(df)
    if filter_name is not None:
        if isinstance(filter_name, list):
            filtered_df = filtered_df[filtered_df["filter"].isin(filter_name)]
        else:
            filtered_df = filtered_df[filtered_df["filter"] == filter_name]
    
    if facility_name is not None:
        if isinstance(facility_name, list):
            filtered_df = filtered_df[filtered_df["facility"].isin(facility_name)]
        else:
            filtered_df = filtered_df[filtered_df["facility"] == facility_name]

    if exclude_time_range is not None:
        filtered_df = filtered_df[~filtered_df["time"].between(exclude_time_range[0], exclude_time_range[1])]

    if remove_upper_limits:
        upper_limits = filtered_df["upper_limit"]
        filtered_df = filtered_df[~upper_limits]

    return filtered_df
