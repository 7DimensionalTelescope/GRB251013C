import copy
import pandas as pd

import numpy as np

from .const import CIRCULAR_DATA_FILENAME, XRT_DATA_FILENAME

def read_data(filename):
    if filename == "circular":
        filename = CIRCULAR_DATA_FILENAME
        kwargs = {}
    elif filename == "xrt":
        filename = XRT_DATA_FILENAME
        kwargs = {
            "sep": "\t",
            "header": None,
            "names": ["Time", "Time_high", "Time_low", "Flux", "Flux_high", "Flux_low"]
        }

    if filename.endswith(".csv"):
        return pd.read_csv(filename, **kwargs)
    elif filename.endswith(".xlsx"):
        return pd.read_excel(filename, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {filename}")

def filter_data(df, filter_name=None, facility_name = None, exclude_time_range=None, remove_upper_limits=False):
    filtered_df = copy.deepcopy(df)
    if filter_name is not None:
        if isinstance(filter_name, list):
            filtered_df = filtered_df[filtered_df["Filter"].isin(filter_name)]
        else:
            filtered_df = filtered_df[filtered_df["Filter"] == filter_name]
    
    if facility_name is not None:
        if isinstance(facility_name, list):
            filtered_df = filtered_df[filtered_df["Facility"].isin(facility_name)]
        else:
            filtered_df = filtered_df[filtered_df["Facility"] == facility_name]

    if exclude_time_range is not None:
        filtered_df = filtered_df[~filtered_df["Time"].between(exclude_time_range[0], exclude_time_range[1])]

    if remove_upper_limits:
        _test = filtered_df["Mag"]
        upper_limits = np.array([">" in str(v) for v in _test])
        filtered_df = filtered_df[~upper_limits]

    return filtered_df
