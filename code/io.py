import copy
import pandas as pd

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

def filter_data(df, filter_name=None, exclude_time_range=None, remove_upper_limits=False, invert_selection=False):
    filtered_df = copy.deepcopy(df)
    if filter_name is not None:
        if isinstance(filter_name, list):
            filtered_df = df[df["Filter"].isin(filter_name)]
        else:
            filtered_df = df[df["Filter"] == filter_name]

        if remove_upper_limits:
            filtered_df = filtered_df[filtered_df["Error"].map(type).ne(str)]

    if exclude_time_range is not None:
        filtered_df = filtered_df[~filtered_df["Time"].between(exclude_time_range[0], exclude_time_range[1])]

    return filtered_df
