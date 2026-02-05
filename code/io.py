import copy
import pandas as pd

def read_data(filename, correct_galactic_extinction=False):
    if filename == "circular":
        from .const import CIRCULAR_DATA_FILENAME
        filename = CIRCULAR_DATA_FILENAME
    elif filename == "xrt":
        from .const import XRT_DATA_FILENAME
        filename = XRT_DATA_FILENAME
        kwargs = {
            "sep": "\t",
            "header": None,
            "names": ["Time", "Time_high", "Time_low", "Flux", "Flux_high", "Flux_low"]
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
    
    if correct_galactic_extinction:
        from .extinction import correct_galactic_extinction
        df = correct_galactic_extinction(df)
    
    return df

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
        upper_limits = filtered_df["Upper Limit"]
        filtered_df = filtered_df[~upper_limits]

    return filtered_df

