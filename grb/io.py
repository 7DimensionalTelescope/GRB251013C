import copy
import pandas as pd
import warnings

warnings.simplefilter("ignore", RuntimeWarning)

def read_data(filename, correct_galactic_extinction=False, correct_host_extinction=False, host_av=0.0, host_z=0.0, add_converted_flux=False, **kwargs) -> pd.DataFrame:
    if filename == "circular":
        from .const import CIRCULAR_DATA_FILENAME
        filename = CIRCULAR_DATA_FILENAME
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
    elif filename == "xrt_unabsorb":
        from .const import XRT_UNABSORB_DATA_FILENAME
        filename = XRT_UNABSORB_DATA_FILENAME
        kwargs = {
            "sep": ",",
        }
    elif filename == "sdt":
        from .const import SDT_DATA_FILENAME
        filename = SDT_DATA_FILENAME
        kwargs = {
            "sep": ",",
        }
    elif filename == "sdt_pivot":
        from .const import SDT_PIVOT_DATA_FILENAME
        filename = SDT_PIVOT_DATA_FILENAME
        kwargs = {
            "sep": ",",
        }
    elif filename == "sdt_tractor":
        from .const import SDT_TRACER_DATA_FILENAME
        filename = SDT_TRACER_DATA_FILENAME
        kwargs = {
            "sep": ",",
        }
    elif filename == "circular_wavelength":
        from .const import CIRCULAR_WAVELENGTH_DATA_FILENAME
        filename = CIRCULAR_WAVELENGTH_DATA_FILENAME
        kwargs = {
            "sep": ",",
        }
    if filename.endswith(".csv"):
        df = pd.read_csv(filename, **kwargs)
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(filename, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {filename}")
    
    if "filter" in df.columns and "wavelength" not in df.columns:
        from .utils import filter_to_wavelength, filter_width
        df["wavelength"] = filter_to_wavelength(df["filter"])
        df["filter_width"] = filter_width(df["filter"])

    if correct_galactic_extinction:
        from .extinction import correct_galactic_extinction
        df = correct_galactic_extinction(df)
    
    if correct_host_extinction:
        from .extinction import correct_host_galaxy_extinction
        df = correct_host_galaxy_extinction(df, Av=host_av, z=host_z)
    
    if add_converted_flux:
        from .utils import mag_to_flux_mJy
        df = mag_to_flux_mJy(df)
    
    return df

def filter_data(df, filter_name=None, facility_name = None, exclude_time_range=None, remove_upper_limits=False):
    """
    Filter data by filter name, facility name, exclude time range, and remove upper limits
    
    Args:
        df: DataFrame to filter
        filter_name: Filter name to filter by
        facility_name: Facility name to filter by
        exclude_time_range: Time range to exclude
        remove_upper_limits: Remove upper limits
    
    Returns:
        Filtered DataFrame
    """
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

