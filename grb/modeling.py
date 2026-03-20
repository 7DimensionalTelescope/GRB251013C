import pandas as pd
from .utils import mJy_to_erg_cm2_s_Hz
from VegasAfterglow import ObsData
from .utils import unit_conversion

def add_observation(df, obs_data=None, input_type="flux_density", **kwargs):
    if obs_data is None:
        obs_data = ObsData()
    elif not isinstance(obs_data, ObsData):
        raise ValueError("obs_data must be an instance of ObsData")
    
    if isinstance(df, pd.DataFrame):
        if input_type == "flux_density": 
            if not all(df["gal_corrected"]):
                raise ValueError("Galactic extinction not corrected")
            if not all(df["host_corrected"]):
                raise ValueError("Host galaxy extinction not corrected")

            add_function = obs_data.add_flux_density
            data = {
                "nu": df["frequency_Hz"].to_list()[0],
                "t": df["time"].to_list(),
                "f_nu": mJy_to_erg_cm2_s_Hz(df["flux_mJy"]).to_list(),
                "err": mJy_to_erg_cm2_s_Hz(df["flux_mJy_error"]).to_list()
            }
        elif input_type == "flux":
            add_function = obs_data.add_flux
            nu_min = kwargs.pop("nu_min", unit_conversion(0.3, "keV", "Hz"))
            nu_max = kwargs.pop("nu_max", unit_conversion(10, "keV", "Hz"))
            num_points=kwargs.pop("num_points", 5)

            if "flux_high" and "flux_low" in df.columns:
                df["flux_error"] = [max(f_max, f_min) for f_max, f_min in zip(df["flux_high"], df["flux_low"])]

            data = {
                "t": df["time"].to_list(),
                "flux": df["flux"].to_list(),
                "err": df["flux_error"].to_list(),
                "nu_min": nu_min,
                "nu_max": nu_max,
                "num_points": num_points
            }

        add_function(**data)
        return obs_data
    else:
        raise ValueError("Invalid input types")
