import pandas as pd
import numpy as np
from .utils import mJy_to_erg_cm2_s_Hz
from VegasAfterglow import ObsData, Model, Observer, ISM, TophatJet, Radiation, ParamDef, Scale
from VegasAfterglow.units import keV


from .utils import unit_conversion, flux_error, seconds_from_trigger
from .io import read_data, filter_data

from .const import D_L, REDSHIFT, MODEL_RESOLUTIONS

def load_all_data():

    """Load ALL optical and XRT data (following late_phase.py structure)"""
    # XRT data
    xrt_data = read_data("xrt")
    xrt_dict = {
        'time': xrt_data['time'].to_numpy(float),
        'flux': xrt_data['flux'].to_numpy(float),
        'flux_error': flux_error(xrt_data),
    }
    
    optical_datasets = []
    
    # 1. i-band data (primary)
    i_data = read_data("i_data", correct_galactic_extinction=True, add_converted_flux=True)
    optical_datasets.append({
        'name': 'i-band',
        'frequency': float(i_data['frequency_Hz'].iloc[0]),
        'time': i_data['time'].to_numpy(float),
        'flux_mJy': i_data['flux_mJy'].to_numpy(float),
        'flux_err': i_data['flux_mJy_error'].to_numpy(float),
    })
    
    # 2. Leavitt Rc and Ic data
    circular = read_data("circular", correct_galactic_extinction=True, add_converted_flux=True)
    for filter_name in ("Rc", "Ic"):
        data = filter_data(circular, filter_name=filter_name, 
                          facility_name="Leavitt", remove_upper_limits=True)
        if len(data) > 0:
            optical_datasets.append({
                'name': f'Leavitt_{filter_name}',
                'frequency': float(data['frequency_Hz'].iloc[0]),
                'time': data['time'].to_numpy(float),
                'flux_mJy': data['flux_mJy'].to_numpy(float),
                'flux_err': data['flux_mJy_error'].to_numpy(float),
            })
    
    # 3. SDT/7DT data - Each filter is a separate dataset!
    sdt_data = read_data("sdt", correct_galactic_extinction=True, add_converted_flux=True)
    sdt_data = sdt_data[~sdt_data["is_upper_limit"].astype(bool)].copy()
    for _, row in sdt_data.iterrows():
        optical_datasets.append({
            'name': f'7DT_{row["filter_name"]}',
            'frequency': float(row["frequency_Hz"]),
            'time': np.array([seconds_from_trigger(row["date_obs"])]),
            'flux_mJy': np.array([float(row["flux_mJy"])]),
            'flux_err': np.array([float(row["flux_mJy_error"])]),
        })
    
    return xrt_dict, optical_datasets

# def add_observation(df, obs_data=None, input_type="flux_density", **kwargs):
#     if obs_data is None:
#         obs_data = ObsData()
#     elif not isinstance(obs_data, ObsData):
#         raise ValueError("obs_data must be an instance of ObsData")
    
#     if isinstance(df, pd.DataFrame):
#         if input_type == "flux_density": 
#             if not all(df["gal_corrected"]):
#                 raise ValueError("Galactic extinction not corrected")
#             if not all(df["host_corrected"]):
#                 raise ValueError("Host galaxy extinction not corrected")

#             add_function = obs_data.add_flux_density
#             data = {
#                 "nu": df["frequency_Hz"].to_list()[0],
#                 "t": df["time"].to_list(),
#                 "f_nu": mJy_to_erg_cm2_s_Hz(df["flux_mJy"]).to_list(),
#                 "err": mJy_to_erg_cm2_s_Hz(df["flux_mJy_error"]).to_list()
#             }
#         elif input_type == "flux":
#             add_function = obs_data.add_flux
#             nu_min = kwargs.pop("nu_min", unit_conversion(0.3, "keV", "Hz"))
#             nu_max = kwargs.pop("nu_max", unit_conversion(10, "keV", "Hz"))
#             num_points=kwargs.pop("num_points", 5)

#             if "flux_high" and "flux_low" in df.columns:
#                 df["flux_error"] = [max(f_max, f_min) for f_max, f_min in zip(df["flux_high"], df["flux_low"])]

#             data = {
#                 "t": df["time"].to_list(),
#                 "flux": df["flux"].to_list(),
#                 "err": df["flux_error"].to_list(),
#                 "nu_min": nu_min,
#                 "nu_max": nu_max,
#                 "num_points": num_points
#             }

#         add_function(**data)
#         return obs_data
#     else:
#         raise ValueError("Invalid input types")


def make_core_model(params):
    """Core jet with reverse shock"""
    observer = Observer(lumi_dist=D_L, z=REDSHIFT, theta_obs=0)
    medium = ISM(n_ism=params["n_ism"])
    jet = TophatJet(
        E_iso=params["E_iso_core"],
        Gamma0=params["Gamma0_core"],
        theta_c=params["theta_c_core"],
        spreading=True,
        duration=params.get("tau", 10.0),
    )
    fwd_radiation = Radiation(
        eps_e=params["eps_e"],
        eps_B=params["eps_B"],
        p=params["p"],
        xi_e=params["xi"],
        ssc=False,
        kn=False,
    )
    
    rvs_radiation = None
    
    if "p_r" in params and "eps_e_r" in params and "eps_B_r" in params:
        rvs_radiation = Radiation(
            eps_e=params["eps_e_r"],
            eps_B=params["eps_B_r"],
            p=params["p_r"],
            xi_e=params.get("xi_r", params["xi"]),
            ssc=False,
            kn=False,
        )
    
    return Model(jet=jet, medium=medium, observer=observer, 
                 fwd_rad=fwd_radiation, rvs_rad=rvs_radiation, 
                 resolutions=MODEL_RESOLUTIONS)


def make_wing_model(params):
    """Wing jet (no reverse shock, with spreading for late-time emission)"""
    observer = Observer(lumi_dist=D_L, z=REDSHIFT, theta_obs=0)
    medium = ISM(n_ism=params["n_ism"])
    jet = TophatJet(
        E_iso=params["E_iso_wing"],
        Gamma0=params["Gamma0_wing"],
        theta_c=params["theta_c_wing"],
        spreading=True,  # Enable spreading to maintain flux at late times
        duration=params.get("tau", 10.0),
    )
    radiation = Radiation(
        eps_e=params.get("eps_e_wing", params["eps_e"]),
        eps_B=params.get("eps_B_wing", params["eps_B"]),
        p=params.get("p_wing", params["p"]),
        xi_e=params.get("xi_wing", params["xi"]),
        ssc=False,
        kn=False,
    )
    return Model(jet=jet, medium=medium, observer=observer, 
                 fwd_rad=radiation, resolutions=MODEL_RESOLUTIONS)

