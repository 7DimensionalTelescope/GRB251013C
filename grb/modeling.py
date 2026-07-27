"""Afterglow model construction and observational data assembly.

Builds the VegasAfterglow core-jet (with optional reverse shock) and wing-jet
models, and loads every XRT and optical dataset the fit uses.
"""
import numpy as np

from VegasAfterglow import ISM, Model, Observer, Radiation, TophatJet

from .const import D_L, REDSHIFT, MODEL_RESOLUTIONS
from .io import read_data, filter_data
from .utils import flux_error, seconds_from_trigger


def load_all_optical_data():
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


def make_wing_model(params, spreading=True):
    """Wing jet (no reverse shock).

    spreading=True (default) enables lateral spreading to maintain flux at late
    times, which is what the final model wants. partial_data.py fits the wing
    without a jet break and passes spreading=False.
    """
    observer = Observer(lumi_dist=D_L, z=REDSHIFT, theta_obs=0)
    medium = ISM(n_ism=params["n_ism"])
    jet = TophatJet(
        E_iso=params["E_iso_wing"],
        Gamma0=params["Gamma0_wing"],
        theta_c=params["theta_c_wing"],
        spreading=spreading,
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

