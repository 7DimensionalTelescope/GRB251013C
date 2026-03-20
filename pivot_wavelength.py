import numpy as np
import pandas as pd

def calculate_pivot_wavelength(wavelength, qe):
    """
    Calculates the pivot wavelength given wavelength and QE arrays.
    Formula: sqrt( (integral lambda * QE d_lambda) / (integral (1/lambda) * QE d_lambda) )
    """
    # Numerator: integral of (lambda * QE)
    numerator = np.trapezoid(wavelength * qe, wavelength)
    
    # Denominator: integral of (QE / lambda)
    denominator = np.trapezoid(qe / wavelength, wavelength)
    
    pivot_lambda = np.sqrt(numerator / denominator)
    return pivot_lambda

def calculate_effective_width(wavelength, qe):
    """
    Calculates the effective width of the detector QE or filter.
    Formula: integral(QE d_lambda) / max(QE)
    """
    total_area = np.trapezoid(qe, wavelength)
    return total_area / np.max(qe)

# Load your QE data
# Assuming your file is a CSV with columns 'wavelength' and 'qe'
# Replace 'camera_qe.csv' with your actual file name
data = pd.read_csv('/home/hongyp007/hongyp/GRB251013C/GRB_fitting/data/Sony_ICS285AL_QE_calapai.csv', names=['wavelength', 'qe'])

# Ensure wavelength is in Angstroms or nm consistently
lam = data['wavelength'].values
qe = data['qe'].values

lp = calculate_pivot_wavelength(lam, qe)
w_eff = calculate_effective_width(lam, qe)

print(f"The Pivot Wavelength (λp) is: {lp:.3f} nm / {lp*10:.2f} Å")
print(f"The Effective Width (W_eff) is: {w_eff:.3f} nm / {w_eff*10:.2f} Å")