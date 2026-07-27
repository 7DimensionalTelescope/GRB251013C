# Smooth Spectral Index Around Cooling Break

## Summary

Based on Granot & Sari 2001 (arXiv:astro-ph/0108027v1), spectral breaks are **smooth**, not sharp. The spectral index β changes continuously as a function of ν/ν_break, not as a step function.

## Key Results

### The Problem with Step Functions

Previously, you might have used:
```python
if nu < nu_c:
    beta = (p - 1) / 2  # Slow cooling
else:
    beta = p / 2         # Fast cooling
```

This is **wrong** when ν is near ν_c!

### The Smooth Transition

According to Granot & Sari Eq. (1), the flux near a break is:

```
F_ν = F_ν,break × [(ν/ν_break)^(-s·β₁) + (ν/ν_break)^(-s·β₂)]^(-1/s)
```

Where:
- β₁, β₂ = asymptotic spectral slopes below/above break
- s = sharpness parameter (depends on p, from Table 2)

For the **cooling break** (ν_c):
- β₁ = (p-1)/2 (below ν_c)
- β₂ = p/2 (above ν_c)  
- s(p) = 1.15 - 0.06·p

The **local spectral index** β(ν) = d(ln F)/d(ln ν) varies smoothly between β₁ and β₂.

### Error Magnitude

When ν/ν_c is between 0.5 and 2.0 (within factor of 2 of the break):
- **Error in β ≈ 0.2-0.25** if you use a step function
- This translates to **~20% error in inferring p**!

## Implementation for Fitting

### Option 1: Precomputed Interpolation Table (Recommended)

```python
from spectral_index_interpolator import SpectralIndexCalculator

# One-time initialization (builds lookup tables)
calc = SpectralIndexCalculator()

# During fitting, for each parameter set:
nu_obs = 3e17  # XRT center frequency ~1.2 keV
nu_m = ...     # Computed from model parameters
nu_c = ...     # Computed from model parameters  
p = ...        # Current p value in fit

# Get accurate spectral index accounting for smooth transition
beta = calc.beta_at_frequency(nu_obs, nu_m, nu_c, p)
```

**Advantages:**
- Fast (just interpolation, no integration)
- Accurate for all ν/ν_c ratios
- No need to assume "slow" or "fast" cooling regime

### Option 2: Direct Calculation

If you want to compute β(ν) directly without interpolation:

```python
from spectral_index_interpolator import (
    local_spectral_index, 
    sharpness_parameter_cooling
)

nu_ratio = nu_obs / nu_c
beta1_GS = (1 - p) / 2  # Granot & Sari notation
beta2_GS = -p / 2
s = sharpness_parameter_cooling(p)

beta_GS = local_spectral_index(nu_ratio, beta1_GS, beta2_GS, s)
beta = -beta_GS  # Convert to X-ray convention
```

## Notation Conventions

⚠️ **Important:** Granot & Sari use F_ν ∝ ν^β (β can be negative)

For X-ray spectroscopy, we use F_ν ∝ ν^(-β_X) where:
- β_X = Γ - 1 (Γ is photon index)
- β_X = -β_GS (conversion)

For cooling break at p=2.2:
- Below ν_c: β_X = (p-1)/2 = 0.6 (softer spectrum)
- Above ν_c: β_X = p/2 = 1.1 (harder spectrum)

## When Does This Matter?

Use smooth transitions when:

1. **XRT observations near break:** If ν_XRT/ν_c is between 0.3 and 3
2. **Fitting spectral index data:** When you have measurements of β(t) or Γ(t)
3. **Joint XRT + optical fits:** Different bands may be on different sides of ν_c

You can use step function when:
- ν ≪ ν_c (factor >10 below) 
- ν ≫ ν_c (factor >10 above)

## Example Values

For p = 2.2 and different ν/ν_c ratios:

| ν/ν_c | β (smooth) | β (step) | Error |
|-------|------------|----------|-------|
| 0.5   | 0.806      | 0.600    | 0.206 |
| 1.0   | 0.850      | 1.100    | -0.250|
| 2.0   | 0.894      | 1.100    | -0.206|

At the break (ν/ν_c = 1), the step function error is 0.25!

## Visualization

Run the demonstration:
```bash
python3 demo_smooth_spectral_index_standalone.py
```

This generates `smooth_spectral_index.png` showing:
1. How β varies for different p values
2. Comparison of smooth vs step function transitions
3. The transition region where errors are largest

## References

- Granot & Sari 2001: "The Shape of Spectral Breaks in GRB Afterglows"
  - arXiv:astro-ph/0108027v1
  - Equation (1): Smooth break formula
  - Table 1: Power law segments  
  - Table 2: Break frequencies and sharpness parameters

## Files

- `spectral_index_interpolator.py` - Main implementation with interpolation tables
- `demo_smooth_spectral_index_standalone.py` - Standalone demonstration
- `smooth_spectral_index.png` - Generated plot showing smooth transitions
- `utils.py` - Contains `compute_break_frequencies()` helper
