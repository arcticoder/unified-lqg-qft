"""
Smearing Functions for ANEC Integrals

This module provides temporal and spatial smearing functions used in
Averaged Null Energy Condition (ANEC) calculations. Includes Ford-Roman
sampling functions, Gaussian smearing kernels, and specialized functions
for quantum inequality bounds.

Key Functions:
- Ford-Roman Gaussian sampling
- Exponential and polynomial smearing kernels  
- Spectral smearing for frequency-domain analysis
- Polymer-modified smearing with LQG corrections
"""

import numpy as np
from typing import Union, Optional, Callable
from scipy.special import erf
import warnings


def ford_roman_sampling_function(t: Union[float, np.ndarray], 
                                tau: float) -> Union[float, np.ndarray]:
    """
    Ford-Roman Gaussian sampling function for ANEC integrals.
    
    f(t) = exp(-t²/2τ²) / (√(2π) τ)
    
    Args:
        t: Time coordinate(s)
        tau: Sampling timescale
        
    Returns:
        Sampling function value(s)
    """
    if tau <= 0:
        raise ValueError("Sampling timescale τ must be positive")
    
    normalization = 1.0 / (np.sqrt(2 * np.pi) * tau)
    return normalization * np.exp(-t**2 / (2 * tau**2))


def exponential_smearing(t: Union[float, np.ndarray], 
                        tau: float, 
                        power: float = 1.0) -> Union[float, np.ndarray]:
    """
    Exponential smearing function with adjustable decay rate.
    
    f(t) = exp(-|t|^power / τ^power) / (2τΓ(1/power))
    
    Args:
        t: Time coordinate(s)
        tau: Characteristic timescale
        power: Decay exponent (default: 1.0 for standard exponential)
        
    Returns:
        Smearing function value(s)
    """
    if tau <= 0:
        raise ValueError("Timescale τ must be positive")
    if power <= 0:
        raise ValueError("Power must be positive")
    
    from scipy.special import gamma
    normalization = 1.0 / (2 * tau * gamma(1.0 / power))
    return normalization * np.exp(-np.abs(t)**power / tau**power)


def polynomial_cutoff_smearing(t: Union[float, np.ndarray], 
                              tau: float, 
                              n: int = 4) -> Union[float, np.ndarray]:
    """
    Polynomial cutoff smearing with compact support.
    
    f(t) = N * (1 - t²/τ²)^n for |t| < τ, 0 otherwise
    
    Args:
        t: Time coordinate(s)
        tau: Cutoff timescale
        n: Polynomial order (higher n → smoother cutoff)
        
    Returns:
        Smearing function value(s)
    """
    if tau <= 0:
        raise ValueError("Cutoff timescale τ must be positive")
    if n < 1:
        raise ValueError("Polynomial order n must be >= 1")
    
    # Normalization constant
    from scipy.special import beta
    normalization = (2 * n + 1) / (2 * tau * beta(0.5, n + 1))
    
    t_array = np.asarray(t)
    result = np.zeros_like(t_array)
    mask = np.abs(t_array) < tau
    
    result[mask] = normalization * (1 - (t_array[mask] / tau)**2)**n
    
    return result if isinstance(t, np.ndarray) else float(result)


def sinc_smearing(t: Union[float, np.ndarray], 
                 tau: float) -> Union[float, np.ndarray]:
    """
    Sinc function smearing kernel.
    
    f(t) = sin(π t/τ) / (π t) for t ≠ 0, f(0) = 1/τ
    
    Args:
        t: Time coordinate(s)
        tau: Characteristic timescale
        
    Returns:
        Smearing function value(s)
    """
    if tau <= 0:
        raise ValueError("Timescale τ must be positive")
    
    t_array = np.asarray(t)
    result = np.zeros_like(t_array, dtype=float)
    
    # Handle t = 0 case
    zero_mask = (t_array == 0)
    result[zero_mask] = 1.0 / tau
    
    # Handle t ≠ 0 case
    nonzero_mask = ~zero_mask
    arg = np.pi * t_array[nonzero_mask] / tau
    result[nonzero_mask] = np.sin(arg) / (np.pi * t_array[nonzero_mask])
    
    return result if isinstance(t, np.ndarray) else float(result)


def polymer_modified_sampling(t: Union[float, np.ndarray], 
                             tau: float, 
                             mu: float = 0.1) -> Union[float, np.ndarray]:
    """
    Polymer-modified sampling function with LQG corrections.
    
    Includes sinc(π μ t) modulation from polymer quantization.
    
    Args:
        t: Time coordinate(s)
        tau: Gaussian width parameter
        mu: Polymer correction parameter
        
    Returns:
        Modified sampling function value(s)
    """
    if tau <= 0:
        raise ValueError("Sampling timescale τ must be positive")
    if mu < 0:
        raise ValueError("Polymer parameter μ must be non-negative")
    
    # Base Ford-Roman function
    base_function = ford_roman_sampling_function(t, tau)
    
    if mu == 0:
        return base_function
    
    # Polymer correction via sinc function
    t_array = np.asarray(t)
    polymer_arg = np.pi * mu * t_array
    
    # Handle sinc function properly
    sinc_factor = np.ones_like(t_array, dtype=float)
    nonzero_mask = (polymer_arg != 0)
    sinc_factor[nonzero_mask] = np.sin(polymer_arg[nonzero_mask]) / polymer_arg[nonzero_mask]
    
    return base_function * sinc_factor


def spectral_smearing_kernel(omega: Union[float, np.ndarray], 
                           omega_c: float, 
                           kernel_type: str = "lorentzian") -> Union[float, np.ndarray]:
    """
    Frequency-domain smearing kernels for spectral analysis.
    
    Args:
        omega: Frequency coordinate(s)
        omega_c: Characteristic frequency scale
        kernel_type: Type of kernel ("lorentzian", "gaussian", "exponential")
        
    Returns:
        Spectral kernel value(s)
    """
    if omega_c <= 0:
        raise ValueError("Characteristic frequency ω_c must be positive")
    
    omega_array = np.asarray(omega)
    
    if kernel_type == "lorentzian":
        return (omega_c / np.pi) / (omega_array**2 + omega_c**2)
    
    elif kernel_type == "gaussian":
        return (1 / (np.sqrt(2 * np.pi) * omega_c)) * np.exp(-omega_array**2 / (2 * omega_c**2))
    
    elif kernel_type == "exponential":
        return (omega_c / 2) * np.exp(-omega_c * np.abs(omega_array))
    
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def week_scale_sampling_kernel(t: Union[float, np.ndarray], 
                              tau_week: float = 7 * 24 * 3600) -> Union[float, np.ndarray]:
    """
    Week-scale sampling kernel for macroscopic ANEC bounds.
    
    Specialized for τ ~ 1 week scale used in advanced QI circumvention.
    
    Args:
        t: Time coordinate(s) in seconds
        tau_week: Week timescale in seconds (default: 1 week)
        
    Returns:
        Sampling kernel value(s)
    """
    if tau_week <= 0:
        raise ValueError("Week timescale must be positive")
    
    # Use Gaussian with exponential wings for better behavior
    t_array = np.asarray(t)
    tau_ratio = t_array / tau_week
    
    # Gaussian core for |t| < 2τ
    gaussian_part = np.exp(-tau_ratio**2 / 2)
    
    # Exponential wings for |t| > 2τ
    wing_mask = np.abs(tau_ratio) > 2
    exponential_part = np.exp(-2 + 2 * np.abs(tau_ratio[wing_mask]))
    
    result = gaussian_part.copy()
    result[wing_mask] = exponential_part
    
    # Normalization
    normalization = 1.0 / (np.sqrt(2 * np.pi) * tau_week)
    
    return normalization * result


def compute_smearing_integral(stress_tensor: np.ndarray,
                             t_grid: np.ndarray,
                             x_grid: np.ndarray,
                             smearing_function: Callable,
                             **smearing_params) -> float:
    """
    Compute smeared integral of stress tensor.
    
    ∫ dt dx f(t) T_00(t,x)
    
    Args:
        stress_tensor: Stress tensor values T_00(t,x)
        t_grid: Time grid
        x_grid: Spatial grid  
        smearing_function: Smearing function f(t)
        **smearing_params: Parameters for smearing function
        
    Returns:
        Smeared integral value
    """
    if stress_tensor.shape != (len(t_grid), len(x_grid)):
        raise ValueError("Stress tensor shape must match grid dimensions")
    
    # Compute smearing weights
    smearing_weights = smearing_function(t_grid, **smearing_params)
    
    # Integrate over space first, then time
    spatial_integrals = np.trapz(stress_tensor, x_grid, axis=1)
    temporal_integral = np.trapz(spatial_integrals * smearing_weights, t_grid)
    
    return temporal_integral


def anec_integral_with_smearing(null_geodesic: np.ndarray,
                               stress_tensor: np.ndarray,
                               smearing_function: Callable,
                               **smearing_params) -> float:
    """
    Compute ANEC integral along null geodesic with temporal smearing.
    
    Args:
        null_geodesic: Array of points along null geodesic
        stress_tensor: Stress tensor field
        smearing_function: Temporal smearing function
        **smearing_params: Parameters for smearing function
        
    Returns:
        Smeared ANEC integral value
    """
    # Extract parameter λ along geodesic
    lambda_values = np.linspace(0, 1, len(null_geodesic))
    
    # Evaluate stress tensor along geodesic
    stress_along_geodesic = []
    for point in null_geodesic:
        # This would need to be interpolated from the full stress tensor field
        # For now, use a simplified approach
        stress_value = np.sum(stress_tensor)  # Placeholder
        stress_along_geodesic.append(stress_value)
    
    stress_along_geodesic = np.array(stress_along_geodesic)
    
    # Apply temporal smearing
    smearing_weights = smearing_function(lambda_values, **smearing_params)
    
    # Compute smeared integral
    return np.trapz(stress_along_geodesic * smearing_weights, lambda_values)


# Convenience functions for common use cases
def compute_ford_roman_bound(tau: float) -> float:
    """
    Compute classical Ford-Roman bound for given timescale.
    
    Bound = -ħc/(32π²τ²) in natural units (ħ = c = 1)
    
    Args:
        tau: Sampling timescale
        
    Returns:
        Ford-Roman bound (negative value)
    """
    if tau <= 0:
        raise ValueError("Timescale τ must be positive")
    
    return -1.0 / (32 * np.pi**2 * tau**2)


def optimal_sampling_timescale(energy_scale: float, 
                              violation_target: float = 0.1) -> float:
    """
    Estimate optimal sampling timescale for desired violation magnitude.
    
    Args:
        energy_scale: Characteristic energy scale of the system
        violation_target: Target violation magnitude (dimensionless)
        
    Returns:
        Optimal sampling timescale
    """
    if energy_scale <= 0:
        raise ValueError("Energy scale must be positive")
    if violation_target <= 0:
        raise ValueError("Violation target must be positive")
    
    # Heuristic based on quantum inequality scaling
    return np.sqrt(violation_target) / (4 * np.pi * energy_scale)


# Export main functions
__all__ = [
    'ford_roman_sampling_function',
    'exponential_smearing', 
    'polynomial_cutoff_smearing',
    'sinc_smearing',
    'polymer_modified_sampling',
    'spectral_smearing_kernel',
    'week_scale_sampling_kernel',
    'compute_smearing_integral',
    'anec_integral_with_smearing',
    'compute_ford_roman_bound',
    'optimal_sampling_timescale'
]
