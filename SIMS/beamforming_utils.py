"""
Beamforming Utility Functions

This module provides common functions for ultrasound beamforming simulations,
including element directivity and tissue attenuation calculations.

References:
    Jensen, "Field II: A program for simulating ultrasound systems", 1996.
"""

import numpy as np
from scipy.special import j1

def element_directivity(theta, width, wavelength, use_piston_model=True, apply_obliquity=True):
    """
    Calculate the element directivity pattern.

    Args:
        theta (ndarray): Angle from element normal in radians.
        width (float): Element width (meters).
        wavelength (float): Current wavelength (meters).
        use_piston_model (bool): If True, use the piston model (default: True).
        apply_obliquity (bool): If True, applies a cosine obliquity factor to account for off-axis attenuation (default: True).

    Returns:
        ndarray: Directivity factor (dimensionless).
    """
    # Precompute sine and cosine of theta for efficiency.
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    if use_piston_model:
        a = width / 2.0  # effective radius of the element
        k = 2 * np.pi / wavelength
        arg = k * a * sin_theta
        tol = 1e-5
        # Use series expansion for small arguments (2*j1(x)/x ≈ 1 - x^2/8 for small x).
        directivity = np.where(np.abs(arg) < tol, 1 - (arg**2)/8.0, 2 * j1(arg) / arg)
    else:
        directivity = np.sinc((width * sin_theta) / wavelength)

    if apply_obliquity:
        directivity *= cos_theta
    return directivity

def calculate_attenuation(distance, freq_MHz, attenuation_coeff=0.5, attenuation_exponent=1.0):
    """
    Calculate the tissue attenuation factor.

    Args:
        distance (ndarray): Path length (meters).
        freq_MHz (float): Frequency in MHz.
        attenuation_coeff (float): Attenuation coefficient in dB/cm/MHz (default: 0.5).
        attenuation_exponent (float): Frequency exponent (default: 1.0).

    Returns:
        ndarray: Attenuation factor (dimensionless) in linear scale.
    """
    # Convert distance from meters to centimeters.
    distance_cm = distance * 100
    attn_dB = attenuation_coeff * (freq_MHz ** attenuation_exponent) * distance_cm
    # Use logarithmic conversion for better precision: 10^(-dB/20) = exp(-dB * ln(10)/20)
    return np.exp(-attn_dB * np.log(10) / 20) 