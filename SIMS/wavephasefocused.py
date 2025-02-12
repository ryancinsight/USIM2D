import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1  # Requires scipy for Bessel functions
from matplotlib.colors import TwoSlopeNorm
from beamforming_utils import element_directivity, calculate_attenuation
from viz import plot_panels
import os
from scipy.ndimage import gaussian_filter

"""
Physics of Ultrasound Beamforming Simulation

This script simulates ultrasound beamforming using improved physical modeling based on recent literature.

1. Wave Propagation:
   - The simulation solves the wave equation in a homogeneous medium using linear acoustics.
   - Speed of sound is set to 1540 m/s to represent soft tissue conditions.
   - Amplitude decay due to tissue attenuation and diffraction (1/r spherical spreading) is included.

2. Phased Array Principles:
   - The transducer is modeled as a finite-size piston array with element spacing of λ/2.
   - The piston model (using a Bessel function) is employed to compute each element's directivity.

3. Delay-and-Sum Beamforming:
   - The time delay is computed as the difference between the actual propagation distance and the focal distance, divided by the speed of sound.
   - The phase factor is obtained via exp(1j * 2πf·time_delay), ensuring accurate modeling of the propagation delay.
   - Apodization (e.g., Hanning window) is applied to reduce sidelobe levels.

4. Field Calculation:
   - Pressure field is computed as the coherent summation of the weighted contributions from all elements.
   - Geometric spreading, tissue attenuation, and the improved phase delay are applied to each element's contribution.
   - Optional field normalization is available to scale the computed pressure field to match a specified source pressure.

Modeling Improvements:
   - Updated phase delay computation using the actual time-of-flight difference.
   - Adjusted speed of sound to 1540 m/s for soft tissue, in line with current literature.
   - The simulation follows advanced beamforming strategies outlined in Field II and similar research.
"""

# Constants
frequency = 180e3  # Frequency in Hz
speed_of_sound = 1540  # Speed of sound in m/s for soft tissue (updated per current literature)
wavelength = speed_of_sound / frequency  # Wavelength in meters
element_spacing = wavelength / 2  # Element spacing (λ/2) to avoid grating lobes
element_width = wavelength / 2  # Element width (typical for PZT elements)
num_elements = 22  # Number of PZT elements
focal_distance = 0.05  # Focal distance in meters (50 mm)
attenuation_coeff = 0.5  # Attenuation coefficient in dB/cm/MHz
attenuation_exponent = 1.0  # Frequency exponent for attenuation (typically 1.0)
spreading_model = "spherical"  # Options: "spherical" (1/r) or "cylindrical" (1/√r); default is spherical.

source_pressure = 0.8e6  # Source pressure amplitude in Pascals (0.8 MPa)

# Define x-axis range (±50 mm), high resolution
x_range = np.linspace(-0.05, 0.05, 1500)
# Define z-axis range (0 to 100 mm), high resolution
z_range = np.linspace(0, 0.1, 1500)

# Create a meshgrid for x and z coordinates
x, z = np.meshgrid(x_range, z_range)

# Calculate the positions of the PZT elements along the x-axis
element_positions = np.linspace(
    -(num_elements - 1) * element_spacing / 2,
    (num_elements - 1) * element_spacing / 2,
    num_elements
)

# Initialize the pressure field
pressure_field = np.zeros_like(x)

# Precompute frequency in MHz for attenuation calculation
freq_MHz = frequency / 1e6

# Beamforming Improvements:
#
# 1. Apodization:
#    The simulation now supports selectable apodization types to optimize the beamforming
#    performance (sidelobe reduction). Options include "hanning", "hamming", and "rectangular".
#
# 2. Complex Summation:
#    A complex phasor sum is computed so that the resultant pressure field is the magnitude of the
#    coherent sum of contributions. This more accurately captures the interference effects.

apodization_type = "hanning"  # Options: "hanning", "hamming", "rectangular", "chebyshev"
if apodization_type == "hanning":
    weights = np.hanning(num_elements)
elif apodization_type == "hamming":
    weights = np.hamming(num_elements)
elif apodization_type == "rectangular":
    weights = np.ones(num_elements)
else:
    # Default (or chebyshev not yet implemented)
    weights = np.ones(num_elements)

# Option to use complex summation for beamforming
use_complex = True

# -------------------------------
# Beamforming Computation
# -------------------------------
# Reshape the weights and element positions for broadcasting over the grid
weights_3d = weights.reshape(num_elements, 1, 1)
element_positions_3d = element_positions.reshape(num_elements, 1, 1)

# Expand the spatial grid to 3D (element, x, z)
x3 = x[np.newaxis, :, :]  # shape: (1, M, N)
z3 = z[np.newaxis, :, :]  # shape: (1, M, N)

# Compute the difference between each element's x-position and every grid point
delta_x = x3 - element_positions_3d                  # shape: (num_elements, M, N)
# Compute the Euclidean distance from each element to each point in the grid
distance_to_point = np.sqrt(delta_x**2 + z3**2)        # shape: (num_elements, M, N)
# Compute the distance from each element to the focal point (for delay correction).
r_focal = np.sqrt(element_positions_3d**2 + focal_distance**2)  # shape: (num_elements, 1, 1)

# Compute the angle between the element's normal and the point on the grid
theta = np.arctan2(delta_x, z3)                        # shape: (num_elements, M, N)

# Calculate the directivity pattern for each element over the grid
directivity = element_directivity(theta, element_width, wavelength)  # shape: (num_elements, M, N)

# Calculate tissue attenuation over the grid for each element
attenuation = calculate_attenuation(distance_to_point, freq_MHz, attenuation_coeff, attenuation_exponent)  # shape: (num_elements, M, N)

# --- Improved Phase Computation for Accurate Physics ---
# Instead of directly computing the phase as:
#     phase_const = 2 * np.pi * frequency / speed_of_sound
#     complex_field = exp(1j * phase_const * (distance_to_point - r_focal))
# we now explicitly compute the propagation time delay for each element:
time_delay = (distance_to_point - r_focal) / speed_of_sound  # Time delay in seconds

# Compute the phase factor using the time delay. This factor captures the propagation delay in the medium.
if use_complex:
    # Complex summation preserves the full phase information
    complex_field = np.exp(1j * 2 * np.pi * frequency * time_delay)
else:
    # Alternatively, a real-valued beamforming can use the cosine of the phase delay
    complex_field = np.cos(2 * np.pi * frequency * time_delay)
# --------------------------------------------------------------------

# Geometric spreading: use spherical spreading (with a small epsilon to avoid division by zero)
if spreading_model == "spherical":
    geom_spreading = np.maximum(distance_to_point, 1e-6)
elif spreading_model == "cylindrical":
    geom_spreading = np.maximum(np.sqrt(distance_to_point), 1e-6)
else:
    geom_spreading = np.maximum(distance_to_point, 1e-6)

# Sum the contributions (applying apodization, directivity, phase delay, attenuation, and spreading correction)
if use_complex:
    summed_field = np.sum(weights_3d * directivity * complex_field * attenuation / geom_spreading, axis=0)
    # Use the real part for the final pressure field, preserving the correct sign of the pressure.
    pressure_field = np.real(summed_field)
else:
    pressure_field = np.sum(weights_3d * directivity * complex_field * attenuation / geom_spreading, axis=0)

# Optional normalization: Scale the pressure field so that its peak equals the source pressure.
# Note: While normalization helps in comparing relative pressure levels, it may mask the absolute amplitude predictions.
apply_normalization = True
if apply_normalization:
    pressure_field = pressure_field / np.max(np.abs(pressure_field)) * source_pressure

# For LIFU, we focus on the actual pressure delivered:
# 1. Instantaneous Pressure Field: the raw (signed) pressure field.
# 2. Peak Pressure Field: the absolute value of the pressure field (representing peak pressure).
instantaneous_field = pressure_field
peak_field = np.abs(pressure_field)

## Optional Gaussian smoothing can be applied to emulate the finite spatial resolution of the transducer.
apply_smoothing = False  # Set to True to apply smoothing if desired
smoothing_sigma = 0.1    # Smoothing parameter (in pixels)
if apply_smoothing:
    instantaneous_field = gaussian_filter(instantaneous_field, sigma=smoothing_sigma)
    peak_field = gaussian_filter(peak_field, sigma=smoothing_sigma)

# Create a two-panel figure: left for instantaneous pressure, right for peak pressure.
fig, axs = plot_panels(
    x, z,
    fields=[instantaneous_field, peak_field],
    titles=[
        "Focused Pressure Field (Instantaneous)",
        "Focused Pressure Field (Peak)"
    ],
    cb_labels=["Pressure (Pa)", "Pressure (Pa)"],
    cmaps=["RdBu", "inferno"],
    figsize=(20,8)
)

export_dir = "exort"
if not os.path.exists(export_dir):
    os.makedirs(export_dir)
export_filename = os.path.join(export_dir, "wavephasefocused.png")
plt.savefig(export_filename, dpi=300)
plt.close()
