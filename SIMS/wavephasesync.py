import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from matplotlib.colors import TwoSlopeNorm
from beamforming_utils import element_directivity, calculate_attenuation
from viz import plot_panels
import os

"""
Physics of Ultrasound Beamforming Simulation - Synchronous Phase

This script simulates ultrasound beamforming using synchronous excitation of a 22-element PZT array.
In synchronous (or "in-phase") beamforming, all elements are excited with the same phase (i.e. zero phase delay),
so that the pressure field is formed by the coherent summation of contributions that are solely determined
by the physical propagation delays. Specifically, the pressure field is given by:

    P(x,z) = Σ D(θ) · cos(2πf·r/c) · A(r) / S(r)

where:
    - r is the distance from an individual element to the field point,
    - D(θ) represents the directivity function of each finite-width element (modeled here using a normalized sinc),
    - A(r) is the tissue attenuation factor (computed from the frequency, attenuation coefficient, and r),
    - S(r) is the geometric spreading factor (here modeled as a 1/√r decay for pressure),
    - and the summation (Σ) is taken over all elements.

Since no delay correction is applied, the interference pattern inherently exhibits both positive (compressive)
and negative (rarefactive) pressure regions, which can be directly visualized.
"""

# Constants
frequency = 180e3             # Frequency in Hz
speed_of_sound = 1480         # Speed of sound in m/s for water
wavelength = speed_of_sound / frequency   # Wavelength in meters
element_spacing = wavelength / 2          # Element spacing (λ/2) to avoid grating lobes
element_width = wavelength / 2            # Element width (typical for PZT elements)
num_elements = 22                         # Number of PZT elements
# (Focal distance is used only for plotting a marker in synchronous beamforming.)
focal_distance = 0.05                     # Focal distance in meters (50 mm)

# Tissue attenuation (in dB/cm/MHz) and exponent (commonly ~1 for soft tissue)
attenuation_coeff = 0.5                   # Attenuation coefficient in dB/cm/MHz
attenuation_exponent = 1.0                # Frequency exponent for attenuation (typically 1.0)

# Selectable geometric spreading model.
# Options: "spherical" (1/r decay as in far-field diffraction per Field II)
#          "cylindrical" (1/sqrt(r) decay, sometimes appropriate for quasi-2D propagation)
spreading_model = "spherical"             # Default is spherical spreading.

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

# Precompute frequency in MHz and the phase constant (2πf/c)
freq_MHz = frequency / 1e6
phase_const = 2 * np.pi * frequency / speed_of_sound

# Vectorized Synchronous Beamforming:
#
# All elements are processed simultaneously. In synchronous beamforming, every element
# is excited in-phase so that the phase term is simply:
#     phase_term = cos(phase_const * distance_to_point)
# The contributions are summed over all elements to yield the final pressure field.

# Reshape element positions for broadcasting: shape (num_elements, 1, 1)
element_positions_3d = element_positions.reshape(num_elements, 1, 1)

# Expand the spatial grid to 3D for element-wise operations.
x3 = x[np.newaxis, :, :]  # shape: (1, M, N)
z3 = z[np.newaxis, :, :]  # shape: (1, M, N)

# Compute lateral differences and distances from every element to all grid points.
delta_x = x3 - element_positions_3d  # shape: (num_elements, M, N)
distance_to_point = np.sqrt(delta_x**2 + z3**2)  # shape: (num_elements, M, N)

# Calculate the angle for directivity.
theta = np.arctan2(x3 - element_positions_3d, z3)  # shape: (num_elements, M, N)

# Compute directivity and attenuation for each element over the grid.
directivity = element_directivity(theta, element_width, wavelength)  # shape: (num_elements, M, N)
attenuation = calculate_attenuation(distance_to_point, freq_MHz)  # shape: (num_elements, M, N)

# Synchronous phase term: all elements are excited with zero delay.
phase_term = np.cos(phase_const * distance_to_point)  # shape: (num_elements, M, N)

# Geometric Spreading: Choose between spherical (1/r) and cylindrical (1/sqrt(r)) models.
if spreading_model == "spherical":
    geom_spreading = np.maximum(distance_to_point, 1e-6)
elif spreading_model == "cylindrical":
    geom_spreading = np.maximum(np.sqrt(distance_to_point), 1e-6)
else:
    geom_spreading = np.maximum(distance_to_point, 1e-6)

# Sum contributions from all elements to form the final pressure field.
pressure_field = np.sum(directivity * phase_term * attenuation / geom_spreading, axis=0)

# Scale the pressure field so that the maximum pressure equals the source pressure (0.8 MPa)
pressure_field = pressure_field / np.max(np.abs(pressure_field)) * source_pressure

# For synchronous simulation (single frequency) we compare:
#
# 1. Coherent (instantaneous) field: the real (signed) pressure field.
# 2. RSS (envelope) field: the magnitude (absolute value) of the pressure field.

coherent_field = pressure_field  # pressure_field is assumed to be the already computed real pressure.
rss_field = np.abs(pressure_field)

fig, axs = plot_panels(
    x, z,
    fields=[coherent_field, rss_field],
    titles=[
        "Synchronous Pressure Field (Coherent Sum)\n(Instantaneous, signed)",
        "Synchronous Pressure Field (RSS Sum)\n(Envelope)"
    ],
    cb_labels=["Pressure (Pa)", "Pressure (Pa)"],
    cmaps=["RdBu", "inferno"],
    figsize=(20,8)
)

export_dir = "exort"
if not os.path.exists(export_dir):
    os.makedirs(export_dir)
export_filename = os.path.join(export_dir, "wavephasesync.png")
plt.savefig(export_filename, dpi=300)
plt.close()
