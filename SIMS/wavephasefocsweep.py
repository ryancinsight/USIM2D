import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1  # Requires scipy for Bessel functions
from matplotlib.colors import TwoSlopeNorm
import math
from beamforming_utils import element_directivity, calculate_attenuation
from viz import plot_panels
import os

"""
Physics of Ultrasound Beamforming Simulation: Frequency Sweep for Focused Ultrasound

This script simulates ultrasound beamforming with a frequency sweep from 140 kHz to 220 kHz,
emulating a focused ultrasound transducer. The simulation uses delay-and-sum beamforming with
phase correction (focusing) and applies an apodization (Hanning window) to the array elements.
Physical parameters:
  - The transducer is assumed to be in water (speed_of_sound = 1480 m/s).
  - The nominal (center) frequency is 180 kHz; however, the simulation sweeps from 140 kHz to 220 kHz.
  - The source pressure amplitude is fixed at 0.8 MPa.
  - Geometric spreading (spherical) and frequency-dependent attenuation are included.
  - Element directivity is modeled with a piston model using a Bessel function.

The ultrasound field is computed over a 2D spatial grid (lateral x and axial z),
and the pressure fields for each frequency are displayed in a subplot grid.

Reference:
  Jensen, "Field II: A program for simulating ultrasound systems", 1996.
"""

# Fixed geometry based on the center frequency (nominal)
center_frequency = 180e3          # 180 kHz nominal
speed_of_sound = 1480             # m/s in water
wavelength_center = speed_of_sound / center_frequency  # Nominal wavelength
element_spacing = wavelength_center / 2  # Fixed transducer geometry
element_width = wavelength_center / 2    # Fixed element width
num_elements = 22                        # Number of PZT elements
focal_distance = 0.05                    # Focal distance in meters (50 mm)

# Transducer properties and medium parameters
attenuation_coeff = 0.5       # Attenuation coefficient in dB/cm/MHz
attenuation_exponent = 1.0    # Frequency exponent for attenuation (typically 1.0)
spreading_model = "spherical" # Options: "spherical" (1/r) or "cylindrical" (1/sqrt(r)); default is spherical.
source_pressure = 0.8e6       # Source pressure amplitude (0.8 MPa in Pascals)

# Spatial grid definitions (common for all frequency simulations)
x_range = np.linspace(-0.05, 0.05, 1500)  # Lateral range: ±50 mm
z_range = np.linspace(0, 0.1, 1500)         # Axial range: 0 to 100 mm
x, z = np.meshgrid(x_range, z_range)

# Fixed transducer element positions (based on fixed geometry)
element_positions = np.linspace(
    -(num_elements - 1) * element_spacing / 2,
    (num_elements - 1) * element_spacing / 2,
    num_elements
)

# Apodization: use a Hanning window for sidelobe reduction
apodization_type = "hanning"  # Options: "hanning", "hamming", "rectangular"
if apodization_type == "hanning":
    weights = np.hanning(num_elements)
elif apodization_type == "hamming":
    weights = np.hamming(num_elements)
elif apodization_type == "rectangular":
    weights = np.ones(num_elements)
else:
    weights = np.ones(num_elements)
weights_3d = weights.reshape(num_elements, 1, 1)

# Reshape fixed element positions for broadcasting
element_positions_3d = element_positions.reshape(num_elements, 1, 1)

# Expand spatial grid arrays for vectorized computations
x3 = x[np.newaxis, :, :]  # shape: (1, M, N)
z3 = z[np.newaxis, :, :]  # shape: (1, M, N)
delta_x = x3 - element_positions_3d            # shape: (num_elements, M, N)
distance_to_point = np.sqrt(delta_x**2 + z3**2)  # shape: (num_elements, M, N)
r_focal = np.sqrt(element_positions_3d**2 + focal_distance**2)  # shape: (num_elements, 1, 1)
theta = np.arctan2(delta_x, z3)                  # shape: (num_elements, M, N)

# Frequency sweep parameters: from 140 kHz to 220 kHz
num_freq = 5
freqs = np.linspace(140e3, 220e3, num_freq)

# Loop over sweep frequencies and compute and store fields for composite processing
fields = []
for f in freqs:
    # Update frequency-dependent parameters
    wavelength_current = speed_of_sound / f
    phase_const_current = 2 * np.pi * f / speed_of_sound
    freq_MHz_current = f / 1e6

    directivity = element_directivity(theta, element_width, wavelength_current)
    attenuation = calculate_attenuation(distance_to_point, freq_MHz_current)
    complex_field = np.exp(1j * phase_const_current * (distance_to_point - r_focal))
    
    if spreading_model == "spherical":
        geom_spreading = np.maximum(distance_to_point, 1e-6)
    elif spreading_model == "cylindrical":
        geom_spreading = np.maximum(np.sqrt(distance_to_point), 1e-6)
    else:
        geom_spreading = np.maximum(distance_to_point, 1e-6)

    summed_field = np.sum(weights_3d * directivity * complex_field * attenuation / geom_spreading, axis=0)
    # Normalize the complex field (by its maximum magnitude) while preserving its phase.
    normalized_complex_field = summed_field / np.max(np.abs(summed_field)) * source_pressure
    fields.append(normalized_complex_field)

fields_array = np.array(fields)

# Compute two composite fields:
# 1. Coherent summation: arithmetic mean of the complex fields, then take the real part.
coherent_complex = np.sum(fields_array, axis=0) / num_freq
coherent_field = np.real(coherent_complex)

# 2. RSS summation: envelope computed with the root-sum-square method.
rss_field = np.sqrt(np.sum(np.abs(fields_array)**2, axis=0) / num_freq)

fig, axs = plot_panels(
    x, z,
    fields=[coherent_field, rss_field],
    titles=[
        "Composite Focused Pressure Field (Coherent Sum)\n(Frequency range: 140-220 kHz)",
        "Composite Focused Pressure Field (RSS Sum)\n(Frequency range: 140-220 kHz)"
    ],
    cb_labels=["Pressure (Pa)", "Pressure (Pa)"],
    cmaps=["RdBu", "inferno"],
    figsize=(20,8)
)

export_dir = "exort"
if not os.path.exists(export_dir):
    os.makedirs(export_dir)
export_filename = os.path.join(export_dir, "wavephasefocsweep.png")
plt.savefig(export_filename, dpi=300)
plt.close()
