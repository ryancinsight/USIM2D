"""
Standing Wave Simulation with Frequency Sweep for Focused Beamforming
----------------------------------------------------------------------
This script simulates standing waves in the presence of a reflective boundary
using a focused beamforming approach with a frequency sweep (from 140 kHz to 220 kHz).
For each frequency in the sweep, the incident field is computed using delay-and-sum
beamforming with a focusing delay (phase computed using (distance_to_point - r_focal)).
The reflected field is simulated by flipping the incident field along the axial (z)
direction with a 180° phase inversion. Averaging the fields across the frequency
sweep incoherently suppresses the fixed standing wave resonance, yielding composite
fields with reduced pronounced nodes and antinodes.

Reference:
    Jensen, "Field II: A program for simulating ultrasound systems", 1996.
"""

import numpy as np
import matplotlib.pyplot as plt
from beamforming_utils import element_directivity, calculate_attenuation
from viz import plot_panels
import os

# Simulation parameters for frequency sweep
low_freq = 140e3          # Lower frequency bound (Hz)
high_freq = 220e3         # Upper frequency bound (Hz)
num_freq = 5              # Number of frequencies in the sweep
freqs = np.linspace(low_freq, high_freq, num_freq)

speed_of_sound = 1480     # Speed of sound in m/s (water)
# Use a fixed (center) frequency for geometry/calculation setup.
center_frequency = 180e3  
wavelength_center = speed_of_sound / center_frequency  # Nominal wavelength based on center frequency
element_spacing = wavelength_center / 2                # Element spacing (λ/2)
element_width = wavelength_center / 2                  # Element width
num_elements = 22                                      # Number of elements
# Focusing is used in this simulation—focal delay correction is applied.
focal_distance = 0.05         # Focal distance in meters (50 mm)
source_pressure = 0.8e6       # Source pressure amplitude (0.8 MPa in Pascals)

# Tissue and geometric spreading parameters.
attenuation_coeff = 0.5       # Attenuation coefficient in dB/cm/MHz
attenuation_exponent = 1.0    # Frequency exponent for attenuation
spreading_model = "spherical" # Use spherical spreading (1/r decay)

# Define spatial grid: lateral x from -0.05 to 0.05 m, axial z from 0 to 0.1 m.
x_range = np.linspace(-0.05, 0.05, 1500)  # Lateral (meters)
z_range = np.linspace(0, 0.1, 1500)         # Axial (meters)
x, z = np.meshgrid(x_range, z_range)

# Determine transducer element positions along x-axis.
element_positions = np.linspace(
    -(num_elements - 1) * element_spacing / 2,
    (num_elements - 1) * element_spacing / 2,
    num_elements
)
# Reshape for broadcasting.
element_positions_3d = element_positions.reshape(num_elements, 1, 1)

# Expand spatial arrays for vectorized computation (add element axis).
x3 = x[np.newaxis, :, :]   # Shape: (1, M, N)
z3 = z[np.newaxis, :, :]   # Shape: (1, M, N)

# Compute distances and angles:
delta_x = x3 - element_positions_3d                     # (num_elements, M, N)
distance_to_point = np.sqrt(delta_x**2 + z3**2)           # (num_elements, M, N)
# For focusing, compute r_focal: distance from each element to the focal point.
r_focal = np.sqrt(element_positions_3d**2 + focal_distance**2)  # (num_elements, 1, 1)
# Angle for directivity computation.
theta = np.arctan2(delta_x, z3)                           # (num_elements, M, N)

# Apodization: Hanning window for sidelobe reduction.
weights = np.hanning(num_elements)
weights_3d = weights.reshape(num_elements, 1, 1)

# Prepare lists to store fields computed at each frequency.
incident_fields = []
reflected_fields = []

# Loop over frequencies in the sweep.
for f in freqs:
    # Update frequency-dependent parameters.
    wavelength_current = speed_of_sound / f
    phase_const_current = 2 * np.pi * f / speed_of_sound
    freq_MHz_current = f / 1e6
    
    # Compute element directivity (using current wavelength).
    directivity = element_directivity(theta, element_width, wavelength_current)
    # Compute tissue attenuation with current frequency.
    attenuation = calculate_attenuation(distance_to_point, freq_MHz_current)
    
    # Compute the incident field using delay-and-sum focused beamforming.
    # Incorporate the focusing delay: (distance_to_point - r_focal)
    complex_field = np.exp(1j * phase_const_current * (distance_to_point - r_focal))
    
    # Compute geometric spreading with spherical model.
    geom_spreading = np.maximum(distance_to_point, 1e-6)
    
    # Sum contributions over elements.
    summed_field = np.sum(weights_3d * directivity * complex_field * attenuation / geom_spreading, axis=0)
    
    # Normalize the resulting field and scale to source pressure.
    incident_field = (summed_field / np.max(np.abs(summed_field))) * source_pressure
    incident_field = np.real(incident_field)   # Use real part (instantaneous pressure)
    
    # Simulate the reflected field by flipping incident field along z-axis.
    # For a hard reflection, apply a 180° phase inversion (multiply by -1).
    reflected_field = -np.flipud(incident_field)
    
    incident_fields.append(incident_field)
    reflected_fields.append(reflected_field)

# Convert the list of fields to arrays.
incident_fields = np.array(incident_fields)
reflected_fields = np.array(reflected_fields)

# Compute composite fields via incoherent averaging.
composite_incident = np.mean(incident_fields, axis=0)
composite_reflected = np.mean(reflected_fields, axis=0)
composite_standing = composite_incident + composite_reflected

# Visualize the composite fields using the shared plotting function.
fig, axs = plot_panels(
    x, z,
    fields=[composite_incident, composite_reflected, composite_standing],
    titles=["Composite Incident Field (Focused)", "Composite Reflected Field (Focused)", "Composite Standing Field (Focused)"],
    cb_labels=["Pressure (Pa)", "Pressure (Pa)", "Pressure (Pa)"],
    cmaps=["RdBu", "RdBu", "RdBu"],
    figsize=(30, 8)
)

# Save the figure to the export folder.
export_dir = "exort"
if not os.path.exists(export_dir):
    os.makedirs(export_dir)
export_filename = os.path.join(export_dir, "wavephasestandingfocsweep.png")
fig.savefig(export_filename, dpi=300)
plt.close(fig) 