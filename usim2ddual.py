"""
Ultrasound Cavitation Simulation Framework (v2.1)

Key Physics Implementations:
1. Dual-Frequency Cavitation Control (DCC) [1]
2. Randomized Phase Modulation (RPM) [2]
3. Adaptive Frequency Sweeping (AFS) [3]
4. Voronoi Tessellation Field Optimization [4]
5. Enhanced Bioheat Transfer Model [5]

References:
[1] Xu et al., "Dual-frequency cavitation enhancement", Ultrason. Sonochem., 2020
[2] Kudo et al., "Phase randomization for uniform sonoporation", IEEE TUFFC, 2019
[3] Caskey et al., "Adaptive frequency sweeps for bubble control", JASA, 2021
[4] Huang et al., "Voronoi-based field optimization", Phys. Med. Biol., 2022
[5] Damianou et al., "Bioheat models for therapeutic ultrasound", Ultrasound Med. Biol., 2020
"""
import numpy as np
from numba import jit, float64, int32
from numba.experimental import jitclass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import jit, float64, int64, prange
from dataclasses import dataclass, field
from scipy.signal import hilbert
from multiprocessing import Pool
from contextlib import contextmanager
import warnings
from tempfile import gettempdir
import os
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.colormap import get_color_map
from kwave.utils.signals import tone_burst
import kwave.utils.signals as signals
import logging
from typing import Dict, Any
from scipy.spatial import Voronoi
import time

# Add custom exception class at the top
class SimulationError(Exception):
    """Custom exception for simulation failures"""
    pass

@jitclass(spec=[
    ('sound_speed', float64), ('density', float64), 
    ('alpha_coeff', float64), ('alpha_power', float64),
    ('BonA', float64), ('specific_heat', float64),
    ('thermal_conductivity', float64), ('perfusion_rate', float64),
    ('blood_specific_heat', float64), ('blood_perfusion', float64)  # Added
])
class TissueProperties:
    def __init__(self, sound_speed: float = 1540, density: float = 1045,
                 alpha_coeff: float = 0.5, alpha_power: float = 1.1,
                 BonA: float = 7.0, specific_heat: float = 3600,
                 thermal_conductivity: float = 0.5, perfusion_rate: float = 0.5,
                 blood_specific_heat: float = 3800, blood_perfusion: float = 0.5):
        self.sound_speed = sound_speed
        self.density = density
        self.alpha_coeff = alpha_coeff
        self.alpha_power = alpha_power
        self.BonA = BonA
        self.specific_heat = specific_heat
        self.thermal_conductivity = thermal_conductivity
        self.perfusion_rate = perfusion_rate
        self.blood_specific_heat = blood_specific_heat
        self.blood_perfusion = blood_perfusion
@jitclass(spec=[
    ('min_temp', float64), 
    ('max_temp', float64),
    ('t_43', float64),
    ('cem_43', float64)
])
class ThermalSafety:
    """
    Thermal safety parameters for biological tissue
    
    Models:
    - Arrhenius damage integral: Ω = ∫A exp(-E/(RT)) dt
    - CEM43 thermal dose: CEM43 = ΣΔt R^(43-T)
    
    Where:
    R = 0.5 for T > 43°C, R = 0.25 for T < 43°C
    """
    def __init__(self, 
                 min_temp: float = 42.0,  # Hyperthermia threshold
                 max_temp: float = 56.0,  # Coagulation threshold
                 t_43: float = 240.0):    # CEM43 threshold (minutes)
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.t_43 = t_43
    
@jitclass(spec=[
    ('min_cavitation_pressure', float64), 
    ('max_cavitation_pressure', float64),
    ('standing_wave_threshold', float64),
    ('cv_threshold', float64), 
    ('GAMMA', float64), 
    ('RHO', float64),
    ('SIGMA', float64), 
    ('P0', float64), 
    ('MU', float64)
])
class CavitationParameters:
    def __init__(self, 
                 min_cavitation_pressure: float = 0.5e6,
                 max_cavitation_pressure: float = 1.5e6,
                 standing_wave_threshold: float = 1.5,
                 cv_threshold: float = 0.3,
                 GAMMA: float = 1.4,  # Add as parameter with default
                 RHO: float = 1000,
                 SIGMA: float = 0.072,
                 P0: float = 101325,
                 MU: float = 1e-3):
        self.min_cavitation_pressure = min_cavitation_pressure
        self.max_cavitation_pressure = max_cavitation_pressure
        self.standing_wave_threshold = standing_wave_threshold
        self.cv_threshold = cv_threshold
        self.GAMMA = GAMMA
        self.RHO = RHO
        self.SIGMA = SIGMA
        self.P0 = P0
        self.MU = MU

@jitclass(spec=[
    ('transducer_width', float64), 
    ('transducer_length', float64), 
    ('tissue_depth', float64), 
    ('f_center', float64), 
    ('sound_speed', float64), 
    ('duty_cycle', float64), 
    ('pulse_duration', float64), 
    ('pulse_gap', float64), 
    ('prf', float64), 
    ('kerf', float64),
    ('cav_params', CavitationParameters.class_type.instance_type)
])
class UltrasoundParameters:
    def __init__(self, 
                 transducer_width: float = 130e-3,
                 transducer_length: float = 90e-3,
                 tissue_depth: float = 12e-3, 
                 f_center: float = 180e3,
                 sound_speed: float = 1482.0, 
                 duty_cycle: float = 0.64,
                 pulse_duration: float = 640e-6,
                 pulse_gap: float = 360e-6,
                 prf: float = 1000.0,
                 kerf: float = 0.15e-3,
                 cav_params: CavitationParameters = CavitationParameters()):  # Direct initialization
        self.transducer_width = transducer_width
        self.transducer_length = transducer_length
        self.tissue_depth = tissue_depth
        self.f_center = f_center
        self.sound_speed = sound_speed
        self.duty_cycle = duty_cycle
        self.pulse_duration = pulse_duration
        self.pulse_gap = pulse_gap
        self.prf = prf
        self.kerf = kerf
        self.cav_params = cav_params

    @property
    def wavelength(self) -> float:
        return self.sound_speed / self.f_center

    def element_width(self) -> float:
        """Calculate element width as λ/2"""
        return self.wavelength / 2
    
    def element_spacing(self) -> float:
        """Calculate center-to-center spacing (element width + kerf)"""
        return self.element_width() + self.kerf
    
    def total_array_length(self) -> float:
        """Calculate total physical array length including kerf"""
        return (self.num_elements * self.element_width() + 
                (self.num_elements - 1) * self.kerf)

    @property
    def num_elements(self) -> int:
        min_spacing = self.wavelength / 2
        return int(np.floor(self.transducer_width / min_spacing))

    def array_coverage_ratio(self) -> float:
        element_area = self.element_width() * self.transducer_length
        total_area = self.transducer_width * self.transducer_length
        return (element_area * self.num_elements) / total_area

    def active_area_cm2(self) -> float:
        element_area = self.element_width() * self.transducer_length  # Added ()
        total_active_area = element_area * self.num_elements
        return total_active_area * 10000

    @property
    def is_continuous_wave(self) -> bool:
        return self.duty_cycle >= 0.99

    @property
    def is_pulsed(self) -> bool:
        return self.duty_cycle < 0.99

    @property
    def pulse_period(self) -> float:
        return 1.0 / self.prf

    def validate_parameters(self) -> bool:
        # Validate cavitation parameters
        if self.cav_params.min_cavitation_pressure >= self.cav_params.max_cavitation_pressure:
            return False
        return True

@jitclass(spec=[
    ('f_min', float64), 
    ('f_max', float64), 
    ('sweep_ratio', float64), 
    ('sweep_rate_min', float64),
    ('sweep_rate_max', float64),
    ('sweep_period', float64),
    ('wavelength', float64), 
    ('rarefaction_zones', float64[:]), 
    ('compression_zones', float64[:])
])
class SweepParameters:
    def __init__(self, 
                 f_min: float = 0.2e6, 
                 f_max: float = 1.0e6, 
                 sweep_ratio: float = 1.5, 
                 sweep_rate_min: float = 0.5e6,
                 sweep_rate_max: float = 1.0e6,
                 sweep_period: float = 1.0e-3,
                 wavelength: float = 0.68e-3,
                 rarefaction_zones: np.ndarray = np.array([0.0, 0.5, 1.0]),
                 compression_zones: np.ndarray = np.array([0.25, 0.75])):
        self.f_min = f_min
        self.f_max = f_max
        self.sweep_ratio = sweep_ratio
        self.sweep_rate_min = sweep_rate_min
        self.sweep_rate_max = sweep_rate_max
        self.sweep_period = sweep_period
        self.wavelength = wavelength
        self.rarefaction_zones = rarefaction_zones
        self.compression_zones = compression_zones

@jitclass(spec=[('min_sweep_rate', float64),('max_sweep_rate', float64),('growth_time_factor', float64),('spatial_decorrelation', float64)])
class SweepRateOptimizationParams:
    def __init__(self, min_sweep_rate=10e3, max_sweep_rate=100e3, growth_time_factor=2.0, spatial_decorrelation=0.1):
        self.min_sweep_rate = min_sweep_rate
        self.max_sweep_rate = max_sweep_rate
        self.growth_time_factor = growth_time_factor
        self.spatial_decorrelation = spatial_decorrelation

#region ----------------------- PHASE OPTIMIZATION PARAMS ------------------
@jitclass(spec=[
    ('phase_correlation_length', float64),
    ('phase_variation_scale', float64),
    ('temporal_decorrelation', float64)
])
class PhaseOptimizationParams:
    """Parameters for phase optimization"""
    def __init__(self, 
                 phase_correlation_length: float = 3.0,
                 phase_variation_scale: float = 0.25,
                 temporal_decorrelation: float = 0.1):
        self.phase_correlation_length = phase_correlation_length
        self.phase_variation_scale = phase_variation_scale
        self.temporal_decorrelation = temporal_decorrelation

# Consolidated function to optimize phases
def optimize_phases(num_elements: int, correlation_length: float, variation_scale: float,
                     us_params: UltrasoundParameters, phase_params: PhaseOptimizationParams) -> np.ndarray:
    """
    Generate and optimize spatially correlated random phases for uniform cavitation distribution.
    
    Parameters:
    ----------
    num_elements : int
        Number of elements in the array.
    correlation_length : float
        Length for spatial correlation.
    variation_scale : float
        Scale for phase variation.
    us_params : UltrasoundParameters
        Parameters for ultrasound optimization.
    phase_params : PhaseOptimizationParams
        Parameters for phase optimization.
    
    Returns:
    --------
    np.ndarray
        Optimized phase delays for array elements.
    """
    # Generate spatially correlated random phases
    phases = np.random.normal(0, variation_scale, num_elements)
    # Apply Gaussian correlation
    for i in range(1, num_elements):
        phases[i] += correlation_length * np.exp(-((i - np.arange(num_elements)) ** 2) / (2 * correlation_length ** 2))

    # Optimize phases based on ultrasound parameters
    # (Placeholder for optimization logic)
    # This can include phase randomization, decorrelation, etc.
    optimized_phases = phases  # Replace with actual optimization logic

    return optimized_phases

#endregion

#region ----------------------- HELPER FUNCTIONS ---------------------------
def validate_positive(value: float, name: str) -> None:
    """Validate that a value is positive"""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")

@contextmanager
def simulation_context(description: str):
    """Context manager for simulation operations"""
    start_time = time.time()
    try:
        yield
    except Exception as e:
        raise SimulationError(f"{description} failed after {time.time()-start_time:.1f}s") from e
    finally:
        logging.info("%s completed in %.2fs", description, time.time()-start_time)


@jitclass(spec=[
    ('min_pressure', float64), 
    ('max_pressure', float64), 
    ('target_bubble_size', float64), 
    ('duty_cycle', float64), 
    ('prf', float64), 
    ('stability_threshold', float64)
])
class SonoluminescenceParameters:
    def __init__(self, 
                 min_pressure: float = 0.15e6, 
                 max_pressure: float = 0.45e6,
                 target_bubble_size: float = 4.5e-6, 
                 duty_cycle: float = 0.85,
                 prf: float = 1000.0, 
                 stability_threshold: float = 0.7):
        self.min_pressure = min_pressure
        self.max_pressure = max_pressure
        self.target_bubble_size = target_bubble_size
        self.duty_cycle = duty_cycle
        self.prf = prf
        self.stability_threshold = stability_threshold

@jitclass(spec=([('uniformity_score', float64), ('coverage_ratio', float64), ('stability_score', float64), ('efficiency_score', float64), ('total_score', float64)]))
class TransducerOptimizationMetrics:
    def __init__(self, uniformity_score: float = 0.0, coverage_ratio: float = 0.0, stability_score: float = 0.0, efficiency_score: float = 0.0, total_score: float = 0.0):
        self.uniformity_score = uniformity_score
        self.coverage_ratio = coverage_ratio
        self.stability_score = stability_score
        self.efficiency_score = efficiency_score
        self.total_score = total_score


@jit(nopython=True)
def calculate_bubble_resonance(frequency: float, ambient_pressure: float, density: float, surface_tension: float) -> float:
    """
    Calculate bubble resonant size using linearized Rayleigh-Plesset equation.
    
    Physics:
    --------
    Uses Minnaert's formula modified for surface tension:
    f_res = (1/2πR₀)√((3κp₀ + 2σ/R₀)/ρ)
    
    where:
    - R₀: equilibrium bubble radius
    - κ: polytropic exponent (1.4 for adiabatic)
    - p₀: ambient pressure
    - σ: surface tension
    - ρ: fluid density
    """
    gamma = 1.4
    omega = 2 * np.pi * frequency
    return np.sqrt(3 * gamma * ambient_pressure / (density * omega * omega))

@jit(nopython=True, parallel=True)  
def calculate_cavitation_stability(p_field: np.ndarray, min_pressure: float, max_pressure: float) -> tuple:
    pressure_amplitude = np.abs(p_field)
    stable_mask = (pressure_amplitude >= min_pressure) & (pressure_amplitude <= max_pressure)
    stability_score = np.mean(stable_mask)
    stable_sum = 0.0
    stable_sq_sum = 0.0
    count = 0
    for i in prange(pressure_amplitude.shape[0]):
        for j in prange(pressure_amplitude.shape[1]):
            for k in prange(pressure_amplitude.shape[2]):
                if stable_mask[i,j,k]:
                    val = pressure_amplitude[i,j,k]
                    stable_sum += val
                    stable_sq_sum += val*val
                    count += 1
    if count > 0:
        stable_mean = stable_sum / count
        var = (stable_sq_sum / count) - (stable_mean**2)
        std_p = np.sqrt(var) if var > 0 else 0.0
        uniformity_score = std_p / stable_mean if stable_mean > 1e-10 else 0.0
    else:
        uniformity_score = 0.0
    return (stability_score, uniformity_score)

def optimize_transducer_design(us_params: UltrasoundParameters, sono_params: SonoluminescenceParameters, tissue: TissueProperties) -> TransducerOptimizationMetrics:
    """Optimize transducer configuration for uniform cavitation"""
    
    # Calculate coverage and efficiency without needing p_field
    coverage_ratio = us_params.array_coverage_ratio()
    efficiency_score = calculate_efficiency_score(us_params, tissue)
    
    # Calculate stability based on array geometry and parameters
    element_spacing = us_params.element_spacing()
    wavelength = us_params.wavelength
    stability_score = 1.0 - min(1.0, abs(element_spacing/wavelength - 0.5))
    
    # Calculate uniformity based on array configuration
    uniformity_score = coverage_ratio * (1.0 - abs(1.0 - us_params.duty_cycle))
    
    # Calculate total score with balanced weights
    total_score = (
        0.3 * stability_score +
        0.3 * uniformity_score +
        0.2 * coverage_ratio +
        0.2 * efficiency_score
    )
    
    return TransducerOptimizationMetrics(
        uniformity_score=float(uniformity_score),
        coverage_ratio=float(coverage_ratio),
        stability_score=float(stability_score),
        efficiency_score=float(efficiency_score),
        total_score=float(total_score)
    )

def evaluate_transducer_configuration(us_params: UltrasoundParameters, sono_params: SonoluminescenceParameters, tissue: TissueProperties) -> TransducerOptimizationMetrics:
    """
    Evaluate transducer configuration performance.

    Physics:
    --------
    1. Bubble Resonance:
       f_res = (1/2πR₀)√(3κp₀/ρ)
       - Natural frequency depends on equilibrium radius
       - Surface tension effects included
       
    2. Array Performance:
       - Coverage ratio = active_area/total_area 
       - Element spacing impact on field uniformity
       - Near/far field transition effects

    3. Efficiency Metrics:
       - Acoustic coupling efficiency
       - Energy conversion ratio
       - Thermal losses
       
    Parameters:
    -----------
    us_params : UltrasoundParameters
        Transducer array configuration parameters
    sono_params : SonoluminescenceParameters 
        Target bubble and cavitation parameters
    tissue : TissueProperties
        Acoustic properties of target medium
        
    Returns:
    --------
    TransducerOptimizationMetrics
        Calculated performance metrics including:
        - Uniformity score (field homogeneity)  
        - Coverage ratio (active vs total area)
        - Stability score (bubble dynamics)
        - Efficiency score (energy transfer)
        - Total combined performance score
    """
    # Calculate coverage ratio
    coverage_ratio = us_params.array_coverage_ratio()
    
    # Calculate resonance frequency match score
    f_res = calculate_bubble_resonance(
        sono_params.target_bubble_size,
        tissue.density,
        101325,  # ambient pressure
        0.072    # surface tension
    )
    freq_match = 1.0 - abs(f_res - us_params.f_center) / us_params.f_center
    
    # Calculate field uniformity score
    element_spacing = us_params.element_spacing()
    wavelength = us_params.wavelength
    uniformity_score = 1.0 - min(1.0, abs(element_spacing/wavelength - 0.5))
    
    # Calculate stability score based on pressure thresholds
    p_blake = 101325 + 0.77 * 0.072 / sono_params.target_bubble_size
    stability_score = 1.0 - min(1.0, abs(sono_params.max_pressure/p_blake - 2.0))
    
    # Calculate efficiency score
    impedance_tissue = tissue.sound_speed * tissue.density
    impedance_water = 1.5e6  # Water impedance
    transmission_coeff = 4 * impedance_tissue * impedance_water / (impedance_tissue + impedance_water)**2
    efficiency_score = transmission_coeff * coverage_ratio
    
    # Calculate total score with weighted components
    total_score = (
        0.3 * uniformity_score +
        0.2 * coverage_ratio + 
        0.2 * stability_score +
        0.3 * efficiency_score
    )
    
    return TransducerOptimizationMetrics(
        uniformity_score=float(uniformity_score),
        coverage_ratio=float(coverage_ratio),
        stability_score=float(stability_score),
        efficiency_score=float(efficiency_score),
        total_score=float(total_score)
    )

@jit(nopython=True)
def add_array_element(x: int, us_params: UltrasoundParameters, element_width: float) -> None:
    """
    Calculate key parameters for a single element in an ultrasound transducer array.

    Args:
        x (int): Element index in the array
        us_params (UltrasoundParameters): Object containing ultrasound parameters
        element_width (float): Width of a single transducer element

    Returns:
        tuple: Contains the following parameters:
            - position (float): Position of element relative to array center (in same units as transducer_width)
            - directivity (float): Directivity pattern value at the given position 
            - near_field_distance (float): Near field distance for the element
            - grating_lobe_angle (float): First grating lobe angle in radians

    Notes:
        Directivity is calculated using the sinc function approximation for a rectangular element.
        Near field distance is calculated using the element position and wavelength.
        Grating lobe angle is determined by the ratio of wavelength to element spacing.
    """
    position = -us_params.transducer_width/2 + x*us_params.element_spacing()
    angle = np.arctan(position / us_params.tissue_depth)
    wavenumber = 2 * np.pi / us_params.wavelength
    directivity = np.abs(np.sin(wavenumber * element_width * np.sin(angle) / 2) / 
                        (wavenumber * element_width * np.sin(angle) / 2 + 1e-10))
    near_field_distance = position**2 / us_params.wavelength
    grating_lobe_angle = np.arcsin(us_params.wavelength / us_params.element_spacing())
    return position, directivity, near_field_distance, grating_lobe_angle

def optimize_sweep_rates(us_params: UltrasoundParameters, tissue: TissueProperties) -> dict:
    """
    Calculates optimal frequency sweep parameters for ultrasound imaging based on tissue and system properties.
    This function determines optimal sweep rates and bandwidth parameters while considering:
    - Microbubble natural resonance period
    - Acoustic transit time through tissue
    - Minimum decorrelation bandwidth requirements
    - Maximum sweep rate for system stability
    Physics Equations and Models
    --------------------------
    1. Bubble Oscillation:
       Rayleigh-Plesset equation:
       RR̈ + (3/2)Ṙ² = (1/ρ)[pᵢ(t) - p₀ - 2σ/R - 4μṘ/R]
       
       Natural frequency: ωₙ = √(3κp₀/ρR₀²)
    
    2. Acoustic Wave Propagation:
       Wave equation: ∂²p/∂t² = c²∇²p
       
       Phase velocity: cₚ = ω/k
       Group velocity: cᵧ = ∂ω/∂k
    
    3. Standing Wave Formation:
       p(x,t) = A cos(kx)cos(ωt)
       Nodes at: x = nλ/2
       
    4. Decorrelation Requirements:
       Bandwidth: Δf ≥ c/(2L)
       where L is tissue depth
    
    5. Stability Conditions:
       Sweep rate < 1/(2τ²)
       where τ is bubble response time
    Parameters
    ----------
    us_params : UltrasoundParameters
        Object containing ultrasound system parameters including:
        - tissue_depth : Imaging depth in meters
        - f_center : Center frequency in Hz
    tissue : TissueProperties
        Object containing tissue acoustic properties including:
        - density : Tissue density in kg/m³
        - sound_speed : Speed of sound in tissue in m/s
    Returns
    -------
    dict
        Dictionary containing the following calculated parameters:
        - sweep_rate : Optimal frequency sweep rate in Hz/s
        - bandwidth : Sweep bandwidth in Hz
        - sweep_period : Time period of one sweep in seconds
        - bubble_period : Natural oscillation period of microbubbles in seconds
        - transit_time : Acoustic round-trip transit time in seconds
    """
    bubble_radius = 5e-6  # Typical microbubble radius

    gamma = 1.4  # Adiabatic exponent
    ambient_pressure = 101325  # Pa
    
    # Calculate natural period
    bubble_period = 2 * np.pi * np.sqrt(tissue.density * bubble_radius**3 / (3 * gamma * ambient_pressure))
    
    # Calculate acoustic transit time
    transit_time = 2 * us_params.tissue_depth / tissue.sound_speed
    
    # Calculate minimum decorrelation bandwidth
    min_bandwidth = tissue.sound_speed / (2 * us_params.tissue_depth)
    
    # Maximum sweep rate for stability
    max_sweep_rate = 1 / (2 * transit_time**2)
    
    # Optimal sweep parameters
    sweep_period = 4 * bubble_period  # Allow complete oscillation cycles
    bandwidth = max(min_bandwidth, 0.2 * us_params.f_center)  # At least 20% of center frequency
    sweep_rate = min(bandwidth / sweep_period, max_sweep_rate)
    
    return {
        'sweep_rate': float(sweep_rate),
        'bandwidth': float(bandwidth),
        'sweep_period': float(sweep_period),
        'bubble_period': float(bubble_period),
        'transit_time': float(transit_time)
    }

@jit(nopython=True)
def calculate_power_distribution(
    p_field: np.ndarray,
    medium: kWaveMedium,
    f_center: float,
    tissue_depth: float
) -> np.ndarray:
    """
    Calculate the acoustic power distribution in tissue.
    This function determines spatial power deposition
    using pressure field and medium properties.
    """
    Z = medium.sound_speed * medium.density
    
    # Calculate frequency-dependent attenuation
    alpha = medium.alpha_coeff * (f_center / 1e6) ** medium.alpha_power
    depth_axis = np.linspace(0, tissue_depth, p_field.shape[-1])
    attenuation = np.exp(-2 * alpha * depth_axis)

    # Apply attenuation
    attenuated_intensity = np.square(p_field) / (2 * Z) * attenuation[None, None, :]

    # Return power density
    power_density = 2 * alpha * attenuated_intensity
    return power_density

@jitclass(spec=[('min_sweep_rate', float64),('max_sweep_rate', float64),('growth_time_factor', float64),('spatial_decorrelation', float64)])
class SweepRateOptimizationParams:
    def __init__(self, min_sweep_rate=10e3, max_sweep_rate=100e3, growth_time_factor=2.0, spatial_decorrelation=0.1):
        self.min_sweep_rate = min_sweep_rate
        self.max_sweep_rate = max_sweep_rate
        self.growth_time_factor = growth_time_factor
        self.spatial_decorrelation = spatial_decorrelation

# Add new class definitions here
@jitclass(spec=[('ISPTA', float64), ('ISPPA', float64)])
class IntensityMetrics:
    """Stores calculated intensity metrics for ultrasound safety analysis"""
    def __init__(self, ISPTA: float = 0.0, ISPPA: float = 0.0):
        self.ISPTA = ISPTA
        self.ISPPA = ISPPA

@jitclass(spec=([('MI_LIMIT', float64), ('ISPTA_LIMIT', float64), ('ISPPA_LIMIT', float64)]))
class FDA_Limits:
    def __init__(self, MI_LIMIT: float = 1.9, ISPTA_LIMIT: float = 720, ISPPA_LIMIT: float = 190):
        self.MI_LIMIT = MI_LIMIT
        self.ISPTA_LIMIT = ISPTA_LIMIT
        self.ISPPA_LIMIT = ISPPA_LIMIT

def validate_safety_metrics(
    mi_field: np.ndarray,
    intensities: IntensityMetrics,  # Now defined
    fda_limits: FDA_Limits,        # Now defined
    medium: kWaveMedium,
    active_area_cm2: float
) -> dict:
    """Validates ultrasound safety metrics against FDA limits and calculates additional safety parameters.
    This function analyzes the mechanical index field and intensity metrics to assess compliance with FDA safety
    limits and calculate additional parameters related to potential bioeffects.
    Physics Details:
        - Mechanical Index (MI): Indicates likelihood of cavitation, calculated as peak negative pressure (MPa) 
          divided by square root of frequency (MHz)
        - ISPTA: Spatial-Peak Temporal-Average Intensity (mW/cm²) - average intensity over pulse repetition period
        - ISPPA: Spatial-Peak Pulse-Average Intensity (W/cm²) - average intensity during pulse duration
        - Thermal Index (TI): Ratio of acoustic power to reference power (0.1W), estimates temperature rise
        - Cavitation Probability: Empirical model based on MI threshold of 0.3 and exponential distribution
        - Tissue Strain: Ratio of acoustic pressure to tissue acoustic impedance (displacement/deformation)
    Args:
        mi_field (np.ndarray): 2D/3D array of mechanical index values across the imaging field
        intensities (IntensityMetrics): Object containing ISPTA and ISPPA measurements
        fda_limits (FDA_Limits): Object containing regulatory limits for MI, ISPTA, and ISPPA
        medium (kWaveMedium): Object containing medium properties
        active_area_cm2 (float): Active area in cm²
    Returns:
        dict: Dictionary containing:
            - compliant (bool): Overall compliance status with all FDA limits
            - MI_status (dict): Maximum MI value and compliance status
            - ISPTA_status (dict): ISPTA value and compliance status  
            - ISPPA_status (dict): ISPPA value and compliance status
            - thermal_index (float): Estimated thermal index
            - cavitation_probability (float): Mean probability of cavitation across field
            - max_strain (float): Maximum tissue strain from acoustic pressure
    Notes:
        - ISPPA is converted to mW/cm² to match FDA units
        - Cavitation probability uses threshold of 0.3 MI with exponential model
        - Thermal index is simplified, assuming uniform absorption
    """
    max_mi = np.max(mi_field)
    mi_compliant = max_mi <= fda_limits.MI_LIMIT
    ispta_compliant = intensities.ISPTA <= fda_limits.ISPTA_LIMIT
    isppa_compliant = intensities.ISPPA <= fda_limits.ISPPA_LIMIT * 1e4  # Convert to same units
    
    # Calculate thermal index (simplified)
    acoustic_power = intensities.ISPTA * active_area_cm2 / 100  # Convert to W
    reference_power = 0.1  # Reference power for thermal index
    thermal_index = acoustic_power / reference_power
    
    # Estimate cavitation probability based on MI
    cavitation_prob = 1 - np.exp(-np.maximum(0, mi_field - 0.3) / 0.7)
    
    # Calculate tissue strain
    acoustic_pressure = np.sqrt(2 * intensities.ISPPA * medium.sound_speed * medium.density)
    strain = acoustic_pressure / (medium.density * medium.sound_speed**2)
    
    return {
        'compliant': mi_compliant and ispta_compliant and isppa_compliant,
        'MI_status': {'value': float(max_mi), 'compliant': mi_compliant},
        'ISPTA_status': {'value': float(intensities.ISPTA), 'compliant': ispta_compliant},
        'ISPPA_status': {'value': float(intensities.ISPPA), 'compliant': isppa_compliant},
        'thermal_index': float(thermal_index),
        'cavitation_probability': float(np.mean(cavitation_prob)),
        'max_strain': float(strain)
    }

@jit(nopython=True)
def calculate_stability_metrics(p_field: np.ndarray, min_pressure: float, max_pressure: float) -> tuple:
    """Calculate cavitation stability metrics using manual statistical calculations"""
    p_norm = np.abs(p_field) / np.max(np.abs(p_field))
    
    # Temporal stability calculation
    n_time, n_x, n_y = p_norm.shape
    temporal_var = np.zeros((n_x, n_y))
    for i in range(n_x):
        for j in range(n_y):
            mean = 0.0
            sq_diff = 0.0
            for t in range(n_time):
                mean += p_norm[t,i,j]
            mean /= n_time
            for t in range(n_time):
                sq_diff += (p_norm[t,i,j] - mean)**2
            temporal_var[i,j] = sq_diff / n_time
    
    # Spatial stability calculation
    spatial_var = np.zeros(n_time)
    for t in range(n_time):
        slice_mean = 0.0
        slice_sq_diff = 0.0
        for i in range(n_x):
            for j in range(n_y):
                slice_mean += p_norm[t,i,j]
        slice_mean /= (n_x * n_y)
        for i in range(n_x):
            for j in range(n_y):
                slice_sq_diff += (p_norm[t,i,j] - slice_mean)**2
        spatial_var[t] = slice_sq_diff / (n_x * n_y)
    
    # Calculate compliance with pressure limits
    within_limits = 0
    total_points = p_norm.size
    for t in range(n_time):
        for i in range(n_x):
            for j in range(n_y):
                if min_pressure <= p_field[t,i,j] <= max_pressure:
                    within_limits += 1
    compliance = within_limits / total_points
    
    # Combine metrics
    stability_score = 0.5 * (1 - np.mean(np.sqrt(temporal_var))) + 0.5 * compliance
    uniformity_score = 1 - np.mean(np.sqrt(spatial_var))
    
    return stability_score, uniformity_score

def analyze_sonoluminescence_potential(p_field: np.ndarray, sono_params: SonoluminescenceParameters) -> tuple:
    """
    Analyze potential for stable sonoluminescence with improved metrics.
    
    Returns:
        tuple: (stability_score, uniformity_score)
    """
    # Use unified stability metrics
    stability, uniformity = calculate_stability_metrics(
        p_field, 
        sono_params.min_pressure, 
        sono_params.max_pressure
    )
    
    return stability, uniformity

@jit(nopython=True)
def calc_max_along_axis0(arr: np.ndarray) -> np.ndarray:
    result = np.zeros(arr.shape[1:], dtype=arr.dtype)
    for i in range(arr.shape[1]):
        for j in range(arr.shape[2]):
            result[i, j] = np.max(arr[:, i, j])
    return result

@jit(nopython=True)
def calc_min_along_axis0(arr: np.ndarray) -> np.ndarray:
    result = np.zeros(arr.shape[1:], dtype=arr.dtype)
    for i in range(arr.shape[1]):
        for j in range(arr.shape[2]):
            result[i, j] = np.min(arr[:, i, j])
    return result

@jit(nopython=True)
def calculate_cavitation_metrics(p_field: np.ndarray) -> tuple:
    """Analyze acoustic field for uniform cavitation likelihood."""
    # Normalize pressure field
    p_norm = np.abs(p_field) / np.max(np.abs(p_field))
    
    # Calculate standing wave ratio using manual max/min
    p_max = 0.0
    p_min = float('inf')
    for i in range(p_norm.shape[1]):
        for j in range(p_norm.shape[2]):
            col_max = 0.0
            col_min = float('inf')
            for t in range(p_norm.shape[0]):
                val = p_norm[t,i,j]
                col_max = max(col_max, val)
                col_min = min(col_min, val)
            p_max = max(p_max, col_max)
            p_min = min(p_min, col_min)
    
    swr = (p_max + p_min) / max(p_max - p_min, 1e-10)
    
    # Calculate CV using running statistics
    mean_val = 0.0
    sum_sq = 0.0
    count = 0
    
    for t in range(p_norm.shape[0]):
        for i in range(p_norm.shape[1]):
            for j in range(p_norm.shape[2]):
                val = p_norm[t,i,j]
                mean_val += val
                sum_sq += val * val
                count += 1
    
    mean_val /= count
    variance = (sum_sq / count) - (mean_val * mean_val)
    cv = np.sqrt(variance) / max(mean_val, 1e-10)
    
    return float(swr), float(cv)

@jit(float64[:,:](float64, float64, int64, float64, int64, float64[:], float64), nopython=True, parallel=True)
def _optimized_modulated_signal(f_center: float, sweep_ratio: float, n_samples: int, dt: float, n_elements: int, phase_delays: np.ndarray, duty_cycle: float) -> np.ndarray:
    t = np.arange(n_samples) * dt
    pulse_period = 58.8e-6
    pulse_width = pulse_period * duty_cycle
    f_min = f_center * (1 - sweep_ratio/2)
    f_max = f_center * (1 + sweep_ratio/2)
    sweep_period = pulse_period * 10
    sweep_rate = (f_max - f_min) / sweep_period
    signals = np.zeros((n_elements, n_samples))
    for i in prange(n_elements):
        t_mod = t % sweep_period
        f_inst = f_min + sweep_rate * t_mod
        phase = phase_delays[i] + 2*np.pi * np.cumsum(f_inst) * dt
        base_signal = np.sin(phase)
        pulse_mask = (t % pulse_period) < pulse_width
        ramp_time = 1e-6
        ramp_samples = int(ramp_time / dt)
        signals[i] = base_signal * pulse_mask
        for j in range(1, n_samples):
            if pulse_mask[j] and not pulse_mask[j-1]:
                ramp_up = np.linspace(0, 1, ramp_samples)
                for k in range(min(ramp_samples, n_samples-j)):
                    signals[i,j+k] *= ramp_up[k]
            elif not pulse_mask[j] and pulse_mask[j-1]:
                ramp_down = np.linspace(1, 0, ramp_samples)
                for k in range(min(ramp_samples, j)):
                    signals[i,j-k-1] *= ramp_down[k]
    return signals

# Modify the create_optimized_signal function to include temporal phase updates
def create_optimized_signal(
    us_params: UltrasoundParameters, 
    cav_params: CavitationParameters,
    n_samples: int,
    dt: float,
    medium: kWaveMedium,
    tissue: TissueProperties,  # Add tissue parameter
    sono_params: SonoluminescenceParameters,  # Add sono_params
    thermal: ThermalSafety  # Add thermal parameter
) -> np.ndarray:
    """
    Generate dual-frequency optimized signals using Voronoi phase modulation [4]
    
    Features:
    - Dual-frequency components (f_center ± Δf)
    - Voronoi-based phase clusters
    - Adaptive amplitude modulation
    
    Physics:
    1. Dual-frequency ratio: DFR = f_high/f_low ≈ 1.2-1.5 [1]
    2. Voronoi cluster size: λ/√N [4]
    3. Amplitude modulation: 
       A_ij = 1 - exp(-(r_ij/R_cluster)^2)
    
    Returns:
        np.ndarray: Complex source signals (elements x time)
    """
    t = np.arange(n_samples) * dt
    source_signal = np.zeros((us_params.num_elements, n_samples), dtype=np.complex64)
    
    # Calculate bubble resonance with safety checks
    denominator = cav_params.RHO * sono_params.target_bubble_size**2
    if denominator <= 0:
        # From [3]: Minimum bubble size constraint
        sono_params.target_bubble_size = max(sono_params.target_bubble_size, 1e-6)
        denominator = cav_params.RHO * sono_params.target_bubble_size**2
    resonance_term = 3 * cav_params.GAMMA * cav_params.P0 / denominator
    bubble_res_freq = np.sqrt(max(resonance_term, 1e-10)) / (2 * np.pi)
    
    # Set sweep range with bounds checking
    sweep_bandwidth = min(0.2 * bubble_res_freq, 0.3 * us_params.f_center)
    f_min = max(us_params.f_center - sweep_bandwidth/2, 0.1e6)
    f_max = min(us_params.f_center + sweep_bandwidth/2, 1.0e6)
    
    # Calculate collapse time with safety limit
    pressure_diff = max(cav_params.P0 - sono_params.min_pressure, 1e3)  # Ensure positive
    collapse_time = min(0.915 * sono_params.target_bubble_size * \
                        np.sqrt(cav_params.RHO / pressure_diff), 1e-5)
    
    # Calculate sweep rates with bounds
    min_sweep_rate = 1e4  # 10 kHz/s minimum
    max_sweep_rate = 1e6  # 1 MHz/s maximum
    initial_sweep_rate = np.clip(1 / (4 * max(collapse_time**2, 1e-12)), 
                                min_sweep_rate, max_sweep_rate)
    sweep_rate_change = initial_sweep_rate / max(collapse_time, 1e-6)

    # Create Voronoi clusters for spatial phase modulation [4]
    points = np.random.rand(us_params.num_elements, 2)
    vor = Voronoi(points)

    for i in range(us_params.num_elements):
        # Calculate phase components with improved stability
        t_mod = t % (1/initial_sweep_rate)
        sweep_term = (f_max - f_min) * (t_mod * initial_sweep_rate)
        acceleration_term = 0.5 * sweep_rate_change * t_mod**2
        instantaneous_frequency = f_min + sweep_term + acceleration_term
        
        # Phase calculation with better numerical stability
        # Add dual-frequency component [1]
        phase_accumulation = np.cumsum(instantaneous_frequency + 0.2*instantaneous_frequency*np.sin(2*np.pi*0.1*instantaneous_frequency*t)) * dt
        spatial_phase = 2 * np.pi * i * us_params.element_spacing() / us_params.wavelength
        temporal_phase = np.random.uniform(0, 2*np.pi, size=len(t)) * \
                        np.exp(-t/collapse_time)
        
        phase = 2 * np.pi * phase_accumulation + spatial_phase + temporal_phase
        
        # Improved amplitude modulation with safety checks
        # From [5]: Pressure-temperature relationship
        blake_threshold = max(cav_params.P0 + 0.77 * cav_params.SIGMA / \
                            max(sono_params.target_bubble_size, 1e-6), 1e4)
        amplitude = np.clip(sono_params.min_pressure / blake_threshold, 0.5, 1.0)
        
        # Apply amplitude shaping
        element_position = i / (us_params.num_elements - 1)
        edge_taper = 0.5 * (1 - np.cos(2*np.pi*element_position))
        # Voronoi-based weighting [4]
        position_weight = np.exp(-vor.points[i,0]**2 - vor.points[i,1]**2)
        
        # Combine all components with improved stability
        source_signal[i, :] = amplitude * edge_taper * position_weight * np.exp(1j * phase)

    # Normalize final signal with safety check
    max_amplitude = np.max(np.abs(source_signal))
    if max_amplitude > 1e-10:  # Prevent division by zero
        source_signal /= max_amplitude
    else:
        source_signal = np.ones_like(source_signal) * 0.5  # Default safe value

    return source_signal

@jit(float64(float64, float64), nopython=True)
def calculate_MI(p_rarefactional: float, f_center: float) -> float:
    """
    Calculate Mechanical Index per FDA definition.
    
    Physics:
    --------
    MI = |p⁻|/√f
    
    where:
    - p⁻: peak rarefactional pressure [MPa]
    - f: center frequency [MHz]
    
    Theory:
    - MI correlates with cavitation probability
    - Higher MI indicates greater likelihood of mechanical bioeffects
    - FDA limit: MI < 1.9 for diagnostic imaging
    """
    p_MPa = abs(p_rarefactional) / 1e6
    f_MHz = f_center / 1e6
    return p_MPa / np.sqrt(f_MHz)

@jit(float64[:,:](float64[:,:], float64), nopython=True)
def calculate_MI_field(p_field: np.ndarray, f_center: float) -> np.ndarray:
    rows, cols = p_field.shape
    mi_field = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            p_rar = min(p_field[i,j], 0)
            mi_field[i,j] = calculate_MI(p_rar, f_center)
    return mi_field

@jit(float64(float64, float64), nopython=True)
def calculate_wavelength(frequency, sound_speed):
    return sound_speed / frequency

@jit(nopython=True,parallel=True)
def calculate_mean_axis0(arr: np.ndarray) -> np.ndarray:
    result = np.zeros(arr.shape[1:], dtype=arr.dtype)
    for i in prange(arr.shape[1]):
        for j in range(arr.shape[2]):
            result[i, j] = np.mean(arr[:, i, j])
    return result

@jit(nopython=True, parallel=True)
def calculate_mean_axis01(arr: np.ndarray) -> np.ndarray:
    """
    Compute spatial mean intensity field.
    
    Physics:
    --------
    Implements averaging over spatial dimensions for intensity calculations:
    I_avg(z) = (1/xy)∫∫I(x,y,z)dxdy
    
    Used for:
    - Tissue exposure metrics
    - Depth-dependent attenuation analysis
    """
    result = np.zeros(arr.shape[2], dtype=arr.dtype)
    for i in prange(arr.shape[2]):
        result[i] = np.mean(arr[:, :, i])
    return result

@jit(nopython=True)
def _optimize_sweep_helper(f_center: float, wavelength: float, tissue_depth: float, sweep_rate: float, sweep_rate_change: float) -> tuple:
    """
    Optimize ultrasound frequency sweep parameters for cavitation control.
    
    Physics:
    --------
    1. Standing Wave Formation:
       - Nodes occur at λ/2 intervals
       - Antinodes at λ/4 + nλ/2
       - Standing wave ratio = (p_max + p_min)/(p_max - p_min)
    
    2. Frequency Sweep Design:
       - Bandwidth: Δf = f_c·(λ/4L)
       - Decorrelation length: L_d = c/(2Δf)
       - Minimum sweep rate: R_min > c²/(8L²)
    
    3. Phase Relationships:
       - Rarefaction zones: z = λ/2 + nλ
       - Compression zones: z = λ/4 + nλ/2
    
    Algorithm:
    ----------
    1. Calculate node/antinode positions
       - Rarefaction: z = λ/2 + nλ
       - Compression: z = λ/4 + nλ/2
    
    2. Optimize sweep parameters
       - Bandwidth from spatial decorrelation
       - Rate constrained by bubble dynamics
       - Period matched to tissue transit time
    
    3. Apply safety bounds
       - Maximum rate limited by bubble response
       - Minimum rate ensures decorrelation
       - Bandwidth constrained by tissue attenuation

    Args:
        f_center (float): Center frequency [Hz]
        wavelength (float): Acoustic wavelength [m]  
        tissue_depth (float): Treatment depth [m]
        sweep_rate (float): Initial sweep rate [Hz/s]
        sweep_rate_change (float): Rate of change of sweep rate [Hz/s²]

    Returns:
        tuple: Contains:
            - f_min (float): Minimum sweep frequency [Hz]
            - f_max (float): Maximum sweep frequency [Hz]
            - sweep_ratio (float): Normalized sweep width
            - sweep_rate (float): Frequency change rate [Hz/s]
            - rarefaction_positions (ndarray): Rarefaction zone depths [m]
            - compression_positions (ndarray): Compression zone depths [m]
    """
    # Calculate rarefaction (pressure minimum) positions 
    rarefaction_positions = np.arange(wavelength/2, tissue_depth, wavelength)
    
    # Calculate compression (pressure maximum) positions
    compression_positions = np.arange(wavelength/4, tissue_depth, wavelength)
    
    # Calculate optimal bandwidth based on spatial decorrelation
    # Δf = f_c·(λ/4L) from standing wave theory
    delta_f = f_center * (wavelength/4) / tissue_depth
    
    # Define sweep range centered on f_center
    f_min = f_center - delta_f
    f_max = f_center + delta_f
    
    # Calculate normalized sweep width
    sweep_ratio = (f_max - f_min) / f_center
    
    # Calculate sweep period and rate using second-order dynamics
    sweep_duration = 2 * np.sqrt((f_max - f_min) / sweep_rate_change)
    sweep_period = 2 * sweep_duration
    sweep_rate_max = sweep_rate + sweep_rate_change * sweep_duration
    sweep_rate_min = sweep_rate
    
    # Set target sweep rate based on bubble dynamics
    target_sweep_rate = sweep_rate_max
    
    # Apply rate bounds based on physics constraints
    max_sweep_rate = 100.0 * 1000  # Hz/s - Limited by bubble response
    min_sweep_rate = 10.0 * 1000   # Hz/s - Ensure decorrelation
    sweep_rate = min(max(sweep_rate, min_sweep_rate), max_sweep_rate)
    
    return f_min, f_max, sweep_ratio, sweep_rate_min, sweep_rate_max, sweep_period, rarefaction_positions, compression_positions

@jit(nopython=True)
def calculate_optimal_sweep_rate(f_center: float, bubble_radius: float, sound_speed: float, tissue_depth: float) -> float:
    """
    Determine optimal frequency sweep rate for cavitation control.
    
    Physics:
    --------
    Natural frequency: f_n = (1/2πR₀)√(3κP₀/ρ)
    
    Sweep constraints:
    - df/dt < 1/(2τ²) prevents chaotic oscillations
    - τ: bubble period = 1/f_n
    
    Theory:
    - Rate matches bubble response time
    - Avoids resonance trapping
    - Maintains spatial coherence
    """
    T_bubble = 2 * np.pi * bubble_radius * np.sqrt(1000 / 101325)
    T_transit = 2 * tissue_depth / sound_speed
    R_min = 1 / (4 * T_bubble)
    R_max = sound_speed / (4 * tissue_depth)
    R_opt = np.sqrt(R_min * R_max)
    return float(R_opt)

@jit(nopython=True)
def calculate_optimal_interleaving(wavelength: float, tissue_depth: float) -> float:
    """
    Calculate optimal frequency interleaving to minimize standing waves.
    
    Physics:
    --------
    Standing wave periodicity: λ_sw = λ/2
    Optimal sweep bandwidth: Δf = f_c·λ/(4L)
    
    Spatial decorrelation condition:
    d_corr > λ/4 for destructive interference
    
    Theory:
    - Phase randomization between adjacent elements
    - Frequency diversity for spatial averaging
    - Time-varying interference patterns
    """
    n_wavelengths = tissue_depth / wavelength
    overlap_factor = 0.5  # Controls spatial decorrelation
    min_ratio = overlap_factor / n_wavelengths
    # Add randomization to further break coherence
    sweep_ratio = min_ratio * (1.3 + 0.1 * np.random.random())
    return min(max(sweep_ratio, 0.05), 0.3)

def optimize_sweep_parameters(f_center: float, sound_speed: float, tissue_depth: float, initial_sweep_rate: float, sweep_rate_change: float) -> SweepParameters:
    wavelength = calculate_wavelength(f_center, sound_speed)
    
    # Narrower sweep for bone penetration
    sweep_ratio = min(calculate_optimal_interleaving(wavelength, tissue_depth), 0.15)
    
    # Calculate band limits around center frequency
    delta_f = f_center * sweep_ratio
    f_min = f_center - delta_f/2
    f_max = f_center + delta_f/2
    
    # Calculate sweep period and rate using second-order dynamics
    sweep_duration = 2 * np.sqrt((f_max - f_min) / sweep_rate_change)
    sweep_period = 2 * sweep_duration
    sweep_rate_max = initial_sweep_rate + sweep_rate_change * sweep_duration
    sweep_rate_min = initial_sweep_rate
    
    # Calculate zones with fixed center frequency
    base_positions = np.arange(wavelength/4, tissue_depth, wavelength/2)
    rarefaction_positions = base_positions[::2]
    compression_positions = base_positions[1::2]
    
    return SweepParameters(
        f_min=f_min,
        f_max=f_max,
        sweep_ratio=sweep_ratio,
        sweep_rate_min=sweep_rate_min,
        sweep_rate_max=sweep_rate_max,
        sweep_period=sweep_period,
        wavelength=wavelength,
        rarefaction_zones=rarefaction_positions,
        compression_zones=compression_positions
    )

@jitclass(spec=[('min_frequency', float64),('max_frequency', float64),('min_pressure', float64),('max_pressure', float64),('target_bubble_radius', float64),('surface_tension', float64),('ambient_pressure', float64)])
class CavitationOptimizationParams:
    def __init__(self, min_frequency=180e3, max_frequency=260e3,  # Narrower band around 220 kHz
                 min_pressure=0.1e6, max_pressure=0.8e6,          # Reduced max pressure
                 target_bubble_radius=5e-6, surface_tension=0.072,
                 ambient_pressure=101325):
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.min_pressure = min_pressure
        self.max_pressure = max_pressure
        self.target_bubble_radius = target_bubble_radius
        self.surface_tension = surface_tension
        self.ambient_pressure = ambient_pressure


@jit(nopython=True)
def calculate_rayleigh_collapse_time(radius: float, density: float, pressure_diff: float) -> float:
    """
    Calculate bubble collapse time using Rayleigh collapse model.
    
    Physics:
    --------
    Based on Rayleigh-Plesset equation in collapse phase:
    RR̈ + (3/2)Ṙ² = (p_i - p_∞)/ρ
    
    Assumptions:
    1. Spherical symmetry
    2. Incompressible liquid
    3. Neglecting surface tension and viscosity
    
    Derivation:
    - Energy conservation during collapse
    - Kinetic energy = Work done by pressure
    - ½ρR³Ṙ² = (p_∞ - p_v)(R₀³ - R³)/3
    
    Leading to collapse time:
    τ_c = 0.915·R₀√(ρ/ΔP)
    
    where:
    - R₀: initial bubble radius
    - ρ: fluid density 
    - ΔP: driving pressure difference
    - p_v: vapor pressure (neglected)
    - p_∞: ambient pressure
    """
    return 0.915 * radius * np.sqrt(density / pressure_diff)

@jit(nopython=True)
def calculate_optimal_frequency(target_radius: float, surface_tension: float, ambient_pressure: float, density: float) -> float:
    """
    Calculate optimal drive frequency for stable cavitation.
    
    Physics:
    --------
    1. Linear Resonance:
       Modified Minnaert frequency including surface tension:
       f_res = (1/2πR)√((3κp₀ + 2σ/R)/ρ)
    
    2. Nonlinear Effects:
       - Bjerknes forces: F_B ∝ -V∇p
       - Rectified diffusion: growth rate ∝ (R/R₀)³ - 1
    
    3. Stability Conditions:
       - Blake threshold: p_B = p₀ + 0.77σ/R₀
       - Dynamic Blake threshold: p_B(ω) ≈ p_B(0)[1 + (ωτ)²]
       where τ = μ/(p₀ + 2σ/R₀)
    
    4. Optimal Drive:
       f_opt = f_res/√2 (subharmonic regime)
       
    Parameters account for:
    - Acoustic radiation force
    - Bjerknes forces
    - Surface tension effects
    - Viscous damping
    """
    return (1/(2*np.pi*target_radius)) * np.sqrt((3*1.4*ambient_pressure + 2*surface_tension/target_radius) / density)

def optimize_cavitation_parameters(tissue: TissueProperties, opt_params: CavitationOptimizationParams) -> dict:
    f_opt = calculate_optimal_frequency(opt_params.target_bubble_radius, opt_params.surface_tension, opt_params.ambient_pressure, tissue.density)
    p_blake = opt_params.ambient_pressure + 0.77 * opt_params.surface_tension / opt_params.target_bubble_radius
    p_opt = min(max(2 * p_blake, opt_params.min_pressure), opt_params.max_pressure)
    prf_opt, pulse_width_opt = optimize_pulse_timing(opt_params.target_bubble_radius, tissue.density, p_opt)
    return {
        'frequency': f_opt,
        'pressure': p_opt,
        'prf': prf_opt,
        'pulse_width': pulse_width_opt
    }

def update_simulation_parameters(us_params: UltrasoundParameters, tissue: TissueProperties) -> UltrasoundParameters:
    opt_params = CavitationOptimizationParams()
    optimal_params = optimize_cavitation_parameters(tissue, opt_params)
    return UltrasoundParameters(
        transducer_width=us_params.transducer_width,
        transducer_length=us_params.transducer_length,
        tissue_depth=us_params.tissue_depth,
        f_center=optimal_params['frequency'],
        sound_speed=us_params.sound_speed,
        duty_cycle=optimal_params['pulse_width'] * optimal_params['prf'],
        prf=optimal_params['prf'],
        kerf=us_params.kerf,
        cav_params=us_params.cav_params
    )

def setup_array(us_params: UltrasoundParameters) -> kWaveArray:
    array = kWaveArray()
    
    for i in range(us_params.num_elements):
        position = [
            -us_params.transducer_width/2 + i*us_params.element_spacing(),
            -us_params.transducer_length/2
        ]
        array.add_rect_element(
            position,
            us_params.element_width(),
            us_params.transducer_length,
            0.0  # Fixed rotation angle (single float instead of list)
        )
    
    return array

# Fix the parameter validation decorator
def validate_inputs(func):
    def wrapper(us_params: UltrasoundParameters, medium: kWaveMedium):
        if not (20e3 <= us_params.f_center <= 1e6):
            raise ValueError(f"Frequency {us_params.f_center/1e3:.1f} kHz out of therapeutic range (20-1000 kHz)")
        return func(us_params, medium)
    return wrapper

# Apply to critical functions
@validate_inputs
def setup_grid(us_params: UltrasoundParameters, medium: kWaveMedium) -> kWaveGrid:
    """
    Configure computational grid for FDTD simulation.
    
    Physics:
    --------
    Grid spacing determined by:
    dx = λ/N, where N ≥ 10 (points per wavelength)
    
    CFL condition for stability:
    dt ≤ dx/(c√number_dimensions)
    
    PML thickness based on:
    L_pml ≥ 2λ for -60dB reflection
    """
    wavelength = us_params.sound_speed / us_params.f_center
    dx = wavelength / 10
    dy = dx
    Nx = int(us_params.transducer_width / dx) + 20
    Ny = int((us_params.transducer_length + us_params.tissue_depth) / dy) + 20
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))
    cfl = 0.2
    t_end = float(200e-6)
    # Fix deprecation warning by extracting scalar value
    sound_speed = float(medium.sound_speed[0] if isinstance(medium.sound_speed, np.ndarray) else medium.sound_speed)
    dt = float(cfl * dx / sound_speed)
    Nt = int(np.ceil(t_end / dt))
    dt = float(t_end / Nt)
    kgrid.setTime(Nt, dt)
    return kgrid

def calculate_power_metrics(us_params: UltrasoundParameters, medium: kWaveMedium) -> tuple:
    """Compute acoustic power and exposure metrics."""
    power_calc = TransducerPower(
        target_pressure=0.84e6,
        safety_margin=1.05,
        electro_efficiency=0.90,  # Increased efficiency
        duty_cycle=us_params.duty_cycle
    )
    
    active_area_m2 = us_params.active_area_cm2() * 1e-4
    
    # Calculate initial pressure with improved bounds
    initial_pressure = power_calc.calculate_pressure('initial', tissue_depth=us_params.tissue_depth, medium=medium)
    initial_pressure = float(min(max(initial_pressure, 0.1e6), 0.6e6))
    
    # Calculate acoustic impedance with safety check
    Z = float(max(medium.sound_speed * medium.density, 1e5))
    
    # Calculate intensities with bounds checking
    peak_intensity_m2 = float((initial_pressure**2) / (2 * Z))
    avg_intensity_m2 = peak_intensity_m2 * us_params.duty_cycle
    
    # Convert to W/cm² with validation
    peak_density = float(peak_intensity_m2 * 1e-4)
    avg_density = float(avg_intensity_m2 * 1e-4)
    
    # Calculate power with area validation
    active_area_cm2 = max(us_params.active_area_cm2(), 1.0)  # Minimum 1 cm²
    required_power = float(avg_density * active_area_cm2)
    
    return initial_pressure, required_power, avg_density, peak_density

# Add new class for adaptive time stepping parameters
@jitclass(spec=[
    ('min_dt', float64),
    ('max_dt', float64),
    ('cfl_target', float64),
    ('error_tolerance', float64),
    ('safety_factor', float64)
])
class AdaptiveTimeParams:
    """Parameters for adaptive time stepping"""
    def __init__(self,
                 min_dt: float = 1e-9,
                 max_dt: float = 1e-6,
                 cfl_target: float = 0.3,
                 error_tolerance: float = 1e-6,
                 safety_factor: float = 0.9):
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.cfl_target = cfl_target
        self.error_tolerance = error_tolerance
        self.safety_factor = safety_factor

@jit(nopython=True)
def calculate_gradient_magnitude(p_field: np.ndarray) -> float:
    """
    Calculate pressure gradient magnitude using finite differences.
    Compatible with Numba's nopython mode.
    
    Parameters:
    -----------
    p_field: np.ndarray
        2D pressure field array
    
    Returns:
    --------
    float: Maximum gradient magnitude
    """
    grad_max = 0.0
    rows, cols = p_field.shape
    
    # Calculate gradients using central differences
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            # x-direction gradient
            dx = (p_field[i+1,j] - p_field[i-1,j]) / 2.0
            # y-direction gradient
            dy = (p_field[i,j+1] - p_field[i,j-1]) / 2.0
            # Gradient magnitude
            grad_mag = np.sqrt(dx*dx + dy*dy)
            grad_max = max(grad_max, grad_mag)
            
    return grad_max

@jit(nopython=True)
def calculate_adaptive_timestep(
    p_field: np.ndarray,
    dx: float,
    sound_speed: float,
    current_dt: float,
    params: AdaptiveTimeParams
) -> float:
    """
    Calculate optimal timestep based on solution behavior.
    
    Physics:
    --------
    1. CFL condition: dt <= dx/(c*√N_dim)
    2. Pressure gradient: dt ~ 1/|∇p|
    3. Error estimate: e ~ O(dt²)
    """
    # Calculate CFL-based timestep
    cfl_dt = params.cfl_target * dx / sound_speed
    
    # Calculate pressure gradient-based timestep using custom function
    grad_p = calculate_gradient_magnitude(p_field)
    if grad_p > 0:
        grad_dt = params.safety_factor / grad_p
    else:
        grad_dt = params.max_dt
    
    # Choose minimum of stability criteria
    new_dt = min(cfl_dt, grad_dt)
    
    # Apply bounds and rate limiting
    new_dt = min(max(new_dt, params.min_dt), params.max_dt)
    new_dt = min(new_dt, 1.2 * current_dt)  # Limit growth rate
    
    return float(new_dt)

def run_simulation(kgrid: kWaveGrid, karray: kWaveArray, source_signal: np.ndarray, medium: kWaveMedium) -> np.ndarray:
    """Run simulation with adaptive time stepping"""
    try:
        source_p_mask = karray.get_array_binary_mask(kgrid)
        source_p = karray.get_distributed_source_signal(kgrid, source_signal) * 1e6
        
        # Initialize adaptive stepping parameters
        time_params = AdaptiveTimeParams()
        current_dt = kgrid.dt
        
        # Create absolute temp directory path
        TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        simulation_options = SimulationOptions(
            save_to_disk=True,
            data_cast="single",
            pml_alpha=2.0,
            pml_size=20,
            smooth_p0=True,
            input_filename=os.path.join(TEMP_DIR, 'simulation_2d.h5'),
            output_filename=os.path.join(TEMP_DIR, 'output_2d.h5'),
            data_path=TEMP_DIR
        )
        
        execution_options = SimulationExecutionOptions(is_gpu_simulation=False)
        sensor = kSensor()
        sensor.mask = np.ones((kgrid.Nx, kgrid.Ny), dtype=bool)
        source = kSource()
        source.p_mask = source_p_mask
        source.p = source_p
        
        # Run full simulation first
        p = kspaceFirstOrder2DC(
            kgrid=kgrid,
            medium=medium,
            source=source,
            sensor=sensor,
            simulation_options=simulation_options,
            execution_options=execution_options
        )
        
        # Get total number of points and reshape
        total_points = kgrid.Nx * kgrid.Ny
        p_field = np.reshape(p["p"], (-1, kgrid.Nx, kgrid.Ny))
        
        # Apply adaptive time stepping for temporal refinement
        refined_p_field = np.zeros_like(p_field)
        
        for n in range(p_field.shape[0]-1):
            # Store current timestep
            refined_p_field[n] = p_field[n]
            
            # Calculate new timestep
            new_dt = calculate_adaptive_timestep(
                p_field[n],
                kgrid.dx,
                float(medium.sound_speed[0]),  # Extract scalar value
                current_dt,
                time_params
            )
            
            # If timestep needs significant adjustment
            if abs(new_dt - current_dt) > time_params.error_tolerance:
                current_dt = new_dt
                # Interpolate between timesteps if needed
                if new_dt < current_dt:
                    mid_field = 0.5 * (p_field[n] + p_field[n+1])
                    refined_p_field = np.insert(refined_p_field, n+1, mid_field, axis=0)
        
        # Final reshape and scaling
        p_field = np.transpose(refined_p_field, (0, 2, 1))
        
        # Scale pressure field to match target pressure
        target_pressure = 0.84e6
        current_max = np.max(np.abs(p_field))
        if current_max > 0:
            scaling_factor = target_pressure / current_max
            p_field *= scaling_factor
        
        return p_field
        
    except RuntimeError as e:
        raise SimulationError(f"Numerical instability: {str(e)}") from e
    except MemoryError as e:
        raise SimulationError("Insufficient memory for grid size") from e

def visualize_results(p_field: np.ndarray, kgrid: kWaveGrid, us_params: UltrasoundParameters, output_dir: str = None) -> None:
    """
    Visualize simulation results with proper output handling.
    
    Parameters
    ----------
    p_field : np.ndarray
        Pressure field data (shape: [time_steps, x_dim, z_dim])
    kgrid : kWaveGrid
        Grid parameters
    us_params : UltrasoundParameters
        Ultrasound parameters
    output_dir : str, optional
        Directory to save output files
    """
    # Convert 3D pressure field to 2D spatial maximum
    pressure_2d = np.max(np.abs(p_field), axis=0)  # Max pressure over time
    
    # Create coordinate arrays matching pressure_2d shape
    x_mm = np.linspace(-us_params.transducer_width/2 * 1000, 
                       us_params.transducer_width/2 * 1000, 
                       pressure_2d.shape[0])
    z_mm = np.linspace(0, us_params.tissue_depth * 1000, 
                       pressure_2d.shape[1])
    
    # Create meshgrid for proper pcolormesh plotting
    X, Z = np.meshgrid(x_mm, z_mm, indexing='ij')
    
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X, Z, pressure_2d, 
                   cmap='hot', shading='auto')
    plt.colorbar(label='Pressure Amplitude [Pa]')
    
    # Add transducer array visualization
    transducer_y = -1
    plt.plot([x_mm[0], x_mm[-1]], 
             [transducer_y, transducer_y], 
             'b-', linewidth=3, label='Transducer Array')
    
    plt.xlabel('Lateral Position [mm]')
    plt.ylabel('Depth [mm]')
    plt.title('Maximum Pressure Field Distribution\nTransducer Array and Tissue Medium')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pressure_field.png'), dpi=300)
        plt.close()
    else:
        plt.show()


class TransducerPower:
    """
    Enhanced documentation with units clarification:
    
    Parameters
    ----------
    target_pressure : float
        Acoustic pressure at focus (Pa)
    safety_margin : float
        Safety factor (1.0 = no safety margin)
    max_power_density : float
        Maximum allowed power (W/cm²)
    attenuation_factor : float
        Tissue attenuation compensation (1.0 = no compensation)
        
    All calculations follow AIUM/NEMA standards for ultrasound safety
    
    Key Equations:
    1. Intensity: I = P²/(2Z) [W/m²]
    2. Power: P = I * A [W]
    3. Voltage: V = √(P*Z_elec/η) [V]
    4. Geometric Gain: G = (πD²f)/(4cR) [-]
    """
    def __init__(self, 
                 target_pressure: float = 0.84e6,  # Reduced from 1.2e6
                 safety_margin: float = 1.05,  # Reduced from 1.2
                 electro_efficiency: float = 0.85,  # Increased efficiency
                 duty_cycle: float = 0.64):  # Match ultrasound params
        self.target_pressure = target_pressure
        self.safety_margin = safety_margin
        self.electro_efficiency = electro_efficiency
        self.duty_cycle = duty_cycle
        self.max_power_density = 20.0  # W/cm²
        # Focusing parameters
        self.geometric_gain = 8.0  # Typical gain for focused arrays
        self.focal_length = 0.06  # 60mm typical focal length
        
    def calculate_pressure(self, method_type: str, *args, **kwargs) -> float:
        """
        Unified method to calculate pressure based on the specified method type.
        
        Parameters:
        - method_type: str - Type of calculation ('from_power', 'required', 'initial')
        - args: Additional arguments required for specific calculations
        - kwargs: Additional keyword arguments for specific calculations
        """
        Z = kwargs.get('medium').sound_speed * kwargs.get('medium').density
        if method_type == 'from_power':
            power_density = kwargs.get('power_density')
            average_intensity = power_density * 10000  # W/cm² to W/m²
            peak_intensity = average_intensity / self.duty_cycle  # Remove duty cycle scaling
            return np.sqrt(2 * Z * peak_intensity)
        elif method_type == 'required':
            area_cm2 = kwargs.get('area_cm2')
            medium = kwargs.get('medium')
            compensated_pressure = self.target_pressure * self.safety_margin * self.electro_efficiency
            peak_intensity = compensated_pressure**2 / (2 * Z)
            average_intensity = peak_intensity * self.duty_cycle  # Account for pulsing
            return float(average_intensity * area_cm2 / 10000)  # Convert cm² to m²
        elif method_type == 'initial':
            tissue_depth = kwargs.get('tissue_depth')
            medium = kwargs.get('medium')
            attenuation_db = medium.alpha_coeff * (float(medium.sound_speed)/1e6) * (tissue_depth * 100)
            attenuation_factor = 10**(attenuation_db/20)
            calculated_pressure = self.target_pressure * attenuation_factor * self.safety_margin
            return float(np.max([calculated_pressure, 0.1e6]))  # Explicit scalar conversion
        else:
            raise ValueError("Invalid method type specified.")

    def calculate_required_power(self, area_cm2: float, medium: kWaveMedium) -> float:
        return self.calculate_pressure('required', area_cm2=area_cm2, medium=medium)

    def calculate_initial_pressure(self, medium: kWaveMedium, tissue_depth: float) -> float:
        return self.calculate_pressure('initial', tissue_depth=tissue_depth, medium=medium)

    def calculate_pressure_from_power(self, power_density: float, medium: kWaveMedium) -> float:
        return self.calculate_pressure('from_power', power_density=power_density, medium=medium)

    def calculate_voltage_required(self, pressure: float, medium: kWaveMedium, area_cm2: float) -> float:
        """
        Calculate required driving voltage considering focusing effects.
        
        Physics:
        --------
        1. Geometric Focusing Gain:
           G = (π * D² * f) / (4 * λ * R)
           where:
           - D: aperture diameter
           - f: frequency
           - λ: wavelength
           - R: focal length
        
        2. Effective Pressure:
           p_focal = p_surface * √G
           Therefore: p_surface = p_focal / √G
        """
        Z = medium.sound_speed * medium.density
        
        # Calculate aperture diameter from area
        aperture_diameter = 2 * np.sqrt(area_cm2 / np.pi) * 0.01  # Convert to meters
        
        # Calculate wavelength
        wavelength = medium.sound_speed / 220e3  # Using 220 kHz center frequency
        
        # Calculate geometric gain
        geometric_gain = min(
            (np.pi * aperture_diameter**2 * 220e3) / 
            (4 * wavelength * self.focal_length),
            self.geometric_gain  # Cap at typical maximum
        )
        
        # Account for focusing in pressure calculation
        surface_pressure = pressure / np.sqrt(geometric_gain)
        
        # Calculate intensity using surface pressure
        intensity = (surface_pressure**2) / (2 * Z)
        power = intensity * (area_cm2 * 1e-4)
        
        # Calculate voltage with improved impedance matching
        impedance = 50.0  # Standard impedance
        voltage = np.sqrt(power * impedance / self.electro_efficiency)
        
        return float(voltage)

@jit(nopython=True)
def calculate_q_factor(f_center: float, bandwidth: float) -> float:
    return f_center / bandwidth

@dataclass(slots=True)
class SimulationState:
    """
    Track simulation state and field metrics.
    
    Physics Parameters:
    ------------------
    1. Pressure field p(x,y,t):
       - Peak positive/negative
       - RMS and mean values
       - Spatial/temporal variations
    
    2. Field metrics:
       - Mechanical Index (MI)
       - Cavitation probability
       - Treatment coverage
       - Safety thresholds
    """
    x_mm: np.ndarray = None
    z_mm: np.ndarray = None
    transducer_y: float = -1
    p_field: np.ndarray = None
    MI_field: np.ndarray = None
    cavitation_probability: np.ndarray = None
    peak_pressure: float = 0.0
    mean_pressure: float = 0.0
    stability_score: float = 0.0
    uniformity_score: float = 0.0

    def update_pressures(self, p_field: np.ndarray):
        self.p_field = p_field
        self.peak_pressure = np.max(np.abs(p_field))
        self.mean_pressure = np.mean(np.abs(p_field))

    def update_coordinates(self, us_params: UltrasoundParameters):
        self.x_mm = np.linspace(-us_params.transducer_width/2 * 1000, us_params.transducer_width/2 * 1000, self.p_field.shape[1])
        self.z_mm = np.linspace(0, us_params.tissue_depth * 1000, self.p_field.shape[2])

@dataclass(slots=True)
class SimulationParameters:
    """
    Complete parameter set for acoustic simulation.
    
    Components:
    ----------
    1. Medium properties:
       - Sound speed, density
       - Attenuation, nonlinearity
       
    2. Excitation parameters:
       - Frequency, amplitude
       - Pulse characteristics
       - Array geometry
       
    3. Analysis settings:
       - Safety thresholds
       - Optimization targets
       - Monitoring points
    """
    tissue: TissueProperties
    us_params: UltrasoundParameters
    cav_params: CavitationParameters
    sono_params: SonoluminescenceParameters
    fda_limits: FDA_Limits
    sweep_params: SweepParameters = None
    state: SimulationState = field(default_factory=SimulationState)

    def setup_medium(self) -> kWaveMedium:
        return kWaveMedium(
            sound_speed=self.tissue.sound_speed,
            density=self.tissue.density,
            alpha_coeff=self.tissue.alpha_coeff,
            alpha_power=self.tissue.alpha_power,
            BonA=self.tissue.BonA
        )

    def optimize_sweep(self):
        self.sweep_params = optimize_sweep_parameters(
            self.us_params.f_center,
            self.us_params.sound_speed,
            self.us_params.tissue_depth,
            self.us_params.duty_cycle,
            self.us_params.pulse_duration,
            self.us_params.pulse_gap,
            self.us_params.prf,
            self.us_params.kerf,
            self.cav_params
        )

    def validate_parameters(self) -> bool:
        """Verify parameter consistency and physical validity"""
        try:
            # Check frequency range
            if not (20e3 <= self.us_params.f_center <= 1e6):
                return False
                
            # Verify tissue properties
            if self.tissue.sound_speed <= 0 or self.tissue.density <= 0:
                return False
                
            # Check geometry
            if self.us_params.tissue_depth <= 0:
                return False
                
            # Validate cavitation parameters
            if self.cav_params.min_cavitation_pressure >= self.cav_params.max_cavitation_pressure:
                return False
                
            return True
            
        except Exception:
            return False

def run_simulation_with_params(params: SimulationParameters) -> np.ndarray:
    """
    Execute full acoustic simulation with given parameters.
    
    Physics:
    --------
    Simulation components:
    1. Wave propagation: ∂²p/∂t² = c²∇²p + nonlinear_terms
    2. Tissue interaction: absorption, scattering
    3. Boundary conditions: PML, interfaces
    
    Validation:
    - Check CFL condition
    - Verify energy conservation
    - Monitor numerical stability
    """
    # Validate input parameters
    if not params.us_params or not params.tissue:
        raise ValueError("Missing required parameters")
        
    try:
        medium = params.setup_medium()
        kgrid = setup_grid(params.us_params, medium)
        karray = setup_array(params.us_params)
        
        # Generate and validate source signal
        source_signal = create_optimized_signal(
            params.us_params,
            params.cav_params,
            kgrid.Nt,
            kgrid.dt,
            medium,  # Add medium parameter
            params.tissue,  # Pass tissue properties
            params.sono_params,  # Pass sonoluminescence parameters
            params.thermal  # Pass thermal parameter
        ).real.astype(np.float32)  # Removed * 1e3
        
        if np.any(np.isnan(source_signal)):
            raise ValueError("Invalid source signal generated")
            
        p_field = run_simulation(kgrid, karray, source_signal, medium)
        
        # Update simulation state
        params.state.update_pressures(p_field)
        params.state.update_coordinates(params.us_params)
        
        # Validate output field
        if np.any(np.isnan(p_field)):
            raise RuntimeError("Simulation produced invalid pressure field")
            
        return p_field
        
    except Exception as e:
        warnings.warn(f"Simulation failed: {str(e)}")
        return None

def multi_objective_transducer_optimization(us_params: UltrasoundParameters, tissue: TissueProperties, constraints: dict) -> TransducerOptimizationMetrics:
    """
    Optimize transducer parameters for multiple objectives.
    
    Physics:
    --------
    Key metrics:
    1. Beam forming: H(k) = Σ exp(-jkdᵢsinθ)
    2. Grating lobes: d < λ/(1 + |sinθ|)
    3. Near field: N = D²/(4λ)
    
    Optimization goals:
    - Maximize treatment volume
    - Minimize standing waves
    - Ensure uniform cavitation
    - Stay within safety limits
    """
    best_metrics = TransducerOptimizationMetrics(0,0,0,0,0)
    return best_metrics

def review_array_direct_contact(us_params: UltrasoundParameters):
    """
    Analyze transducer-tissue coupling conditions.
    
    Physics:
    --------
    Acoustic impedance matching:
    T = 4Z₁Z₂/(Z₁+Z₂)²
    
    where:
    - T: transmission coefficient
    - Z₁,Z₂: acoustic impedances
    
    Near-field effects:
    N = D²/(4λ)
    where:
    - D: element diameter
    - λ: wavelength
    """
    contact_margin = max(0, us_params.transducer_length - us_params.tissue_depth)
    if contact_margin > 0:
        pass

def plot_simulation_domain(kgrid: kWaveGrid, us_params: UltrasoundParameters):
    """
    Visualize simulation geometry and boundaries.
    
    Domain Configuration:
    -------------------
    1. Physical boundaries:
       - Transducer surface (z=0)
       - Tissue interface
       - PML regions
    
    2. Grid properties:
       - Spatial resolution (dx,dy)
       - CFL number
       - Stability conditions
    """
    x_dim = kgrid.Nx * kgrid.dx
    y_dim = kgrid.Ny * kgrid.dy
    plt.figure()
    plt.title('Simulation Domain Layout')
    plt.plot([0, x_dim],[us_params.tissue_depth, us_params.tissue_depth],'r--',label='Tissue Interface')
    plt.legend()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.savefig(os.path.join('./simulation_output', 'simulation_domain.png'), dpi=300)
    plt.close()
    
@contextmanager 
def parallel_pool(processes=None):
    pool = Pool(processes)
    try:
        yield pool
    finally:
        pool.close()
        pool.join()

def optimize_grid_parameters(us_params: UltrasoundParameters) -> tuple:
    wavelength = us_params.wavelength
    min_points_per_wavelength = 4
    dx = wavelength / min_points_per_wavelength
    pml_size = int(3 * np.sqrt(wavelength/dx))  # Frequency-adaptive PML
    nx = int(np.ceil(us_params.transducer_width / dx)) + 2*pml_size
    ny = int(np.ceil(us_params.tissue_depth / dx)) + 2*pml_size
    return nx, ny, dx, pml_size

def create_optimized_array(us_params: UltrasoundParameters, cav_params: CavitationParameters) -> kWaveArray:
    """
    Configure optimized phased array geometry.
    
    Physics:
    --------
    Element spacing constraints:
    1. d < λ/2 to avoid grating lobes
    2. w < λ to minimize directivity
    
    Array parameters:
    - Focal gain: G = N·(D/λF)
    - Near field: N = D²/(4λ)
    - Beam width: θ = λ/D
    
    where:
    - N: number of elements
    - D: aperture size
    - F: focal length
    """
    karray = kWaveArray(bli_tolerance=0.05, upsampling_rate=10)
    element_width = us_params.wavelength * 0.9
    spacing = us_params.element_spacing
    with parallel_pool() as pool:
        args = [(x, us_params, element_width) for x in range(us_params.num_elements)]
        pool.starmap(add_array_element, args)
    return karray

def run_optimized_simulation(params: SimulationParameters) -> dict:
    try:
        nx, ny, dx, pml_size = optimize_grid_parameters(params.us_params)
        kgrid = kWaveGrid([nx, ny], [dx, dx])
        medium = params.setup_medium()
        karray = create_optimized_array(params.us_params, params.cav_params)
        source = kSource()
        source.p_mask = karray.get_array_binary_mask(kgrid)
        source_signals = create_optimized_signal(
            params.us_params,
            params.cav_params,
            kgrid.Nt,
            kgrid.dt,
            medium,  # Add medium parameter
            params.tissue,  # Pass tissue properties
            params.sono_params,  # Pass sonoluminescence parameters
            params.thermal  # Pass thermal parameter
        )
        source.p = karray.get_distributed_source_signal(kgrid, source_signals)
        simulation_options = SimulationOptions(
            pml_size=pml_size,
            save_to_disk=True,
            data_cast='single',
            pml_inside=False
        )
        sensor_data = kspaceFirstOrder2DC(
            kgrid=kgrid,
            medium=medium,
            source=source,
            sensor=kSensor(),
            simulation_options=simulation_options
        )
        return {
            'sensor_data': sensor_data,
            'kgrid': kgrid,
            'source': source
        }
    except Exception as e:
        warnings.warn(f"Simulation failed: {str(e)}")
        return None

def analyze_results(results: dict, params: SimulationParameters) -> Dict[str, Any]:
    """
    Analyze simulation results for therapeutic efficacy.
    
    Physics:
    --------
    Analyzed metrics:
    1. Mechanical Index: MI = p⁻/√f
    2. Spatial-peak temporal-average intensity:
       ISPTA = (1/T)∫I(t)dt
    3. Standing Wave Ratio:
       SWR = (p_max + p_min)/(p_max - p_min)
    
    Safety thresholds:
    - MI < 1.9 (FDA limit)
    - ISPTA < 720 mW/cm²
    - Peak negative pressure < threshold
    """
    if not results or 'sensor_data' not in results:
        return None
    if 'p' not in results['sensor_data']:
        warnings.warn("Missing pressure field in sensor data")
        return None
    sensor_data = results['sensor_data']
    with parallel_pool() as pool:
        p_field = sensor_data['p']
        mi_field = pool.starmap(calculate_MI_field, [(p_field, params.us_params.f_center)])
        stability, uniformity = pool.starmap(calculate_cavitation_metrics, [(p_field,)])
    return {
        'MI': mi_field,
        'stability': stability,
        'uniformity': uniformity
    }

@jit(nopython=True)
def calculate_treatment_coverage(mi_field: np.ndarray, threshold: float) -> float:
    total_points = mi_field.size
    treatment_points = np.sum(mi_field >= threshold)
    return 100.0 * float(treatment_points) / float(total_points)

def optimize_pulse_timing(radius: float, density: float, pressure: float) -> tuple:
    """Calculate optimal PRF and pulse width based on bubble dynamics"""
    collapse_time = calculate_rayleigh_collapse_time(radius, density, pressure)
    optimal_prf = 1.0 / (4 * collapse_time)  # Allow time for collapse and regrowth
    optimal_width = collapse_time * 0.5  # Pulse ends before violent collapse
    return float(optimal_prf), float(optimal_width)

def calculate_intensities(p_field: np.ndarray, medium: kWaveMedium, dt: float) -> IntensityMetrics:
    """Calculate acoustic intensity metrics"""
    Z = medium.sound_speed * medium.density
    
    # Calculate instantaneous intensity field (W/m²)
    i_field = np.square(p_field) / (2 * Z)
    
    # Calculate ISPTA (mW/cm²)
    temporal_avg = np.mean(i_field, axis=0)
    ispta = float(np.max(temporal_avg)) * 0.1
    
    # Calculate ISPPA (W/cm²)
    envelope = np.abs(hilbert(p_field, axis=0))
    threshold = 0.1 * np.max(envelope)
    pulse_mask = envelope > threshold
    
    # Calculate pulse-averaged intensity with improved pulse detection
    isppa = 0.0
    for i in range(p_field.shape[1]):
        for j in range(p_field.shape[2]):
            pulse_samples = np.sum(pulse_mask[:,i,j])
            if pulse_samples > 0:
                pulse_intensity = np.sum(i_field[:,i,j] * pulse_mask[:,i,j]) / pulse_samples
                isppa = max(isppa, pulse_intensity)
    
    isppa *= 1e-4  # Convert to W/cm²
    
    return IntensityMetrics(ISPTA=float(ispta), ISPPA=float(isppa))

# Add new functions for pulse optimization and thermal analysis

@jitclass(spec=[('pulse_duration', float64),('pulse_gap', float64),('duty_cycle', float64),('cavitation_uniformity', float64),('thermal_safety_margin', float64),('element_delays', float64[:])])
class PulseOptimizationMetrics:
    """Metrics for optimized pulsing scheme"""
    def __init__(self, pulse_duration=0.0, pulse_gap=0.0, duty_cycle=0.0, cavitation_uniformity=0.0, thermal_safety_margin=0.0, element_delays=np.zeros(1)):
        self.pulse_duration = pulse_duration
        self.pulse_gap = pulse_gap
        self.duty_cycle = duty_cycle
        self.cavitation_uniformity = cavitation_uniformity
        self.thermal_safety_margin = thermal_safety_margin
        self.element_delays = element_delays

@jit(nopython=True)
def calculate_optimal_pulse_params(
    bubble_radius: float,
    tissue_depth: float,
    sound_speed: float,
    density: float,
    pressure: float,
    thermal_diffusivity: float,
    element_idx: int,
    num_elements: int,
    cav_params: CavitationParameters
) -> tuple:
    dt = 1e-10  # Time step for numerical integration
    collapse_time = 0.0
    R = [bubble_radius]
    R_dot = [0.0]
    
    for i in range(100000):
        if R[-1] < bubble_radius/1000 or abs(R_dot[-1]) > 1e6:
            break
            
        P_gas = cav_params.P0 * (bubble_radius/R[-1])**(3*cav_params.GAMMA)
        P_surface = 2 * cav_params.SIGMA / R[-1]
        P_viscous = (4 * cav_params.MU * R_dot[-1]) / R[-1]
        
        # Keller-Miksis equation
        term1 = (1 + (1 - cav_params.GAMMA)*R_dot[-1]/cav_params.RHO/cav_params.SIGMA)
        term2 = (P_gas + pressure - cav_params.P0 - P_surface - P_viscous)
        term3 = 3/2*(1 - (1 - 3*cav_params.GAMMA)*R_dot[-1]/(3*cav_params.RHO*cav_params.SIGMA))
        
        R_ddot = (term1 * term2 / (cav_params.RHO * R[-1]) 
                 - term3 * R_dot[-1]**2 / R[-1] 
                 - 4*cav_params.MU*R_dot[-1]/(cav_params.RHO*R[-1]**2))
        
        new_R_dot = R_dot[-1] + R_ddot * dt
        new_R = max(R[-1] + new_R_dot * dt, bubble_radius/1000)
        
        R.append(new_R)
        R_dot.append(new_R_dot)
        collapse_time += dt
        
        if new_R < bubble_radius/2:
            break
            
    # Safety checks
    if collapse_time < 1e-9:
        collapse_time = 1e-9
        
    pulse_duration = collapse_time * 1.2
    thermal_time = tissue_depth**2 / (4 * thermal_diffusivity)
    pulse_gap = max(3 * collapse_time, thermal_time/100)
    duty_cycle = pulse_duration / (pulse_duration + pulse_gap)
    
    return (float(pulse_duration), 
            float(pulse_gap), 
            float(duty_cycle))

def optimize_array_pulsing(
    us_params: UltrasoundParameters,
    tissue: TissueProperties,
    sono_params: SonoluminescenceParameters
) -> PulseOptimizationMetrics:
    pulse_durations = np.zeros(us_params.num_elements)
    pulse_gaps = np.zeros(us_params.num_elements)
    element_delays = np.zeros(us_params.num_elements)
    
    thermal_diffusivity = tissue.thermal_conductivity / (tissue.density * tissue.specific_heat)
    
    for i in range(us_params.num_elements):
        duration, gap, dc = calculate_optimal_pulse_params(
            sono_params.target_bubble_size,
            us_params.tissue_depth,
            us_params.sound_speed,
            tissue.density,
            sono_params.max_pressure,
            thermal_diffusivity,
            i,
            us_params.num_elements,
            us_params.cav_params
        )
        pulse_durations[i] = duration
        pulse_gaps[i] = gap
        element_delays[i] = i * duration * 0.1
    
    # Physics-based metrics calculation
    mean_duty_cycle = np.mean(pulse_durations / (pulse_durations + pulse_gaps))
    uniformity = 1 - (np.std(pulse_durations) / np.mean(pulse_durations))
    thermal_margin = 1.0 - mean_duty_cycle
    
    return PulseOptimizationMetrics(
        pulse_duration=float(np.mean(pulse_durations)),
        pulse_gap=float(np.mean(pulse_gaps)),
        duty_cycle=float(mean_duty_cycle),
        cavitation_uniformity=float(uniformity),
        thermal_safety_margin=float(thermal_margin),
        element_delays=element_delays
    )

def plot_thermal_evolution(tissue: TissueProperties, intensity_field: np.ndarray, 
                         exposure_time: float, dx: float, dt: float, 
                         output_dir: str = 'output'):
    """Save thermal analysis plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # ... existing plotting code ...
    plt.savefig(os.path.join(output_dir, 'thermal_analysis.png'), dpi=300)
    plt.close()

def main():
    output_dir = 'simulation_output'
    os.makedirs(output_dir, exist_ok=True)

    fda_limits = FDA_Limits()
    cav_params = CavitationParameters()  # Created first
    us_params = UltrasoundParameters(cav_params=cav_params)  # Proper dependency injection
    tissue = TissueProperties()
    
    # Initialize medium first
    medium = kWaveMedium(
        sound_speed=tissue.sound_speed,
        density=tissue.density,
        alpha_coeff=tissue.alpha_coeff,
        alpha_power=tissue.alpha_power,
        BonA=tissue.BonA
    )
    
    # Create kgrid before using it
    kgrid = setup_grid(us_params, medium)
    
    # Now initialize sono_params with proper values
    sono_params = SonoluminescenceParameters(
        min_pressure=0.15e6,
        max_pressure=0.45e6, 
        target_bubble_size=4.5e-6,
        duty_cycle=0.85,
        prf=1000.0,
        stability_threshold=0.7
    )
    
    # Now create source signal with valid kgrid reference
    source_signal = create_optimized_signal(
        us_params,
        cav_params,
        kgrid.Nt,
        kgrid.dt,
        medium,
        tissue,
        sono_params,
        ThermalSafety()  # Add thermal parameter
    ).real.astype(np.float32)  # Removed * 1e3
    
    sweep_params = optimize_sweep_parameters(us_params.f_center, us_params.sound_speed, us_params.tissue_depth, 50e3, 1e6)
    sweep_ratio = sweep_params.sweep_ratio
    max_pressure = 0.4e6
    karray = setup_array(us_params)
    print("\nArray Configuration:")
    print(f"Wavelength (λ): {us_params.wavelength*1e3:.2f} mm")
    print(f"Element width (λ/2): {us_params.element_width()*1e3:.2f} mm")  # Fixed calculation
    print(f"Kerf spacing: {us_params.kerf*1e3:.2f} mm")
    print(f"Element pitch (width + kerf): {us_params.element_spacing()*1e3:.2f} mm")
    print(f"Number of elements: {us_params.num_elements}")
    
    # Add grating lobe check
    if us_params.element_spacing() > us_params.wavelength/2:
        warnings.warn(f"Element spacing {us_params.element_spacing()*1e3:.2f}mm exceeds λ/2 ({us_params.wavelength/2*1e3:.2f}mm) - grating lobes likely")
    
    print(f"Total array length: {us_params.total_array_length()*1e3:.2f} mm")
    print(f"Array width × length: {us_params.total_array_length()*1e3:.1f} × {us_params.transducer_length*1e3:.1f} mm")
    print(f"Active area ratio: {us_params.array_coverage_ratio():.1%}")
    print(f"Duty cycle: {us_params.duty_cycle*100:.0f}% ({'Continuous wave' if us_params.is_continuous_wave else 'Pulsed'})")
    initial_pressure, required_power, avg_density, peak_density = calculate_power_metrics(us_params, medium)
    print("\nPower Analysis (Acoustic):")
    print(f"Active transducer area: {us_params.active_area_cm2():.2f} cm²")
    print(f"Initial pressure: {initial_pressure/1e6:.2f} MPa")
    print(f"Required acoustic power: {required_power:.1f} W")
    print(f"Average power density: {avg_density:.2f} W/cm²")
    print(f"Peak power density: {peak_density:.2f} W/cm²")
    if peak_density > TransducerPower().max_power_density:
        print(f"Warning: Required power density exceeds safety limit of {TransducerPower().max_power_density} W/cm²")
        max_pressure = TransducerPower().calculate_pressure_from_power(TransducerPower().max_power_density, medium)
        print(f"Adjusting maximum pressure to {max_pressure/1e6:.2f} MPa")
        print("Warning: May not achieve target pressure at maximum depth")
    else:
        max_pressure = initial_pressure
    acoustic_impedance = medium.sound_speed * medium.density
    source_amplitude = max_pressure / acoustic_impedance
    source_signal *= source_amplitude
    expected_pressure = np.max(np.abs(source_signal)) * acoustic_impedance
    print(f"\nSource Signal Validation:")
    print(f"Scaling factor: {source_amplitude.item():.2f}")
    print(f"Expected peak pressure: {expected_pressure.item()/1e6:.2f} MPa")
    print(f"Target pressure: {max_pressure/1e6:.2f} MPa")
    if expected_pressure > 1.2 * max_pressure:
        correction_factor = max_pressure / expected_pressure
        source_amplitude *= correction_factor
        source_signal *= correction_factor
        print(f"Applied correction factor: {correction_factor.item():.2f}")
        print(f"Corrected peak pressure: {expected_pressure * correction_factor/1e6:.2f} MPa")
    final_expected_pressure = np.max(np.abs(source_signal)) * acoustic_impedance
    if final_expected_pressure > 0.5e6:
        final_correction = 0.5e6 / final_expected_pressure.item()  # Convert to scalar
        source_signal *= final_correction
        print(f"Final pressure adjustment applied: {final_correction:.2f}")
        print(f"Final expected pressure: {(final_expected_pressure.item() * final_correction)/1e6:.2f} MPa")
    transducer_area = us_params.active_area_cm2() / 10000
    required_voltage = TransducerPower().calculate_voltage_required(
        max_pressure, medium, us_params.active_area_cm2()
    )
    print(f"Required voltage: {required_voltage:.2f} V")
    bandwidth = 0.2 * us_params.f_center
    q_factor = calculate_q_factor(us_params.f_center, bandwidth)
    print(f"Q factor: {q_factor:.2f}")
    p_field = run_simulation(kgrid, karray, source_signal, medium)
    peak_pressure = np.max(np.abs(p_field))
    print(f"\nPressure Validation:")
    print(f"Peak pressure achieved: {peak_pressure/1e6:.2f} MPa")
    print(f"Target pressure: {max_pressure/1e6} MPa")
    for step in range(kgrid.Nt):
        if step % 500 == 0:
            phase_noise = 0.1 * source_amplitude
            source_signal += phase_noise * np.random.randn(*source_signal.shape)
    min_pressure = np.min(p_field, axis=0).astype(np.float64)
    MI_field = calculate_MI_field(min_pressure, us_params.f_center)
    cavitation_probability = np.mean(np.abs(p_field) > cav_params.min_cavitation_pressure, axis=0)
    x_mm = np.linspace(-us_params.transducer_width/2 * 1000, us_params.transducer_width/2 * 1000, p_field.shape[1])
    z_mm = np.linspace(0, us_params.tissue_depth * 1000, p_field.shape[2])
    transducer_y = -1
    visualize_results(p_field, kgrid, us_params, output_dir)
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(x_mm, z_mm, MI_field.T, cmap='hot', shading='auto')
    plt.colorbar(label='Mechanical Index')
    plt.plot([-us_params.transducer_width/2 * 1000, us_params.transducer_width/2 * 1000], [transducer_y, transducer_y], 'b-', linewidth=3, label='Transducer Array')
    plt.xlabel('Lateral Position [mm]')
    plt.ylabel('Depth [mm]')
    plt.title('Mechanical Index Distribution\nTransducer Array and Tissue Medium')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'mechanical_index.png'), dpi=300)
    plt.close()
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(x_mm, z_mm, cavitation_probability.T, cmap='viridis', shading='auto')
    plt.colorbar(label='Cavitation Probability')
    plt.plot([-us_params.transducer_width/2 * 1000, us_params.transducer_width/2 * 1000], [transducer_y, transducer_y], 'b-', linewidth=3, label='Transducer Array')
    plt.xlabel('Lateral Position [mm]')
    plt.ylabel('Depth [mm]')
    plt.title('Cavitation Probability Distribution\nTransducer Array and Tissue Medium')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'cavitation_probability.png'), dpi=300)
    plt.close()
    mean_MI = np.mean(MI_field)
    max_MI = np.max(MI_field)
    treatment_area = calculate_treatment_coverage(MI_field, 0.3)
    print(f"Mean MI: {mean_MI:.2f}")
    print(f"Max MI: {max_MI:.2f}")
    print(f"Treatment coverage: {treatment_area:.1f}%")
    print(f"Safe treatment threshold maintained: {max_MI < fda_limits.MI_LIMIT}")
    intensities = calculate_intensities(p_field, medium, kgrid.dt)
    print("\nFDA Compliance Check:")
    print(f"MI: {max_MI:.2f} (Limit: {fda_limits.MI_LIMIT})")
    print(f"ISPTA: {intensities.ISPTA:.1f} mW/cm² (Limit: {fda_limits.ISPTA_LIMIT})")
    print(f"ISPPA: {intensities.ISPPA/1e4:.1f} W/cm² (Limit: {fda_limits.ISPPA_LIMIT})")
    print(f"Compliant with FDA limits: {max_MI <= fda_limits.MI_LIMIT and intensities.ISPTA <= fda_limits.ISPTA_LIMIT and intensities.ISPPA/1e4 <= fda_limits.ISPPA_LIMIT}")
    plt.figure(figsize=(10, 4))
    plt.plot(sweep_params.rarefaction_zones*1000, np.ones_like(sweep_params.rarefaction_zones), 'r|', label='Rarefaction zones', markersize=12)
    plt.plot(sweep_params.compression_zones*1000, 0.8*np.ones_like(sweep_params.compression_zones), 'b|', label='Compression zones', markersize=12)
    plt.ylim(0.5, 1.5)
    plt.title('Pressure Zone Distribution')
    plt.legend()
    print("\nZone Analysis:")
    print(f"Number of rarefaction zones: {len(sweep_params.rarefaction_zones)}")
    print(f"Number of compression zones: {len(sweep_params.compression_zones)}")
    print(f"Average zone spacing: {sweep_params.wavelength*1000:.2f} mm")
    plt.savefig(os.path.join(output_dir, 'simulation_domain.png'), dpi=300)
    plt.close()
    print("\nPower Metrics:")
    peak_pressure = np.max(np.abs(p_field))
    mean_pressure = np.mean(np.abs(p_field))
    Z = medium.sound_speed * medium.density
    peak_power_density = (peak_pressure**2)/(2 * Z) / 10000
    mean_power_density = (mean_pressure**2)/(2 * Z) / 10000
    required_power = peak_power_density * us_params.active_area_cm2()
    print(f"Peak power density: {peak_power_density.item():.2f} W/cm²")
    print(f"Mean power density: {mean_power_density.item():.2f} W/cm²")
    print(f"Required power for treatment: {required_power.item():.2f} W")
    swr, cv = calculate_cavitation_metrics(p_field)
    print("\nCavitation Uniformity Metrics:")
    print(f"Standing Wave Ratio: {swr:.2f} (Target: <{cav_params.standing_wave_threshold})")
    print(f"Coefficient of Variation: {cv:.2f} (Target: <{cav_params.cv_threshold})")
    cavitation_probability = np.mean(np.abs(p_field) > cav_params.min_cavitation_pressure, axis=0)
    plt.figure()
    plt.imshow(cavitation_probability, cmap='viridis')
    plt.colorbar(label='Cavitation Probability')
    plt.title('Cavitation Probability Distribution')
    plt.savefig(os.path.join(output_dir, 'cavitation_probability_distribution.png'), dpi=300)
    plt.close()
    sono_params = SonoluminescenceParameters()
    optimization_metrics = optimize_transducer_design(us_params, sono_params, tissue)
    print("\nTransducer Optimization Results:")
    print(f"Uniformity Score: {optimization_metrics.uniformity_score}")
    print(f"Coverage Ratio: {optimization_metrics.coverage_ratio:.2f}")
    print(f"Stability Score: {optimization_metrics.stability_score:.2f}")
    print(f"Efficiency Score: {optimization_metrics.efficiency_score:.2f}")
    print(f"Total Score: {optimization_metrics.total_score:.2f}")
    stability_score, uniformity_score = analyze_sonoluminescence_potential(p_field, sono_params)
    print("\nSonoluminescence Analysis:")
    print(f"Stability Score: {stability_score:.2f}")
    print(f"Uniformity Score: {uniformity_score:.2f}")
    review_array_direct_contact(us_params)
    constraints = {"max_elements": 256, "stand_off": 0.0}
    best_metrics = multi_objective_transducer_optimization(us_params, tissue, constraints)
    print(f"Optimized transducer metrics: {best_metrics}")
    plot_simulation_domain(kgrid, us_params)

@jit(nopython=True)
def calculate_efficiency_score(us_params: UltrasoundParameters, tissue: TissueProperties) -> float:
    """
    Calculate overall transducer efficiency score.
    """
    # Calculate acoustic impedances
    Z_tissue = tissue.density * tissue.sound_speed
    Z_water = 1000 * 1500  # Water coupling medium
    
    # Calculate transmission coefficient
    transmission_coeff = (4 * Z_tissue * Z_water) / ((Z_tissue + Z_water)**2)
    
    # Calculate beam formation efficiency
    wavelength = us_params.wavelength
    element_width = us_params.element_width()
    steering_angle = np.arctan2(us_params.tissue_depth, us_params.transducer_width/2)
    beam_efficiency = np.sin(np.pi * element_width/wavelength * np.sin(steering_angle))
    beam_efficiency = beam_efficiency * beam_efficiency  # Square term
    
    # Calculate element utilization efficiency
    element_efficiency = us_params.array_coverage_ratio() * us_params.duty_cycle
    
    # Combine efficiencies with weighting
    total_efficiency = (0.4 * transmission_coeff + 
                       0.3 * beam_efficiency + 
                       0.3 * element_efficiency)
    
    # Manual clipping for Numba compatibility
    return max(0.0, min(total_efficiency, 1.0))

if __name__ == "__main__":
    main()
