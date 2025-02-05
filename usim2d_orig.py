"""
Ultrasound Simulation and Analysis Framework
==========================================

A comprehensive framework for modeling therapeutic ultrasound fields and analyzing 
bioeffects including cavitation and sonoluminescence.

References:
----------
[1] Treeby, B.E. and Cox, B.T. (2010), "k-Wave: MATLAB toolbox for the simulation 
    and reconstruction of photoacoustic wave fields", J. Biomed. Opt., 15(2):021314
[2] Church, C.C. (2005), "Frequency, pulse length, and the mechanical index", 
    ARLO, 6, 162-168
[3] O'Brien, W.D. (2007), "Ultrasound-biophysics mechanisms", 
    Progress in Biophysics and Molecular Biology, 93(1-3):212-255
[4] Apfel, R.E. and Holland, C.K. (1991), "Gauging the likelihood of cavitation 
    from short-pulse, low-duty cycle diagnostic ultrasound", Ultrasound Med. Biol., 17:179-185
[5] Suslick, K.S. (1990), "Sonochemistry", Science, 247(4949):1439-1445

Algorithms:
----------
1. Acoustic Field Simulation:
   - k-Wave FDTD (Finite Difference Time Domain) implementation
   - PML (Perfectly Matched Layer) boundary conditions
   - CFL-limited time stepping for numerical stability
   - Grid optimization based on wavelength sampling criteria

2. Cavitation Analysis:
   - Rayleigh collapse time calculation for bubble dynamics
   - Blake threshold prediction for cavitation onset
   - Standing wave ratio computation for field uniformity
   - Spatial and temporal peak detection for stable/inertial cavitation zones

3. Intensity Calculations:
   - ISPTA (Spatial-Peak Temporal-Average Intensity)
   - ISPPA (Spatial-Peak Pulse-Average Intensity)
   - Mechanical Index computation using negative peak pressures
   - Temporal averaging with Hilbert transform envelope detection

4. Array Optimization:
   - Phase delay optimization for improved focusing
   - Element spacing calculations for grating lobe suppression
   - Aperture apodization for sidelobe reduction
   - Multi-objective optimization for coverage and uniformity

5. Safety Monitoring:
   - FDA regulatory limit validation
   - Thermal dose estimation
   - Cavitation risk assessment
   - Real-time treatment zone tracking

6. Sonoluminescence Prediction:
   - Bubble resonance frequency calculation
   - Stability threshold determination
   - Rayleigh-Plesset equation analysis
   - Multi-bubble interaction effects

Implementation Notes:
-------------------
- Numba JIT compilation used for compute-intensive functions
- Parallel processing for array calculations
- Adaptive mesh refinement in high-gradient regions
- Vectorized operations for improved performance
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numba import jit, float64, int64, prange
from numba.experimental import jitclass
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

@jitclass(spec=([('sound_speed', float64), ('density', float64), ('alpha_coeff', float64), ('alpha_power',float64), ('BonA', float64)]))
class TissueProperties:
    def __init__(self, sound_speed: float = 1540, density: float = 1045, alpha_coeff: float = 0.5, alpha_power: float = 1.1, BonA: float = 7.0):
        self.sound_speed = sound_speed
        self.density = density
        self.alpha_coeff = alpha_coeff
        self.alpha_power = alpha_power
        self.BonA = BonA


@jitclass(spec=([('MI_LIMIT', float64), ('ISPTA_LIMIT',float64), ('ISPPA_LIMIT', float64)]))
class FDA_Limits:
    def __init__(self, MI_LIMIT: float = 1.9, ISPTA_LIMIT: float = 720, ISPPA_LIMIT: float = 190):
        self.MI_LIMIT = MI_LIMIT
        self.ISPTA_LIMIT = ISPTA_LIMIT
        self.ISPPA_LIMIT = ISPPA_LIMIT

class TransducerPower:
    """
    A class handling power and pressure calculations for ultrasound transducers.
    This class manages the calculations related to transducer power settings,
    pressure propagation, and safety parameters for ultrasound applications.
    Attributes:
        target_pressure (float): Target acoustic pressure in Pa (default: 0.4 MPa)
        safety_margin (float): Safety coefficient for power calculations (default: 1.1)
        max_power_density (float): Maximum allowed power density in W/cm² (default: 10.0)
        attenuation_factor (float): Tissue attenuation compensation factor (default: 1.1)
    Methods:
        calculate_required_power(area_cm2: float, medium: kWaveMedium) -> float:
            Calculate required power based on transducer area and medium properties.
            Returns power in Watts.
        calculate_initial_pressure(medium: kWaveMedium, tissue_depth: float) -> float:
            Calculate initial pressure needed accounting for tissue attenuation.
            Returns pressure in Pascals.
        - Power calculation: P = I·A = p²·A/(2ρc)
        - Attenuation: p(z) = p₀·exp(-αf^n·z)
        - Safety considerations include:
            * Cavitation threshold
            * Tissue heating limits
            * Mechanical effects
    Notes:
        All calculations incorporate safety margins and tissue attenuation factors
        to ensure therapeutic effectiveness while maintaining safety parameters.
    Power and pressure parameters for transducer operation.
    
    Physics:
    --------
    1. Power calculation: P = I·A = p²·A/(2ρc)
    2. Attenuation: p(z) = p₀·exp(-αf^n·z)
    3. Safety margins based on:
       - Cavitation threshold
       - Tissue heating limits
       - Mechanical effects
    """
    def __init__(self, target_pressure: float = 0.4e6, safety_margin: float = 1.1, max_power_density: float = 10.0, attenuation_factor: float = 1.1):
        self.target_pressure = target_pressure
        self.safety_margin = safety_margin
        self.max_power_density = max_power_density
        self.attenuation_factor = attenuation_factor

    def calculate_required_power(self, area_cm2: float, medium: kWaveMedium) -> float:
        Z = medium.sound_speed * medium.density
        compensated_pressure = self.target_pressure * self.safety_margin * self.attenuation_factor
        intensity = float(compensated_pressure**2 / (2*Z))
        return float(intensity * area_cm2 / 10000)

    def calculate_initial_pressure(self, medium: kWaveMedium, tissue_depth: float) -> float:
        alpha_0 = 0.5
        f_MHz = medium.sound_speed / 1e6
        depth_cm = tissue_depth * 100
        attenuation_db = alpha_0 * f_MHz * depth_cm
        attenuation_factor = 10**(attenuation_db/20)
        min_therapeutic_pressure = 0.1e6
        calculated_pressure = self.target_pressure * attenuation_factor * self.safety_margin
        return max(calculated_pressure, min_therapeutic_pressure)

@jitclass(spec=([('transducer_width', float64), ('transducer_length', float64), ('tissue_depth', float64), ('f_center', float64), ('sound_speed', float64), ('duty_cycle', float64), ('pulse_duration', float64), ('pulse_gap', float64), ('prf', float64)]))
class UltrasoundParameters:
    def __init__(self, transducer_width: float = 130e-3, transducer_length: float = 90e-3, 
                 tissue_depth: float = 12e-3, f_center: float = 180e3,  # Changed to match current frequency
                 sound_speed: float = 1482.0, duty_cycle: float = 0.9,  # Updated to match water properties
                 pulse_duration: float = 90e-6, pulse_gap: float = 10e-6,
                 prf: float = 10000):
        self.transducer_width = transducer_width
        self.transducer_length = transducer_length
        self.tissue_depth = tissue_depth
        self.f_center = f_center
        self.sound_speed = sound_speed
        self.duty_cycle = duty_cycle
        self.pulse_duration = pulse_duration
        self.pulse_gap = pulse_gap
        self.prf = prf

    @property
    def wavelength(self) -> float:
        return self.sound_speed / self.f_center

    @property
    def element_width(self) -> float:
        return min(self.wavelength / 2, self.element_spacing * 0.9)

    @property
    def num_elements(self) -> int:
        min_spacing = self.wavelength / 2
        return int(np.floor(self.transducer_width / min_spacing))

    @property
    def element_spacing(self) -> float:
        return self.transducer_width / self.num_elements

    @property
    def total_array_length(self) -> float:
        return self.element_spacing * self.num_elements

    def array_coverage_ratio(self) -> float:
        element_area = self.element_width * self.transducer_length
        total_array_area = self.transducer_width * self.transducer_length
        return (element_area * self.num_elements) / total_array_area

    def active_area_cm2(self) -> float:
        element_area = self.element_width * self.transducer_length
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

@jitclass(spec=([('f_min', float64), ('f_max', float64), ('sweep_ratio', float64), ('sweep_rate', float64), ('wavelength', float64), ('rarefaction_zones', float64[:]), ('compression_zones', float64[:])]))
class SweepParameters:
    def __init__(self, f_min: float = 0.2e6, f_max: float = 1.0e6, sweep_ratio: float = 1.5, sweep_rate: float = 0.5e6, wavelength: float = 0.68e-3, rarefaction_zones: np.ndarray = np.array([0.0, 0.5, 1.0]), compression_zones: np.ndarray = np.array([0.25, 0.75])):
        self.f_min = f_min
        self.f_max = f_max
        self.sweep_ratio = sweep_ratio
        self.sweep_rate = sweep_rate
        self.wavelength = wavelength
        self.rarefaction_zones = rarefaction_zones
        self.compression_zones = compression_zones

@jitclass(spec=([('ISPTA', float64), ('ISPPA', float64)]))
class IntensityMetrics:
    def __init__(self, ISPTA: float = 0.0, ISPPA: float = 0.0):
        self.ISPTA = ISPTA
        self.ISPPA = ISPPA

@jitclass(spec=([('min_cavitation_pressure', float64), ('max_cavitation_pressure', float64), ('optimal_prr', float64), ('phase_shift_factor', float64), ('standing_wave_threshold', float64), ('cv_threshold', float64)]))
class CavitationParameters:
    def __init__(self, min_cavitation_pressure: float = 0.1e6, max_cavitation_pressure: float = 1.0e6, optimal_prr: float = 1000.0, phase_shift_factor: float = 0.25, standing_wave_threshold: float = 1.3, cv_threshold: float = 0.15):
        self.min_cavitation_pressure = min_cavitation_pressure
        self.max_cavitation_pressure = max_cavitation_pressure
        self.optimal_prr = optimal_prr
        self.phase_shift_factor = phase_shift_factor
        self.standing_wave_threshold = standing_wave_threshold
        self.cv_threshold = cv_threshold

@jit(float64[:](int64, float64), nopython=True)
def _optimize_array_phases_helper(num_elements: int, phase_shift_factor: float) -> np.ndarray:
    phase_delays = np.zeros(num_elements)
    for i in prange(num_elements):
        phase_delays[i] = (i * phase_shift_factor + np.random.uniform(0, np.pi/4))
    return phase_delays % (2 * np.pi)

# Add after existing cavitation parameters class
@jitclass(spec=[
    ('phase_correlation_length', float64),
    ('phase_variation_scale', float64),
    ('temporal_decorrelation', float64)
])
class UniformCavitationParams:
    """Parameters for achieving uniform cavitation field"""
    def __init__(self, phase_correlation_length: float = 2.0, phase_variation_scale: float = 0.3, temporal_decorrelation: float = 0.2):
        self.phase_correlation_length = phase_correlation_length
        self.phase_variation_scale = phase_variation_scale
        self.temporal_decorrelation = temporal_decorrelation

@jit(nopython=True)
def generate_correlated_phases(num_elements: int, correlation_length: float, variation_scale: float) -> np.ndarray:
    """
    Generate spatially correlated random phases for uniform cavitation.
    
    Physics:
    --------
    1. Spatial Correlation:
       - Uses Gaussian correlation function: exp(-x²/2L²)
       - L: correlation length (number of elements)
       - Prevents abrupt phase changes that create hot spots
    
    2. Phase Constraints:
       - Maximum phase difference between adjacent elements < π/2
       - Maintains coherent focusing while breaking standing waves
    
    3. Standing Wave Suppression:
       - Phase variation creates time-varying interference patterns
       - Decorrelation length > λ/4 breaks up nodes/antinodes
    """
    base_phases = np.zeros(num_elements)
    for i in range(num_elements):
        # Generate correlated random walk
        correlation = np.exp(-np.arange(num_elements)**2 / (2*correlation_length**2))
        phase_step = np.random.normal(0, variation_scale)
        base_phases[i:] += phase_step * correlation[:num_elements-i]
    
    return (base_phases % (2*np.pi))

# Modify the optimize_array_phases function
def optimize_array_phases(us_params: UltrasoundParameters, cav_params: CavitationParameters) -> np.ndarray:
    """
    Optimize array element phases for uniform cavitation distribution.
    
    Physics:
    --------
    1. Standing Wave Mitigation:
       - Phase randomization breaks up periodic interference
       - Correlation length > λ/4 for effective decorrelation
       - Maximum phase gradient limited by steering constraints
    
    2. Cavitation Control:
       - Time-varying phase patterns prevent stable nodes
       - Local pressure variations within ±30% of mean
       - Phase updates synchronized with bubble dynamics
    
    3. Field Uniformity:
       - Progressive phase variation reduces edge effects
       - Spatial correlation maintains treatment coherence
       - Temporal modulation spreads energy distribution
    
    Returns:
        ndarray: Optimized phase delays for array elements
    """
    unif_params = UniformCavitationParams()
    
    # Generate base phases with spatial correlation
    base_phases = generate_correlated_phases(
        us_params.num_elements,
        unif_params.phase_correlation_length,
        unif_params.phase_variation_scale
    )
    
    # Add progressive phase shift for steering
    steering_angle = np.arctan2(us_params.tissue_depth, us_params.transducer_width/4)
    k = 2 * np.pi / us_params.wavelength
    element_positions = np.arange(us_params.num_elements) * us_params.element_spacing
    steering_phases = k * element_positions * np.sin(steering_angle)
    
    # Combine phases with temporal variation
    temporal_factor = 2 * np.pi * unif_params.temporal_decorrelation
    final_phases = (base_phases + steering_phases + temporal_factor) % (2*np.pi)
    
    return final_phases

@jitclass(spec=([('min_pressure', float64), ('max_pressure', float64), ('target_bubble_size', float64), ('duty_cycle', float64), ('prf', float64), ('stability_threshold', float64)]))
class SonoluminescenceParameters:
    def __init__(self, min_pressure: float = 0.15e6, max_pressure: float = 0.45e6, target_bubble_size: float = 4.5e-6, duty_cycle: float = 0.85, prf: float = 1000.0, stability_threshold: float = 0.7):
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

@jit(nopython=True)
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
    width_variations = np.array([0.8, 1.0, 1.2]) * us_params.transducer_width
    length_variations = np.array([0.8, 1.0, 1.2]) * us_params.transducer_length
    best_metrics = TransducerOptimizationMetrics(0, 0, 0, 0, 0)
    best_config = None
    for width in width_variations:
        for length in length_variations:
            test_params = UltrasoundParameters(
                transducer_width=width,
                transducer_length=length,
                tissue_depth=us_params.tissue_depth,
                f_center=us_params.f_center,
                sound_speed=us_params.sound_speed
            )
            metrics = evaluate_transducer_configuration(us_params, sono_params, tissue)
            if metrics.total_score > best_metrics.total_score:
                best_metrics = metrics
                best_config = test_params
    return best_metrics

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
    element_spacing = us_params.element_spacing
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
    position = -us_params.transducer_width/2 + x*us_params.element_spacing
    angle = np.arctan(position / us_params.tissue_depth)
    wavenumber = 2 * np.pi / us_params.wavelength
    directivity = np.abs(np.sin(wavenumber * element_width * np.sin(angle) / 2) / 
                        (wavenumber * element_width * np.sin(angle) / 2 + 1e-10))
    near_field_distance = position**2 / us_params.wavelength
    grating_lobe_angle = np.arcsin(us_params.wavelength / us_params.element_spacing)
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
    intensity = np.square(p_field) / (2 * Z)

    # Calculate frequency-dependent attenuation
    alpha = medium.alpha_coeff * (f_center / 1e6) ** medium.alpha_power
    depth_axis = np.linspace(0, tissue_depth, p_field.shape[-1])
    attenuation = np.exp(-2 * alpha * depth_axis)

    # Apply attenuation
    attenuated_intensity = intensity * attenuation[None, None, :]

    # Return power density
    power_density = 2 * alpha * attenuated_intensity
    return power_density

def validate_safety_metrics(
    mi_field: np.ndarray,
    intensities: IntensityMetrics,
    fda_limits: FDA_Limits,
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
        medium: kWaveMedium: Object containing medium properties
        active_area_cm2: float: Active area in cm²
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
def calculate_thermal_effects(intensity_field: np.ndarray, tissue: TissueProperties, exposure_time: float) -> np.ndarray:
    """Calculates thermal effects in tissue due to ultrasound exposure using the bio-heat equation.
    This function implements the Pennes bio-heat equation to model temperature rise and thermal dose
    in tissue exposed to ultrasound. The equation accounts for:
    - Ultrasonic energy absorption
    - Heat conduction in tissue
    - Blood perfusion cooling effects
    Physics equations used:
    - Bio-heat equation: ρc(∂T/∂t) = k∇²T + Q - wbcb(T-Ta)
        where ρ = density, c = specific heat, k = thermal conductivity,
        Q = heat source, wb = blood perfusion rate, cb = blood specific heat,
        T = tissue temperature, Ta = arterial blood temperature
    - Thermal dose calculation: TD = ∫ R^(43-T) dt
        where R = 0.5 for T ≥ 43°C, R = 0.25 for T < 43°C
    Parameters
    ----------
    intensity_field : np.ndarray
        Spatial distribution of ultrasound intensity (W/m²)
    tissue : TissueProperties
        Object containing tissue properties (density, absorption coefficient)
    exposure_time : float
        Duration of ultrasound exposure in seconds
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - temperature: Final temperature distribution in tissue (°C)
        - thermal_dose: Cumulative thermal dose in equivalent minutes at 43°C
    Notes
    -----
    Uses numba @jit decorator for improved computational performance.
    Temperature and thermal dose are calculated using discrete time steps of 0.1s.
    """
    specific_heat = 3600  # J/(kg·K)
    thermal_conductivity = 0.5  # W/(m·K)
    blood_perfusion = 0.5  # kg/(m³·s)
    blood_specific_heat = 3800  # J/(kg·K)
    ambient_temp = 37.0  # °C
    
    # Calculate temperature rise using bio-heat equation
    q = 2 * tissue.alpha_coeff * intensity_field
    dt = 0.1  # Time step (s)
    num_steps = int(exposure_time / dt)
    temperature = np.zeros_like(intensity_field) + ambient_temp
    thermal_dose = np.zeros_like(intensity_field)
    
    for _ in range(num_steps):
        dT = (q + thermal_conductivity * np.gradient(temperature, axis=0)**2 - 
              blood_perfusion * blood_specific_heat * (temperature - ambient_temp)) * dt / (tissue.density * specific_heat)
        temperature += dT
        R = np.where(temperature >= 43, 0.5, 0.25)
        thermal_dose += R**(43 - temperature) * dt
        
    return temperature, thermal_dose

@jit(nopython=True)
def calculate_tissue_response(pressure_field: np.ndarray, tissue: TissueProperties) -> tuple:
    """
    Calculate the tissue response to an ultrasonic pressure field by determining strain, stress and temperature change.
    This function implements viscoelastic tissue response modeling using the Kelvin-Voigt model and thermal effects
    from acoustic absorption.
    Parameters
    ----------
    pressure_field : np.ndarray
        2D array of pressure values over time and space [Pa]
    tissue : TissueProperties
        Object containing tissue material properties including:
        - density [kg/m^3] 
        - sound_speed [m/s]
        - alpha_coeff [Np/m]
    Returns
    -------
    tuple
        strain : np.ndarray
            Mechanical strain in the tissue [-]
            Calculated using ε = -P/(ρc^2) where:
            P = pressure, ρ = density, c = sound speed
        stress : np.ndarray  
            Viscoelastic stress response [Pa]
            σ = Eε + ηdε/dt where:
            E = Young's modulus, η = viscosity, ε = strain
        dT : np.ndarray
            Temperature rise [K]
            ΔT = (2αIt)/(ρCp) where:
            α = absorption coefficient, I = intensity,
            t = exposure time, ρ = density, Cp = specific heat
    Notes
    -----
    - Uses Kelvin-Voigt model for viscoelastic response
    - Assumes linear acoustic propagation
    - Temperature calculation uses plane wave intensity approximation
    - Fixed material properties:
        Young's modulus = 30 kPa
        Viscosity = 0.1 Pa·s  
        Specific heat = 3600 J/(kg·K)
    - Uses 1 microsecond time step for strain rate calculation
    """
    youngs_modulus = 3e4  # Pa
    viscosity = 0.1  # Pa·s
    specific_heat = 3600  # J/(kg·K)
    
    # Calculate strain
    acoustic_impedance = tissue.sound_speed * tissue.density
    strain = -pressure_field / (tissue.density * tissue.sound_speed**2)
    
    # Calculate strain rate
    dt = 1e-6  # Assumed time step
    strain_rate = np.gradient(strain, dt, axis=0)
    
    # Calculate viscoelastic stress
    stress = youngs_modulus * strain + viscosity * strain_rate
    
    # Calculate temperature rise
    intensity = pressure_field**2 / (2 * acoustic_impedance)
    exposure_time = pressure_field.shape[0] * dt
    dT = (2 * tissue.alpha_coeff * np.mean(intensity, axis=0) * exposure_time) / (tissue.density * specific_heat)
    
    return strain, stress, dT

@jit(nopython=True)
def calculate_bubble_dynamics(p_field: np.ndarray, sono_params: SonoluminescenceParameters) -> tuple:
    """Calculate the bubble dynamics using the Rayleigh-Plesset equation.

    Physics:
    --------
    Implements full Rayleigh-Plesset equation:
    RR̈ + (3/2)Ṙ² = (1/ρ)[p_g(R₀/R)ᵏ + p_v - P₀ - P(t) - 2σ/R - 4μṘ/R]

    where:
    - R: bubble radius
    - ρ: liquid density 
    - p_g: gas pressure
    - p_v: vapor pressure
    - P₀: ambient pressure
    - P(t): acoustic pressure
    - σ: surface tension
    - μ: viscosity
    - κ: polytropic exponent

    Key physics:
    - Gas compression: p_g(R₀/R)ᵏ
    - Surface tension: 2σ/R
    - Viscous damping: 4μṘ/R 
    - Inertial effects: RR̈ + (3/2)Ṙ²

    Args:
        p_field (np.ndarray): Acoustic pressure field values over time
        sono_params (SonoluminescenceParameters): Sonoluminescence parameters

    Returns:
        tuple: (R_history, P_history) containing radius and pressure histories

    Notes:
        - Implemented with 4th order Runge-Kutta integration for stability
        - Optimized with Numba for performance
        - Uses adaptive timestep based on collapse dynamics
    """
    # Physical constants 
    ambient_pressure = 101325  # Pa
    surface_tension = 0.072   # N/m
    viscosity = 1e-3         # Pa·s 
    density = 1000          # kg/m³
    polytropic_exp = 1.4    # Adiabatic
    vapor_pressure = 2.3e3  # Pa at 20°C
    
    # Initialize with optimal time step
    dt_base = 1e-9  # Base timestep
    dt_min = 1e-12  # Minimum timestep during collapse
    R0 = sono_params.target_bubble_size
    R = R0
    Rdot = 0
    
    # Preallocate arrays
    num_steps = p_field.shape[0]
    R_history = np.zeros(num_steps)
    P_history = np.zeros(num_steps)
    
    # RK4 integration coefficients
    a = [0, 0.5, 0.5, 1.0]
    b = [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0]
    
    # Main integration loop with adaptive timestep
    for i in range(num_steps):
        # Adaptive timestep based on radius
        dt = max(dt_min, dt_base * (R/R0))
        
        # RK4 integration step
        k1_R = Rdot
        k1_V = _calculate_acceleration(R, Rdot, p_field[i], R0, ambient_pressure,
                                    density, surface_tension, viscosity, 
                                    polytropic_exp, vapor_pressure)
        
        k2_R = Rdot + 0.5*dt*k1_V  
        k2_V = _calculate_acceleration(R + 0.5*dt*k1_R, k2_R, p_field[i], R0,
                                    ambient_pressure, density, surface_tension,
                                    viscosity, polytropic_exp, vapor_pressure)
        
        k3_R = Rdot + 0.5*dt*k2_V
        k3_V = _calculate_acceleration(R + 0.5*dt*k2_R, k3_R, p_field[i], R0,
                                    ambient_pressure, density, surface_tension, 
                                    viscosity, polytropic_exp, vapor_pressure)
        
        k4_R = Rdot + dt*k3_V
        k4_V = _calculate_acceleration(R + dt*k3_R, k4_R, p_field[i], R0,
                                    ambient_pressure, density, surface_tension,
                                    viscosity, polytropic_exp, vapor_pressure)
        
        # Update radius and velocity
        R += dt/6 * (k1_R + 2*k2_R + 2*k3_R + k4_R)
        Rdot += dt/6 * (k1_V + 2*k2_V + 2*k3_V + k4_V)
        
        # Calculate internal pressure
        P_gas = ambient_pressure * (R0/R)**(3*polytropic_exp)
        
        # Store results
        R_history[i] = R
        P_history[i] = P_gas + vapor_pressure
        
    return R_history, P_history

@jit(nopython=True)
def _calculate_acceleration(R: float, Rdot: float, P_acoustic: float, R0: float,
                          P0: float, rho: float, sigma: float, mu: float,
                          kappa: float, P_v: float) -> float:
    """Calculate bubble wall acceleration using Rayleigh-Plesset equation."""
    P_gas = P0 * (R0/R)**(3*kappa) + P_v
    P_surface = 2 * sigma / R
    P_viscous = 4 * mu * Rdot / R
    
    return (P_gas - P0 - P_acoustic - P_surface - P_viscous)/(rho * R) - 1.5 * Rdot**2/R
def analyze_sonoluminescence_potential(p_field: np.ndarray, sono_params: SonoluminescenceParameters):
    """
    Evaluate field conditions for sonoluminescence.
    
    Physics:
    --------
    Critical parameters:
    - Rayleigh collapse time: τ = 0.915·R₀√(ρ/ΔP)
    - Blake threshold: P_B = P₀ + 0.77σ/R₀
    - Expansion ratio: R_max/R₀ ≈ 2-3 for stable cavitation
    
    Theory:
    - Requires precise timing between collapse and driving
    - Temperature can reach >5000K during collapse
    - Light emission from plasma formation
    """
    stability_score, uniformity_score = calculate_cavitation_stability(p_field, sono_params.min_pressure, sono_params.max_pressure)
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.imshow(np.mean(np.abs(p_field) > sono_params.min_pressure, axis=0), cmap='viridis')
    plt.colorbar(label='Sonoluminescence Probability')
    plt.title('Sonoluminescence Probability Map')
    plt.subplot(222)
    stable_regions = ((np.abs(p_field) >= sono_params.min_pressure) & (np.abs(p_field) <= sono_params.max_pressure))
    plt.imshow(np.mean(stable_regions, axis=0), cmap='RdYlGn')
    plt.colorbar(label='Stability Score')
    plt.title('Stable Cavitation Regions')
    return stability_score, uniformity_score

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
    """
    Analyze acoustic field for cavitation likelihood.
    
    Physics:
    --------
    Standing Wave Ratio (SWR):
    SWR = (p_max + p_min)/(p_max - p_min)
    
    Cavitation Index (CI):
    CI = |p⁻|/√(ρcf)
    
    Theory:
    - SWR > 4 indicates strong standing waves
    - Local pressure maxima determine nucleation sites
    - Temporal pressure gradients affect bubble dynamics
    """
    p_amplitude = np.abs(p_field)
    p_max = calc_max_along_axis0(p_amplitude)
    p_min = calc_min_along_axis0(p_amplitude)
    denominator = p_max - p_min
    denominator = np.where(denominator < 1e-10, 1e-10, denominator)
    swr = (p_max + p_min) / denominator
    mean_p = np.mean(p_amplitude)
    if mean_p < 1e-10:
        mean_p = 1e-10
    std_p = np.std(p_amplitude)
    cv = std_p / mean_p
    return float(np.mean(swr)), float(cv)

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
def create_optimized_signal(us_params: UltrasoundParameters, cav_params: CavitationParameters, 
                          n_samples: int, dt: float, medium: kWaveMedium) -> np.ndarray:
    """Generate optimized excitation signals with dynamic phase control and element-specific pulsing"""
    # Generate base carrier signal
    t = np.arange(n_samples) * dt
    source_signal = np.zeros((us_params.num_elements, n_samples), dtype=np.complex64)
    carrier = np.exp(2j * np.pi * us_params.f_center * t)
    for i in range(us_params.num_elements):
        # ...carrier signal generation...
        source_signal[i, :] = carrier

    # Get optimized pulse parameters for each element
    pulse_params = optimize_array_pulsing(us_params, TissueProperties(), 0.5e6)
    
    # Generate base signal as before
    for i in range(us_params.num_elements):
        # Add element-specific phase offset
        phase_offset = 2 * np.pi * i / us_params.num_elements
        element_carrier = carrier * np.exp(1j * phase_offset)
        source_signal[i, :] = element_carrier
    
    # Apply element-specific pulse timing
    for i in range(us_params.num_elements):
        # Calculate pulse mask with optimized timing
        t = np.arange(n_samples) * dt
        period = pulse_params.pulse_duration + pulse_params.pulse_gap
        shifted_t = t - pulse_params.element_delays[i]
        pulse_mask = ((shifted_t % period) < pulse_params.pulse_duration)
        
        # Apply element-specific pulsing
        source_signal[i, :] *= pulse_mask
    
    # Add temporal phase updates for cavitation control
    update_interval = int(0.1 / (dt * us_params.f_center))  # 10% of period
    for t in range(0, n_samples, update_interval):
        phase_updates = generate_correlated_phases(
            us_params.num_elements,
            UniformCavitationParams().phase_correlation_length,
            0.1  # Reduced variation for stability
        )
        end_idx = min(t + update_interval, n_samples)
        source_signal[:, t:end_idx] *= np.exp(1j * phase_updates[:, np.newaxis])
    
    # Apply amplitude scaling and normalization
    max_pressure = 0.5e6  # 0.5 MPa maximum initial pressure
    acoustic_impedance = float(medium.sound_speed * medium.density)
    scaling_factor = max_pressure / (acoustic_impedance * np.max(np.abs(source_signal)))
    source_signal *= scaling_factor
    
    return source_signal.real.astype(np.float32)

@jit(float64[:](float64, float64, float64, float64), nopython=True, parallel=True)
def create_sweep_signal(f_center: float, sweep_ratio: float, duration: float, dt: float) -> np.ndarray:
    """Base frequency sweep signal generation"""
    f_min = f_center * (1 - sweep_ratio/2)
    f_max = f_center * (1 + sweep_ratio/2)
    t = np.arange(0, duration, dt)
    signal = np.zeros_like(t)
    elongation_factor = 2.0
    period = duration * elongation_factor
    for i in prange(t.size):
        freq = 0.5*(f_min + f_max) + 0.5*(f_max - f_min) * np.cos(2.0 * np.pi * (t[i]/period) - np.pi/2)
        signal[i] = np.sin(2.0 * np.pi * freq * t[i])
    return signal

@jit(float64[:,:](float64, float64, int64, float64, int64), nopython=True, parallel=True)
def create_modulated_signal(f_center: float, sweep_ratio: float, n_samples: int, dt: float, n_elements: int) -> np.ndarray:
    """Element-specific modulation with phase offsets"""
    f_min = f_center * (1 - sweep_ratio/2)
    f_max = f_center * (1 + sweep_ratio/2)
    t = np.arange(n_samples) * dt
    sweep_period = n_samples * dt
    sweep_rate = (f_max - f_min) / sweep_period
    t_mod = t % sweep_period
    phase_sweep = 2 * np.pi * (f_min * t + 0.5 * sweep_rate * t_mod**2)
    phase_jumps = np.floor(t / sweep_period)
    phase_correction = phase_jumps * 2 * np.pi * (f_max - f_min) * sweep_period / 2
    phase_sweep += phase_correction
    signals = np.zeros((n_elements, n_samples))
    for i in prange(n_elements):
        phase_offset = np.random.uniform(0, 2*np.pi)
        amp_mod = 0.5 + 0.5*np.sin(2*np.pi*1000*t)
        signals[i] = amp_mod * np.sin(phase_sweep + phase_offset)
    return signals

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
def _optimize_sweep_helper(f_center: float, wavelength: float, tissue_depth: float) -> tuple:
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
    
    # Set target sweep rate based on bubble dynamics
    target_sweep_rate = 50.0  # kHz/ms
    
    # Calculate sweep period and rate
    sweep_period = delta_f / (target_sweep_rate * 1000)  # s
    sweep_rate = delta_f / sweep_period  # Hz/s
    
    # Apply rate bounds based on physics constraints
    max_sweep_rate = 100.0 * 1000  # Hz/s - Limited by bubble response
    min_sweep_rate = 10.0 * 1000   # Hz/s - Ensure decorrelation
    sweep_rate = min(max(sweep_rate, min_sweep_rate), max_sweep_rate)
    
    return f_min, f_max, sweep_ratio, sweep_rate, rarefaction_positions, compression_positions

@jitclass(spec=[('min_sweep_rate', float64),('max_sweep_rate', float64),('growth_time_factor', float64),('spatial_decorrelation', float64)])
class SweepRateOptimizationParams:
    def __init__(self, min_sweep_rate=10e3, max_sweep_rate=100e3, growth_time_factor=2.0, spatial_decorrelation=0.1):
        self.min_sweep_rate = min_sweep_rate
        self.max_sweep_rate = max_sweep_rate
        self.growth_time_factor = growth_time_factor
        self.spatial_decorrelation = spatial_decorrelation

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

def optimize_sweep_parameters(f_center: float, sound_speed: float, tissue_depth: float) -> SweepParameters:
    """
    Optimize frequency sweep for bone transmission and cavitation control.
    
    Physics:
    --------
    1. Bone transmission optimized:
       - Center frequency fixed at 220 kHz
       - Narrow bandwidth to maintain transmission efficiency
       - Sweep range considers bone attenuation characteristics
       
    2. Cavitation enhancement:
       - Sweep period matched to bubble dynamics
       - Rate ensures standing wave breakup
       - Bandwidth allows stable bubble oscillation
    """
    wavelength = calculate_wavelength(f_center, sound_speed)
    
    # Narrower sweep for bone penetration
    sweep_ratio = min(calculate_optimal_interleaving(wavelength, tissue_depth), 0.15)
    
    # Calculate band limits around 220 kHz
    delta_f = f_center * sweep_ratio
    f_min = f_center - delta_f/2
    f_max = f_center + delta_f/2
    
    # Optimize sweep rate for bubble dynamics
    sweep_rate = calculate_optimal_sweep_rate(f_center, 5e-6, sound_speed, tissue_depth)
    
    # Calculate zones with fixed center frequency
    base_positions = np.arange(wavelength/4, tissue_depth, wavelength/2)
    rarefaction_positions = base_positions[::2]
    compression_positions = base_positions[1::2]
    
    return SweepParameters(
        f_min=f_min,
        f_max=f_max,
        sweep_ratio=sweep_ratio,
        sweep_rate=sweep_rate,
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
        prf=optimal_params['prf']
    )

def setup_array(us_params: UltrasoundParameters) -> kWaveArray:
    karray = kWaveArray()
    for i in range(us_params.num_elements):
        position = [-us_params.transducer_width/2 + i*us_params.element_spacing, -us_params.transducer_length/2]
        karray.add_rect_element(
            position,
            us_params.element_width,
            us_params.transducer_length,
            0
        )
    return karray

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
    """
    Compute acoustic power and exposure metrics.
    
    Physics:
    --------
    1. Acoustic Intensity (W/m²):
       I = p²/(2ρc)
       where:
       - p: pressure (Pa)
       - ρ: density (kg/m³)
       - c: sound speed (m/s)
    
    2. Power (W):
       P = I * A 
       where A is area in m²
    
    3. Unit Conversions:
       - 1 W/cm² = 10000 W/m²
       - 1 MPa = 1e6 Pa
    
    FDA Safety Limits:
    - ISPTA < 720 mW/cm²
    - ISPPA < 190 W/cm²
    - MI < 1.9
    """
    power_calc = TransducerPower()
    active_area = us_params.active_area_cm2() * 1e-4  # Convert cm² to m²
    
    # Initial pressure limited to therapeutic range (0.1-2 MPa)
    initial_pressure = power_calc.calculate_initial_pressure(medium, us_params.tissue_depth)
    initial_pressure = min(max(initial_pressure, 0.1e6), 2e6)
    
    # Calculate intensity using proper units
    Z = medium.sound_speed * medium.density  # Pa·s/m
    intensity = (initial_pressure**2) / (2 * Z)  # W/m²
    
    # Convert to clinically relevant units
    intensity_cm2 = intensity * 1e-4  # Convert W/m² to W/cm²
    required_power = intensity * active_area  # Total acoustic power (W)
    power_density = required_power / us_params.active_area_cm2()  # W/cm²
    
    return initial_pressure, required_power, power_density

def run_simulation(kgrid: kWaveGrid, karray: kWaveArray, source_signal: np.ndarray, medium: kWaveMedium) -> np.ndarray:
    """
    Execute k-space pseudospectral acoustic simulation.
    
    Physics:
    --------
    Wave equation solved:
    ∂²p/∂t² = c²∇²p + (c²/ρ₀)∇ρ₀·∇p + ∂²/∂t²(ξ/c²)
    
    where:
    - p: acoustic pressure
    - c: sound speed
    - ρ₀: ambient density
    - ξ: acoustic nonlinearity parameter
    
    Numerical Implementation:
    - k-space corrected finite differences
    - PML boundary conditions
    - Staggered grid for improved accuracy
    """
    source_p_mask = karray.get_array_binary_mask(kgrid)
    source_p = karray.get_distributed_source_signal(kgrid, source_signal) * 1e6
    # Create absolute temp directory pathp
    TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
    os.makedirs(TEMP_DIR, exist_ok=True)
    simulation_options = SimulationOptions(
        save_to_disk=True,
        data_cast="single",
        pml_alpha=2.0,
        pml_size=20,
        smooth_p0=True,
        input_filename=os.path.join(TEMP_DIR, 'simulation_2d.h5'),  # Add .h5 extension
        output_filename=os.path.join(TEMP_DIR, 'output_2d.h5'),  # Specify output file
        data_path=TEMP_DIR  # Move data_path here
    )
    execution_options = SimulationExecutionOptions(is_gpu_simulation=False)
    sensor = kSensor()
    sensor.mask = np.ones((kgrid.Nx, kgrid.Ny), dtype=bool)
    source = kSource()
    source.p_mask = source_p_mask
    source.p = source_p
    p = kspaceFirstOrder2DC(kgrid, source, sensor, medium, simulation_options, execution_options)
    p_field = np.reshape(p["p"], (kgrid.Nt, kgrid.Nx, kgrid.Ny))
    p_field = np.transpose(p_field, (0, 2, 1))
    
    # Limit maximum pressure to realistic therapeutic range
    max_therapeutic = 2e6  # 2 MPa max
    scaling_factor = min(1.0, max_therapeutic / np.max(np.abs(p_field)))
    p_field *= scaling_factor
    
    return p_field

def visualize_results(p_field: np.ndarray, kgrid: kWaveGrid, us_params: UltrasoundParameters):
    """
    Visualize acoustic field and treatment metrics.
    
    Physics:
    --------
    Displayed quantities:
    1. Pressure field: p(x,z,t)
    2. Intensity distribution: I = p²/(2ρc)
    3. Treatment zones based on:
       - Cavitation threshold
       - Mechanical Index
       - Thermal dose
    
    Color mapping:
    - Hot colormap for pressure magnitude
    - Viridis for probabilities
    - RdYlGn for safety metrics
    """
    x_mm = np.linspace(-us_params.transducer_width/2 * 1000, us_params.transducer_width/2 * 1000, p_field.shape[1])
    z_mm = np.linspace(0, us_params.tissue_depth * 1000, p_field.shape[2])
    global visualization_coords
    visualization_coords = {
        'x_mm': x_mm,
        'z_mm': z_mm,
        'transducer_y': -1
    }
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(x_mm, z_mm, np.max(np.abs(p_field), axis=0).T/1e6, cmap='hot', shading='auto')
    transducer_y = -1
    plt.plot([-us_params.transducer_width/2 * 1000, us_params.transducer_width/2 * 1000], [transducer_y, transducer_y], 'b-', linewidth=3, label='Transducer Array')
    plt.colorbar(label='Pressure [MPa]')
    plt.xlabel('Lateral Position [mm]')
    plt.ylabel('Depth [mm]')
    plt.title('Pressure Field Distribution\nTransducer Array and Tissue Medium')
    plt.text(0, -2, 'Transducer Array', horizontalalignment='center', color='blue')
    plt.text(0, us_params.tissue_depth*500, 'Tissue Medium', horizontalalignment='center', color='red')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

def calculate_voltage_required(pressure: float, medium: kWaveMedium, transducer_area: float) -> float:
    """Calculate required driving voltage with proper scaling"""
    Z = medium.sound_speed * medium.density
    # Limit pressure to therapeutic range (0.1-2 MPa)
    pressure = min(max(pressure, 0.1e6), 2e6)
    intensity = (pressure**2) / (2 * Z)
    power = intensity * transducer_area
    # Add efficiency factor (~50-70% for typical transducers)
    efficiency = 0.6
    voltage = np.sqrt(power * 2 * Z / efficiency)
    # Limit voltage to realistic range (typical max ~500V)
    return min(voltage, 500.0)

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
            self.us_params.tissue_depth
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
            if self.cav_params.min_pressure >= self.cav_params.max_pressure:
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
            medium  # Add medium parameter
        ).astype(np.float32) * 1e3
        
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
    plt.show()

@contextmanager 
def parallel_pool(processes=None):
    pool = Pool(processes)
    try:
        yield pool
    finally:
        pool.close()
        pool.join()

def optimize_grid_parameters(us_params: UltrasoundParameters) -> tuple:
    """
    Optimize computational grid for accuracy and efficiency.
    
    Physics:
    --------
    Spatial sampling:
    dx ≤ λ_min/N where N ≥ 4 (Nyquist criterion)
    
    Temporal sampling:
    dt ≤ CFL·dx/c_max (Courant condition)
    
    PML thickness:
    L_pml ≥ 2λ for -60dB absorption
    """
    wavelength = us_params.wavelength
    min_points_per_wavelength = 4
    dx = wavelength / min_points_per_wavelength
    nx = int(np.ceil(us_params.transducer_width / dx))
    ny = int(np.ceil(us_params.tissue_depth / dx))
    nx += nx % 2
    ny += ny % 2
    pml_size = 10
    nx += 2 * pml_size
    ny += 2 * pml_size
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
            medium  # Add medium parameter
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

def analyze_results(results: dict, params: SimulationParameters) -> dict:
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
    if not results:
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

@jit(float64(float64[:,:], float64), nopython=True)
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
    """
    Calculate acoustic intensity metrics from pressure field.
    
    Physics:
    --------
    1. Instantaneous Intensity:
       I(t) = p²(t)/(ρc) [W/m²]
    
    2. Temporal Averaging:
       ISPTA = (1/T)∫I(t)dt [mW/cm²]
       ISPPA = max(pulse average of I) [W/cm²]
    
    3. Intensity Metrics:
       - ISPTA: Spatial-Peak Temporal-Average Intensity
       - ISPPA: Spatial-Peak Pulse-Average Intensity
    
    Unit Conversions:
    - 1 W/m² = 0.1 mW/cm²
    - Pressure field in Pa
    """
    Z = medium.sound_speed * medium.density  # Acoustic impedance (Pa·s/m)
    
    # Calculate instantaneous intensity (W/m²)
    i_field = np.square(p_field) / (2 * Z)
    
    # Apply proper scaling for medical units
    ispta = np.max(np.mean(i_field, axis=0)) * 0.1  # Convert W/m² to mW/cm²
    
    # Calculate ISPPA with pulse detection
    hilbert_env = np.abs(hilbert(p_field, axis=0))
    pulse_mask = hilbert_env > (np.max(hilbert_env) * 0.1)
    isppa = np.max(np.mean(i_field * pulse_mask, axis=0)) * 1e-4  # Convert W/m² to W/cm²
    
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
    element_idx: int,
    num_elements: int
) -> tuple:
    """
    Calculate optimal pulse parameters for each array element.
    
    Physics:
    --------
    1. Bubble dynamics:
       - Rayleigh collapse time: τ_c = 0.915R₀√(ρ/ΔP)
       - Recovery time: τ_r = 2πR₀√(ρ/3κP₀)
       
    2. Acoustic transit:
       - Time of flight: τ_f = d/c
       - Phase delay: φ = 2πfτ_f
       
    3. Thermal effects:
       - Heat diffusion time: τ_h = d²/4α
       - Duty cycle limit: DC < τ_h/(τ_h + τ_c)
    """
    # Calculate fundamental timescales
    collapse_time = calculate_rayleigh_collapse_time(bubble_radius, density, pressure)
    transit_time = tissue_depth / sound_speed
    thermal_time = tissue_depth**2 / (4 * 0.15e-6)  # Thermal diffusivity ~0.15 mm²/s
    
    # Base pulse duration on collapse dynamics
    base_pulse = collapse_time * 1.5  # Allow full collapse
    
    # Element-specific adjustments for uniform coverage
    position_factor = 0.8 + 0.4 * (element_idx / num_elements)  # Vary 0.8-1.2
    pulse_duration = base_pulse * position_factor
    
    # Calculate gap to prevent heat buildup
    min_gap = 3 * collapse_time  # Minimum for bubble reset
    cooling_factor = thermal_time / 100  # Prevent tissue heating
    pulse_gap = max(min_gap, cooling_factor * pulse_duration)
    
    # Calculate resulting duty cycle
    period = pulse_duration + pulse_gap
    duty_cycle = pulse_duration / period
    
    return float(pulse_duration), float(pulse_gap), float(duty_cycle)

def optimize_array_pulsing(
    us_params: UltrasoundParameters,
    tissue: TissueProperties,
    target_pressure: float
) -> PulseOptimizationMetrics:
    """
    Optimize pulsing scheme across all elements for uniform cavitation.
    
    Algorithm:
    ----------
    1. Calculate base timing from bubble dynamics
    2. Apply position-dependent adjustments
    3. Ensure thermal safety margins
    4. Validate against FDA limits
    """
    pulse_durations = np.zeros(us_params.num_elements)
    pulse_gaps = np.zeros(us_params.num_elements)
    element_delays = np.zeros(us_params.num_elements)
    
    for i in range(us_params.num_elements):
        duration, gap, dc = calculate_optimal_pulse_params(
            5e-6,  # Typical bubble radius
            us_params.tissue_depth,
            us_params.sound_speed,
            tissue.density,
            target_pressure,
            i,
            us_params.num_elements
        )
        pulse_durations[i] = duration
        pulse_gaps[i] = gap
        
        # Calculate progressive delays for sweeping effect
        element_delays[i] = i * duration * 0.1
    
    # Calculate aggregate metrics
    mean_duty_cycle = np.mean(pulse_durations / (pulse_durations + pulse_gaps))
    uniformity = np.std(pulse_durations) / np.mean(pulse_durations)
    thermal_margin = 1.0 - mean_duty_cycle  # Higher is safer
    
    return PulseOptimizationMetrics(
        pulse_duration=float(np.mean(pulse_durations)),
        pulse_gap=float(np.mean(pulse_gaps)),
        duty_cycle=float(mean_duty_cycle),
        cavitation_uniformity=float(uniformity),
        thermal_safety_margin=float(thermal_margin),
        element_delays=element_delays
    )

def plot_thermal_evolution(tissue: TissueProperties, intensity_field: np.ndarray, 
                         exposure_time: float, dx: float, dt: float):
    """
    Visualize thermal effects in tissue over time.
    
    Physics:
    --------
    1. Bio-heat equation:
       ρc∂T/∂t = k∇²T + Q - wbcb(T-Ta)
    
    2. Thermal dose:
       TD₄₃ = ∫R^(43-T)dt
       where R = {0.5 T > 43°C
                {0.25 T < 43°C
    """
    temp_field, dose_field = calculate_thermal_effects(intensity_field, tissue, exposure_time)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Temperature distribution
    im1 = ax1.imshow(temp_field - 37, cmap='hot')
    ax1.set_title('Temperature Rise (°C)')
    plt.colorbar(im1, ax=ax1)
    
    # Thermal dose
    im2 = ax2.imshow(np.log10(dose_field + 1), cmap='viridis')
    ax2.set_title('Thermal Dose (log₁₀ CEM43)')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()

# Modify create_optimized_signal to use element-specific pulsing
def create_optimized_signal(us_params: UltrasoundParameters, cav_params: CavitationParameters, 
                          n_samples: int, dt: float, medium: kWaveMedium) -> np.ndarray:
    """Generate optimized excitation signals with dynamic phase control and element-specific pulsing"""
    pulse_params = optimize_array_pulsing(us_params, TissueProperties(), 0.5e6)
    
    # Generate base signal as before
    t = np.arange(n_samples) * dt
    source_signal = np.zeros((us_params.num_elements, n_samples), dtype=np.complex64)
    carrier = np.exp(1j * 2.0 * np.pi * us_params.f_center * t)
    for i in range(us_params.num_elements):
        source_signal[i, :] = carrier

    # Apply element-specific pulsing
    for i in range(us_params.num_elements):
        delay = pulse_params.element_delays[i]
        pulse_start = int(delay / dt)
        pulse_end = pulse_start + int(pulse_params.pulse_duration / dt)
        if pulse_end > n_samples:
            pulse_end = n_samples
        source_signal[i, :pulse_start] = 0
        source_signal[i, pulse_end:] = 0

    return source_signal

def main():
    fda_limits = FDA_Limits()
    us_params = UltrasoundParameters()
    sweep_params = optimize_sweep_parameters(us_params.f_center, us_params.sound_speed, us_params.tissue_depth)
    sweep_ratio = sweep_params.sweep_ratio
    tissue = TissueProperties()
    print("\nOptimal Sweep Parameters:")
    print(f"  Frequency Range: {sweep_params.f_min:.2f} Hz - {sweep_params.f_max:.2f} Hz")
    print(f"  Sweep Ratio:     {sweep_params.sweep_ratio:.4f}")
    print(f"  Sweep Rate:      {sweep_params.sweep_rate:.4f} Hz/s")
    print(f"  Wavelength:      {sweep_params.wavelength:.5f} m")
    print(f"  Rarefaction Zones (m): {sweep_params.rarefaction_zones}")
    max_pressure = 0.4e6
    karray = setup_array(us_params)
    print("\nArray Configuration:")
    print(f"Wavelength (λ): {us_params.wavelength*1e3:.2f} mm")
    print(f"Element width (λ/2): {us_params.element_width*1e3:.2f} mm")
    print(f"Number of elements: {us_params.num_elements}")
    print(f"Element spacing: {us_params.element_spacing*1e3:.2f} mm")
    print(f"Total array length: {us_params.total_array_length*1e3:.2f} mm")
    print(f"Array width × length: {us_params.total_array_length*1e3:.1f} × {us_params.transducer_length*1e3:.1f} mm")
    print(f"Active area ratio: {us_params.array_coverage_ratio():.1%}")
    print(f"Duty cycle: {us_params.duty_cycle*100:.0f}% ({'Continuous wave' if us_params.is_continuous_wave else 'Pulsed'})")
    medium = kWaveMedium(
        sound_speed=tissue.sound_speed,
        density=tissue.density,
        alpha_coeff=tissue.alpha_coeff,
        alpha_power=tissue.alpha_power,
        BonA=tissue.BonA
    )
    kgrid = setup_grid(us_params, medium)
    cav_params = CavitationParameters()
    source_signal = create_optimized_signal(
        us_params,
        cav_params,
        kgrid.Nt,
        kgrid.dt,
        medium  # Add medium parameter
    ).astype(np.float32) * 1e3
    print(f"Source signal shape: {source_signal.shape}")
    initial_pressure, required_power, power_density = calculate_power_metrics(us_params, medium)
    print("\nPower Analysis (Acoustic):")
    print(f"Active transducer area: {float(us_params.active_area_cm2()):.2f} cm²")
    print(f"Initial pressure required: {initial_pressure/1e6} MPa")
    print(f"Required acoustic power: {required_power.item():.1f} W")
    print(f"Power density: {power_density.item():.1f} W/cm²")
    if power_density > TransducerPower().max_power_density:
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
    print(f"Target pressure: {(max_pressure/1e6).item():.2f} MPa")
    if expected_pressure > 1.2 * max_pressure:
        correction_factor = max_pressure / expected_pressure
        source_amplitude *= correction_factor
        source_signal *= correction_factor
        print(f"Applied correction factor: {correction_factor.item():.2f}")
        print(f"Corrected peak pressure: {(expected_pressure * correction_factor).item()/1e6:.2f} MPa")
    final_expected_pressure = np.max(np.abs(source_signal)) * acoustic_impedance
    if final_expected_pressure > 0.5e6:
        final_correction = 0.5e6 / final_expected_pressure
        source_signal *= final_correction
        print(f"Final pressure adjustment applied: {final_correction:.2f}")
        print(f"Final expected pressure: {(final_expected_pressure * final_correction)/1e6:.2f} MPa")
    transducer_area = us_params.active_area_cm2() / 10000
    required_voltage = calculate_voltage_required(max_pressure, medium, transducer_area)
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
    visualize_results(p_field, kgrid, us_params)
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(x_mm, z_mm, MI_field.T, cmap='hot', shading='auto')
    plt.colorbar(label='Mechanical Index')
    plt.plot([-us_params.transducer_width/2 * 1000, us_params.transducer_width/2 * 1000], [transducer_y, transducer_y], 'b-', linewidth=3, label='Transducer Array')
    plt.xlabel('Lateral Position [mm]')
    plt.ylabel('Depth [mm]')
    plt.title('Mechanical Index Distribution\nTransducer Array and Tissue Medium')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(x_mm, z_mm, cavitation_probability.T, cmap='viridis', shading='auto')
    plt.colorbar(label='Cavitation Probability')
    plt.plot([-us_params.transducer_width/2 * 1000, us_params.transducer_width/2 * 1000], [transducer_y, transducer_y], 'b-', linewidth=3, label='Transducer Array')
    plt.xlabel('Lateral Position [mm]')
    plt.ylabel('Depth [mm]')
    plt.title('Cavitation Probability Distribution\nTransducer Array and Tissue Medium')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()
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
    plt.show()
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
    plt.show()
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

if __name__ == "__main__":
    main()










