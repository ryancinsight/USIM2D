# Technical Notes & Learnings

- **Beamforming Utilities:**
  - The `element_directivity()` function implements a piston model using `scipy.special.j1` while also offering a fallback to a normalized sinc function.
  - The use of `np.where` guarantees numerical stability when the argument of the Bessel function is near zero.
  
- **Attenuation Calculations:**
  - Tissue attenuation is computed based on the given attenuation coefficient, frequency (in MHz), and path length (converted from meters to centimeters).
  
- **Broadcasting & Array Shapes:**
  - Reshaping arrays using `np.newaxis` (or `.reshape()`) is crucial for aligning element positions with spatial grids.
  - Maintaining consistent shapes in the simulation scripts ensures efficient vectorized computing.
  
- **Frequency Sweep Implementations:**
  - Both focused and synchronous frequency sweep scripts update wavelength, phase constant, and attenuation factors dynamically across the frequency range.
  
- **Code Style & Best Practices:**
  - Unused imports (e.g., `math` and unnecessary `j1` in some scripts) were noted and have been removed where appropriate.
  - The codebase now adheres to PEP8 guidelines, and inline documentation has been improved.
  
- **Dependencies:**
  - Dependencies are using current versions as verified against the latest releases (NumPy, SciPy, Matplotlib).
  
- **General Observation:**
  - The simulation scripts, while independent in their visualization style (focused vs. synchronous), share a common structure that reinforces DRY principles.
  - Future work may focus on further modularizing shared components and enhancing numerical robustness.

*These notes are consolidated from insights collected during the review of all simulations and periodic code updates.*
