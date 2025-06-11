#!/usr/bin/env python3
"""
Universal Squeezing Parameter Optimization Framework
==================================================

This module implements comprehensive numerical validation of optimal squeezing
parameters across broader electric field regimes, systematically characterizing
negative-energy enhancements for the complete parameter space.

Objectives:
1. Validate universal squeezing parameter r_opt across E-field regimes 10^12 - 10^18 V/m
2. Characterize negative-energy enhancement scaling laws
3. Quantify field-dependent squeezing thresholds and saturation limits
4. Map optimal parameter combinations for maximum efficiency
5. Establish universal scaling relationships for practical implementation

Mathematical Framework:
- Squeezing Enhancement: ξ(r,E) = cosh(2r) × f_field(E/E_crit)
- Negative Energy: ρ_neg = -ρ_vacuum × ξ(r,E) × g_spatial(x,y,z)
- Universal Scaling: r_opt = α × log(E/E_crit) + β × (E/E_crit)^γ
- Efficiency: η = (Energy_created / Energy_input) × ξ(r_opt,E)
"""

import numpy as np
import scipy.optimize as optimize
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Try to import JAX with fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap, pmap
    JAX_AVAILABLE = True
    jax.config.update("jax_enable_x64", True)
    print("JAX acceleration enabled for squeezing optimization")
except ImportError:
    JAX_AVAILABLE = False
    jnp = np
    def jit(func):
        return func
    def grad(func):
        # Simple numerical gradient fallback
        def grad_func(x, *args, **kwargs):
            h = 1e-8
            return (func(x + h, *args, **kwargs) - func(x - h, *args, **kwargs)) / (2 * h)
        return grad_func
    def vmap(func):
        return np.vectorize(func)
    def pmap(func):
        return func
    print("Using NumPy fallback for squeezing optimization")

@dataclass
class SqueezingSpecs:
    """Specifications for squeezing parameter optimization"""
    # Field regime specifications
    E_min: float = 1e12        # Minimum field strength (V/m)
    E_max: float = 1e18        # Maximum field strength (V/m) 
    E_crit: float = 1.32e18    # Critical Schwinger field (V/m)
    n_field_points: int = 1000 # Number of field points to sample
    
    # Squeezing parameter range
    r_min: float = 0.0         # Minimum squeezing parameter
    r_max: float = 5.0         # Maximum squeezing parameter (r=5 → 74 dB)
    n_squeeze_points: int = 500 # Number of squeezing points
    
    # Spatial resolution
    spatial_resolution: int = 128 # Grid points per dimension
    spatial_extent: float = 1e-6   # Physical size (1 μm)
    
    # Optimization parameters
    optimization_tolerance: float = 1e-12
    max_iterations: int = 10000
    convergence_threshold: float = 1e-15

@dataclass
class SqueezingResults:
    """Results from squeezing parameter optimization"""
    optimal_squeezing: np.ndarray = field(default_factory=lambda: np.array([]))
    enhancement_factors: np.ndarray = field(default_factory=lambda: np.array([]))
    negative_energy_density: np.ndarray = field(default_factory=lambda: np.array([]))
    efficiency_map: np.ndarray = field(default_factory=lambda: np.array([]))
    universal_scaling_coeffs: Dict[str, float] = field(default_factory=dict)
    field_regimes: np.ndarray = field(default_factory=lambda: np.array([]))
    computational_performance: Dict[str, float] = field(default_factory=dict)

class UniversalSqueezingOptimizer:
    """Advanced squeezing parameter optimization with universal scaling laws"""
    
    def __init__(self, specs: SqueezingSpecs = None):
        self.specs = specs or SqueezingSpecs()
        
        # Physical constants
        self.c = 2.998e8           # Speed of light (m/s)
        self.hbar = 1.055e-34      # Reduced Planck constant (J⋅s)
        self.epsilon_0 = 8.854e-12 # Vacuum permittivity (F/m)
        self.alpha = 1/137.036     # Fine structure constant
        self.e = 1.602e-19         # Elementary charge (C)
        self.m_e = 9.109e-31       # Electron mass (kg)
        
        # Derived quantities
        self.l_planck = 1.616e-35  # Planck length (m)
        self.rho_planck = self.c**5 / (self.hbar * 6.674e-11**2)  # Planck energy density
        self.vacuum_energy_density = self.hbar * self.c / self.l_planck**4
        
        # Initialize computational arrays
        self.initialize_computational_grids()
        
        # JAX-compiled functions for performance
        self.compile_jax_functions()
        
    def initialize_computational_grids(self):
        """Initialize computational grids for parameter space exploration"""
        print("🔧 Initializing Universal Squeezing Parameter Grids")
        print(f"   Field Range: {self.specs.E_min/1e12:.1f} - {self.specs.E_max/1e15:.0f} × 10^15 V/m")
        print(f"   Squeezing Range: {self.specs.r_min:.1f} - {self.specs.r_max:.1f}")
        print(f"   Grid Resolution: {self.specs.n_field_points} × {self.specs.n_squeeze_points}")
        
        # Logarithmic field sampling for broad dynamic range
        self.E_field_array = np.logspace(
            np.log10(self.specs.E_min), 
            np.log10(self.specs.E_max),
            self.specs.n_field_points
        )
        
        # Linear squeezing parameter sampling
        self.r_squeeze_array = np.linspace(
            self.specs.r_min,
            self.specs.r_max, 
            self.specs.n_squeeze_points
        )
        
        # Create 2D parameter meshes
        self.E_mesh, self.r_mesh = np.meshgrid(self.E_field_array, self.r_squeeze_array)
        
        # Spatial grid for 3D field calculations
        x = np.linspace(0, self.specs.spatial_extent, self.specs.spatial_resolution)
        y = np.linspace(0, self.specs.spatial_extent, self.specs.spatial_resolution)  
        z = np.linspace(0, self.specs.spatial_extent, self.specs.spatial_resolution)
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')
        
        print(f"✅ Computational grids initialized")
        print(f"   Total parameter combinations: {self.specs.n_field_points * self.specs.n_squeeze_points:,}")
        print(f"   Spatial grid points: {self.specs.spatial_resolution**3:,}")
        
    def compile_jax_functions(self):
        """Compile JAX functions for high-performance computation"""
        print("🔧 Compiling JAX functions for optimal performance...")
        
        @jit
        def squeezing_enhancement_jax(r: jnp.ndarray, E: jnp.ndarray) -> jnp.ndarray:
            """JAX-compiled squeezing enhancement calculation"""
            # Basic hyperbolic enhancement
            basic_enhancement = jnp.cosh(2 * r)
            
            # Field-dependent correction factor
            E_ratio = E / self.specs.E_crit
            field_correction = 1 + 0.5 * jnp.tanh(E_ratio - 0.1)
            
            # Nonlinear saturation at high squeezing
            saturation_factor = 1 / (1 + 0.1 * r**2)
            
            return basic_enhancement * field_correction * saturation_factor
        
        @jit  
        def negative_energy_density_jax(r: jnp.ndarray, E: jnp.ndarray, 
                                       x: jnp.ndarray, y: jnp.ndarray, 
                                       z: jnp.ndarray) -> jnp.ndarray:
            """JAX-compiled negative energy density calculation"""
            # Squeezing enhancement
            xi = squeezing_enhancement_jax(r, E)
            
            # Spatial localization factor
            r_spatial = jnp.sqrt(x**2 + y**2 + z**2)
            spatial_factor = jnp.exp(-r_spatial**2 / (100e-9)**2)  # 100 nm localization
            
            # Base vacuum energy density (negative)
            rho_base = -self.vacuum_energy_density
            
            return rho_base * xi * spatial_factor
        
        @jit
        def efficiency_calculation_jax(r: jnp.ndarray, E: jnp.ndarray) -> jnp.ndarray:
            """JAX-compiled efficiency calculation"""
            # Enhancement factor
            xi = squeezing_enhancement_jax(r, E)
            
            # Input energy density
            input_energy = 0.5 * self.epsilon_0 * E**2
            
            # Created energy (via Schwinger effect with squeezing)
            # Γ ∝ exp(-π m²c³ / eEℏ) enhanced by squeezing
            gamma_schwinger = jnp.exp(-jnp.pi * self.m_e**2 * self.c**3 / 
                                     (self.e * E * self.hbar)) * xi
            
            # Energy per particle pair
            pair_energy = 2 * self.m_e * self.c**2
            
            # Creation rate density
            creation_rate = (self.e**2 * E**2) / (4 * jnp.pi**3 * self.hbar**2 * self.c) * gamma_schwinger
            
            # Output energy rate
            output_energy = creation_rate * pair_energy
            
            # Efficiency
            efficiency = output_energy / (input_energy * self.c + 1e-50)  # Avoid division by zero
            
            return efficiency
        
        @jit
        def universal_scaling_law_jax(E: jnp.ndarray, alpha: float, beta: float, 
                                     gamma: float) -> jnp.ndarray:
            """JAX-compiled universal scaling law"""
            E_ratio = E / self.specs.E_crit
            return alpha * jnp.log(E_ratio + 1e-10) + beta * E_ratio**gamma
        
        # Store compiled functions
        self.squeezing_enhancement_jax = squeezing_enhancement_jax
        self.negative_energy_density_jax = negative_energy_density_jax  
        self.efficiency_calculation_jax = efficiency_calculation_jax
        self.universal_scaling_law_jax = universal_scaling_law_jax
        
        # Vectorized versions for parameter sweeps
        self.squeezing_enhancement_vmap = vmap(squeezing_enhancement_jax, in_axes=(0, 0))
        self.efficiency_calculation_vmap = vmap(efficiency_calculation_jax, in_axes=(0, 0))
        
        print("✅ JAX functions compiled and optimized")
        
    def execute_comprehensive_parameter_sweep(self) -> SqueezingResults:
        """Execute comprehensive parameter space exploration"""
        print("\n🚀 Executing Comprehensive Squeezing Parameter Sweep")
        print("=" * 60)
        
        start_time = time.time()
        results = SqueezingResults()
        
        # Initialize result arrays
        enhancement_map = np.zeros((self.specs.n_squeeze_points, self.specs.n_field_points))
        efficiency_map = np.zeros((self.specs.n_squeeze_points, self.specs.n_field_points))
        optimal_squeezing = np.zeros(self.specs.n_field_points)
        negative_energy_map = np.zeros((self.specs.n_squeeze_points, self.specs.n_field_points))
        
        print(f"📊 Processing {self.specs.n_field_points:,} field values...")
        
        # Process in batches for memory efficiency
        batch_size = 100
        n_batches = (self.specs.n_field_points + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, self.specs.n_field_points)
            batch_fields = self.E_field_array[start_idx:end_idx]
            
            # Create batch meshes
            batch_E_mesh, batch_r_mesh = np.meshgrid(batch_fields, self.r_squeeze_array)
            
            # Calculate enhancement factors for batch
            batch_enhancement = self.squeezing_enhancement_jax(
                jnp.array(batch_r_mesh), jnp.array(batch_E_mesh)
            )
            enhancement_map[:, start_idx:end_idx] = np.array(batch_enhancement)
            
            # Calculate efficiencies for batch
            batch_efficiency = self.efficiency_calculation_jax(
                jnp.array(batch_r_mesh), jnp.array(batch_E_mesh)
            )
            efficiency_map[:, start_idx:end_idx] = np.array(batch_efficiency)
            
            # Calculate negative energy densities
            for field_idx, E_field in enumerate(batch_fields):
                global_field_idx = start_idx + field_idx
                
                # Sample spatial points for integration
                n_spatial_samples = 1000
                x_samples = np.random.uniform(0, self.specs.spatial_extent, n_spatial_samples)
                y_samples = np.random.uniform(0, self.specs.spatial_extent, n_spatial_samples)
                z_samples = np.random.uniform(0, self.specs.spatial_extent, n_spatial_samples)
                
                for r_idx, r_squeeze in enumerate(self.r_squeeze_array):
                    # Calculate average negative energy density
                    rho_neg_samples = self.negative_energy_density_jax(
                        jnp.array(r_squeeze), jnp.array(E_field),
                        jnp.array(x_samples), jnp.array(y_samples), jnp.array(z_samples)
                    )
                    negative_energy_map[r_idx, global_field_idx] = np.mean(rho_neg_samples)
            
            # Find optimal squeezing for each field in batch
            for field_idx, E_field in enumerate(batch_fields):
                global_field_idx = start_idx + field_idx
                field_efficiencies = efficiency_map[:, global_field_idx]
                
                # Find maximum efficiency
                max_idx = np.argmax(field_efficiencies)
                optimal_squeezing[global_field_idx] = self.r_squeeze_array[max_idx]
            
            # Progress update
            progress = (batch_idx + 1) / n_batches * 100
            if batch_idx % 10 == 0:
                print(f"   📈 Progress: {progress:.1f}% - "
                      f"Batch {batch_idx+1}/{n_batches}")
        
        # Store results
        results.optimal_squeezing = optimal_squeezing
        results.enhancement_factors = enhancement_map
        results.efficiency_map = efficiency_map
        results.negative_energy_density = negative_energy_map
        results.field_regimes = self.E_field_array
        
        # Calculate universal scaling coefficients
        print("\n🔧 Deriving Universal Scaling Laws...")
        scaling_coeffs = self.derive_universal_scaling_laws(optimal_squeezing)
        results.universal_scaling_coeffs = scaling_coeffs
        
        # Performance metrics
        computation_time = time.time() - start_time
        results.computational_performance = {
            'total_time': computation_time,
            'points_per_second': (self.specs.n_field_points * self.specs.n_squeeze_points) / computation_time,
            'memory_usage_gb': self.estimate_memory_usage(),
            'optimization_efficiency': self.calculate_optimization_efficiency(results)
        }
        
        print(f"✅ Parameter sweep completed in {computation_time:.1f} seconds")
        print(f"   Processing rate: {results.computational_performance['points_per_second']:.1e} points/second")
        print(f"   Memory usage: {results.computational_performance['memory_usage_gb']:.2f} GB")
        
        return results
    
    def derive_universal_scaling_laws(self, optimal_squeezing: np.ndarray) -> Dict[str, float]:
        """Derive universal scaling law coefficients"""
        print("   🔬 Fitting universal scaling law: r_opt = α·log(E/E_c) + β·(E/E_c)^γ")
        
        # Prepare data for fitting
        E_ratio = self.E_field_array / self.specs.E_crit
        
        # Define fitting function
        def scaling_law(E_ratio, alpha, beta, gamma):
            return alpha * np.log(E_ratio + 1e-10) + beta * E_ratio**gamma
        
        # Fit using robust optimization
        try:
            # Initial guess
            p0 = [1.0, 0.5, 0.5]
            
            # Robust fitting with bounds
            bounds = ([-10, 0, 0.1], [10, 10, 2.0])
            
            from scipy.optimize import curve_fit
            popt, pcov = curve_fit(scaling_law, E_ratio, optimal_squeezing, 
                                 p0=p0, bounds=bounds, maxfev=10000)
            
            alpha, beta, gamma = popt
            
            # Calculate goodness of fit
            r_predicted = scaling_law(E_ratio, alpha, beta, gamma)
            r_squared = 1 - np.sum((optimal_squeezing - r_predicted)**2) / \
                           np.sum((optimal_squeezing - np.mean(optimal_squeezing))**2)
            
            # Calculate parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov))
            
            coefficients = {
                'alpha': alpha,
                'beta': beta, 
                'gamma': gamma,
                'alpha_error': param_errors[0],
                'beta_error': param_errors[1],
                'gamma_error': param_errors[2],
                'r_squared': r_squared,
                'fitting_success': True
            }
            
            print(f"   📊 Scaling law: r_opt = {alpha:.3f}·log(E/E_c) + {beta:.3f}·(E/E_c)^{gamma:.3f}")
            print(f"   📊 Goodness of fit: R² = {r_squared:.6f}")
            print(f"   📊 Parameter uncertainties: α±{param_errors[0]:.3f}, β±{param_errors[1]:.3f}, γ±{param_errors[2]:.3f}")
            
        except Exception as e:
            print(f"   ⚠️  Scaling law fitting failed: {e}")
            coefficients = {'fitting_success': False}
        
        return coefficients
    
    def validate_negative_energy_enhancements(self, results: SqueezingResults) -> Dict[str, float]:
        """Validate negative energy enhancement characterization"""
        print("\n🔬 Validating Negative Energy Enhancement Characteristics")
        print("=" * 55)
        
        validation_results = {}
        
        # 1. Enhancement scaling analysis
        max_enhancements = np.max(results.enhancement_factors, axis=0)
        min_field_for_enhancement = self.E_field_array[max_enhancements > 2.0][0]  # >2× enhancement
        saturation_field = self.E_field_array[max_enhancements > 0.95 * np.max(max_enhancements)][0]
        
        validation_results['min_enhancement_field'] = min_field_for_enhancement
        validation_results['saturation_field'] = saturation_field
        validation_results['max_enhancement_factor'] = np.max(max_enhancements)
        
        print(f"📊 Minimum field for 2× enhancement: {min_field_for_enhancement/1e15:.2f} × 10^15 V/m")
        print(f"📊 Enhancement saturation field: {saturation_field/1e15:.2f} × 10^15 V/m") 
        print(f"📊 Maximum enhancement factor: {np.max(max_enhancements):.1f}×")
        
        # 2. Negative energy density scaling
        max_negative_energy = np.min(results.negative_energy_density, axis=0)  # Most negative
        negative_energy_range = np.max(max_negative_energy) - np.min(max_negative_energy)
        
        validation_results['max_negative_energy_density'] = np.min(max_negative_energy)
        validation_results['negative_energy_range'] = negative_energy_range
        validation_results['energy_density_scaling_exponent'] = self.calculate_scaling_exponent(
            self.E_field_array, -max_negative_energy
        )
        
        print(f"📊 Maximum negative energy density: {np.min(max_negative_energy):.2e} J/m³")
        print(f"📊 Energy density dynamic range: {negative_energy_range:.2e} J/m³")
        print(f"📊 Energy density scaling exponent: {validation_results['energy_density_scaling_exponent']:.3f}")
        
        # 3. Efficiency validation
        max_efficiencies = np.max(results.efficiency_map, axis=0)
        efficiency_threshold_field = self.E_field_array[max_efficiencies > 0.01][0]  # >1% efficiency
        peak_efficiency = np.max(max_efficiencies)
        peak_efficiency_field = self.E_field_array[np.argmax(max_efficiencies)]
        
        validation_results['efficiency_threshold_field'] = efficiency_threshold_field
        validation_results['peak_efficiency'] = peak_efficiency
        validation_results['peak_efficiency_field'] = peak_efficiency_field
        
        print(f"📊 Field for 1% efficiency threshold: {efficiency_threshold_field/1e15:.2f} × 10^15 V/m")
        print(f"📊 Peak efficiency: {peak_efficiency*100:.3f}%")
        print(f"📊 Peak efficiency field: {peak_efficiency_field/1e15:.2f} × 10^15 V/m")
        
        # 4. Universal scaling validation
        if results.universal_scaling_coeffs.get('fitting_success', False):
            alpha = results.universal_scaling_coeffs['alpha']
            beta = results.universal_scaling_coeffs['beta']
            gamma = results.universal_scaling_coeffs['gamma']
            r_squared = results.universal_scaling_coeffs['r_squared']
            
            validation_results['universal_scaling_quality'] = r_squared
            validation_results['scaling_law_valid'] = r_squared > 0.95
            
            print(f"📊 Universal scaling law R²: {r_squared:.6f}")
            if r_squared > 0.95:
                print("✅ Universal scaling law validated (R² > 0.95)")
            else:
                print("⚠️  Universal scaling law needs refinement")
        
        return validation_results
    
    def calculate_scaling_exponent(self, x_data: np.ndarray, y_data: np.ndarray) -> float:
        """Calculate power-law scaling exponent"""
        # Fit log(y) = log(a) + b*log(x)
        valid_indices = (x_data > 0) & (y_data > 0)
        if np.sum(valid_indices) < 10:
            return 0.0
        
        log_x = np.log(x_data[valid_indices])
        log_y = np.log(y_data[valid_indices])
        
        # Linear regression in log space
        coeffs = np.polyfit(log_x, log_y, 1)
        return coeffs[0]  # Scaling exponent
    
    def estimate_memory_usage(self) -> float:
        """Estimate memory usage in GB"""
        # Arrays: E_mesh, r_mesh, enhancement_map, efficiency_map, negative_energy_map
        array_elements = (2 + 3) * self.specs.n_field_points * self.specs.n_squeeze_points
        array_elements += self.specs.spatial_resolution**3 * 3  # Spatial grids
        
        # Assume 8 bytes per float64 element
        memory_bytes = array_elements * 8
        return memory_bytes / (1024**3)  # Convert to GB
    
    def calculate_optimization_efficiency(self, results: SqueezingResults) -> float:
        """Calculate computational optimization efficiency"""
        # Efficiency = (Useful computations) / (Total computations)
        total_computations = self.specs.n_field_points * self.specs.n_squeeze_points
        
        # Count computations that led to meaningful results
        meaningful_efficiencies = results.efficiency_map[results.efficiency_map > 1e-10]
        useful_computations = len(meaningful_efficiencies)
        
        return useful_computations / total_computations
    
    def generate_optimization_report(self, results: SqueezingResults, 
                                   validation_results: Dict[str, float]) -> str:
        """Generate comprehensive optimization report"""
        report = []
        report.append("="*80)
        report.append("UNIVERSAL SQUEEZING PARAMETER OPTIMIZATION REPORT")
        report.append("="*80)
        report.append("")
        
        # Summary statistics
        report.append("📊 SUMMARY STATISTICS")
        report.append("-" * 40)
        report.append(f"Parameter Space Coverage:")
        report.append(f"  • Field Range: {self.specs.E_min/1e12:.1f} - {self.specs.E_max/1e15:.0f} × 10^15 V/m")
        report.append(f"  • Squeezing Range: {self.specs.r_min:.1f} - {self.specs.r_max:.1f}")
        report.append(f"  • Total Combinations: {self.specs.n_field_points * self.specs.n_squeeze_points:,}")
        report.append("")
        
        # Key findings
        report.append("🔬 KEY FINDINGS")
        report.append("-" * 40)
        report.append(f"Maximum Enhancement Factor: {validation_results.get('max_enhancement_factor', 0):.1f}×")
        report.append(f"Peak Efficiency: {validation_results.get('peak_efficiency', 0)*100:.3f}%")
        report.append(f"Maximum Negative Energy Density: {validation_results.get('max_negative_energy_density', 0):.2e} J/m³")
        report.append(f"Universal Scaling R²: {validation_results.get('universal_scaling_quality', 0):.6f}")
        report.append("")
        
        # Performance metrics
        report.append("⚡ COMPUTATIONAL PERFORMANCE")
        report.append("-" * 40)
        perf = results.computational_performance
        report.append(f"Total Computation Time: {perf['total_time']:.1f} seconds")
        report.append(f"Processing Rate: {perf['points_per_second']:.2e} points/second")
        report.append(f"Memory Usage: {perf['memory_usage_gb']:.2f} GB")
        report.append(f"Optimization Efficiency: {perf['optimization_efficiency']*100:.1f}%")
        report.append("")
        
        # Universal scaling law
        if results.universal_scaling_coeffs.get('fitting_success', False):
            report.append("🎯 UNIVERSAL SCALING LAW")
            report.append("-" * 40)
            alpha = results.universal_scaling_coeffs['alpha']
            beta = results.universal_scaling_coeffs['beta']
            gamma = results.universal_scaling_coeffs['gamma']
            report.append(f"r_opt = {alpha:.3f}·log(E/E_c) + {beta:.3f}·(E/E_c)^{gamma:.3f}")
            report.append(f"Goodness of Fit: R² = {results.universal_scaling_coeffs['r_squared']:.6f}")
            report.append("")
        
        # Recommendations
        report.append("💡 OPTIMIZATION RECOMMENDATIONS")
        report.append("-" * 40)
        
        peak_field = validation_results.get('peak_efficiency_field', 0)
        if peak_field > 0:
            optimal_r = results.optimal_squeezing[np.argmin(np.abs(results.field_regimes - peak_field))]
            report.append(f"• Optimal Operating Point: E = {peak_field/1e15:.2f} × 10^15 V/m, r = {optimal_r:.2f}")
        
        threshold_field = validation_results.get('efficiency_threshold_field', 0)
        if threshold_field > 0:
            report.append(f"• Minimum Practical Field: {threshold_field/1e15:.2f} × 10^15 V/m (1% efficiency)")
        
        saturation_field = validation_results.get('saturation_field', 0)
        if saturation_field > 0:
            report.append(f"• Enhancement Saturation: {saturation_field/1e15:.2f} × 10^15 V/m")
        
        report.append("")
        report.append("="*80)
        
        return "\n".join(report)
    
    def save_results_and_visualizations(self, results: SqueezingResults, 
                                      validation_results: Dict[str, float],
                                      save_plots: bool = True) -> None:
        """Save results and generate visualizations"""
        if save_plots:
            print("\n📊 Generating Optimization Visualizations...")
            
            # Create comprehensive figure
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Universal Squeezing Parameter Optimization Results', fontsize=16)
            
            # 1. Enhancement factor map
            im1 = axes[0,0].contourf(self.E_field_array/1e15, self.r_squeeze_array, 
                                   results.enhancement_factors, levels=50, cmap='viridis')
            axes[0,0].set_xlabel('Electric Field (×10¹⁵ V/m)')
            axes[0,0].set_ylabel('Squeezing Parameter r')
            axes[0,0].set_title('Enhancement Factor ξ(r,E)')
            axes[0,0].set_xscale('log')
            plt.colorbar(im1, ax=axes[0,0])
            
            # 2. Efficiency map
            im2 = axes[0,1].contourf(self.E_field_array/1e15, self.r_squeeze_array,
                                   results.efficiency_map * 100, levels=50, cmap='plasma')
            axes[0,1].set_xlabel('Electric Field (×10¹⁵ V/m)')
            axes[0,1].set_ylabel('Squeezing Parameter r')
            axes[0,1].set_title('Conversion Efficiency (%)')
            axes[0,1].set_xscale('log')
            plt.colorbar(im2, ax=axes[0,1])
            
            # 3. Optimal squeezing vs field
            axes[0,2].semilogx(self.E_field_array/1e15, results.optimal_squeezing, 'b-', linewidth=2)
            axes[0,2].set_xlabel('Electric Field (×10¹⁵ V/m)')
            axes[0,2].set_ylabel('Optimal Squeezing r_opt')
            axes[0,2].set_title('Universal Squeezing Law')
            axes[0,2].grid(True, alpha=0.3)
            
            # Universal scaling law overlay
            if results.universal_scaling_coeffs.get('fitting_success', False):
                alpha = results.universal_scaling_coeffs['alpha']
                beta = results.universal_scaling_coeffs['beta']
                gamma = results.universal_scaling_coeffs['gamma']
                E_ratio = self.E_field_array / self.specs.E_crit
                r_fit = alpha * np.log(E_ratio + 1e-10) + beta * E_ratio**gamma
                axes[0,2].semilogx(self.E_field_array/1e15, r_fit, 'r--', linewidth=2, 
                                 label=f'Fit: α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}')
                axes[0,2].legend()
            
            # 4. Negative energy density
            max_negative = np.min(results.negative_energy_density, axis=0)
            axes[1,0].loglog(self.E_field_array/1e15, -max_negative, 'g-', linewidth=2)
            axes[1,0].set_xlabel('Electric Field (×10¹⁵ V/m)')
            axes[1,0].set_ylabel('Max Negative Energy Density (J/m³)')
            axes[1,0].set_title('Negative Energy Enhancement')
            axes[1,0].grid(True, alpha=0.3)
            
            # 5. Maximum efficiency vs field
            max_eff = np.max(results.efficiency_map, axis=0)
            axes[1,1].loglog(self.E_field_array/1e15, max_eff * 100, 'm-', linewidth=2)
            axes[1,1].set_xlabel('Electric Field (×10¹⁵ V/m)')
            axes[1,1].set_ylabel('Maximum Efficiency (%)')
            axes[1,1].set_title('Peak Efficiency Scaling')
            axes[1,1].grid(True, alpha=0.3)
            
            # 6. Enhancement factor distribution
            max_enhancement = np.max(results.enhancement_factors, axis=0)
            axes[1,2].loglog(self.E_field_array/1e15, max_enhancement, 'c-', linewidth=2)
            axes[1,2].set_xlabel('Electric Field (×10¹⁵ V/m)')
            axes[1,2].set_ylabel('Maximum Enhancement Factor')
            axes[1,2].set_title('Enhancement Scaling')
            axes[1,2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('universal_squeezing_optimization_results.png', dpi=300, bbox_inches='tight')
            print("   💾 Saved: universal_squeezing_optimization_results.png")
            
            plt.show()

def main():
    """Main universal squeezing parameter optimization"""
    print("🔬 Universal Squeezing Parameter Optimization Framework")
    print("=" * 65)
    
    # Create optimization specifications
    specs = SqueezingSpecs(
        E_min=1e12,             # 1 TV/m
        E_max=1e18,             # 1 EV/m  
        n_field_points=500,     # Field sampling points
        n_squeeze_points=300,   # Squeezing sampling points
        spatial_resolution=64,  # 64³ spatial grid
        optimization_tolerance=1e-12
    )
    
    # Initialize optimizer
    optimizer = UniversalSqueezingOptimizer(specs)
    
    # Execute comprehensive optimization
    print(f"\n🎯 Optimization Target:")
    print(f"   Field Range: {specs.E_min/1e12:.0f} - {specs.E_max/1e15:.0f} × 10^15 V/m")
    print(f"   Parameter Space: {specs.n_field_points * specs.n_squeeze_points:,} combinations")
    print(f"   Spatial Resolution: {specs.spatial_resolution}³ = {specs.spatial_resolution**3:,} points")
    
    # Run optimization
    results = optimizer.execute_comprehensive_parameter_sweep()
    
    # Validate results
    validation_results = optimizer.validate_negative_energy_enhancements(results)
    
    # Generate comprehensive report
    report = optimizer.generate_optimization_report(results, validation_results)
    print("\n" + report)
    
    # Save results and visualizations
    optimizer.save_results_and_visualizations(results, validation_results, save_plots=True)
    
    # Final summary
    print("\n" + "="*65)
    print("🎯 UNIVERSAL SQUEEZING OPTIMIZATION COMPLETE")
    print("="*65)
    
    perf = results.computational_performance
    print(f"✅ Processing Rate: {perf['points_per_second']:.2e} points/second")
    print(f"✅ Optimization Efficiency: {perf['optimization_efficiency']*100:.1f}%")
    print(f"✅ Universal Scaling R²: {validation_results.get('universal_scaling_quality', 0):.6f}")
    
    if validation_results.get('scaling_law_valid', False):
        print("🎉 Universal scaling law successfully validated!")
    else:
        print("📊 Additional optimization recommended for scaling law refinement")
    
    return results, validation_results

if __name__ == "__main__":
    main()
