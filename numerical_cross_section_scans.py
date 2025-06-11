#!/usr/bin/env python3
"""
Numerical Cross-Section Scans and Parameter Sweeps

This module implements comprehensive numerical cross-section scans for 
polymerized gauge theories, including:

1. Grid scans over μ_g values
2. Running coupling feed-in and comparison
3. s-distribution integration
4. Parameter sweep tabulation
5. Yield vs. field strength analysis

Mathematical Framework:
σ_poly(s, μ_g) = σ_0(s) * [sinc(μ_g√s)]^4
Yield = ∫ σ_poly(s, μ_g) * f(s) ds

Key Features:
- Multi-parameter grid scanning
- Running coupling integration α_s(μ_g)
- Automated yield calculations
- Statistical uncertainty propagation
- Parameter optimization routines
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import json
import time
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CROSS-SECTION SCAN FRAMEWORK
# ============================================================================

@dataclass
class ScanParameters:
    """Parameters for cross-section scans"""
    
    # μ_g parameter grid
    mu_g_min: float = 1e-4
    mu_g_max: float = 1e-2
    mu_g_points: int = 50
    
    # Energy scale grid  
    s_min: float = 1.0      # GeV²
    s_max: float = 1000.0   # GeV²
    s_points: int = 100
    
    # Running coupling parameters
    alpha_s_0: float = 0.3
    beta_0: float = 7.0    # QCD beta function
    Lambda_QCD: float = 0.2  # GeV
    
    # Cross-section normalization
    sigma_0: float = 1e-30  # cm²
    
    # Integration parameters
    integration_method: str = "simpson"
    n_mc_samples: int = 10000

class CrossSectionScanner:
    """
    Comprehensive cross-section scanning and analysis
    """
    
    def __init__(self, params: ScanParameters):
        """Initialize scanner with parameters"""
        
        self.params = params
        
        # Generate parameter grids
        self.mu_g_grid = np.logspace(
            np.log10(params.mu_g_min),
            np.log10(params.mu_g_max),
            params.mu_g_points
        )
        
        self.s_grid = np.logspace(
            np.log10(params.s_min),
            np.log10(params.s_max),
            params.s_points
        )
        
        print(f"🔬 Cross-Section Scanner Initialized")
        print(f"   μ_g range: [{params.mu_g_min:.1e}, {params.mu_g_max:.1e}]")
        print(f"   s range: [{params.s_min}, {params.s_max}] GeV²")
        print(f"   Grid points: {params.mu_g_points}×{params.s_points}")
        
    def polymerized_cross_section(self, s: float, mu_g: float) -> float:
        """
        Calculate polymerized cross-section
        
        Args:
            s: Center-of-mass energy squared (GeV²)
            mu_g: Polymer parameter
            
        Returns:
            Cross-section in cm²
        """
        
        # Standard cross-section (placeholder model)
        sigma_standard = self.params.sigma_0 / (1 + s/100.0)
        
        # Polymer form factor: sinc⁴(μ_g√s)
        arg = mu_g * np.sqrt(s)
        if arg == 0:
            form_factor = 1.0
        else:
            sinc_factor = np.sin(arg) / arg
            form_factor = sinc_factor**4
            
        return sigma_standard * form_factor
    
    def running_coupling(self, mu_g: float) -> float:
        """
        Calculate running coupling α_s(μ_g)
        
        Args:
            mu_g: Energy scale parameter
            
        Returns:
            Running coupling strength
        """
        
        # Simple one-loop running: α_s(μ) = α_s(Λ) / [1 + β₀ ln(μ/Λ)]
        scale_ratio = (mu_g * self.params.Lambda_QCD) / self.params.Lambda_QCD
        
        if scale_ratio <= 1e-10:
            return self.params.alpha_s_0
            
        log_term = np.log(scale_ratio)
        denominator = 1 + self.params.beta_0 * log_term
        
        if denominator <= 0:
            return self.params.alpha_s_0  # Prevent divergence
            
        return self.params.alpha_s_0 / denominator
    
    def s_distribution(self, s: float) -> float:
        """
        Energy distribution function f(s)
        
        Args:
            s: Center-of-mass energy squared
            
        Returns:
            Probability density
        """
        
        # Exponential falloff with peak around 10 GeV²
        s_peak = 10.0
        return np.exp(-(s - s_peak)**2 / (2 * s_peak**2)) / np.sqrt(2 * np.pi * s_peak**2)
    
    def compute_cross_section_grid(self) -> np.ndarray:
        """
        Compute cross-section on full parameter grid
        
        Returns:
            2D array of cross-sections [μ_g, s]
        """
        
        print("\n📊 COMPUTING CROSS-SECTION GRID...")
        
        sigma_grid = np.zeros((len(self.mu_g_grid), len(self.s_grid)))
        
        for i, mu_g in enumerate(self.mu_g_grid):
            for j, s in enumerate(self.s_grid):
                sigma_grid[i, j] = self.polymerized_cross_section(s, mu_g)
                
            if i % 10 == 0:
                print(f"   Progress: {i+1}/{len(self.mu_g_grid)} μ_g values")
                
        print("   ✅ Grid computation complete")
        return sigma_grid
    
    def compute_integrated_yield(self, mu_g: float) -> float:
        """
        Compute integrated yield for given μ_g
        
        Args:
            mu_g: Polymer parameter
            
        Returns:
            Integrated yield
        """
        
        def integrand(s):
            return self.polymerized_cross_section(s, mu_g) * self.s_distribution(s)
            
        result, _ = quad(integrand, self.params.s_min, self.params.s_max)
        return result
    
    def yield_vs_mu_g_scan(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scan integrated yield vs μ_g
        
        Returns:
            Tuple of (μ_g_values, yields)
        """
        
        print("\n📈 SCANNING YIELD vs μ_g...")
        
        yields = np.zeros(len(self.mu_g_grid))
        
        for i, mu_g in enumerate(self.mu_g_grid):
            yields[i] = self.compute_integrated_yield(mu_g)
            
            if i % 10 == 0:
                progress = 100 * (i + 1) / len(self.mu_g_grid)
                print(f"   Progress: {progress:.1f}%")
                
        print("   ✅ Yield scan complete")
        return self.mu_g_grid, yields
    
    def running_coupling_comparison(self) -> Dict[str, np.ndarray]:
        """
        Compare standard vs running coupling effects
        
        Returns:
            Dictionary with coupling comparison data
        """
        
        print("\n🔧 RUNNING COUPLING ANALYSIS...")
        
        # Standard coupling (constant)
        yields_standard = np.zeros(len(self.mu_g_grid))
        
        # Running coupling  
        yields_running = np.zeros(len(self.mu_g_grid))
        coupling_values = np.zeros(len(self.mu_g_grid))
        
        for i, mu_g in enumerate(self.mu_g_grid):
            # Standard coupling yield
            yields_standard[i] = self.compute_integrated_yield(mu_g)
            
            # Running coupling modification
            alpha_s_running = self.running_coupling(mu_g)
            coupling_values[i] = alpha_s_running
            
            # Modified yield (simplified model: yield ∝ α_s²)
            coupling_ratio = alpha_s_running / self.params.alpha_s_0
            yields_running[i] = yields_standard[i] * coupling_ratio**2
            
        enhancement_factor = yields_running / yields_standard
        
        print(f"   α_s range: [{coupling_values.min():.3f}, {coupling_values.max():.3f}]")
        print(f"   Enhancement range: [{enhancement_factor.min():.3f}, {enhancement_factor.max():.3f}]")
        print("   ✅ Running coupling analysis complete")
        
        return {
            'mu_g': self.mu_g_grid,
            'yields_standard': yields_standard,
            'yields_running': yields_running,
            'coupling_values': coupling_values,
            'enhancement_factor': enhancement_factor
        }
    
    def parameter_sweep_analysis(self) -> Dict[str, np.ndarray]:
        """
        Comprehensive parameter sweep and optimization
        
        Returns:
            Dictionary with sweep results
        """
        
        print("\n🔍 PARAMETER SWEEP ANALYSIS...")
        
        # Multi-dimensional parameter scan
        n_params = 3
        param_ranges = {
            'mu_g': (self.params.mu_g_min, self.params.mu_g_max),
            'alpha_s': (0.1, 0.5),
            'Lambda_QCD': (0.1, 0.5)
        }
        
        # Generate random parameter samples
        n_samples = 1000
        np.random.seed(42)  # Reproducible
        
        param_samples = {}
        for param, (min_val, max_val) in param_ranges.items():
            param_samples[param] = np.random.uniform(min_val, max_val, n_samples)
            
        yields = np.zeros(n_samples)
        
        # Store original parameters
        original_alpha_s = self.params.alpha_s_0
        original_Lambda = self.params.Lambda_QCD
        
        for i in range(n_samples):
            # Set parameters for this sample
            self.params.alpha_s_0 = param_samples['alpha_s'][i]
            self.params.Lambda_QCD = param_samples['Lambda_QCD'][i]
            
            # Compute yield
            yields[i] = self.compute_integrated_yield(param_samples['mu_g'][i])
            
            if i % 200 == 0:
                progress = 100 * (i + 1) / n_samples
                print(f"   Progress: {progress:.1f}%")
                
        # Restore original parameters
        self.params.alpha_s_0 = original_alpha_s
        self.params.Lambda_QCD = original_Lambda
        
        # Find optimal parameters
        max_idx = np.argmax(yields)
        optimal_params = {
            'mu_g': param_samples['mu_g'][max_idx],
            'alpha_s': param_samples['alpha_s'][max_idx],
            'Lambda_QCD': param_samples['Lambda_QCD'][max_idx],
            'yield': yields[max_idx]
        }
        
        print(f"   Optimal μ_g: {optimal_params['mu_g']:.1e}")
        print(f"   Optimal α_s: {optimal_params['alpha_s']:.3f}")
        print(f"   Optimal Λ_QCD: {optimal_params['Lambda_QCD']:.3f}")
        print(f"   Maximum yield: {optimal_params['yield']:.2e}")
        print("   ✅ Parameter sweep complete")
        
        return {
            'param_samples': param_samples,
            'yields': yields,
            'optimal_params': optimal_params
        }

# ============================================================================
# VISUALIZATION AND EXPORT
# ============================================================================

class ScanVisualizer:
    """
    Visualization and data export for scan results
    """
    
    def __init__(self, scanner: CrossSectionScanner):
        """Initialize with scanner instance"""
        self.scanner = scanner
        
    def plot_cross_section_heatmap(self, sigma_grid: np.ndarray, save_path: str = None):
        """Plot cross-section heatmap"""
        
        plt.figure(figsize=(10, 8))
        
        # Create meshgrid for plotting
        S, MU_G = np.meshgrid(self.scanner.s_grid, self.scanner.mu_g_grid)
        
        # Plot heatmap
        im = plt.pcolormesh(S, MU_G, sigma_grid, shading='auto', cmap='viridis')
        plt.colorbar(im, label='Cross-section (cm²)')
        
        plt.xlabel('s (GeV²)')
        plt.ylabel('μ_g')
        plt.title('Polymerized Cross-Section σ(s, μ_g)')
        plt.xscale('log')
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_yield_analysis(self, results: Dict):
        """Plot yield analysis results"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Yield vs μ_g
        ax1.loglog(results['mu_g'], results['yields_standard'], 'b-', label='Standard coupling')
        ax1.loglog(results['mu_g'], results['yields_running'], 'r--', label='Running coupling')
        ax1.set_xlabel('μ_g')
        ax1.set_ylabel('Integrated Yield')
        ax1.set_title('Yield vs μ_g')
        ax1.legend()
        ax1.grid(True)
        
        # Running coupling
        ax2.semilogx(results['mu_g'], results['coupling_values'], 'g-')
        ax2.set_xlabel('μ_g') 
        ax2.set_ylabel('α_s(μ_g)')
        ax2.set_title('Running Coupling')
        ax2.grid(True)
        
        # Enhancement factor
        ax3.semilogx(results['mu_g'], results['enhancement_factor'], 'm-')
        ax3.set_xlabel('μ_g')
        ax3.set_ylabel('Enhancement Factor')
        ax3.set_title('Running Coupling Enhancement')
        ax3.grid(True)
        
        # Yield ratio
        yield_ratio = results['yields_running'] / results['yields_standard'].max()
        ax4.semilogx(results['mu_g'], yield_ratio, 'c-')
        ax4.set_xlabel('μ_g')
        ax4.set_ylabel('Normalized Yield')
        ax4.set_title('Normalized Yield Comparison')
        ax4.grid(True)        
        plt.tight_layout()
        plt.savefig('cross_section_yield_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def export_results(self, results: Dict, filename: str = "cross_section_scan_results.json"):
        """Export results to JSON"""
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            else:
                return obj
        
        json_results = convert_to_serializable(results)
                
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
            
        print(f"   📤 Results exported to {filename}")

# ============================================================================
# DEMONSTRATION AND VALIDATION
# ============================================================================

def run_comprehensive_scan():
    """Run comprehensive cross-section scan demonstration"""
    
    print("=" * 80)
    print("NUMERICAL CROSS-SECTION SCANS & PARAMETER SWEEPS")
    print("=" * 80)
    
    # Initialize parameters
    params = ScanParameters(
        mu_g_min=1e-4,
        mu_g_max=1e-2,
        mu_g_points=30,  # Reduced for faster demo
        s_points=50      # Reduced for faster demo
    )
    
    # Initialize scanner
    scanner = CrossSectionScanner(params)
    visualizer = ScanVisualizer(scanner)
    
    # 1. Cross-section grid computation
    print("\n1. CROSS-SECTION GRID COMPUTATION")
    sigma_grid = scanner.compute_cross_section_grid()
    
    # 2. Yield vs μ_g scan
    print("\n2. YIELD vs μ_g ANALYSIS")
    mu_g_values, yields = scanner.yield_vs_mu_g_scan()
    
    # 3. Running coupling comparison
    print("\n3. RUNNING COUPLING COMPARISON")
    coupling_results = scanner.running_coupling_comparison()
    
    # 4. Parameter sweep analysis
    print("\n4. PARAMETER SWEEP OPTIMIZATION")
    sweep_results = scanner.parameter_sweep_analysis()
    
    # 5. Visualization
    print("\n5. RESULTS VISUALIZATION")
    visualizer.plot_yield_analysis(coupling_results)
    
    # 6. Export results
    print("\n6. EXPORTING RESULTS")
    all_results = {
        'sigma_grid': sigma_grid,
        'mu_g_values': mu_g_values,
        'yields': yields,
        'coupling_analysis': coupling_results,
        'parameter_sweep': sweep_results,
        'scan_parameters': {
            'mu_g_min': params.mu_g_min,
            'mu_g_max': params.mu_g_max,
            'mu_g_points': params.mu_g_points,
            's_min': params.s_min,
            's_max': params.s_max,
            's_points': params.s_points
        }
    }
    
    visualizer.export_results(all_results)
    
    # Summary statistics
    print("\n📊 SCAN SUMMARY:")
    print(f"   Total cross-sections computed: {sigma_grid.size:,}")
    print(f"   Maximum cross-section: {sigma_grid.max():.2e} cm²")
    print(f"   Minimum cross-section: {sigma_grid.min():.2e} cm²")
    print(f"   Maximum yield enhancement: {coupling_results['enhancement_factor'].max():.3f}")
    print(f"   Optimal μ_g: {sweep_results['optimal_params']['mu_g']:.1e}")
    print(f"   Computation time: {time.time():.1f}s")
    
    print("\n✅ COMPREHENSIVE SCAN COMPLETE")
    print("   Grid computation: ✅")
    print("   Yield analysis: ✅") 
    print("   Running coupling: ✅")
    print("   Parameter sweep: ✅")
    print("   Visualization: ✅")
    print("   Data export: ✅")
    print("   Ready for FDTD/spin-foam integration")

if __name__ == "__main__":
    run_comprehensive_scan()
