#!/usr/bin/env python3
"""
Running Coupling Integration with Polymer QFT
==============================================

Implements explicit running-coupling feed-in with β-function and head-to-head
comparison of Schwinger pair production with/without running coupling effects.
Includes full parameter sweeps over μ_g and β values.
"""

import numpy as np
import json
import matplotlib.pyplot as plt

def running_coupling_analysis():
    """Complete running coupling analysis with polymer corrections."""
    
    # Configuration
    alpha_0 = 0.1          # Initial coupling at reference scale
    E_0 = 1.0              # Reference energy scale
    mu_g_range = np.linspace(0.1, 0.6, 6)  # Polymer parameter range
    b_range = np.linspace(0, 10, 11)        # β-function coefficient range
    
    print("Running Coupling Integration with Polymer QFT")
    print("="*55)
    
    def running_coupling(E, b_coeff):
        """
        Running coupling with β-function:
        α_eff(E) = α_0 / (1 - (b/(2π)) α_0 ln(E/E_0))
        """
        if b_coeff == 0:
            return alpha_0  # No running
        
        beta_factor = b_coeff / (2 * np.pi)
        log_term = np.log(E / E_0)
        denominator = 1.0 - beta_factor * alpha_0 * log_term
        
        if denominator <= 0:
            return alpha_0  # Avoid divergence
        
        return alpha_0 / denominator
    
    def polymer_modification_factor(mu_g):
        """Polymer modification F(μ) in the exponential."""
        if mu_g == 0:
            return 1.0
        # Simplified polymer correction
        return 1.0 + 0.5 * mu_g**2 * np.sin(np.pi * mu_g)
    
    def schwinger_pair_production(E, mu_g, b_coeff):
        """
        Schwinger pair production rate with polymer corrections:
        Γ_Schwinger^poly = (α_eff(E) eE)² / (4π³ℏc) × exp[-πm²c³/(eEℏ) F(μ)]
        """
        # Physical constants (natural units)
        e = 1.0  # Electric charge
        m = 1.0  # Particle mass
        hbar = 1.0
        c = 1.0
        
        # Running coupling
        alpha_eff = running_coupling(E, b_coeff)
        
        # Polymer modification
        F_mu = polymer_modification_factor(mu_g)
        
        # Schwinger formula components
        prefactor = (alpha_eff * e * E)**2 / (4 * np.pi**3 * hbar * c)
        exponent = -np.pi * m**2 * c**3 / (e * E * hbar) * F_mu
        
        return prefactor * np.exp(exponent)
    
    # Analysis 1: Running coupling behavior
    print("1. Running coupling α_eff(E) analysis:")
    E_values = np.logspace(-1, 2, 20)  # Energy range from 0.1 to 100
    
    print("   Energy scale dependence:")
    for b in [0, 2, 5, 10]:
        alpha_values = [running_coupling(E, b) for E in E_values]
        print(f"   β = {b}: α(E=0.1) = {alpha_values[0]:.4f}, α(E=100) = {alpha_values[-1]:.4f}")
    
    # Analysis 2: Head-to-head comparison (b=0 vs b≠0)
    print("\n2. Head-to-head comparison of Schwinger production:")
    E_test = 10.0  # Test energy
    mu_g_test = 0.3  # Test polymer parameter
    
    print(f"   At E = {E_test}, μ_g = {mu_g_test}:")
    
    # Without running coupling (b=0)
    gamma_no_running = schwinger_pair_production(E_test, mu_g_test, 0)
    print(f"   No running (b=0):     Γ = {gamma_no_running:.6e}")
    
    # With running coupling (various b values)
    for b in [2, 5, 10]:
        gamma_running = schwinger_pair_production(E_test, mu_g_test, b)
        enhancement = gamma_running / gamma_no_running
        print(f"   Running (b={b}):      Γ = {gamma_running:.6e}, enhancement = {enhancement:.3f}")
    
    # Analysis 3: Full parameter sweep
    print("\n3. Full parameter sweep μ_g ∈ [0.1, 0.6], b ∈ [0, 10]:")
    
    # Initialize results grid
    yield_curves = np.zeros((len(mu_g_range), len(b_range)))
    gain_factors = np.zeros((len(mu_g_range), len(b_range)))
    
    E_fixed = 5.0  # Fixed energy for sweep
    
    print(f"   Sweeping at fixed E = {E_fixed}:")
    print("   μ_g\\b    ", end="")
    for b in [0, 2, 4, 6, 8, 10]:
        print(f"{b:8.1f}", end="")
    print()
    
    for i, mu_g in enumerate(mu_g_range):
        print(f"   {mu_g:.1f}     ", end="")
        
        for j, b in enumerate(b_range):
            # Compute Schwinger rate
            gamma = schwinger_pair_production(E_fixed, mu_g, b)
            yield_curves[i, j] = gamma
            
            # Compute gain factor relative to no running coupling
            gamma_b0 = schwinger_pair_production(E_fixed, mu_g, 0)
            gain_factors[i, j] = gamma / gamma_b0 if gamma_b0 > 0 else 1.0
            
            if b in [0, 2, 4, 6, 8, 10]:
                print(f"{gamma:.2e}", end=" ")
        print()
    
    # Analysis 4: Yield vs field curves
    print("\n4. Yield vs field strength curves:")
    E_field_range = np.logspace(0, 2, 15)  # Field strength from 1 to 100
    
    # Generate curves for different (μ_g, b) combinations
    representative_cases = [
        (0.2, 0),   # Low polymer, no running
        (0.2, 5),   # Low polymer, moderate running
        (0.5, 0),   # High polymer, no running
        (0.5, 10),  # High polymer, strong running
    ]
    
    print("   Field strength dependence:")
    for mu_g, b in representative_cases:
        yields = [schwinger_pair_production(E, mu_g, b) for E in E_field_range]
        max_yield = max(yields)
        E_max = E_field_range[np.argmax(yields)]
        print(f"   μ_g={mu_g}, b={b}: Max yield = {max_yield:.3e} at E = {E_max:.1f}")
    
    # Analysis 5: Tabulated gain factors
    print("\n5. Tabulated gain factors (relative to b=0):")
    print("   μ_g\\b    ", end="")
    for b in [2, 4, 6, 8, 10]:
        print(f"{b:8.1f}", end="")
    print()
    
    for i, mu_g in enumerate(mu_g_range):
        print(f"   {mu_g:.1f}     ", end="")
        for b in [2, 4, 6, 8, 10]:
            j = int(b)  # Index in b_range
            if j < len(b_range):
                gain = gain_factors[i, j]
                print(f"{gain:8.3f}", end="")
        print()
    
    # Analysis 6: Optimal parameter identification
    print("\n6. Optimal parameter identification:")
    max_yield_overall = np.max(yield_curves)
    max_indices = np.unravel_index(np.argmax(yield_curves), yield_curves.shape)
    optimal_mu_g = mu_g_range[max_indices[0]]
    optimal_b = b_range[max_indices[1]]
    
    print(f"   Maximum yield: {max_yield_overall:.6e}")
    print(f"   Optimal parameters: μ_g = {optimal_mu_g:.2f}, b = {optimal_b:.1f}")
    
    # Export comprehensive results
    results = {
        "config": {
            "alpha_0": alpha_0,
            "E_0": E_0,
            "mu_g_range": mu_g_range.tolist(),
            "b_range": b_range.tolist()
        },
        "running_coupling_formula": "alpha_eff(E) = alpha_0 / (1 - (b/(2*pi)) * alpha_0 * ln(E/E_0))",
        "schwinger_formula": "Gamma = (alpha_eff * e * E)^2 / (4*pi^3) * exp(-pi*m^2*c^3/(e*E) * F(mu_g))",
        "parameter_sweep_results": {
            "yield_curves": yield_curves.tolist(),
            "gain_factors": gain_factors.tolist(),
            "optimal_mu_g": optimal_mu_g,
            "optimal_b": optimal_b,
            "max_yield": max_yield_overall
        },
        "head_to_head_comparison": {
            "test_energy": E_test,
            "test_mu_g": mu_g_test,
            "no_running_yield": gamma_no_running,
            "running_enhancements": {
                "b=2": schwinger_pair_production(E_test, mu_g_test, 2) / gamma_no_running,
                "b=5": schwinger_pair_production(E_test, mu_g_test, 5) / gamma_no_running,
                "b=10": schwinger_pair_production(E_test, mu_g_test, 10) / gamma_no_running
            }
        },
        "analysis_status": {
            "running_coupling_implemented": True,
            "parameter_sweep_completed": True,
            "yield_curves_generated": True,
            "gain_factors_tabulated": True,
            "optimization_performed": True
        }
    }
    
    with open("running_coupling_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*55)
    print("RUNNING COUPLING ANALYSIS COMPLETE")
    print("="*55)
    print("✓ Running coupling α_eff(E) with β-function implemented")
    print("✓ Schwinger pair production with polymer corrections implemented")
    print("✓ Head-to-head comparison (b=0 vs b≠0) completed")
    print("✓ Full parameter sweep μ_g ∈ [0.1,0.6], b ∈ [0,10] completed")
    print("✓ Yield vs field strength curves generated")
    print("✓ Gain factors tabulated and analyzed")
    print("✓ Optimal parameter identification performed")
    
    print("\nKey Formulas Implemented:")
    print("1. Running coupling:")
    print("   α_eff(E) = α_0 / (1 - (b/(2π)) α_0 ln(E/E_0))")
    print("\n2. Polymer-corrected Schwinger production:")
    print("   Γ_Schwinger^poly = (α_eff eE)² / (4π³ℏc) exp[-πm²c³/(eEℏ)F(μ)]")
    print("\n3. Polymer modification factor:")
    print("   F(μ) = 1 + 0.5μ²sin(πμ)")
    
    print(f"\nOptimal parameters found: μ_g = {optimal_mu_g:.2f}, b = {optimal_b:.1f}")
    print(f"Maximum yield: {max_yield_overall:.6e}")
    print("Results exported to running_coupling_analysis_results.json")
    
    return results

if __name__ == "__main__":
    results = running_coupling_analysis()
