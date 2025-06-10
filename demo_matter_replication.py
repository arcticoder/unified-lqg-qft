#!/usr/bin/env python3
"""
Matter Replication Demonstration Script

This script demonstrates the integration of the matter_polymer module with
the unified LQG-QFT framework, showcasing matter creation analysis based on
the parameter sweep results.

Usage:
    python demo_matter_replication.py [--quick] [--full-analysis] [--optimization]

Features:
- Parameter sweep validation using optimal values
- Matter creation rate analysis
- Energy feasibility assessment  
- Integration with existing warp bubble framework
- Visualization of results

Based on parameter sweep findings:
- Optimal μ = 0.20 (polymer scale)
- Optimal λ = 0.01 (matter-geometry coupling)
- Optimal α = 2.0 (curvature strength)
- Optimal R = 1.0 (bubble radius)
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import framework components
try:
    from src.matter_polymer import (
        matter_hamiltonian, interaction_hamiltonian, total_hamiltonian,
        matter_creation_rate, OptimalParameters, example_matter_creation_analysis
    )
    MATTER_POLYMER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: matter_polymer not available: {e}")
    MATTER_POLYMER_AVAILABLE = False

try:
    from src.polymer_quantization import (
        polymer_correction, ReplicatorSimulation, optimal_replicator_parameters,
        matter_creation_hamiltonian, replicator_objective_function
    )
    POLYMER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: polymer_quantization extensions not available: {e}")
    POLYMER_AVAILABLE = False

try:
    from src.warp_bubble_analysis import (
        matter_replication_analysis, replicator_feasibility_study,
        visualize_replication_analysis, run_replication_analysis_example
    )
    WARP_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: warp_bubble_analysis extensions not available: {e}")
    WARP_ANALYSIS_AVAILABLE = False

try:
    from src.warp_bubble_solver import WarpBubbleSolver
    WARP_SOLVER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: warp_bubble_solver not available: {e}")
    WARP_SOLVER_AVAILABLE = False


def demo_basic_matter_creation():
    """Demonstrate basic matter creation with optimal parameters."""
    print("\n" + "="*60)
    print("BASIC MATTER CREATION DEMONSTRATION")
    print("="*60)
    
    if not MATTER_POLYMER_AVAILABLE:
        print("Skipping: matter_polymer module not available")
        return None
    
    # Run the built-in example
    print("Running matter-polymer analysis with optimal parameters...")
    results = example_matter_creation_analysis()
    
    print(f"\nResults Summary:")
    print(f"- Total Hamiltonian: {results['H_total']:.3e}")
    print(f"- Matter creation rate: {results['creation_rate']:.3e}")  
    print(f"- Peak matter density: {np.max(results['H_matter']):.3e}")
    print(f"- Peak interaction density: {np.max(np.abs(results['H_interaction'])):.3e}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Matter Creation Analysis with Optimal Parameters", fontsize=14)
    
    r = np.linspace(0, 10, len(results['phi']))
    
    # Field configuration
    axes[0,0].plot(r, results['phi'], 'b-', label='φ field')
    axes[0,0].plot(r, results['pi'], 'r-', label='π field')
    axes[0,0].set_xlabel('Radius r')
    axes[0,0].set_ylabel('Field amplitude')
    axes[0,0].set_title('Field Configuration')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Curvature profile
    axes[0,1].plot(r, results['R'], 'g-', label='Ricci curvature')
    axes[0,1].set_xlabel('Radius r')
    axes[0,1].set_ylabel('R(r)')
    axes[0,1].set_title('Spacetime Curvature')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Energy densities
    axes[1,0].plot(r, results['H_matter'], 'b-', label='Matter density')
    axes[1,0].plot(r, results['H_interaction'], 'r-', label='Interaction density')
    axes[1,0].set_xlabel('Radius r')
    axes[1,0].set_ylabel('Energy density')
    axes[1,0].set_title('Energy Density Components')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Parameter summary
    params = OptimalParameters.BEST
    axes[1,1].text(0.1, 0.9, 'OPTIMAL PARAMETERS', fontweight='bold', transform=axes[1,1].transAxes)
    param_text = f"""
μ = {params['mu']} (polymer scale)
λ = {params['lambda']} (coupling strength)  
α = {params['alpha']} (curvature strength)
R = {params['R_bubble']} (bubble radius)

Total H = {results['H_total']:.2e}
Creation = {results['creation_rate']:.2e}
"""
    axes[1,1].text(0.1, 0.8, param_text, fontfamily='monospace', transform=axes[1,1].transAxes)
    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('matter_creation_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results


def demo_parameter_optimization():
    """Demonstrate parameter optimization for matter creation."""
    print("\n" + "="*60)
    print("PARAMETER OPTIMIZATION DEMONSTRATION")  
    print("="*60)
    
    if not POLYMER_AVAILABLE:
        print("Skipping: polymer_quantization extensions not available")
        return None
    
    # Initialize replicator simulation
    print("Initializing replicator simulation...")
    sim = ReplicatorSimulation()
    
    # Create mock field configurations
    N = 100
    r = np.linspace(0, 10, N)
    phi_init = np.exp(-(r - 5)**2 / 4)  # Gaussian initial field
    pi_init = 0.1 * np.sin(r) * np.exp(-(r - 5)**2 / 8)
    
    # Mock spacetime (negative curvature bubble)
    R_curvature = -2.0 * np.exp(-(r - 5)**2 / 3)
    f_metric = np.ones_like(r)
    
    print("Running parameter optimization...")
    opt_results = sim.parameter_optimization(phi_init, pi_init, R_curvature, f_metric)
    
    print(f"\nOptimization Results:")
    print(f"- Optimal μ: {opt_results['optimal_mu']:.3f}")
    print(f"- Optimal λ: {opt_results['optimal_lambda']:.3f}")
    print(f"- Best objective: {opt_results['best_objective']:.3f}")
    print(f"- Net creation: {opt_results['best_result']['net_creation']:.3e}")
    
    # Visualize optimization results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Parameter Optimization Results", fontsize=14)
    
    result = opt_results['best_result']
    
    # Field evolution
    axes[0,0].plot(result['phi_final'], 'b-', label='Final φ')
    axes[0,0].plot(phi_init, 'b--', alpha=0.5, label='Initial φ')
    axes[0,0].set_xlabel('Spatial index')
    axes[0,0].set_ylabel('Field amplitude')
    axes[0,0].set_title('Field Evolution')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Creation rate evolution
    time = np.linspace(0, len(result['creation_rate'])*sim.dt, len(result['creation_rate']))
    axes[0,1].plot(time, result['creation_rate'], 'g-')
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('Creation rate dN/dt')
    axes[0,1].set_title('Matter Creation Rate')
    axes[0,1].grid(True)
    
    # Hamiltonian evolution
    axes[1,0].plot(time, result['hamiltonian'], 'r-')
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('Hamiltonian')
    axes[1,0].set_title('Energy Conservation')
    axes[1,0].grid(True)
    
    # Optimization summary
    axes[1,1].text(0.1, 0.9, 'OPTIMIZATION SUMMARY', fontweight='bold', transform=axes[1,1].transAxes)
    summary_text = f"""
Optimal Parameters:
μ = {opt_results['optimal_mu']:.3f}
λ = {opt_results['optimal_lambda']:.3f}

Performance:
Net Creation = {result['net_creation']:.2e}
Final Energy = {result['energy_final']:.2e}
Objective = {opt_results['best_objective']:.3f}

Status: {'SUCCESS' if result['net_creation'] > 0 else 'OPTIMIZATION_NEEDED'}
"""
    axes[1,1].text(0.1, 0.8, summary_text, fontfamily='monospace', transform=axes[1,1].transAxes)
    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('parameter_optimization_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return opt_results


def demo_full_replication_analysis():
    """Demonstrate full replication analysis with visualization."""
    print("\n" + "="*60)
    print("FULL REPLICATION ANALYSIS DEMONSTRATION")
    print("="*60)
    
    if not WARP_ANALYSIS_AVAILABLE:
        print("Skipping: warp_bubble_analysis extensions not available")
        return None
    
    print("Running comprehensive replication analysis...")
    print("This may take a few minutes...")
    
    try:
        # Run the full analysis example
        analysis, feasibility, fig = run_replication_analysis_example()
        
        print(f"\nFull Analysis Results:")
        print(f"- Parameter combinations tested: {analysis['parameter_count']}")
        print(f"- Best creation rate: {analysis['best_creation_params']['creation_rate']:.3e}")
        print(f"- Energy feasibility: {feasibility['recommendation']}")
        print(f"- Energy ratio: {feasibility['energy_ratio']:.2f}")
        
        # Save the comprehensive figure
        fig.savefig('full_replication_analysis.png', dpi=150, bbox_inches='tight')
        
        return analysis, feasibility
        
    except Exception as e:
        print(f"Full analysis failed: {e}")
        print("This is expected if all dependencies are not available")
        return None


def demo_integration_with_warp_solver():
    """Demonstrate integration with existing warp bubble solver."""
    print("\n" + "="*60)
    print("WARP SOLVER INTEGRATION DEMONSTRATION")
    print("="*60)
    
    if not (WARP_SOLVER_AVAILABLE and MATTER_POLYMER_AVAILABLE):
        print("Skipping: Required modules not available")
        return None
    
    try:
        from src.matter_polymer import integrate_with_warp_solver, create_matter_polymer_source
        
        # Create matter-polymer enhanced source
        print("Creating polymer-enhanced energy source...")
        source = create_matter_polymer_source(mu=0.20, lam=0.01)
        
        if source is None:
            print("Could not create enhanced source, using mock")
            return None
        
        # Initialize solver
        solver = WarpBubbleSolver()
        
        # Run simulation with matter creation analysis
        print("Running warp bubble simulation with matter creation...")
        result = integrate_with_warp_solver(solver, source, radius=5.0, resolution=30)
        
        print(f"\nIntegration Results:")
        print(f"- Simulation success: {result.success}")
        print(f"- Total energy: {result.energy_total:.2e} J")
        print(f"- Stability: {result.stability:.3f}")
        
        if hasattr(result, 'matter_creation_rate'):
            print(f"- Matter creation rate: {result.matter_creation_rate:.3e}")
            print(f"- Polymer parameter: {result.polymer_parameter}")
            print(f"- Coupling strength: {result.coupling_strength}")
        
        return result
        
    except Exception as e:
        print(f"Integration failed: {e}")
        return None


def main():
    """Main demonstration script."""
    parser = argparse.ArgumentParser(description="Matter Replication Demonstration")
    parser.add_argument('--quick', action='store_true', help='Run quick demo only')
    parser.add_argument('--full-analysis', action='store_true', help='Run full parameter analysis')  
    parser.add_argument('--optimization', action='store_true', help='Run parameter optimization')
    parser.add_argument('--integration', action='store_true', help='Test warp solver integration')
    
    args = parser.parse_args()
    
    print("UNIFIED LQG-QFT MATTER REPLICATION DEMONSTRATION")
    print("=" * 60)
    print("Based on parameter sweep analysis findings:")
    print("- Optimal μ = 0.20 (polymer scale)")
    print("- Optimal λ = 0.01 (matter-geometry coupling)")
    print("- Optimal α = 2.0 (curvature strength)")
    print("- Optimal R = 1.0 (bubble radius)")
    print()
    
    results = {}
    
    # Always run basic demo
    results['basic'] = demo_basic_matter_creation()
    
    if not args.quick:
        # Run additional demos unless quick mode
        if args.optimization or not any([args.full_analysis, args.integration]):
            results['optimization'] = demo_parameter_optimization()
        
        if args.full_analysis:
            results['full_analysis'] = demo_full_replication_analysis()
            
        if args.integration:
            results['integration'] = demo_integration_with_warp_solver()
    
    print("\n" + "="*60)
    print("DEMONSTRATION SUMMARY")
    print("="*60)
    
    for demo_name, result in results.items():
        status = "SUCCESS" if result is not None else "SKIPPED/FAILED"
        print(f"- {demo_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nGenerated files:")
    output_files = [
        'matter_creation_demo.png',
        'parameter_optimization_demo.png', 
        'full_replication_analysis.png'
    ]
    
    for filename in output_files:
        if Path(filename).exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ○ {filename} (not generated)")
    
    print(f"\nNext steps:")
    print(f"1. Install JAX for GPU acceleration: pip install jax jaxlib")
    print(f"2. Integrate with CMA-ES optimization: pip install cma")
    print(f"3. Run full parameter sweeps on GPU clusters")
    print(f"4. Validate against experimental constraints")
    
    return results


if __name__ == "__main__":
    results = main()
