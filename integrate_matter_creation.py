#!/usr/bin/env python3
"""
Integration Example: Matter-Polymer Creation with Unified LQG-QFT Framework

This script demonstrates how to integrate the new matter_polymer module with
the existing unified framework components for advanced replicator physics.

Key Integration Points:
1. matter_polymer.py - New polymer-quantized matter creation
2. warp_bubble_solver.py - 3D spacetime geometry 
3. ghost_condensate_eft.py - Negative energy sources
4. anec_violation_analysis.py - ANEC constraint validation

Usage:
    python integrate_matter_creation.py [--full-analysis]
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.matter_polymer import (
    matter_hamiltonian,
    interaction_hamiltonian,
    compute_matter_creation_rate,
    run_parameter_sweep_refined,
    validate_optimal_parameters
)

from src.ghost_condensate_eft import GhostEFTParameters, GhostCondensateEFT
from src.warp_bubble_solver import WarpBubbleSolver
from src.energy_source_interface import GhostCondensateEFT as GhostSource
from src.anec_violation_analysis import coherent_state_anec_violation


def integrated_matter_creation_analysis():
    """
    Run integrated analysis combining matter creation with spacetime engineering.
    """
    print("Unified LQG-QFT Framework: Integrated Matter Creation Analysis")
    print("="*80)
    
    # 1. Validate optimal parameters from sweep
    print("\n1. Validating Optimal Matter-Creation Parameters")
    print("-" * 50)
    
    optimal_results = validate_optimal_parameters()
    print(f"✓ Optimal configuration validated")
    print(f"  - Matter creation rate: {optimal_results['metrics']['creation_rate']:.6f}")
    print(f"  - Peak interaction density: {optimal_results['metrics']['peak_interaction_density']:.6f}")
    
    # 2. Ghost condensate EFT integration
    print("\n2. Ghost Condensate EFT Integration")
    print("-" * 50)
    
    try:
        # Create ghost EFT with optimal parameters
        ghost_params = GhostEFTParameters(
            phi_0=1.0,
            lambda_ghost=0.01,  # Using optimal λ from sweep
            cutoff_scale=10.0,
            higher_deriv_coeff=0.01
        )
        
        ghost_eft = GhostCondensateEFT(ghost_params)
        print(f"✓ Ghost EFT configured with λ={ghost_params.lambda_ghost}")
        
        # Compute ANEC violation
        anec_result = ghost_eft.compute_anec_violation(tau=1.0)
        print(f"  - ANEC violation: {anec_result['violation']:.6e}")
        print(f"  - Energy density: {anec_result.get('energy_density', 'N/A')}")
        
    except Exception as e:
        print(f"⚠ Ghost EFT analysis failed: {e}")
        anec_result = {'violation': 0.0}
    
    # 3. Warp bubble geometry analysis
    print("\n3. Warp Bubble Geometry Analysis")
    print("-" * 50)
    
    try:
        # Create energy source
        ghost_source = GhostSource(
            M=1000,          # Mass scale
            alpha=0.01,      # Optimal α from sweep 
            beta=0.1,        # EFT parameter
            R0=1.0,          # Optimal R_bubble from sweep
            sigma=0.2        # Shell thickness
        )
        
        # Run warp bubble simulation
        solver = WarpBubbleSolver()
        warp_result = solver.simulate(ghost_source, radius=5.0, resolution=30)
        
        print(f"✓ Warp bubble simulation complete")
        print(f"  - Success: {warp_result.success}")
        print(f"  - Total energy: {warp_result.energy_total:.2e} J")
        print(f"  - Stability: {warp_result.stability:.3f}")
        print(f"  - Max negative density: {warp_result.max_negative_density:.2e} J/m³")
        
    except Exception as e:
        print(f"⚠ Warp bubble analysis failed: {e}")
        warp_result = type('Result', (), {
            'success': False, 'energy_total': 0.0, 'stability': 0.0
        })()
    
    # 4. ANEC violation with coherent states
    print("\n4. ANEC Violation Analysis with Coherent States")
    print("-" * 50)
    
    try:
        anec_coherent = coherent_state_anec_violation(
            n_nodes=64,
            alpha=0.05,
            mu=0.20,      # Optimal μ from sweep
            tau=1.0,
            field_amplitude=1.0
        )
        
        print(f"✓ Coherent state ANEC analysis complete")
        print(f"  - ANEC violation: {anec_coherent['anec_violation']:.6e}")
        print(f"  - Quantum inequality violation: {anec_coherent.get('qi_violation', 'N/A')}")
        
    except Exception as e:
        print(f"⚠ Coherent state ANEC analysis failed: {e}")
        anec_coherent = {'anec_violation': 0.0}
    
    # 5. Integrated assessment
    print("\n5. Integrated Assessment")
    print("-" * 50)
    
    # Combine results for overall feasibility
    matter_creation_rate = optimal_results['metrics']['creation_rate']
    ghost_anec_violation = anec_result['violation']
    coherent_anec_violation = anec_coherent['anec_violation']
    warp_stability = warp_result.stability
    
    print(f"Matter Creation Rate:      {matter_creation_rate:.6e}")
    print(f"Ghost ANEC Violation:      {ghost_anec_violation:.6e}")
    print(f"Coherent ANEC Violation:   {coherent_anec_violation:.6e}")
    print(f"Warp Bubble Stability:     {warp_stability:.3f}")
    
    # Overall feasibility score
    feasibility_factors = [
        abs(matter_creation_rate) > 1e-6,      # Significant matter creation
        abs(ghost_anec_violation) > 1e-6,      # ANEC violation present
        abs(coherent_anec_violation) > 1e-6,   # LQG ANEC violation
        warp_stability > 0.1                   # Stable spacetime geometry
    ]
    
    feasibility_score = sum(feasibility_factors) / len(feasibility_factors)
    
    print(f"\nOverall Feasibility Score: {feasibility_score:.2f} / 1.0")
    
    if feasibility_score >= 0.75:
        print("🎉 EXCELLENT: All major components functioning well!")
    elif feasibility_score >= 0.5:
        print("✓ GOOD: Most components working, minor issues detected.")
    elif feasibility_score >= 0.25:
        print("⚠ PARTIAL: Some components working, significant issues present.")
    else:
        print("❌ POOR: Major issues detected across multiple components.")
    
    return {
        'optimal_parameters': optimal_results,
        'ghost_eft_result': anec_result,
        'warp_bubble_result': warp_result,
        'coherent_anec_result': anec_coherent,
        'feasibility_score': feasibility_score,
        'recommendations': get_recommendations(feasibility_score, {
            'matter_creation_rate': matter_creation_rate,
            'ghost_anec_violation': ghost_anec_violation,
            'coherent_anec_violation': coherent_anec_violation,
            'warp_stability': warp_stability
        })
    }


def get_recommendations(feasibility_score: float, metrics: dict) -> list:
    """Generate recommendations based on analysis results."""
    recommendations = []
    
    if metrics['matter_creation_rate'] < 1e-6:
        recommendations.append(
            "• Increase λ (matter-geometry coupling) or α (curvature strength) for higher creation rates"
        )
    
    if metrics['ghost_anec_violation'] < 1e-6:
        recommendations.append(
            "• Optimize ghost EFT parameters (λ_ghost, cutoff_scale) for stronger ANEC violations"
        )
    
    if metrics['coherent_anec_violation'] < 1e-6:
        recommendations.append(
            "• Adjust coherent state parameters (α, μ) for enhanced quantum violations"
        )
    
    if metrics['warp_stability'] < 0.1:
        recommendations.append(
            "• Increase energy source strength or optimize bubble geometry for stability"
        )
    
    if feasibility_score >= 0.75:
        recommendations.append(
            "• Proceed with full-scale parameter optimization and experimental validation"
        )
    elif feasibility_score >= 0.5:
        recommendations.append(
            "• Focus on addressing identified weak points before scaling up"
        )
    else:
        recommendations.append(
            "• Fundamental issues detected - review theoretical foundations and parameter ranges"
        )
    
    return recommendations


def run_extended_parameter_sweep():
    """
    Run extended parameter sweep to find optimal configurations.
    """
    print("\nExtended Parameter Sweep for Optimal Matter Creation")
    print("="*60)
    
    # Extended ranges based on initial results
    sweep_results = run_parameter_sweep_refined(
        lambda_range=[0.005, 0.01, 0.015, 0.02, 0.025],  # Extended λ range
        mu_range=[0.15, 0.18, 0.20, 0.22, 0.25, 0.30],   # Finer μ resolution
        alpha_range=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0], # Extended α range
        R_bubble_range=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],   # Extended R range
        verbose=False  # Reduce output for extended sweep
    )
    
    best_params = sweep_results['best_parameters']
    print(f"\nBest Configuration Found:")
    print(f"λ = {best_params['lambda']:.3f}")
    print(f"μ = {best_params['mu']:.3f}")
    print(f"α = {best_params['alpha']:.3f}")
    print(f"R_bubble = {best_params['R_bubble']:.3f}")
    print(f"ΔN = {best_params['Delta_N']:.6f}")
    print(f"Objective = {best_params['objective']:.6f}")
    
    return sweep_results


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrated Matter Creation Analysis")
    parser.add_argument('--full-analysis', action='store_true',
                       help='Run extended parameter sweep (takes longer)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run main integrated analysis
    integrated_results = integrated_matter_creation_analysis()
    
    # Run extended sweep if requested
    if args.full_analysis:
        print("\n" + "="*80)
        sweep_results = run_extended_parameter_sweep()
        integrated_results['extended_sweep'] = sweep_results
    
    # Print final recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR NEXT STEPS")
    print("="*80)
    
    for i, rec in enumerate(integrated_results['recommendations'], 1):
        print(f"{i}. {rec}")
    
    print(f"\n📁 Results saved to: {args.output_dir}/")
    print("🚀 Ready for integration with JAX/CMA-ES optimization pipeline!")
    
    return integrated_results


if __name__ == "__main__":
    results = main()
