#!/usr/bin/env python3
"""
Simple 3D Replicator Test
Tests basic functionality without complex imports
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt

def simple_3d_test():
    """
    Simple test of 3D replicator concepts
    """
    print("=== Simple 3D Replicator Test ===")
    
    # Create 3D grid
    N = 32
    L = 3.0
    dx = 2 * L / N
    
    x = np.linspace(-L, L, N)
    y, z = x, x
    grid = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
    r_grid = np.linalg.norm(grid, axis=-1)
    
    print(f"Grid: {N}³ points, dx={dx:.3f}")
    
    # Define replicator metric
    def replicator_metric_3d(r, mu=0.20, alpha=0.10, R0=2.0, M=1.0):
        # LQG polymer-corrected baseline
        f_lqg = 1.0 - 2*M / np.maximum(r, 0.1)
        
        # μ² polymer corrections
        mu_correction = (mu**2 * M**2) / (6 * np.maximum(r, 0.1)**4)
        f_lqg = f_lqg + mu_correction
        
        # Gaussian enhancement
        gaussian_enhancement = alpha * np.exp(-(r/R0)**2)
        
        # Combined metric
        f_total = f_lqg + gaussian_enhancement
        return np.maximum(f_total, 0.01)  # Ensure f > 0
    
    # Compute 3D metric
    f3d = replicator_metric_3d(r_grid)
    
    print(f"Metric computed: min={np.min(f3d):.6f}, max={np.max(f3d):.6f}")
    
    # Compute discrete Ricci scalar
    def compute_ricci_3d(f3d, dx):
        # Gradients
        df_dx = np.gradient(f3d, dx, axis=0)
        df_dy = np.gradient(f3d, dx, axis=1)
        df_dz = np.gradient(f3d, dx, axis=2)
        
        # Second derivatives
        d2f_dx2 = np.gradient(df_dx, dx, axis=0)
        d2f_dy2 = np.gradient(df_dy, dx, axis=1)
        d2f_dz2 = np.gradient(df_dz, dx, axis=2)
        
        # Laplacian
        laplacian_f = d2f_dx2 + d2f_dy2 + d2f_dz2
        
        # Gradient magnitude squared
        grad_f_squared = df_dx**2 + df_dy**2 + df_dz**2
        
        # Ricci scalar
        f_safe = np.maximum(f3d, 0.01)
        R = -laplacian_f / (2 * f_safe**2) + grad_f_squared / (4 * f_safe**3)
        
        return R
    
    # Compute Ricci curvature
    R3d = compute_ricci_3d(f3d, dx)
    
    print(f"Ricci computed: min={np.min(R3d):.6f}, max={np.max(R3d):.6f}")
    
    # Initialize matter fields
    phi = 0.1 * np.exp(-(r_grid / 1.0)**2)  # Gaussian initial condition
    pi = np.zeros_like(phi)  # Start from rest
    
    print(f"Fields initialized: max φ = {np.max(np.abs(phi)):.6f}")
    
    # Evolution step function
    def evolution_step(phi, pi, f3d, R3d, params, dt):
        mu = params['mu']
        lambda_param = params['lambda']
        
        # Polymer-corrected φ̇
        phi_dot = np.sin(mu * pi) / mu
        
        # 3D Laplacian of φ
        d2phi_dx2 = np.gradient(np.gradient(phi, dx, axis=0), dx, axis=0)
        d2phi_dy2 = np.gradient(np.gradient(phi, dx, axis=1), dx, axis=1)
        d2phi_dz2 = np.gradient(np.gradient(phi, dx, axis=2), dx, axis=2)
        laplacian_phi = d2phi_dx2 + d2phi_dy2 + d2phi_dz2
        
        # Field equation: π̇ = ∇²φ - 2λ√f R φ
        interaction_term = 2 * lambda_param * np.sqrt(f3d) * R3d * phi
        pi_dot = laplacian_phi - interaction_term
        
        # Update
        phi_new = phi + dt * phi_dot
        pi_new = pi + dt * pi_dot
        
        return phi_new, pi_new
    
    # Run evolution
    params = {'mu': 0.20, 'lambda': 0.01}
    dt = 0.01
    steps = 500
    
    print(f"Running evolution: {steps} steps, dt={dt}")
    
    # Store evolution
    creation_rates = []
    
    for step in range(steps):
        phi, pi = evolution_step(phi, pi, f3d, R3d, params, dt)
        
        # Matter creation rate
        creation_rate = 2 * params['lambda'] * np.sum(R3d * phi * pi) * (dx**3)
        creation_rates.append(creation_rate)
        
        if step % 100 == 0:
            print(f"  Step {step}: rate={creation_rate:.6f}, max φ={np.max(np.abs(phi)):.6f}")
    
    # Final results
    final_creation_rate = creation_rates[-1]
    max_field = np.max(np.abs(phi))
    max_curvature = np.max(np.abs(R3d))
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Final creation rate: {final_creation_rate:.6f}")
    print(f"Max field amplitude: {max_field:.6f}")
    print(f"Max curvature: {max_curvature:.6f}")
    print(f"Grid resolution: {N}³ = {N**3:,} points")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Central slices
    mid = N // 2
    
    # Field φ - XY slice
    phi_slice = phi[:, :, mid]
    im1 = axes[0,0].imshow(phi_slice, extent=[-L, L, -L, L], cmap='RdBu', origin='lower')
    axes[0,0].set_title('Matter Field φ (z=0 slice)')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Momentum π - XY slice
    pi_slice = pi[:, :, mid]
    im2 = axes[0,1].imshow(pi_slice, extent=[-L, L, -L, L], cmap='viridis', origin='lower')
    axes[0,1].set_title('Field Momentum π (z=0 slice)')
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Metric f - XY slice
    f_slice = f3d[:, :, mid]
    im3 = axes[1,0].imshow(f_slice, extent=[-L, L, -L, L], cmap='plasma', origin='lower')
    axes[1,0].set_title('Metric Function f (z=0 slice)')
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1,0])
    
    # Creation rate evolution
    axes[1,1].plot(creation_rates)
    axes[1,1].set_title('Matter Creation Rate Evolution')
    axes[1,1].set_xlabel('Step')
    axes[1,1].set_ylabel('Creation Rate')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('simple_3d_replicator_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Visualization saved as 'simple_3d_replicator_test.png'")
    
    # Export simple blueprint
    blueprint = {
        "test_type": "simple_3d_replicator",
        "parameters": params,
        "results": {
            "final_creation_rate": float(final_creation_rate),
            "max_field_amplitude": float(max_field),
            "max_curvature": float(max_curvature),
            "grid_size": [N, N, N],
            "spatial_domain": [-L, L]
        },
        "performance": {
            "total_grid_points": N**3,
            "evolution_steps": steps,
            "time_step": dt
        },
        "status": "SUCCESSFUL"
    }
    
    with open('simple_3d_test_results.json', 'w') as f:
        import json
        json.dump(blueprint, f, indent=2)
    
    print(f"✓ Results exported to 'simple_3d_test_results.json'")
    
    return blueprint

if __name__ == "__main__":
    print("🚀 Starting Simple 3D Replicator Test...")
    
    try:
        results = simple_3d_test()
        print(f"\n✅ SUCCESS: 3D replicator test completed!")
        print(f"📊 Creation rate: {results['results']['final_creation_rate']:.6f}")
        print(f"📈 Max field: {results['results']['max_field_amplitude']:.6f}")
        print(f"🎯 Grid: {results['results']['grid_size']}")
        print(f"💾 Files: simple_3d_test_results.json, simple_3d_replicator_test.png")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
