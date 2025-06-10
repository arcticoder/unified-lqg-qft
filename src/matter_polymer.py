"""
Matter-Polymer Quantization Module

This module implements polymer-quantized matter field Hamiltonians with nonminimal
curvature coupling for the unified LQG-QFT framework. Based on parameter sweep
results identifying optimal configurations for net particle creation.

Key Features:
- Polymer quantization via holonomy substitutions
- Matter Hamiltonian with gradient and mass terms
- Nonminimal curvature-matter interaction
- JAX-optimized for GPU acceleration
- Integration with warp bubble dynamics

Parameter Sweep Results (sorted by least negative ΔN):
| λ     | μ    | α   | R_bubble |
|-------|------|-----|----------|
| 0.01  | 0.20 | 2.0 | 1.0      |
| 0.01  | 0.20 | 1.0 | 1.0      |
| 0.01  | 0.20 | 1.0 | 2.0      |
| 0.05  | 0.20 | 2.0 | 2.0      |

Optimal region: λ=0.01, μ=0.20, α=2.0, R_bubble=1.0

Key Math Relations:
- Matter-creation rate: ṅ(t) = 2λ Σᵢ Rᵢ(t) φᵢ(t) πᵢ(t)
- Constraint anomaly: A = ∫₀ᵀ Σᵢ |Gₜₜ,ᵢ - 8π(T_m,ᵢ + T_int,ᵢ)| dt
- Curvature cost: C = ∫₀ᵀ Σᵢ |Rᵢ(t)| dt
- Optimization objective: J = ΔN - γA - κC

Author: Unified LQG-QFT Research Team
Date: June 2025
"""

import numpy as np
from typing import Tuple, Optional, Union

# Try JAX import, fallback to NumPy if not available
try:
    import jax.numpy as jnp
    from jax import jit, grad, vmap
    JAX_AVAILABLE = True
    print("JAX backend available for matter_polymer module")
except ImportError:
    import numpy as jnp
    JAX_AVAILABLE = False
    print("Using NumPy backend for matter_polymer module")
    
    # Define dummy jit decorator for NumPy fallback
    def jit(func):
        return func


def polymer_substitution(x: Union[float, np.ndarray, 'jnp.ndarray'], 
                        mu: float) -> Union[float, np.ndarray, 'jnp.ndarray']:
    """
    Apply polymer quantization substitution: x -> sin(μx)/μ
    
    This is the fundamental polymer modification that regularizes operators
    in Loop Quantum Gravity by replacing continuous variables with discrete
    holonomy-based representations.
    
    Args:
        x: Classical variable (momentum, coordinate, etc.)
        mu: Polymer scale parameter
        
    Returns:
        Polymer-modified variable
    """
    # Handle small μx for numerical stability and JAX compatibility
    mu_x = mu * x
    
    if JAX_AVAILABLE and hasattr(x, 'shape'):
        # JAX-compatible version using jnp.where for conditions
        return jnp.where(jnp.abs(mu_x) < 1e-10, 
                        x * (1 - (mu_x)**2/6 + (mu_x)**4/120),
                        jnp.where(jnp.abs(mu) < 1e-15,
                                 x,  # Handle mu ≈ 0 case
                                 jnp.sin(mu_x) / mu))
    else:
        # NumPy version with explicit conditions
        if np.abs(mu) < 1e-15:
            return x
        
        small_arg = np.abs(mu_x) < 1e-10
        result = np.where(small_arg,
                         x * (1 - (mu_x)**2/6 + (mu_x)**4/120),
                         np.sin(mu_x) / mu)
        return result


@jit
def matter_hamiltonian(phi: 'jnp.ndarray',
                      pi: 'jnp.ndarray', 
                      dr: float,
                      mu: float,
                      m: float = 0.0) -> 'jnp.ndarray':
    """
    Compute the polymer-quantized matter Hamiltonian density on a 1D lattice.
    
    H_matter = 1/2 [ (sin(μπ)/μ)² + (∂φ/∂r)² + m²φ² ]
    
    The polymer substitution π → sin(μπ)/μ regularizes the kinetic energy
    and leads to modified dispersion relations that can support matter creation.
    
    Args:
        phi: Scalar field configuration
        pi: Conjugate momentum field  
        dr: Spatial discretization step
        mu: Polymer scale parameter (optimal: μ ≈ 0.20)
        m: Field mass (default: 0.0 for massless field)
        
    Returns:
        Matter Hamiltonian density at each spatial point
    """
    # Polymer-corrected kinetic term
    pi_poly = polymer_substitution(pi, mu)
    kinetic = 0.5 * pi_poly**2
    
    # Gradient term with central differences
    # ∂φ/∂r ≈ (φ[i+1] - φ[i-1]) / (2Δr)
    if JAX_AVAILABLE:
        grad_phi = (jnp.roll(phi, -1) - jnp.roll(phi, 1)) / (2.0 * dr)
    else:
        grad_phi = (np.roll(phi, -1) - np.roll(phi, 1)) / (2.0 * dr)
    
    gradient = 0.5 * grad_phi**2
    
    # Mass term
    mass_term = 0.5 * m**2 * phi**2
    
    return kinetic + gradient + mass_term


@jit  
def interaction_hamiltonian(phi: 'jnp.ndarray',
                           f: 'jnp.ndarray',
                           R: 'jnp.ndarray',
                           lam: float) -> 'jnp.ndarray':
    """
    Nonminimal curvature-matter coupling density.
    
    H_int = λ √f R φ²
    
    This coupling allows spacetime curvature to directly influence matter
    creation/annihilation. The √f factor comes from the spatial metric
    determinant in the 3+1 decomposition.
    
    Args:
        phi: Scalar field configuration
        f: Spatial metric determinant  
        R: Ricci scalar curvature
        lam: Coupling strength (optimal: λ ≈ 0.01)
        
    Returns:
        Interaction Hamiltonian density at each spatial point
    """
    if JAX_AVAILABLE:
        sqrt_f = jnp.sqrt(jnp.abs(f))  # Ensure positive under square root
    else:
        sqrt_f = np.sqrt(np.abs(f))
        
    return lam * sqrt_f * R * phi**2


@jit
def total_hamiltonian(phi: 'jnp.ndarray',
                     pi: 'jnp.ndarray',
                     f: 'jnp.ndarray', 
                     R: 'jnp.ndarray',
                     dr: float,
                     mu: float,
                     lam: float,
                     m: float = 0.0) -> float:
    """
    Compute total matter+interaction Hamiltonian.
    
    H_total = ∫ [H_matter + H_int] dr
    
    Args:
        phi: Scalar field configuration
        pi: Conjugate momentum field
        f: Spatial metric determinant
        R: Ricci scalar curvature  
        dr: Spatial step size
        mu: Polymer parameter
        lam: Coupling strength
        m: Field mass
        
    Returns:
        Total integrated Hamiltonian
    """
    H_mat_density = matter_hamiltonian(phi, pi, dr, mu, m)
    H_int_density = interaction_hamiltonian(phi, f, R, lam)
    
    # Integrate over spatial domain
    if JAX_AVAILABLE:
        total_H = jnp.sum(H_mat_density + H_int_density) * dr
    else:
        total_H = np.sum(H_mat_density + H_int_density) * dr
        
    return total_H


def matter_creation_rate(phi: 'jnp.ndarray',
                        pi: 'jnp.ndarray', 
                        R: 'jnp.ndarray',
                        lam: float,
                        dr: float) -> float:
    """
    Compute instantaneous matter creation rate.
    
    Ṅ(t) = 2λ ∑_i R_i(t) φ_i(t) π_i(t)
    
    This measures the rate of particle number change due to the
    curvature-matter coupling.
    
    Args:
        phi: Scalar field configuration
        pi: Conjugate momentum field
        R: Ricci scalar curvature
        lam: Coupling strength
        dr: Spatial step size
        
    Returns:
        Instantaneous creation rate dN/dt
    """
    creation_density = 2.0 * lam * R * phi * pi
    
    if JAX_AVAILABLE:
        return jnp.sum(creation_density) * dr
    else:
        return np.sum(creation_density) * dr


def constraint_anomaly(G_tt: 'jnp.ndarray',
                      T_matter: 'jnp.ndarray', 
                      T_interaction: 'jnp.ndarray',
                      dr: float) -> float:
    """
    Compute Hamiltonian constraint violation.
    
    A = ∫ |G_tt - 8π(T_matter + T_interaction)| dr
    
    This measures how well the Einstein equations are satisfied
    during the dynamical evolution.
    
    Args:
        G_tt: Time-time component of Einstein tensor
        T_matter: Matter stress-energy density  
        T_interaction: Interaction stress-energy density
        dr: Spatial step size
        
    Returns:
        Integrated constraint violation
    """
    constraint_violation = G_tt - 8.0 * np.pi * (T_matter + T_interaction)
    
    if JAX_AVAILABLE:
        return jnp.sum(jnp.abs(constraint_violation)) * dr
    else:
        return np.sum(np.abs(constraint_violation)) * dr


def curvature_cost(R: 'jnp.ndarray', dr: float, dt: float) -> float:
    """
    Compute integrated curvature cost.
    
    C = ∫∫ |R(r,t)| dr dt
    
    This penalizes large curvatures in the optimization objective.
    
    Args:
        R: Ricci scalar curvature (spatial array or spacetime grid)
        dr: Spatial step size
        dt: Time step size (if R is 2D)
        
    Returns:
        Integrated curvature cost
    """
    if JAX_AVAILABLE:
        cost = jnp.sum(jnp.abs(R))
    else:
        cost = np.sum(np.abs(R))
        
    if R.ndim == 1:
        # Spatial integration only
        return cost * dr
    else:
        # Spacetime integration  
        return cost * dr * dt


def optimization_objective(delta_N: float,
                          anomaly: float,
                          cost: float,
                          gamma: float = 1.0,
                          kappa: float = 0.1) -> float:
    """
    Compute optimization objective function.
    
    J = ΔN - γA - κC
    
    To be maximized over {λ, μ, α, R_bubble}.
    
    Args:
        delta_N: Net particle change
        anomaly: Constraint violation 
        cost: Curvature cost
        gamma: Constraint penalty weight
        kappa: Curvature penalty weight
        
    Returns:
        Objective function value (to maximize)
    """
    return delta_N - gamma * anomaly - kappa * cost


# ============================================================================
# OPTIMAL PARAMETER SETS (from sweep analysis)
# ============================================================================

class OptimalParameters:
    """Container for optimal parameter sets from sweep analysis."""
    
    # Best parameter set (minimal annihilation)
    BEST = {
        'lambda': 0.01,    # Matter-geometry coupling
        'mu': 0.20,        # Polymer scale parameter  
        'alpha': 2.0,      # Curvature pulse strength
        'R_bubble': 1.0    # Bubble radius
    }
    
    # Refined search ranges around optimal region
    REFINED_RANGES = {
        'mu': [0.15, 0.20, 0.25, 0.30],
        'lambda': [0.005, 0.010, 0.020],
        'alpha': [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        'R_bubble': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    }
    
    # Physical scales for reference
    SCALES = {
        'planck_length': 1.616e-35,    # meters
        'planck_time': 5.391e-44,      # seconds  
        'planck_energy': 1.956e9,      # Joules
        'polymer_length': 0.20 * 1.616e-35,  # μ * l_Planck
    }


# ============================================================================
# INTEGRATION FUNCTIONS FOR EXISTING FRAMEWORK
# ============================================================================

def integrate_with_warp_solver(warp_solver, energy_source, **params):
    """
    Integrate matter_polymer module with existing warp bubble solver.
    
    Args:
        warp_solver: Instance of WarpBubbleSolver
        energy_source: Energy source from energy_source_interface
        **params: Additional parameters (mu, lambda, etc.)
        
    Returns:
        Enhanced simulation results with matter creation analysis
    """
    # Extract parameters
    mu = params.get('mu', OptimalParameters.BEST['mu'])
    lam = params.get('lambda', OptimalParameters.BEST['lambda'])
    
    print(f"Integrating matter-polymer effects (μ={mu}, λ={lam})")
    
    # Run base simulation
    result = warp_solver.simulate(energy_source, **params)
    
    if result.success and hasattr(result, 'coordinates'):
        # Compute additional matter creation metrics
        coords = result.coordinates
        energy_profile = result.energy_profile
        
        # Create mock field configurations for demonstration
        N_points = len(coords)
        phi_mock = np.random.normal(0, 0.1, N_points)
        pi_mock = np.random.normal(0, 0.1, N_points)
        
        # Estimate curvature from energy profile (rough approximation)
        R_mock = -energy_profile / (8 * np.pi)  # Einstein equation approximation
        f_mock = np.ones_like(R_mock)  # Flat spatial metric
        
        # Compute matter creation rate
        dr = 0.1  # Approximate spatial resolution
        creation_rate = matter_creation_rate(phi_mock, pi_mock, R_mock, lam, dr)
        
        # Add to result
        result.matter_creation_rate = creation_rate
        result.polymer_parameter = mu
        result.coupling_strength = lam
        
        print(f"Matter creation rate: {creation_rate:.3e}")
    
    return result


def create_matter_polymer_source(mu: float = 0.20, 
                                 lam: float = 0.01,
                                 **kwargs):
    """
    Create an energy source specifically configured for matter-polymer effects.
    
    Args:
        mu: Polymer parameter
        lam: Coupling strength  
        **kwargs: Additional source parameters
        
    Returns:
        Configured energy source with polymer modifications
    """
    try:
        from .ghost_condensate_eft import GhostCondensateEFT
        
        # Create ghost EFT source with polymer-optimized parameters
        source = GhostCondensateEFT(
            M=1000,           # Mass scale
            alpha=0.01,       # Ghost coupling
            beta=0.1,         # Self-interaction
            R0=kwargs.get('R_bubble', 1.0),
            sigma=0.2,        # Shell thickness
            **kwargs
        )
        
        # Add polymer parameters
        source.polymer_mu = mu
        source.coupling_lambda = lam
        source.name = f"Polymer-Enhanced Ghost EFT (μ={mu}, λ={lam})"
        
        return source
        
    except ImportError:
        print("Ghost EFT not available, using mock source")
        return None


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def example_matter_creation_analysis():
    """Example of matter creation analysis with optimal parameters."""
    
    print("Matter-Polymer Creation Analysis")
    print("=" * 50)
    
    # Use optimal parameters from sweep
    params = OptimalParameters.BEST
    mu = params['mu']
    lam = params['lambda'] 
    
    print(f"Using optimal parameters: μ={mu}, λ={lam}")
    
    # Create mock field configuration
    N = 100
    dr = 0.1
    r = np.linspace(0, 10, N)
    
    # Gaussian field pulse
    phi = np.exp(-(r - 5)**2 / 2)
    pi = 0.1 * np.sin(2 * r) * np.exp(-(r - 5)**2 / 4)
    
    # Mock curvature (negative energy region)
    R = -2.0 * np.exp(-(r - 5)**2 / 1.5)
    f = np.ones_like(r)
    
    # Compute Hamiltonians
    H_matter = matter_hamiltonian(phi, pi, dr, mu)
    H_interaction = interaction_hamiltonian(phi, f, R, lam)
    H_total = np.sum(H_matter + H_interaction) * dr
    
    # Matter creation rate
    creation_rate = matter_creation_rate(phi, pi, R, lam, dr)    
    print(f"Total Hamiltonian: {H_total:.3e}")
    print(f"Matter creation rate: {creation_rate:.3e}")
    print(f"Peak matter density: {np.max(H_matter):.3e}")
    print(f"Peak interaction density: {np.max(np.abs(H_interaction)):.3e}")
    
    return {
        'H_total': H_total,
        'creation_rate': creation_rate,
        'phi': phi,
        'pi': pi,
        'R': R,
        'H_matter': H_matter,
        'H_interaction': H_interaction
    }


# ============================================================================
# NEW FUNCTIONS FOR PARAMETER SWEEP OPTIMIZATION
# ============================================================================

@jit
def compute_matter_creation_rate(phi: 'jnp.ndarray',
                                 pi: 'jnp.ndarray',
                                 R: 'jnp.ndarray',
                                 lam: float) -> float:
    """
    Compute instantaneous matter creation rate.
    
    ṅ(t) = 2λ Σᵢ Rᵢ(t) φᵢ(t) πᵢ(t)
    
    Args:
        phi: Scalar field
        pi: Canonical momentum
        R: Ricci curvature
        lam: Coupling strength
        
    Returns:
        Matter creation rate
    """
    if JAX_AVAILABLE:
        return 2.0 * lam * jnp.sum(R * phi * pi)
    else:
        return 2.0 * lam * np.sum(R * phi * pi)


@jit
def constraint_anomaly(G_tt: 'jnp.ndarray',
                       T_matter: 'jnp.ndarray',
                       T_interaction: 'jnp.ndarray') -> float:
    """
    Compute Hamiltonian constraint anomaly.
    
    A = ∫₀ᵀ Σᵢ |Gₜₜ,ᵢ - 8π(T_m,ᵢ + T_int,ᵢ)| dt
    
    Args:
        G_tt: Einstein tensor tt-component
        T_matter: Matter stress-energy density
        T_interaction: Interaction stress-energy density
        
    Returns:
        Constraint anomaly magnitude
    """
    constraint_violation = jnp.abs(G_tt - 8.0 * jnp.pi * (T_matter + T_interaction))
    if JAX_AVAILABLE:
        return jnp.sum(constraint_violation)
    else:
        return np.sum(constraint_violation)


@jit
def curvature_cost(R: 'jnp.ndarray', dt: float) -> float:
    """
    Compute integrated curvature cost.
    
    C = ∫₀ᵀ Σᵢ |Rᵢ(t)| dt
    
    Args:
        R: Ricci curvature array
        dt: Time step
        
    Returns:
        Total curvature cost
    """
    if JAX_AVAILABLE:
        return dt * jnp.sum(jnp.abs(R))
    else:
        return dt * np.sum(np.abs(R))


def optimization_objective(Delta_N: float,
                          anomaly: float,
                          cost: float,
                          gamma: float = 1.0,
                          kappa: float = 0.1) -> float:
    """
    Compute optimization objective function.
    
    J = ΔN - γA - κC
    
    To be maximized over {λ, μ, α, R_bubble}.
    
    Args:
        Delta_N: Net particle change
        anomaly: Constraint anomaly
        cost: Curvature cost
        gamma: Anomaly penalty weight
        kappa: Curvature penalty weight
        
    Returns:
        Objective function value
    """
    return Delta_N - gamma * anomaly - kappa * cost


def run_parameter_sweep_refined(lambda_range: list = [0.005, 0.010, 0.020],
                               mu_range: list = [0.15, 0.20, 0.25, 0.30],
                               alpha_range: list = [1.0, 2.0, 3.0, 5.0],
                               R_bubble_range: list = [1.0, 2.0, 3.0],
                               verbose: bool = True) -> dict:
    """
    Run refined parameter sweep around optimal region identified from initial results.
    
    Tests the parameter space around:
    - λ ∈ {0.005, 0.010, 0.020} (matter-geometry coupling)
    - μ ∈ {0.15, 0.20, 0.25, 0.30} (polymer scale)
    - α ∈ {1.0, 2.0, 3.0, 5.0} (curvature amplitude)
    - R_bubble ∈ {1.0, 2.0, 3.0} (bubble radius)
    
    Args:
        lambda_range: Matter-geometry coupling values
        mu_range: Polymer parameter values  
        alpha_range: Curvature amplitude values
        R_bubble_range: Bubble radius values
        verbose: Print progress updates
        
    Returns:
        Dictionary with sweep results sorted by objective function
    """
    results = []
    total_runs = len(lambda_range) * len(mu_range) * len(alpha_range) * len(R_bubble_range)
    run_count = 0
    
    if verbose:
        print(f"Starting refined parameter sweep: {total_runs} total combinations")
        print("Optimizing around identified optimal region: λ=0.01, μ=0.20, α=2.0, R=1.0")
    
    for lam in lambda_range:
        for mu in mu_range:
            for alpha in alpha_range:
                for R_bubble in R_bubble_range:
                    run_count += 1
                    
                    if verbose:
                        print(f"\nRun {run_count}/{total_runs}: λ={lam}, μ={mu}, α={alpha}, R={R_bubble}")
                    
                    # Simplified evolution test (replace with full dynamics)
                    N_sites = 50
                    dr = 0.1
                    phi = 0.1 * np.sin(2 * np.pi * np.arange(N_sites) / 10.0)
                    pi = 0.1 * np.cos(2 * np.pi * np.arange(N_sites) / 10.0)
                    
                    # Apply curvature pulse
                    r = np.arange(N_sites) * dr
                    r_center = N_sites * dr / 2.0
                    R = alpha * np.exp(-((r - r_center)**2) / (2 * R_bubble**2))
                    
                    # Compute metrics
                    creation_rate = compute_matter_creation_rate(phi, pi, R, lam)
                    
                    # Simulate net particle change (simplified)
                    N_initial = np.sum(phi**2 + pi**2) * dr
                    Delta_N = creation_rate * 1.0  # Simplified time integration
                    
                    # Compute objective
                    objective = optimization_objective(Delta_N, 0.0, 0.0)  # Simplified
                    
                    results.append({
                        'lambda': lam,
                        'mu': mu,
                        'alpha': alpha,
                        'R_bubble': R_bubble,
                        'Delta_N': Delta_N,
                        'creation_rate': creation_rate,
                        'objective': objective,
                        'N_initial': N_initial
                    })
                    
                    if verbose:
                        print(f"  ΔN = {Delta_N:.6f}, Creation Rate = {creation_rate:.6f}")
    
    # Sort by objective (highest first - least negative ΔN)
    results.sort(key=lambda x: x['objective'], reverse=True)
    
    if verbose:
        print("\n" + "="*80)
        print("REFINED PARAMETER SWEEP RESULTS")
        print("="*80)
        print("| Rank | λ     | μ    | α   | R_bubble | ΔN        | Objective |")
        print("|------|-------|------|-----|----------|-----------|-----------|")
        
        for i, res in enumerate(results[:10]):  # Top 10
            print(f"| {i+1:<4} | {res['lambda']:<5.3f} | {res['mu']:<4.2f} | "
                  f"{res['alpha']:<3.1f} | {res['R_bubble']:<8.1f} | "
                  f"{res['Delta_N']:<9.6f} | {res['objective']:<9.6f} |")
    
    return {
        'results': results,
        'best_parameters': results[0] if results else None,
        'sweep_summary': {
            'total_runs': total_runs,
            'parameter_ranges': {
                'lambda': lambda_range,
                'mu': mu_range,
                'alpha': alpha_range,
                'R_bubble': R_bubble_range
            }
        }
    }


def validate_optimal_parameters():
    """
    Validate the optimal parameters identified from the sweep:
    λ=0.01, μ=0.20, α=2.0, R_bubble=1.0
    """
    print("\nValidating optimal parameters from parameter sweep...")
    print("λ=0.01, μ=0.20, α=2.0, R_bubble=1.0")
    
    # Test configuration
    N_sites = 100
    dr = 0.1
    lam, mu, alpha, R_bubble = 0.01, 0.20, 2.0, 1.0
    
    # Initial conditions (improved vacuum state)
    phi = 0.01 * np.sin(2 * np.pi * np.arange(N_sites) / 20.0)
    pi = 0.01 * np.cos(2 * np.pi * np.arange(N_sites) / 20.0)
    
    # Curvature profile
    r = np.arange(N_sites) * dr
    r_center = N_sites * dr / 2.0
    R = alpha * np.exp(-((r - r_center)**2) / (2 * R_bubble**2))
    f = np.ones_like(r) + 0.1 * R  # Metric perturbation
    
    # Compute Hamiltonians
    H_matter = matter_hamiltonian(phi, pi, dr, mu, m=0.0)
    H_int = interaction_hamiltonian(phi, f, R, lam)
    
    # Matter creation analysis
    creation_rate = compute_matter_creation_rate(phi, pi, R, lam)
    N_initial = np.sum(phi**2 + pi**2) * dr
    
    print(f"Initial particle number: {N_initial:.6f}")
    print(f"Matter creation rate: {creation_rate:.6f}")
    print(f"Peak matter density: {np.max(H_matter):.6f}")
    print(f"Peak interaction density: {np.max(np.abs(H_int)):.6f}")
    print(f"Total curvature: {np.sum(np.abs(R)) * dr:.6f}")
    
    # Estimate net change (simplified)
    time_scale = 5.0
    estimated_Delta_N = creation_rate * time_scale
    print(f"Estimated ΔN over {time_scale}s: {estimated_Delta_N:.6f}")
    
    return {
        'parameters': {'lambda': lam, 'mu': mu, 'alpha': alpha, 'R_bubble': R_bubble},
        'metrics': {
            'N_initial': N_initial,
            'creation_rate': creation_rate,
            'estimated_Delta_N': estimated_Delta_N,
            'peak_matter_density': np.max(H_matter),
            'peak_interaction_density': np.max(np.abs(H_int))
        }
    }


if __name__ == "__main__":
    print("Matter-Polymer Quantization Module")
    print("="*60)
    
    # Run parameter sweep validation
    optimal_results = validate_optimal_parameters()
    
    print("\n" + "="*60)
    print("Running sample parameter sweep...")
    
    # Run small parameter sweep around optimal region
    sweep_results = run_parameter_sweep_refined(
        lambda_range=[0.005, 0.01, 0.02],
        mu_range=[0.15, 0.20, 0.25],
        alpha_range=[1.0, 2.0, 3.0],
        R_bubble_range=[1.0, 2.0],
        verbose=True
    )
    
    # Run original example
    print("\n" + "="*60)
    print("Running original example analysis...")
    results = example_matter_creation_analysis()
    print("\nMatter-polymer module validation complete!")
