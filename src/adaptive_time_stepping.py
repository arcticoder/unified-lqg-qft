#!/usr/bin/env python3
"""
Adaptive Time-Stepping Module for 3D Replicator Framework
Implements adaptive time step control based on field gradients and stability analysis
"""

import numpy as np
from typing import Tuple, Dict, Optional
import warnings

class AdaptiveTimeStepper:
    """
    Adaptive time-stepping controller for replicator field evolution
    Automatically adjusts time steps based on field dynamics and stability
    """
    
    def __init__(self, 
                 dt_min: float = 1e-6,
                 dt_max: float = 0.1,
                 dt_initial: float = 0.01,
                 safety_factor: float = 0.8,
                 tolerance: float = 1e-4):
        """
        Initialize adaptive time stepper
        
        Args:
            dt_min: Minimum allowed time step
            dt_max: Maximum allowed time step  
            dt_initial: Initial time step
            safety_factor: Safety factor for step size adjustment
            tolerance: Error tolerance for adaptive control
        """
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt = dt_initial
        self.safety_factor = safety_factor
        self.tolerance = tolerance
        
        # History tracking
        self.dt_history = [dt_initial]
        self.error_history = []
        self.rejected_steps = 0
        self.accepted_steps = 0
        
        print(f"✓ Adaptive time stepper initialized: dt={dt_initial:.6f}")
        
    def estimate_local_error(self, 
                           phi_full: np.ndarray, 
                           pi_full: np.ndarray,
                           phi_half1: np.ndarray,
                           pi_half1: np.ndarray,
                           phi_half2: np.ndarray, 
                           pi_half2: np.ndarray) -> float:
        """
        Estimate local truncation error using Richardson extrapolation
        
        Args:
            phi_full, pi_full: Result from full time step
            phi_half1, pi_half1: Result from first half step
            phi_half2, pi_half2: Result from second half step
            
        Returns:
            Estimated local error
        """
        # Richardson extrapolation: error ≈ |y_full - y_half|
        phi_error = np.linalg.norm(phi_full - phi_half2)
        pi_error = np.linalg.norm(pi_full - pi_half2)
        
        # Combined relative error
        phi_norm = np.linalg.norm(phi_full) + 1e-12
        pi_norm = np.linalg.norm(pi_full) + 1e-12
        
        relative_error = (phi_error / phi_norm + pi_error / pi_norm) / 2
        
        return relative_error
        
    def compute_stability_metrics(self, 
                                phi: np.ndarray, 
                                pi: np.ndarray,
                                dx: float) -> Dict[str, float]:
        """
        Compute stability metrics for adaptive control
        
        Args:
            phi, pi: Current field states
            dx: Spatial grid spacing
            
        Returns:
            Dictionary of stability metrics
        """
        # Field gradient magnitudes
        if phi.ndim == 3:
            # 3D gradients
            grad_phi_x = np.gradient(phi, dx, axis=0)
            grad_phi_y = np.gradient(phi, dx, axis=1)
            grad_phi_z = np.gradient(phi, dx, axis=2)
            grad_phi_mag = np.sqrt(grad_phi_x**2 + grad_phi_y**2 + grad_phi_z**2)
            
            grad_pi_x = np.gradient(pi, dx, axis=0)
            grad_pi_y = np.gradient(pi, dx, axis=1) 
            grad_pi_z = np.gradient(pi, dx, axis=2)
            grad_pi_mag = np.sqrt(grad_pi_x**2 + grad_pi_y**2 + grad_pi_z**2)
        else:
            # 1D gradients
            grad_phi_mag = np.abs(np.gradient(phi, dx))
            grad_pi_mag = np.abs(np.gradient(pi, dx))
        
        # Maximum gradients
        max_grad_phi = np.max(grad_phi_mag)
        max_grad_pi = np.max(grad_pi_mag)
        
        # Field magnitudes
        max_phi = np.max(np.abs(phi))
        max_pi = np.max(np.abs(pi))
        
        # Courant-Friedrichs-Lewy (CFL) condition estimate
        # dt < dx / (max wave speed)
        wave_speed = np.sqrt(max_grad_phi**2 + max_grad_pi**2) + 1e-12
        cfl_dt = dx / wave_speed
        
        return {
            'max_grad_phi': max_grad_phi,
            'max_grad_pi': max_grad_pi,
            'max_phi': max_phi,
            'max_pi': max_pi,
            'cfl_dt': cfl_dt,
            'wave_speed': wave_speed
        }
        
    def suggest_time_step(self, 
                         phi: np.ndarray,
                         pi: np.ndarray, 
                         dx: float,
                         error: Optional[float] = None) -> float:
        """
        Suggest next time step based on stability and error analysis
        
        Args:
            phi, pi: Current field states
            dx: Spatial grid spacing
            error: Optional local error estimate
            
        Returns:
            Suggested time step
        """
        # Compute stability metrics
        metrics = self.compute_stability_metrics(phi, pi, dx)
        
        # CFL-based time step
        dt_cfl = self.safety_factor * metrics['cfl_dt']
        
        # Error-based time step adjustment
        if error is not None:
            if error > self.tolerance:
                # Reduce step size
                dt_error = self.dt * (self.tolerance / error)**0.2
            else:
                # Can increase step size
                dt_error = self.dt * (self.tolerance / error)**0.1
        else:
            dt_error = self.dt
            
        # Gradient-based time step
        max_derivative = max(metrics['max_grad_phi'], metrics['max_grad_pi'])
        if max_derivative > 0:
            dt_gradient = 0.1 / max_derivative  # Conservative estimate
        else:
            dt_gradient = self.dt_max
            
        # Take most conservative estimate
        dt_new = min(dt_cfl, dt_error, dt_gradient)
        
        # Enforce bounds
        dt_new = np.clip(dt_new, self.dt_min, self.dt_max)
        
        return dt_new
        
    def adaptive_step(self, 
                     evolution_func, 
                     phi: np.ndarray,
                     pi: np.ndarray,
                     *args, **kwargs) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        """
        Perform adaptive time step with error control
        
        Args:
            evolution_func: Function to perform evolution step
            phi, pi: Current field states
            *args, **kwargs: Additional arguments for evolution_func
            
        Returns:
            Tuple of (phi_new, pi_new, accept_step, info)
        """
        dt_current = self.dt
        
        # Full step
        phi_full, pi_full = evolution_func(phi, pi, dt_current, *args, **kwargs)
        
        # Two half steps for error estimation
        phi_half1, pi_half1 = evolution_func(phi, pi, dt_current/2, *args, **kwargs)
        phi_half2, pi_half2 = evolution_func(phi_half1, pi_half1, dt_current/2, *args, **kwargs)
        
        # Estimate error
        error = self.estimate_local_error(phi_full, pi_full, 
                                        phi_half1, pi_half1,
                                        phi_half2, pi_half2)
        
        # Check if step should be accepted
        accept_step = error <= self.tolerance
        
        if accept_step:
            self.accepted_steps += 1
            phi_new, pi_new = phi_half2, pi_half2  # Use more accurate half-step result
            
            # Suggest next time step
            dx = kwargs.get('dx', 0.1)  # Default spatial spacing
            dt_next = self.suggest_time_step(phi_new, pi_new, dx, error)
            self.dt = dt_next
            
        else:
            self.rejected_steps += 1
            phi_new, pi_new = phi, pi  # Reject step
            
            # Reduce time step
            self.dt = max(self.dt * 0.5, self.dt_min)
            
        # Update history
        self.dt_history.append(self.dt)
        self.error_history.append(error)
        
        # Return info
        info = {
            'error': error,
            'dt_used': dt_current,
            'dt_next': self.dt,
            'accepted': accept_step,
            'total_accepted': self.accepted_steps,
            'total_rejected': self.rejected_steps
        }
        
        return phi_new, pi_new, accept_step, info

    def integrate_with_3d_replicator(self, replicator, params: Dict, total_time: float = 1.0):
        """
        Integrate 3D replicator with adaptive time stepping
        
        Args:
            replicator: JAXReplicator3D instance
            params: Replicator parameters
            total_time: Total integration time
            
        Returns:
            Integration results with adaptive time history
        """
        print(f"Starting adaptive integration: T={total_time}, tol={self.tolerance}")
        
        # Initialize fields
        phi, pi = replicator.initialize_gaussian_field()
        
        # Compute static metric/curvature
        f3d = replicator.metric_3d(params)
        R3d = replicator.ricci_3d(f3d)
        
        # Storage
        time_current = 0.0
        history = {
            'times': [0.0],
            'dt_values': [self.dt],
            'phi_snapshots': [phi],
            'pi_snapshots': [pi],
            'objectives': [],
            'creation_rates': [],
            'stability_metrics': []
        }
        
        step_count = 0
        
        while time_current < total_time:
            # Remaining time
            time_remaining = total_time - time_current
            dt_max_remaining = min(self.dt_max, time_remaining)
            
            # Adaptive step with error control
            phi_new, pi_new, accepted, info = self.adaptive_step(
                replicator.evolution_step, phi, pi, f3d, R3d, params
            )
            
            if accepted:
                # Update state
                phi, pi = phi_new, pi_new
                time_current += self.dt
                self.accepted_steps += 1
                
                # Compute metrics
                objective = replicator._compute_objective(phi, pi, f3d, R3d, params)
                creation_rate = 2 * params['lambda'] * np.sum(R3d * phi * pi) * (replicator.dx**3)
                
                # Store history
                history['times'].append(time_current)
                history['dt_values'].append(self.dt)
                history['phi_snapshots'].append(phi)
                history['pi_snapshots'].append(pi)
                history['objectives'].append(float(objective))
                history['creation_rates'].append(float(creation_rate))
                history['stability_metrics'].append(info['stability_metrics'])
                
                # Progress reporting
                if step_count % 100 == 0:
                    print(f"  T={time_current:.3f}, dt={self.dt:.6f}, obj={objective:.6f}, rate={creation_rate:.6f}")
                    
            else:
                self.rejected_steps += 1
                
            step_count += 1
            
            # Safety check
            if step_count > 1000000:
                warnings.warn("Maximum step count exceeded")
                break
                
        print(f"✓ Adaptive integration complete: {self.accepted_steps} accepted, {self.rejected_steps} rejected")
        
        return {
            'phi_final': phi,
            'pi_final': pi, 
            'f3d': f3d,
            'R3d': R3d,
            'history': history,
            'final_time': time_current,
            'total_steps': step_count,
            'acceptance_rate': self.accepted_steps / (self.accepted_steps + self.rejected_steps),
            'average_dt': np.mean(history['dt_values']),
            'params': params
        }
        
class AdaptiveReplicatorSimulator:
    """
    Replicator simulator with adaptive time stepping
    """
    
    def __init__(self, base_simulator, adaptive_stepper: AdaptiveTimeStepper):
        """
        Initialize adaptive replicator simulator
        
        Args:
            base_simulator: Base replicator simulation class
            adaptive_stepper: AdaptiveTimeStepper instance
        """
        self.base_sim = base_simulator
        self.stepper = adaptive_stepper
        
    def evolution_step_wrapper(self, phi, pi, dt, f, R, params, dx):
        """
        Wrapper for base evolution step to work with adaptive stepper
        """
        # Convert to base simulator format if needed
        if hasattr(self.base_sim, 'evolution_step'):
            return self.base_sim.evolution_step(phi, pi, f, R, params, dx, dt)
        else:
            # Fallback simple evolution
            return self.simple_evolution_step(phi, pi, dt, f, R, params, dx)
            
    def simple_evolution_step(self, phi, pi, dt, f, R, params, dx):
        """
        Simple evolution step for demonstration
        """
        # Polymer-corrected φ̇
        phi_dot = np.sin(params['mu'] * pi) / params['mu']
        
        # Field equation π̇ 
        if phi.ndim == 1:
            # 1D case
            d2phi_dr2 = np.gradient(np.gradient(phi, dx), dx)
        else:
            # 3D case - simplified Laplacian
            d2phi_dx2 = np.gradient(np.gradient(phi, dx, axis=0), dx, axis=0)
            d2phi_dy2 = np.gradient(np.gradient(phi, dx, axis=1), dx, axis=1)
            d2phi_dz2 = np.gradient(np.gradient(phi, dx, axis=2), dx, axis=2)
            d2phi_dr2 = d2phi_dx2 + d2phi_dy2 + d2phi_dz2
            
        mass_term = params.get('mass', 0.0) * phi
        interaction_term = 2 * params['lambda'] * np.sqrt(np.abs(f)) * R * phi
        pi_dot = d2phi_dr2 - mass_term - interaction_term
        
        # Update
        phi_new = phi + dt * phi_dot
        pi_new = pi + dt * pi_dot
        
        return phi_new, pi_new
        
    def simulate_adaptive(self, 
                         phi_init: np.ndarray,
                         pi_init: np.ndarray,
                         f: np.ndarray,
                         R: np.ndarray,
                         params: Dict,
                         dx: float,
                         t_final: float = 5.0,
                         max_steps: int = 10000) -> Dict:
        """
        Run adaptive simulation to final time
        
        Args:
            phi_init, pi_init: Initial field states
            f, R: Metric and curvature
            params: Replicator parameters
            dx: Spatial grid spacing
            t_final: Final simulation time
            max_steps: Maximum number of steps
            
        Returns:
            Simulation results with adaptive time stepping data
        """
        print(f"Starting adaptive simulation to t={t_final}")
        
        phi, pi = phi_init.copy(), pi_init.copy()
        t = 0.0
        step = 0
        
        # Results storage
        time_points = [0.0]
        objectives = []
        dt_evolution = [self.stepper.dt]
        
        while t < t_final and step < max_steps:
            # Adaptive evolution step
            phi_new, pi_new, accepted, info = self.stepper.adaptive_step(
                self.evolution_step_wrapper, phi, pi,
                f=f, R=R, params=params, dx=dx
            )
            
            if accepted:
                phi, pi = phi_new, pi_new
                t += info['dt_used']
                time_points.append(t)
                dt_evolution.append(info['dt_next'])
                
                # Compute objective periodically
                if step % 50 == 0:
                    creation_rate = 2 * params['lambda'] * np.sum(R * phi * pi) * dx**(phi.ndim)
                    objectives.append(creation_rate)
                    
                    if step % 500 == 0:
                        print(f"  t={t:.3f}, dt={info['dt_next']:.6f}, "
                              f"error={info['error']:.2e}, creation={creation_rate:.6f}")
            
            step += 1
            
            # Safety check
            if step >= max_steps:
                warnings.warn(f"Maximum steps ({max_steps}) reached before t_final")
                break
                
        # Final analysis
        final_creation = 2 * params['lambda'] * np.sum(R * phi * pi) * dx**(phi.ndim)
        
        efficiency = self.stepper.accepted_steps / (self.stepper.accepted_steps + self.stepper.rejected_steps)
        
        results = {
            'phi_final': phi,
            'pi_final': pi, 
            'final_time': t,
            'total_steps': step,
            'final_creation_rate': final_creation,
            'time_points': time_points,
            'dt_evolution': dt_evolution,
            'objectives': objectives,
            'step_efficiency': efficiency,
            'accepted_steps': self.stepper.accepted_steps,
            'rejected_steps': self.stepper.rejected_steps,
            'error_history': self.stepper.error_history[-len(time_points):],
            'parameters': params
        }
        
        print(f"✓ Adaptive simulation complete: {step} steps, efficiency={efficiency:.1%}")
        
        return results

def demo_adaptive_time_stepping():
    """
    Demonstration of adaptive time stepping for replicator simulation
    """
    print("=== ADAPTIVE TIME STEPPING DEMONSTRATION ===")
    
    # Initialize adaptive stepper
    stepper = AdaptiveTimeStepper(
        dt_min=1e-5,
        dt_max=0.05,
        dt_initial=0.01,
        tolerance=1e-4
    )
    
    # Create mock simulator (normally would use actual replicator)
    class MockSimulator:
        def evolution_step(self, phi, pi, f, R, params, dx, dt):
            # Simple mock evolution for demonstration
            phi_dot = np.sin(params['mu'] * pi) / params['mu']
            d2phi_dr2 = np.gradient(np.gradient(phi, dx), dx)
            pi_dot = d2phi_dr2 - 2 * params['lambda'] * np.sqrt(np.abs(f)) * R * phi
            return phi + dt * phi_dot, pi + dt * pi_dot
    
    mock_sim = MockSimulator()
    adaptive_sim = AdaptiveReplicatorSimulator(mock_sim, stepper)
    
    # Setup test problem
    N = 64
    L = 5.0
    dx = 2 * L / N
    r = np.linspace(-L, L, N)
    
    # Initial fields (localized perturbation)
    phi_init = 0.1 * np.exp(-r**2/2)
    pi_init = 0.05 * np.exp(-r**2/4)
    
    # Mock metric and curvature
    f = 1.0 - 2.0/np.maximum(np.abs(r), 0.1) + 0.1*np.exp(-r**2/4)
    f = np.maximum(f, 0.01)  # Ensure positivity
    R = -np.gradient(np.gradient(f, dx), dx) / (2*f**2) + np.gradient(f, dx)**2 / (4*f**3)
    
    # Parameters
    params = {
        'lambda': 0.01,
        'mu': 0.20,
        'mass': 0.0
    }
    
    print(f"Grid: {N} points, dx={dx:.3f}")
    print(f"Parameters: {params}")
    
    # Run adaptive simulation
    results = adaptive_sim.simulate_adaptive(
        phi_init, pi_init, f, R, params, dx,
        t_final=2.0, max_steps=5000
    )
    
    print(f"\n=== RESULTS ===")
    print(f"Final time: {results['final_time']:.3f}")
    print(f"Total steps: {results['total_steps']}")
    print(f"Step efficiency: {results['step_efficiency']:.1%}")
    print(f"Final creation rate: {results['final_creation_rate']:.6f}")
    print(f"Average dt: {np.mean(results['dt_evolution']):.6f}")
    print(f"Min dt: {np.min(results['dt_evolution']):.6f}")
    print(f"Max dt: {np.max(results['dt_evolution']):.6f}")
    
    return results, stepper

if __name__ == "__main__":
    results, stepper = demo_adaptive_time_stepping()
    
    print(f"\n=== ADAPTIVE TIME STEPPING READY ===")
    print(f"✓ Error-controlled evolution implemented")
    print(f"✓ CFL stability analysis included")
    print(f"✓ Richardson extrapolation for error estimation")
    print(f"✓ Automatic step size adjustment")
    print(f"Framework ready for integration with 3D JAX replicator!")
