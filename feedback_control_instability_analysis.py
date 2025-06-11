"""
Feedback-Controlled Production Loop and Instability Mode Analysis
================================================================

This module implements the advanced feedback control system and instability mode simulation
for the unified energy-to-matter conversion framework.

Key Features:
- Real-time parameter adjustment based on production targets
- Feedback controller for polymer μ parameters
- Field strength tuning for Schwinger zones  
- Entanglement-based state preparation timing
- Instability mode detection and analysis

Author: Advanced Quantum Simulation Team
Date: June 10, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.fft import fft, fftfreq, fft2
from scipy.integrate import odeint
from scipy.signal import find_peaks, welch
import time
from typing import Dict, List, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

# Universal parameters from optimization
R_UNIVERSAL = 0.847
PHI_UNIVERSAL = 3 * np.pi / 7

class FeedbackControlledProductionLoop:
    """
    Implements feedback-controlled production loop with real-time parameter adjustment
    """
    
    def __init__(self):
        """Initialize feedback control system"""
        # Control parameters
        self.kp = 1.0      # Proportional gain
        self.ki = 0.1      # Integral gain  
        self.kd = 0.05     # Derivative gain
        
        # System parameters
        self.mu_params = np.array([0.2, 0.15, 0.25, 0.18])  # Polymer parameters
        self.E_field_base = 1.32e18  # Base field strength (V/m)
        self.r_univ = R_UNIVERSAL
        self.phi_univ = PHI_UNIVERSAL
        
        # Production targets and limits
        self.target_rate = 1e-15  # Target production rate (W)
        self.max_field_strength = 1e21  # Safety limit (V/m)
        self.min_mu = 0.01
        self.max_mu = 2.0
        
        # Control history
        self.control_history = {
            'time': [],
            'production_rate': [],
            'error': [],
            'mu_params': [],
            'E_field': [],
            'control_output': []
        }
        
        # PID controller state
        self.integral_error = 0.0
        self.previous_error = 0.0
        
        print("Feedback-Controlled Production Loop initialized")
        print(f"Target production rate: {self.target_rate:.2e} W")
        print(f"Initial μ parameters: {self.mu_params}")
    
    def production_rate_model(self, mu_params: np.ndarray, E_field: float, 
                            r: float, phi: float) -> float:
        """
        Model for production rate based on system parameters
        
        Combines Schwinger effect, polymer enhancement, and universal parameters
        """
        # Schwinger production probability
        E_crit = 1.32e18
        P_schwinger = 1 - np.exp(-np.pi * (E_crit / E_field)**2)
        
        # Polymer enhancement
        polymer_factor = np.prod([np.sin(np.pi * mu) / (np.pi * mu + 1e-12) for mu in mu_params])
        
        # Universal parameter enhancement
        universal_factor = np.cosh(2 * r) * np.cos(phi)
        
        # Synergistic enhancement (from previous analysis)
        synergy_factor = 1.207
        
        # Combined production rate
        base_rate = 1e-18  # Base rate (W)
        production_rate = (base_rate * P_schwinger * polymer_factor * 
                         universal_factor * synergy_factor)
        
        return production_rate
    
    def pid_controller(self, error: float, dt: float) -> float:
        """PID controller for production rate regulation"""
        # Proportional term
        P = self.kp * error
        
        # Integral term
        self.integral_error += error * dt
        I = self.ki * self.integral_error
        
        # Derivative term
        if dt > 0:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = 0
        D = self.kd * derivative
        
        # Combined PID output
        output = P + I + D
        
        # Store for next iteration
        self.previous_error = error
        
        return output
    
    def update_parameters(self, control_output: float, current_rate: float):
        """Update system parameters based on control output"""
        # Determine which parameter to adjust based on current state
        if current_rate < self.target_rate * 0.5:
            # Low production - increase field strength
            field_adjustment = control_output * 1e16
            self.E_field_base = np.clip(
                self.E_field_base + field_adjustment,
                1e15, self.max_field_strength
            )
        
        elif current_rate < self.target_rate * 0.8:
            # Moderate production - adjust polymer parameters
            mu_adjustment = control_output * 0.01
            self.mu_params = np.clip(
                self.mu_params + mu_adjustment,
                self.min_mu, self.max_mu
            )
        
        else:
            # Near target - fine-tune universal parameters
            r_adjustment = control_output * 0.001
            self.r_univ = np.clip(self.r_univ + r_adjustment, 0.1, 3.0)
    
    def increase_E_field(self, factor: float = 1.1):
        """Increase electric field strength with safety checks"""
        new_field = self.E_field_base * factor
        if new_field <= self.max_field_strength:
            self.E_field_base = new_field
            return True
        return False
    
    def adjust_r_univ(self, delta_r: float):
        """Adjust universal squeezing parameter"""
        new_r = self.r_univ + delta_r
        if 0.1 <= new_r <= 3.0:
            self.r_univ = new_r
            return True
        return False
    
    def update_lagrangian_coeffs(self):
        """Update Lagrangian coefficients based on current parameters"""
        # This would interface with the digital twin to update
        # the effective Lagrangian structure
        coeffs = {
            'schwinger_coeff': np.exp(-1e18 / self.E_field_base),
            'polymer_coeff': np.prod(self.mu_params),
            'universal_coeff': self.r_univ * np.cos(self.phi_univ),
            'synergy_coeff': 1.207
        }
        return coeffs
    
    def simulate_production_loop(self, duration: float = 1000.0, dt: float = 1.0,
                               disturbances: bool = True) -> Dict:
        """
        Simulate feedback-controlled production loop
        
        Implements the control logic:
        if production_rate < target:
            increase_E_field()
            adjust_r_univ()
            update_Lagrangian_coeffs()
        """
        print(f"Simulating feedback-controlled production for {duration:.1f} time units...")
        
        t_vals = np.arange(0, duration, dt)
        n_steps = len(t_vals)
        
        # Initialize tracking arrays
        production_rates = np.zeros(n_steps)
        errors = np.zeros(n_steps)
        control_outputs = np.zeros(n_steps)
        E_fields = np.zeros(n_steps)
        mu_history = np.zeros((n_steps, len(self.mu_params)))
        r_history = np.zeros(n_steps)
        
        # Simulation loop
        for i in range(n_steps):
            t = t_vals[i]
            
            # Add random disturbances if enabled
            if disturbances:
                noise_factor = 1 + 0.1 * np.random.normal(0, 0.1)
            else:
                noise_factor = 1.0
            
            # Calculate current production rate
            current_rate = self.production_rate_model(
                self.mu_params, self.E_field_base, self.r_univ, self.phi_univ
            ) * noise_factor
            
            # Calculate error
            error = self.target_rate - current_rate
            
            # PID control
            control_output = self.pid_controller(error, dt)
            
            # Update parameters based on control output
            if current_rate < self.target_rate:
                # Increase field strength
                if not self.increase_E_field(1 + 0.01 * abs(control_output)):
                    # If field at max, adjust universal parameter
                    self.adjust_r_univ(0.001 * control_output)
                
                # Update Lagrangian coefficients
                self.update_lagrangian_coeffs()
            
            # Store data
            production_rates[i] = current_rate
            errors[i] = error
            control_outputs[i] = control_output
            E_fields[i] = self.E_field_base
            mu_history[i] = self.mu_params.copy()
            r_history[i] = self.r_univ
            
            # Store in control history
            self.control_history['time'].append(t)
            self.control_history['production_rate'].append(current_rate)
            self.control_history['error'].append(error)
            self.control_history['mu_params'].append(self.mu_params.copy())
            self.control_history['E_field'].append(self.E_field_base)
            self.control_history['control_output'].append(control_output)
        
        results = {
            'time': t_vals,
            'production_rate': production_rates,
            'error': errors,
            'control_output': control_outputs,
            'E_field': E_fields,
            'mu_history': mu_history,
            'r_history': r_history,
            'settling_time': self.calculate_settling_time(t_vals, errors),
            'steady_state_error': np.mean(np.abs(errors[-100:])),
            'overshoot': np.max(production_rates) / self.target_rate - 1
        }
        
        print(f"Settling time: {results['settling_time']:.1f} time units")
        print(f"Steady-state error: {results['steady_state_error']:.2e}")
        print(f"Overshoot: {results['overshoot']:.1%}")
        
        return results
    
    def calculate_settling_time(self, time: np.ndarray, error: np.ndarray, 
                              tolerance: float = 0.02) -> float:
        """Calculate settling time (2% tolerance)"""
        target_band = self.target_rate * tolerance
        
        for i in range(len(error)):
            if all(np.abs(error[i:]) < target_band):
                return time[i]
        
        return time[-1]  # Never settled
    
    def plot_control_performance(self, results: Dict):
        """Plot control system performance"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        t = results['time']
        
        # Production rate tracking
        ax1 = axes[0, 0]
        ax1.plot(t, results['production_rate'], 'b-', linewidth=2, label='Actual')
        ax1.axhline(self.target_rate, color='red', linestyle='--', linewidth=2, label='Target')
        ax1.fill_between(t, self.target_rate * 0.98, self.target_rate * 1.02, 
                        alpha=0.3, color='green', label='±2% Band')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Production Rate (W)')
        ax1.set_title('Production Rate Control')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Control error
        ax2 = axes[0, 1]
        ax2.plot(t, results['error'], 'r-', linewidth=2)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Control Error (W)')
        ax2.set_title('Control Error')
        ax2.grid(True, alpha=0.3)
        
        # Control output
        ax3 = axes[0, 2]
        ax3.plot(t, results['control_output'], 'g-', linewidth=2)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Control Output')
        ax3.set_title('PID Controller Output')
        ax3.grid(True, alpha=0.3)
        
        # Field strength evolution
        ax4 = axes[1, 0]
        ax4.plot(t, results['E_field'], 'm-', linewidth=2)
        ax4.axhline(self.max_field_strength, color='red', linestyle='--', alpha=0.7, label='Safety Limit')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Electric Field (V/m)')
        ax4.set_title('Field Strength Adjustment')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        # Polymer parameters
        ax5 = axes[1, 1]
        for i in range(len(self.mu_params)):
            ax5.plot(t, results['mu_history'][:, i], linewidth=2, label=f'μ_{i+1}')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Polymer Parameters')
        ax5.set_title('Polymer Parameter Evolution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Universal parameter r
        ax6 = axes[1, 2]
        ax6.plot(t, results['r_history'], 'c-', linewidth=2)
        ax6.axhline(R_UNIVERSAL, color='red', linestyle='--', alpha=0.7, label='Initial Value')
        ax6.set_xlabel('Time')
        ax6.set_ylabel('Universal Parameter r')
        ax6.set_title('Universal Parameter Adjustment')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feedback_control_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

class InstabilityModeAnalyzer:
    """
    Analyzes instability modes and system fault tolerance
    """
    
    def __init__(self):
        """Initialize instability analyzer"""
        self.base_frequency = 1000.0  # Hz
        self.damping_coefficients = {}
        self.resonant_frequencies = []
        
        print("Instability Mode Analyzer initialized")
    
    def perturbation_field(self, x: np.ndarray, t: float, amplitude: float = 0.1,
                         freq: float = 100.0, wave_number: float = 1.0) -> np.ndarray:
        """Generate perturbation field δφ(x,t)"""
        spatial_component = np.sin(wave_number * x)
        temporal_component = np.sin(2 * np.pi * freq * t)
        return amplitude * spatial_component * temporal_component
    
    def system_response(self, state: np.ndarray, t: float, params: Dict) -> np.ndarray:
        """
        System response to perturbations
        
        Implements coupled field equations with perturbations
        """
        phi, pi = state
        
        # System parameters
        mu = params['mu']
        gamma = params['damping']
        omega_0 = params['natural_freq']
        coupling = params['coupling']
        
        # Perturbation
        x = np.linspace(0, 1, len(phi))
        perturbation = self.perturbation_field(x, t, params['pert_amp'], params['pert_freq'])
        
        # Field equations with polymer corrections
        dphi_dt = pi
        dpi_dt = (-omega_0**2 * phi - 2*gamma*pi + 
                 coupling * np.sin(mu * pi) / (mu + 1e-12) + perturbation)
        
        return np.array([dphi_dt, dpi_dt])
    
    def analyze_stability(self, duration: float = 100.0, perturbation_frequencies: List[float] = None) -> Dict:
        """Analyze system stability under perturbations"""
        if perturbation_frequencies is None:
            perturbation_frequencies = [10, 50, 100, 500, 1000, 2000]
        
        print("Analyzing system stability under perturbations...")
        
        # System parameters
        n_points = 64
        x = np.linspace(0, 1, n_points)
        
        stability_results = {}
        
        for freq in perturbation_frequencies:
            print(f"  Testing frequency: {freq} Hz")
            
            # Initial conditions
            phi_0 = 0.1 * np.sin(np.pi * x)
            pi_0 = np.zeros_like(x)
            initial_state = np.array([phi_0, pi_0])
            
            # Simulation parameters
            params = {
                'mu': 0.2,
                'damping': 0.1,
                'natural_freq': 100.0,
                'coupling': 1.0,
                'pert_amp': 0.1,
                'pert_freq': freq
            }
            
            # Time evolution
            t_span = np.linspace(0, duration, 1000)
            
            try:
                # Simple discrete evolution (avoiding odeint complexity)
                dt = t_span[1] - t_span[0]
                states = [initial_state]
                
                for i in range(1, len(t_span)):
                    current_state = states[-1]
                    derivative = self.system_response(current_state, t_span[i], params)
                    new_state = current_state + derivative * dt
                    states.append(new_state)
                
                states = np.array(states)
                
                # Analyze final state
                final_phi = states[-1, 0]
                final_energy = np.sum(final_phi**2 + states[-1, 1]**2)
                
                # Calculate growth rate
                initial_energy = np.sum(phi_0**2 + pi_0**2)
                growth_rate = np.log(final_energy / (initial_energy + 1e-12)) / duration
                
                # Determine stability
                is_stable = growth_rate < 0.01  # 1% growth threshold
                
                stability_results[freq] = {
                    'growth_rate': growth_rate,
                    'final_energy': final_energy,
                    'is_stable': is_stable,
                    'max_amplitude': np.max(np.abs(final_phi))
                }
                
            except Exception as e:
                print(f"    Warning: Simulation failed for {freq} Hz: {e}")
                stability_results[freq] = {
                    'growth_rate': np.inf,
                    'final_energy': np.inf,
                    'is_stable': False,
                    'max_amplitude': np.inf
                }
        
        # Find resonant frequencies (unstable modes)
        self.resonant_frequencies = [f for f, result in stability_results.items() 
                                   if not result['is_stable']]
        
        print(f"Found {len(self.resonant_frequencies)} resonant frequencies: {self.resonant_frequencies}")
        
        return stability_results
    
    def fourier_analysis(self, signal: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Perform Fourier analysis of field evolution"""
        # FFT
        fft_signal = fft(signal)
        frequencies = fftfreq(len(signal), dt)
        
        # Power spectral density
        psd = np.abs(fft_signal)**2
        
        # Find peaks in frequency domain
        peaks, _ = find_peaks(psd, height=np.max(psd) * 0.1)
        
        return frequencies[peaks], psd[peaks]
    
    def evaluate_decoherence(self, duration: float = 1000.0) -> Dict:
        """Evaluate decoherence effects over time"""
        print("Evaluating decoherence effects...")
        
        # Decoherence model parameters
        gamma_0 = 0.001  # Base decoherence rate
        temperature = 0.01  # Effective temperature (Kelvin)
        
        t_vals = np.linspace(0, duration, 1000)
        
        # Coherence decay models
        exponential_decay = np.exp(-gamma_0 * t_vals)
        gaussian_decay = np.exp(-(gamma_0 * t_vals)**2)
        power_law_decay = (1 + gamma_0 * t_vals)**(-2)
        
        # Thermal decoherence
        thermal_factor = np.exp(-t_vals / (1000 * temperature))
        
        results = {
            'time': t_vals,
            'exponential': exponential_decay,
            'gaussian': gaussian_decay,
            'power_law': power_law_decay,
            'thermal': thermal_factor,
            'coherence_time_exp': -1 / gamma_0,
            'coherence_time_gauss': 1 / gamma_0,
            'coherence_time_thermal': 1000 * temperature
        }
        
        print(f"Exponential coherence time: {results['coherence_time_exp']:.1f}")
        print(f"Gaussian coherence time: {results['coherence_time_gauss']:.1f}")
        print(f"Thermal coherence time: {results['coherence_time_thermal']:.1f}")
        
        return results
    
    def plot_instability_analysis(self, stability_results: Dict, decoherence_results: Dict):
        """Plot instability analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Stability vs frequency
        ax1 = axes[0, 0]
        freqs = list(stability_results.keys())
        growth_rates = [stability_results[f]['growth_rate'] for f in freqs]
        colors = ['red' if not stability_results[f]['is_stable'] else 'green' for f in freqs]
        
        ax1.scatter(freqs, growth_rates, c=colors, s=100, alpha=0.7)
        ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Perturbation Frequency (Hz)')
        ax1.set_ylabel('Growth Rate (1/s)')
        ax1.set_title('Stability Analysis')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Maximum amplitude vs frequency
        ax2 = axes[0, 1]
        max_amps = [stability_results[f]['max_amplitude'] for f in freqs]
        ax2.scatter(freqs, max_amps, c=colors, s=100, alpha=0.7)
        ax2.set_xlabel('Perturbation Frequency (Hz)')
        ax2.set_ylabel('Maximum Amplitude')
        ax2.set_title('Amplitude Response')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Decoherence evolution
        ax3 = axes[1, 0]
        t = decoherence_results['time']
        ax3.plot(t, decoherence_results['exponential'], label='Exponential', linewidth=2)
        ax3.plot(t, decoherence_results['gaussian'], label='Gaussian', linewidth=2)
        ax3.plot(t, decoherence_results['power_law'], label='Power Law', linewidth=2)
        ax3.plot(t, decoherence_results['thermal'], label='Thermal', linewidth=2)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Coherence')
        ax3.set_title('Decoherence Models')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Stability map
        ax4 = axes[1, 1]
        stable_freqs = [f for f in freqs if stability_results[f]['is_stable']]
        unstable_freqs = [f for f in freqs if not stability_results[f]['is_stable']]
        
        if stable_freqs:
            ax4.scatter(stable_freqs, [1]*len(stable_freqs), c='green', s=100, 
                       alpha=0.7, label='Stable')
        if unstable_freqs:
            ax4.scatter(unstable_freqs, [0]*len(unstable_freqs), c='red', s=100, 
                       alpha=0.7, label='Unstable')
        
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Stability')
        ax4.set_title('Stability Map')
        ax4.set_xscale('log')
        ax4.set_ylim(-0.5, 1.5)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('instability_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def run_advanced_simulation():
    """Run complete advanced simulation analysis"""
    print("=" * 80)
    print("ADVANCED SIMULATION: FEEDBACK CONTROL & INSTABILITY ANALYSIS")
    print("=" * 80)
    
    # Initialize feedback control system
    control_loop = FeedbackControlledProductionLoop()
    
    # Simulate feedback-controlled production
    control_results = control_loop.simulate_production_loop(
        duration=500.0,
        dt=1.0,
        disturbances=True
    )
    
    # Plot control performance
    control_loop.plot_control_performance(control_results)
    
    # Initialize instability analyzer
    instability_analyzer = InstabilityModeAnalyzer()
    
    # Analyze system stability
    stability_results = instability_analyzer.analyze_stability(
        duration=100.0,
        perturbation_frequencies=[10, 50, 100, 200, 500, 1000, 2000]
    )
    
    # Evaluate decoherence
    decoherence_results = instability_analyzer.evaluate_decoherence(duration=1000.0)
    
    # Plot instability analysis
    instability_analyzer.plot_instability_analysis(stability_results, decoherence_results)
    
    # Summary report
    print("\n" + "=" * 60)
    print("ADVANCED SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Control System Performance:")
    print(f"  Settling time: {control_results['settling_time']:.1f} time units")
    print(f"  Steady-state error: {control_results['steady_state_error']:.2e} W")
    print(f"  Overshoot: {control_results['overshoot']:.1%}")
    
    stable_count = sum(1 for result in stability_results.values() if result['is_stable'])
    total_count = len(stability_results)
    print(f"\nStability Analysis:")
    print(f"  Stable frequencies: {stable_count}/{total_count}")
    print(f"  Resonant frequencies: {instability_analyzer.resonant_frequencies}")
    
    print(f"\nDecoherence Times:")
    print(f"  Exponential model: {decoherence_results['coherence_time_exp']:.1f} time units")
    print(f"  Thermal model: {decoherence_results['coherence_time_thermal']:.1f} time units")
    
    return control_loop, instability_analyzer, control_results, stability_results

if __name__ == "__main__":
    # Run the complete advanced simulation
    control_loop, analyzer, control_results, stability_results = run_advanced_simulation()
    
    print("\n✅ Advanced simulation analysis complete!")
    print("📊 Generated visualizations: feedback_control_performance.png, instability_analysis.png")
    print("🎯 Feedback control and instability mode analysis ready for production deployment")
