#!/usr/bin/env python3
"""
Real-Time Field Monitoring and Control System
============================================

This module implements comprehensive real-time monitoring and control
for electromagnetic fields in energy-to-matter conversion experiments:

1. THz Time-Domain Spectroscopy for field measurement
2. Real-time feedback control with microsecond response
3. Multi-channel field monitoring across 3D space
4. Automated field optimization and stability control
5. Safety monitoring with emergency shutdown protocols

Technical Specifications:
- Temporal Resolution: 100 fs (THz-TDS)
- Spatial Resolution: 10 μm (scanning probe)
- Field Range: 10¹² - 10¹⁸ V/m
- Response Time: < 1 μs (feedback control)
- Bandwidth: 0.1 - 10 THz
"""

import numpy as np
import scipy.signal as signal
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import time
import threading
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import queue
import asyncio
import warnings
warnings.filterwarnings('ignore')

class FieldType(Enum):
    """Types of electromagnetic fields"""
    DC = "dc"
    AC = "ac" 
    PULSED = "pulsed"
    CHIRPED = "chirped"
    SQUEEZED = "squeezed"
    CASIMIR = "casimir"

class MonitoringMode(Enum):
    """Field monitoring modes"""
    CONTINUOUS = "continuous"
    TRIGGERED = "triggered"
    BURST = "burst"
    SINGLE_SHOT = "single_shot"

class ControlMode(Enum):
    """Field control modes"""
    MANUAL = "manual"
    FEEDBACK = "feedback"
    ADAPTIVE = "adaptive"
    OPTIMIZATION = "optimization"

@dataclass
class FieldSpecs:
    """Electromagnetic field specifications"""
    frequency_range: Tuple[float, float] = (1e11, 1e13)  # 0.1-10 THz
    amplitude_range: Tuple[float, float] = (1e12, 1e18)  # V/m
    pulse_duration_range: Tuple[float, float] = (1e-15, 1e-9)  # fs-ns
    repetition_rate_range: Tuple[float, float] = (1, 1e6)  # Hz-MHz
    phase_stability: float = 0.01  # 1% phase stability
    amplitude_stability: float = 0.001  # 0.1% amplitude stability

@dataclass
class MonitoringSpecs:
    """Field monitoring system specifications"""
    temporal_resolution: float = 100e-15  # 100 fs
    spatial_resolution: float = 10e-6     # 10 μm
    dynamic_range: float = 80             # 80 dB
    sensitivity: float = 1e12             # V/m minimum
    bandwidth: float = 10e12              # 10 THz
    sampling_rate: float = 25e12          # 25 THz (Nyquist for 12.5 THz)

@dataclass
class ControlSpecs:
    """Field control system specifications"""
    response_time: float = 1e-6           # 1 μs
    stability_time: float = 10e-6         # 10 μs settling
    control_bandwidth: float = 1e6        # 1 MHz
    precision: float = 0.001              # 0.1% control precision
    correction_range: float = 0.1         # ±10% adjustment range

class RealTimeFieldMonitor:
    """Real-time electromagnetic field monitoring system"""
    
    def __init__(self, field_specs: FieldSpecs = None,
                 monitoring_specs: MonitoringSpecs = None,
                 control_specs: ControlSpecs = None):
        
        # System specifications
        self.field_specs = field_specs or FieldSpecs()
        self.monitoring_specs = monitoring_specs or MonitoringSpecs()
        self.control_specs = control_specs or ControlSpecs()
        
        # Physical constants
        self.c = 2.998e8  # m/s
        self.epsilon_0 = 8.854e-12  # F/m
        self.mu_0 = 4e-7 * np.pi  # H/m
        self.hbar = 1.055e-34  # J⋅s
        
        # System state
        self.monitoring_active = False
        self.control_active = False
        self.emergency_stop_active = False
        
        # Current field measurements
        self.current_field_amplitude = 0.0
        self.current_field_frequency = 0.0
        self.current_field_phase = 0.0
        self.current_field_polarization = np.array([1, 0, 0])
        
        # Target field parameters
        self.target_field_amplitude = 0.0
        self.target_field_frequency = 0.0
        self.target_field_phase = 0.0
        
        # Spatial field map
        self.field_grid_size = 64
        self.field_grid_spacing = 10e-6  # 10 μm
        self.field_map = np.zeros((self.field_grid_size, self.field_grid_size, 3))
        
        # Data logging
        self.field_history = []
        self.control_history = []
        self.timestamp_history = []
        
        # Control threads
        self.monitoring_thread = None
        self.control_thread = None
        self.optimization_thread = None
        self.stop_threads = threading.Event()
        
        # Initialize monitoring systems
        self.initialize_monitoring_systems()
        
    def initialize_monitoring_systems(self):
        """Initialize all field monitoring systems"""
        print("🔧 Initializing Real-Time Field Monitoring System")
        print(f"   Frequency Range: {self.field_specs.frequency_range[0]/1e12:.1f}-"
              f"{self.field_specs.frequency_range[1]/1e12:.1f} THz")
        print(f"   Amplitude Range: {self.field_specs.amplitude_range[0]/1e15:.1f}-"
              f"{self.field_specs.amplitude_range[1]/1e15:.1f} × 10^15 V/m")
        print(f"   Temporal Resolution: {self.monitoring_specs.temporal_resolution*1e15:.0f} fs")
        print(f"   Response Time: {self.control_specs.response_time*1e6:.1f} μs")
        
        # Initialize THz time-domain spectroscopy
        self.initialize_thz_tds()
        
        # Initialize spatial scanning system
        self.initialize_spatial_scanner()
        
        # Initialize feedback control system
        self.initialize_feedback_control()
        
        # Initialize field generation system
        self.initialize_field_generation()
        
        # Start monitoring threads
        self.start_monitoring_threads()
        
        print("✅ Field monitoring system initialized")
        
    def initialize_thz_tds(self):
        """Initialize THz time-domain spectroscopy system"""
        print("   🔹 Initializing THz time-domain spectroscopy...")
        
        # THz pulse parameters
        self.thz_pulse_duration = 200e-15  # 200 fs
        self.thz_pulse_energy = 1e-9       # 1 nJ
        self.thz_repetition_rate = 1e3     # 1 kHz
        
        # Detection system
        self.electro_optic_crystal = "ZnTe"  # Zinc telluride
        self.detection_bandwidth = 5e12      # 5 THz
        
        print(f"      Pulse Duration: {self.thz_pulse_duration*1e15:.0f} fs")
        print(f"      Detection Bandwidth: {self.detection_bandwidth/1e12:.1f} THz")
        
    def initialize_spatial_scanner(self):
        """Initialize 3D spatial field scanning system"""
        print("   🔹 Initializing 3D spatial scanner...")
        
        # Scanning probe specifications
        self.probe_diameter = 5e-6         # 5 μm diameter
        self.scanning_speed = 1e-3         # 1 mm/s
        self.position_accuracy = 100e-9    # 100 nm
        
        # Create spatial grid
        x_grid = np.linspace(0, self.field_grid_size * self.field_grid_spacing, 
                           self.field_grid_size)
        y_grid = np.linspace(0, self.field_grid_size * self.field_grid_spacing,
                           self.field_grid_size)
        self.X_grid, self.Y_grid = np.meshgrid(x_grid, y_grid)
        
        print(f"      Grid Size: {self.field_grid_size} × {self.field_grid_size}")
        print(f"      Spatial Resolution: {self.field_grid_spacing*1e6:.0f} μm")
        
    def initialize_feedback_control(self):
        """Initialize real-time feedback control system"""
        print("   🔹 Initializing feedback control system...")
        
        # PID controller parameters
        self.pid_kp = 1.0     # Proportional gain
        self.pid_ki = 0.1     # Integral gain  
        self.pid_kd = 0.01    # Derivative gain
        
        # Control state variables
        self.control_error_integral = 0.0
        self.control_error_previous = 0.0
        self.control_output = 0.0
        
        print(f"      Response Time: {self.control_specs.response_time*1e6:.1f} μs")
        print(f"      Control Bandwidth: {self.control_specs.control_bandwidth/1e6:.1f} MHz")
        
    def initialize_field_generation(self):
        """Initialize field generation and control hardware"""
        print("   🔹 Initializing field generation system...")
        
        # Field generation capabilities
        self.max_field_amplitude = 5e17   # V/m (below breakdown)
        self.field_rise_time = 100e-15    # 100 fs rise time
        self.phase_control_resolution = 0.01  # 0.01 radian resolution
        
        print(f"      Maximum Field: {self.max_field_amplitude/1e17:.1f} × 10^17 V/m")
        print(f"      Rise Time: {self.field_rise_time*1e15:.0f} fs")
        
    def start_monitoring_threads(self):
        """Start all real-time monitoring and control threads"""
        print("   🔹 Starting real-time monitoring threads...")
        
        # Field monitoring thread
        self.monitoring_thread = threading.Thread(target=self._field_monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Control thread
        self.control_thread = threading.Thread(target=self._field_control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        # Optimization thread
        self.optimization_thread = threading.Thread(target=self._field_optimization_loop)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
        
        self.monitoring_active = True
        
    def _field_monitoring_loop(self):
        """Real-time field monitoring loop"""
        while not self.stop_threads.is_set() and self.monitoring_active:
            current_time = time.time()
            
            # Measure field parameters
            field_data = self.measure_field_thz_tds()
            
            # Update current field state
            self.current_field_amplitude = field_data['amplitude']
            self.current_field_frequency = field_data['frequency']
            self.current_field_phase = field_data['phase']
            
            # Log measurements
            self.field_history.append({
                'timestamp': current_time,
                'amplitude': self.current_field_amplitude,
                'frequency': self.current_field_frequency,
                'phase': self.current_field_phase,
                'polarization': self.current_field_polarization.copy()
            })
            
            # Spatial field mapping (periodic)
            if len(self.field_history) % 100 == 0:  # Every 100 measurements
                self.update_spatial_field_map()
            
            # Safety monitoring
            if self.current_field_amplitude > 0.95 * self.max_field_amplitude:
                self.trigger_emergency_stop("Field amplitude approaching maximum")
            
            # Maintain history length
            if len(self.field_history) > 10000:
                self.field_history = self.field_history[-5000:]
            
            time.sleep(self.monitoring_specs.temporal_resolution)
    
    def _field_control_loop(self):
        """Real-time field control loop"""
        while not self.stop_threads.is_set() and not self.emergency_stop_active:
            if self.control_active:
                current_time = time.time()
                
                # Calculate control error
                amplitude_error = self.target_field_amplitude - self.current_field_amplitude
                frequency_error = self.target_field_frequency - self.current_field_frequency
                phase_error = self.target_field_phase - self.current_field_phase
                
                # PID control for amplitude
                control_output = self.calculate_pid_output(amplitude_error)
                
                # Apply field corrections
                if abs(control_output) > 1e-6:  # Minimum correction threshold
                    self.apply_field_correction(control_output, frequency_error, phase_error)
                
                # Log control actions
                self.control_history.append({
                    'timestamp': current_time,
                    'amplitude_error': amplitude_error,
                    'frequency_error': frequency_error,
                    'phase_error': phase_error,
                    'control_output': control_output
                })
                
                # Maintain control history
                if len(self.control_history) > 5000:
                    self.control_history = self.control_history[-2500:]
            
            time.sleep(self.control_specs.response_time)
    
    def _field_optimization_loop(self):
        """Automated field optimization loop"""
        while not self.stop_threads.is_set() and not self.emergency_stop_active:
            if len(self.field_history) > 1000:  # Need sufficient data
                # Analyze field stability
                recent_amplitudes = [h['amplitude'] for h in self.field_history[-1000:]]
                amplitude_std = np.std(recent_amplitudes)
                amplitude_mean = np.mean(recent_amplitudes)
                
                stability_ratio = amplitude_std / (amplitude_mean + 1e-30)
                
                # Optimize if stability is poor
                if stability_ratio > self.field_specs.amplitude_stability * 2:
                    self.optimize_field_stability()
            
            time.sleep(1.0)  # Optimization every second
    
    def measure_field_thz_tds(self) -> Dict[str, float]:
        """Measure electromagnetic field using THz time-domain spectroscopy"""
        # Simulate THz-TDS measurement
        
        # Generate field pulse simulation
        t_measurement = np.linspace(0, 10e-12, 1000)  # 10 ps window
        
        # Simulated field components
        base_amplitude = 1e15  # V/m
        base_frequency = 1e12  # 1 THz
        
        # Add noise and environmental effects
        amplitude_noise = np.random.normal(0, base_amplitude * 0.001)  # 0.1% noise
        frequency_drift = np.random.normal(0, base_frequency * 0.0001)  # 0.01% drift
        phase_jitter = np.random.normal(0, 0.01)  # 0.01 radian jitter
        
        measured_amplitude = base_amplitude + amplitude_noise
        measured_frequency = base_frequency + frequency_drift
        measured_phase = phase_jitter
        
        # Polarization analysis
        polarization_x = np.random.normal(0.7, 0.05)  # Mostly x-polarized
        polarization_y = np.sqrt(1 - polarization_x**2)
        polarization_z = 0.0
        
        self.current_field_polarization = np.array([polarization_x, polarization_y, polarization_z])
        
        return {
            'amplitude': measured_amplitude,
            'frequency': measured_frequency,
            'phase': measured_phase,
            'polarization': self.current_field_polarization,
            'signal_to_noise': measured_amplitude / (amplitude_noise + 1e12),
            'measurement_time': time.time()
        }
    
    def update_spatial_field_map(self):
        """Update 3D spatial field map"""
        # Simulate spatial field distribution
        for i in range(self.field_grid_size):
            for j in range(self.field_grid_size):
                x = i * self.field_grid_spacing
                y = j * self.field_grid_spacing
                
                # Distance from field source (center)
                center_x = self.field_grid_size * self.field_grid_spacing / 2
                center_y = self.field_grid_size * self.field_grid_spacing / 2
                r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                # Field distribution model
                field_strength = self.current_field_amplitude * np.exp(-r**2 / (100e-6)**2)
                
                # Store E-field components
                self.field_map[i, j, 0] = field_strength  # Ex
                self.field_map[i, j, 1] = 0  # Ey
                self.field_map[i, j, 2] = 0  # Ez
    
    def calculate_pid_output(self, error: float) -> float:
        """Calculate PID controller output"""
        current_time = time.time()
        
        # Proportional term
        p_term = self.pid_kp * error
        
        # Integral term
        self.control_error_integral += error * self.control_specs.response_time
        i_term = self.pid_ki * self.control_error_integral
        
        # Derivative term
        error_derivative = (error - self.control_error_previous) / self.control_specs.response_time
        d_term = self.pid_kd * error_derivative
        
        # Combined output
        self.control_output = p_term + i_term + d_term
        
        # Clamp output to safe range
        max_output = self.control_specs.correction_range * self.max_field_amplitude
        self.control_output = np.clip(self.control_output, -max_output, max_output)
        
        # Update previous error
        self.control_error_previous = error
        
        return self.control_output
    
    def apply_field_correction(self, amplitude_correction: float,
                             frequency_correction: float,
                             phase_correction: float):
        """Apply real-time field corrections"""
        # Amplitude correction
        if abs(amplitude_correction) > 1e12:  # Minimum 1 TV/m correction
            # Simulate field amplitude adjustment
            correction_factor = 1 + amplitude_correction / self.current_field_amplitude
            print(f"   📊 Applying amplitude correction: {correction_factor:.4f}×")
        
        # Frequency correction
        if abs(frequency_correction) > 1e9:  # Minimum 1 GHz correction
            # Simulate frequency tuning
            print(f"   🎵 Applying frequency correction: {frequency_correction/1e9:.3f} GHz")
        
        # Phase correction
        if abs(phase_correction) > 0.001:  # Minimum 1 mrad correction
            # Simulate phase adjustment
            print(f"   🔄 Applying phase correction: {phase_correction*1000:.1f} mrad")
    
    def optimize_field_stability(self):
        """Optimize field parameters for improved stability"""
        print("🔧 Optimizing field stability...")
        
        # Analyze recent field history
        if len(self.field_history) < 100:
            return
        
        recent_data = self.field_history[-1000:]
        amplitudes = [d['amplitude'] for d in recent_data]
        frequencies = [d['frequency'] for d in recent_data]
        phases = [d['phase'] for d in recent_data]
        
        # Calculate stability metrics
        amp_stability = np.std(amplitudes) / np.mean(amplitudes)
        freq_stability = np.std(frequencies) / np.mean(frequencies)
        phase_stability = np.std(phases)
        
        print(f"   Current stability: Amp {amp_stability*100:.3f}%, "
              f"Freq {freq_stability*100:.3f}%, Phase {phase_stability*1000:.1f} mrad")
        
        # Optimization strategies
        if amp_stability > self.field_specs.amplitude_stability:
            # Adjust control gains
            self.pid_kp *= 0.9
            self.pid_ki *= 1.1
            print("   📈 Adjusted PID gains for amplitude stability")
        
        if freq_stability > 0.0001:  # 0.01% frequency stability target
            # Implement frequency lock
            mean_frequency = np.mean(frequencies)
            self.target_field_frequency = mean_frequency
            print(f"   🔒 Locked frequency to {mean_frequency/1e12:.3f} THz")
        
        if phase_stability > 0.1:  # 0.1 radian phase stability target
            # Phase lock optimization
            mean_phase = np.mean(phases)
            self.target_field_phase = mean_phase
            print(f"   🔄 Locked phase to {mean_phase:.3f} rad")
    
    def set_target_field(self, amplitude: float, frequency: float, 
                        phase: float = 0.0) -> bool:
        """Set target field parameters for control system"""
        # Validate parameters
        if amplitude > self.max_field_amplitude:
            print(f"⚠️  Amplitude {amplitude/1e17:.2f}×10^17 V/m exceeds maximum")
            return False
        
        if not (self.field_specs.frequency_range[0] <= frequency <= 
                self.field_specs.frequency_range[1]):
            print(f"⚠️  Frequency {frequency/1e12:.2f} THz outside range")
            return False
        
        # Set targets
        self.target_field_amplitude = amplitude
        self.target_field_frequency = frequency
        self.target_field_phase = phase
        self.control_active = True
        
        print(f"🎯 Target field set:")
        print(f"   Amplitude: {amplitude/1e15:.2f} × 10^15 V/m")
        print(f"   Frequency: {frequency/1e12:.2f} THz")
        print(f"   Phase: {phase:.3f} rad")
        
        return True
    
    def measure_field_at_position(self, position: np.ndarray) -> Dict[str, float]:
        """Measure field at specific spatial position"""
        x, y, z = position
        
        # Convert to grid indices
        i = int(x / self.field_grid_spacing)
        j = int(y / self.field_grid_spacing)
        
        if 0 <= i < self.field_grid_size and 0 <= j < self.field_grid_size:
            # Interpolate field from map
            field_vector = self.field_map[i, j, :]
            field_magnitude = np.linalg.norm(field_vector)
            
            # Add measurement noise
            noise = np.random.normal(0, field_magnitude * 0.001)
            measured_field = field_magnitude + noise
        else:
            measured_field = 0.0
            field_vector = np.zeros(3)
        
        return {
            'position': position,
            'field_magnitude': measured_field,
            'field_vector': field_vector,
            'measurement_noise': abs(noise) if 'noise' in locals() else 0.0
        }
    
    def trigger_emergency_stop(self, reason: str):
        """Trigger emergency field shutdown"""
        print(f"🚨 EMERGENCY STOP: {reason}")
        
        self.emergency_stop_active = True
        self.control_active = False
        
        # Zero all fields immediately
        self.target_field_amplitude = 0.0
        self.current_field_amplitude = 0.0
        
        print("   🔴 All fields disabled")
        print("   🔴 Control system halted")
    
    def get_field_statistics(self) -> Dict[str, float]:
        """Calculate comprehensive field statistics"""
        if len(self.field_history) < 10:
            return {}
        
        recent_data = self.field_history[-1000:] if len(self.field_history) >= 1000 else self.field_history
        
        amplitudes = [d['amplitude'] for d in recent_data]
        frequencies = [d['frequency'] for d in recent_data]
        phases = [d['phase'] for d in recent_data]
        
        return {
            'amplitude_mean': np.mean(amplitudes),
            'amplitude_std': np.std(amplitudes),
            'amplitude_stability': np.std(amplitudes) / np.mean(amplitudes),
            'frequency_mean': np.mean(frequencies),
            'frequency_std': np.std(frequencies),
            'frequency_stability': np.std(frequencies) / np.mean(frequencies),
            'phase_std': np.std(phases),
            'data_points': len(recent_data),
            'measurement_duration': recent_data[-1]['timestamp'] - recent_data[0]['timestamp']
        }
    
    def shutdown(self):
        """Graceful system shutdown"""
        print("\n🔌 Shutting down field monitoring system...")
        
        # Stop all threads
        self.monitoring_active = False
        self.control_active = False
        self.stop_threads.set()
        
        # Wait for threads to finish
        threads = [self.monitoring_thread, self.control_thread, self.optimization_thread]
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=1.0)
        
        # Zero all fields
        self.current_field_amplitude = 0.0
        self.target_field_amplitude = 0.0
        
        print("✅ Field monitoring system shutdown complete")

def main():
    """Main field monitoring demonstration"""
    print("🔬 Real-Time Field Monitoring System Demonstration")
    print("=" * 60)
    
    # Create monitoring system
    monitor = RealTimeFieldMonitor()
    
    try:
        # Test sequence
        print("\n🧪 Running Field Monitoring Tests...")
        print("=" * 40)
        
        # Test 1: Set target field
        print("Test 1: Setting target field parameters...")
        success = monitor.set_target_field(
            amplitude=1e16,    # 10^16 V/m
            frequency=1e12,    # 1 THz
            phase=0.0
        )
        
        if success:
            # Let control system stabilize
            time.sleep(0.1)
            
            # Check field statistics
            stats = monitor.get_field_statistics()
            if stats:
                print(f"   Field Stability: {stats['amplitude_stability']*100:.3f}%")
                print(f"   Frequency Stability: {stats['frequency_stability']*100:.4f}%")
                print(f"   Phase Jitter: {stats['phase_std']*1000:.1f} mrad")
        
        # Test 2: Spatial field mapping
        print("\nTest 2: Spatial field mapping...")
        positions = [
            np.array([0, 0, 0]),
            np.array([100e-6, 0, 0]),
            np.array([0, 100e-6, 0]),
            np.array([100e-6, 100e-6, 0])
        ]
        
        for pos in positions:
            field_data = monitor.measure_field_at_position(pos)
            print(f"   Position ({pos[0]*1e6:.0f}, {pos[1]*1e6:.0f}) μm: "
                  f"{field_data['field_magnitude']/1e15:.2f} × 10^15 V/m")
        
        # Test 3: Control system response
        print("\nTest 3: Testing control response...")
        
        # Change target amplitude
        monitor.set_target_field(2e16, 1e12, 0.0)
        time.sleep(0.05)
        
        # Check control history
        if monitor.control_history:
            recent_control = monitor.control_history[-10:]
            avg_error = np.mean([abs(c['amplitude_error']) for c in recent_control])
            print(f"   Average control error: {avg_error/1e15:.3f} × 10^15 V/m")
            
            if avg_error < 1e14:  # < 0.1 × 10^15 V/m
                print("   ✅ Control system performing well")
            else:
                print("   ⚠️  Control system needs tuning")
        
        # Final statistics
        print("\n📊 Final System Performance:")
        final_stats = monitor.get_field_statistics()
        if final_stats:
            print(f"   Amplitude Stability: {final_stats['amplitude_stability']*100:.3f}%")
            print(f"   Frequency Stability: {final_stats['frequency_stability']*100:.4f}%")
            print(f"   Total Data Points: {final_stats['data_points']:,}")
            print(f"   Measurement Duration: {final_stats['measurement_duration']:.3f} s")
    
    except KeyboardInterrupt:
        print("\n⚠️  User interrupt detected")
        monitor.trigger_emergency_stop("User interrupt")
    
    except Exception as e:
        print(f"\n❌ System error: {e}")
        monitor.trigger_emergency_stop(f"Software error: {e}")
    
    finally:
        # Graceful shutdown
        monitor.shutdown()

if __name__ == "__main__":
    main()
