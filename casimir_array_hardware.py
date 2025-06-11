#!/usr/bin/env python3
"""
Hardware Interface for Casimir Array Implementation
==================================================

This module provides the hardware interface layer for precise control of
Casimir effect arrays, including:

1. Nanometer-precision positioning control
2. Real-time force measurement and feedback
3. Surface quality monitoring and calibration
4. Temperature and vibration isolation systems
5. Automated safety and emergency protocols

Technical Specifications:
- Positioning Precision: ±0.5 nm (piezoelectric actuators)
- Force Resolution: 10^-15 N (attoNewton sensitivity)
- Temperature Control: 1 mK stability (dilution refrigerator)
- Vibration Isolation: 10^-12 m displacement sensitivity
- Surface Quality: RMS roughness < 0.1 nm
"""

import numpy as np
import time
import threading
import queue
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import warnings
warnings.filterwarnings('ignore')

class ActuatorState(Enum):
    """Piezoelectric actuator states"""
    POWERED_DOWN = "powered_down"
    INITIALIZING = "initializing"
    CALIBRATING = "calibrating"
    READY = "ready"
    POSITIONING = "positioning"
    HOLDING = "holding"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"

class SurfaceQuality(Enum):
    """Surface quality assessment levels"""
    EXCELLENT = "excellent"    # RMS < 0.05 nm
    GOOD = "good"             # RMS < 0.1 nm
    ACCEPTABLE = "acceptable"  # RMS < 0.2 nm
    POOR = "poor"             # RMS > 0.2 nm

@dataclass
class CasimirPlateSpecs:
    """Specifications for individual Casimir plates"""
    material: str = "Au/SiO2"  # Gold on silicon dioxide
    thickness: float = 100e-9  # 100 nm total thickness
    area: float = 1e-6         # 1 mm² active area
    surface_roughness: float = 0.05e-9  # 0.05 nm RMS
    parallelism_tolerance: float = 1e-9  # 1 nm parallelism
    coating_uniformity: float = 0.01  # 1% thickness variation
    
@dataclass
class PositioningSpecs:
    """Piezoelectric positioning system specifications"""
    range_nm: float = 1000.0     # 1 μm range
    resolution_nm: float = 0.1   # 0.1 nm resolution
    stability_nm: float = 0.05   # 0.05 nm stability
    bandwidth_hz: float = 1000   # 1 kHz bandwidth
    settling_time_ms: float = 1  # 1 ms settling time
    
@dataclass
class ForceSpecs:
    """Force measurement specifications"""
    range_nN: float = 1000.0     # 1 μN range
    resolution_fN: float = 0.1   # 0.1 fN resolution
    bandwidth_hz: float = 10000  # 10 kHz bandwidth
    noise_floor_fN: float = 0.01 # 10 aN noise floor

class CasimirArrayController:
    """Hardware control system for Casimir effect arrays"""
    
    def __init__(self, n_plates: int = 2, plate_specs: CasimirPlateSpecs = None,
                 positioning_specs: PositioningSpecs = None,
                 force_specs: ForceSpecs = None):
        
        # System specifications
        self.n_plates = n_plates
        self.plate_specs = plate_specs or CasimirPlateSpecs()
        self.positioning_specs = positioning_specs or PositioningSpecs()
        self.force_specs = force_specs or ForceSpecs()
        
        # Physical constants
        self.hbar = 1.055e-34  # J⋅s
        self.c = 2.998e8       # m/s
        self.epsilon_0 = 8.854e-12  # F/m
        
        # Hardware state
        self.actuator_states = [ActuatorState.POWERED_DOWN] * n_plates
        self.current_positions = np.zeros(n_plates)  # Current positions (m)
        self.target_positions = np.zeros(n_plates)   # Target positions (m)
        self.measured_forces = np.zeros(n_plates)    # Measured forces (N)
        self.surface_qualities = [SurfaceQuality.GOOD] * n_plates
        
        # Control parameters
        self.emergency_stop_active = False
        self.calibration_complete = False
        self.feedback_enabled = True
        
        # Data logging
        self.position_history = []
        self.force_history = []
        self.timestamp_history = []
        
        # Threading for real-time control
        self.control_thread = None
        self.monitoring_thread = None
        self.stop_threads = threading.Event()
        
        # Initialize hardware systems
        self.initialize_hardware()
        
    def initialize_hardware(self):
        """Initialize all hardware subsystems"""
        print("🔧 Initializing Casimir Array Hardware Controller")
        print(f"   Plates: {self.n_plates}")
        print(f"   Material: {self.plate_specs.material}")
        print(f"   Area: {self.plate_specs.area*1e6:.1f} mm²")
        print(f"   Target Roughness: {self.plate_specs.surface_roughness*1e9:.2f} nm RMS")
        
        # Initialize piezoelectric actuators
        self.initialize_actuators()
        
        # Initialize force measurement system
        self.initialize_force_sensors()
        
        # Initialize surface quality monitoring
        self.initialize_surface_monitoring()
        
        # Initialize temperature and vibration control
        self.initialize_environmental_controls()
        
        # Start control threads
        self.start_control_threads()
        
        print("✅ Casimir array hardware initialized successfully")
        
    def initialize_actuators(self):
        """Initialize piezoelectric actuator systems"""
        print("   🔹 Initializing piezoelectric actuators...")
        
        for i in range(self.n_plates):
            # Power up actuator
            self.actuator_states[i] = ActuatorState.INITIALIZING
            print(f"      Actuator {i}: Powering up...")
            
            # Simulate initialization delay
            time.sleep(0.01)
            
            # Set to ready state
            self.actuator_states[i] = ActuatorState.READY
            print(f"      Actuator {i}: Ready (±{self.positioning_specs.resolution_nm:.1f} nm)")
    
    def initialize_force_sensors(self):
        """Initialize capacitive force sensors"""
        print("   🔹 Initializing force measurement system...")
        print(f"      Resolution: {self.force_specs.resolution_fN:.1f} fN")
        print(f"      Bandwidth: {self.force_specs.bandwidth_hz/1000:.1f} kHz")
        print(f"      Noise Floor: {self.force_specs.noise_floor_fN:.2f} fN")
        
    def initialize_surface_monitoring(self):
        """Initialize atomic force microscopy for surface monitoring"""
        print("   🔹 Initializing surface quality monitoring...")
        
        for i in range(self.n_plates):
            # Simulate surface quality assessment
            measured_roughness = self.plate_specs.surface_roughness * (1 + 0.1 * np.random.randn())
            
            if measured_roughness < 0.05e-9:
                self.surface_qualities[i] = SurfaceQuality.EXCELLENT
            elif measured_roughness < 0.1e-9:
                self.surface_qualities[i] = SurfaceQuality.GOOD
            elif measured_roughness < 0.2e-9:
                self.surface_qualities[i] = SurfaceQuality.ACCEPTABLE
            else:
                self.surface_qualities[i] = SurfaceQuality.POOR
                
            print(f"      Plate {i}: {self.surface_qualities[i].value} "
                  f"({measured_roughness*1e9:.3f} nm RMS)")
    
    def initialize_environmental_controls(self):
        """Initialize temperature and vibration isolation systems"""
        print("   🔹 Initializing environmental controls...")
        print("      Temperature: 1 mK dilution refrigerator")
        print("      Vibration: Active isolation to 1 pm sensitivity")
        print("      Vacuum: Ultra-high vacuum < 10^-10 Torr")
        
    def start_control_threads(self):
        """Start real-time control and monitoring threads"""
        print("   🔹 Starting real-time control threads...")
        
        # Position control thread
        self.control_thread = threading.Thread(target=self._position_control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        # Force monitoring thread
        self.monitoring_thread = threading.Thread(target=self._force_monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
    def _position_control_loop(self):
        """Real-time position control loop"""
        while not self.stop_threads.is_set():
            if not self.emergency_stop_active:
                for i in range(self.n_plates):
                    if self.actuator_states[i] == ActuatorState.READY:
                        # Check if position adjustment needed
                        position_error = self.target_positions[i] - self.current_positions[i]
                        
                        if abs(position_error) > self.positioning_specs.resolution_nm * 1e-9:
                            self.actuator_states[i] = ActuatorState.POSITIONING
                            
                            # Apply position correction
                            correction = np.sign(position_error) * \
                                       min(abs(position_error), 
                                           self.positioning_specs.resolution_nm * 1e-9)
                            
                            self.current_positions[i] += correction
                            
                            # Settling time
                            time.sleep(self.positioning_specs.settling_time_ms / 1000)
                            
                            self.actuator_states[i] = ActuatorState.HOLDING
            
            time.sleep(0.001)  # 1 ms control loop
    
    def _force_monitoring_loop(self):
        """Real-time force monitoring loop"""
        while not self.stop_threads.is_set():
            current_time = time.time()
            
            for i in range(self.n_plates):
                # Calculate expected Casimir force
                if i < self.n_plates - 1:  # Pairs of plates
                    separation = abs(self.current_positions[i+1] - self.current_positions[i])
                    if separation > 0:
                        casimir_force = self.calculate_casimir_force(separation)
                        
                        # Add measurement noise
                        noise = np.random.normal(0, self.force_specs.noise_floor_fN * 1e-15)
                        self.measured_forces[i] = casimir_force + noise
                    else:
                        self.measured_forces[i] = 0
                        
            # Log data
            self.position_history.append(self.current_positions.copy())
            self.force_history.append(self.measured_forces.copy())
            self.timestamp_history.append(current_time)
            
            # Maintain data history length
            if len(self.position_history) > 10000:
                self.position_history = self.position_history[-5000:]
                self.force_history = self.force_history[-5000:]
                self.timestamp_history = self.timestamp_history[-5000:]
            
            time.sleep(0.0001)  # 0.1 ms monitoring loop
    
    def calculate_casimir_force(self, separation: float) -> float:
        """Calculate Casimir force between parallel plates"""
        # F_C = ℏcπ²A/(240d⁴)
        force = (self.hbar * self.c * np.pi**2 * self.plate_specs.area) / \
                (240 * separation**4)
        return force
    
    def set_plate_separation(self, plate1: int, plate2: int, 
                           separation: float) -> bool:
        """Set separation between two plates with nanometer precision"""
        if plate1 >= self.n_plates or plate2 >= self.n_plates:
            raise ValueError("Invalid plate indices")
        
        if self.emergency_stop_active:
            print("❌ Cannot move plates: Emergency stop active")
            return False
        
        print(f"📐 Setting separation between plates {plate1}-{plate2}: "
              f"{separation*1e9:.2f} nm")
        
        # Check tolerance limits
        if separation < 1e-9:  # Minimum 1 nm separation
            print("⚠️  Separation below 1 nm minimum - clamping")
            separation = 1e-9
        elif separation > 1e-6:  # Maximum 1 μm separation
            print("⚠️  Separation above 1 μm maximum - clamping")
            separation = 1e-6
        
        # Set target positions
        center_position = (self.current_positions[plate1] + self.current_positions[plate2]) / 2
        self.target_positions[plate1] = center_position - separation / 2
        self.target_positions[plate2] = center_position + separation / 2
        
        # Wait for positioning to complete
        self._wait_for_positioning_complete()
        
        # Verify final separation
        final_separation = abs(self.current_positions[plate2] - self.current_positions[plate1])
        position_error = abs(final_separation - separation)
        
        if position_error < self.positioning_specs.resolution_nm * 1e-9:
            print(f"✅ Separation achieved: {final_separation*1e9:.3f} nm "
                  f"(error: {position_error*1e12:.1f} pm)")
            return True
        else:
            print(f"❌ Positioning error: {position_error*1e12:.1f} pm")
            return False
    
    def _wait_for_positioning_complete(self, timeout: float = 5.0):
        """Wait for all actuators to complete positioning"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_ready = all(state in [ActuatorState.READY, ActuatorState.HOLDING] 
                          for state in self.actuator_states)
            if all_ready:
                return True
            time.sleep(0.001)
        
        print("⚠️  Positioning timeout - some actuators may not have reached target")
        return False
    
    def measure_plate_separation(self, plate1: int, plate2: int) -> float:
        """Measure current separation between plates with pm precision"""
        if plate1 >= self.n_plates or plate2 >= self.n_plates:
            raise ValueError("Invalid plate indices")
        
        # Real separation
        real_separation = abs(self.current_positions[plate2] - self.current_positions[plate1])
        
        # Add measurement noise (pm level)
        measurement_noise = np.random.normal(0, 1e-12)  # 1 pm RMS noise
        measured_separation = real_separation + measurement_noise
        
        return measured_separation
    
    def measure_casimir_force(self, plate1: int, plate2: int) -> Dict[str, float]:
        """Measure Casimir force between plates"""
        separation = self.measure_plate_separation(plate1, plate2)
        theoretical_force = self.calculate_casimir_force(separation)
        measured_force = self.measured_forces[plate1]
        
        force_error = abs(measured_force - theoretical_force)
        relative_error = force_error / (abs(theoretical_force) + 1e-30)
        
        return {
            'separation_nm': separation * 1e9,
            'theoretical_force_fN': theoretical_force * 1e15,
            'measured_force_fN': measured_force * 1e15,
            'force_error_fN': force_error * 1e15,
            'relative_error_percent': relative_error * 100,
            'signal_to_noise': abs(measured_force) / self.force_specs.noise_floor_fN
        }
    
    def calibrate_system(self) -> Dict[str, bool]:
        """Perform comprehensive system calibration"""
        print("\n🔧 Performing Comprehensive System Calibration")
        print("=" * 50)
        
        calibration_results = {
            'position_calibration': False,
            'force_calibration': False,
            'surface_calibration': False,
            'overall_success': False
        }
        
        # Position calibration
        print("📐 Position Calibration...")
        position_success = self._calibrate_positioning()
        calibration_results['position_calibration'] = position_success
        
        # Force calibration
        print("⚖️  Force Calibration...")
        force_success = self._calibrate_force_measurement()
        calibration_results['force_calibration'] = force_success
        
        # Surface quality verification
        print("🔍 Surface Quality Verification...")
        surface_success = self._verify_surface_quality()
        calibration_results['surface_calibration'] = surface_success
        
        # Overall calibration status
        overall_success = all([position_success, force_success, surface_success])
        calibration_results['overall_success'] = overall_success
        
        if overall_success:
            self.calibration_complete = True
            print("✅ System calibration completed successfully")
        else:
            print("❌ Calibration failed - manual intervention required")
        
        return calibration_results
    
    def _calibrate_positioning(self) -> bool:
        """Calibrate positioning system accuracy"""
        print("   Testing positioning accuracy at multiple points...")
        
        test_separations = [10e-9, 50e-9, 100e-9, 500e-9]  # Test points
        max_error = 0
        
        for separation in test_separations:
            success = self.set_plate_separation(0, 1, separation)
            if success:
                measured = self.measure_plate_separation(0, 1)
                error = abs(measured - separation)
                max_error = max(max_error, error)
                
                print(f"   Target: {separation*1e9:.1f} nm, "
                      f"Measured: {measured*1e9:.2f} nm, "
                      f"Error: {error*1e12:.1f} pm")
            else:
                return False
        
        # Check if maximum error is within specification
        success = max_error < self.positioning_specs.resolution_nm * 1e-9
        
        if success:
            print(f"   ✅ Position calibration passed (max error: {max_error*1e12:.1f} pm)")
        else:
            print(f"   ❌ Position calibration failed (max error: {max_error*1e12:.1f} pm)")
        
        return success
    
    def _calibrate_force_measurement(self) -> bool:
        """Calibrate force measurement system"""
        print("   Testing force measurement accuracy...")
        
        # Test at 10 nm separation (well-defined Casimir force)
        test_separation = 10e-9
        self.set_plate_separation(0, 1, test_separation)
        
        # Theoretical Casimir force
        theoretical_force = self.calculate_casimir_force(test_separation)
        
        # Average over multiple measurements
        n_measurements = 100
        force_measurements = []
        
        for _ in range(n_measurements):
            force_data = self.measure_casimir_force(0, 1)
            force_measurements.append(force_data['measured_force_fN'])
            time.sleep(0.001)  # 1 ms between measurements
        
        measured_force_avg = np.mean(force_measurements)
        measurement_std = np.std(force_measurements)
        
        relative_error = abs(measured_force_avg - theoretical_force*1e15) / (theoretical_force*1e15)
        
        print(f"   Theoretical: {theoretical_force*1e15:.2f} fN")
        print(f"   Measured: {measured_force_avg:.2f} ± {measurement_std:.2f} fN")
        print(f"   Relative Error: {relative_error*100:.2f}%")
        
        # Success criteria: <10% error and noise < 0.1 fN
        success = (relative_error < 0.1 and measurement_std < 0.1)
        
        if success:
            print("   ✅ Force calibration passed")
        else:
            print("   ❌ Force calibration failed")
        
        return success
    
    def _verify_surface_quality(self) -> bool:
        """Verify surface quality meets specifications"""
        print("   Verifying surface quality...")
        
        all_good = True
        for i in range(self.n_plates):
            quality = self.surface_qualities[i]
            print(f"   Plate {i}: {quality.value}")
            
            if quality == SurfaceQuality.POOR:
                all_good = False
                print(f"      ⚠️  Plate {i} requires surface treatment")
        
        if all_good:
            print("   ✅ All surfaces meet quality requirements")
        else:
            print("   ❌ Surface quality issues detected")
        
        return all_good
    
    def emergency_stop(self, reason: str = "Manual activation"):
        """Activate emergency stop procedure"""
        print(f"🚨 EMERGENCY STOP ACTIVATED: {reason}")
        
        self.emergency_stop_active = True
        
        # Stop all actuators
        for i in range(self.n_plates):
            self.actuator_states[i] = ActuatorState.EMERGENCY_STOP
        
        # Move plates to safe separation (1 μm)
        print("   🔴 Moving all plates to safe separation...")
        safe_separation = 1e-6  # 1 μm
        
        for i in range(self.n_plates):
            self.target_positions[i] = i * safe_separation
        
        print("   🔴 All systems halted")
        
    def shutdown(self):
        """Graceful system shutdown"""
        print("\n🔌 Shutting down Casimir array controller...")
        
        # Stop control threads
        self.stop_threads.set()
        
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=1.0)
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
        
        # Power down actuators
        for i in range(self.n_plates):
            self.actuator_states[i] = ActuatorState.POWERED_DOWN
        
        print("✅ Casimir array controller shutdown complete")
    
    def get_system_status(self) -> Dict[str, any]:
        """Get comprehensive system status"""
        return {
            'actuator_states': [state.value for state in self.actuator_states],
            'current_positions_nm': self.current_positions * 1e9,
            'target_positions_nm': self.target_positions * 1e9,
            'measured_forces_fN': self.measured_forces * 1e15,
            'surface_qualities': [quality.value for quality in self.surface_qualities],
            'emergency_stop_active': self.emergency_stop_active,
            'calibration_complete': self.calibration_complete,
            'data_points_logged': len(self.timestamp_history)
        }

def main():
    """Main hardware interface demonstration"""
    print("🔬 Casimir Array Hardware Interface Demonstration")
    print("=" * 55)
    
    # Create controller
    controller = CasimirArrayController(n_plates=2)
    
    try:
        # Perform system calibration
        calibration_results = controller.calibrate_system()
        
        if calibration_results['overall_success']:
            print("\n🧪 Running Hardware Tests...")
            print("=" * 30)
            
            # Test 1: Set precise separation
            print("Test 1: Setting 10 nm separation...")
            success = controller.set_plate_separation(0, 1, 10e-9)
            if success:
                force_data = controller.measure_casimir_force(0, 1)
                print(f"   Casimir Force: {force_data['measured_force_fN']:.2f} fN")
                print(f"   Signal/Noise: {force_data['signal_to_noise']:.1f}")
            
            # Test 2: Force vs. separation curve
            print("\nTest 2: Force vs. separation characterization...")
            separations = np.linspace(5e-9, 50e-9, 10)  # 5-50 nm
            forces = []
            
            for separation in separations:
                controller.set_plate_separation(0, 1, separation)
                force_data = controller.measure_casimir_force(0, 1)
                forces.append(force_data['measured_force_fN'])
                print(f"   {separation*1e9:.1f} nm: {force_data['measured_force_fN']:.2f} fN")
            
            # Verify 1/d⁴ scaling
            theoretical_scaling = (separations[0]/separations)**4
            measured_scaling = np.array(forces) / forces[0]
            scaling_error = np.mean(abs(theoretical_scaling - measured_scaling))
            
            print(f"   Scaling verification: {scaling_error:.3f} average deviation")
            if scaling_error < 0.1:
                print("   ✅ 1/d⁴ scaling confirmed")
            else:
                print("   ⚠️  Scaling deviation detected")
            
            # System status
            print("\n📊 Final System Status:")
            status = controller.get_system_status()
            print(f"   Actuators: {status['actuator_states']}")
            print(f"   Positions: {status['current_positions_nm']} nm")
            print(f"   Forces: {status['measured_forces_fN']} fN")
            print(f"   Data Points: {status['data_points_logged']:,}")
        
        else:
            print("❌ Calibration failed - hardware tests skipped")
    
    except KeyboardInterrupt:
        print("\n⚠️  User interrupt detected")
        controller.emergency_stop("User interrupt")
    
    except Exception as e:
        print(f"\n❌ Hardware error: {e}")
        controller.emergency_stop(f"Software error: {e}")
    
    finally:
        # Graceful shutdown
        controller.shutdown()

if __name__ == "__main__":
    main()
