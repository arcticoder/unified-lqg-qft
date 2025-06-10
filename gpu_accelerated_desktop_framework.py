#!/usr/bin/env python3
"""
GPU-Accelerated Desktop 3D Replicator Framework
==============================================

Enhanced version with proper GPU detection, utilization, and monitoring.
Designed to maximize both CPU and GPU resources on high-end desktop systems.
"""

import time
import json
import numpy as np
import multiprocessing as mp
import psutil
from typing import Dict, Tuple, List, Any, Optional
from pathlib import Path

# GPU monitoring imports
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("GPUtil not available - install with: pip install GPUtil")

# JAX with GPU support
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, pmap
    from jax.config import config
    # Enable GPU if available
    jax.config.update('jax_enable_x64', True)
    JAX_AVAILABLE = True
    
    # Check for GPU devices
    gpu_devices = [d for d in jax.devices() if 'gpu' in str(d).lower()]
    GPU_AVAILABLE = len(gpu_devices) > 0
    
    if GPU_AVAILABLE:
        print(f"🎮 JAX GPU detected: {len(gpu_devices)} GPU(s)")
    else:
        print("🖥️  JAX CPU-only mode")
        
except ImportError:
    import numpy as jnp
    JAX_AVAILABLE = False
    GPU_AVAILABLE = False
    print("JAX not available - install with: pip install jax[gpu] or jax[cpu]")

class GPUAccelerated3DReplicator:
    """GPU-accelerated 3D replicator with comprehensive monitoring"""
    
    def __init__(self, grid_size=80, extent=3.0, force_gpu=True):
        self.N = grid_size
        self.L = extent
        self.dx = 2 * self.L / (self.N - 1)
        self.force_gpu = force_gpu
        
        # Hardware detection
        self.detect_hardware()
        
        # Configure JAX device usage
        self.configure_compute_devices()
        
        print(f"🚀 GPU-Accelerated Desktop 3D Replicator")
        print(f"=" * 55)
        print(f"Hardware: {self.cpu_cores} cores, {self.total_ram_gb:.1f} GB RAM")
        if self.gpu_available:
            print(f"GPU: {self.gpu_info}")
        print(f"Compute: {self.compute_device}")
        print(f"Grid: {self.N}³ = {self.N**3:,} points")
        print(f"Memory: {(self.N**3 * 8 * 8) / (1024**3):.3f} GB estimated")
        
        # Setup computation
        self.setup_gpu_grid()
        self.initialize_gpu_fields()
        
        # Enhanced metrics with GPU monitoring
        self.metrics = {
            'step_times': [],
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'gpu_memory': [],
            'throughput': [],
            'field_evolution': [],
            'gpu_efficiency': []
        }
    
    def detect_hardware(self):
        """Comprehensive hardware detection including GPU"""
        # CPU and RAM
        self.cpu_cores = mp.cpu_count()
        memory = psutil.virtual_memory()
        self.total_ram_gb = memory.total / (1024**3)
        self.available_ram_gb = memory.available / (1024**3)
        
        # GPU detection
        self.gpu_available = False
        self.gpu_info = "None detected"
        self.gpu_memory_gb = 0
        
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.gpu_available = True
                    gpu = gpus[0]  # Use first GPU
                    self.gpu_info = f"{gpu.name} ({gpu.memoryTotal}MB)"
                    self.gpu_memory_gb = gpu.memoryTotal / 1024
                    self.gpu_id = gpu.id
            except:
                pass
        
        # JAX GPU verification
        if JAX_AVAILABLE and GPU_AVAILABLE:
            self.jax_gpu_available = True
            self.jax_devices = jax.devices()
            self.jax_gpu_devices = [d for d in self.jax_devices if 'gpu' in str(d).lower()]
        else:
            self.jax_gpu_available = False
            self.jax_gpu_devices = []
    
    def configure_compute_devices(self):
        """Configure optimal compute device usage"""
        if self.force_gpu and self.jax_gpu_available:
            # Force GPU usage
            self.compute_device = "GPU"
            self.primary_device = self.jax_gpu_devices[0]
            print(f"🎮 Using GPU acceleration: {self.primary_device}")
        elif JAX_AVAILABLE:
            # Use CPU with JAX
            self.compute_device = "JAX CPU"
            self.primary_device = jax.devices('cpu')[0]
            print(f"🖥️  Using JAX CPU acceleration")
        else:
            # Pure NumPy fallback
            self.compute_device = "NumPy CPU"
            self.primary_device = None
            print(f"⚙️  Using NumPy (no JAX)")
    
    def monitor_gpu_usage(self):
        """Monitor GPU utilization and memory"""
        gpu_usage = 0.0
        gpu_memory = 0.0
        
        if GPUTIL_AVAILABLE and self.gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_usage = gpu.load * 100  # Convert to percentage
                    gpu_memory = (gpu.memoryUsed / gpu.memoryTotal) * 100
            except:
                pass
        
        return gpu_usage, gpu_memory
    
    def setup_gpu_grid(self):
        """Setup 3D grid with GPU optimization"""
        print(f"🔄 Creating {self.N}³ GPU-optimized grid...")
        start_time = time.time()
        
        # Create coordinate arrays
        x = np.linspace(-self.L, self.L, self.N)
        y = np.linspace(-self.L, self.L, self.N)
        z = np.linspace(-self.L, self.L, self.N)
        
        # Use JAX for GPU acceleration if available
        if self.compute_device == "GPU":
            with jax.default_device(self.primary_device):
                x_jax = jnp.array(x)
                y_jax = jnp.array(y)
                z_jax = jnp.array(z)
                
                X, Y, Z = jnp.meshgrid(x_jax, y_jax, z_jax, indexing='ij')
                self.grid_coords = jnp.stack([X, Y, Z], axis=-1)
                self.r_3d = jnp.sqrt(X**2 + Y**2 + Z**2)
        else:
            # CPU implementation
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            self.grid_coords = jnp.array(np.stack([X, Y, Z], axis=-1))
            self.r_3d = jnp.array(np.sqrt(X**2 + Y**2 + Z**2))
        
        setup_time = time.time() - start_time
        print(f"Grid setup: {setup_time:.2f} seconds")
        print(f"Grid memory: {self.grid_coords.nbytes / (1024**3):.3f} GB")
        
        # Monitor GPU after grid creation
        gpu_usage, gpu_memory = self.monitor_gpu_usage()
        if self.gpu_available:
            print(f"GPU usage: {gpu_usage:.1f}%, GPU memory: {gpu_memory:.1f}%")
    
    def initialize_gpu_fields(self):
        """Initialize fields with GPU acceleration"""
        print(f"🔄 Initializing {self.N}³ GPU fields...")
        
        # Physics parameters
        self.lambda_coupling = 0.002
        self.mu_polymer = 0.12
        self.alpha_enhancement = 0.03
        self.R0_scale = 2.5
        self.M_mass = 1.0
        
        # Create initial fields on GPU if available
        if self.compute_device == "GPU":
            with jax.default_device(self.primary_device):
                # Use JAX random number generation
                key = jax.random.PRNGKey(42)
                key1, key2 = jax.random.split(key)
                
                # Smooth Gaussian initialization
                center = self.N // 2
                i, j, k = jnp.meshgrid(jnp.arange(self.N), jnp.arange(self.N), jnp.arange(self.N), indexing='ij')
                gaussian_weight = jnp.exp(-((i - center)**2 + (j - center)**2 + (k - center)**2) / (self.N/4)**2)
                
                # Initialize on GPU
                self.phi = (5e-5 * gaussian_weight + 
                           1e-6 * jax.random.normal(key1, shape=(self.N, self.N, self.N)))
                self.pi = (1e-6 * gaussian_weight + 
                          1e-7 * jax.random.normal(key2, shape=(self.N, self.N, self.N)))
        else:
            # CPU initialization
            np.random.seed(42)
            center = self.N // 2
            i, j, k = np.meshgrid(range(self.N), range(self.N), range(self.N), indexing='ij')
            gaussian_weight = np.exp(-((i - center)**2 + (j - center)**2 + (k - center)**2) / (self.N/4)**2)
            
            self.phi = jnp.array(5e-5 * gaussian_weight + 1e-6 * np.random.normal(size=(self.N, self.N, self.N)))
            self.pi = jnp.array(1e-6 * gaussian_weight + 1e-7 * np.random.normal(size=(self.N, self.N, self.N)))
        
        # Compute initial geometry
        self.f3d = self.compute_gpu_metric()
        self.R3d = self.compute_gpu_ricci()
        
        print(f"Initial field statistics:")
        print(f"  φ: mean = {jnp.mean(self.phi):.2e}, std = {jnp.std(self.phi):.2e}")
        print(f"  π: mean = {jnp.mean(self.pi):.2e}, std = {jnp.std(self.pi):.2e}")
        print(f"  f3d: range = [{jnp.min(self.f3d):.3f}, {jnp.max(self.f3d):.3f}]")
        print(f"  R3d: range = [{jnp.min(self.R3d):.3f}, {jnp.max(self.R3d):.3f}]")
        
        # Check GPU usage after initialization
        gpu_usage, gpu_memory = self.monitor_gpu_usage()
        if self.gpu_available:
            print(f"GPU after init: {gpu_usage:.1f}% usage, {gpu_memory:.1f}% memory")
    
    @jit
    def compute_gpu_metric(self):
        """GPU-accelerated metric computation"""
        r_safe = jnp.maximum(self.r_3d, 0.1)
        
        # LQG component
        f_lqg = (1 - 2*self.M_mass/r_safe + 
                (self.mu_polymer**2 * self.M_mass**2)/(6 * r_safe**4))
        
        # Gaussian enhancement
        gaussian = (self.alpha_enhancement * 
                   jnp.exp(-(self.r_3d/self.R0_scale)**2))
        
        # Apply bounds
        f_total = f_lqg + gaussian
        return jnp.clip(f_total, 0.1, 8.0)
    
    @jit
    def compute_gpu_laplacian(self, field):
        """GPU-accelerated 3D Laplacian computation"""
        dx2 = self.dx**2
        
        # X-direction
        d2_dx2 = jnp.zeros_like(field)
        d2_dx2 = d2_dx2.at[1:-1, :, :].set(
            (field[2:, :, :] - 2*field[1:-1, :, :] + field[:-2, :, :]) / dx2
        )
        
        # Y-direction
        d2_dy2 = jnp.zeros_like(field)
        d2_dy2 = d2_dy2.at[:, 1:-1, :].set(
            (field[:, 2:, :] - 2*field[:, 1:-1, :] + field[:, :-2, :]) / dx2
        )
        
        # Z-direction
        d2_dz2 = jnp.zeros_like(field)
        d2_dz2 = d2_dz2.at[:, :, 1:-1].set(
            (field[:, :, 2:] - 2*field[:, :, 1:-1] + field[:, :, :-2]) / dx2
        )
        
        return d2_dx2 + d2_dy2 + d2_dz2
    
    @jit 
    def compute_gpu_ricci(self):
        """GPU-accelerated Ricci scalar computation"""
        laplacian_phi = self.compute_gpu_laplacian(self.phi)
        coupling_term = self.lambda_coupling * self.phi * laplacian_phi
        return jnp.clip(coupling_term, -2.0, 2.0)
    
    @jit
    def gpu_evolution_step(self, phi, pi, f3d, dt):
        """GPU-accelerated evolution step"""
        # Compute Laplacian
        laplacian_phi = self.compute_gpu_laplacian(phi)
        
        # Update momentum
        source_term = self.lambda_coupling * laplacian_phi + f3d * phi
        source_term = jnp.clip(source_term, -5.0, 5.0)
        
        pi_new = pi + dt * source_term
        
        # Update field
        phi_new = phi + dt * pi_new
        
        # Apply bounds
        phi_new = jnp.clip(phi_new, -0.02, 0.02)
        pi_new = jnp.clip(pi_new, -0.02, 0.02)
        
        return phi_new, pi_new
    
    def run_gpu_accelerated_simulation(self, total_steps=2000, dt=0.0005, report_interval=200):
        """Run GPU-accelerated simulation with comprehensive monitoring"""
        print(f"\\n🚀 Starting GPU-Accelerated Desktop Simulation")
        print(f"Device: {self.compute_device}")
        print(f"Steps: {total_steps}, dt = {dt}")
        
        start_time = time.time()
        
        for step in range(total_steps):
            step_start = time.time()
            
            # Evolution step (GPU accelerated if available)
            if self.compute_device == "GPU":
                self.phi, self.pi = self.gpu_evolution_step(self.phi, self.pi, self.f3d, dt)
                # Recompute geometry on GPU
                self.f3d = self.compute_gpu_metric()
                self.R3d = self.compute_gpu_ricci()
            else:
                # CPU evolution step
                self.evolution_step_cpu(dt)
            
            # Comprehensive monitoring
            step_time = time.time() - step_start
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            gpu_usage, gpu_memory = self.monitor_gpu_usage()
            
            # Calculate throughput
            throughput = self.N**3 / step_time
            
            # Record metrics
            self.metrics['step_times'].append(step_time)
            self.metrics['cpu_usage'].append(cpu_percent)
            self.metrics['memory_usage'].append(memory_percent)
            self.metrics['gpu_usage'].append(gpu_usage)
            self.metrics['gpu_memory'].append(gpu_memory)
            self.metrics['throughput'].append(throughput)
            
            # Field evolution tracking
            phi_rms = float(jnp.sqrt(jnp.mean(self.phi**2)))
            self.metrics['field_evolution'].append(phi_rms)
            
            # GPU efficiency calculation
            if self.gpu_available and gpu_usage > 0:
                gpu_efficiency = min(100, gpu_usage / 100 * 100)
            else:
                gpu_efficiency = 0
            self.metrics['gpu_efficiency'].append(gpu_efficiency)
            
            # Progress reporting
            if step % report_interval == 0:
                print(f"  Step {step:4d}: φ_rms = {phi_rms:.2e}, "
                      f"time = {step_time*1000:.1f}ms, "
                      f"throughput = {throughput:,.0f} pts/s")
                if self.gpu_available:
                    print(f"             GPU: {gpu_usage:.1f}% usage, {gpu_memory:.1f}% memory")
                print(f"             CPU: {cpu_percent:.1f}%, RAM: {memory_percent:.1f}%")
        
        total_time = time.time() - start_time
        
        # Comprehensive results
        print(f"\\n✅ GPU-accelerated simulation completed!")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Average step time: {np.mean(self.metrics['step_times'])*1000:.2f} ms")
        print(f"Average throughput: {np.mean(self.metrics['throughput']):,.0f} points/second")
        print(f"Peak memory usage: {max(self.metrics['memory_usage']):.1f}%")
        print(f"Average CPU usage: {np.mean(self.metrics['cpu_usage']):.1f}%")
        
        if self.gpu_available:
            print(f"Average GPU usage: {np.mean(self.metrics['gpu_usage']):.1f}%")
            print(f"Peak GPU memory: {max(self.metrics['gpu_memory']):.1f}%")
            print(f"GPU efficiency: {np.mean(self.metrics['gpu_efficiency']):.1f}%")
        
        return self.metrics
    
    def evolution_step_cpu(self, dt):
        """CPU fallback evolution step"""
        # Compute Laplacian
        laplacian_phi = self.compute_laplacian_cpu(self.phi)
        
        # Update momentum
        source_term = self.lambda_coupling * laplacian_phi + self.f3d * self.phi
        source_term = jnp.clip(source_term, -5.0, 5.0)
        
        self.pi += dt * source_term
        
        # Update field
        self.phi += dt * self.pi
        
        # Apply bounds
        self.phi = jnp.clip(self.phi, -0.02, 0.02)
        self.pi = jnp.clip(self.pi, -0.02, 0.02)
        
        # Recompute geometry
        self.f3d = self.compute_gpu_metric()
        self.R3d = self.compute_gpu_ricci()
    
    def compute_laplacian_cpu(self, field):
        """CPU implementation of Laplacian"""
        laplacian = jnp.zeros_like(field)
        dx2 = self.dx**2
        
        # X-direction
        laplacian = laplacian.at[1:-1, :, :].add(
            (field[2:, :, :] - 2*field[1:-1, :, :] + field[:-2, :, :]) / dx2
        )
        
        # Y-direction
        laplacian = laplacian.at[:, 1:-1, :].add(
            (field[:, 2:, :] - 2*field[:, 1:-1, :] + field[:, :-2, :]) / dx2
        )
        
        # Z-direction
        laplacian = laplacian.at[:, :, 1:-1].add(
            (field[:, :, 2:] - 2*field[:, :, 1:-1] + field[:, :, :-2]) / dx2
        )
        
        return laplacian

def benchmark_gpu_vs_cpu(grid_sizes=[48, 64, 80], steps=100):
    """Benchmark GPU vs CPU performance"""
    print(f"\\n🔬 GPU vs CPU Performance Benchmark")
    print(f"=" * 45)
    
    results = {}
    
    for N in grid_sizes:
        print(f"\\n📊 Testing {N}³ grid...")
        
        # GPU test
        if GPU_AVAILABLE:
            print(f"  🎮 GPU test...")
            gpu_sim = GPUAccelerated3DReplicator(grid_size=N, force_gpu=True)
            
            start_time = time.time()
            for step in range(steps):
                gpu_sim.phi, gpu_sim.pi = gpu_sim.gpu_evolution_step(
                    gpu_sim.phi, gpu_sim.pi, gpu_sim.f3d, 0.0005
                )
            gpu_time = time.time() - start_time
            gpu_throughput = N**3 * steps / gpu_time
            
            # Monitor final GPU usage
            gpu_usage, gpu_memory = gpu_sim.monitor_gpu_usage()
        else:
            gpu_time = float('inf')
            gpu_throughput = 0
            gpu_usage = 0
            gpu_memory = 0
        
        # CPU test
        print(f"  🖥️  CPU test...")
        cpu_sim = GPUAccelerated3DReplicator(grid_size=N, force_gpu=False)
        
        start_time = time.time()
        for step in range(steps):
            cpu_sim.evolution_step_cpu(0.0005)
        cpu_time = time.time() - start_time
        cpu_throughput = N**3 * steps / cpu_time
        
        # Results
        speedup = gpu_throughput / cpu_throughput if cpu_throughput > 0 else 0
        
        results[N] = {
            'gpu_time': gpu_time,
            'cpu_time': cpu_time,
            'gpu_throughput': gpu_throughput,
            'cpu_throughput': cpu_throughput,
            'speedup': speedup,
            'gpu_usage': gpu_usage,
            'gpu_memory': gpu_memory
        }
        
        print(f"    GPU: {gpu_throughput:,.0f} pts/s ({gpu_time:.2f}s)")
        print(f"    CPU: {cpu_throughput:,.0f} pts/s ({cpu_time:.2f}s)")
        if speedup > 0:
            print(f"    Speedup: {speedup:.1f}x")
            print(f"    GPU utilization: {gpu_usage:.1f}%")
    
    return results

def main():
    """Main GPU-accelerated framework"""
    print("🎮 GPU-Accelerated Desktop Unified LQG-QFT Framework")
    print("=" * 65)
    
    # Hardware detection
    print("🔍 Detecting hardware...")
    simulator = GPUAccelerated3DReplicator(grid_size=80, force_gpu=True)
    
    # Run benchmark
    print("\\n🔬 Running GPU vs CPU benchmark...")
    benchmark_results = benchmark_gpu_vs_cpu([48, 64, 80], steps=50)
    
    # Save benchmark results
    with open("gpu_cpu_benchmark.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)
    
    # Run main simulation
    print(f"\\n🚀 Running main GPU-accelerated simulation...")
    results = simulator.run_gpu_accelerated_simulation(total_steps=1000)
    
    # Generate comprehensive report
    report = {
        "gpu_accelerated_results": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hardware": {
                "cpu_cores": simulator.cpu_cores,
                "ram_gb": simulator.total_ram_gb,
                "gpu_available": simulator.gpu_available,
                "gpu_info": simulator.gpu_info,
                "compute_device": simulator.compute_device
            },
            "performance": {
                "avg_throughput": float(np.mean(results['throughput'])),
                "avg_step_time_ms": float(np.mean(results['step_times']) * 1000),
                "avg_cpu_usage": float(np.mean(results['cpu_usage'])),
                "avg_gpu_usage": float(np.mean(results['gpu_usage'])),
                "peak_memory": float(max(results['memory_usage'])),
                "gpu_efficiency": float(np.mean(results['gpu_efficiency']))
            },
            "benchmark_results": benchmark_results
        }
    }
    
    # Save report
    with open("gpu_accelerated_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Summary
    print(f"\\n🎯 GPU-Accelerated Framework Results:")
    print(f"   Device: {simulator.compute_device}")
    print(f"   Throughput: {np.mean(results['throughput']):,.0f} points/second")
    print(f"   CPU usage: {np.mean(results['cpu_usage']):.1f}%")
    if simulator.gpu_available:
        print(f"   GPU usage: {np.mean(results['gpu_usage']):.1f}%")
        print(f"   GPU efficiency: {np.mean(results['gpu_efficiency']):.1f}%")
    
    print(f"\\n📁 Generated Files:")
    print(f"   - gpu_cpu_benchmark.json")
    print(f"   - gpu_accelerated_report.json")
    
    if simulator.gpu_available and np.mean(results['gpu_usage']) > 20:
        print(f"\\n🚀 GPU acceleration successful!")
    elif simulator.gpu_available:
        print(f"\\n⚠️  GPU detected but low utilization - may need optimization")
    else:
        print(f"\\n🖥️  Running on CPU - consider installing GPU-enabled JAX")

if __name__ == "__main__":
    main()
