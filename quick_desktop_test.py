#!/usr/bin/env python3
"""
Quick Desktop Hardware Test
"""

import numpy as np
import multiprocessing as mp
import psutil
import time

def main():
    print("Desktop Hardware Quick Test")
    print("=" * 30)
    
    # CPU
    cpu_cores = mp.cpu_count()
    print(f"CPU cores: {cpu_cores}")
    
    # Memory
    memory = psutil.virtual_memory()
    total_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    print(f"RAM: {total_gb:.1f} GB total, {available_gb:.1f} GB available")
    
    # Grid size test
    for N in [32, 48, 64]:
        memory_gb = (N**3 * 8 * 8) / (1024**3)
        print(f"Grid {N}³: {memory_gb:.3f} GB")
    
    # Simple computation test
    print(f"\\nTesting 48³ grid computation...")
    N = 48
    start_time = time.time()
    
    # Create test arrays
    x = np.linspace(-3, 3, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)
    
    # Simple computation
    field = 1e-4 * np.exp(-r**2)
    
    # Laplacian test
    laplacian = np.zeros_like(field)
    dx = 6.0 / (N-1)
    laplacian[1:-1,:,:] = (field[2:,:,:] - 2*field[1:-1,:,:] + field[:-2,:,:]) / dx**2
    
    elapsed = time.time() - start_time
    points_per_sec = N**3 / elapsed
    
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Throughput: {points_per_sec:,.0f} points/second")
    print(f"Field RMS: {np.sqrt(np.mean(field**2)):.2e}")
    
    print(f"\\n✅ Desktop test completed successfully!")

if __name__ == "__main__":
    main()
