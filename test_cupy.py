#!/usr/bin/env python3
"""Test CuPy GPU functionality"""

try:
    import cupy as cp
    print("✅ CuPy loaded successfully")
    
    # Check device count
    device_count = cp.cuda.runtime.getDeviceCount()
    print(f"GPU devices: {device_count}")
    
    # Test basic GPU operation
    a = cp.array([1, 2, 3, 4, 5])
    b = cp.array([10, 20, 30, 40, 50])
    c = a + b
    print(f"GPU computation test: {cp.asnumpy(c)}")
    
    # Memory info
    device = cp.cuda.Device(0)
    mem_info = device.mem_info
    print(f"GPU memory: {mem_info[1]/1e9:.1f} GB total, {mem_info[0]/1e9:.1f} GB free")
    
    print("🎮 CuPy GPU acceleration ready!")
    
except Exception as e:
    print(f"❌ CuPy error: {e}")
    import traceback
    traceback.print_exc()
