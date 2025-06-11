"""
Test script for mathematical simulation steps
"""

print("Testing mathematical simulation...")

import sys
sys.path.append(r'c:\Users\echo_\Code\asciimath\unified-lqg-qft')

try:
    from mathematical_simulation_steps import ClosedFormEffectivePotential
    print("✅ Import successful")
    
    # Test Step 1
    potential = ClosedFormEffectivePotential()
    r_opt, V_max = potential.find_optimal_r()
    
    print(f"✅ Step 1 complete: r_opt = {r_opt:.6f}, V_max = {V_max:.2e}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
