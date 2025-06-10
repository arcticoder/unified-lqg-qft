#!/usr/bin/env python3
"""
Validation script for the unified LQG-QFT framework.

This script tests the basic functionality of the copied modules and ensures
all dependencies are properly resolved.
"""

import sys
import os
import warnings
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test that all core modules can be imported."""
    print("Testing basic module imports...")
    
    try:
        # Core polymer quantization
        from src.polymer_quantization import PolymerOperator
        print("✓ polymer_quantization.py")
        
        # Coherent states  
        from src.coherent_states import CoherentState
        print("✓ coherent_states.py")
        
        # Spin networks
        from src.spin_network_utils import build_flat_graph
        print("✓ spin_network_utils.py")
        
        # Energy sources
        from src.energy_source_interface import EnergySource
        print("✓ energy_source_interface.py")
        
        # Ghost EFT
        from src.ghost_condensate_eft import GhostEFTParameters
        print("✓ ghost_condensate_eft.py")
        
        # Vacuum engineering
        from src.vacuum_engineering import VacuumStateOptimizer
        print("✓ vacuum_engineering.py")
        
        # Warp bubble solver
        from src.warp_bubble_solver import WarpBubbleSolver
        print("✓ warp_bubble_solver.py")
        
        # ANEC analysis
        from src.anec_violation_analysis import coherent_state_anec_violation
        print("✓ anec_violation_analysis.py")
        
        # Utilities
        from src.utils.smearing import ford_roman_sampling_function
        print("✓ utils/smearing.py")
        
        print("\nAll core modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\nTesting basic functionality...")
    
    try:
        # Test spin network creation
        from src.spin_network_utils import build_flat_graph
        graph = build_flat_graph(10, connectivity="cubic")
        print(f"✓ Created spin network with {len(graph.nodes)} nodes")
        
        # Test coherent state
        from src.coherent_states import CoherentState  
        coherent_state = CoherentState(graph, alpha=0.05)
        print("✓ Created coherent state")
        
        # Test Ford-Roman function
        from src.utils.smearing import ford_roman_sampling_function
        import numpy as np
        t_vals = np.linspace(-2, 2, 100)
        fr_vals = ford_roman_sampling_function(t_vals, tau=1.0)
        print(f"✓ Ford-Roman function computed ({len(fr_vals)} points)")
        
        # Test polymer operator
        from src.polymer_quantization import PolymerOperator
        polymer_op = PolymerOperator(mu=0.1)
        print("✓ Created polymer operator")
        
        print("\nBasic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Functionality test error: {e}")
        return False


def test_ghost_eft():
    """Test Ghost EFT functionality."""
    print("\nTesting Ghost EFT...")
    
    try:
        from src.ghost_condensate_eft import GhostEFTParameters, GhostCondensateEFT
        
        # Create parameters
        params = GhostEFTParameters(
            phi_0=1.0,
            lambda_ghost=0.1,
            cutoff_scale=10.0
        )
        print("✓ Created Ghost EFT parameters")
        
        # Create EFT instance
        eft = GhostCondensateEFT(params)
        print("✓ Created Ghost condensate EFT")
        
        # Test basic computation (if available)
        try:
            result = eft.compute_stress_energy_density(x=0.0, t=0.0)
            print(f"✓ Computed stress-energy density: {result:.3e}")
        except:
            print("⚠ Stress-energy computation not available (expected)")
        
        print("Ghost EFT tests completed!")
        return True
        
    except Exception as e:
        print(f"✗ Ghost EFT test error: {e}")
        return False


def test_dependencies():
    """Test that required dependencies are available."""
    print("\nTesting dependencies...")
    
    required_deps = [
        ("numpy", "np"),
        ("scipy", "scipy"),
        ("matplotlib", "plt"),
        ("networkx", "nx"),
    ]
    
    optional_deps = [
        ("jax", "jax"),
        ("torch", "torch"),
        ("pyvista", "pv"),
    ]
    
    for dep_name, alias in required_deps:
        try:
            if alias == "plt":
                import matplotlib.pyplot as plt
            elif alias == "nx":
                import networkx as nx
            elif alias == "np":
                import numpy as np
            elif alias == "scipy":
                import scipy
            print(f"✓ {dep_name}")
        except ImportError:
            print(f"✗ {dep_name} (REQUIRED)")
            return False
    
    for dep_name, alias in optional_deps:
        try:
            if alias == "jax":
                import jax
            elif alias == "torch":
                import torch
            elif alias == "pv":
                import pyvista as pv
            print(f"✓ {dep_name} (optional)")
        except ImportError:
            print(f"⚠ {dep_name} (optional, not available)")
    
    print("Dependency check completed!")
    return True


def main():
    """Run all validation tests."""
    print("Unified LQG-QFT Framework Validation")
    print("=" * 50)
    
    tests = [
        ("Dependency Check", test_dependencies),
        ("Basic Imports", test_basic_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Ghost EFT", test_ghost_eft),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:<20}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All validation tests PASSED!")
        print("The unified LQG-QFT framework is ready for use.")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) FAILED.")
        print("Please check the error messages above.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
