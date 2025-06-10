#!/usr/bin/env python3
"""
Setup script for Unified LQG-QFT Framework

A comprehensive framework for Loop Quantum Gravity (LQG) and Quantum Field Theory (QFT)
integration with focus on matter creation, replicator physics, and exotic spacetime geometries.
"""

from setuptools import setup, find_packages
import os

# Read the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="unified-lqg-qft",
    version="1.0.0",
    description="Unified Loop Quantum Gravity and Quantum Field Theory Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LQG-QFT Research Team",
    author_email="research@lqg-qft.org",
    url="https://github.com/your-org/unified-lqg-qft",
    
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    
    # Optional dependencies for advanced features
    extras_require={
        "gpu": ["jax>=0.4.0", "jaxlib>=0.4.0", "torch>=2.0.0"],
        "visualization": ["pyvista>=0.40.0", "vtk>=9.2.0"],
        "fenics": ["dolfin-adjoint>=2023.0.0", "fenics>=2019.1.0"],
        "dev": ["pytest>=6.0.0", "pytest-cov>=3.0.0", "black>=22.0.0", "flake8>=4.0.0"],
        "all": [
            "jax>=0.4.0", "jaxlib>=0.4.0", "torch>=2.0.0",
            "pyvista>=0.40.0", "vtk>=9.2.0",
            "dolfin-adjoint>=2023.0.0", "fenics>=2019.1.0",
            "pytest>=6.0.0", "pytest-cov>=3.0.0", "black>=22.0.0", "flake8>=4.0.0"
        ]
    },
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "unified-lqg-qft=src.main:main",
            "lqg-anec-analysis=automated_ghost_eft_scanner:main",
            "ghost-eft-scanner=automated_ghost_eft_scanner:main",
        ],
    },
    
    # Package data
    package_data={
        "src": ["*.yaml", "*.json", "*.txt"],
        "scripts": ["*.py"],
    },
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    keywords="quantum-gravity loop-quantum-gravity quantum-field-theory anec spacetime physics",
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/your-org/unified-lqg-qft/issues",
        "Source": "https://github.com/your-org/unified-lqg-qft",
        "Documentation": "https://unified-lqg-qft.readthedocs.io/",
    },
)
