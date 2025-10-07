#!/usr/bin/env python3
"""
Setup script for Bio-Inspired Navigation Data Generation Package

This script provides installation configuration for the bio_nav_data package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()

setup(
    name="bio-nav-data",
    version="2.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="A comprehensive Python package for generating and visualizing bio-inspired navigation research data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bio-nav-data",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bio-nav-data=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "bio_nav_data": ["*.txt", "*.md"],
    },
    keywords=[
        "bio-inspired",
        "navigation",
        "robotics",
        "SLAM",
        "data-generation",
        "visualization",
        "research",
        "simulation",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/bio-nav-data/issues",
        "Source": "https://github.com/yourusername/bio-nav-data",
        "Documentation": "https://github.com/yourusername/bio-nav-data#readme",
    },
) 