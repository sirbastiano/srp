"""
SARPyX: SAR Processing library in Python

This package provides tools and utilities for SAR (Synthetic Aperture Radar) data processing.
"""

__version__ = "0.1.0"
__author__ = "ESA Phi-Lab"

# Import main components to make them available at package level
from . import sla
from . import utils
from . import processor # Uncomment when processor has content
# from . import snap      # Uncomment when snap has content

__all__ = [
    "sla",
    "utils",
    "processor",
    "snap",
    "__version__",
    "__author__"
]