"""
SAR Processing Module.

This module provides comprehensive SAR (Synthetic Aperture Radar) processing
capabilities including focusing algorithms, autofocus methods, data I/O,
and various processing utilities.

Submodules:
    core: Core processing algorithms (focus, decode, transforms)
    autofocus: Autofocus algorithms and quality metrics
    algorithms: High-level processing algorithms (RDA, back-projection)
    data: Data readers, writers, and format converters
    utils: Processing utilities and helper functions
    
Legacy:
    AutoFocusNet: Original AutoFocusNet implementation (maintained for compatibility)
"""

# Import main submodules
from . import core
from . import autofocus  
from . import algorithms
from . import data
from . import utils

# Keep AutoFocusNet for backward compatibility
from . import AutoFocusNet

__all__ = [
    'core',
    'autofocus',
    'algorithms', 
    'data',
    'utils',
    'AutoFocusNet',  # For backward compatibility
]

__version__ = "0.2.0"