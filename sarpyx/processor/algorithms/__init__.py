"""
SAR Processing Algorithms.

This submodule contains high-level processing algorithms for SAR data,
including range-doppler algorithms, back-projection, and other focusing methods.
"""

from .rda import simple_rda
from .backprojection import time_domain_backprojection

__all__ = [
    # Exports will be properly defined based on actual module contents
    'simple_rda',
    'time_domain_backprojection',
]
