"""SARPyX: SAR Processing library in Python.

This package provides tools and utilities for SAR (Synthetic Aperture Radar) 
data processing, including sub-look analysis, visualization, and integration
with processing frameworks.
"""

__version__ = '0.1.0'
__author__ = 'ESA Phi-Lab'

# Import main components to make them available at package level
from . import sla
from . import utils
from . import processor
# from . import snap      # Uncomment when snap has content
# from . import science   # Uncomment when science has content

__all__ = [
    'sla',
    'utils', 
    'processor',
    'snap', # type: ignore
    # 'science',
    '__version__',
    '__author__'
]