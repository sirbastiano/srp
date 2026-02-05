"""
CLI module for sarpyx package.

This module provides command-line interfaces for various sarpyx functionalities,
including ship detection, SAR processing, and other utilities.
"""

from .main import main as cli_main
from .shipdet import main as shipdet_main
from . import utils

__all__ = [
    'cli_main',
    'shipdet_main',
    'utils',
]

__version__ = '0.1.6'
