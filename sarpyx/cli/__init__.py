"""
CLI module for sarpyx package.

This module provides command-line interfaces for various sarpyx functionalities,
including ship detection, SAR processing, and other utilities.
"""

# Keep imports lazy to reduce CLI startup time.
__all__ = [
    'cli_main',
    'shipdet_main',
    'utils',
]

def cli_main(*args, **kwargs):
    from .main import main
    return main(*args, **kwargs)

def shipdet_main(*args, **kwargs):
    from .shipdet import main
    return main(*args, **kwargs)

def utils():
    from . import utils as _utils
    return _utils

__version__ = '0.1.5'
