"""SARPyX: SAR Processing library in Python.

This package provides tools and utilities for SAR (Synthetic Aperture Radar) 
data processing, including sub-look analysis, visualization, and integration
with processing frameworks.
"""

__version__ = '0.1.5'
__author__ = 'ESA Phi-Lab'

__all__ = [
    'sla',
    'utils', 
    'processor',
    'snap',
    'cli',
    'science',
    '__version__',
    '__author__'
]

# Lazy imports using __getattr__ with caching to avoid circular imports
_module_cache = {}

def __getattr__(name):
    """Lazy import of submodules to prevent circular imports."""
    if name in _module_cache:
        return _module_cache[name]
    
    if name in ('sla', 'utils', 'processor', 'snap', 'cli', 'science'):
        import importlib
        module = importlib.import_module(f'.{name}', __name__)
        _module_cache[name] = module
        return module
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
