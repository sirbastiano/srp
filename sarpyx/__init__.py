"""SARPyX: SAR Processing library in Python.

This package provides tools and utilities for SAR (Synthetic Aperture Radar)
data processing, including sub-look analysis, visualization, and integration
with processing frameworks.
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _resolve_version() -> str:
    try:
        return version('sarpyx')
    except PackageNotFoundError:
        try:
            import tomllib

            pyproject = Path(__file__).resolve().parents[1] / 'pyproject.toml'
            with pyproject.open('rb') as handle:
                return tomllib.load(handle)['project']['version']
        except Exception:
            return '0+unknown'


__version__ = _resolve_version()
__author__ = 'ESA Phi-Lab'

__all__ = [
    'sla',
    'utils',
    'processor',
    'snapflow',
    'snap',
    'cli',
    'science',
    '__version__',
    '__author__'
]

# Lazy imports using __getattr__ with caching to avoid circular imports
_module_cache = {}
_module_aliases = {
    'snap': 'snapflow',
}
_public_modules = {'sla', 'utils', 'processor', 'snapflow', 'snap', 'cli', 'science'}


def __getattr__(name):
    """Lazy import of submodules to prevent circular imports."""
    if name in _module_cache:
        return _module_cache[name]

    if name in _public_modules:
        import importlib
        module_name = _module_aliases.get(name, name)
        module = importlib.import_module(f'.{module_name}', __name__)
        _module_cache[name] = module
        _module_cache.setdefault(module_name, module)
        return module

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
