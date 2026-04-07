"""Utilities package with lazy exports.

Importing ``sarpyx.utils`` should not eagerly load visualization helpers that
pull optional heavy dependencies.
"""

import importlib

_EXPORT_MAP = {
    'show_image': 'viz',
    'image_histogram_equalization': 'viz',
    'show_histogram': 'viz',
    'show_histogram_equalization': 'viz',
    'download_tiles_for_wkt': 'dem_utils',
    'tiles_from_wkt': 'dem_utils',
    'tile_name': 'dem_utils',
    'tile_url': 'dem_utils',
    'build_vrt': 'dem_utils',
}

__all__ = list(_EXPORT_MAP)
__version__ = '0.1.5'

_module_cache = {}
_value_cache = {}


def __getattr__(name):
    if name in _value_cache:
        return _value_cache[name]

    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module = _module_cache.get(module_name)
    if module is None:
        module = importlib.import_module(f'.{module_name}', __name__)
        _module_cache[module_name] = module

    value = getattr(module, name)
    _value_cache[name] = value
    return value
