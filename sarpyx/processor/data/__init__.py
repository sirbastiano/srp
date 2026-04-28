"""Data handling and I/O for SAR processing.

The package exposes common reader, writer, and formatter helpers lazily so that
importing ``sarpyx.processor.data`` does not immediately import heavier I/O
libraries such as rasterio and zarr.
"""

import importlib

_SUBMODULES = ("readers", "writers", "formatters")
_EXPORT_MODULES = {
    "read_sentinel1": "readers",
    "read_cosmo_skymed": "readers",
    "read_terrasar_x": "readers",
    "read_tif": "readers",
    "read_zarr_file": "readers",
    "write_geotiff": "writers",
    "write_hdf5": "writers",
    "write_binary": "writers",
    "format_converter": "formatters",
    "preprocess_data": "formatters",
    "validate_format": "formatters",
}

__all__ = [*_SUBMODULES, *_EXPORT_MODULES]

_module_cache = {}
_value_cache = {}


def __getattr__(name):
    if name in _value_cache:
        return _value_cache[name]

    module_name = name if name in _SUBMODULES else _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module = _module_cache.get(module_name)
    if module is None:
        module = importlib.import_module(f".{module_name}", __name__)
        _module_cache[module_name] = module

    value = module if name == module_name else getattr(module, name)
    _value_cache[name] = value
    return value


def __dir__():
    return sorted({*globals(), *__all__})
