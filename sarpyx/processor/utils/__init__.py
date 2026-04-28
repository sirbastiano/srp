"""Processor utilities for SAR data handling.

Utilities are exported lazily so importing ``sarpyx.processor.utils`` remains
lightweight and does not immediately load plotting or torch dependencies.
"""

import importlib

_SUBMODULES = ("viz", "summary", "unzip", "mem", "metrics")
_EXPORT_MODULES = {
    "plot_with_logscale": "viz",
    "plot_with_cdf": "viz",
    "plot2_with_cdf": "viz",
    "get_lognorm": "viz",
    "find_checkpoint": "viz",
    "plot_histogram": "viz",
    "dump": "viz",
    "load": "viz",
    "summarize_2d_array": "summary",
    "cleanup_memory": "mem",
    "luminance": "metrics",
    "contrast": "metrics",
    "structure": "metrics",
    "create_window": "metrics",
    "ssim": "metrics",
    "psnr": "metrics",
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
