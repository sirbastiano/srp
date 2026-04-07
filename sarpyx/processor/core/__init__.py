"""Core processing components with lazy attribute resolution.

Avoid importing focus/torch-heavy modules just to reach unrelated helpers such
as ``dim_updater``.
"""

import importlib

_SUBMODULES = ('focus', 'decode', 'transforms', 'constants')
_cache = {}

__all__ = list(_SUBMODULES)


def __getattr__(name):
    if name in _cache:
        return _cache[name]

    if name in _SUBMODULES:
        module = importlib.import_module(f'.{name}', __name__)
        _cache[name] = module
        return module

    for module_name in _SUBMODULES:
        module = _cache.get(module_name)
        if module is None:
            module = importlib.import_module(f'.{module_name}', __name__)
            _cache[module_name] = module
        if hasattr(module, name):
            value = getattr(module, name)
            _cache[name] = value
            return value

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
