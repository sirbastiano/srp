"""SNAP integration module for sarpyx with lazy exports."""

import importlib

__all__ = [
    "GPT",
    "PairProducts",
    "SNAP2STAMPS_PIPELINES",
    "SNAP2STAMPS_WORKFLOWS",
    "SNAP2STAMPS_WORKFLOW_INPUTS",
    "build_gpt",
    "get_pipeline_definition",
    "list_pipeline_names",
    "pipeline_requires_multi_input",
    "pipeline_requires_pair",
    "prepare_pair",
    "run_pair_workflow",
    "run_processing_pipeline",
    "update_dim_add_bands_from_data_dir",
]

__version__ = '0.1.5'

_EXPORT_MODULES = {
    "GPT": "engine",
    "PairProducts": "snap2stamps",
    "SNAP2STAMPS_PIPELINES": "snap2stamps",
    "SNAP2STAMPS_WORKFLOWS": "snap2stamps",
    "SNAP2STAMPS_WORKFLOW_INPUTS": "snap2stamps",
    "build_gpt": "snap2stamps",
    "get_pipeline_definition": "snap2stamps",
    "list_pipeline_names": "snap2stamps",
    "pipeline_requires_multi_input": "snap2stamps",
    "pipeline_requires_pair": "snap2stamps",
    "prepare_pair": "snap2stamps",
    "run_pair_workflow": "snap2stamps",
    "run_processing_pipeline": "snap2stamps",
    "update_dim_add_bands_from_data_dir": "dim_updater",
}

_module_cache = {}
_value_cache = {}


def __getattr__(name):
    if name in _value_cache:
        return _value_cache[name]

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module = _module_cache.get(module_name)
    if module is None:
        module = importlib.import_module(f'.{module_name}', __name__)
        _module_cache[module_name] = module

    value = getattr(module, name)
    _value_cache[name] = value
    return value
