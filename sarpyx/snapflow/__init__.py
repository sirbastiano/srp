"""SNAP integration module for sarpyx."""

from sarpyx.snapflow.engine import GPT
from sarpyx.snapflow.snap2stamps import (
    PairProducts,
    SNAP2STAMPS_PIPELINES,
    SNAP2STAMPS_WORKFLOWS,
    SNAP2STAMPS_WORKFLOW_INPUTS,
    build_gpt,
    get_pipeline_definition,
    list_pipeline_names,
    pipeline_requires_multi_input,
    pipeline_requires_pair,
    prepare_pair,
    run_pair_workflow,
    run_processing_pipeline,
)

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
]

__version__ = '0.1.5'
