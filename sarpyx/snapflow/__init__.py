"""SNAP integration module for sarpyx.

This module provides integration with the Sentinel Application Platform (SNAP)
for SAR data processing workflows.
"""

from sarpyx.snapflow.engine import GPT

__all__ = ["GPT"]

# Keep snapflow import resilient while snap2stamps APIs evolve.
try:  # pragma: no cover - exercised by import smoke tests
    from sarpyx.snapflow.snap2stamps_pipelines import Snap2StampsRunner
except ImportError:
    pass
else:
    __all__.append("Snap2StampsRunner")

__version__ = '0.1.5'
