"""SNAP integration module for sarpyx.

This module provides integration with the Sentinel Application Platform (SNAP)
for SAR data processing workflows.
"""

from sarpyx.snapflow.engine import GPT
from sarpyx.snapflow.snap2stamps_pipelines import Snap2StampsRunner

__all__ = ["GPT", "Snap2StampsRunner"]

__version__ = '0.1.5'
