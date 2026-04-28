"""Compatibility alias for :mod:`sarpyx.snapflow`."""

import sys

from . import snapflow as _snapflow

sys.modules[__name__] = _snapflow
