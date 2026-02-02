"""Sub-look metrics.

Example:
    from srp.sarpyx.sla.metrics import stack_metrics
"""

from __future__ import annotations

import numpy as np


def _intensity(x: np.ndarray) -> np.ndarray:
    return np.abs(x) ** 2 if np.iscomplexobj(x) else np.asarray(x)


def enl(x: np.ndarray, axis=None, eps: float = 1e-12) -> np.ndarray:
    """Computes equivalent number of looks (ENL).

    Args:
        x (np.ndarray): Intensity or complex samples.
        axis: Axis of looks.
        eps (float): Small value to avoid division by zero.

    Returns:
        np.ndarray: ENL estimate.
    """
    i = _intensity(x)
    m = np.mean(i, axis=axis)
    v = np.var(i, axis=axis)
    return (m * m) / (v + eps)


def dispersion_ratio(x: np.ndarray, axis=None, eps: float = 1e-12) -> np.ndarray:
    """Computes dispersion ratio (normalized variance).

    Args:
        x (np.ndarray): Intensity or complex samples.
        axis: Axis of looks.
        eps (float): Small value to avoid division by zero.

    Returns:
        np.ndarray: Dispersion ratio.
    """
    i = _intensity(x)
    m = np.mean(i, axis=axis)
    v = np.var(i, axis=axis)
    return v / (m * m + eps)


def interlook_coherence(a: np.ndarray, b: np.ndarray, axis=None, eps: float = 1e-12) -> np.ndarray:
    """Computes inter-look coherence between two complex looks.

    Args:
        a (np.ndarray): First complex look.
        b (np.ndarray): Second complex look.
        axis: Averaging axis.
        eps (float): Small value to avoid division by zero.

    Returns:
        np.ndarray: Coherence magnitude in [0, 1].
    """
    num = np.abs(np.mean(a * np.conj(b), axis=axis))
    den = np.sqrt(np.mean(np.abs(a) ** 2, axis=axis) * np.mean(np.abs(b) ** 2, axis=axis))
    return num / (den + eps)


def phase_variance(x: np.ndarray, axis=None) -> np.ndarray:
    """Computes circular phase variance.

    Args:
        x (np.ndarray): Complex samples.
        axis: Axis of looks.

    Returns:
        np.ndarray: Circular phase variance in [0, 1].
    """
    ph = np.angle(x)
    return 1.0 - np.abs(np.mean(np.exp(1j * ph), axis=axis))


def stack_metrics(stack: np.ndarray, look_axis: int = 0, pair=(0, 1), eps: float = 1e-12):
    """Computes all metrics from a sub-look stack.

    Args:
        stack (np.ndarray): Complex sub-look stack.
        look_axis (int): Axis of looks.
        pair (tuple[int, int]): Look indices for coherence.
        eps (float): Small value to avoid division by zero.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ENL, coherence,
        dispersion ratio, phase variance.
    """
    s = np.moveaxis(stack, look_axis, 0)
    a, b = s[pair[0]], s[pair[1]]
    return (
        enl(s, axis=0, eps=eps),
        interlook_coherence(a, b, axis=0, eps=eps),
        dispersion_ratio(s, axis=0, eps=eps),
        phase_variance(s, axis=0),
    )
