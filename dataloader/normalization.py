import torch
import torch.nn as nn
import numpy as np
import functools
from typing import Optional, Callable, Union
from utils import minmax_normalize, minmax_inverse, RC_MIN, RC_MAX, GT_MIN, GT_MAX

class BaseTransformModule(nn.Module):
    """Base class for SAR data transformations."""
    
    def __init__(self):
        super(BaseTransformModule, self).__init__()
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply transformation to input data."""
        raise NotImplementedError("Subclasses must implement forward method")
    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Inverse transformation, if applicable."""
        raise NotImplementedError("Subclasses must implement inverse method")


class NormalizationModule(BaseTransformModule):
    """Normalization module for SAR data."""
    
    def __init__(self, data_min: float, data_max: float):
        super(NormalizationModule, self).__init__()
        self.data_min = data_min
        self.data_max = data_max
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Normalize array to range [0, 1]."""
        # Apply normalization directly on numpy array
        return minmax_normalize(x, self.data_min, self.data_max)
    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Inverse normalization."""
        # Apply inverse normalization
        return minmax_inverse(x, self.data_min, self.data_max)


class IdentityModule(BaseTransformModule):
    """Identity transformation that returns input unchanged."""
    
    def __init__(self):
        super(IdentityModule, self).__init__()
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return input unchanged."""
        return x
    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Return input unchanged."""
        return x


class ComplexNormalizationModule(BaseTransformModule):
    """Complex-valued normalization module for SAR data."""
    
    def __init__(self, real_min: float, real_max: float, imag_min: float, imag_max: float):
        super(ComplexNormalizationModule, self).__init__()
        self.real_min = real_min
        self.real_max = real_max
        self.imag_min = imag_min
        self.imag_max = imag_max
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Normalize complex array separately for real and imaginary parts."""
        if np.iscomplexobj(x):
            # Normalize real and imaginary parts separately
            # Use numpy functions to extract real and imaginary parts
            real_part = minmax_normalize(np.real(x), self.real_min, self.real_max)
            imag_part = minmax_normalize(np.imag(x), self.imag_min, self.imag_max)

            #real_part = np.clip(real_part, 0, 1)
            #imag_part = np.clip(imag_part, 0, 1)
            
            normalized = real_part + 1j * imag_part
        else:
            # Assume magnitude data
            normalized = minmax_normalize(x, self.real_min, self.real_max)

        return normalized
    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Inverse normalization."""
        if np.iscomplexobj(x):
            # Inverse normalize real and imaginary parts separately
            real_part = minmax_inverse(np.real(x), self.real_min, self.real_max)
            imag_part = minmax_inverse(np.imag(x), self.imag_min, self.imag_max)
            return real_part + 1j * imag_part
        else:
            # Assume magnitude data
            return minmax_inverse(x, self.real_min, self.real_max)

class AdaptiveNormalizationModule(BaseTransformModule):
    """Complex-valued normalization module for SAR data."""
    
    def __init__(self):
        super(AdaptiveNormalizationModule, self).__init__()

    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Normalize complex array separately for real and imaginary parts."""
        if np.iscomplexobj(x):
            # Normalize real and imaginary parts separately
            # Use numpy functions to extract real and imaginary parts
            re = np.real(x)
            im = np.imag(x)
            real_part = minmax_normalize(re, np.min(re), np.max(re))
            imag_part = minmax_normalize(im, np.min(im), np.max(im))

            #real_part = np.clip(real_part, 0, 1)
            #imag_part = np.clip(imag_part, 0, 1)
            
            normalized = real_part + 1j * imag_part
        else:
            # Assume magnitude data
            normalized = minmax_normalize(x, np.min(x), np.max(x))

        return normalized
    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Inverse normalization."""
        if np.iscomplexobj(x):
            # Inverse normalize real and imaginary parts separately
            real_part = minmax_inverse(np.real(x), np.min(x), np.max(x))
            imag_part = minmax_inverse(np.imag(x), np.min(x), np.max(x))
            return real_part + 1j * imag_part
        else:
            # Assume magnitude data
            return minmax_inverse(x, np.min(x), np.max(x))

class SARTransform(nn.Module):
    """
    PyTorch transform module for normalizing SAR data patches at different processing levels.

    This class uses separate PyTorch modules for each SAR processing level (e.g., 'raw', 'rc', 'rcmc', 'az').
    Each module can be any PyTorch nn.Module that implements the transformation.

    Args:
        transform_raw (BaseTransformModule, optional): Module to transform 'raw' level data.
        transform_rc (BaseTransformModule, optional): Module to transform 'rc' level data.
        transform_rcmc (BaseTransformModule, optional): Module to transform 'rcmc' level data.
        transform_az (BaseTransformModule, optional): Module to transform 'az' level data.
    """
    def __init__(
        self,
        transform_raw: Optional[Union[Callable[[np.ndarray], np.ndarray], functools.partial, BaseTransformModule]] = None,
        transform_rc: Optional[Union[Callable[[np.ndarray], np.ndarray], functools.partial, BaseTransformModule]] = None,
        transform_rcmc: Optional[Union[Callable[[np.ndarray], np.ndarray], functools.partial, BaseTransformModule]] = None,
        transform_az: Optional[Union[Callable[[np.ndarray], np.ndarray], functools.partial, BaseTransformModule]] = None,
    ):
        super(SARTransform, self).__init__()
        
        # Register transform modules
        self.transform_raw = transform_raw if transform_raw is not None else IdentityModule()
        self.transform_rc = transform_rc if transform_rc is not None else IdentityModule()
        self.transform_rcmc = transform_rcmc if transform_rcmc is not None else IdentityModule()
        self.transform_az = transform_az if transform_az is not None else IdentityModule()
        
        # Create a mapping for easy access
        self.transforms = {
            'raw': self.transform_raw,
            'rc': self.transform_rc,
            'rcmc': self.transform_rcmc,
            'az': self.transform_az,
        }

    def forward(self, x: np.ndarray, level: str) -> np.ndarray:
        """
        Apply the appropriate transform module to the input array for the specified SAR processing level.

        Args:
            x (np.ndarray): Input data array.
            level (str): SAR processing level ('raw', 'rc', 'rcmc', or 'az').

        Returns:
            np.ndarray: Transformed data array.
        """
        assert level in self.transforms, f"Transform for level '{level}' not defined."
        return self.transforms[level](x)
    def inverse(self, x: np.ndarray, level: str) -> np.ndarray:
        """
        Apply the appropriate inverse transform module to the input array for the specified SAR processing level.

        Args:
            x (np.ndarray): Input data array.
            level (str): SAR processing level ('raw', 'rc', 'rcmc', or 'az').

        Returns:
            np.ndarray: Inverse transformed data array.
        """
        assert level in self.transforms, f"Transform for level '{level}' not defined."
        if isinstance(self.transforms[level], BaseTransformModule):
            # If the transform is a custom module, use its inverse method
            return self.transforms[level].inverse(x)
        print("[WARNING] Inverse transform not defined, returning input unchanged.")
        return x

    @classmethod
    def create_minmax_normalized_transform(
        cls,
        normalize: bool = True,
        adaptive: bool = False,
        rc_min: float = RC_MIN,
        rc_max: float = RC_MAX,
        gt_min: float = GT_MIN,
        gt_max: float = GT_MAX,
        complex_valued: bool = True
    ):
        """
        Factory method to create a SARTransform with normalization modules.
        
        Args:
            normalize (bool): Whether to apply normalization.
            adaptive (bool): Whether to use adaptive normalization based on patch min/max.
            rc_min (float): Minimum value for RC data normalization.
            rc_max (float): Maximum value for RC data normalization.
            gt_min (float): Minimum value for ground truth data normalization.
            gt_max (float): Maximum value for ground truth data normalization.
            complex_valued (bool): Whether data is complex-valued.
        
        Returns:
            SARTransform: Configured transform instance.
        """
        if not normalize:
            return cls()
        if not adaptive:
            if complex_valued:
                raw_transform = ComplexNormalizationModule(rc_min, rc_max, rc_min, rc_max)
                rc_transform = ComplexNormalizationModule(gt_min, gt_max, gt_min, gt_max)
                rcmc_transform = ComplexNormalizationModule(gt_min, gt_max, gt_min, gt_max)
                az_transform = ComplexNormalizationModule(gt_min, gt_max, gt_min, gt_max)
            else:
                # Use simple normalization for magnitude data
                raw_transform = NormalizationModule(rc_min, rc_max)
                rc_transform = NormalizationModule(gt_min, gt_max)
                rcmc_transform = NormalizationModule(gt_min, gt_max)
                az_transform = NormalizationModule(gt_min, gt_max)
        else:
            raw_transform = AdaptiveNormalizationModule()
            rc_transform = AdaptiveNormalizationModule()
            rcmc_transform = AdaptiveNormalizationModule()
            az_transform = AdaptiveNormalizationModule()
        return cls(
            transform_raw=raw_transform,
            transform_rc=rc_transform,
            transform_rcmc=rcmc_transform,
            transform_az=az_transform
        )
    @classmethod
    def create_zscore_normalized_transform(cls, normalize=True, adaptive=True, 
                                          rc_mean=0.0, rc_std=1.0, 
                                          gt_mean=0.0, gt_std=1.0,
                                          complex_valued=True):
        """Create z-score normalized transform."""
        if adaptive:
            # Compute statistics from data during first batch
            raw_transform = AdaptiveZScoreNormalize(complex_valued=complex_valued)
            rc_transform = AdaptiveZScoreNormalize(complex_valued=complex_valued)
            rcmc_transform = AdaptiveZScoreNormalize(complex_valued=complex_valued)
            az_transform = AdaptiveZScoreNormalize(complex_valued=complex_valued)

        else:
            # Use provided statistics
            raw_transform = ZScoreNormalize(mean=rc_mean, std=rc_std, complex_valued=complex_valued)
            rc_transform = ZScoreNormalize(mean=gt_mean, std=gt_std, complex_valued=complex_valued)
            rcmc_transform = ZScoreNormalize(mean=gt_mean, std=gt_std, complex_valued=complex_valued)
            az_transform = ZScoreNormalize(mean=gt_mean, std=gt_std, complex_valued=complex_valued)
        
        return cls(
            transform_raw=raw_transform,
            transform_rc=rc_transform,
            transform_rcmc=rcmc_transform,
            transform_az=az_transform
        )
    
    @classmethod
    def create_robust_normalized_transform(cls, normalize=True, adaptive=True, complex_valued=True):
        """Create robust normalized transform using median and IQR."""
        raw_transform = RobustNormalize(adaptive=adaptive, complex_valued=complex_valued)
        rc_transform = RobustNormalize(adaptive=adaptive, complex_valued=complex_valued)
        rcmc_transform = RobustNormalize(adaptive=adaptive, complex_valued=complex_valued)
        az_transform = RobustNormalize(adaptive=adaptive, complex_valued=complex_valued)
        return cls(
            transform_raw=raw_transform,
            transform_rc=rc_transform,
            transform_rcmc=rcmc_transform,
            transform_az=az_transform
        )

class ZScoreNormalize(BaseTransformModule):
    """Z-score normalization: (x - mean) / std"""
    def __init__(self, mean=0.0, std=1.0, complex_valued=True):
        super(ZScoreNormalize, self).__init__()
        self.mean = mean
        self.std = std
        self.complex_valued = complex_valued
    
    def forward(self, x):
        if self.complex_valued and np.iscomplexobj(x):
            # Normalize real and imaginary parts separately
            real_norm = (x.real - self.mean) / self.std
            imag_norm = (x.imag - self.mean) / self.std
            return real_norm + 1j * imag_norm
        else:
            return (x - self.mean) / self.std
    def inverse(self, x):
        if self.complex_valued and np.iscomplexobj(x):
            # Inverse normalize real and imaginary parts separately
            real_inv = x.real * self.std + self.mean
            imag_inv = x.imag * self.std + self.mean
            return real_inv + 1j * imag_inv
        else:
            return x * self.std + self.mean

class AdaptiveZScoreNormalize(BaseTransformModule):
    """Adaptive z-score normalization that computes statistics from data."""
    def __init__(self, complex_valued=True, momentum=0.1):
        self.complex_valued = complex_valued
        self.momentum = momentum
        self.running_mean = None
        self.running_std = None
        self.initialized = False
    
    def forward(self, x):
        if not self.initialized:
            # Initialize with first batch statistics
            if self.complex_valued and np.iscomplexobj(x):
                self.running_mean = np.mean(np.abs(x))
                self.running_std = np.std(np.abs(x))
            else:
                self.running_mean = np.mean(x)
                self.running_std = np.std(x)
            self.initialized = True
        else:
            # Update running statistics
            if self.complex_valued and np.iscomplexobj(x):
                batch_mean = np.mean(np.abs(x))
                batch_std = np.std(np.abs(x))
            else:
                batch_mean = np.mean(x)
                batch_std = np.std(x)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_std = (1 - self.momentum) * self.running_std + self.momentum * batch_std
        
        # Apply normalization
        if self.complex_valued and np.iscomplexobj(x):
            magnitude = np.abs(x)
            normalized_magnitude = (magnitude - self.running_mean) / (self.running_std + 1e-8)
            phase = np.angle(x)
            return normalized_magnitude * np.exp(1j * phase)
        else:
            return (x - self.running_mean) / (self.running_std + 1e-8)
    

class RobustNormalize(BaseTransformModule):
    """
    Robust normalization using median and IQR: (x - median) / IQR
    More resistant to outliers than standard z-score normalization.
    
    Args:
        adaptive (bool): If True, compute statistics from data. If False, use provided values.
        median (float): Fixed median value (only used if adaptive=False)
        iqr (float): Fixed IQR value (only used if adaptive=False)
        complex_valued (bool): Handle complex-valued data
        momentum (float): Running statistics momentum for adaptive mode
        percentile_range (tuple): Percentiles for IQR calculation, default (25, 75)
    """
    
    def __init__(self, adaptive=True, median=0.0, iqr=1.0, complex_valued=True, 
                 momentum=0.1, percentile_range=(25, 75)):
        self.adaptive = adaptive
        self.complex_valued = complex_valued
        self.momentum = momentum
        self.percentile_range = percentile_range
        
        if not adaptive:
            self.median = median
            self.iqr = max(iqr, 1e-8)  # Prevent division by zero
        else:
            self.running_median = None
            self.running_iqr = None
            self.initialized = False

    def _compute_robust_stats(self, x):
        """Compute median and IQR from numpy array."""
        if self.complex_valued and np.iscomplexobj(x):
            magnitude = np.abs(x)
        else:
            magnitude = x

        flat_data = magnitude.flatten()
        median_val = np.median(flat_data)
        q1 = np.percentile(flat_data, self.percentile_range[0])
        q3 = np.percentile(flat_data, self.percentile_range[1])
        iqr_val = q3 - q1
        iqr_val = max(iqr_val, 1e-8)
        return median_val, iqr_val

    def forward(self, x):
        """Apply robust normalization."""
        if self.adaptive:
            batch_median, batch_iqr = self._compute_robust_stats(x)
            if not self.initialized:
                self.running_median = batch_median
                self.running_iqr = batch_iqr
                self.initialized = True
            else:
                self.running_median = (1 - self.momentum) * self.running_median + self.momentum * batch_median
                self.running_iqr = (1 - self.momentum) * self.running_iqr + self.momentum * batch_iqr
            median_val = self.running_median
            iqr_val = self.running_iqr
        else:
            median_val = self.median
            iqr_val = self.iqr

        if self.complex_valued and np.iscomplexobj(x):
            magnitude = np.abs(x)
            phase = np.angle(x)
            normalized_magnitude = (magnitude - median_val) / iqr_val
            return normalized_magnitude * np.exp(1j * phase)
        else:
            return (x - median_val) / iqr_val

class AdaptiveRobustNormalize(RobustNormalize):
    """Alias for RobustNormalize with adaptive=True"""
    def __init__(self, **kwargs):
        super().__init__(adaptive=True, **kwargs)

class FixedRobustNormalize(RobustNormalize):
    """Alias for RobustNormalize with adaptive=False"""
    def __init__(self, median=0.0, iqr=1.0, **kwargs):
        super().__init__(adaptive=False, median=median, iqr=iqr, **kwargs)