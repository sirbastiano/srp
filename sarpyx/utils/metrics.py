"""
Performance Metrics for Complex-Valued SAR Image Super-Resolution.

This module provides comprehensive evaluation metrics for Synthetic Aperture Radar (SAR)
super-resolution algorithms. SAR super-resolution aims to enhance spatial resolution while
preserving both amplitude (magnitude) and phase information. 

The metrics are organized into three main categories:

1. **Magnitude Fidelity Metrics** (Amplitude Domain):
   - Mean Squared Error (MSE) and Root MSE (RMSE)
   - Peak Signal-to-Noise Ratio (PSNR)
   - Structural Similarity Index (SSIM)
   - Amplitude Correlation Coefficient (CC)

2. **Phase Accuracy Metrics** (Phase Domain):
   - Mean Phase Error (MAE/RMSE of Phase)
   - Complex Correlation Coefficient (Interferometric Coherence)
   - Phase-Only Correlation (Phase Consistency)

3. **SAR-Specific Structural Metrics**:
   - Equivalent Number of Looks (ENL)
   - Resolution Gain (Spatial Resolution Improvement)

All metrics accept complex-valued numpy arrays representing SAR images and provide
quantitative assessment of super-resolution quality for both research and operational
applications.

References:
    IEEE Geoscience and Remote Sensing literature for SAR image quality assessment,
    Remote Sensing journal publications on SAR super-resolution, and IEEE TGRS
    articles on interferometric coherence and SAR image processing.
"""

import numpy as np
from scipy.ndimage import uniform_filter
from typing import Tuple, Union


def _safe_divide(numerator: Union[float, np.floating], denominator: Union[float, np.floating], default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default value if denominator is zero or invalid.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value  
        default: Value to return if division is invalid
        
    Returns:
        Result of division or default value
    """
    num = float(numerator)
    den = float(denominator) 
    
    if den == 0 or not np.isfinite(den) or not np.isfinite(num):
        return default
    return num / den


def _validate_arrays(ref: np.ndarray, pred: np.ndarray) -> bool:
    """
    Validate that input arrays are suitable for metric computation.
    
    Args:
        ref: Reference array
        pred: Predicted array
        
    Returns:
        True if arrays are valid, False otherwise
    """
    # Check for NaN or infinite values
    if not (np.isfinite(ref).all() and np.isfinite(pred).all()):
        return False
    
    # Check for all-zero arrays (common with incomplete downloads)
    if np.all(ref == 0) or np.all(pred == 0):
        return False
        
    return True


# Magnitude Fidelity Metrics (Amplitude Domain)

def mse_complex(ref: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute mean squared error between two complex SAR images.
    
    Takes magnitude difference into account by using absolute difference.
    MSE directly measures fidelity in the amplitude domain and is fundamental
    for evaluating SAR super-resolution quality.
    
    Args:
        ref: Reference complex SAR image array
        pred: Predicted complex SAR image array
        
    Returns:
        MSE value as float. Lower values indicate higher amplitude fidelity.
        
    Raises:
        AssertionError: If input arrays have different shapes
    """
    assert ref.shape == pred.shape, 'Reference and predicted arrays must have same shape'
    
    error = ref - pred
    mse_val = np.mean(np.abs(error)**2)  # |error|^2 yields magnitude squared difference
    return float(mse_val)


def rmse_complex(ref: np.ndarray, pred: np.ndarray) -> float:
    """
    Root mean squared error for complex SAR images.
    
    Args:
        ref: Reference complex SAR image array
        pred: Predicted complex SAR image array
        
    Returns:
        RMSE value as float in original units
    """
    return np.sqrt(mse_complex(ref, pred))


def psnr_amplitude(ref: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute PSNR (in dB) between two complex SAR images by comparing their amplitudes.
    
    PSNR relates peak signal intensity to reconstruction error and is widely used
    in SAR super-resolution benchmarks. Higher PSNR indicates less distortion
    and closer amplitude resemblance to the true high-resolution image.
    
    Formula: PSNR = 10 * log10(MAX^2 / MSE)
    where MAX is the maximum amplitude value in the reference.
    
    Args:
        ref: Reference complex SAR image array (ground truth)
        pred: Predicted complex SAR image array (generated)
        
    Returns:
        PSNR value in decibels. Higher values indicate better quality.
        Returns inf if MSE is zero (perfect reconstruction).
        
    Raises:
        AssertionError: If input arrays have different shapes
    """
    assert ref.shape == pred.shape, 'Reference and predicted arrays must have same shape'
    
    # Compute amplitude (magnitude) images
    ref_mag = np.abs(ref)
    pred_mag = np.abs(pred)
    
    # Mean squared error on amplitude
    mse_val = np.mean((ref_mag - pred_mag) ** 2)
    if mse_val == 0 or not np.isfinite(mse_val):
        return float('inf')  # No error
        
    # Peak signal value (use max amplitude of reference as MAX)
    max_val = ref_mag.max()
    if max_val == 0 or not np.isfinite(max_val):
        return 0.0  # Invalid reference image
        
    psnr = 10 * np.log10(_safe_divide(max_val**2, mse_val, 1.0))
    return float(psnr)


def ssim_amplitude(ref: np.ndarray, pred: np.ndarray, window_size: int = 7) -> float:
    """
    Compute SSIM between two complex SAR images using their amplitudes.
    
    SSIM measures perceptual image quality by comparing local patterns of pixel
    intensities. It accounts for luminance, contrast, and structural similarity,
    correlating better with human visual assessment than MSE. In SAR super-resolution,
    SSIM indicates how well fine textures and structural details are preserved.
    
    Args:
        ref: Reference complex SAR image array
        pred: Predicted complex SAR image array  
        window_size: Size of local window for computing statistics
        
    Returns:
        Mean SSIM value (0 to 1). Higher values indicate better structural similarity.
        
    Raises:
        AssertionError: If input arrays have different shapes
    """
    assert ref.shape == pred.shape, 'Reference and predicted arrays must have same shape'
    
    x = np.abs(ref).astype(np.float64)
    y = np.abs(pred).astype(np.float64)
    
    # Local mean
    mu_x = uniform_filter(x, size=window_size)
    mu_y = uniform_filter(y, size=window_size)
    
    # Local variance and covariance
    sigma_x2 = uniform_filter(x**2, size=window_size) - mu_x**2
    sigma_y2 = uniform_filter(y**2, size=window_size) - mu_y**2
    sigma_xy = uniform_filter(x*y, size=window_size) - mu_x * mu_y
    
    # Stability constants (small fractions of dynamic range)
    c1 = (0.01 * x.max())**2
    c2 = (0.03 * x.max())**2
    
    # SSIM formula
    ssim_map = ((2*mu_x*mu_y + c1) * (2*sigma_xy + c2)) / \
               ((mu_x**2 + mu_y**2 + c1) * (sigma_x2 + sigma_y2 + c2))
    
    return float(ssim_map.mean())


def amplitude_correlation(ref: np.ndarray, pred: np.ndarray) -> float:
    """
    Pearson correlation coefficient between two images' amplitudes.
    
    Measures linear correlation between amplitude images of super-resolved output
    and ground truth. High correlation means bright and dark features (scatterers)
    in the reference are well-reproduced in the super-resolved image.
    
    Args:
        ref: Reference complex SAR image array
        pred: Predicted complex SAR image array
        
    Returns:
        Correlation coefficient (0 to 1 for non-negative amplitudes).
        Value of 1 indicates perfect linear similarity.
        
    Raises:
        AssertionError: If input arrays have different shapes
    """
    assert ref.shape == pred.shape, 'Reference and predicted arrays must have same shape'
    
    x = np.abs(ref).ravel()
    y = np.abs(pred).ravel()
    
    # Subtract mean
    x_mean = x.mean()
    y_mean = y.mean()
    
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    
    return float(_safe_divide(num, den, 0.0))


# Phase Accuracy Metrics (Phase Domain)

def phase_error_stats(ref: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    """
    Compute mean absolute error (MAE) and root mean squared error (RMSE) of phase.
    
    Directly computes phase difference between reconstructed and true complex image,
    accounting for 2π wrapping ambiguity. Low phase error indicates the generative
    model preserved phase information accurately - crucial for interferometry.
    
    Args:
        ref: Reference complex SAR image array
        pred: Predicted complex SAR image array
        
    Returns:
        Tuple of (mae, rmse) in radians. Lower values indicate higher phase fidelity.
        
    Raises:
        AssertionError: If input arrays have different shapes
    """
    assert ref.shape == pred.shape, 'Reference and predicted arrays must have same shape'
    
    # Extract phase [−π, π) of each complex pixel
    phi_ref = np.angle(ref)
    phi_pred = np.angle(pred)
    
    # Phase difference, wrapped to [-π, π]
    diff = phi_pred - phi_ref
    # Wrap the difference to [-π, π]
    diff = (diff + np.pi) % (2*np.pi) - np.pi
    
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    
    return float(mae), float(rmse)


def complex_coherence(ref: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute the complex coherence (magnitude of correlation) between two complex images.
    
    Coherence is a key interferometric SAR metric that measures correlation between
    two complex images, sensitive to both amplitude and phase differences. High
    coherence (close to 1) implies the super-resolved complex image is very similar
    to ground truth in both phase alignment and proportional amplitude.
    
    Formula: γ = |Σ(X_i * conj(Y_i))| / sqrt(Σ|X_i|^2 * Σ|Y_i|^2)
    
    Args:
        ref: Reference complex SAR image array
        pred: Predicted complex SAR image array
        
    Returns:
        Coherence value (0 to 1). Value of 1 indicates perfect complex match.
        
    Raises:
        AssertionError: If input arrays have different shapes
    """
    assert ref.shape == pred.shape, 'Reference and predicted arrays must have same shape'
    
    # Flatten to vectors
    x = ref.ravel()
    y = pred.ravel()
    
    # Compute cross-correlation and energy terms
    num = np.vdot(x, y)  # equivalent to sum(conj(x) * y)
    den = np.sqrt(np.vdot(x, x) * np.vdot(y, y))
    
    coherence = _safe_divide(np.abs(num), den, 0.0)
    return float(coherence)


def phase_coherence(ref: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute phase-only coherence between two complex images.
    
    Isolates phase agreement by disregarding amplitude differences. Measures
    the magnitude of the average phase difference vector. Value of 1 means
    the phase of every pixel matches the reference exactly (up to constant offset).
    
    Args:
        ref: Reference complex SAR image array
        pred: Predicted complex SAR image array
        
    Returns:
        Phase coherence value (0 to 1). Higher values indicate better phase consistency.
        
    Raises:
        AssertionError: If input arrays have different shapes
    """
    assert ref.shape == pred.shape, 'Reference and predicted arrays must have same shape'
    
    # Unit-magnitude versions of ref and pred
    u = ref / (np.abs(ref) + 1e-8)  # Avoid division by zero
    v = pred / (np.abs(pred) + 1e-8)
    
    # Compute average of elementwise conjugate product
    coh = np.abs(np.mean(u * np.conj(v)))
    return float(coh)


# SAR-Specific Structural Metrics

def enl(intensity_image: np.ndarray) -> float:
    """
    Compute Equivalent Number of Looks (ENL) for a given intensity image.
    
    ENL measures speckle noise in SAR image by relating mean intensity to variance
    in homogeneous regions. Formula: ENL = μ²/σ² for pixel intensities.
    Higher ENL means lower relative variance (less speckle noise).
    
    Args:
        intensity_image: 2D array of intensity values (|amplitude|²)
        
    Returns:
        ENL value. Higher values indicate less speckle noise.
        
    Raises:
        AssertionError: If input is not 2D array
    """
    assert intensity_image.ndim == 2, 'Input must be 2D intensity image'
    
    roi = intensity_image  # Assume whole image as homogeneous region for simplicity
    mean_val = roi.mean()
    var_val = roi.var()
    
    return float((mean_val**2) / (var_val + 1e-8))


def resolution_gain(orig: np.ndarray, sr: np.ndarray, threshold: float = 0.5) -> float:
    """
    Estimate resolution gain by comparing autocorrelation widths of original vs super-res images.
    
    Uses autocorrelation-based method to quantify spatial resolution improvement.
    Compares mainlobe areas above threshold in autocorrelation functions.
    Resolution gain > 1 confirms the model created finer detail than original.
    
    Args:
        orig: Original lower resolution complex SAR image array
        sr: Super-resolved complex SAR image array
        threshold: Threshold fraction (0-1) for defining mainlobe width
        
    Returns:
        Resolution gain factor. Value > 1 indicates resolution improvement.
        Value of ~2 indicates roughly double linear resolution.
        
    Raises:
        AssertionError: If input arrays have different shapes
    """
    assert orig.shape == sr.shape, 'Original and super-resolved arrays must have same shape'
    
    def autocorr_map(img: np.ndarray) -> np.ndarray:
        """Compute normalized autocorrelation map via FFT."""
        # Use power spectrum (|FFT|^2) and inverse FFT to get autocorrelation
        img = img - img.mean()  # remove mean
        corr = np.fft.ifft2(np.abs(np.fft.fft2(img))**2)
        corr = np.real(np.fft.fftshift(corr))
        # Normalize to peak=1 with protection against zero max
        max_val = corr.max()
        if max_val == 0:
            return np.zeros_like(corr)
        return corr / (max_val + 1e-8)
    
    # Check for valid input arrays
    if not _validate_arrays(orig, sr):
        return 0.0
    
    try:
        corr_orig = autocorr_map(np.abs(orig))
        corr_sr = autocorr_map(np.abs(sr))
        
        # Create boolean masks for mainlobe region above threshold
        orig_mask = corr_orig > threshold
        sr_mask = corr_sr > threshold
        
        # Count pixels in main lobe above threshold
        area_orig = int(np.sum(orig_mask))
        area_sr = int(np.sum(sr_mask))
        
        if area_sr == 0:
            return float('inf')
            
        return float(_safe_divide(area_orig, area_sr, 0.0))
    except (ValueError, RuntimeError) as e:
        return 0.0


# Convenience function for comprehensive evaluation

def evaluate_sar_metrics(ref: np.ndarray, pred: np.ndarray, 
                        window_size: int = 7, threshold: float = 0.5) -> dict:
    """
    Compute comprehensive SAR super-resolution metrics.
    
    Evaluates all major metrics for complex-valued SAR image super-resolution
    including magnitude fidelity, phase accuracy, and structural quality.
    
    Args:
        ref: Reference complex SAR image array (ground truth)
        pred: Predicted complex SAR image array (super-resolved)
        window_size: Window size for SSIM computation
        threshold: Threshold for resolution gain computation
        
    Returns:
        Dictionary containing all computed metrics with descriptive keys.
        Returns default values if arrays are invalid (e.g., from incomplete downloads).
        
    Raises:
        AssertionError: If input arrays have different shapes
    """
    assert ref.shape == pred.shape, 'Reference and predicted arrays must have same shape'
    
    # Validate arrays for common concurrency issues (incomplete downloads, etc.)
    if not _validate_arrays(ref, pred):
        # Return default metrics for invalid data (common with partial downloads)
        return {
            'mse': float('inf'), 'rmse': float('inf'), 'psnr_db': 0.0,
            'ssim': 0.0, 'amplitude_correlation': 0.0,
            'phase_mae_rad': float('inf'), 'phase_rmse_rad': float('inf'),
            'phase_mae_deg': float('inf'), 'phase_rmse_deg': float('inf'),
            'complex_coherence': 0.0, 'phase_coherence': 0.0,
            'enl_reference': 0.0, 'enl_predicted': 0.0, 'enl_ratio': 0.0,
            'resolution_gain': 0.0
        }
    
    # Magnitude fidelity metrics
    try:
        mse_val = mse_complex(ref, pred)
        rmse_val = rmse_complex(ref, pred)
        psnr_val = psnr_amplitude(ref, pred)
        ssim_val = ssim_amplitude(ref, pred, window_size)
        corr_val = amplitude_correlation(ref, pred)
    except (ValueError, ZeroDivisionError, RuntimeError) as e:
        # Handle errors from corrupted/incomplete data
        mse_val = rmse_val = float('inf')
        psnr_val = ssim_val = corr_val = 0.0
    
    # Phase accuracy metrics
    try:
        phase_mae, phase_rmse = phase_error_stats(ref, pred)
        complex_coh = complex_coherence(ref, pred)
        phase_coh = phase_coherence(ref, pred)
    except (ValueError, ZeroDivisionError, RuntimeError) as e:
        # Handle errors from corrupted/incomplete data
        phase_mae = phase_rmse = float('inf')
        complex_coh = phase_coh = 0.0
    
    # SAR-specific metrics with error handling
    try:
        ref_intensity = np.abs(ref)**2
        pred_intensity = np.abs(pred)**2
        enl_ref = enl(ref_intensity)
        enl_pred = enl(pred_intensity)
        res_gain = resolution_gain(ref, pred, threshold)
    except (ValueError, ZeroDivisionError, RuntimeError) as e:
        # Handle errors from corrupted/incomplete data
        enl_ref = enl_pred = res_gain = 0.0
    
    return {
        # Magnitude fidelity
        'mse': mse_val,
        'rmse': rmse_val,
        'psnr_db': psnr_val,
        'ssim': ssim_val,
        'amplitude_correlation': corr_val,
        
        # Phase accuracy
        'phase_mae_rad': phase_mae,
        'phase_rmse_rad': phase_rmse,
        'phase_mae_deg': np.degrees(phase_mae),
        'phase_rmse_deg': np.degrees(phase_rmse),
        'complex_coherence': complex_coh,
        'phase_coherence': phase_coh,
        
        # SAR-specific (with safe division)
        'enl_reference': enl_ref,
        'enl_predicted': enl_pred,
        'enl_ratio': _safe_divide(enl_pred, enl_ref, 0.0),
        'resolution_gain': res_gain
    }
