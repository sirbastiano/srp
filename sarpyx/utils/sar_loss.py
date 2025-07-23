"""Loss Functions for Complex-Valued SAR Image Super-Resolution.

This module provides specialized loss functions for Synthetic Aperture Radar (SAR)
image super-resolution tasks, handling both amplitude and phase components of
complex-valued SAR data.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. AMPLITUDE-DOMAIN LOSSES
# =============================================================================

def amplitude_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute L1 loss between magnitudes of predicted and target complex SAR images.
    
    This loss measures the average absolute difference in intensity, focusing on
    overall amplitude fidelity. It is robust to outliers and preserves edges,
    commonly used to ensure overall intensity accuracy in SAR super-resolution.
    
    Args:
        pred: Predicted complex SAR image tensor of shape (..., H, W) or (..., H, W, 2)
        target: Target complex SAR image tensor of same shape as pred
        
    Returns:
        Scalar tensor representing the L1 amplitude loss
        
    Note:
        If input tensors are complex, takes absolute value for magnitude.
        Otherwise assumes inputs are already magnitudes.
        
    Reference:
        Zhang et al., Remote Sensing, 2023 - combined L1 with adversarial loss
        for phase image restoration, where L1 recovered low-frequency content.
    """
    assert pred.shape == target.shape, f'Shape mismatch: pred {pred.shape} vs target {target.shape}'
    
    pred_mag = torch.abs(pred) if pred.is_complex() else pred
    target_mag = torch.abs(target) if target.is_complex() else target
    
    return torch.mean(torch.abs(pred_mag - target_mag))


def amplitude_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute L2 loss (MSE) between magnitudes of predicted and target SAR images.
    
    This loss squares the intensity differences, heavily penalizing large errors.
    L2 amplitude loss is known to improve objective metrics like PSNR, making it
    a common choice for SAR image super-resolution.
    
    Args:
        pred: Predicted complex SAR image tensor of shape (..., H, W) or (..., H, W, 2)
        target: Target complex SAR image tensor of same shape as pred
        
    Returns:
        Scalar tensor representing the L2 amplitude loss
        
    Note:
        By focusing on minimizing power differences, it ensures the reconstructed
        image's overall brightness distribution matches the reference. May produce
        over-smoothed results when used alone.
        
    Reference:
        Addabbo et al., IEEE Access, 2023 - used MSE on complex magnitudes as
        part of a hybrid loss to maximize PSNR in SAR super-resolution.
    """
    assert pred.shape == target.shape, f'Shape mismatch: pred {pred.shape} vs target {target.shape}'
    
    pred_mag = torch.abs(pred) if pred.is_complex() else pred
    target_mag = torch.abs(target) if target.is_complex() else target
    
    return torch.mean((pred_mag - target_mag) ** 2)


def amplitude_ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Compute SSIM loss between magnitudes of predicted and target SAR images.
    
    This loss evaluates structural similarity (local means, variances, covariances)
    to compare textures and edges. Using 1 - SSIM as loss encourages the model to
    produce images with high structural similarity to ground truth.
    
    Args:
        pred: Predicted complex SAR image tensor of shape (B, C, H, W) or (H, W)
        target: Target complex SAR image tensor of same shape as pred
        window_size: Size of Gaussian window for SSIM computation
        
    Returns:
        Scalar tensor representing the SSIM loss (1 - SSIM)
        
    Note:
        Particularly beneficial for SAR super-resolution as it preserves fine
        details and speckle patterns that simple pixel-wise losses might miss.
        Often combined with pixel-wise losses for balanced training.
        
    Reference:
        Zhu et al., IET Image Processing, 2019 - introduced SSIM into loss for
        SAR super-resolution to better preserve structural detail.
    """
    assert pred.shape == target.shape, f'Shape mismatch: pred {pred.shape} vs target {target.shape}'
    assert window_size % 2 == 1, f'Window size must be odd, got {window_size}'
    
    # Extract magnitudes and ensure proper dimensions
    pred_mag = torch.abs(pred) if pred.is_complex() else pred
    target_mag = torch.abs(target) if target.is_complex() else target
    
    # Add batch and channel dimensions if needed
    if pred_mag.ndim == 2:
        pred_mag = pred_mag.unsqueeze(0).unsqueeze(0)
        target_mag = target_mag.unsqueeze(0).unsqueeze(0)
    elif pred_mag.ndim == 3:
        pred_mag = pred_mag.unsqueeze(1)
        target_mag = target_mag.unsqueeze(1)
    
    device = pred_mag.device
    channels = pred_mag.shape[1]
    
    # Create 1D Gaussian kernel
    coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
    g_kernel_1d = torch.exp(-coords**2 / (2.0 * (window_size / 5)**2))
    g_kernel_1d = g_kernel_1d / g_kernel_1d.sum()
    
    # Create 2D Gaussian kernel
    gauss_kernel = g_kernel_1d[:, None] * g_kernel_1d[None, :]
    gauss_kernel = gauss_kernel.expand(channels, 1, window_size, window_size)
    
    # Compute local means
    mu1 = F.conv2d(pred_mag, gauss_kernel, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(target_mag, gauss_kernel, padding=window_size//2, groups=channels)
    mu1_sq, mu2_sq = mu1**2, mu2**2
    mu1_mu2 = mu1 * mu2
    
    # Compute local variances and covariance
    sigma1_sq = F.conv2d(pred_mag * pred_mag, gauss_kernel, padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target_mag * target_mag, gauss_kernel, padding=window_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred_mag * target_mag, gauss_kernel, padding=window_size//2, groups=channels) - mu1_mu2
    
    # SSIM constants (for dynamic range [0,1])
    c1, c2 = 1e-4, 9e-4
    
    # SSIM computation
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    ssim_val = ssim_map.mean()
    
    return 1.0 - ssim_val


def amplitude_log_cosh_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute log-cosh loss between magnitudes of predicted and target SAR images.
    
    The log-cosh loss behaves quadratically near zero error and linearly for large
    errors, making it less sensitive to outliers than L2 while still penalizing
    small errors more than L1.
    
    Args:
        pred: Predicted complex SAR image tensor of shape (..., H, W) or (..., H, W, 2)
        target: Target complex SAR image tensor of same shape as pred
        
    Returns:
        Scalar tensor representing the log-cosh amplitude loss
        
    Note:
        For SAR super-resolution, log-cosh can improve training stability by
        reducing impact of extremely high-error pixels (possibly due to speckle
        spikes). Results in smoother convergence and often sharper detail preservation.
        
    Reference:
        Navacchi et al., Remote Sensing, 2025 - applied log-cosh as loss for SAR
        backscatter modeling, noting its robustness to outliers in training.
    """
    assert pred.shape == target.shape, f'Shape mismatch: pred {pred.shape} vs target {target.shape}'
    
    pred_mag = torch.abs(pred) if pred.is_complex() else pred
    target_mag = torch.abs(target) if target.is_complex() else target
    
    diff = pred_mag - target_mag
    return torch.mean(torch.log(torch.cosh(diff)))


# =============================================================================
# 2. PHASE-DOMAIN LOSSES
# =============================================================================

def phase_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute L1 loss between phases of predicted and target complex SAR images.
    
    This loss computes mean absolute wrapped phase difference, ensuring that
    2π phase errors are not penalized (reflecting cyclic nature of phase).
    
    Args:
        pred: Predicted complex SAR image tensor
        target: Target complex SAR image tensor of same shape as pred
        
    Returns:
        Scalar tensor representing the phase L1 loss
        
    Note:
        Phase differences are wrapped to [-π, π] before computing absolute values.
        Encourages overall phase alignment and is particularly useful for preserving
        large-scale phase structures like interferometric fringes.
        
    Reference:
        Zhang et al., Remote Sensing, 2023 - used L1 phase loss to bring generated
        unwrapped phase closer to real phase, recovering low-frequency content.
    """
    assert pred.is_complex() and target.is_complex(), 'Inputs must be complex tensors for phase computation'
    assert pred.shape == target.shape, f'Shape mismatch: pred {pred.shape} vs target {target.shape}'
    
    # Compute phases
    pred_phase = torch.atan2(pred.imag, pred.real)
    target_phase = torch.atan2(target.imag, target.real)
    
    # Wrap difference to [-π, π]
    diff = pred_phase - target_phase
    diff_wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))
    
    return torch.mean(torch.abs(diff_wrapped))


def phase_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute L2 loss between phases of predicted and target complex SAR images.
    
    Similar to phase L1, but squares the wrapped phase error, emphasizing larger
    phase deviations. More strongly penalizes pixels with significant phase errors.
    
    Args:
        pred: Predicted complex SAR image tensor
        target: Target complex SAR image tensor of same shape as pred
        
    Returns:
        Scalar tensor representing the phase L2 loss
        
    Note:
        Effective for enforcing accuracy in areas with significant phase variation.
        Large deviations (e.g., shifted fringe by > π) incur high penalty.
        Often combined with phase L1 for balanced training.
        
    Reference:
        Derived from interferometric phase error correction strategies where
        wrapped phase differences are minimized for precise phase preservation.
    """
    assert pred.is_complex() and target.is_complex(), 'Inputs must be complex tensors for phase computation'
    assert pred.shape == target.shape, f'Shape mismatch: pred {pred.shape} vs target {target.shape}'
    
    pred_phase = torch.atan2(pred.imag, pred.real)
    target_phase = torch.atan2(target.imag, target.real)
    
    diff = pred_phase - target_phase
    diff_wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))
    
    return torch.mean(diff_wrapped ** 2)


def phase_wrapping_error_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute phase wrapping error loss between predicted and target SAR images.
    
    This loss focuses on penalizing phase differences corresponding to incorrect
    wrap (cycle) counts. Implemented as mean absolute wrapped phase difference.
    
    Args:
        pred: Predicted complex SAR image tensor
        target: Target complex SAR image tensor of same shape as pred
        
    Returns:
        Scalar tensor representing the phase wrapping error loss
        
    Note:
        Equals zero if only difference is integer multiple of 2π (no true error).
        Helps avoid introducing artificial 2π phase jumps and ensures continuity
        of phase field. Critical for interferogram reconstruction.
        
    Reference:
        Based on Itoh's phase continuity criterion - assumption that phase
        differences between neighboring pixels stay < π.
    """
    # This is effectively identical to phase_l1_loss
    return phase_l1_loss(pred, target)


def phase_correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute phase correlation loss between predicted and target complex SAR images.
    
    Uses complex phase coherence between images to measure how consistently
    aligned the predicted phase is with true phase. Loss = 1 - coherence.
    
    Args:
        pred: Predicted complex SAR image tensor
        target: Target complex SAR image tensor of same shape as pred
        
    Returns:
        Scalar tensor representing the phase correlation loss
        
    Note:
        Measures global similarity of fringe patterns and phase distribution.
        Sensitive to relative phase errors but ignores constant phase bias.
        Maximizing coherence preserves interferometric structure in output.
        
    Reference:
        Wei et al., Remote Sensing, 2025 - defined custom phase correlation metric
        for evaluating multi-aspect SAR image interpolation using interferometric
        coherence measure.
    """
    assert pred.is_complex() and target.is_complex(), 'Inputs must be complex tensors for phase computation'
    assert pred.shape == target.shape, f'Shape mismatch: pred {pred.shape} vs target {target.shape}'
    
    pred_phase = torch.atan2(pred.imag, pred.real)
    target_phase = torch.atan2(target.imag, target.real)
    
    # Complex exponential of phase difference
    phase_diff_complex = torch.exp(1j * (pred_phase - target_phase))
    
    # Mean coherence
    coherence = torch.abs(phase_diff_complex.mean())
    
    return 1.0 - coherence


# =============================================================================
# 3. COMPLEX-DOMAIN LOSSES
# =============================================================================

def complex_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute MSE between predicted and target complex SAR images.
    
    Treats complex pixels as 2D vectors (real & imaginary) and measures squared
    distance. Ensures super-resolved image's complex values are close to ground
    truth in both amplitude and phase.
    
    Args:
        pred: Predicted complex SAR image tensor
        target: Target complex SAR image tensor of same shape as pred
        
    Returns:
        Scalar tensor representing the complex L2 loss
        
    Note:
        Essential when output will be used for coherent processing (e.g.,
        interferometry) as it preserves complex information content. Often
        combined with structural terms for improved results.
        
    Reference:
        Addabbo et al., IEEE Access, 2023 - employed hybrid loss including
        complex MSE to retain complex image accuracy in SAR super-resolution.
    """
    assert pred.shape == target.shape, f'Shape mismatch: pred {pred.shape} vs target {target.shape}'
    
    diff = pred - target
    
    if diff.is_complex():
        # |diff|^2 = diff.real^2 + diff.imag^2
        loss = (diff.real**2 + diff.imag**2).mean()
    else:
        # Assume last dimension represents [real, imag]
        if diff.shape[-1] == 2:
            loss = (diff[..., 0]**2 + diff[..., 1]**2).mean()
        else:
            # Treat as real-valued
            loss = (diff**2).mean()
    
    return loss


def complex_correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute loss based on complex correlation between predicted and target SAR images.
    
    Calculates normalized cross-correlation and uses its magnitude (coherence)
    as similarity measure. Loss = 1 - |correlation|.
    
    Args:
        pred: Predicted complex SAR image tensor
        target: Target complex SAR image tensor of same shape as pred
        
    Returns:
        Scalar tensor representing the complex correlation loss
        
    Note:
        Encourages super-resolved image to maintain high overall coherence with
        ground truth. Useful for preserving interferometric integrity as it
        aligns complex pixel patterns globally.
        
    Reference:
        Based on InSAR coherence concept - high coherence implies strong
        correlation between two complex images.
    """
    assert pred.shape == target.shape, f'Shape mismatch: pred {pred.shape} vs target {target.shape}'
    
    # Flatten for correlation computation
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Compute complex correlation
    numerator = (pred_flat * torch.conj(target_flat)).sum()
    denom = torch.sqrt(torch.sum(torch.abs(pred_flat)**2) * torch.sum(torch.abs(target_flat)**2))
    
    corr = numerator / (denom + 1e-8)
    coherence = torch.abs(corr)
    
    return 1.0 - coherence


def coherence_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute interferometric coherence loss between predicted and target SAR images.
    
    Coherence is magnitude of complex correlation between two SAR images.
    Loss = 1 - coherence, focusing only on correlation strength, not phase offset.
    
    Args:
        pred: Predicted complex SAR image tensor
        target: Target complex SAR image tensor of same shape as pred
        
    Returns:
        Scalar tensor representing the coherence loss
        
    Note:
        Ensures super-resolved image maintains high coherence with ground truth,
        meaning fine-grained complex patterns (speckle, phase structure) are
        preserved. Critical for interferometric applications.
        
    Reference:
        Wei et al., 2025 - used coherence coefficient to validate fidelity of
        generated SAR images' phase structure.
    """
    assert pred.shape == target.shape, f'Shape mismatch: pred {pred.shape} vs target {target.shape}'
    
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Compute coherence
    numerator = torch.abs((pred_flat * torch.conj(target_flat)).mean())
    denom = torch.sqrt((torch.abs(pred_flat)**2).mean() * (torch.abs(target_flat)**2).mean())
    
    coherence_val = numerator / (denom + 1e-8)
    
    return 1.0 - coherence_val


# =============================================================================
# 4. ADVERSARIAL AND PERCEPTUAL LOSSES
# =============================================================================

def gan_generator_loss(disc_pred: torch.Tensor) -> torch.Tensor:
    """Compute adversarial loss for generator in GAN training.
    
    Generator tries to fool discriminator into thinking outputs are real.
    Uses binary cross-entropy with target "1" (real) for discriminator predictions.
    
    Args:
        disc_pred: Discriminator's output probabilities for generated images
        
    Returns:
        Scalar tensor representing the generator's adversarial loss
        
    Note:
        In SAR super-resolution, drives generator to create images with realistic
        SAR characteristics (speckle distribution, texture). Complements pixel-wise
        losses by introducing high-frequency content similar to real SAR data.
        
    Reference:
        Kong et al., Remote Sensing, 2024 - applied adversarial loss (cGAN with
        PatchGAN) to SAR super-resolution to enhance image details.
    """
    bce = nn.BCEWithLogitsLoss()
    target_real = torch.ones_like(disc_pred)
    return bce(disc_pred, target_real)


def gan_discriminator_loss(disc_real: torch.Tensor, disc_fake: torch.Tensor) -> torch.Tensor:
    """Compute discriminator loss in GAN for SAR super-resolution.
    
    Discriminator aims to output 1 for real high-resolution SAR images and 0
    for generated images. Total loss = BCE(real, 1) + BCE(fake, 0).
    
    Args:
        disc_real: Discriminator predictions on real images
        disc_fake: Discriminator predictions on generated images
        
    Returns:
        Scalar tensor representing the discriminator loss
        
    Note:
        Well-trained discriminator forces generator to produce outputs classified
        as real, improving realism of super-resolved SAR images. Usually designed
        to focus on local patches (PatchGAN) for texture details.
        
    Reference:
        Based on pix2pix GAN (Isola et al. 2017) - used PatchGAN for image-to-image
        translation, adapted for SAR with focus on speckle texture.
    """
    bce = nn.BCEWithLogitsLoss()
    target_real = torch.ones_like(disc_real)
    target_fake = torch.zeros_like(disc_fake)
    
    loss_real = bce(disc_real, target_real)
    loss_fake = bce(disc_fake, target_fake)
    
    return loss_real + loss_fake


def perceptual_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    feature_extractor: nn.Module
) -> torch.Tensor:
    """Compute perceptual feature loss using pretrained feature extractor.
    
    Images are passed through feature_extractor to obtain high-level feature maps.
    Loss is MSE between feature representations of prediction and target.
    
    Args:
        pred: Predicted SAR image tensor
        target: Target SAR image tensor of same shape as pred
        feature_extractor: Pretrained CNN for feature extraction (e.g., VGG)
        
    Returns:
        Scalar tensor representing the perceptual loss
        
    Note:
        Encourages generator to produce outputs with similar high-level
        representations. In SAR context, preserves shape outlines, textural
        patterns, and semantic content. Recently introduced to enhance
        perceptual quality in SAR super-resolution.
        
    Reference:
        Kong et al., Remote Sensing, 2024 (DMSC-GAN) - first applied perceptual
        and feature matching losses in SAR image SR, yielding improved structural
        and perceptual quality.
    """
    assert pred.shape == target.shape, f'Shape mismatch: pred {pred.shape} vs target {target.shape}'
    
    # Extract features (feature_extractor should be in eval mode and frozen)
    with torch.no_grad():
        feat_target = feature_extractor(target)
    
    feat_pred = feature_extractor(pred)
    
    # MSE in feature space
    return F.mse_loss(feat_pred, feat_target)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def combined_sar_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    amplitude_weight: float = 1.0,
    phase_weight: float = 0.5,
    complex_weight: float = 0.3,
    ssim_weight: float = 0.2
) -> torch.Tensor:
    """Compute combined SAR loss with multiple components.
    
    Combines amplitude L1, phase L1, complex L2, and SSIM losses with specified
    weights for comprehensive SAR super-resolution training.
    
    Args:
        pred: Predicted complex SAR image tensor
        target: Target complex SAR image tensor
        amplitude_weight: Weight for amplitude L1 loss
        phase_weight: Weight for phase L1 loss  
        complex_weight: Weight for complex L2 loss
        ssim_weight: Weight for SSIM loss
        
    Returns:
        Scalar tensor representing the combined loss
        
    Note:
        This is a common combination used in SAR super-resolution literature.
        Weights should be tuned based on specific application requirements.
    """
    total_loss = 0.0
    
    if amplitude_weight > 0:
        total_loss += amplitude_weight * amplitude_l1_loss(pred, target)
    
    if phase_weight > 0 and pred.is_complex() and target.is_complex():
        total_loss += phase_weight * phase_l1_loss(pred, target)
    
    if complex_weight > 0:
        total_loss += complex_weight * complex_l2_loss(pred, target)
    
    if ssim_weight > 0:
        total_loss += ssim_weight * amplitude_ssim_loss(pred, target)
    
    return total_loss
