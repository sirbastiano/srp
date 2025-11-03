"""
Differentiable loss functions for SAR applications.

This module provides various loss functions optimized for SAR data processing,
including standard regression and classification losses, as well as specialized
losses for complex-valued SAR data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from abc import ABC, abstractmethod
import kornia.losses as losses
import kornia as K
import math
from typing import Callable, Sequence

def complex_abs(x: torch.Tensor) -> torch.Tensor:
    # works with complex dtype or real tensors representing complex values
    if torch.is_complex(x):
        return x.abs()
    else:
        # assume last dim = 2 for real/imag
        return torch.sqrt((x[..., 0] ** 2) + (x[..., 1] ** 2))

def complex_angle(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        return torch.angle(x)
    else:
        # last dim: real, imag
        return torch.atan2(x[..., 1], x[..., 0])

def angle_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # principal value difference: atan2(sin(d), cos(d))
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d))

class BaseLoss(nn.Module, ABC):
    """
    Base class for all loss functions.
    
    Args:
        reduction (str): Specifies the reduction to apply to the output.
            'none' | 'mean' | 'sum'. Default: 'mean'
    """
    
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        assert reduction in ['none', 'mean', 'sum'], f'Invalid reduction: {reduction}'
        self.reduction = reduction
    
    @abstractmethod
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss function.
        
        Args:
            prediction (torch.Tensor): Predicted values
            target (torch.Tensor): Ground truth values
            
        Returns:
            torch.Tensor: Computed loss
        """
        pass
    
    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply reduction to loss tensor."""
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class MSELoss(BaseLoss):
    """
    Mean Squared Error loss.
    
    Args:
        reduction (str): Reduction method. Default: 'mean'
    """
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss.
        
        Args:
            prediction (torch.Tensor): Predicted values of shape (N, ...)
            target (torch.Tensor): Target values of shape (N, ...)
            
        Returns:
            torch.Tensor: MSE loss
        """
        assert prediction.shape == target.shape, f'Shape mismatch: {prediction.shape} vs {target.shape}'
        loss = (prediction - target) ** 2
        return self._reduce(loss)


class DistributionAwareMSELoss(BaseLoss):
    """
    Distribution-Aware MSE Loss that weights errors by local statistics.
    
    This loss normalizes the squared error by the target's variance, effectively
    making the loss distribution-aware. It encourages the model to match both
    the reconstruction quality AND the statistical distribution of the target.
    
    Formula:
        For each pixel: loss = ((pred - target) / (sigma + eps))^2
        Where sigma = sqrt(variance of target in the batch/patch)
        
    Alternatively (normalized version):
        loss = ((pred - target) * (target - mean) / (variance + eps))^2
    
    This creates a weighted MSE where:
    - Errors in high-variance regions are weighted less (natural variation)
    - Errors in low-variance regions are weighted more (should be consistent)
    - Implicitly encourages matching the distribution's shape
    
    Args:
        normalization_mode (str): How to compute statistics
            'batch': Use batch-level mean/variance (default)
            'spatial': Use spatial (per-sample) mean/variance
            'channel': Use per-channel statistics
        use_standardization (bool): If True, uses (x - mean) / std weighting
            If False, uses 1 / std weighting (simpler). Default: True
        eps (float): Small constant for numerical stability. Default: 1e-6
        reduction (str): Reduction method. Default: 'mean'
    """
    
    def __init__(
        self, 
        normalization_mode: str = 'batch',
        use_standardization: bool = True,
        eps: float = 1e-6,
        reduction: str = 'mean'
    ) -> None:
        super().__init__(reduction)
        assert normalization_mode in ['batch', 'spatial', 'channel'], \
            f"normalization_mode must be 'batch', 'spatial', or 'channel', got {normalization_mode}"
        assert eps > 0, f'eps must be positive, got {eps}'
        
        self.normalization_mode = normalization_mode
        self.use_standardization = use_standardization
        self.eps = eps
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute distribution-aware MSE loss.
        
        Args:
            prediction (torch.Tensor): Predicted values of shape (B, ..., C) or (B, ...)
                Can be complex-valued or real-valued
            target (torch.Tensor): Target values of same shape as prediction
            
        Returns:
            torch.Tensor: Distribution-aware MSE loss
        """
        assert prediction.shape == target.shape, f'Shape mismatch: {prediction.shape} vs {target.shape}'
        
        # Convert complex to magnitude for statistics computation
        if torch.is_complex(prediction):
            pred_mag = prediction.abs()
            target_mag = target.abs()
        elif prediction.shape[-1] == 2:
            # Assume last dim is [real, imag]
            pred_mag = torch.sqrt(prediction[..., 0]**2 + prediction[..., 1]**2 + self.eps)
            target_mag = torch.sqrt(target[..., 0]**2 + target[..., 1]**2 + self.eps)
        else:
            pred_mag = prediction
            target_mag = target
        
        # Compute statistics based on normalization mode (using magnitudes)
        if self.normalization_mode == 'batch':
            # Statistics over entire batch (all dimensions)
            mean_target = target_mag.mean()
            var_target = target_mag.var()
        elif self.normalization_mode == 'spatial':
            # Statistics per sample (over spatial dimensions, keep batch separate)
            if target_mag.dim() == 4:  # (B, C, H, W)
                mean_target = target_mag.mean(dim=[2, 3], keepdim=True)
                var_target = target_mag.var(dim=[2, 3], keepdim=True)
            elif target_mag.dim() == 3:  # (B, L, C) or (B, H, W)
                mean_target = target_mag.mean(dim=1, keepdim=True)
                var_target = target_mag.var(dim=1, keepdim=True)
            else:
                # Fallback to batch statistics
                mean_target = target_mag.mean()
                var_target = target_mag.var()
        elif self.normalization_mode == 'channel':
            # Statistics per channel
            if target_mag.dim() >= 2:
                # Compute over all dims except batch and channel
                reduce_dims = list(range(2, target_mag.dim()))
                if len(reduce_dims) > 0:
                    mean_target = target_mag.mean(dim=reduce_dims, keepdim=True)
                    var_target = target_mag.var(dim=reduce_dims, keepdim=True)
                else:
                    mean_target = target_mag.mean(dim=1, keepdim=True)
                    var_target = target_mag.var(dim=1, keepdim=True)
            else:
                mean_target = target_mag.mean()
                var_target = target_mag.var()
        
        std_target = torch.sqrt(var_target + self.eps)
        
        # Compute squared error (on magnitudes)
        squared_error = (pred_mag - target_mag) ** 2
        
        if self.use_standardization:
            # Weight by standardized target: (x - mean) / std
            # This emphasizes matching the distribution shape
            target_normalized = (target_mag - mean_target) / std_target
            weighted_error = squared_error * target_normalized ** 2
        else:
            # Weight by inverse variance: 1 / std^2
            # Simpler, focuses on relative error magnitude
            weighted_error = squared_error / (var_target + self.eps)
        
        return self._reduce(weighted_error)


class MAELoss(BaseLoss):
    """
    Mean Absolute Error loss.
    
    Args:
        reduction (str): Reduction method. Default: 'mean'
    """
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MAE loss.
        
        Args:
            prediction (torch.Tensor): Predicted values of shape (N, ...)
            target (torch.Tensor): Target values of shape (N, ...)
            
        Returns:
            torch.Tensor: MAE loss
        """
        assert prediction.shape == target.shape, f'Shape mismatch: {prediction.shape} vs {target.shape}'
        loss = torch.abs(prediction - target)
        return self._reduce(loss)


class HuberLoss(BaseLoss):
    """
    Huber loss (smooth L1 loss).
    
    Args:
        delta (float): Threshold for switching between quadratic and linear loss. Default: 1.0
        reduction (str): Reduction method. Default: 'mean'
    """
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean') -> None:
        super().__init__(reduction)
        assert delta > 0, f'Delta must be positive, got {delta}'
        self.delta = delta
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Huber loss.
        
        Args:
            prediction (torch.Tensor): Predicted values of shape (N, ...)
            target (torch.Tensor): Target values of shape (N, ...)
            
        Returns:
            torch.Tensor: Huber loss
        """
        assert prediction.shape == target.shape, f'Shape mismatch: {prediction.shape} vs {target.shape}'
        residual = torch.abs(prediction - target)
        loss = torch.where(
            residual < self.delta,
            0.5 * residual ** 2,
            self.delta * residual - 0.5 * self.delta ** 2
        )
        return self._reduce(loss)


class FocalLoss(BaseLoss):
    """
    Focal Loss for addressing class imbalance.
    
    Args:
        alpha (float, optional): Weighting factor for rare class. Default: 1.0
        gamma (float): Focusing parameter. Default: 2.0
        reduction (str): Reduction method. Default: 'mean'
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean') -> None:
        super().__init__(reduction)
        assert gamma >= 0, f'Gamma must be non-negative, got {gamma}'
        assert alpha > 0, f'Alpha must be positive, got {alpha}'
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal loss.
        
        Args:
            prediction (torch.Tensor): Predicted logits of shape (N, C)
            target (torch.Tensor): Target class indices of shape (N,)
            
        Returns:
            torch.Tensor: Focal loss
        """
        assert prediction.dim() == 2, f'Prediction must be 2D, got {prediction.dim()}D'
        assert target.dim() == 1, f'Target must be 1D, got {target.dim()}D'
        assert prediction.size(0) == target.size(0), f'Batch size mismatch: {prediction.size(0)} vs {target.size(0)}'
        
        ce_loss = F.cross_entropy(prediction, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return self._reduce(focal_loss)


class ComplexMSELoss(BaseLoss):
    """
    Mean Squared Error loss for complex-valued tensors.
    
    Args:
        reduction (str): Reduction method. Default: 'mean'
    """
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss for complex tensors.
        
        Args:
            prediction (torch.Tensor): Predicted complex values of shape (N, ...)
            target (torch.Tensor): Target complex values of shape (N, ...)
            
        Returns:
            torch.Tensor: Complex MSE loss (real-valued)
        """
        assert prediction.shape == target.shape, f'Shape mismatch: {prediction.shape} vs {target.shape}'
        assert prediction.is_complex(), 'Prediction tensor must be complex'
        assert target.is_complex(), 'Target tensor must be complex'
        
        diff = prediction - target
        loss = torch.abs(diff) ** 2
        return self._reduce(loss)

class ComplexMAELoss(BaseLoss):
    """
    Mean Absolute Error loss for complex-valued tensors.

    Args:
        reduction (str): Reduction method. Default: 'mean'
    """

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MAE loss for complex tensors.

        Args:
            prediction (torch.Tensor): Predicted complex values of shape (N, ...)
            target (torch.Tensor): Target complex values of shape (N, ...)
        """
        assert prediction.shape == target.shape, f'Shape mismatch: {prediction.shape} vs {target.shape}'
        assert prediction.is_complex(), 'Prediction tensor must be complex'
        assert target.is_complex(), 'Target tensor must be complex'

        diff = prediction - target
        loss = torch.abs(diff)
        return self._reduce(loss)

class PhaseLoss(BaseLoss):
    """
    Phase-based loss for complex-valued tensors.
    
    Args:
        reduction (str): Reduction method. Default: 'mean'
    """
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute phase difference loss.
        
        Args:
            prediction (torch.Tensor): Predicted complex values of shape (N, ...)
            target (torch.Tensor): Target complex values of shape (N, ...)
            
        Returns:
            torch.Tensor: Phase loss
        """
        assert prediction.shape == target.shape, f'Shape mismatch: {prediction.shape} vs {target.shape}'
        assert prediction.is_complex(), 'Prediction tensor must be complex'
        assert target.is_complex(), 'Target tensor must be complex'
        
        pred_phase = torch.angle(prediction)
        target_phase = torch.angle(target)
        
        # Compute circular distance between phases
        phase_diff = torch.abs(pred_phase - target_phase)
        phase_diff = torch.min(phase_diff, 2 * torch.pi - phase_diff)
        
        return self._reduce(phase_diff)


class CombinedComplexLoss(BaseLoss):
    """
    Combined magnitude and phase loss for complex-valued tensors.
    
    Args:
        magnitude_weight (float): Weight for magnitude loss. Default: 1.0
        phase_weight (float): Weight for phase loss. Default: 1.0
        reduction (str): Reduction method. Default: 'mean'
    """
    
    def __init__(
        self, 
        magnitude_weight: float = 1.0, 
        phase_weight: float = 1.0, 
        reduction: str = 'mean'
    ) -> None:
        super().__init__(reduction)
        assert magnitude_weight >= 0, f'Magnitude weight must be non-negative, got {magnitude_weight}'
        assert phase_weight >= 0, f'Phase weight must be non-negative, got {phase_weight}'
        
        self.magnitude_weight = magnitude_weight
        self.phase_weight = phase_weight
        self.complex_mse = ComplexMSELoss(reduction='none')
        self.phase_loss = PhaseLoss(reduction='none')
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined magnitude and phase loss.
        
        Args:
            prediction (torch.Tensor): Predicted complex values of shape (N, ...)
            target (torch.Tensor): Target complex values of shape (N, ...)
            
        Returns:
            torch.Tensor: Combined loss
        """
        mag_loss = self.complex_mse(prediction, target)
        phase_loss = self.phase_loss(prediction, target)
        
        combined_loss = (
            self.magnitude_weight * mag_loss + 
            self.phase_weight * phase_loss
        )
        
        return self._reduce(combined_loss)
class MagnitudeL1Loss(BaseLoss):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # operate on magnitudes
        pred_mag = prediction.abs() if prediction.is_complex() else prediction
        tgt_mag = target.abs() if target.is_complex() else target
        loss = torch.abs(pred_mag - tgt_mag)
        return self._reduce(loss)

class LogMagMSELoss(BaseLoss):
    """MSE on log(1 + alpha*|z|) to preserve dynamic range."""
    def __init__(self, alpha:float=1e-3, reduction='mean'):
        super().__init__(reduction)
        self.alpha = alpha
    def forward(self, prediction, target):
        pm = prediction.abs() if prediction.is_complex() else prediction
        tm = target.abs() if target.is_complex() else target
        pl = torch.log1p(self.alpha * pm)
        tl = torch.log1p(self.alpha * tm)
        return self._reduce((pl - tl)**2)

class PhaseLossMasked(BaseLoss):
    """
    Angular difference, weighted by target magnitude to avoid tiny-mag noise.

    Expects:
      - prediction and target either complex tensors (dtype=torch.cfloat / torch.cdouble)
        OR real tensors where the last dimension is size 2 and contains [real, imag],
        e.g. shape (B, L, 2). If your real tensors are (B, 2, L) adapt before calling.
    """
    def __init__(self, mag_threshold: float = 1e-3, reduction: str = 'mean'):
        super().__init__(reduction)
        self.th = float(mag_threshold)

    def _to_complex(self, z: torch.Tensor) -> torch.Tensor:
        # Convert real pair representation (last dim = 2) to complex, otherwise return as-is.
        if z.is_complex():
            return z
        if z.dim() >= 1 and z.shape[-1] == 2:
            return torch.view_as_complex(z)
        raise ValueError("Input must be a complex tensor or real tensor with last-dim == 2 ([real,imag]).")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.shape == target.shape, f"Shape mismatch {pred.shape} vs {target.shape}"

        # Convert to complex if necessary
        pred_c = self._to_complex(pred)
        target_c = self._to_complex(target)

        # Compute magnitudes and mask
        tgt_mag = target_c.abs()                       # shape (...), real dtype
        mask = (tgt_mag > self.th).to(tgt_mag.dtype)   # same dtype as magnitudes

        # Compute phase difference (wrapped to [-pi, pi])
        pred_phase = torch.angle(pred_c)
        tgt_phase  = torch.angle(target_c)
        phase_diff = torch.abs(pred_phase - tgt_phase)

        # Use tensor PI on same device/dtype to avoid any casting issues
        PI = torch.tensor(math.pi, device=phase_diff.device, dtype=phase_diff.dtype)
        phase_diff = torch.minimum(phase_diff, 2.0 * PI - phase_diff)

        # Weighted (masked) sum per sample
        # We'll treat the 0-th dim as batch; if your data has no batch, this still works with batch=first dim
        bsz = phase_diff.shape[0]
        # flatten everything except batch
        rest = phase_diff.shape[1:]
        n_elems = int(torch.tensor(rest).prod().item()) if len(rest) > 0 else 1

        phase_flat = phase_diff.reshape(bsz, -1)      # (B, N)
        mask_flat  = mask.reshape(bsz, -1)            # (B, N)

        num = (phase_flat * mask_flat).sum(dim=1)     # (B,)
        denom = mask_flat.sum(dim=1).clamp_min(1.0)   # (B,) avoid division by 0

        loss_per_sample = num / denom                 # (B,)

        # Use BaseLoss's reduction (mean/sum/none)
        return self._reduce(loss_per_sample)

class EdgeLoss(BaseLoss): # Currently not using this
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        # Compute gradients (edges) for predictions
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        # Compute gradients (edges) for targets
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        # Compute L1 loss between gradients
        loss_x = self.l1_loss(pred_dx, target_dx)
        loss_y = self.l1_loss(pred_dy, target_dy)

        return loss_x + loss_y

def gaussian_1d_kernel(window_size: int, sigma: Optional[float] = None, device=None, dtype=torch.float32):
    if sigma is None:
        sigma = 0.5 * window_size
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2.
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.view(1, 1, -1)  # shape (1,1,L) for conv1d

class SSIM1DLoss(BaseLoss):
    def __init__(self, window_size=11, sigma=None, K1=0.01, K2=0.03,
                 data_range=1.0, eps=1e-6, reduction='mean'):
        super().__init__(reduction)
        assert window_size % 2 == 1, "window_size should be odd"
        self.wsize = window_size
        self.sigma = sigma
        self.K1 = float(K1)
        self.K2 = float(K2)
        self.data_range = float(data_range)
        self.eps = float(eps)
        self.C1 = (self.K1 * self.data_range) ** 2
        self.C2 = (self.K2 * self.data_range) ** 2

    def forward(self, pred, target):
        # normalize shapes to (B, C, L)
        if pred.ndim == 2:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)
        # ensure shapes (B,C,L)
        if pred.ndim == 4 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
            target = target.squeeze(-1)
        if pred.ndim == 3 and pred.shape[-1] == 1:
            pred = pred.permute(0, 2, 1)
            target = target.permute(0, 2, 1)

        pred = pred.to(dtype=target.dtype, device=target.device)

        B, C, L = pred.shape
        device = pred.device
        dtype = pred.dtype

        # kernel
        win = gaussian_1d_kernel(self.wsize, self.sigma, device=device, dtype=dtype)  # (1,1,K)
        win = win.repeat(C, 1, 1)  # (C,1,K)
        pad = self.wsize // 2

        # local stats
        mu1 = F.conv1d(pred, win, padding=pad, groups=C)
        mu2 = F.conv1d(target, win, padding=pad, groups=C)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv1d(pred * pred, win, padding=pad, groups=C) - mu1_sq
        sigma2_sq = F.conv1d(target * target, win, padding=pad, groups=C) - mu2_sq
        sigma12 = F.conv1d(pred * target, win, padding=pad, groups=C) - mu1_mu2

        # Numerical stability: clamp variances to >= 0
        sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

        # SSIM map
        num = (2.0 * mu1_mu2 + self.C1) * (2.0 * sigma12 + self.C2)
        den = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)

        ssim_map = num / (den + self.eps)   # add eps to avoid div-by-zero
        # clamp to [0,1]
        ssim_map = torch.clamp(ssim_map, min=0.0, max=1.0)

        ssim_val = ssim_map.mean(dim=[1,2])   # per-batch mean
        loss = 1.0 - ssim_val                 # non-negative

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
class ComplexSSIMCombinedLoss(BaseLoss):
    def __init__(self, alpha=0.0, beta=1.0, gamma=1.0, delta=0.1, ssim_window=11, reduction='mean'):
        super().__init__(reduction)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.ssim = SSIM1DLoss(window_size=ssim_window, reduction='mean')
        self.cmse = ComplexMSELoss(reduction='none')
        self.mag_l1 = MagnitudeL1Loss(reduction='none')
        self.phase_mask = PhaseLossMasked(reduction='none')

    def forward(self, pred, tgt):
        # per-pixel terms
        cmse_scalar = self.cmse(pred, tgt).mean()
        magl1_scalar = self.mag_l1(pred, tgt).mean()

        # SSIM on magnitudes; ensure proper shape (B,L) -> (B,1,L)
        pred_mag = (pred.abs() if pred.is_complex() else pred).to(dtype=cmse_scalar.dtype, device=cmse_scalar.device)
        tgt_mag  = (tgt.abs()  if tgt.is_complex()  else tgt).to(dtype=cmse_scalar.dtype, device=cmse_scalar.device)
        # call ssim (returns scalar if reduction='mean')
        ssim_loss = self.ssim(pred_mag, tgt_mag)  # already 1 - ssim averaged per batch

        phase_scalar = self.phase_mask(pred, tgt)

        loss = self.alpha * cmse_scalar + self.beta * magl1_scalar + self.gamma * ssim_loss + self.delta * phase_scalar
        # loss should be non-negative; nevertheless clamp small negative numerical noise to 0
        loss = torch.clamp(loss, min=0.0)
        return self._reduce(loss)
    

class SplitRealImagLoss(BaseLoss):
    """Lsplit = ||Re(X^) - Re(X)||^2 + ||Im(X^) - Im(X)||^2"""
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(prediction):
            loss = (prediction.real - target.real).pow(2) + (prediction.imag - target.imag).pow(2)
        else:
            # assume last dim size 2: [..., 2] = (real, imag)
            loss = (prediction[..., 0] - target[..., 0]).pow(2) + (prediction[..., 1] - target[..., 1]).pow(2)
        return self._reduce(loss)


class PolarLoss(BaseLoss):
    """
    Lpolar = w_mag * |||X^| - |X|||^2 + w_phase * ||phase_diff||^2
    Uses robust principal-value angle-difference.
    """
    def __init__(self, w_mag: float = 1.0, w_phase: float = 1.0, reduction: str = 'mean') -> None:
        super().__init__(reduction=reduction)
        self.w_mag = w_mag
        self.w_phase = w_phase

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mag_pred = complex_abs(prediction)
        mag_tgt = complex_abs(target)
        mag_loss = (mag_pred - mag_tgt).pow(2)

        ang_pred = complex_angle(prediction)
        ang_tgt = complex_angle(target)
        ang_diff = angle_difference(ang_pred, ang_tgt)
        phase_loss = ang_diff.pow(2)

        loss = self.w_mag * mag_loss + self.w_phase * phase_loss
        return self._reduce(loss)


class ComplexMSELoss(BaseLoss):
    """LcMSE = ||X^ - X||^2 (complex squared error)"""
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(prediction):
            diff_sq = (prediction.real - target.real).pow(2) + (prediction.imag - target.imag).pow(2)
        else:
            diff_sq = (prediction - target).pow(2)
        return self._reduce(diff_sq)


class RobustComplexLoss(BaseLoss):
    """
    Robust alternatives. Two modes:
      - 'quartic' -> sum |diff|^4
      - 'logcosh' -> sum log(cosh(|diff|))
    """
    def __init__(self, mode: str = 'logcosh', reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        assert mode in ['quartic', 'logcosh'], "mode must be 'quartic' or 'logcosh'"
        self.mode = mode

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = prediction - target
        mag = complex_abs(diff)
        if self.mode == 'quartic':
            loss = mag.pow(4)
        else:
            # logcosh(|diff|)
            # log(cosh(x)) is approx. x^2/2 for small x and linear for large x
            # to keep numerical stability clip magnitude
            loss = torch.log(torch.cosh(mag + 1e-12))
        return self._reduce(loss)


class LogMagnitudeLoss(BaseLoss):
    """
    Llogmag = || ln|X^| - ln|X| ||^2
    Avoids -inf at zeros with eps.
    """
    def __init__(self, eps: float = 1e-6, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mag_pred = complex_abs(prediction).clamp(min=self.eps)
        mag_tgt = complex_abs(target).clamp(min=self.eps)
        loss = (torch.log(mag_pred) - torch.log(mag_tgt)).pow(2)
        return self._reduce(loss)


class SymmetryConstrainedLoss(BaseLoss):
    """
    L = ||X^ - X||^2 + lambda * ||X^ - S(X^)||^2,
    where S is a symmetry operator (callable). If S is None, defaults to conjugate transpose along last two dims (Hermitian)
    """
    def __init__(self, lam: float = 1.0, symmetry_op: Optional[Callable[[torch.Tensor], torch.Tensor]] = None, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.lam = lam
        self.symmetry_op = symmetry_op

    def default_symmetry(self, x: torch.Tensor) -> torch.Tensor:
        # Attempt a generic conjugate symmetry: conj and reverse spatial dims if 2D
        if torch.is_complex(x):
            return x.conj()
        else:
            # if real-imag stacked: flip imag sign
            out = x.clone()
            out[..., 1] = -out[..., 1]
            return out

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        recon_loss = ComplexMSELoss(reduction='none')(prediction, target)
        op = self.symmetry_op or self.default_symmetry
        sym_loss = ComplexMSELoss(reduction='none')(prediction, op(prediction))
        loss = recon_loss + self.lam * sym_loss
        return self._reduce(loss)


class PolarimetricDecompositionLoss(BaseLoss):
    """
    For PolSAR: expects predictions and targets as sequence/tuple/list of complex components cpred_i.
    Usage: pass prediction as tensor with extra channel dim or a sequence of component tensors.
    Simple implementation supports two styles:
     - prediction/target: (B, C, H, W) complex with C = number of polarimetric channels
     - or prediction/target: list/tuple of length C of (B, H, W) complex tensors
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction=reduction)

    def forward(self, prediction, target):
        if isinstance(prediction, (list, tuple)):
            # list of components
            loss = 0.0
            for p, t in zip(prediction, target):
                loss = loss + ComplexMSELoss(reduction='none')(p, t)
            return self._reduce(loss)
        else:
            # assume tensor shape (B, C, H, W)
            # compute per-channel complex mse and sum over channels
            # if real layout (B, C, H, W, 2) treat accordingly
            if torch.is_complex(prediction):
                diff_sq = (prediction.real - target.real).pow(2) + (prediction.imag - target.imag).pow(2)
            else:
                diff_sq = (prediction - target).pow(2)
            # sum across channel dimension (1)
            loss = diff_sq.sum(dim=1)
            return self._reduce(loss)


class PowerLoss(BaseLoss):
    """
    Lpower = (sum |X^|^2 - sum |X|^2)^2
    Computes global power difference per-batch.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction=reduction)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        P_pred = complex_abs(prediction).pow(2).sum(dim=list(range(1, prediction.dim())))  # sum over non-batch dims
        P_tgt = complex_abs(target).pow(2).sum(dim=list(range(1, target.dim())))
        diff_sq = (P_pred - P_tgt).pow(2)
        # diff_sq is per-batch scalar; we want to reduce according to reduction
        if self.reduction == 'none':
            return diff_sq
        return self._reduce(diff_sq)


class SpeckleNLLLoss(BaseLoss):
    """
    Negative log-likelihood for circular complex Gaussian speckle:
    p(x | mu, sigma^2) = 1/(pi sigma^2) * exp(-|x-mu|^2 / sigma^2)
    NLL per pixel: ln(pi sigma^2) + |x-mu|^2 / sigma^2
    Two modes:
       - fixed_sigma: scalar sigma provided
       - predicted_sigma: second output channel gives variance estimate
    prediction can be:
       - complex tensor (mu) and sigma given as float
       - tuple (mu, sigma_pred) where sigma_pred is real non-negative (same shape as mu magnitude)
    """
    def __init__(self, sigma: Optional[float] = None, eps: float = 1e-6, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.sigma = sigma
        self.eps = eps

    def forward(self, prediction, target):
        if isinstance(prediction, (tuple, list)) and len(prediction) == 2:
            mu, sigma_pred = prediction
            sigma2 = sigma_pred.clamp(min=self.eps)
        else:
            mu = prediction
            if self.sigma is None:
                raise ValueError("Either pass (mu, sigma_pred) or set fixed sigma in constructor")
            sigma2 = torch.tensor(self.sigma ** 2, device=mu.device, dtype=mu.dtype)

        mag_sq = (complex_abs(target - mu)).pow(2)
        # if sigma2 is scalar tensor expand to match
        if sigma2.dim() == 0:
            nll = torch.log(torch.tensor(torch.pi, device=mu.device, dtype=mu.dtype) * sigma2) + mag_sq / sigma2
        else:
            nll = torch.log(torch.pi * sigma2) + mag_sq / sigma2
        return self._reduce(nll)


class ComplexTVLoss(BaseLoss):
    """
    Isotropic TV on complex-valued image:
    sum_{m,n} (|X_{m+1,n} - X_{m,n}| + |X_{m,n+1} - X_{m,n}|)
    Implemented with magnitude of complex differences.
    """
    def forward(self, prediction: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        # target not used; TV is regularizer on prediction
        x = prediction
        # compute forward differences
        dx = x[..., 1:] - x[..., :-1]
        dy = x[..., :, 1:] - x[..., :, :-1] if x.dim() >= 3 else (x[..., :, 1:] - x[..., :, :-1])
        # Because indexing differs with dims, we compute generically:
        # We'll compute differences along last two spatial dims if available.
        if x.dim() < 2:
            raise ValueError("Input must have at least 2 dims (batch, spatial...)")
        # We assume shape (B, H, W) or (B, C, H, W)
        if x.dim() >= 3:
            # spatial dims assumed last two
            dh = x[..., 1:, :] - x[..., :-1, :]
            dw = x[..., :, 1:] - x[..., :, :-1]
            loss = complex_abs(dh).sum(-1).sum(-1) if torch.is_complex(x) else complex_abs(dh).sum()
            # but better: compute elementwise and sum all
            loss = complex_abs(dh)
            loss = loss.sum()
            loss = loss + complex_abs(dw).sum()
        else:
            # fallback 1D difference
            loss = complex_abs(dx).sum()
        # return scalar or per-batch?
        # We'll average per-batch if reduction == 'mean', else sum or none (none not well-defined here)
        if self.reduction == 'none':
            return loss  # note: returns scalar; user can adapt
        return self._reduce(loss)


class GradientMatchingLoss(BaseLoss):
    """
    Lgrad = ||grad |X^| - grad |X|||^2 + ||grad angle(X^) - grad angle(X)||^2
    Uses simple finite differences to compute gradients (sobel-ish could be added).
    """
    def __init__(self, w_mag: float = 1.0, w_phase: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.w_mag = w_mag
        self.w_phase = w_phase

    def _spatial_grads(self, t: torch.Tensor):
        # expects shape (B, H, W) or (B, C, H, W)
        # compute simple forward differences on last two dims
        if t.dim() < 3:
            raise ValueError("Tensor must be at least 3D (B, H, W)")
        gx = t[..., 1:, :] - t[..., :-1, :]
        gy = t[..., :, 1:] - t[..., :, :-1]
        # pad to same shape
        gx = F.pad(gx, (0, 0, 0, 1))
        gy = F.pad(gy, (0, 1, 0, 0))
        return gx, gy

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mag_p = complex_abs(prediction)
        mag_t = complex_abs(target)
        ang_p = complex_angle(prediction)
        ang_t = complex_angle(target)

        gmag_p_x, gmag_p_y = self._spatial_grads(mag_p)
        gmag_t_x, gmag_t_y = self._spatial_grads(mag_t)
        grad_mag_loss = (gmag_p_x - gmag_t_x).pow(2) + (gmag_p_y - gmag_t_y).pow(2)

        gang_p_x, gang_p_y = self._spatial_grads(ang_p)
        gang_t_x, gang_t_y = self._spatial_grads(ang_t)
        # angle differences wrapped
        d1 = angle_difference(gang_p_x, gang_t_x).pow(2)
        d2 = angle_difference(gang_p_y, gang_t_y).pow(2)
        grad_phase_loss = d1 + d2

        loss = self.w_mag * grad_mag_loss + self.w_phase * grad_phase_loss
        return self._reduce(loss)


class PhaseSmoothnessLoss(BaseLoss):
    """
    Penalize Laplacian (second-order differences) of the phase to encourage smooth phase.
    Lphase = sum (Delta angle(Xhat))^2
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction=reduction)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        ang = complex_angle(prediction)
        # discrete Laplacian on last two dims
        # assume (B, H, W) or (B, C, H, W)
        # create padded version for neighbors
        pad = (1, 1, 1, 1)  # pad left,right,top,bottom
        ang_p = F.pad(ang, pad, mode='replicate')
        center = ang_p[..., 1:-1, 1:-1]
        up = ang_p[..., :-2, 1:-1]
        down = ang_p[..., 2:, 1:-1]
        left = ang_p[..., 1:-1, :-2]
        right = ang_p[..., 1:-1, 2:]
        lap = (up + down + left + right - 4 * center)
        loss = angle_difference(lap, torch.zeros_like(lap)).pow(2)
        return self._reduce(loss)


class GANLoss(BaseLoss):
    """
    Standard GAN loss wrappers. Expects discriminator outputs.
    mode:
      - 'vanilla' -> BCE with logits
      - 'hinge' -> hinge loss
    The forward signature:
      - forward(pred_fake_logits, pred_real_logits=None)
      or provide labels tensors.
    """
    def __init__(self, mode: str = 'vanilla', reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        assert mode in ['vanilla', 'hinge']
        self.mode = mode
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def fake_loss(self, pred_fake: torch.Tensor) -> torch.Tensor:
        if self.mode == 'vanilla':
            labels = torch.zeros_like(pred_fake)
            return self._reduce(self.bce(pred_fake, labels))
        else:
            # hinge
            return self._reduce(F.relu(1.0 + pred_fake).mean())

    def real_loss(self, pred_real: torch.Tensor) -> torch.Tensor:
        if self.mode == 'vanilla':
            labels = torch.ones_like(pred_real)
            return self._reduce(self.bce(pred_real, labels))
        else:
            return self._reduce(F.relu(1.0 - pred_real).mean())

    def forward(self, pred_fake: torch.Tensor, pred_real: Optional[torch.Tensor] = None, which: str = 'both') -> torch.Tensor:
        """
        which: 'both', 'gen' (generator wants fake to be real), 'disc' (compute discriminator loss using pred_real and pred_fake)
        If which == 'gen' -> generator loss = -E[log D(fake)] (vanilla) approximated by BCE with target ones.
        """
        if which == 'gen':
            if self.mode == 'vanilla':
                labels = torch.ones_like(pred_fake)
                return self._reduce(self.bce(pred_fake, labels))
            else:
                # hinge generator loss
                return self._reduce((-pred_fake).mean())
        else:
            # discriminator loss
            if pred_real is None:
                raise ValueError("pred_real must be provided for discriminator loss")
            real_l = self.real_loss(pred_real)
            fake_l = self.fake_loss(pred_fake)
            return real_l + fake_l


class FeatureLoss(BaseLoss):
    """
    Lfeat = ||Phi(X^) - Phi(X)||^2
    feature_extractor should accept complex or real tensors and produce a real feature tensor.
    Provide preprocess_fn if conversion from complex to feature extractor input is needed.
    """
    def __init__(self, feature_extractor: Callable[[torch.Tensor], torch.Tensor], preprocess_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.feature_extractor = feature_extractor
        self.preprocess_fn = preprocess_fn

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_in = prediction if self.preprocess_fn is None else self.preprocess_fn(prediction)
        tgt_in = target if self.preprocess_fn is None else self.preprocess_fn(target)
        feat_pred = self.feature_extractor(pred_in)
        feat_tgt = self.feature_extractor(tgt_in)
        loss = (feat_pred - feat_tgt).pow(2)
        return self._reduce(loss)


class CompositeLoss(BaseLoss):
    """
    Composite weighted combination of sub-losses:
    L = sum_i alpha_i * Li
    sub_losses: sequence of (weight, loss_instance)
    """
    def __init__(self, sub_losses: Sequence[tuple], reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        # sub_losses: [(weight, loss_module), ...]
        self.sub_losses = nn.ModuleList([l for _, l in sub_losses])
        self.weights = [w for w, _ in sub_losses]

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = 0.0
        # use each loss with reduction='none' to combine spatially/elementwise if possible
        for w, loss_module in zip(self.weights, self.sub_losses):
            # temporarily force none to get elementwise loss if module supports it
            # if module has attribute reduction, set and restore
            prev_reduction = getattr(loss_module, 'reduction', None)
            try:
                loss_module.reduction = 'none'
            except Exception:
                pass
            l = loss_module(prediction, target)
            # restore
            if prev_reduction is not None:
                loss_module.reduction = prev_reduction
            total = total + w * l
        return self._reduce(total)


class MultiDomainSARLoss(BaseLoss):
    """
    Multi-domain loss for physics-aware SAR compression.
    Combines spatial, frequency, phase, sparsity, rate, SSIM, perceptual, and edge losses.
    
    ENHANCED VERSION (Oct 2024): Added structure-preserving losses for better detail retention.
    
    Designed for use with PhysicsAwareSpatialTransformer model.
    
    Args:
        spatial_weight: Weight for spatial domain loss (default: 1.0)
        frequency_weight: Weight for frequency domain loss (default: 0.3)
        phase_weight: Weight for phase consistency loss (default: 0.5)
        sparsity_weight: Weight for sparsity penalty (default: 0.01)
        rate_weight: Weight for rate penalty (default: 0.01)
        ssim_weight: Weight for SSIM loss (default: 0.0)
        perceptual_weight: Weight for perceptual loss (default: 0.0)
        edge_weight: Weight for edge-preserving loss (default: 0.0)
        reduction: Loss reduction method ('mean', 'sum', 'none')
    """
    def __init__(
        self,
        spatial_weight: float = 1.0,
        frequency_weight: float = 0.3,
        phase_weight: float = 0.5,
        sparsity_weight: float = 0.01,
        rate_weight: float = 0.01,
        ssim_weight: float = 0.0,
        perceptual_weight: float = 0.0,
        edge_weight: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__(reduction=reduction)
        self.spatial_weight = spatial_weight
        self.frequency_weight = frequency_weight
        self.phase_weight = phase_weight
        self.sparsity_weight = sparsity_weight
        self.rate_weight = rate_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
    
    def complex_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Complex MSE: |pred - target|^2"""
        return F.mse_loss(pred, target, reduction=self.reduction)
    
    def frequency_domain_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE in azimuth frequency domain."""
        # Handle both complex tensor format [B, H, W, 2] and native complex
        if pred.dtype in [torch.complex64, torch.complex128]:
            pred_complex = pred
            target_complex = target
        else:
            pred_complex = torch.complex(pred[..., 0], pred[..., 1])
            target_complex = torch.complex(target[..., 0], target[..., 1])
        
        # FFT along azimuth (axis 1 for [B, H, W] or axis -2 for [B, ..., H, W])
        pred_fft = torch.fft.fft(pred_complex, dim=-2)
        target_fft = torch.fft.fft(target_complex, dim=-2)
        
        # MSE on magnitude
        mag_loss = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft), reduction=self.reduction)
        
        return mag_loss
    
    def phase_consistency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Phase loss using complex inner product.
        L = 1 - |<pred, target>| / (||pred|| ||target||)
        """
        # Handle format
        if pred.dtype in [torch.complex64, torch.complex128]:
            pred_complex = pred
            target_complex = target
        else:
            pred_complex = torch.complex(pred[..., 0], pred[..., 1])
            target_complex = torch.complex(target[..., 0], target[..., 1])
        
        # Flatten spatial dimensions for inner product
        pred_flat = pred_complex.flatten(start_dim=-2)  # [B, ..., H*W]
        target_flat = target_complex.flatten(start_dim=-2)
        
        # Complex inner product
        inner_product = torch.sum(pred_flat * torch.conj(target_flat), dim=-1)
        
        # Norms
        pred_norm = torch.sqrt(torch.sum(torch.abs(pred_flat)**2, dim=-1) + 1e-8)
        target_norm = torch.sqrt(torch.sum(torch.abs(target_flat)**2, dim=-1) + 1e-8)
        
        # Normalized inner product (cosine similarity in complex space)
        similarity = torch.abs(inner_product) / (pred_norm * target_norm + 1e-8)
        
        # Loss (1 - similarity)
        loss = 1.0 - similarity
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def ssim_loss(self, pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """
        Structural Similarity Index (SSIM) loss for structure preservation.
        Computes SSIM on magnitude of complex data.
        """
        # Convert to magnitude
        if pred.dtype in [torch.complex64, torch.complex128]:
            pred_mag = torch.abs(pred)
            target_mag = torch.abs(target)
        else:
            pred_mag = torch.sqrt(pred[..., 0]**2 + pred[..., 1]**2)
            target_mag = torch.sqrt(target[..., 0]**2 + target[..., 1]**2)
        
        # Add channel dimension if needed [B, H, W] -> [B, 1, H, W]
        if pred_mag.dim() == 3:
            pred_mag = pred_mag.unsqueeze(1)
            target_mag = target_mag.unsqueeze(1)
        
        # Compute SSIM using pytorch_msssim if available, else manual implementation
        try:
            from pytorch_msssim import ssim
            ssim_val = ssim(pred_mag, target_mag, data_range=target_mag.max() - target_mag.min(), 
                           size_average=True, win_size=window_size)
            return 1.0 - ssim_val  # Convert to loss (lower is better)
        except ImportError:
            # Manual SSIM implementation
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            mu_pred = F.avg_pool2d(pred_mag, window_size, stride=1, padding=window_size//2)
            mu_target = F.avg_pool2d(target_mag, window_size, stride=1, padding=window_size//2)
            
            mu_pred_sq = mu_pred ** 2
            mu_target_sq = mu_target ** 2
            mu_pred_target = mu_pred * mu_target
            
            sigma_pred_sq = F.avg_pool2d(pred_mag ** 2, window_size, stride=1, padding=window_size//2) - mu_pred_sq
            sigma_target_sq = F.avg_pool2d(target_mag ** 2, window_size, stride=1, padding=window_size//2) - mu_target_sq
            sigma_pred_target = F.avg_pool2d(pred_mag * target_mag, window_size, stride=1, padding=window_size//2) - mu_pred_target
            
            ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                       ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
            
            return 1.0 - ssim_map.mean()
    
    def edge_preserving_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Edge-preserving loss using gradient differences.
        Ensures sharp edges are maintained after compression.
        """
        # Convert to magnitude
        if pred.dtype in [torch.complex64, torch.complex128]:
            pred_mag = torch.abs(pred)
            target_mag = torch.abs(target)
        else:
            pred_mag = torch.sqrt(pred[..., 0]**2 + pred[..., 1]**2)
            target_mag = torch.sqrt(target[..., 0]**2 + target[..., 1]**2)
        
        # Compute gradients
        pred_grad_h = torch.abs(pred_mag[:, 1:, :] - pred_mag[:, :-1, :])
        pred_grad_w = torch.abs(pred_mag[:, :, 1:] - pred_mag[:, :, :-1])
        
        target_grad_h = torch.abs(target_mag[:, 1:, :] - target_mag[:, :-1, :])
        target_grad_w = torch.abs(target_mag[:, :, 1:] - target_mag[:, :, :-1])
        
        # L1 loss on gradients
        loss_h = F.l1_loss(pred_grad_h, target_grad_h, reduction=self.reduction)
        loss_w = F.l1_loss(pred_grad_w, target_grad_w, reduction=self.reduction)
        
        return (loss_h + loss_w) / 2.0
    
    def perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Simple perceptual loss using multi-scale gradients.
        Captures features at different scales for better detail preservation.
        """
        # Convert to magnitude
        if pred.dtype in [torch.complex64, torch.complex128]:
            pred_mag = torch.abs(pred)
            target_mag = torch.abs(target)
        else:
            pred_mag = torch.sqrt(pred[..., 0]**2 + pred[..., 1]**2)
            target_mag = torch.sqrt(target[..., 0]**2 + target[..., 1]**2)
        
        # Multi-scale feature extraction using pooling
        scales = [1, 2, 4]
        total_loss = 0.0
        
        for scale in scales:
            if scale > 1:
                pred_scaled = F.avg_pool2d(pred_mag.unsqueeze(1) if pred_mag.dim() == 3 else pred_mag, 
                                          scale, stride=scale).squeeze(1)
                target_scaled = F.avg_pool2d(target_mag.unsqueeze(1) if target_mag.dim() == 3 else target_mag,
                                            scale, stride=scale).squeeze(1)
            else:
                pred_scaled = pred_mag
                target_scaled = target_mag
            
            # Compute L1 loss at this scale
            scale_loss = F.l1_loss(pred_scaled, target_scaled, reduction=self.reduction)
            total_loss += scale_loss / scale  # Weight by scale
        
        return total_loss / len(scales)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        aux_outputs: Optional[dict] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Compute multi-domain loss.
        
        Args:
            pred: Predicted tensor [B, H, W, 2] or complex format
            target: Ground truth tensor [B, H, W, 2] or complex format
            aux_outputs: Optional dictionary with 'sparsity' and 'rate' from model
        
        Returns:
            If aux_outputs is None: scalar loss
            Otherwise: (loss, loss_dict) tuple with individual components
        """
        losses = {}
        
        # 1. Spatial domain loss
        losses['spatial'] = self.complex_mse(pred, target) * self.spatial_weight
        
        # 2. Frequency domain loss
        losses['frequency'] = self.frequency_domain_loss(pred, target) * self.frequency_weight
        
        # 3. Phase consistency
        losses['phase'] = self.phase_consistency_loss(pred, target) * self.phase_weight
        
        # 4. SSIM loss (structure preservation - NEW)
        if self.ssim_weight > 0:
            losses['ssim'] = self.ssim_loss(pred, target) * self.ssim_weight
        else:
            losses['ssim'] = torch.tensor(0.0, device=pred.device)
        
        # 5. Edge-preserving loss (detail preservation - NEW)
        if self.edge_weight > 0:
            losses['edge'] = self.edge_preserving_loss(pred, target) * self.edge_weight
        else:
            losses['edge'] = torch.tensor(0.0, device=pred.device)
        
        # 6. Perceptual loss (multi-scale features - NEW)
        if self.perceptual_weight > 0:
            losses['perceptual'] = self.perceptual_loss(pred, target) * self.perceptual_weight
        else:
            losses['perceptual'] = torch.tensor(0.0, device=pred.device)
        
        # 7. Sparsity penalty (from auxiliary outputs)
        if aux_outputs is not None and 'sparsity' in aux_outputs:
            sparsity = aux_outputs['sparsity']
            if isinstance(sparsity, torch.Tensor):
                losses['sparsity'] = sparsity * self.sparsity_weight
            else:
                losses['sparsity'] = torch.tensor(0.0, device=pred.device)
        else:
            losses['sparsity'] = torch.tensor(0.0, device=pred.device)
        
        # 8. Rate penalty (from auxiliary outputs)
        if aux_outputs is not None and 'rate' in aux_outputs:
            rate = aux_outputs['rate']
            if isinstance(rate, torch.Tensor):
                losses['rate'] = rate.mean() if rate.dim() > 0 else rate * self.rate_weight
            else:
                losses['rate'] = torch.tensor(0.0, device=pred.device)
        else:
            losses['rate'] = torch.tensor(0.0, device=pred.device)
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        # Return format depends on whether we have aux_outputs
        if aux_outputs is not None:
            return total_loss, losses
        else:
            return total_loss



class SARFocusingLoss(BaseLoss):
    """
    Comprehensive SAR focusing loss combining reconstruction, focus quality, 
    distribution matching, and total variation.
    
    IMPROVED VERSION with better scaling and normalization for stable training.
    
    Full combined loss:
        L = λ_rec * L_rec + λ_focus * L_focus + λ_dist * L_dist + λ_tv * L_tv
    
    where:
        L_rec    = Reconstruction loss (L1, L2, or complex MSE to ground truth)
        L_focus  = Focus quality metric (normalized variance, entropy, or contrast)
        L_dist   = Distribution matching (statistical moments or histogram L1)
        L_tv     = Total variation regularization (suppress speckle artifacts)
    
    Args:
        lambda_rec (float): Weight for reconstruction loss. Default: 1.0
        lambda_focus (float): Weight for focus quality. Default: 0.1
        lambda_dist (float): Weight for distribution matching. Default: 0.5
        lambda_tv (float): Weight for total variation. Default: 0.01
        rec_loss_type (str): Type of reconstruction loss ('mse', 'mae', 'complex_mse'). Default: 'complex_mse'
        focus_metric (str): Focus quality metric ('variance', 'entropy', 'contrast'). Default: 'contrast'
        dist_metric (str): Distribution matching metric ('moments', 'histogram', 'none'). Default: 'moments'
        use_tv (bool): Whether to use total variation regularization. Default: True
        use_adaptive_weights (bool): Dynamically balance loss components. Default: True
        reduction (str): Loss reduction method. Default: 'mean'
    
    Hyperparameter tips:
        - λ_rec = 1.0 (baseline)
        - λ_focus = 0.1-0.5 (higher than before due to normalization)
        - λ_dist = 0.5-2.0 (distribution matching is now more important)
        - λ_tv = 0.01-0.1 (can be higher due to magnitude-based computation)
    """
    
    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_focus: float = 0.1,
        lambda_dist: float = 0.5,
        lambda_tv: float = 0.01,
        rec_loss_type: str = 'complex_mse',
        focus_metric: str = 'contrast',
        dist_metric: str = 'moments',
        use_tv: bool = True,
        use_adaptive_weights: bool = True,
        reduction: str = 'mean'
    ):
        super().__init__(reduction=reduction)
        self.lambda_rec = lambda_rec
        self.lambda_focus = lambda_focus
        self.lambda_dist = lambda_dist
        self.lambda_tv = lambda_tv
        self.rec_loss_type = rec_loss_type
        self.focus_metric = focus_metric
        self.dist_metric = dist_metric
        self.use_tv = use_tv
        self.use_adaptive_weights = use_adaptive_weights
        
        # Running statistics for adaptive weighting
        self.register_buffer('rec_scale', torch.tensor(1.0))
        self.register_buffer('focus_scale', torch.tensor(1.0))
        self.register_buffer('dist_scale', torch.tensor(1.0))
        self.register_buffer('tv_scale', torch.tensor(1.0))
        self.register_buffer('num_updates', torch.tensor(0))
        
        # Initialize reconstruction loss
        if rec_loss_type == 'mse':
            self.rec_loss = MSELoss(reduction=reduction)
        elif rec_loss_type == 'mae':
            self.rec_loss = MAELoss(reduction=reduction)
        elif rec_loss_type == 'complex_mse':
            self.rec_loss = ComplexMSELoss(reduction=reduction)
            
        elif rec_loss_type == 'complex_mae':
            self.rec_loss = ComplexMAELoss(reduction=reduction)
        elif rec_loss_type == 'distribution_aware_mse':
            # NEW: Distribution-aware MSE that weights by target statistics
            self.rec_loss = DistributionAwareMSELoss(
                normalization_mode='batch',  # Use batch statistics
                use_standardization=True,    # Weight by (x-mean)/std
                eps=1e-6,
                reduction=reduction
            )
        else:
            raise ValueError(f"Unknown rec_loss_type: {rec_loss_type}. Options: 'mse', 'mae', 'complex_mse', 'distribution_aware_mse'")
    
    def reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward-model reconstruction or L1/L2 to ground truth."""
        return self.rec_loss(pred, target)
    
    def focus_quality_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Focus quality metric based on spatial concentration.
        IMPROVED with better normalization for stable gradients.
        
        Methods:
            - 'variance': Normalized centroid variance
            - 'entropy': Normalized image entropy  
            - 'contrast': Image contrast/sharpness (NEW - RECOMMENDED)
        """
        # Convert to magnitude if complex
        if torch.is_complex(pred):
            mag = pred.abs()
        elif pred.shape[-1] == 2:
            mag = torch.sqrt(pred[..., 0]**2 + pred[..., 1]**2 + 1e-8)
        else:
            mag = pred.abs()
        
        if self.focus_metric == 'variance':
            # Normalized centroid variance
            return self._normalized_centroid_variance(mag)
        
        elif self.focus_metric == 'entropy':
            # Normalized image entropy
            return self._normalized_entropy(mag)
        
        elif self.focus_metric == 'contrast':
            # Image contrast - measures sharpness
            # Maximize contrast = minimize negative contrast
            return -self._image_contrast(mag)
        
        else:
            raise ValueError(f"Unknown focus_metric: {self.focus_metric}")
    
    def _normalized_centroid_variance(self, mag: torch.Tensor) -> torch.Tensor:
        """
        Compute NORMALIZED variance of intensity around centroid.
        Normalized by image size for scale-invariance.
        """
        if mag.dim() == 4:
            mag = mag.mean(dim=1)
        
        B, H, W = mag.shape
        
        # Normalize to probability distribution
        eps = 1e-8
        mag_sum = mag.sum(dim=(-2, -1), keepdim=True) + eps
        mag_norm = mag / mag_sum
        
        # Create normalized coordinate grids [0, 1]
        y_coords = torch.linspace(0, 1, H, device=mag.device, dtype=mag.dtype).view(1, H, 1)
        x_coords = torch.linspace(0, 1, W, device=mag.device, dtype=mag.dtype).view(1, 1, W)
        
        # Compute centroids
        cy = (mag_norm * y_coords).sum(dim=(-2, -1))
        cx = (mag_norm * x_coords).sum(dim=(-2, -1))
        
        # Compute normalized variance
        var_y = (mag_norm * (y_coords - cy.view(B, 1, 1))**2).sum(dim=(-2, -1))
        var_x = (mag_norm * (x_coords - cx.view(B, 1, 1))**2).sum(dim=(-2, -1))
        
        # Combined variance (already normalized to [0, 1] range)
        variance = var_y + var_x
        
        if self.reduction == 'mean':
            return variance.mean()
        elif self.reduction == 'sum':
            return variance.sum()
        else:
            return variance
    
    def _normalized_entropy(self, mag: torch.Tensor) -> torch.Tensor:
        """
        Compute NORMALIZED image entropy: H / log(N)
        Normalized by maximum possible entropy for scale-invariance.
        """
        if mag.dim() == 4:
            mag = mag.mean(dim=1)
        
        B, H, W = mag.shape
        N = H * W
        
        # Normalize to probability distribution
        eps = 1e-8
        mag_norm = mag / (mag.sum(dim=(-2, -1), keepdim=True) + eps)
        
        # Compute entropy
        entropy = -(mag_norm * torch.log(mag_norm + eps)).sum(dim=(-2, -1))
        
        # Normalize by maximum entropy (uniform distribution)
        max_entropy = math.log(N)
        normalized_entropy = entropy / max_entropy
        
        if self.reduction == 'mean':
            return normalized_entropy.mean()
        elif self.reduction == 'sum':
            return normalized_entropy.sum()
        else:
            return normalized_entropy
    
    def _image_contrast(self, mag: torch.Tensor) -> torch.Tensor:
        """
        Compute image contrast using standard deviation normalized by mean.
        Higher contrast = better focus.
        
        Contrast = std(I) / (mean(I) + eps)
        """
        if mag.dim() == 4:
            mag = mag.mean(dim=1)
        
        # Flatten spatial dimensions
        B = mag.shape[0]
        mag_flat = mag.view(B, -1)
        
        eps = 1e-8
        mean_val = mag_flat.mean(dim=1) + eps
        std_val = mag_flat.std(dim=1) + eps
        
        # Normalized contrast (dimensionless)
        contrast = std_val / mean_val
        
        # Return negative because we minimize loss (want to maximize contrast)
        if self.reduction == 'mean':
            return contrast.mean()
        elif self.reduction == 'sum':
            return contrast.sum()
        else:
            return contrast
    
    def _centroid_variance(self, mag: torch.Tensor) -> torch.Tensor:
        """
        Compute variance of intensity around centroid.
        Lower variance indicates better focus.
        """
        # mag: (B, H, W) or (B, C, H, W)
        if mag.dim() == 4:
            # Average over channels if present
            mag = mag.mean(dim=1)
        
        B, H, W = mag.shape
        
        # Normalize to probability distribution
        eps = 1e-8
        mag_norm = mag / (mag.sum(dim=(-2, -1), keepdim=True) + eps)
        
        # Create coordinate grids
        y_coords = torch.arange(H, device=mag.device, dtype=mag.dtype).view(1, H, 1)
        x_coords = torch.arange(W, device=mag.device, dtype=mag.dtype).view(1, 1, W)
        
        # Compute centroids
        cy = (mag_norm * y_coords).sum(dim=(-2, -1))  # (B,)
        cx = (mag_norm * x_coords).sum(dim=(-2, -1))  # (B,)
        
        # Compute variance around centroid
        var_y = (mag_norm * (y_coords - cy.view(B, 1, 1))**2).sum(dim=(-2, -1))
        var_x = (mag_norm * (x_coords - cx.view(B, 1, 1))**2).sum(dim=(-2, -1))
        
        variance = var_y + var_x
        
        if self.reduction == 'mean':
            return variance.mean()
        elif self.reduction == 'sum':
            return variance.sum()
        else:
            return variance
    
    def _image_entropy(self, mag: torch.Tensor) -> torch.Tensor:
        """
        Compute image entropy: H = -sum(p * log(p))
        Lower entropy indicates sharper focus.
        """
        if mag.dim() == 4:
            mag = mag.mean(dim=1)
        
        # Normalize to probability distribution
        eps = 1e-8
        mag_norm = mag / (mag.sum(dim=(-2, -1), keepdim=True) + eps)
        
        # Compute entropy
        entropy = -(mag_norm * torch.log(mag_norm + eps)).sum(dim=(-2, -1))
        
        if self.reduction == 'mean':
            return entropy.mean()
        elif self.reduction == 'sum':
            return entropy.sum()
        else:
            return entropy
    
    def _image_kurtosis(self, mag: torch.Tensor) -> torch.Tensor:
        """
        Compute image kurtosis (fourth standardized moment).
        Higher kurtosis indicates sharper peaks (better focus).
        """
        if mag.dim() == 4:
            mag = mag.mean(dim=1)
        
        # Flatten spatial dimensions
        B = mag.shape[0]
        mag_flat = mag.view(B, -1)
        
        # Compute moments
        mean = mag_flat.mean(dim=1, keepdim=True)
        var = ((mag_flat - mean)**2).mean(dim=1, keepdim=True)
        std = torch.sqrt(var + 1e-8)
        
        # Standardized fourth moment
        kurtosis = (((mag_flat - mean) / std)**4).mean(dim=1)
        
        if self.reduction == 'mean':
            return kurtosis.mean()
        elif self.reduction == 'sum':
            return kurtosis.sum()
        else:
            return kurtosis
    
    def distribution_matching_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Distribution matching loss - IMPROVED for better gradients.
        
        Methods:
            - 'moments': Match mean, std, skewness, kurtosis (RECOMMENDED)
            - 'histogram': L1 distance between histograms
            - 'none': Disable distribution matching
        """
        if self.dist_metric is None or self.dist_metric == 'none':
            return torch.tensor(0.0, device=pred.device)
        
        if self.dist_metric == 'moments':
            return self._statistical_moments_loss(pred, target)
        
        elif self.dist_metric == 'histogram':
            return self._histogram_l1_loss(pred, target)
        
        elif self.dist_metric == 'mmd':
            # Keep MMD as option but use simplified version
            return self._simplified_mmd(pred, target)
        
        else:
            raise ValueError(f"Unknown dist_metric: {self.dist_metric}")
    
    def _statistical_moments_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Match statistical moments: mean, std, skewness, kurtosis.
        This provides strong gradients and is computationally efficient.
        """
        # Convert to magnitude
        if torch.is_complex(pred):
            pred_mag = pred.abs()
            target_mag = target.abs()
        elif pred.shape[-1] == 2:
            pred_mag = torch.sqrt(pred[..., 0]**2 + pred[..., 1]**2 + 1e-8)
            target_mag = torch.sqrt(target[..., 0]**2 + target[..., 1]**2 + 1e-8)
        else:
            pred_mag = pred.abs()
            target_mag = target.abs()
        
        # Flatten
        B = pred_mag.shape[0]
        pred_flat = pred_mag.view(B, -1)
        target_flat = target_mag.view(B, -1)
        
        eps = 1e-8
        
        # First moment: mean
        pred_mean = pred_flat.mean(dim=1)
        target_mean = target_flat.mean(dim=1)
        loss_mean = F.mse_loss(pred_mean, target_mean, reduction='none')
        
        # Second moment: standard deviation
        pred_std = pred_flat.std(dim=1) + eps
        target_std = target_flat.std(dim=1) + eps
        loss_std = F.mse_loss(pred_std, target_std, reduction='none')
        
        # Third moment: skewness (optional, can be noisy)
        pred_centered = pred_flat - pred_mean.unsqueeze(1)
        target_centered = target_flat - target_mean.unsqueeze(1)
        pred_skew = (pred_centered**3).mean(dim=1) / (pred_std**3 + eps)
        target_skew = (target_centered**3).mean(dim=1) / (target_std**3 + eps)
        loss_skew = F.mse_loss(pred_skew, target_skew, reduction='none')
        
        # Fourth moment: kurtosis
        pred_kurt = (pred_centered**4).mean(dim=1) / (pred_std**4 + eps)
        target_kurt = (target_centered**4).mean(dim=1) / (target_std**4 + eps)
        loss_kurt = F.mse_loss(pred_kurt, target_kurt, reduction='none')
        
        # Weighted combination (mean and std are most important)
        total_loss = 2.0 * loss_mean + 2.0 * loss_std + 0.5 * loss_skew + 0.5 * loss_kurt
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss
    
    def _histogram_l1_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        KDE-based distribution matching using Gaussian kernel density estimation.
        
        This is superior to raw histogram comparison because:
        1. Provides smooth, continuous probability density estimates
        2. More robust to bin placement and sample size
        3. Better gradients for optimization
        
        Note: Requires sufficient samples (typically >100) for reliable KDE.
              Works best with larger batch sizes or epoch-level aggregation.
        """
        # Convert to magnitude
        if torch.is_complex(pred):
            pred_mag = pred.abs()
            target_mag = target.abs()
        elif pred.shape[-1] == 2:
            pred_mag = torch.sqrt(pred[..., 0]**2 + pred[..., 1]**2 + 1e-8)
            target_mag = torch.sqrt(target[..., 0]**2 + target[..., 1]**2 + 1e-8)
        else:
            pred_mag = pred.abs()
            target_mag = target.abs()
        
        # Flatten
        B = pred_mag.shape[0]
        pred_flat = pred_mag.view(B, -1)
        target_flat = target_mag.view(B, -1)
        
        # Number of evaluation points for KDE
        num_eval_points = 100
        
        total_loss = torch.tensor(0.0, device=pred.device)
        
        for b in range(B):
            pred_samples = pred_flat[b]
            target_samples = target_flat[b]
            
            # Automatic bandwidth selection using Scott's rule
            # h = n^(-1/5) * std
            n_pred = pred_samples.numel()
            n_target = target_samples.numel()
            
            pred_std = pred_samples.std() + 1e-8
            target_std = target_samples.std() + 1e-8
            
            # Scott's bandwidth (robust for near-Gaussian distributions)
            h_pred = pred_std * (n_pred ** (-1/5))
            h_target = target_std * (n_target ** (-1/5))
            
            # Use average bandwidth for fair comparison
            h = (h_pred + h_target) / 2.0
            
            # Create evaluation grid covering both distributions
            min_val = torch.min(pred_samples.min(), target_samples.min())
            max_val = torch.max(pred_samples.max(), target_samples.max())
            
            # Add margin to avoid edge effects
            margin = (max_val - min_val) * 0.1
            eval_points = torch.linspace(min_val - margin, max_val + margin, 
                                        num_eval_points, device=pred.device)
            
            # Compute KDE for prediction
            # KDE(x) = (1/nh) * Σ K((x - x_i) / h)
            # Using Gaussian kernel: K(u) = (1/√(2π)) * exp(-u²/2)
            pred_kde = self._gaussian_kde(pred_samples, eval_points, h)
            
            # Compute KDE for target
            target_kde = self._gaussian_kde(target_samples, eval_points, h)
            
            # L1 distance between KDEs (integrated over evaluation points)
            # Approximate integral using trapezoidal rule
            dx = (max_val - min_val + 2*margin) / num_eval_points
            kde_diff = torch.abs(pred_kde - target_kde)
            kde_l1 = kde_diff.sum() * dx
            
            total_loss += kde_l1
        
        if self.reduction == 'mean':
            return total_loss / B
        elif self.reduction == 'sum':
            return total_loss
        else:
            return total_loss / B
    
    def _gaussian_kde(self, samples: torch.Tensor, eval_points: torch.Tensor, 
                     bandwidth: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian kernel density estimate at evaluation points.
        
        Args:
            samples: Data samples [N]
            eval_points: Points to evaluate KDE [M]
            bandwidth: KDE bandwidth (scalar)
        
        Returns:
            kde: Density estimates at eval_points [M]
        """
        # samples: [N], eval_points: [M]
        # Compute distance matrix: [M, N]
        # (eval_points[i] - samples[j]) for all i, j
        
        # Reshape for broadcasting
        eval_points = eval_points.view(-1, 1)  # [M, 1]
        samples = samples.view(1, -1)           # [1, N]
        
        # Squared distances / (2 * h²)
        z = ((eval_points - samples) / bandwidth) ** 2 / 2.0  # [M, N]
        
        # Gaussian kernel: (1/√(2π)) * exp(-z)
        gaussian_constant = 1.0 / torch.sqrt(torch.tensor(2.0 * 3.14159265359, device=samples.device))
        kernel_values = gaussian_constant * torch.exp(-z)  # [M, N]
        
        # Sum over samples and normalize
        n_samples = samples.shape[1]
        kde = kernel_values.sum(dim=1) / (n_samples * bandwidth)  # [M]
        
        return kde
    
    def _simplified_mmd(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Simplified MMD using only mean embedding distance.
        Much faster and more stable than full MMD.
        """
        # Convert to magnitude
        if torch.is_complex(pred):
            pred_mag = pred.abs()
            target_mag = target.abs()
        elif pred.shape[-1] == 2:
            pred_mag = torch.sqrt(pred[..., 0]**2 + pred[..., 1]**2 + 1e-8)
            target_mag = torch.sqrt(target[..., 0]**2 + target[..., 1]**2 + 1e-8)
        else:
            pred_mag = pred.abs()
            target_mag = target.abs()
        
        # Flatten
        B = pred_mag.shape[0]
        pred_flat = pred_mag.view(B, -1)
        target_flat = target_mag.view(B, -1)
        
        # Simple mean squared distance
        mmd = ((pred_flat.mean(dim=1) - target_flat.mean(dim=1))**2).mean()
        
        return mmd
    
    def _maximum_mean_discrepancy(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Maximum Mean Discrepancy (MMD) between prediction and target distributions.
        Uses kernel trick to measure distance between distributions.
        """
        # Convert to magnitude for distribution comparison
        if torch.is_complex(pred):
            pred_mag = pred.abs()
            target_mag = target.abs()
        elif pred.shape[-1] == 2:
            pred_mag = torch.sqrt(pred[..., 0]**2 + pred[..., 1]**2)
            target_mag = torch.sqrt(target[..., 0]**2 + target[..., 1]**2)
        else:
            pred_mag = pred.abs()
            target_mag = target.abs()
        
        # Flatten to (B, N) where N is number of pixels
        B = pred_mag.shape[0]
        pred_flat = pred_mag.view(B, -1)
        target_flat = target_mag.view(B, -1)
        
        # Sample subset for efficiency (use all if small enough)
        max_samples = 1000
        if pred_flat.shape[1] > max_samples:
            indices = torch.randperm(pred_flat.shape[1], device=pred_flat.device)[:max_samples]
            pred_flat = pred_flat[:, indices]
            target_flat = target_flat[:, indices]
        
        # Compute kernel matrices
        def kernel(x, y, bandwidth):
            """Gaussian or Laplacian kernel."""
            # x, y: (B, N)
            xx = x.unsqueeze(2)  # (B, N, 1)
            yy = y.unsqueeze(1)  # (B, 1, N)
            diff = xx - yy  # (B, N, N)
            
            if self.mmd_kernel == 'gaussian':
                return torch.exp(-diff**2 / (2 * bandwidth**2))
            elif self.mmd_kernel == 'laplacian':
                return torch.exp(-torch.abs(diff) / bandwidth)
            else:
                raise ValueError(f"Unknown kernel: {self.mmd_kernel}")
        
        # MMD^2 = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
        k_xx = kernel(pred_flat, pred_flat, self.mmd_bandwidth)
        k_yy = kernel(target_flat, target_flat, self.mmd_bandwidth)
        k_xy = kernel(pred_flat, target_flat, self.mmd_bandwidth)
        
        # Compute MMD (remove diagonal for unbiased estimate)
        n = pred_flat.shape[1]
        mmd = (k_xx.sum(dim=(1, 2)) - k_xx.diagonal(dim1=1, dim2=2).sum(dim=1)) / (n * (n - 1))
        mmd += (k_yy.sum(dim=(1, 2)) - k_yy.diagonal(dim1=1, dim2=2).sum(dim=1)) / (n * (n - 1))
        mmd -= 2 * k_xy.mean(dim=(1, 2))
        
        if self.reduction == 'mean':
            return mmd.mean()
        elif self.reduction == 'sum':
            return mmd.sum()
        else:
            return mmd
    
    def _histogram_matching_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Histogram matching loss: measures difference between histograms.
        """
        # Convert to magnitude
        if torch.is_complex(pred):
            pred_mag = pred.abs()
            target_mag = target.abs()
        elif pred.shape[-1] == 2:
            pred_mag = torch.sqrt(pred[..., 0]**2 + pred[..., 1]**2)
            target_mag = torch.sqrt(target[..., 0]**2 + target[..., 1]**2)
        else:
            pred_mag = pred.abs()
            target_mag = target.abs()
        
        # Compute histograms
        num_bins = 64
        B = pred_mag.shape[0]
        
        # Flatten
        pred_flat = pred_mag.view(B, -1)
        target_flat = target_mag.view(B, -1)
        
        # Find common range
        min_val = torch.min(torch.min(pred_flat.min(), target_flat.min()), torch.tensor(0.0, device=pred.device))
        max_val = torch.max(pred_flat.max(), target_flat.max())
        
        # Compute histograms (simple binning)
        hist_loss = torch.tensor(0.0, device=pred.device)
        
        for b in range(B):
            pred_hist = torch.histc(pred_flat[b], bins=num_bins, min=min_val.item(), max=max_val.item())
            target_hist = torch.histc(target_flat[b], bins=num_bins, min=min_val.item(), max=max_val.item())
            
            # Normalize
            pred_hist = pred_hist / (pred_hist.sum() + 1e-8)
            target_hist = target_hist / (target_hist.sum() + 1e-8)
            
            # L2 distance between histograms
            hist_loss += ((pred_hist - target_hist)**2).sum()
        
        if self.reduction == 'mean':
            return hist_loss / B
        elif self.reduction == 'sum':
            return hist_loss
        else:
            return hist_loss
    
    def total_variation_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Total variation regularization to suppress speckle artifacts.
        Use small weight to avoid smearing point targets.
        
        Computes isotropic TV: sum(|grad_x| + |grad_y|)
        """
        if not self.use_tv:
            return torch.tensor(0.0, device=pred.device)
        
        # Convert to magnitude for TV computation
        if torch.is_complex(pred):
            mag = pred.abs()
        elif pred.shape[-1] == 2:
            mag = torch.sqrt(pred[..., 0]**2 + pred[..., 1]**2)
        else:
            mag = pred.abs()
        
        # Compute spatial gradients
        # mag: (B, H, W) or (B, C, H, W)
        if mag.dim() == 4:
            # Average over channels if present
            mag = mag.mean(dim=1)
        
        # Gradients along height and width
        grad_h = torch.abs(mag[:, 1:, :] - mag[:, :-1, :])
        grad_w = torch.abs(mag[:, :, 1:] - mag[:, :, :-1])
        
        tv = grad_h.sum(dim=(-2, -1)) + grad_w.sum(dim=(-2, -1))
        
        if self.reduction == 'mean':
            return tv.mean()
        elif self.reduction == 'sum':
            return tv.sum()
        else:
            return tv
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Compute combined SAR focusing loss with ADAPTIVE WEIGHTING.
        
        Args:
            pred: Predicted focused SAR image
            target: Ground truth (focused or reference)
            return_components: If True, return dictionary with individual loss components
        
        Returns:
            If return_components=False: scalar total loss
            If return_components=True: (total_loss, loss_dict)
        """
        losses = {}
        losses_raw = {}
        
        # Compute raw losses (unweighted)
        rec_raw = self.reconstruction_loss(pred, target)
        focus_raw = self.focus_quality_loss(pred)
        dist_raw = self.distribution_matching_loss(pred, target)
        tv_raw = self.total_variation_loss(pred)
        
        losses_raw['rec'] = rec_raw
        losses_raw['focus'] = focus_raw
        losses_raw['dist'] = dist_raw
        losses_raw['tv'] = tv_raw
        
        # Adaptive weighting: normalize by running average of loss magnitudes
        if self.use_adaptive_weights and self.training:
            with torch.no_grad():
                # Update running statistics
                momentum = 0.1 if self.num_updates < 100 else 0.01
                
                self.rec_scale = (1 - momentum) * self.rec_scale + momentum * rec_raw.detach().abs()
                self.focus_scale = (1 - momentum) * self.focus_scale + momentum * focus_raw.detach().abs()
                self.dist_scale = (1 - momentum) * self.dist_scale + momentum * dist_raw.detach().abs()
                self.tv_scale = (1 - momentum) * self.tv_scale + momentum * tv_raw.detach().abs()
                
                self.num_updates += 1
            
            # Normalize by running scale (prevents one loss from dominating)
            eps = 1e-8
            losses['rec'] = self.lambda_rec * rec_raw / (self.rec_scale + eps)
            losses['focus'] = self.lambda_focus * focus_raw / (self.focus_scale + eps)
            losses['dist'] = self.lambda_dist * dist_raw / (self.dist_scale + eps)
            losses['tv'] = self.lambda_tv * tv_raw / (self.tv_scale + eps)
        else:
            # Standard weighting (no adaptation)
            losses['rec'] = self.lambda_rec * rec_raw
            losses['focus'] = self.lambda_focus * focus_raw
            losses['dist'] = self.lambda_dist * dist_raw
            losses['tv'] = self.lambda_tv * tv_raw
        
        # Total loss
        total_loss = losses['rec'] + losses['focus'] + losses['dist'] + losses['tv']
        losses['total'] = total_loss
        
        # Add raw values for logging
        if return_components:
            losses['rec_raw'] = rec_raw
            losses['focus_raw'] = focus_raw
            losses['dist_raw'] = dist_raw
            losses['tv_raw'] = tv_raw
            return total_loss, losses
        else:
            return total_loss

    
def get_loss_function(loss_name: str, **kwargs) -> BaseLoss:
    """
    Factory function to get loss function by name.
    
    Args:
        loss_name (str): Name of the loss function
        **kwargs: Additional arguments for the loss function
        
    Returns:
        BaseLoss: Instantiated loss function
        
    Raises:
        ValueError: If loss_name is not recognized
    """
    loss_registry = {
        'mse': MSELoss,
        'mae': MAELoss,
        'huber': HuberLoss,
        'focal': FocalLoss,
        'complex_mse': ComplexMSELoss,
        'phase': PhaseLoss,
        'combined_complex': CombinedComplexLoss,
        'ssim': SSIM1DLoss(window_size=7, reduction='mean'), 
        'edge': EdgeLoss, 
        'combined_complex_ssim': ComplexSSIMCombinedLoss,
        'split_real_imag': SplitRealImagLoss,
        'polar': PolarLoss,
        'robust_complex': RobustComplexLoss,
        'log_magnitude': LogMagnitudeLoss,
        'symmetry': SymmetryConstrainedLoss,
        'polarimetric': PolarimetricDecompositionLoss,
        'power': PowerLoss,
        'speckle_nll': SpeckleNLLLoss,
        'complex_tv': ComplexTVLoss,
        'gradient_matching': GradientMatchingLoss,
        'phase_smoothness': PhaseSmoothnessLoss,
        'gan': GANLoss,
        'feature': FeatureLoss,
        'composite': CompositeLoss,
        'multi_domain_sar': MultiDomainSARLoss,  # Physics-aware multi-domain loss
        'sar_focusing': SARFocusingLoss,  # Comprehensive SAR focusing loss
    }
    
    if loss_name.lower() not in loss_registry:
        available_losses = list(loss_registry.keys())
        raise ValueError(f'Unknown loss function: {loss_name}. Available: {available_losses}')
    
    return loss_registry[loss_name.lower()](**kwargs)
