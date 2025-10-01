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
    }
    
    if loss_name.lower() not in loss_registry:
        available_losses = list(loss_registry.keys())
        raise ValueError(f'Unknown loss function: {loss_name}. Available: {available_losses}')
    
    return loss_registry[loss_name.lower()](**kwargs)
