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
    }
    
    if loss_name.lower() not in loss_registry:
        available_losses = list(loss_registry.keys())
        raise ValueError(f'Unknown loss function: {loss_name}. Available: {available_losses}')
    
    return loss_registry[loss_name.lower()](**kwargs)
