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

class EdgeLoss(nn.Module): # Currently not using this
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
        'ssim': losses.SSIMLoss(window_size=7, reduction='mean'), 
        'edge': EdgeLoss
    }
    
    if loss_name.lower() not in loss_registry:
        available_losses = list(loss_registry.keys())
        raise ValueError(f'Unknown loss function: {loss_name}. Available: {available_losses}')
    
    return loss_registry[loss_name.lower()](**kwargs)
