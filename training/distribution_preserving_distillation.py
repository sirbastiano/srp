#!/usr/bin/env python3
"""
Distribution-Preserving Knowledge Distillation

This module implements enhanced knowledge distillation techniques that specifically
preserve the original distribution characteristics of the data, preventing the 
common issue where student models make overly conservative (centered) predictions.

Key Features:
- Variance regularization to maintain prediction spread
- Moment matching to preserve distribution statistics
- Dynamic temperature scaling based on prediction confidence
- Distribution alignment loss
- Confidence calibration mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math
from sarpyx.utils.losses import get_loss_function


class DistributionPreservingLoss(nn.Module):
    """
    Enhanced knowledge distillation loss that preserves the original data distribution
    by preventing overly conservative (centered) predictions from the student model.
    """
    
    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.7,          # Ground truth weight
        beta: float = 0.2,           # Distillation weight  
        gamma: float = 0.1,          # Feature weight
        variance_weight: float = 0.15,   # NEW: Variance preservation weight
        moment_weight: float = 0.1,      # NEW: Moment matching weight
        confidence_weight: float = 0.05, # NEW: Confidence calibration weight
        dynamic_temperature: bool = True, # NEW: Enable dynamic temperature
        loss_fn_name: str = "complex_mse"
    ):
        super().__init__()
        self.base_temperature = temperature
        self.alpha = alpha
        self.beta = beta  
        self.gamma = gamma
        self.variance_weight = variance_weight
        self.moment_weight = moment_weight
        self.confidence_weight = confidence_weight
        self.dynamic_temperature = dynamic_temperature
        
        from sarpyx.utils.losses import get_loss_function
        self.mse_loss = get_loss_function(loss_fn_name)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
        # Moving averages for dynamic temperature scaling
        self.register_buffer('teacher_var_ema', torch.tensor(1.0))
        self.register_buffer('student_var_ema', torch.tensor(1.0))
        self.ema_momentum = 0.99
        
    def compute_adaptive_temperature(self, teacher_output: torch.Tensor, student_output: torch.Tensor) -> float:
        """
        Compute adaptive temperature based on prediction confidence and variance.
        Higher variance predictions get lower temperature (sharper), 
        lower variance predictions get higher temperature (softer).
        """
        if not self.dynamic_temperature:
            return self.base_temperature
            
        # Handle complex tensors by using magnitude for variance computation
        if torch.is_complex(teacher_output):
            teacher_values = torch.abs(teacher_output)
        else:
            teacher_values = teacher_output
            
        if torch.is_complex(student_output):
            student_values = torch.abs(student_output)
        else:
            student_values = student_output
            
        # Compute current variances
        teacher_var = torch.var(teacher_values).item()
        student_var = torch.var(student_values).item()
        
        # Update EMA of variances
        with torch.no_grad():
            self.teacher_var_ema = self.ema_momentum * self.teacher_var_ema + (1 - self.ema_momentum) * teacher_var
            self.student_var_ema = self.ema_momentum * self.student_var_ema + (1 - self.ema_momentum) * student_var
        
        # Adaptive temperature: lower temperature when teacher has high variance
        # This encourages the student to maintain high variance predictions
        variance_ratio = self.teacher_var_ema / (self.student_var_ema + 1e-8)
        adaptive_temp = self.base_temperature * torch.sqrt(1.0 / (variance_ratio + 1e-8)).clamp(0.5, 2.0)
        
        return adaptive_temp.item()
    
    def compute_variance_preservation_loss(self, student_output: torch.Tensor, teacher_output: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Compute loss that encourages the student to maintain the same variance as the ground truth.
        This prevents the student from making overly conservative (low-variance) predictions.
        """
        # Handle complex tensors by using magnitude for variance calculations
        if torch.is_complex(ground_truth):
            gt_values = torch.abs(ground_truth)
        else:
            gt_values = ground_truth
            
        if torch.is_complex(student_output):
            student_values = torch.abs(student_output)
        else:
            student_values = student_output
            
        if torch.is_complex(teacher_output):
            teacher_values = torch.abs(teacher_output)
        else:
            teacher_values = teacher_output
        
        # Compute variances along different dimensions
        gt_var_total = torch.var(gt_values)
        student_var_total = torch.var(student_values)
        teacher_var_total = torch.var(teacher_values)
        
        # Compute per-sequence variance (preserves temporal dynamics)
        gt_var_seq = torch.var(gt_values, dim=1, keepdim=True)  # Variance across sequence length
        student_var_seq = torch.var(student_values, dim=1, keepdim=True)
        
        # Compute per-feature variance (preserves feature dynamics)
        gt_var_feat = torch.var(gt_values, dim=(0, 1), keepdim=True)  # Variance across batch and sequence
        student_var_feat = torch.var(student_values, dim=(0, 1), keepdim=True)
        
        # Multi-scale variance loss
        total_var_loss = F.mse_loss(student_var_total, gt_var_total)
        seq_var_loss = F.mse_loss(student_var_seq, gt_var_seq)
        feat_var_loss = F.mse_loss(student_var_feat, gt_var_feat)
        
        # Combined variance preservation loss
        variance_loss = total_var_loss + 0.5 * seq_var_loss + 0.3 * feat_var_loss
        
        return variance_loss
    
    def compute_moment_matching_loss(self, student_output: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Compute moment matching loss to preserve higher-order statistics of the distribution.
        This helps maintain the shape of the distribution beyond just mean and variance.
        """
        # Handle complex tensors by using magnitude for statistical calculations
        if torch.is_complex(ground_truth):
            gt_values = torch.abs(ground_truth)
        else:
            gt_values = ground_truth
            
        if torch.is_complex(student_output):
            student_values = torch.abs(student_output)
        else:
            student_values = student_output
        
        # First moment (mean)
        gt_mean = torch.mean(gt_values)
        student_mean = torch.mean(student_values)
        mean_loss = F.mse_loss(student_mean, gt_mean)

        # Second moment (variance) - already handled in variance preservation
        gt_var = torch.var(gt_values)
        student_var = torch.var(student_values)
        var_loss = F.mse_loss(student_var, gt_var)
        
        # Third moment (skewness) - measures asymmetry
        gt_centered = gt_values - gt_mean
        student_centered = student_values - student_mean
        gt_skew = torch.mean(gt_centered ** 3) / (gt_var ** 1.5 + 1e-8)
        student_skew = torch.mean(student_centered ** 3) / (student_var ** 1.5 + 1e-8)
        skew_loss = F.mse_loss(student_skew, gt_skew)
        
        # Fourth moment (kurtosis) - measures tail heaviness
        gt_kurt = torch.mean(gt_centered ** 4) / (gt_var ** 2 + 1e-8)
        student_kurt = torch.mean(student_centered ** 4) / (student_var ** 2 + 1e-8)
        kurt_loss = F.mse_loss(student_kurt, gt_kurt)
        
        # Combine moments with decreasing weights (higher moments are less stable)
        moment_loss = mean_loss + 0.8 * var_loss + 0.3 * skew_loss + 0.1 * kurt_loss
        
        return moment_loss
    
    def compute_confidence_calibration_loss(self, student_output: torch.Tensor, teacher_output: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence calibration loss to ensure student maintains appropriate confidence levels.
        This prevents the student from being overly conservative in high-confidence regions.
        """
        # Handle complex tensors by using magnitude
        if torch.is_complex(teacher_output):
            teacher_values = torch.abs(teacher_output)
            student_values = torch.abs(student_output)
        else:
            teacher_values = teacher_output
            student_values = student_output
        
        # Measure confidence as distance from mean prediction
        teacher_mean = torch.mean(teacher_values)
        student_mean = torch.mean(student_values)
        
        # Compute confidence scores (distance from mean)
        teacher_confidence = torch.abs(teacher_values - teacher_mean)
        student_confidence = torch.abs(student_values - student_mean)
        
        # Encourage student to match teacher's confidence pattern
        confidence_loss = F.mse_loss(student_confidence, teacher_confidence)
        
        # Add percentile-based calibration
        # Ensure student maintains similar percentile values as teacher
        # Convert to float if needed for quantile calculation
        teacher_flat = teacher_values.flatten().float()
        student_flat = student_values.flatten().float()
        
        teacher_percentiles = torch.quantile(teacher_flat, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]).to(teacher_output.device))
        student_percentiles = torch.quantile(student_flat, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]).to(student_output.device))
        percentile_loss = F.mse_loss(student_percentiles, teacher_percentiles)
        
        return confidence_loss + 0.5 * percentile_loss
    
    def compute_distribution_alignment_loss(self, student_output: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Compute distribution alignment loss using Wasserstein-like distance approximation.
        This encourages the student's output distribution to match the ground truth distribution.
        """
        # Handle complex tensors by using magnitude
        if torch.is_complex(student_output):
            student_values = torch.abs(student_output)
        else:
            student_values = student_output
            
        if torch.is_complex(ground_truth):
            gt_values = torch.abs(ground_truth)
        else:
            gt_values = ground_truth
        
        # Sort both distributions to compute Wasserstein-1 distance approximation
        # Convert to float for sorting if needed
        student_flat = student_values.flatten().float()
        gt_flat = gt_values.flatten().float()
        
        student_sorted, _ = torch.sort(student_flat)
        gt_sorted, _ = torch.sort(gt_flat)
        
        # Ensure same length for comparison
        min_len = min(len(student_sorted), len(gt_sorted))
        student_sorted = student_sorted[:min_len]
        gt_sorted = gt_sorted[:min_len]
        
        # Wasserstein-1 distance approximation
        wasserstein_loss = F.mse_loss(student_sorted, gt_sorted)
        
        return wasserstein_loss
    
    def _preprocess_for_loss(self, output: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess output and target tensors to ensure compatible shapes for loss computation.
        Based on TrainSSM.preprocess_output_and_prediction_before_comparison
        """
        # Handle extra dimensions in output
        if output.shape[-1] > 2:
            output = output[..., :2]
        elif output.shape[-1] == 2:
            if torch.is_complex(output):
                output = output[..., :1]
        
        # Handle extra dimensions in target
        if target.shape[-1] > 2:
            target = target[..., :2]
        elif target.shape[-1] == 2:
            if torch.is_complex(target):
                target = target[..., :1]

        # Squeeze extra dimensions
        if len(output.shape) > 3:
            output = output.squeeze(-1)
        if len(target.shape) > 3:
            target = target.squeeze(-1)

        return output, target
    
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        ground_truth: torch.Tensor,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distribution-preserving knowledge distillation loss.
        """
        # Standard student loss (ground truth alignment)
        student_loss = self.mse_loss(student_output, ground_truth)
        
        # Adaptive temperature for distillation
        adaptive_temp = self.compute_adaptive_temperature(teacher_output, student_output)
        
        # Distillation loss with adaptive temperature
        if torch.is_complex(student_output):
            # Handle complex data
            teacher_mag = torch.abs(teacher_output)
            student_mag = torch.abs(student_output)
            
            # Use magnitude for distillation with adaptive temperature
            teacher_soft = F.softmax(teacher_mag / adaptive_temp, dim=-1)
            student_log_soft = F.log_softmax(student_mag / adaptive_temp, dim=-1)
            distillation_loss = self.kl_div(student_log_soft, teacher_soft) * (adaptive_temp ** 2)
            
            # Also preserve phase relationships
            teacher_phase = torch.angle(teacher_output)
            student_phase = torch.angle(student_output)
            phase_loss = F.mse_loss(student_phase, teacher_phase)
            distillation_loss = distillation_loss + 0.3 * phase_loss
        else:
            # Real-valued distillation
            teacher_soft = F.softmax(teacher_output / adaptive_temp, dim=-1)
            student_log_soft = F.log_softmax(student_output / adaptive_temp, dim=-1)
            distillation_loss = self.kl_div(student_log_soft, teacher_soft) * (adaptive_temp ** 2)
        
        # NEW: Variance preservation loss
        variance_loss = self.compute_variance_preservation_loss(student_output, teacher_output, ground_truth)
        
        # NEW: Moment matching loss  
        moment_loss = self.compute_moment_matching_loss(student_output, ground_truth)
        
        # NEW: Confidence calibration loss
        confidence_loss = self.compute_confidence_calibration_loss(student_output, teacher_output)
        
        # NEW: Distribution alignment loss
        distribution_loss = self.compute_distribution_alignment_loss(student_output, ground_truth)
        
        # Feature matching loss (if available)
        feature_loss = torch.tensor(0.0, device=student_output.device)
        if student_features is not None and teacher_features is not None:
            if student_features.shape == teacher_features.shape:
                feature_loss = self.mse_loss(student_features, teacher_features)
        
        # Combined loss with distribution preservation terms
        total_loss = (
            self.alpha * student_loss +
            self.beta * distillation_loss +
            self.gamma * feature_loss +
            self.variance_weight * variance_loss +
            self.moment_weight * moment_loss +
            self.confidence_weight * confidence_loss +
            0.05 * distribution_loss  # Small weight for distribution alignment
        )
        
        return {
            'total_loss': total_loss,
            'student_loss': student_loss,
            'distillation_loss': distillation_loss,
            'feature_loss': feature_loss,
            'variance_loss': variance_loss,
            'moment_loss': moment_loss,
            'confidence_loss': confidence_loss,
            'distribution_loss': distribution_loss,
            'adaptive_temperature': torch.tensor(adaptive_temp, device=student_output.device)
        }


class DistributionStatisticsTracker:
    """
    Utility class to track and compare distribution statistics between
    student predictions, teacher predictions, and ground truth.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked statistics"""
        self.stats = {
            'student': {'mean': [], 'var': [], 'std': [], 'min': [], 'max': [], 'percentiles': []},
            'teacher': {'mean': [], 'var': [], 'std': [], 'min': [], 'max': [], 'percentiles': []},
            'ground_truth': {'mean': [], 'var': [], 'std': [], 'min': [], 'max': [], 'percentiles': []}
        }
    
    def update(self, student_output: torch.Tensor, teacher_output: torch.Tensor, ground_truth: torch.Tensor):
        """Update statistics with new batch of predictions"""
        
        def compute_stats(tensor: torch.Tensor) -> Dict:
            """Compute comprehensive statistics for a tensor"""
            # Handle complex tensors by using magnitude
            if torch.is_complex(tensor):
                flat_tensor = torch.abs(tensor).detach().cpu().flatten().float()
            else:
                flat_tensor = tensor.detach().cpu().flatten().float()
            
            percentiles = torch.quantile(flat_tensor, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
            
            return {
                'mean': torch.mean(flat_tensor).item(),
                'var': torch.var(flat_tensor).item(), 
                'std': torch.std(flat_tensor).item(),
                'min': torch.min(flat_tensor).item(),
                'max': torch.max(flat_tensor).item(),
                'percentiles': percentiles.tolist()
            }
        
        # Compute statistics for each type
        student_stats = compute_stats(student_output)
        teacher_stats = compute_stats(teacher_output)
        gt_stats = compute_stats(ground_truth)
        
        # Update tracked statistics
        for key in self.stats['student'].keys():
            self.stats['student'][key].append(student_stats[key])
            self.stats['teacher'][key].append(teacher_stats[key])
            self.stats['ground_truth'][key].append(gt_stats[key])
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics over all tracked batches"""
        summary = {}
        
        for model_type in ['student', 'teacher', 'ground_truth']:
            summary[model_type] = {}
            for stat_name in ['mean', 'var', 'std', 'min', 'max']:
                values = self.stats[model_type][stat_name]
                if values:
                    summary[model_type][f'{stat_name}_avg'] = sum(values) / len(values)
                    summary[model_type][f'{stat_name}_std'] = torch.tensor(values).std().item()
        
        return summary
    
    def compute_distribution_similarity(self) -> Dict:
        """Compute similarity metrics between distributions"""
        if not self.stats['student']['mean']:
            return {}
        
        similarity = {}
        
        # Compare student vs ground truth
        student_means = torch.tensor(self.stats['student']['mean'])
        gt_means = torch.tensor(self.stats['ground_truth']['mean'])
        student_vars = torch.tensor(self.stats['student']['var'])
        gt_vars = torch.tensor(self.stats['ground_truth']['var'])
        
        similarity['student_gt_mean_mse'] = F.mse_loss(student_means, gt_means).item()
        similarity['student_gt_var_mse'] = F.mse_loss(student_vars, gt_vars).item()
        
        # Compare student vs teacher
        teacher_means = torch.tensor(self.stats['teacher']['mean'])
        teacher_vars = torch.tensor(self.stats['teacher']['var'])
        
        similarity['student_teacher_mean_mse'] = F.mse_loss(student_means, teacher_means).item()
        similarity['student_teacher_var_mse'] = F.mse_loss(student_vars, teacher_vars).item()
        
        return similarity