"""
Knowledge Distillation Pipeline for sarSSM Models

This module implements knowledge distillation to transfer knowledge from a larger,
complex teacher model to a smaller, simpler student model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import copy
import yaml
from typing import Optional, Dict, Any, Tuple
import os
from pathlib import Path
from sarpyx.utils.losses import get_loss_function

# Import wandb with fallback
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Gradient tracking will be disabled.")

from model.model_utils import get_model_from_configs
from training.training_loops import TrainSSM
from dataloader.dataloader import SARDataloader

# Import distribution preserving distillation
from training.distribution_preserving_distillation import DistributionPreservingLoss, DistributionStatisticsTracker


class GradientTracker:
    """
    Utility class to track and log model gradients to wandb
    """
    
    def __init__(self, log_frequency: int = 100):
        self.log_frequency = log_frequency
        self.step_count = 0
    
    def log_gradients(self, model: nn.Module, model_name: str, global_step: int):
        """
        Log gradient statistics to wandb
        
        Args:
            model: The model to track gradients for
            model_name: Name prefix for logging (e.g., 'teacher', 'student')
            global_step: Current training step
        """
        if global_step % self.log_frequency != 0:
            return
            
        gradient_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                # Compute gradient statistics
                grad_norm = torch.norm(grad)
                grad_mean = torch.mean(grad)
                grad_std = torch.std(grad)
                grad_max = torch.max(torch.abs(grad))
                
                # Clean parameter name for logging
                clean_name = name.replace('.', '/')
                
                gradient_stats.update({
                    f"{model_name}/gradients/{clean_name}/norm": grad_norm.item(),
                    f"{model_name}/gradients/{clean_name}/mean": grad_mean.item(),
                    f"{model_name}/gradients/{clean_name}/std": grad_std.item(),
                    f"{model_name}/gradients/{clean_name}/max": grad_max.item()
                })
        
        # Log overall gradient statistics
        if gradient_stats:
            all_norms = [v for k, v in gradient_stats.items() if 'norm' in k]
            if all_norms:
                gradient_stats[f"{model_name}/gradients/overall/total_norm"] = sum(all_norms)
                gradient_stats[f"{model_name}/gradients/overall/mean_norm"] = sum(all_norms) / len(all_norms)
                gradient_stats[f"{model_name}/gradients/overall/max_norm"] = max(all_norms)
        
        # Log to wandb
        try:
            wandb.log(gradient_stats, step=global_step)
        except Exception as e:
            print(f"Warning: Failed to log gradients to wandb: {e}")
    
    def log_model_weights(self, model: nn.Module, model_name: str, global_step: int, frequency: int = 500):
        """
        Log model weight statistics to wandb
        
        Args:
            model: The model to track weights for
            model_name: Name prefix for logging
            global_step: Current training step
            frequency: How often to log weights
        """
        if global_step % frequency != 0:
            return
            
        weight_stats = {}
        
        for name, param in model.named_parameters():
            if param.data is not None:
                weight = param.data
                
                # Compute weight statistics
                weight_norm = torch.norm(weight)
                weight_mean = torch.mean(weight)
                weight_std = torch.std(weight)
                weight_max = torch.max(torch.abs(weight))
                
                # Clean parameter name for logging
                clean_name = name.replace('.', '/')
                
                weight_stats.update({
                    f"{model_name}/weights/{clean_name}/norm": weight_norm.item(),
                    f"{model_name}/weights/{clean_name}/mean": weight_mean.item(),
                    f"{model_name}/weights/{clean_name}/std": weight_std.item(),
                    f"{model_name}/weights/{clean_name}/max": weight_max.item()
                })
        
        # Log to wandb
        try:
            wandb.log(weight_stats, step=global_step)
        except Exception as e:
            print(f"Warning: Failed to log weights to wandb: {e}")


class GradientTracker:
    """
    Utility class to track and log model gradients to wandb
    """
    
    def __init__(self, log_frequency: int = 100):
        self.log_frequency = log_frequency
        self.step_count = 0
        self.enabled = WANDB_AVAILABLE
    
    def log_gradients(self, model: nn.Module, model_name: str, global_step: int):
        """
        Log gradient statistics to wandb
        
        Args:
            model: The model to track gradients for
            model_name: Name prefix for logging (e.g., 'teacher', 'student')
            global_step: Current training step
        """
        if not self.enabled or global_step % self.log_frequency != 0:
            return
            
        gradient_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                # Compute gradient statistics
                grad_norm = torch.norm(grad)
                grad_mean = torch.mean(grad)
                grad_std = torch.std(grad)
                grad_max = torch.max(torch.abs(grad))
                
                # Clean parameter name for logging
                clean_name = name.replace('.', '/')
                
                gradient_stats.update({
                    f"{model_name}/gradients/{clean_name}/norm": grad_norm.item(),
                    f"{model_name}/gradients/{clean_name}/mean": grad_mean.item(),
                    f"{model_name}/gradients/{clean_name}/std": grad_std.item(),
                    f"{model_name}/gradients/{clean_name}/max": grad_max.item()
                })
        
        # Log overall gradient statistics
        if gradient_stats:
            all_norms = [v for k, v in gradient_stats.items() if 'norm' in k]
            if all_norms:
                gradient_stats[f"{model_name}/gradients/overall/total_norm"] = sum(all_norms)
                gradient_stats[f"{model_name}/gradients/overall/mean_norm"] = sum(all_norms) / len(all_norms)
                gradient_stats[f"{model_name}/gradients/overall/max_norm"] = max(all_norms)
        
        # Log to wandb
        if WANDB_AVAILABLE:
            try:
                wandb.log(gradient_stats, step=global_step)
            except Exception as e:
                print(f"Warning: Failed to log gradients to wandb: {e}")
    
    def log_model_weights(self, model: nn.Module, model_name: str, global_step: int, frequency: int = 500):
        """
        Log model weight statistics to wandb
        
        Args:
            model: The model to track weights for
            model_name: Name prefix for logging
            global_step: Current training step
            frequency: How often to log weights
        """
        if not self.enabled or global_step % frequency != 0:
            return
            
        weight_stats = {}
        
        for name, param in model.named_parameters():
            if param.data is not None:
                weight = param.data
                
                # Compute weight statistics
                weight_norm = torch.norm(weight)
                weight_mean = torch.mean(weight)
                weight_std = torch.std(weight)
                weight_max = torch.max(torch.abs(weight))
                
                # Clean parameter name for logging
                clean_name = name.replace('.', '/')
                
                weight_stats.update({
                    f"{model_name}/weights/{clean_name}/norm": weight_norm.item(),
                    f"{model_name}/weights/{clean_name}/mean": weight_mean.item(),
                    f"{model_name}/weights/{clean_name}/std": weight_std.item(),
                    f"{model_name}/weights/{clean_name}/max": weight_max.item()
                })
        
        # Log to wandb
        if WANDB_AVAILABLE:
            try:
                wandb.log(weight_stats, step=global_step)
            except Exception as e:
                print(f"Warning: Failed to log weights to wandb: {e}")


class KnowledgeDistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation including:
    - Student loss (MSE with ground truth)
    - Distillation loss (KL divergence between teacher and student)
    - Feature matching loss (optional)
    """
    
    def __init__(
        self,
        temperature: float = 2.5,  # Reduced from 4.0 to provide sharper teacher signals
        alpha: float = 0.8,        # Increased from 0.5 to emphasize ground truth learning
        beta: float = 0.15,        # Reduced from 0.3 to reduce teacher dependency
        feature_matching: bool = True, 
        gt_loss_fn_name: str = "complex_mse",
        feature_loss_fn_name: str = "complex_mse"
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for student loss
        self.beta = beta    # Weight for distillation loss
        self.gamma = 1.0 - alpha - beta  # Weight for feature matching
        self.feature_matching = feature_matching
        self.gt_loss = get_loss_function(gt_loss_fn_name) #nn.MSELoss()
        self.feature_loss = get_loss_function(feature_loss_fn_name) #nn.MSELoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
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
        Compute knowledge distillation loss
        
        Args:
            student_output: Student model predictions (B, L, D)
            teacher_output: Teacher model predictions (B, L, D)
            ground_truth: Ground truth targets (B, L, D)
            student_features: Optional intermediate features from student
            teacher_features: Optional intermediate features from teacher
            
        Returns:
            Dictionary containing individual loss components and total loss
        """
        
        # Student loss (standard MSE with ground truth)
        # Preprocess outputs to ensure compatible shapes
        student_output_processed, ground_truth_processed = self._preprocess_for_loss(student_output, ground_truth)
        student_loss = self.gt_loss(student_output_processed, ground_truth_processed)
        # Ensure student loss is real
        if torch.is_complex(student_loss):
            student_loss = student_loss.real
        
        # Distillation loss (soft targets from teacher)
        # For regression, we use MSE instead of KL divergence
        
        # Handle different output dimensions between teacher and student
        if student_output.shape != teacher_output.shape:
            # Adapt teacher output to match student output dimensions
            if len(teacher_output.shape) == 3 and len(student_output.shape) == 3:
                B_t, L_t, D_t = teacher_output.shape
                B_s, L_s, D_s = student_output.shape
                
                teacher_adapted = teacher_output
                if L_t != L_s:
                    # Adapt sequence length
                    teacher_adapted = F.adaptive_avg_pool1d(
                        teacher_adapted.transpose(1, 2),
                        L_s
                    ).transpose(1, 2)
                
                if D_t != D_s:
                    # Adapt output dimension using adaptive pooling or projection
                    if D_t > D_s:
                        # Downsample: use adaptive pooling
                        teacher_adapted = F.adaptive_avg_pool1d(
                            teacher_adapted.transpose(-1, -2),
                            D_s
                        ).transpose(-1, -2)
                    else:
                        # Upsample: replicate or use linear projection
                        # Simple replication for now
                        repeat_factor = D_s // D_t
                        remainder = D_s % D_t
                        teacher_adapted = teacher_adapted.repeat(1, 1, repeat_factor)
                        if remainder > 0:
                            teacher_adapted = torch.cat([
                                teacher_adapted,
                                teacher_adapted[..., :remainder]
                            ], dim=-1)
                
                teacher_output_adapted = teacher_adapted
            else:
                teacher_output_adapted = teacher_output
        else:
            teacher_output_adapted = teacher_output
        

        teacher_processed, student_processed = self._preprocess_for_loss(teacher_output_adapted, student_output)
        distillation_loss = self.feature_loss(student_processed, teacher_processed)
        # Ensure distillation loss is real
        if torch.is_complex(distillation_loss):
            distillation_loss = distillation_loss.real
        
        # Feature matching loss (optional) - Now with sophisticated alignment
        feature_loss = torch.tensor(0.0, device=student_output.device)
        if self.feature_matching and student_features is not None and teacher_features is not None:
            # Debug: Print feature shapes
            # print(f"Computing feature matching loss - Student features: {student_features.shape}, Teacher features: {teacher_features.shape}")
            try:
                # For now, use the existing simple alignment
                # The sophisticated alignment will be used in the trainer class
                if student_features.shape != teacher_features.shape:
                    # Adapt feature dimensions if different (for different model/state dimensions)
                    if len(teacher_features.shape) == 3 and len(student_features.shape) == 3:
                        B_t, L_t, D_t = teacher_features.shape
                        B_s, L_s, D_s = student_features.shape
                        
                        if L_t != L_s:
                            # Adapt sequence length
                            teacher_feat_adapted = F.adaptive_avg_pool1d(
                                teacher_features.transpose(1, 2),
                                L_s
                            ).transpose(1, 2)
                        else:
                            teacher_feat_adapted = teacher_features
                        
                        if D_t != D_s:
                            # Adapt feature dimension using linear projection
                            proj = nn.Linear(D_t, D_s, device=teacher_features.device, dtype=teacher_features.dtype)
                            teacher_feat_adapted = proj(teacher_feat_adapted)
                        
                        # Preprocess features before computing loss
                        teacher_feat_processed, student_feat_processed = self._preprocess_for_loss(teacher_feat_adapted, student_features)
                        feature_loss = self.feature_loss(student_feat_processed, teacher_feat_processed)
                        # Ensure feature loss is real
                        if torch.is_complex(feature_loss):
                            feature_loss = feature_loss.real
                else:
                    # Preprocess features before computing loss
                    teacher_feat_processed, student_feat_processed = self._preprocess_for_loss(teacher_features, student_features)
                    feature_loss = self.feature_loss(student_feat_processed, teacher_feat_processed)
                    # Ensure feature loss is real
                    if torch.is_complex(feature_loss):
                        feature_loss = feature_loss.real
            except Exception as e:
                print(f"Warning: Feature matching loss computation failed: {e}")
                feature_loss = torch.tensor(0.0, device=student_output.device)
        
        # Combined loss
        total_loss = (
            self.alpha * student_loss +
            self.beta * distillation_loss +
            self.gamma * feature_loss
        )
        
        # Ensure total loss is real
        if torch.is_complex(total_loss):
            total_loss = total_loss.real
        
        return {
            'total_loss': total_loss,
            'student_loss': student_loss,
            'distillation_loss': distillation_loss,
            'feature_loss': feature_loss
        }


class KnowledgeDistillationTrainer(TrainSSM):
    """
    Knowledge Distillation trainer that extends TrainSSM to support teacher-student training
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        teacher_checkpoint_path: str,
        base_save_dir: str,
        train_loader: SARDataloader,
        val_loader: SARDataloader,
        test_loader: SARDataloader,
        inference_loader: Optional[SARDataloader] = None,
        mode: str = "parallel",
        temperature: float = 2.5,   # Reduced from 4.0 for sharper teacher signals
        alpha: float = 0.8,         # Increased from 0.5 for stronger ground truth focus 
        beta: float = 0.15,         # Reduced from 0.3 to prevent teacher over-dependence
        feature_matching: bool = True,
        freeze_teacher: bool = True,
        lr: float = 1e-4,
        gt_loss_fn_name: str = 'complex_mse',
        feature_loss_fn_name: str = 'complex_mse',
        curriculum_learning: bool = True,  # Enable curriculum learning to prevent mode collapse
        curriculum_epochs: int = 10,       # Number of epochs with pure student learning
        progressive_layers: bool = False,  # Enable progressive layer coupling
        teacher_layers: int = 4,           # Number of teacher layers
        student_layers: int = 4,           # Number of student layers
        stage_epochs: int = 15,            # Epochs per progressive stage
        # NEW: Distribution preservation parameters
        preserve_distribution: bool = True,     # Enable distribution preservation
        variance_weight: float = 0.15,          # Variance preservation weight
        moment_weight: float = 0.1,             # Moment matching weight
        confidence_weight: float = 0.05,        # Confidence calibration weight
        dynamic_temperature: bool = True,       # Enable adaptive temperature
        input_dim: int = 4,
        **kwargs
    ):
        """
        Initialize Knowledge Distillation Trainer
        
        Args:
            student_model: The smaller model to be trained
            teacher_model: The larger, pre-trained model
            teacher_checkpoint_path: Path to teacher model checkpoint
            base_save_dir: Directory to save student model checkpoints
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            inference_loader: Optional inference data loader
            mode: Training mode
            temperature: Temperature for softmax (distillation)
            alpha: Weight for student loss
            beta: Weight for distillation loss
            feature_matching: Whether to include feature matching loss
            freeze_teacher: Whether to freeze teacher model weights
            lr: Learning rate
            input_dim: Input data dimension for student(e.g., 4 for complex data with real and imag channels)
        """
        
        # Initialize parent with student model
        super().__init__(
            base_save_dir=base_save_dir,
            model=student_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            inference_loader=inference_loader,
            mode=mode,
            lr=lr,
            criterion=get_loss_function(gt_loss_fn_name), 
            input_dim=input_dim,
            #**kwargs
        )
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.teacher_checkpoint_path = teacher_checkpoint_path
        self.progressive_layers = progressive_layers
        # Load teacher model checkpoint
        self._load_teacher_checkpoint()
        
        # if freeze_teacher:
        #     # Freeze teacher model parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
        # Store progressive layer parameters
        self.teacher_layers = teacher_layers
        self.student_layers = student_layers
        self.stage_epochs = stage_epochs
        self.preserve_distribution = preserve_distribution
        
        # Initialize distillation loss - SIMPLIFIED TO MOST EFFECTIVE APPROACHES
        if progressive_layers:
            # Use progressive layer coupling - the method you want
            from training.progressive_layer_coupling import ProgressiveLayerCouplingLoss
            self.distillation_criterion = ProgressiveLayerCouplingLoss(
                teacher_layers=teacher_layers,
                student_layers=student_layers,
                stage_epochs=stage_epochs,
                alpha=alpha,
                beta=beta,
                gamma=1.0 - alpha - beta,
                temperature=temperature,
                loss_fn_name=gt_loss_fn_name
            )
            print("✅ Progressive Layer Coupling enabled")
            print(f"   Strategy: Layer-by-layer feature matching across {len(self.distillation_criterion.layer_couplings)} stages")
            print(f"   Stage epochs: {stage_epochs}")
            print(f"   Final stage: Ground truth only")
        else:
            # Use distribution-preserving knowledge distillation (most effective standard approach)
            self.distillation_criterion = DistributionPreservingLoss(
                temperature=temperature,
                alpha=alpha,
                beta=beta,
                gamma=1.0 - alpha - beta,
                variance_weight=variance_weight,
                moment_weight=moment_weight,
                confidence_weight=confidence_weight,
                dynamic_temperature=dynamic_temperature,
                loss_fn_name=gt_loss_fn_name
            )
            print("✅ Distribution-preserving knowledge distillation enabled")
            print(f"   Most effective for preventing mode collapse")
        
        self.preserve_distribution = preserve_distribution and not progressive_layers  # Don't mix strategies
        
        # Curriculum learning parameters
        self.curriculum_learning = curriculum_learning
        self.curriculum_epochs = curriculum_epochs
        self.original_alpha = alpha
        self.original_beta = beta
        
        # Initialize gradient tracking
        self.gradient_tracker = GradientTracker(log_frequency=50)
        self.training_step_count = 0
        
        # Initialize distribution statistics tracker (if using distribution preservation)
        if preserve_distribution:
            self.distribution_tracker = DistributionStatisticsTracker()
            print("✅ Distribution-preserving knowledge distillation enabled")
            print(f"   Variance weight: {variance_weight}")
            print(f"   Moment weight: {moment_weight}")
            print(f"   Confidence weight: {confidence_weight}")
            print(f"   Dynamic temperature: {dynamic_temperature}")
        else:
            self.distribution_tracker = None
        
        # Ensure models are in the correct dtype (float32 for mixed precision compatibility)
        self.student_model = self.student_model.float()
        self.teacher_model = self.teacher_model.float()
        
        # Initialize layer projection modules for progressive feature coupling
        self._layer_projections = nn.ModuleDict()
        
        # Initialize wandb logging
        if WANDB_AVAILABLE:
            try:
                wandb.init(
                    project="knowledge_distillation",
                    name=f"kd_{Path(base_save_dir).name}",
                    config={
                        "temperature": temperature,
                        "alpha": alpha,
                        "beta": beta,
                        "learning_rate": lr,
                        "curriculum_learning": curriculum_learning,
                        "curriculum_epochs": curriculum_epochs,
                        "preserve_distribution": preserve_distribution,
                        "variance_weight": variance_weight,
                        "moment_weight": moment_weight,
                        "confidence_weight": confidence_weight,
                        "dynamic_temperature": dynamic_temperature,
                        "progressive_layers": progressive_layers,
                        "teacher_layers": teacher_layers,
                        "student_layers": student_layers,
                        "stage_epochs": stage_epochs,
                        "feature_matching": feature_matching
                    }
                )
                print("✅ Wandb logging initialized")
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
        
        self.save_hyperparameters(ignore=['student_model', 'teacher_model'])
    
    def log_with_wandb(self, name: str, value, **kwargs):
        """Log to both PyTorch Lightning and wandb"""
        # Log to PyTorch Lightning
        self.log(name, value, **kwargs)
        
        # Log to wandb if available
        if WANDB_AVAILABLE:
            try:
                wandb.log({name: value}, step=self.global_step)
            except Exception as e:
                pass  # Silently continue if wandb logging fails
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        super().on_train_epoch_end()
        
        # Log distribution statistics if tracking is enabled
        if self.distribution_tracker is not None:
            stats = self.distribution_tracker.get_summary_stats()
            similarity = self.distribution_tracker.compute_distribution_similarity()
            
            # Log summary statistics
            for model_type in ['student', 'teacher', 'ground_truth']:
                if model_type in stats:
                    for stat_name, value in stats[model_type].items():
                        self.log_with_wandb(f'distribution/{model_type}_{stat_name}', value, on_epoch=True)
            
            # Log similarity metrics
            for sim_name, value in similarity.items():
                self.log_with_wandb(f'distribution_similarity/{sim_name}', value, on_epoch=True)
            
            # Reset tracker for next epoch
            self.distribution_tracker.reset()
            
            # Log additional distribution preservation metrics if using distribution-preserving loss
            if hasattr(self.distillation_criterion, 'teacher_var_ema'):
                self.log_with_wandb('distribution/teacher_var_ema', self.distillation_criterion.teacher_var_ema.item(), on_epoch=True)
                self.log_with_wandb('distribution/student_var_ema', self.distillation_criterion.student_var_ema.item(), on_epoch=True)
                
                # Compute variance ratio (should be close to 1.0 for good distribution preservation)
                var_ratio = self.distillation_criterion.student_var_ema / (self.distillation_criterion.teacher_var_ema + 1e-8)
                self.log_with_wandb('distribution/variance_ratio', var_ratio.item(), on_epoch=True)
        
    def _load_teacher_checkpoint(self):
        """Load teacher model from checkpoint"""
        if os.path.exists(self.teacher_checkpoint_path):
            checkpoint = torch.load(self.teacher_checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                # PyTorch Lightning checkpoint
                state_dict = checkpoint['state_dict']
                # Remove 'model.' prefix if present
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('model.'):
                        new_key = key[6:]  # Remove 'model.' prefix
                    else:
                        new_key = key
                    # Ensure weights are in float32
                    new_state_dict[new_key] = value.float() if value.dtype in [torch.float16, torch.float64] else value
                self.teacher_model.load_state_dict(new_state_dict)
            else:
                # Direct state dict - ensure float32 dtype
                state_dict = {}
                for key, value in checkpoint.items():
                    # Only convert tensor values, skip non-tensor items
                    if isinstance(value, torch.Tensor):
                        state_dict[key] = value.float() if value.dtype in [torch.float16, torch.float64] else value
                    else:
                        state_dict[key] = value
                self.teacher_model.load_state_dict(state_dict)
                
            print(f"Loaded teacher model from {self.teacher_checkpoint_path}")
            
            # Ensure teacher model is in float32
            self.teacher_model = self.teacher_model.float()
        else:
            raise FileNotFoundError(f"Teacher checkpoint not found: {self.teacher_checkpoint_path}")
    


    def extract_features(self, model: nn.Module, x: torch.Tensor, is_student: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract intermediate features from the model for feature matching
        
        Args:
            model: The model to extract features from
            x: Input tensor
            
        Returns:
            Tuple of (output, intermediate_features)
        """
        if hasattr(model, 'extract_features'):
            # If model has built-in feature extraction
            return model.extract_features(x)
        else:
            # For sarSSM models, extract features from intermediate layers
            ssm_layers = None
            if hasattr(model, 'layers') and len(model.layers) > 0:
                ssm_layers = model.layers
            elif hasattr(model, 'ssm') and len(model.ssm) > 0:
                # For sarSSMFinal model
                ssm_layers = model.ssm
            
            if ssm_layers and len(ssm_layers) > 0:
                # Hook into intermediate layer to extract features
                features = []
                
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        # S4D returns (output, state) - use output for feature matching
                        if is_student:
                            # For student, store the state update: state = state + new_state
                            # This corresponds to the additive update mentioned in requirements
                            features.append(output[0])  # Use the output which represents state update
                        else:
                            features.append(output[0])  # Teacher uses standard output
                    else:
                        features.append(output)
                
                # Register hook on middle layer
                middle_layer_idx = len(ssm_layers) // 2
                hook = ssm_layers[middle_layer_idx].register_forward_hook(hook_fn)
                
                try:
                    output = model(x)
                    intermediate_features = features[0] if features else None
                    
                    # Convert features to consistent format (B, L, D)
                    if intermediate_features is not None and len(intermediate_features.shape) == 3:
                        if intermediate_features.shape[1] != x.shape[1]:  # If (B, D, L) format
                            intermediate_features = intermediate_features.transpose(1, 2)
                    
                    return output, intermediate_features
                finally:
                    hook.remove()
            else:
                # Default: return final output and None for features
                output = model(x)
                return output, None
    
    def _get_strategic_layers(self, num_layers: int) -> list:
        """
        Get strategic layer indices for feature extraction
        
        Args:
            num_layers: Total number of layers in the model
            
        Returns:
            List of layer indices to extract features from
        """
        if num_layers <= 2:
            return [0]  # Just first layer for very small models
        elif num_layers <= 4:
            return [0, num_layers - 1]  # First and last layers
        else:
            # For larger models, extract from early, middle, and late layers
            early = num_layers // 4
            middle = num_layers // 2
            late = 3 * num_layers // 4
            return [early, middle, late]
    
    def _combine_multi_layer_features(self, all_features: list, input_shape: tuple) -> torch.Tensor:
        """
        Combine features from multiple layers into a single representation
        
        Args:
            all_features: List of (layer_idx, feature_tensor) tuples
            input_shape: Shape of the input tensor for reference
            
        Returns:
            Combined feature tensor
        """
        if not all_features:
            return None
        
        # Extract just the feature tensors
        features = [feat for _, feat in all_features]
        
        # Ensure all features have consistent format (B, L, D)
        normalized_features = []
        for feat in features:
            if len(feat.shape) == 3:
                # Check if we need to transpose from (B, D, L) to (B, L, D)
                if feat.shape[1] != input_shape[1] and feat.shape[2] == input_shape[1]:
                    feat = feat.transpose(1, 2)
                normalized_features.append(feat)
        
        if not normalized_features:
            return None
        
        # Resize all features to the same sequence length (use input length as reference)
        target_seq_len = input_shape[1]
        resized_features = []
        
        for feat in normalized_features:
            if feat.shape[1] != target_seq_len:
                # Adapt sequence length
                if torch.is_complex(feat):
                    # Handle complex features separately
                    feat_real = F.adaptive_avg_pool1d(
                        feat.real.transpose(1, 2), target_seq_len
                    ).transpose(1, 2)
                    feat_imag = F.adaptive_avg_pool1d(
                        feat.imag.transpose(1, 2), target_seq_len
                    ).transpose(1, 2)
                    feat_resized = torch.complex(feat_real, feat_imag)
                else:
                    feat_resized = F.adaptive_avg_pool1d(
                        feat.transpose(1, 2), target_seq_len
                    ).transpose(1, 2)
            else:
                feat_resized = feat
            resized_features.append(feat_resized)
        
        # Combine features: for multiple features, concatenate along the feature dimension
        if len(resized_features) == 1:
            return resized_features[0]
        
        combined = torch.cat(resized_features, dim=-1)
        return combined
    
    def _compute_progressive_feature_coupling_loss(self, teacher_features_list: list, student_features_list: list, current_epoch: int) -> torch.Tensor:
        """
        Compute progressive feature coupling loss with gradual layer introduction
        
        This method:
        1. Creates feature correspondences between teacher and student layers
        2. Gradually introduces features from shallow to deep layers across epochs
        3. Weights losses progressively to focus on different layer depths over training
        
        Args:
            teacher_features_list: List of feature tensors from teacher layers
            student_features_list: List of feature tensors from student layers  
            current_epoch: Current training epoch for progressive weighting
            
        Returns:
            Progressive feature coupling loss
        """
        if not teacher_features_list or not student_features_list:
            return torch.tensor(0.0, device=next(iter(teacher_features_list or student_features_list)).device)
        
        num_teacher_layers = len(teacher_features_list)
        num_student_layers = len(student_features_list)
        
        # Create layer correspondences
        layer_correspondences = self._create_layer_correspondences(num_teacher_layers, num_student_layers)
        
        # Progressive weighting based on current epoch
        total_epochs = getattr(self.hparams, 'num_epochs', 100)  # Default to 100 if not specified
        epoch_progress = min(current_epoch / max(total_epochs - 1, 1), 1.0)  # Normalize to [0, 1]
        
        total_loss = torch.tensor(0.0, device=teacher_features_list[0].device)
        total_weight = 0.0
        
        # Compute loss for each layer correspondence with progressive weighting
        for i, (teacher_idx, student_idx) in enumerate(layer_correspondences):
            # Progressive weight: early layers get more weight initially, later layers get more weight later
            layer_depth_ratio = i / max(len(layer_correspondences) - 1, 1)  # [0, 1] from shallow to deep
            
            # Sigmoid-based progressive weighting
            # Early epochs: focus on shallow layers (low layer_depth_ratio)
            # Late epochs: focus on deep layers (high layer_depth_ratio)
            progress_shift = layer_depth_ratio - epoch_progress
            layer_weight = torch.sigmoid(torch.tensor(-5.0 * progress_shift))  # Sharper transition
            
            # Additional emphasis on final output in later epochs
            if i == len(layer_correspondences) - 1:  # Last layer (closest to output)
                output_emphasis = 1.0 + 2.0 * epoch_progress  # Gradually increase importance
                layer_weight = layer_weight * output_emphasis
            
            if layer_weight > 0.01:  # Only compute loss if weight is significant
                # Get corresponding features
                teacher_feat = teacher_features_list[teacher_idx]
                student_feat = student_features_list[student_idx]
                
                # Align features for this layer pair
                teacher_feat_aligned, student_feat_aligned = self._align_single_feature_pair(
                    teacher_feat, student_feat
                )
                
                # Compute feature matching loss for this layer pair
                layer_loss = self._compute_single_layer_feature_loss(
                    teacher_feat_aligned, student_feat_aligned
                )
                
                # Ensure layer loss is real
                if torch.is_complex(layer_loss):
                    layer_loss = layer_loss.real
                
                # Weight and accumulate
                weighted_loss = layer_weight * layer_loss
                total_loss = total_loss + weighted_loss
                total_weight += layer_weight.item()
        
        # Normalize by total weight to maintain consistent loss scale
        if total_weight > 0:
            total_loss = total_loss / total_weight
        
        # Ensure final progressive loss is real
        if torch.is_complex(total_loss):
            total_loss = total_loss.real
        
        return total_loss
    
    def _create_layer_correspondences(self, num_teacher_layers: int, num_student_layers: int) -> list:
        """
        Create correspondences between teacher and student layers
        
        Args:
            num_teacher_layers: Number of teacher layers
            num_student_layers: Number of student layers
            
        Returns:
            List of (teacher_idx, student_idx) tuples
        """
        correspondences = []
        
        if num_teacher_layers == num_student_layers:
            # 1:1 mapping (0-0, 1-1, 2-2, etc.)
            correspondences = [(i, i) for i in range(num_student_layers)]
        elif num_teacher_layers > num_student_layers:
            # Map multiple teacher layers to each student layer
            teacher_per_student = num_teacher_layers / num_student_layers
            for student_idx in range(num_student_layers):
                teacher_idx = int(student_idx * teacher_per_student)
                correspondences.append((teacher_idx, student_idx))
        else:
            # Map teacher layers to subset of student layers
            student_per_teacher = num_student_layers / num_teacher_layers
            for teacher_idx in range(num_teacher_layers):
                student_idx = int(teacher_idx * student_per_teacher)
                correspondences.append((teacher_idx, student_idx))
        
        return correspondences
    
    def _align_single_feature_pair(self, teacher_feat: torch.Tensor, student_feat: torch.Tensor) -> tuple:
        """
        Align a single pair of teacher-student features
        
        Args:
            teacher_feat: Teacher feature tensor (B, L, D_t)
            student_feat: Student feature tensor (B, L, D_s)
            
        Returns:
            Tuple of aligned (teacher_feat, student_feat)
        """
        if teacher_feat is None or student_feat is None:
            return teacher_feat, student_feat
        
        # Ensure both features are in (B, L, D) format
        if len(teacher_feat.shape) != 3 or len(student_feat.shape) != 3:
            return teacher_feat, student_feat
        
        B_t, L_t, D_t = teacher_feat.shape
        B_s, L_s, D_s = student_feat.shape
        
        # Step 1: Align sequence lengths
        if L_t != L_s:
            # Use the student's sequence length as target
            if torch.is_complex(teacher_feat):
                teacher_real = F.adaptive_avg_pool1d(
                    teacher_feat.real.transpose(1, 2), L_s
                ).transpose(1, 2)
                teacher_imag = F.adaptive_avg_pool1d(
                    teacher_feat.imag.transpose(1, 2), L_s
                ).transpose(1, 2)
                teacher_feat = torch.complex(teacher_real, teacher_imag)
            else:
                teacher_feat = F.adaptive_avg_pool1d(
                    teacher_feat.transpose(1, 2), L_s
                ).transpose(1, 2)
        
        # Step 2: Align feature dimensions using learned projection
        if D_t != D_s:
            # Create or reuse a projection layer for this specific dimension pair
            proj_key = f"layer_proj_{D_t}_to_{D_s}"
            
            if not hasattr(self, '_layer_projections'):
                self._layer_projections = nn.ModuleDict()
            
            if proj_key not in self._layer_projections:
                # Create a projection layer with proper initialization
                proj_layer = nn.Sequential(
                    nn.Linear(D_t, D_s, device=teacher_feat.device, dtype=teacher_feat.dtype),
                    nn.LayerNorm(D_s, device=teacher_feat.device, dtype=teacher_feat.dtype) if not torch.is_complex(teacher_feat) else nn.Identity()
                )
                
                # Initialize projection weights
                if D_t > D_s:
                    nn.init.xavier_uniform_(proj_layer[0].weight)
                else:
                    nn.init.kaiming_uniform_(proj_layer[0].weight, a=0.1)
                nn.init.zeros_(proj_layer[0].bias)
                self._layer_projections[proj_key] = proj_layer
            
            # Apply projection
            teacher_feat_aligned = self._layer_projections[proj_key](teacher_feat)
        else:
            teacher_feat_aligned = teacher_feat
        
        return teacher_feat_aligned, student_feat
    
    def _compute_single_layer_feature_loss(self, teacher_feat: torch.Tensor, student_feat: torch.Tensor) -> torch.Tensor:
        """
        Compute feature matching loss for a single layer pair
        
        Args:
            teacher_feat: Aligned teacher feature tensor
            student_feat: Student feature tensor
            
        Returns:
            Feature matching loss for this layer
        """
        if teacher_feat is None or student_feat is None:
            return torch.tensor(0.0, device=student_feat.device if student_feat is not None else teacher_feat.device)
        
        # Preprocess features before computing loss
        teacher_feat_processed, student_feat_processed = self.distillation_criterion._preprocess_for_loss(teacher_feat, student_feat)
        
        # Multi-scale loss computation for single layer
        losses = []
        
        # 1. Direct MSE loss
        direct_loss = self.distillation_criterion.mse_loss(student_feat_processed, teacher_feat_processed)
        # Ensure loss is real
        if torch.is_complex(direct_loss):
            direct_loss = direct_loss.real
        losses.append(direct_loss)
        
        # 2. Cosine similarity loss (reduced weight)
        if teacher_feat_processed.shape[-1] > 1:  # Only if feature dim > 1
            if torch.is_complex(teacher_feat_processed) or torch.is_complex(student_feat_processed):
                # For complex tensors, compute cosine similarity using magnitude
                teacher_mag = torch.abs(teacher_feat_processed)
                student_mag = torch.abs(student_feat_processed)
                teacher_norm = F.normalize(teacher_mag, p=2, dim=-1)
                student_norm = F.normalize(student_mag, p=2, dim=-1)
                cosine_sim = F.cosine_similarity(teacher_norm, student_norm, dim=-1)
            else:
                # For real tensors, use standard cosine similarity
                teacher_norm = F.normalize(teacher_feat_processed, p=2, dim=-1)
                student_norm = F.normalize(student_feat_processed, p=2, dim=-1)
                cosine_sim = F.cosine_similarity(teacher_norm, student_norm, dim=-1)
            cosine_loss = 1 - cosine_sim.mean()  # Convert similarity to loss
            # Ensure loss is real
            if torch.is_complex(cosine_loss):
                cosine_loss = cosine_loss.real
            losses.append(0.1 * cosine_loss)  # Reduced weight
        
        # Combine losses
        total_layer_loss = sum(losses)
        # Ensure final loss is real
        if torch.is_complex(total_layer_loss):
            total_layer_loss = total_layer_loss.real
        return total_layer_loss

    def _align_features_for_distillation(self, teacher_features: torch.Tensor, student_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align teacher and student features for effective knowledge distillation
        
        This method handles:
        1. Different feature dimensions between teacher and student
        2. Different sequence lengths
        3. Complex vs real-valued features
        4. Semantic alignment through learned projections
        
        Args:
            teacher_features: Features from teacher model (B, L, D_t)
            student_features: Features from student model (B, L, D_s)
            
        Returns:
            Tuple of aligned (teacher_features, student_features)
        """
        if teacher_features is None or student_features is None:
            return teacher_features, student_features
        
        # Ensure both features are in (B, L, D) format
        if len(teacher_features.shape) != 3 or len(student_features.shape) != 3:
            return teacher_features, student_features
        
        B_t, L_t, D_t = teacher_features.shape
        B_s, L_s, D_s = student_features.shape
        
        # Step 1: Align sequence lengths
        if L_t != L_s:
            # Use the student's sequence length as target
            if torch.is_complex(teacher_features):
                teacher_real = F.adaptive_avg_pool1d(
                    teacher_features.real.transpose(1, 2), L_s
                ).transpose(1, 2)
                teacher_imag = F.adaptive_avg_pool1d(
                    teacher_features.imag.transpose(1, 2), L_s
                ).transpose(1, 2)
                teacher_features = torch.complex(teacher_real, teacher_imag)
            else:
                teacher_features = F.adaptive_avg_pool1d(
                    teacher_features.transpose(1, 2), L_s
                ).transpose(1, 2)
        
        # Step 2: Align feature dimensions using learned projection
        if D_t != D_s:
            # Create or reuse a projection layer for this specific dimension pair
            proj_key = f"teacher_to_student_{D_t}_to_{D_s}"
            
            if not hasattr(self, '_feature_projections'):
                self._feature_projections = nn.ModuleDict()
            
            if proj_key not in self._feature_projections:
                # Create a projection layer with proper initialization
                proj_layer = nn.Sequential(
                    nn.Linear(D_t, D_s, device=teacher_features.device, dtype=teacher_features.dtype),
                    nn.LayerNorm(D_s, device=teacher_features.device, dtype=teacher_features.dtype) if not torch.is_complex(teacher_features) else nn.Identity()
                )
                
                # Initialize projection weights for better convergence
                if D_t > D_s:
                    # Downsampling: use Xavier initialization
                    nn.init.xavier_uniform_(proj_layer[0].weight)
                else:
                    # Upsampling: use He initialization scaled down
                    nn.init.kaiming_uniform_(proj_layer[0].weight, a=0.1)
                
                nn.init.zeros_(proj_layer[0].bias)
                self._feature_projections[proj_key] = proj_layer
            
            # Apply projection
            teacher_features_aligned = self._feature_projections[proj_key](teacher_features)
        else:
            teacher_features_aligned = teacher_features
        
        return teacher_features_aligned, student_features
    
    def _compute_multi_scale_feature_loss(self, teacher_features: torch.Tensor, student_features: torch.Tensor) -> torch.Tensor:
        """
        Compute feature matching loss with multi-scale considerations
        
        Args:
            teacher_features: Aligned teacher features
            student_features: Student features
            
        Returns:
            Feature matching loss
        """
        complexmse = get_loss_function("complex_mse")
        if teacher_features is None or student_features is None:
            return torch.tensor(0.0, device=student_features.device if student_features is not None else teacher_features.device)
    
        # Preprocess features before computing loss
        teacher_feat_processed, student_feat_processed = self.distillation_criterion._preprocess_for_loss(teacher_features, student_features)
        
        # Multi-scale loss computation
        losses = []
        
        # 1. Direct MSE loss
        direct_loss = self.distillation_criterion.mse_loss(student_feat_processed, teacher_feat_processed)
        # Ensure loss is real
        if torch.is_complex(direct_loss):
            direct_loss = direct_loss.real
        losses.append(direct_loss)
        
        # 2. Cosine similarity loss (for semantic alignment) - Reduced weight
        if teacher_feat_processed.shape[-1] > 1:  # Only if feature dim > 1
            if torch.is_complex(teacher_feat_processed) or torch.is_complex(student_feat_processed):
                # For complex tensors, compute cosine similarity using magnitude
                teacher_mag = torch.abs(teacher_feat_processed)
                student_mag = torch.abs(student_feat_processed)
                teacher_norm = F.normalize(teacher_mag, p=2, dim=-1)
                student_norm = F.normalize(student_mag, p=2, dim=-1)
                cosine_sim = F.cosine_similarity(teacher_norm, student_norm, dim=-1)
            else:
                # For real tensors, use standard cosine similarity
                teacher_norm = F.normalize(teacher_feat_processed, p=2, dim=-1)
                student_norm = F.normalize(student_feat_processed, p=2, dim=-1)
                cosine_sim = F.cosine_similarity(teacher_norm, student_norm, dim=-1)
            cosine_loss = 1 - cosine_sim.mean()  # Convert similarity to loss
            # Ensure loss is real
            if torch.is_complex(cosine_loss):
                cosine_loss = cosine_loss.real
            losses.append(0.02 * cosine_loss)  # Reduced from 0.1 to 0.02
        
        # 3. Distribution matching (statistical moments) - Reduced weight
        teacher_mean = torch.mean(teacher_feat_processed, dim=1)  # (B, D)
        student_mean = torch.mean(student_feat_processed, dim=1)  # (B, D)
        mean_loss = complexmse(student_mean, teacher_mean)  # Fixed: use F.mse_loss instead of self.compute_loss
        # Ensure loss is real
        if torch.is_complex(mean_loss):
            mean_loss = mean_loss.real
        losses.append(0.01 * mean_loss)  # Reduced from 0.05 to 0.01
        
        # Combine all losses
        total_feature_loss = sum(losses)
        # Ensure final loss is real
        if torch.is_complex(total_feature_loss):
            total_feature_loss = total_feature_loss.real
        return total_feature_loss
    
    
    def _update_curriculum_weights(self, current_epoch: int):
        """
        Update loss weights based on curriculum learning schedule
        
        Args:
            current_epoch: Current training epoch
        """
        if not self.curriculum_learning:
            return
            
        if current_epoch < self.curriculum_epochs:
            # Pure student learning phase - focus on ground truth
            alpha = 1.0
            beta = 0.0
            gamma = 0.0
        else:
            # Gradually introduce distillation
            progress = (current_epoch - self.curriculum_epochs) / max(1, self.curriculum_epochs)
            progress = min(progress, 1.0)  # Cap at 1.0
            
            # Smooth transition to final weights
            alpha = 1.0 - progress * (1.0 - self.original_alpha)
            beta = progress * self.original_beta
            gamma = progress * (1.0 - self.original_alpha - self.original_beta)
        
        # Update distillation criterion weights
        self.distillation_criterion.alpha = alpha
        self.distillation_criterion.beta = beta
        self.distillation_criterion.gamma = gamma
        
        # Log weight changes
        if hasattr(self, 'log'):
            self.log_with_wandb('curriculum/alpha', alpha, on_step=False, on_epoch=True)
            self.log_with_wandb('curriculum/beta', beta, on_step=False, on_epoch=True)
            self.log_with_wandb('curriculum/gamma', gamma, on_step=False, on_epoch=True)
    def forward(self, x, device): 
        x_student = self.preprocess_sample(x, device=self.device)
        # Student uses real preprocessing (convert complex to real)
        x_student = self._convert_complex_to_real(x_student)
        student_output_real = self.student_model(x_student)
        student_output = self._convert_real_to_complex(student_output_real)
        return student_output
    
    def training_step(self, batch, batch_idx):
        """Training step with knowledge distillation (standard or progressive)"""
        # Update curriculum weights for standard KD
        if not self.progressive_layers:
            self._update_curriculum_weights(self.current_epoch)
        
        x, y = batch
        
        # Teacher uses complex preprocessing (original input)
        x_teacher = self.preprocess_sample(x, device=self.device)
        
        # Student uses real preprocessing (convert complex to real)
        x_student = self._convert_complex_to_real(x_teacher)

        if self.progressive_layers:
            # Progressive layer coupling approach
            # Student and teacher forward passes with different inputs
            student_output_real, student_features_list = self.extract_features(self.student_model, x_student, is_student=True)
            
            # Teacher forward pass (no gradients) with feature extraction using complex input
            with torch.no_grad():
                teacher_output, teacher_features_list = self.extract_features(self.teacher_model, x_teacher, is_student=False)
            
            # Convert student real output to complex for comparison
            student_output = self._convert_real_to_complex(student_output_real)
            
            y, student_output = self.preprocess_output_and_prediction_before_comparison(y, student_output)
            y, teacher_output = self.preprocess_output_and_prediction_before_comparison(y, teacher_output)
            student_features_list, teacher_features_list = zip(*[self.preprocess_output_and_prediction_before_comparison(student_feat, teacher_feat) for student_feat, teacher_feat in zip(student_features_list, teacher_features_list)]) if student_features_list is not None else (None, None)

            # Compute progressive loss with models and input
            loss_dict = self.distillation_criterion(
                student_output=student_output,
                student_features_list=student_features_list,
                teacher_output=teacher_output,
                teacher_features_list=teacher_features_list,
                ground_truth=y,
                student_model=self.student_model,
                teacher_model=self.teacher_model,
                input_tensor=x_teacher  # Use teacher input for progressive coupling
            )
            
            # Log progressive-specific metrics
            self.log_with_wandb('progressive/stage', loss_dict['stage'], on_step=False, on_epoch=True)
            self.log_with_wandb('progressive/num_layer_pairs', loss_dict['num_layer_pairs'], on_step=False, on_epoch=True)
            self.log_with_wandb('progressive/adaptive_gamma', loss_dict['adaptive_gamma'], on_step=False, on_epoch=True)
            
        else:
            # Standard knowledge distillation approach
            # Student forward pass with feature extraction using real input
            student_output_real, student_features_list = self.extract_features(self.student_model, x_student, is_student=True)
            
            # Teacher forward pass (no gradients) with feature extraction using complex input
            with torch.no_grad():
                teacher_output, teacher_features_list = self.extract_features(self.teacher_model, x_teacher, is_student=False)
            
            # Convert student real output to complex for comparison
            student_output = self._convert_real_to_complex(student_output_real)
            
            y, student_output = self.preprocess_output_and_prediction_before_comparison(y, student_output)
            y, teacher_output = self.preprocess_output_and_prediction_before_comparison(y, teacher_output)

            # Compute progressive feature coupling loss
            if student_features_list is not None and teacher_features_list is not None:
                # Apply progressive feature coupling with list of features
                feature_loss = self._compute_progressive_feature_coupling_loss(
                    teacher_features_list, student_features_list, self.current_epoch
                )
            else:
                feature_loss = torch.tensor(0.0, device=student_output.device)
            
            # Compute standard distillation loss (without features)
            loss_dict = self.distillation_criterion(
                student_output=student_output,
                student_features_list=student_features_list,
                teacher_output=teacher_output,
                teacher_features_list=teacher_features_list,
                ground_truth=y,
                student_features=None,  # We handle features separately now
                teacher_features=None
            )
            
            # Replace the feature loss with progressive coupling loss
            loss_dict['feature_loss'] = feature_loss
            # Recompute total loss
            loss_dict['total_loss'] = (
                self.distillation_criterion.alpha * loss_dict['student_loss'] +
                self.distillation_criterion.beta * loss_dict['distillation_loss'] +
                self.distillation_criterion.gamma * feature_loss
            )
        
        # Log all loss components
        for loss_name, loss_value in loss_dict.items():
            if isinstance(loss_value, torch.Tensor):
                self.log_with_wandb(f'train_{loss_name}', loss_value, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log progressive feature coupling metrics
        if student_features_list is not None and teacher_features_list is not None:
            self.log_with_wandb('progressive/num_teacher_layers', len(teacher_features_list), on_step=False, on_epoch=True)
            self.log_with_wandb('progressive/num_student_layers', len(student_features_list), on_step=False, on_epoch=True)
            self.log_with_wandb('progressive/epoch_progress', min(self.current_epoch / max(getattr(self.hparams, 'num_epochs', 100) - 1, 1), 1.0), on_step=False, on_epoch=True)
        
        # Track distribution statistics (if enabled)
        if self.distribution_tracker is not None:
            if self.progressive_layers:
                # For progressive layers, we need to extract outputs separately using proper preprocessing
                student_output_real_for_tracking = self.student_model(x_student)
                with torch.no_grad():
                    teacher_output_for_tracking = self.teacher_model(x_teacher)
                
                # Convert student real output to complex for tracking
                student_output_for_tracking = self._convert_real_to_complex(student_output_real_for_tracking)
                
                y, student_output_for_tracking = self.preprocess_output_and_prediction_before_comparison(y, student_output_for_tracking)
                y, teacher_output_for_tracking = self.preprocess_output_and_prediction_before_comparison(y, teacher_output_for_tracking)
                self.distribution_tracker.update(student_output_for_tracking, teacher_output_for_tracking, y)
            else:
                # For standard and distribution-preserving KD, outputs are already available
                self.distribution_tracker.update(student_output, teacher_output, y)
        
        # Log gradients to wandb after backward pass
        if hasattr(self, 'training_step_count'):
            self.training_step_count += 1
            # Note: Gradients will be logged in on_after_backward_step
        
        return loss_dict['total_loss']
    
    def validation_step(self, batch, batch_idx):
        """Validation step with knowledge distillation"""
        x, y = batch
        
        # Teacher uses complex preprocessing (original input)
        x_teacher = self.preprocess_sample(x, device=self.device)
        
        # Student uses real preprocessing (convert complex to real)
        x_student = self._convert_complex_to_real(x_teacher)
        
        # Student forward pass with feature extraction using real input
        student_output_real, student_features_list = self.extract_features(self.student_model, x_student, is_student=True)
        # Teacher forward pass (no gradients) with feature extraction using complex input
        with torch.no_grad():
            teacher_output, teacher_features_list = self.extract_features(self.teacher_model, x_teacher, is_student=False)
        
        # Convert student real output to complex for comparison
        student_output = self._convert_real_to_complex(student_output_real)
        # print(f"Teacher output dtype: {teacher_output.dtype}, shape: {teacher_output.shape}")
        # print(f"Ground truth dtype: {y.dtype}, shape: {y.shape}")
        
        # Apply sophisticated feature alignment before computing distillation loss
        if self.progressive_layers:
            # Progressive layer coupling approach
            # Student and teacher forward passes
            # student_output = self.student_model(x)
            # with torch.no_grad():
            #     teacher_output = self.teacher_model(x)
            y, student_output = self.preprocess_output_and_prediction_before_comparison(y, student_output)
            y, teacher_output = self.preprocess_output_and_prediction_before_comparison(y, teacher_output)
            student_features_list, teacher_features_list = zip(*[self.preprocess_output_and_prediction_before_comparison(student_feat, teacher_feat) for student_feat, teacher_feat in zip(student_features_list, teacher_features_list)]) if student_features_list is not None else (None, None)

            # Compute progressive loss with models and input
            loss_dict = self.distillation_criterion(
                student_output=student_output,
                student_features_list=student_features_list,
                teacher_output=teacher_output,
                teacher_features_list=teacher_features_list,
                ground_truth=y,
                student_model=self.student_model,
                teacher_model=self.teacher_model,
                input_tensor=x
            )
            
            # Log progressive-specific metrics
            self.log_with_wandb('progressive/stage', loss_dict['stage'], on_step=False, on_epoch=True)
            self.log_with_wandb('progressive/num_layer_pairs', loss_dict['num_layer_pairs'], on_step=False, on_epoch=True)
            self.log_with_wandb('progressive/adaptive_gamma', loss_dict['adaptive_gamma'], on_step=False, on_epoch=True)
            
        else:
            # Standard knowledge distillation approach
            # Student forward pass with feature extraction
            # student_output, student_features = self.extract_features(self.student_model, x)
            
            # # Teacher forward pass (no gradients) with feature extraction
            # with torch.no_grad():
            #     teacher_output, teacher_features = self.extract_features(self.teacher_model, x)
            
            # Apply sophisticated feature alignment before computing distillation loss
            if student_features_list is not None and teacher_features_list is not None:
                # Apply progressive feature coupling with list of features for validation
                feature_loss = self._compute_progressive_feature_coupling_loss(
                    teacher_features_list, student_features_list, self.current_epoch
                )
            else:
                feature_loss = torch.tensor(0.0, device=student_output.device)
            
            y, student_output = self.preprocess_output_and_prediction_before_comparison(y, student_output)
            y, teacher_output = self.preprocess_output_and_prediction_before_comparison(y, teacher_output)
            student_features_list, teacher_features_list = zip(*[self.preprocess_output_and_prediction_before_comparison(student_feat, teacher_feat) for student_feat, teacher_feat in zip(student_features_list, teacher_features_list)]) if student_features_list is not None else (None, None)


            # Compute distillation loss with aligned features
            loss_dict = self.distillation_criterion(
                student_output=student_output,
                student_features_list=student_features_list,
                teacher_output=teacher_output,
                teacher_features_list=teacher_features_list,
                ground_truth=y,
                student_features=None,  # We handle features separately now
                teacher_features=None
            )
            
            # Replace the feature loss with progressive coupling loss
            loss_dict['feature_loss'] = feature_loss
            # Recompute total loss
            loss_dict['total_loss'] = (
                self.distillation_criterion.alpha * loss_dict['student_loss'] +
                self.distillation_criterion.beta * loss_dict['distillation_loss'] +
                self.distillation_criterion.gamma * feature_loss
            )
        
        # Log validation losses
        for loss_name, loss_value in loss_dict.items():
            self.log_with_wandb(f'val_{loss_name}', loss_value, on_step=False, on_epoch=True, prog_bar=True)
            
        # Track distribution statistics for validation (if enabled)
        if self.distribution_tracker is not None:
            self.distribution_tracker.update(student_output, teacher_output, y)
            
        return loss_dict['total_loss']
    
    def test_step(self, batch, batch_idx):
        """Test step comparing student vs teacher performance"""
        x, y = batch
        
        # Teacher uses complex preprocessing (original input)
        x_teacher = self.preprocess_sample(x, device=self.device)
        
        # Student uses real preprocessing (convert complex to real)
        x_student = self._convert_complex_to_real(x_teacher)
        
        # Student predictions with feature extraction using real input
        student_output_real, student_features_list = self.extract_features(self.student_model, x_student, is_student=True)
        
        # Teacher predictions (for comparison) with feature extraction using complex input
        with torch.no_grad():
            teacher_output, teacher_features_list = self.extract_features(self.teacher_model, x_teacher, is_student=False)
        
        # Convert student real output to complex for comparison
        student_output = self._convert_real_to_complex(student_output_real)
        complexMSE = get_loss_function("complex_mse")()
        # Compute individual losses
        student_mse = complexMSE(student_output, y)
        teacher_mse = complexMSE(teacher_output, y)
        student_teacher_mse = complexMSE(student_output, teacher_output)

        # Log test metrics
        self.log('test_student_mse', student_mse, on_step=False, on_epoch=True)
        self.log('test_teacher_mse', teacher_mse, on_step=False, on_epoch=True)
        self.log('test_student_teacher_mse', student_teacher_mse, on_step=False, on_epoch=True)
        
        return {
            'student_mse': student_mse,
            'teacher_mse': teacher_mse,
            'student_teacher_mse': student_teacher_mse
        }
    
    def configure_optimizers(self):
        """Configure optimizer with adaptive learning rate for knowledge distillation"""
        # Use a lower initial learning rate during curriculum phase
        base_lr = self.hparams.lr
        
        # Start with lower learning rate during curriculum phase
        if self.curriculum_learning:
            initial_lr = base_lr * 0.5  # Start with half the learning rate
        else:
            initial_lr = base_lr
            
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=initial_lr,
            weight_decay=getattr(self.hparams, 'weight_decay', 1e-4)
        )
        
        if hasattr(self.hparams, 'scheduler_type') and self.hparams.scheduler_type == 'cosine':
            # Learning rate scheduler with warmup
            total_epochs = getattr(self.hparams, 'num_epochs', 100)
            warmup_epochs = min(5, self.curriculum_epochs // 2) if self.curriculum_learning else 0
            
            if warmup_epochs > 0:
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[
                        # Warmup phase
                        torch.optim.lr_scheduler.LinearLR(
                            optimizer,
                            start_factor=0.1,
                            end_factor=1.0,
                            total_iters=warmup_epochs
                        ),
                        # Main training phase
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=total_epochs - warmup_epochs,
                            eta_min=base_lr * 0.01
                        )
                    ],
                    milestones=[warmup_epochs]
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_epochs,
                    eta_min=base_lr * 0.01
                )
                
            return [optimizer], [scheduler]
        
        return optimizer
    
    def on_after_backward(self):
        """Hook called after backward pass to log gradients"""
        if hasattr(self, 'gradient_tracker') and hasattr(self, 'training_step_count'):
            # Log student model gradients
            self.gradient_tracker.log_gradients(
                self.student_model, 
                "student", 
                self.training_step_count
            )
            
            # Log teacher model gradients (if not frozen)
            if any(p.requires_grad for p in self.teacher_model.parameters()):
                self.gradient_tracker.log_gradients(
                    self.teacher_model,
                    "teacher", 
                    self.training_step_count
                )
    
    def on_train_epoch_end(self):
        """Hook called at the end of each training epoch"""
        if hasattr(self, 'gradient_tracker') and hasattr(self, 'training_step_count'):
            # Log weight statistics less frequently
            self.gradient_tracker.log_model_weights(
                self.student_model,
                "student",
                self.training_step_count,
                frequency=1  # Log weights every epoch
            )
            
            # Log teacher weights for comparison
            self.gradient_tracker.log_model_weights(
                self.teacher_model,
                "teacher",
                self.training_step_count,
                frequency=1
            )
    
    def on_train_end(self):
        """Called when training ends"""
        super().on_train_end() if hasattr(super(), 'on_train_end') else None
        
        # Close wandb if it was initialized
        if WANDB_AVAILABLE:
            try:
                wandb.finish()
                print("✅ Wandb logging finished")
            except Exception as e:
                print(f"Warning: Failed to finish wandb: {e}")
