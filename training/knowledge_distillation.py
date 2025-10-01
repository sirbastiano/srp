"""
Knowledge Distillation Pipeline for sarSSM Models

This module implements knowledge distillation to transfer knowledge from a larger,
complex teacher model to a smaller, simpler student model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, Dict, Any, Tuple
import os
from pathlib import Path

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
        temperature: float = 4.0,
        alpha: float = 0.5,
        beta: float = 0.3,
        feature_matching: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for student loss
        self.beta = beta    # Weight for distillation loss
        self.gamma = 1.0 - alpha - beta  # Weight for feature matching
        self.feature_matching = feature_matching
        self.mse_loss = nn.MSELoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
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
        student_loss = self.mse_loss(student_output, ground_truth)
        
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
        
        if torch.is_complex(student_output):
            # For complex data, use magnitude-based distillation
            teacher_mag = torch.abs(teacher_output_adapted)
            student_mag = torch.abs(student_output)
            distillation_loss = self.mse_loss(student_mag, teacher_mag)
            
            # Also distill phase information
            teacher_phase = torch.angle(teacher_output_adapted)
            student_phase = torch.angle(student_output)
            phase_loss = self.mse_loss(student_phase, teacher_phase)
            distillation_loss = distillation_loss + 0.5 * phase_loss
        else:
            # For real-valued data
            distillation_loss = self.mse_loss(student_output, teacher_output_adapted)
        
        # Feature matching loss (optional)
        feature_loss = torch.tensor(0.0, device=student_output.device)
        if self.feature_matching and student_features is not None and teacher_features is not None:
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
                    
                    feature_loss = self.mse_loss(student_features, teacher_feat_adapted)
            else:
                feature_loss = self.mse_loss(student_features, teacher_features)
        
        # Combined loss
        total_loss = (
            self.alpha * student_loss +
            self.beta * distillation_loss +
            self.gamma * feature_loss
        )
        
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
        temperature: float = 4.0,
        alpha: float = 0.5,
        beta: float = 0.3,
        feature_matching: bool = True,
        freeze_teacher: bool = True,
        lr: float = 1e-4,
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
            **kwargs
        )
        
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.teacher_checkpoint_path = teacher_checkpoint_path
        
        # Load teacher model checkpoint
        self._load_teacher_checkpoint()
        
        if freeze_teacher:
            # Freeze teacher model parameters
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
        
        # Initialize distillation loss
        self.distillation_criterion = KnowledgeDistillationLoss(
            temperature=temperature,
            alpha=alpha,
            beta=beta,
            feature_matching=feature_matching
        )
        
        # Initialize gradient tracking
        self.gradient_tracker = GradientTracker(log_frequency=50)
        self.global_step = 0
        
        self.save_hyperparameters(ignore=['student_model', 'teacher_model'])
        
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
                    new_state_dict[new_key] = value
                self.teacher_model.load_state_dict(new_state_dict)
            else:
                # Direct state dict
                self.teacher_model.load_state_dict(checkpoint)
                
            print(f"Loaded teacher model from {self.teacher_checkpoint_path}")
        else:
            raise FileNotFoundError(f"Teacher checkpoint not found: {self.teacher_checkpoint_path}")
    
    def extract_features(self, model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
            if hasattr(model, 'layers') and len(model.layers) > 0:
                # Hook into intermediate layer to extract features
                features = []
                
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        features.append(output[0])  # S4D returns (output, state)
                    else:
                        features.append(output)
                
                # Register hook on middle layer
                middle_layer_idx = len(model.layers) // 2
                hook = model.layers[middle_layer_idx].register_forward_hook(hook_fn)
                
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
    
    def training_step(self, batch, batch_idx):
        """Training step with knowledge distillation"""
        x, y = batch
        
        # Student forward pass
        student_output = self.student_model(x)
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_output = self.teacher_model(x)
        
        # Extract features for feature matching (optional)
        student_features = None
        teacher_features = None
        
        # Compute distillation loss
        loss_dict = self.distillation_criterion(
            student_output=student_output,
            teacher_output=teacher_output,
            ground_truth=y,
            student_features=student_features,
            teacher_features=teacher_features
        )
        
        # Log all loss components
        for loss_name, loss_value in loss_dict.items():
            self.log(f'train_{loss_name}', loss_value, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log gradients to wandb after backward pass
        if hasattr(self, 'global_step'):
            self.global_step += 1
            # Note: Gradients will be logged in on_after_backward_step
        
        return loss_dict['total_loss']
    
    def validation_step(self, batch, batch_idx):
        """Validation step with knowledge distillation"""
        x, y = batch
        
        # Student forward pass
        student_output = self.student_model(x)
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_output = self.teacher_model(x)
        
        # Compute distillation loss
        loss_dict = self.distillation_criterion(
            student_output=student_output,
            teacher_output=teacher_output,
            ground_truth=y
        )
        
        # Log validation losses
        for loss_name, loss_value in loss_dict.items():
            self.log(f'val_{loss_name}', loss_value, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss_dict['total_loss']
    
    def test_step(self, batch, batch_idx):
        """Test step comparing student vs teacher performance"""
        x, y = batch
        
        # Student predictions
        student_output = self.student_model(x)
        
        # Teacher predictions (for comparison)
        with torch.no_grad():
            teacher_output = self.teacher_model(x)
        
        # Compute individual losses
        student_mse = F.mse_loss(student_output, y)
        teacher_mse = F.mse_loss(teacher_output, y)
        student_teacher_mse = F.mse_loss(student_output, teacher_output)
        
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
        """Configure optimizer only for student model parameters"""
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.hparams.lr,
            weight_decay=getattr(self.hparams, 'weight_decay', 1e-4)
        )
        
        if hasattr(self.hparams, 'scheduler_type') and self.hparams.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=getattr(self.hparams, 'num_epochs', 100)
            )
            return [optimizer], [scheduler]
        
        return optimizer
    
    def on_after_backward(self):
        """Hook called after backward pass to log gradients"""
        if hasattr(self, 'gradient_tracker') and hasattr(self, 'global_step'):
            # Log student model gradients
            self.gradient_tracker.log_gradients(
                self.student_model, 
                "student", 
                self.global_step
            )
            
            # Log teacher model gradients (if not frozen)
            if any(p.requires_grad for p in self.teacher_model.parameters()):
                self.gradient_tracker.log_gradients(
                    self.teacher_model,
                    "teacher", 
                    self.global_step
                )
    
    def on_train_epoch_end(self):
        """Hook called at the end of each training epoch"""
        if hasattr(self, 'gradient_tracker') and hasattr(self, 'global_step'):
            # Log weight statistics less frequently
            self.gradient_tracker.log_model_weights(
                self.student_model,
                "student",
                self.global_step,
                frequency=1  # Log weights every epoch
            )
            
            # Log teacher weights for comparison
            self.gradient_tracker.log_model_weights(
                self.teacher_model,
                "teacher",
                self.global_step,
                frequency=1
            )
    
    def on_after_backward(self):
        """Hook called after backward pass to log gradients"""
        if hasattr(self, 'gradient_tracker') and hasattr(self, 'global_step'):
            # Log student model gradients
            self.gradient_tracker.log_gradients(
                self.student_model, 
                "student", 
                self.global_step
            )
            
            # Log teacher model gradients (if not frozen)
            if any(p.requires_grad for p in self.teacher_model.parameters()):
                self.gradient_tracker.log_gradients(
                    self.teacher_model,
                    "teacher", 
                    self.global_step
                )
    
    def on_train_epoch_end(self):
        """Hook called at the end of each training epoch"""
        if hasattr(self, 'gradient_tracker') and hasattr(self, 'global_step'):
            # Log weight statistics less frequently
            self.gradient_tracker.log_model_weights(
                self.student_model,
                "student",
                self.global_step,
                frequency=1  # Log weights every epoch
            )
            
            # Log teacher weights for comparison
            self.gradient_tracker.log_model_weights(
                self.teacher_model,
                "teacher",
                self.global_step,
                frequency=1
            )


def create_distillation_config(
    teacher_config: Dict[str, Any],
    student_modifications: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create student model configuration based on teacher config with modifications
    
    Args:
        teacher_config: Original teacher model configuration
        student_modifications: Modifications to apply for student model
        
    Returns:
        Student model configuration
    """
    import copy
    student_config = copy.deepcopy(teacher_config)
    
    # Apply modifications
    for key, value in student_modifications.items():
        if key in student_config:
            student_config[key] = value
        else:
            student_config[key] = value
    
    return student_config


def setup_knowledge_distillation(
    teacher_config_path: str,
    teacher_checkpoint_path: str,
    student_modifications: Dict[str, Any],
    dataloader_config: Dict[str, Any],
    training_config: Dict[str, Any],
    save_dir: str = "./results/distillation"
) -> Tuple[KnowledgeDistillationTrainer, pl.Trainer]:
    """
    Setup complete knowledge distillation pipeline
    
    Args:
        teacher_config_path: Path to teacher model configuration
        teacher_checkpoint_path: Path to teacher model checkpoint
        student_modifications: Modifications for student model
        dataloader_config: Dataloader configuration
        training_config: Training configuration
        save_dir: Directory to save results
        
    Returns:
        Tuple of (distillation_trainer, pytorch_lightning_trainer)
    """
    import yaml
    from training.training_script import create_dataloaders
    
    # Load teacher configuration
    with open(teacher_config_path, 'r') as f:
        teacher_full_config = yaml.safe_load(f)
    
    teacher_model_config = teacher_full_config['model']
    
    # Create student configuration
    student_model_config = create_distillation_config(
        teacher_model_config,
        student_modifications
    )
    
    print("Teacher Model Config:", teacher_model_config)
    print("Student Model Config:", student_model_config)
    
    # Create models
    teacher_model = get_model_from_configs(**teacher_model_config)
    student_model = get_model_from_configs(**student_model_config)
    
    print(f"Teacher model parameters: {sum(p.numel() for p in teacher_model.parameters()):,}")
    print(f"Student model parameters: {sum(p.numel() for p in student_model.parameters()):,}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, inference_loader = create_dataloaders(dataloader_config)
    
    # Create distillation trainer
    distillation_trainer = KnowledgeDistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        teacher_checkpoint_path=teacher_checkpoint_path,
        base_save_dir=save_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        inference_loader=inference_loader,
        **training_config
    )
    
    # Create PyTorch Lightning trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    
    # Initialize loggers
    loggers = []
    
    # Always use TensorBoard logger
    tb_logger = TensorBoardLogger(save_dir=save_dir, name="distillation")
    loggers.append(tb_logger)
    
    # Add wandb logger if available
    if WANDB_AVAILABLE:
        try:
            from pytorch_lightning.loggers import WandbLogger
            wandb_logger = WandbLogger(
                project="sarssm-knowledge-distillation",
                name=f"distillation_{student_model_config.get('model_dim', 'unknown')}dim_{student_model_config.get('num_layers', 'unknown')}layers",
                save_dir=save_dir,
                config={
                    "teacher_config": teacher_model_config,
                    "student_config": student_model_config,
                    "training_config": training_config,
                    "dataloader_config": dataloader_config
                }
            )
            loggers.append(wandb_logger)
            print("✅ Wandb logger initialized for gradient tracking")
        except Exception as e:
            print(f"⚠️  Wandb logger initialization failed: {e}")
    else:
        print("⚠️  Wandb not available - using TensorBoard only")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_dir, "checkpoints"),
        filename="student-{epoch:02d}-{val_total_loss:.4f}",
        monitor="val_total_loss",
        mode="min",
        save_top_k=3,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor="val_total_loss",
        patience=training_config.get('patience', 20),
        mode="min",
        min_delta=training_config.get('delta', 1e-4)
    )
    
    trainer = pl.Trainer(
        max_epochs=training_config.get('num_epochs', 100),
        logger=loggers,  # Use all available loggers
        callbacks=[checkpoint_callback, early_stopping],
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=16 if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        log_every_n_steps=50
    )
    
    return distillation_trainer, trainer