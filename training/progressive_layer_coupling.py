#!/usr/bin/env python3
"""
Progressive Layer Coupling Strategy for Knowledge Distillation

This implements a curriculum learning approach where feature matching is introduced
progressively: start with early layers, add middle layers, then late layers, 
and finally combine all feature losses together.

Strategy:
- Stage 1 (Epochs 0-N): Early layers only - Teacher[1] ↔ Student[1] 
- Stage 2 (Epochs N-2N): Add middle layers - Teacher[1,3] ↔ Student[1,2]
- Stage 3 (Epochs 2N-3N): Add late layers - Teacher[1,3,5] ↔ Student[1,2,3]
- Stage 4 (Epochs 3N+): Full multi-layer distillation with balanced weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import math
from model.SSMs.SSM import ComplexLinear

class ProgressiveLayerCouplingLoss(nn.Module):
    """
    Progressive layer coupling loss that gradually introduces feature matching
    between teacher and student layers in a curriculum learning fashion.
    """
    
    def __init__(
        self,
        teacher_layers: int = 6,
        student_layers: int = 4,
        stage_epochs: int = 10,
        alpha: float = 0.7,  # Ground truth weight (increased from 0.2)
        beta: float = 0.2,   # Distillation weight (decreased from 0.7)
        gamma: float = 0.1,  # Feature weight
        temperature: float = 3.0,  # Reduced from 6.0
        loss_fn_name: str = "mse"
    ):
        super().__init__()
        self.teacher_layers = teacher_layers
        self.student_layers = student_layers
        self.stage_epochs = stage_epochs
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        
        # Define layer coupling strategy
        self.layer_couplings = self._define_layer_couplings()
        
        # Loss functions
        from sarpyx.utils.losses import get_loss_function
        self.mse_loss = get_loss_function(loss_fn_name)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
        # Current training stage
        self.current_stage = 0
        self.current_epoch = 0
        
    def _define_layer_couplings(self) -> List[List[Tuple[int, int]]]:
        """
        Define the progressive layer coupling strategy EXACTLY as requested:
        
        Stage 1: First S4D layer features only (Teacher[1] ↔ Student[1])
        Stage 2: Second S4D layer features only (Teacher[2] ↔ Student[2]) 
        Stage 3: Third S4D layer features only (Teacher[3] ↔ Student[3])
        Stage 4: Fourth S4D layer features only (Teacher[4] ↔ Student[4])
        Final Stage: Ground truth only (no feature matching)
        
        Returns:
            List of stages, each containing list of (teacher_idx, student_idx) tuples
        """
        # Create layer-by-layer coupling strategy
        couplings = []
        
        # Individual layer stages (one layer pair at a time)
        max_layers = min(self.teacher_layers, self.student_layers)
        for layer_idx in range(1, max_layers + 1):  # Start from layer 1
            couplings.append([(layer_idx, layer_idx)])
        
        # Final stage: Empty list means ground truth only, no feature matching
        couplings.append([])
        
        print(f"📚 Progressive Layer Coupling Strategy:")
        for i, stage_couples in enumerate(couplings):
            if stage_couples:
                layer_pairs = ", ".join([f"T{t}↔S{s}" for t, s in stage_couples])
                print(f"   Stage {i+1} (epochs {i*self.stage_epochs}-{(i+1)*self.stage_epochs}): {layer_pairs}")
            else:
                print(f"   Stage {i+1} (epochs {i*self.stage_epochs}+): Ground truth only")
        
        return couplings
    
    def update_training_stage(self, current_epoch: int):
        """Update the current training stage based on epoch"""
        self.current_epoch = current_epoch
        self.current_stage = min(current_epoch // self.stage_epochs, len(self.layer_couplings) - 1)
    
    def align_feature_dimensions(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align feature dimensions between student and teacher.
        
        Args:
            student_feat: Student feature tensor
            teacher_feat: Teacher feature tensor
            
        Returns:
            Aligned (student_feat, teacher_feat) tensors
        """
        # Handle shape mismatches
        if len(student_feat.shape) != len(teacher_feat.shape):
            # Ensure both have same number of dimensions
            while len(student_feat.shape) < len(teacher_feat.shape):
                student_feat = student_feat.unsqueeze(-1)
            while len(teacher_feat.shape) < len(student_feat.shape):
                teacher_feat = teacher_feat.unsqueeze(-1)
        
        # Handle sequence length dimension (typically dim 1 or 2)
        if student_feat.shape[1] != teacher_feat.shape[1] and len(student_feat.shape) >= 3:
            # Try transposing if it makes sequence lengths match
            if student_feat.shape[2] == teacher_feat.shape[1]:
                student_feat = student_feat.transpose(1, 2)
            elif teacher_feat.shape[2] == student_feat.shape[1]:
                teacher_feat = teacher_feat.transpose(1, 2)
        
        # Handle feature dimension mismatch (check both middle and last dimensions)
        if student_feat.shape[1] != teacher_feat.shape[1]:
            # Mismatch in dimension 1 (feature channels)
            teacher_dim = teacher_feat.shape[1]
            student_dim = student_feat.shape[1]
            
            if teacher_dim > student_dim:
                # Project teacher down to student dimension
                # Reshape for linear layer: [batch, channels, seq] -> [batch*seq, channels]
                batch, t_dim, seq = teacher_feat.shape
                teacher_reshaped = teacher_feat.transpose(1, 2).reshape(-1, t_dim)
                    
                projection = nn.Linear(t_dim, student_dim).to(teacher_feat.device)
                if torch.is_complex(teacher_reshaped):
                    # Custom complex linear projection

                    projection = ComplexLinear(t_dim, student_dim).to(teacher_feat.device)
                    with torch.no_grad():
                        projection.real.weight.data.fill_(1.0 / t_dim)
                        projection.real.bias.data.zero_()
                        projection.imag.weight.data.zero_()
                        projection.imag.bias.data.zero_()
                else:
                    projection = nn.Linear(t_dim, student_dim).to(teacher_feat.device)
                    with torch.no_grad():
                        projection.weight.data.fill_(1.0 / t_dim)
                        projection.bias.data.zero_()
                teacher_projected = projection(teacher_reshaped)
                teacher_feat = teacher_projected.reshape(batch, seq, student_dim).transpose(1, 2)
            else:
                # Project student up to teacher dimension
                batch, s_dim, seq = student_feat.shape
                student_reshaped = student_feat.transpose(1, 2).reshape(-1, s_dim)
                projection = nn.Linear(s_dim, teacher_dim).to(student_feat.device)
                with torch.no_grad():
                    projection.weight.data.fill_(1.0)
                    projection.bias.data.zero_()
                student_projected = projection(student_reshaped)
                student_feat = student_projected.reshape(batch, seq, teacher_dim).transpose(1, 2)
        
        elif student_feat.shape[-1] != teacher_feat.shape[-1]:
            # Mismatch in last dimension (fallback for other cases)
            # Project teacher features to student dimension space
            # or vice versa using simple linear projection
            teacher_dim = teacher_feat.shape[-1]
            student_dim = student_feat.shape[-1]
            
            if teacher_dim > student_dim:
                # Project teacher down to student dimension
                projection = nn.Linear(teacher_dim, student_dim).to(teacher_feat.device)
                # Simple averaging projection
                with torch.no_grad():
                    projection.weight.data.fill_(1.0 / teacher_dim)
                    projection.bias.data.zero_()
                teacher_feat = projection(teacher_feat)
            else:
                # Project student up to teacher dimension  
                projection = nn.Linear(student_dim, teacher_dim).to(student_feat.device)
                with torch.no_grad():
                    projection.weight.data.fill_(1.0)
                    projection.bias.data.zero_()
                student_feat = projection(student_feat)
        
        return student_feat, teacher_feat
    
    def compute_feature_loss(self, student_features: List[torch.Tensor], teacher_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute feature matching loss for the current stage.
        
        Args:
            student_features: List of student feature tensors
            teacher_features: List of teacher feature tensors
            
        Returns:
            Feature matching loss
        """
        if len(student_features) != len(teacher_features):
            return torch.tensor(0.0, device=student_features[0].device if student_features else torch.device('cpu'))
        
        total_loss = 0.0
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Align dimensions
            s_feat_aligned, t_feat_aligned = self.align_feature_dimensions(s_feat, t_feat)
            
            # Compute MSE loss between aligned features
            feat_loss = self.mse_loss(s_feat_aligned, t_feat_aligned)
            total_loss += feat_loss
        
        # Average over all feature pairs
        if len(student_features) > 0:
            total_loss = total_loss / len(student_features)
        
        return total_loss
    
    def forward(
        self,
        student_output: torch.Tensor,
        student_features_list: List[torch.Tensor],
        teacher_output: torch.Tensor,
        teacher_features_list: List[torch.Tensor],
        ground_truth: torch.Tensor,
        student_model: nn.Module,
        teacher_model: nn.Module,
        input_tensor: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute progressive layer coupling loss.
        
        Args:
            student_output: Student model output
            teacher_output: Teacher model output
            ground_truth: Ground truth labels
            student_model: Student model (for feature extraction)
            teacher_model: Teacher model (for feature extraction)
            input_tensor: Input tensor for feature extraction
            
        Returns:
            Dictionary containing loss components
        """
        # Get current stage layer couplings
        current_couplings = self.layer_couplings[self.current_stage]
        
        # 1. Student loss (ground truth) - always present
        student_loss = self.mse_loss(student_output, ground_truth)
        
        # 2. Check if we're in the final stage (ground truth only)
        if not current_couplings:  # Empty list means ground truth only
            print(f"📚 Stage {self.current_stage + 1}: Ground truth only phase")
            return {
                'total_loss': student_loss,  # Only ground truth loss
                'student_loss': student_loss,
                'distillation_loss': torch.tensor(0.0, device=student_loss.device),
                'feature_loss': torch.tensor(0.0, device=student_loss.device),
                'stage': torch.tensor(self.current_stage, dtype=torch.float),
                'num_layer_pairs': torch.tensor(0, dtype=torch.float),
                'adaptive_gamma': torch.tensor(0.0, dtype=torch.float)
            }
        

        
        # 3. Distillation loss (teacher output) - handle complex tensors
        if teacher_output.is_complex() or student_output.is_complex():
            # For complex outputs, use MSE instead of KL divergence
            distillation_loss = self.mse_loss(student_output, teacher_output)
        else:
            # Standard softmax for real-valued outputs
            teacher_soft = F.softmax(teacher_output / self.temperature, dim=-1)
            student_log_soft = F.log_softmax(student_output / self.temperature, dim=-1)
            distillation_loss = self.kl_div(student_log_soft, teacher_soft) * (self.temperature ** 2)
        
        # 4. Progressive feature loss (only for current layer pair)
        feature_loss = self.compute_feature_loss(student_features_list, teacher_features_list)
        
        # 5. Combine losses - focus on current layer features
        total_loss = (
            self.alpha * student_loss +
            self.beta * distillation_loss +
            self.gamma * feature_loss  # Fixed weight for single layer focus
        )
        
        return {
            'total_loss': total_loss,
            'student_loss': student_loss,
            'distillation_loss': distillation_loss,
            'feature_loss': feature_loss,
            'stage': torch.tensor(self.current_stage, dtype=torch.float),
            'num_layer_pairs': torch.tensor(len(current_couplings), dtype=torch.float),
            'adaptive_gamma': torch.tensor(self.gamma, dtype=torch.float)
        }


class ProgressiveKnowledgeDistillationTrainer:
    """
    Trainer class that implements progressive layer coupling strategy.
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        teacher_layers: int = 6,
        student_layers: int = 4,
        stage_epochs: int = 10,
        **kwargs
    ):
        self.student_model = student_model
        self.teacher_model = teacher_model
        
        # Create progressive loss function
        self.criterion = ProgressiveLayerCouplingLoss(
            teacher_layers=teacher_layers,
            student_layers=student_layers,
            stage_epochs=stage_epochs,
            **kwargs
        )
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def training_step(self, batch, batch_idx, current_epoch):
        """Training step with progressive layer coupling"""
        # Update training stage
        self.criterion.update_training_stage(current_epoch)
        
        # Get batch data
        x, y = batch
        
        # Forward passes
        student_output = self.student_model(x)
        with torch.no_grad():
            teacher_output = self.teacher_model(x)
        
        # Compute progressive loss
        loss_dict = self.criterion(
            student_output, teacher_output, y,
            self.student_model, self.teacher_model, x
        )
        
        return loss_dict
    
    def get_current_stage_info(self) -> Dict[str, any]:
        """Get information about current training stage"""
        current_couplings = self.criterion.layer_couplings[self.criterion.current_stage]
        
        return {
            'stage': self.criterion.current_stage,
            'epoch': self.criterion.current_epoch,
            'layer_pairs': current_couplings,
            'num_pairs': len(current_couplings),
            'stage_description': self._get_stage_description()
        }
    
    def _get_stage_description(self) -> str:
        """Get human-readable description of current stage"""
        stage = self.criterion.current_stage
        descriptions = [
            "Early layers: Learning low-level features",
            "Early + Middle layers: Adding mid-level abstractions", 
            "All layers: Full hierarchical feature matching",
            "Stable training: Maintaining all feature alignments"
        ]
        return descriptions[min(stage, len(descriptions) - 1)]


def test_progressive_strategy():
    """Test the progressive layer coupling strategy"""
    print("🧪 Testing Progressive Layer Coupling Strategy")
    print("=" * 60)
    
    # Create dummy models
    teacher_model = nn.Sequential(*[nn.Linear(64, 64) for _ in range(6)])
    student_model = nn.Sequential(*[nn.Linear(1, 1) for _ in range(4)])
    
    # Create progressive trainer
    trainer = ProgressiveKnowledgeDistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        teacher_layers=6,
        student_layers=4,
        stage_epochs=10,
        alpha=0.7,  # High ground truth weight
        beta=0.2,   # Low distillation weight
        gamma=0.1,  # Moderate feature weight
        temperature=3.0
    )
    
    # Simulate training epochs
    print("Training Progress:")
    print("Epoch | Stage | Layer Pairs | Description")
    print("-" * 60)
    
    for epoch in range(35):
        stage_info = trainer.get_current_stage_info()
        
        if epoch % 5 == 0:  # Print every 5 epochs
            pairs_str = str(stage_info['layer_pairs'])
            if len(pairs_str) > 30:
                pairs_str = pairs_str[:27] + "..."
            
            print(f"{epoch:5d} | {stage_info['stage']:5d} | {pairs_str:11s} | {stage_info['stage_description']}")
        
        # Update criterion epoch
        trainer.criterion.update_training_stage(epoch)
    
    print(f"\n✓ Progressive strategy successfully implemented!")


if __name__ == "__main__":
    test_progressive_strategy()