#!/usr/bin/env python3
"""
Updated Progressive Layer Coupling with Learnable Projections

This enhanced version addresses the critical dimension mismatch problem by:
1. Supporting learnable projection layers instead of simple averaging
2. Implementing gradual dimension reduction strategies
3. Adding intelligent feature selection mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import math

class LearnableProjection(nn.Module):
    """
    Learnable projection layer that can adapt to preserve most important features
    """
    
    def __init__(self, input_dim: int, output_dim: int, projection_type: str = "linear"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection_type = projection_type
        
        if projection_type == "linear":
            self.projection = nn.Linear(input_dim, output_dim)
        elif projection_type == "attention":
            # Attention-based projection
            self.attention = nn.MultiheadAttention(input_dim, num_heads=1, batch_first=True)
            self.projection = nn.Linear(input_dim, output_dim)
        elif projection_type == "pca_init":
            # Initialize with PCA-like weights
            self.projection = nn.Linear(input_dim, output_dim)
            self._init_pca_weights()
        
    def _init_pca_weights(self):
        """Initialize weights to approximate PCA projection"""
        # Initialize first few output dimensions as principal components
        with torch.no_grad():
            # Create a simple principal component approximation
            for i in range(min(self.output_dim, self.input_dim)):
                weight_vector = torch.zeros(self.input_dim)
                # Emphasize different frequency components for each output dimension
                start_idx = (i * self.input_dim) // self.output_dim
                end_idx = ((i + 1) * self.input_dim) // self.output_dim
                weight_vector[start_idx:end_idx] = 1.0 / (end_idx - start_idx)
                self.projection.weight.data[i] = weight_vector
            
            self.projection.bias.data.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.projection_type == "attention":
            # Use attention to select important features
            attn_out, _ = self.attention(x, x, x)
            return self.projection(attn_out)
        else:
            return self.projection(x)

class EnhancedProgressiveLayerCouplingLoss(nn.Module):
    """
    Enhanced progressive layer coupling with learnable projections
    """
    
    def __init__(
        self,
        teacher_layers: int = 6,
        student_layers: int = 4,
        teacher_dim: int = 64,
        student_dim: int = 16,  # INCREASED from 1
        stage_epochs: int = 15,
        alpha: float = 0.7,
        beta: float = 0.2, 
        gamma: float = 0.1,
        temperature: float = 3.0,
        loss_fn_name: str = "mse",
        projection_type: str = "pca_init"  # "linear", "attention", "pca_init"
    ):
        super().__init__()
        self.teacher_layers = teacher_layers
        self.student_layers = student_layers
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        self.stage_epochs = stage_epochs
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.projection_type = projection_type
        
        # Create learnable projection layers for each layer coupling
        self.projections = nn.ModuleDict()
        
        # Define layer couplings
        self.layer_couplings = self._define_layer_couplings()
        
        # Initialize projection layers for all possible couplings
        all_couplings = set()
        for stage_couplings in self.layer_couplings:
            all_couplings.update(stage_couplings)
        
        for teacher_idx, student_idx in all_couplings:
            proj_name = f"proj_t{teacher_idx}_s{student_idx}"
            self.projections[proj_name] = LearnableProjection(
                teacher_dim, student_dim, projection_type
            )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
        # Current training state
        self.current_stage = 0
        self.current_epoch = 0
        
        print(f"🚀 Enhanced Progressive Coupling initialized:")
        print(f"   Teacher: {teacher_layers} layers × {teacher_dim}D")
        print(f"   Student: {student_layers} layers × {student_dim}D")
        print(f"   Projection: {projection_type} ({teacher_dim}→{student_dim})")
        print(f"   Compression ratio: {teacher_dim/student_dim:.1f}:1")
        print(f"   Expected info retention: ~{(student_dim/teacher_dim)*100:.1f}%")
    
    def _define_layer_couplings(self) -> List[List[Tuple[int, int]]]:
        """Define progressive layer coupling strategy"""
        early_coupling = [(1, 1)]
        middle_coupling = [(1, 1), (3, 2)]
        late_coupling = [(1, 1), (3, 2), (5, 3)]
        
        return [early_coupling, middle_coupling, late_coupling, late_coupling]
    
    def update_training_stage(self, current_epoch: int):
        """Update training stage"""
        self.current_epoch = current_epoch
        self.current_stage = min(current_epoch // self.stage_epochs, len(self.layer_couplings) - 1)
    
    def extract_layer_features(self, model: nn.Module, x: torch.Tensor, layer_indices: List[int]) -> List[torch.Tensor]:
        """Extract features from specific layers"""
        features = []
        hooks = []
        
        def create_hook(layer_idx: int):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    features.append((layer_idx, output[0]))
                else:
                    features.append((layer_idx, output))
            return hook_fn
        
        if hasattr(model, 'layers'):
            for layer_idx in layer_indices:
                if 0 <= layer_idx < len(model.layers):
                    hook = model.layers[layer_idx].register_forward_hook(create_hook(layer_idx))
                    hooks.append(hook)
        
        try:
            with torch.no_grad():
                output = model(x)
            
            features.sort(key=lambda x: x[0])
            extracted_features = [feat[1] for feat in features]
            return extracted_features
        finally:
            for hook in hooks:
                hook.remove()
    
    def align_and_project_features(self, teacher_feat: torch.Tensor, student_feat: torch.Tensor, 
                                  teacher_idx: int, student_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align features using learnable projection
        """
        # Handle shape alignment (sequence length, etc.)
        if len(teacher_feat.shape) != len(student_feat.shape):
            while len(teacher_feat.shape) < len(student_feat.shape):
                teacher_feat = teacher_feat.unsqueeze(-1)
            while len(student_feat.shape) < len(teacher_feat.shape):
                student_feat = student_feat.unsqueeze(-1)
        
        # Handle sequence dimension mismatch
        if teacher_feat.shape[1] != student_feat.shape[1] and len(teacher_feat.shape) >= 3:
            if teacher_feat.shape[2] == student_feat.shape[1]:
                teacher_feat = teacher_feat.transpose(1, 2)
            elif student_feat.shape[2] == teacher_feat.shape[1]:
                student_feat = student_feat.transpose(1, 2)
        
        # Apply learnable projection
        proj_name = f"proj_t{teacher_idx}_s{student_idx}"
        if proj_name in self.projections:
            # Project teacher features to student dimension
            teacher_feat_projected = self.projections[proj_name](teacher_feat)
            return teacher_feat_projected, student_feat
        else:
            # Fallback to simple projection if needed
            if teacher_feat.shape[-1] != student_feat.shape[-1]:
                projection = nn.Linear(teacher_feat.shape[-1], student_feat.shape[-1]).to(teacher_feat.device)
                with torch.no_grad():
                    projection.weight.data.fill_(1.0 / teacher_feat.shape[-1])
                    projection.bias.data.zero_()
                teacher_feat = projection(teacher_feat)
            
            return teacher_feat, student_feat
    
    def compute_feature_loss(self, student_features: List[torch.Tensor], teacher_features: List[torch.Tensor],
                           current_couplings: List[Tuple[int, int]]) -> torch.Tensor:
        """Compute feature matching loss with learnable projections"""
        if len(student_features) != len(teacher_features):
            return torch.tensor(0.0, device=student_features[0].device if student_features else torch.device('cpu'))
        
        total_loss = 0.0
        for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
            teacher_idx, student_idx = current_couplings[i]
            
            # Apply learnable projection
            t_feat_aligned, s_feat_aligned = self.align_and_project_features(
                t_feat, s_feat, teacher_idx, student_idx
            )
            
            # Compute MSE loss
            feat_loss = self.mse_loss(s_feat_aligned, t_feat_aligned.detach())
            total_loss += feat_loss
        
        return total_loss / len(student_features) if len(student_features) > 0 else torch.tensor(0.0)
    
    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor, ground_truth: torch.Tensor,
               student_model: nn.Module, teacher_model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Enhanced forward pass with learnable projections"""
        
        # Get current stage couplings
        current_couplings = self.layer_couplings[self.current_stage]
        
        # Extract features
        teacher_indices = [t_idx for t_idx, s_idx in current_couplings]
        student_indices = [s_idx for t_idx, s_idx in current_couplings]
        
        student_features = self.extract_layer_features(student_model, input_tensor, student_indices)
        teacher_features = self.extract_layer_features(teacher_model, input_tensor, teacher_indices)
        
        # Compute losses
        student_loss = self.mse_loss(student_output, ground_truth)
        
        # Distillation loss - handle complex tensors
        if teacher_output.is_complex() or student_output.is_complex():
            # For complex outputs, use magnitude-based distillation
            teacher_mag = teacher_output.abs() if teacher_output.is_complex() else teacher_output
            student_mag = student_output.abs() if student_output.is_complex() else student_output
            
            # Apply softmax to magnitudes
            teacher_soft = F.softmax(teacher_mag / self.temperature, dim=-1)
            student_log_soft = F.log_softmax(student_mag / self.temperature, dim=-1)
            distillation_loss = self.kl_div(student_log_soft, teacher_soft) * (self.temperature ** 2)
        else:
            # Standard softmax for real-valued outputs
            teacher_soft = F.softmax(teacher_output / self.temperature, dim=-1)
            student_log_soft = F.log_softmax(student_output / self.temperature, dim=-1)
            distillation_loss = self.kl_div(student_log_soft, teacher_soft) * (self.temperature ** 2)
        
        # Enhanced feature loss with learnable projections
        feature_loss = self.compute_feature_loss(student_features, teacher_features, current_couplings)
        
        # Adaptive gamma based on stage
        stage_factor = (self.current_stage + 1) / len(self.layer_couplings)
        adaptive_gamma = self.gamma * stage_factor
        
        # Total loss
        total_loss = (
            self.alpha * student_loss +
            self.beta * distillation_loss +
            adaptive_gamma * feature_loss
        )
        
        return {
            'total_loss': total_loss,
            'student_loss': student_loss,
            'distillation_loss': distillation_loss,
            'feature_loss': feature_loss,
            'stage': torch.tensor(self.current_stage, dtype=torch.float),
            'num_layer_pairs': torch.tensor(len(current_couplings), dtype=torch.float),
            'adaptive_gamma': torch.tensor(adaptive_gamma, dtype=torch.float),
            'compression_ratio': torch.tensor(self.teacher_dim / self.student_dim, dtype=torch.float)
        }

def test_enhanced_progressive():
    """Test the enhanced progressive layer coupling"""
    print("🧪 Testing Enhanced Progressive Layer Coupling")
    print("=" * 70)
    
    # Test with different student dimensions
    test_configs = [
        {'student_dim': 1, 'name': 'Current (Critical)'},
        {'student_dim': 8, 'name': 'Improved (Fair)'},
        {'student_dim': 16, 'name': 'Recommended (Good)'},
        {'student_dim': 32, 'name': 'Optimal (Excellent)'},
    ]
    
    for config in test_configs:
        print(f"\n📊 Testing {config['name']} - Student Dim: {config['student_dim']}")
        print("-" * 50)
        
        loss_fn = EnhancedProgressiveLayerCouplingLoss(
            teacher_layers=6,
            student_layers=4,
            teacher_dim=64,
            student_dim=config['student_dim'],
            projection_type="pca_init"
        )
        
        # Calculate compression ratio and expected performance
        compression_ratio = 64 / config['student_dim']
        expected_retention = min(50, (config['student_dim'] / 64) * 100 + 5)
        
        print(f"   Compression ratio: {compression_ratio:.1f}:1")
        print(f"   Expected info retention: ~{expected_retention:.1f}%")
        print(f"   Learnable projections: {len(loss_fn.projections)} layers")

if __name__ == "__main__":
    test_enhanced_progressive()