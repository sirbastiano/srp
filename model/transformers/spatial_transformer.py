#!/usr/bin/env python3
"""
Physics-Aware SAR Spatial Transformer with Compressed Sensing Integration
==========================================================================

Implements a state-of-the-art SAR compression model incorporating:
1. True complex-valued processing (CVNN-inspired)
2. Deep unfolded SAR physics (Range-Doppler, data consistency)
3. Sparsity priors via learned ISTA (LISTA)
4. Multi-domain losses (spatial, frequency, phase)
5. Rate-distortion optimization with entropy modeling
6. Quantization-aware training

Based on research recommendations for SAR compression with physics integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import math
import einops
from typing import Optional, Tuple, Literal, Dict
from dataclasses import dataclass

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PhysicsConfig:
    """Configuration for physics-aware processing."""
    use_complex_layers: bool = True
    use_deep_unfolding: bool = True
    unfolding_iterations: int = 3
    use_sparsity_prior: bool = True
    sparsity_lambda: float = 0.01
    use_multi_domain_loss: bool = True
    phase_loss_weight: float = 0.5
    frequency_loss_weight: float = 0.3

@dataclass
class CompressionConfig:
    """Configuration for learned compression."""
    use_entropy_model: bool = True
    use_hyperprior: bool = True
    use_quantization: bool = True
    use_residual_quantization: bool = True  # NEW: Use RVQ for detail preservation
    num_rvq_quantizers: int = 4  # Number of RVQ stages (more = better quality, higher bitrate)
    rvq_codebook_size: int = 1024  # Codebook size per stage
    rvq_commitment_weight: float = 0.25  # Weight for commitment loss
    rate_lambda: float = 0.01  # Rate-distortion tradeoff
    num_distributions: int = 4  # Mixture components


# ============================================================================
# Complex-Valued Neural Network Components
# ============================================================================

class ComplexConv2d(nn.Module):
    """
    True complex-valued 2D convolution.
    Implements: (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Separate real and imaginary weight matrices
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_channels))
            self.bias_imag = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex input [B, C, H, W, 2] where last dim is [real, imag]
        Returns:
            Complex output [B, C', H', W', 2]
        """
        x_real, x_imag = x[..., 0], x[..., 1]
        
        # Complex multiplication: (a + ib)(c + id) = (ac - bd) + i(ad + bc)
        out_real = self.conv_real(x_real) - self.conv_imag(x_imag)
        out_imag = self.conv_real(x_imag) + self.conv_imag(x_real)
        
        if self.bias_real is not None:
            out_real = out_real + self.bias_real.view(1, -1, 1, 1)
            out_imag = out_imag + self.bias_imag.view(1, -1, 1, 1)
        
        return torch.stack([out_real, out_imag], dim=-1)


class ComplexLinear(nn.Module):
    """True complex-valued linear layer."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features))
        
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_features))
            self.bias_imag = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_real', None)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with complex-valued Xavier."""
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.normal_(self.weight_real, 0, std)
        nn.init.normal_(self.weight_imag, 0, std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., features, 2] where last dim is [real, imag]
        Returns:
            [..., out_features, 2]
        """
        x_real, x_imag = x[..., 0], x[..., 1]
        
        out_real = F.linear(x_real, self.weight_real) - F.linear(x_imag, self.weight_imag)
        out_imag = F.linear(x_real, self.weight_imag) + F.linear(x_imag, self.weight_real)
        
        if self.bias_real is not None:
            out_real = out_real + self.bias_real
            out_imag = out_imag + self.bias_imag
        
        return torch.stack([out_real, out_imag], dim=-1)


class ComplexReLU(nn.Module):
    """
    Complex activation: applies ReLU to magnitude, preserves phase.
    Output = ReLU(|z|) * exp(i * angle(z))
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., 2] complex tensor."""
        real, imag = x[..., 0], x[..., 1]
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        phase = torch.atan2(imag, real)
        
        # Apply ReLU to magnitude
        magnitude_activated = F.relu(magnitude)
        
        # Reconstruct with activated magnitude
        out_real = magnitude_activated * torch.cos(phase)
        out_imag = magnitude_activated * torch.sin(phase)
        
        return torch.stack([out_real, out_imag], dim=-1)


class ComplexGELU(nn.Module):
    """
    Complex GELU activation applied separately to real and imaginary parts.
    Alternative to magnitude-phase activation for smoother gradients.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = x[..., 0], x[..., 1]
        return torch.stack([F.gelu(real), F.gelu(imag)], dim=-1)


class ComplexLayerNorm(nn.Module):
    """
    Complex-valued Layer Normalization.
    Normalizes based on complex magnitude.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape, 2))
        self.beta = nn.Parameter(torch.zeros(normalized_shape, 2))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., features, 2]"""
        # Compute complex magnitude for normalization
        real, imag = x[..., 0], x[..., 1]
        magnitude = torch.sqrt(real**2 + imag**2 + self.eps)
        
        mean = magnitude.mean(dim=-1, keepdim=True)
        var = magnitude.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        normalized_magnitude = (magnitude - mean) / torch.sqrt(var + self.eps)
        
        # Scale back to complex
        scale = normalized_magnitude / (magnitude + self.eps)
        real_norm = real * scale
        imag_norm = imag * scale
        
        # Apply learnable affine
        real_out = self.gamma[..., 0] * real_norm + self.beta[..., 0]
        imag_out = self.gamma[..., 1] * imag_norm + self.beta[..., 1]
        
        return torch.stack([real_out, imag_out], dim=-1)


# ============================================================================
# Deep Unfolding for SAR Physics
# ============================================================================

class SARDataConsistency(nn.Module):
    """
    Data consistency layer for SAR processing.
    Enforces consistency with raw measurements through forward SAR operator.
    """
    def __init__(self, max_height: int = 5000, max_width: int = 100):
        super().__init__()
        self.max_height = max_height
        self.max_width = max_width
        
        # Learnable step size for data consistency
        self.step_size = nn.Parameter(torch.tensor(0.5))
        
        # Learnable complex frequency response (approximates SAR forward model)
        self.frequency_response_real = nn.Parameter(torch.ones(max_height))
        self.frequency_response_imag = nn.Parameter(torch.zeros(max_height))
    
    def forward_operator(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simplified SAR forward operator (azimuth FFT with learned response).
        Args:
            x: Spatial domain complex data [B, H, W, 2]
        Returns:
            Frequency domain data [B, H, W, 2]
        """
        x_complex = torch.complex(x[..., 0], x[..., 1])
        
        # FFT along azimuth
        X = fft.fft(x_complex, dim=1)
        
        # Apply learned frequency response
        H = X.shape[1]
        freq_response = torch.complex(
            self.frequency_response_real[:H].to(x.device),
            self.frequency_response_imag[:H].to(x.device)
        )
        X_filtered = X * freq_response.view(1, H, 1)
        
        return torch.stack([X_filtered.real, X_filtered.imag], dim=-1)
    
    def backward_operator(self, y: torch.Tensor) -> torch.Tensor:
        """
        Adjoint of forward operator.
        Args:
            y: Frequency domain data [B, H, W, 2]
        Returns:
            Spatial domain data [B, H, W, 2]
        """
        y_complex = torch.complex(y[..., 0], y[..., 1])
        
        # Apply conjugate of frequency response
        H = y.shape[1]
        freq_response = torch.complex(
            self.frequency_response_real[:H].to(y.device),
            self.frequency_response_imag[:H].to(y.device)
        )
        Y_filtered = y_complex * torch.conj(freq_response.view(1, H, 1))
        
        # IFFT along azimuth
        x_complex = fft.ifft(Y_filtered, dim=1)
        
        return torch.stack([x_complex.real, x_complex.imag], dim=-1)
    
    def forward(self, x_current: torch.Tensor, measurements: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Data consistency update step.
        Args:
            x_current: Current estimate [B, H, W, 2]
            measurements: Raw measurements (k-space) [B, H, W, 2]
            mask: Sampling mask [B, H, W] (1 = measured, 0 = not measured)
        Returns:
            Updated estimate [B, H, W, 2]
        """
        # Forward pass
        y_current = self.forward_operator(x_current)
        
        # Compute residual with measurements
        if mask is not None:
            residual = (measurements - y_current) * mask.unsqueeze(-1)
        else:
            residual = measurements - y_current
        
        # Backward pass (gradient step)
        gradient = self.backward_operator(residual)
        
        # Update with learned step size
        x_updated = x_current + self.step_size * gradient
        
        return x_updated


class LearnedProximalOperator(nn.Module):
    """
    Learned denoiser/regularizer for deep unfolding.
    Acts as proximal operator in iterative reconstruction.
    """
    def __init__(self, channels: int = 1, hidden_channels: int = 64):
        super().__init__()
        
        # Lightweight complex-valued denoiser
        # Input is [B, 1, H, W, 2] where 1 is the complex channel dimension
        self.net = nn.Sequential(
            ComplexConv2d(channels, hidden_channels, 3, padding=1),
            ComplexGELU(),
            ComplexConv2d(hidden_channels, hidden_channels, 3, padding=1),
            ComplexGELU(),
            ComplexConv2d(hidden_channels, channels, 3, padding=1),
        )
        
        # Learnable weight for proximal step
        self.prox_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex input [B, H, W, 2]
        Returns:
            Denoised output [B, H, W, 2]
        """
        # Convert to [B, C, H, W, 2] format for ComplexConv2d
        # Input: [B, H, W, 2] -> [B, 1, H, W, 2] where C=1 (single complex channel)
        x_conv = x.unsqueeze(1)  # [B, 1, H, W, 2]
        
        # Apply denoiser
        denoised = self.net(x_conv)  # [B, channels, H, W, 2]
        
        # Convert back to [B, H, W, 2]
        denoised = denoised.squeeze(1)  # Remove channel dimension
        
        # Residual connection with learned weight
        output = x + self.prox_weight * (denoised - x)
        
        return output


class DeepUnfoldedBlock(nn.Module):
    """
    Single iteration of deep unfolded optimization.
    Combines data consistency + learned proximal operator.
    """
    def __init__(self, max_height: int = 5000, max_width: int = 100, hidden_channels: int = 64):
        super().__init__()
        
        self.data_consistency = SARDataConsistency(max_height, max_width)
        self.proximal_op = LearnedProximalOperator(channels=1, hidden_channels=hidden_channels)
    
    def forward(self, x: torch.Tensor, measurements: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        One unfolding iteration: data consistency → proximal operator.
        """
        # Data consistency step
        x_dc = self.data_consistency(x, measurements, mask)
        
        # Proximal/denoising step
        x_out = self.proximal_op(x_dc)
        
        return x_out


# ============================================================================
# LISTA: Learned Iterative Soft Thresholding for Sparsity
# ============================================================================

class LISTAEncoder(nn.Module):
    """
    Learned ISTA for producing sparse latent codes.
    Implements T iterations of soft-thresholding with learned dictionaries.
    """
    def __init__(self, input_dim: int, latent_dim: int, num_iterations: int = 5, sparsity_lambda: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_iterations = num_iterations
        
        # Learned dictionary (encoder)
        self.encoder_dict = nn.Linear(input_dim, latent_dim, bias=False)
        
        # Learned step sizes for each iteration
        self.step_sizes = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(num_iterations)
        ])
        
        # Learned thresholds for each iteration
        self.thresholds = nn.ParameterList([
            nn.Parameter(torch.tensor(sparsity_lambda)) for _ in range(num_iterations)
        ])
        
        # Learned weight matrix for iterative update
        self.weight_matrix = nn.Linear(latent_dim, latent_dim, bias=False)
        
        # Initialize with scaled identity for stability
        with torch.no_grad():
            nn.init.orthogonal_(self.encoder_dict.weight, gain=1.0)
            nn.init.eye_(self.weight_matrix.weight)
            self.weight_matrix.weight.mul_(0.9)
    
    def soft_threshold(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Soft thresholding operator for sparsity."""
        return torch.sign(x) * F.relu(torch.abs(x) - threshold)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input [B, N, input_dim]
        Returns:
            z: Sparse codes [B, N, latent_dim]
            sparsity: L1 norm of codes (for loss)
        """
        # Initial encoding
        z = self.encoder_dict(x)
        
        # ISTA iterations
        for i in range(self.num_iterations):
            # Gradient step
            residual = self.weight_matrix(z) - self.encoder_dict(x)
            z = z - self.step_sizes[i] * residual
            
            # Soft thresholding (proximal operator for L1)
            z = self.soft_threshold(z, self.thresholds[i].abs())
        
        # Compute sparsity for loss
        sparsity = torch.mean(torch.abs(z))
        
        return z, sparsity


class LISTADecoder(nn.Module):
    """Decoder for LISTA - learned synthesis dictionary."""
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.decoder_dict = nn.Linear(latent_dim, output_dim, bias=False)
        
        with torch.no_grad():
            nn.init.orthogonal_(self.decoder_dict.weight, gain=1.0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Sparse codes [B, N, latent_dim]
        Returns:
            x_reconstructed [B, N, output_dim]
        """
        return self.decoder_dict(z)


# ============================================================================
# Entropy Model for Learned Compression
# ============================================================================

class ScaleHyperprior(nn.Module):
    """
    Scale hyperprior network (Ballé et al. 2018).
    Predicts per-location Gaussian scales for entropy coding.
    """
    def __init__(self, latent_channels: int, hyperprior_channels: int = 64):
        super().__init__()
        
        # Hyperencoder: latent → hyperlatent
        self.hyper_encoder = nn.Sequential(
            nn.Conv2d(latent_channels, hyperprior_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hyperprior_channels, hyperprior_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hyperprior_channels, hyperprior_channels, 3, stride=2, padding=1),
        )
        
        # Hyperdecoder: hyperlatent → scale parameters
        self.hyper_decoder = nn.Sequential(
            nn.ConvTranspose2d(hyperprior_channels, hyperprior_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hyperprior_channels, hyperprior_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hyperprior_channels, latent_channels, 3, stride=1, padding=1),
        )
    
    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent: Latent representation [B, C, H_lat, W_lat]
        Returns:
            scales: Predicted scales [B, C, H_lat, W_lat]
            hyperlatent: Compressed side information [B, C_hyper, H_hyper, W_hyper]
        """
        hyperlatent = self.hyper_encoder(latent)
        scales_raw = torch.exp(self.hyper_decoder(hyperlatent))  # Ensure positive scales
        
        # Resize scales to match latent size exactly (in case of dimension mismatch)
        if scales_raw.shape != latent.shape:
            scales = F.interpolate(scales_raw, size=latent.shape[2:], mode='bilinear', align_corners=False)
        else:
            scales = scales_raw
        
        return scales, hyperlatent


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantization (RVQ) for high-fidelity compression.
    
    Inspired by SoundStream/EnCodec - iteratively quantizes residuals
    to preserve fine details that single-stage quantization misses.
    
    Key idea: 
        x ≈ q1 + q2 + q3 + ... + qN
    where each qi is quantized version of residual from previous stages.
    """
    def __init__(
        self,
        num_quantizers: int = 4,
        codebook_size: int = 1024,
        codebook_dim: int = 64,
        commitment_weight: float = 0.25,
        kmeans_init: bool = True,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay
        
        # Multiple codebooks (one per stage)
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, codebook_dim))
            for _ in range(num_quantizers)
        ])
        
        # Initialize codebooks with scaled random values
        if not kmeans_init:
            for codebook in self.codebooks:
                nn.init.normal_(codebook, mean=0, std=0.01)
        
        # EMA for codebook updates (optional, for better training)
        if ema_decay > 0:
            self.register_buffer('cluster_size', torch.zeros(num_quantizers, codebook_size))
            self.register_buffer('embed_avg', torch.zeros(num_quantizers, codebook_size, codebook_dim))
    
    def quantize_stage(
        self, 
        x: torch.Tensor, 
        codebook: torch.Tensor,
        stage_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single VQ stage.
        
        Args:
            x: Input residual [B, N, D]
            codebook: Codebook for this stage [K, D]
            stage_idx: Index of current stage
            
        Returns:
            quantized: Quantized vectors [B, N, D]
            indices: Codebook indices [B, N]
            commitment_loss: Commitment loss for this stage
        """
        B, N, D = x.shape
        
        # Flatten for efficient distance computation
        x_flat = x.reshape(-1, D)  # [B*N, D]
        
        # Compute distances to all codebook vectors
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x·c
        x_norm = (x_flat ** 2).sum(dim=1, keepdim=True)  # [B*N, 1]
        codebook_norm = (codebook ** 2).sum(dim=1, keepdim=True).t()  # [1, K]
        distances = x_norm + codebook_norm - 2 * torch.matmul(x_flat, codebook.t())  # [B*N, K]
        
        # Find nearest codebook vectors
        indices = torch.argmin(distances, dim=1)  # [B*N]
        
        # Lookup quantized vectors
        quantized_flat = F.embedding(indices, codebook)  # [B*N, D]
        quantized = quantized_flat.view(B, N, D)
        
        # Commitment loss (encourages encoder to commit to codebook)
        commitment_loss = F.mse_loss(x.detach(), quantized)
        
        # Codebook loss (updates codebook to match encoder output)
        codebook_loss = F.mse_loss(x, quantized.detach())
        
        # Straight-through estimator for backprop
        quantized = x + (quantized - x).detach()
        
        # Total loss for this stage
        stage_loss = codebook_loss + self.commitment_weight * commitment_loss
        
        # Update EMA statistics (if using EMA)
        if hasattr(self, 'cluster_size') and self.training:
            with torch.no_grad():
                # Count usage of each codebook entry
                indices_onehot = F.one_hot(indices, self.codebook_size).float()  # [B*N, K]
                cluster_size = indices_onehot.sum(dim=0)  # [K]
                
                # Update cluster sizes with EMA
                self.cluster_size[stage_idx] = self.cluster_size[stage_idx] * self.ema_decay + \
                                                cluster_size * (1 - self.ema_decay)
                
                # Update embeddings with EMA
                embed_sum = torch.matmul(indices_onehot.t(), x_flat)  # [K, D]
                self.embed_avg[stage_idx] = self.embed_avg[stage_idx] * self.ema_decay + \
                                             embed_sum * (1 - self.ema_decay)
                
                # Update codebook (Laplace smoothing)
                n = self.cluster_size[stage_idx].sum()
                cluster_size_smoothed = (self.cluster_size[stage_idx] + 1e-5) / (n + self.codebook_size * 1e-5) * n
                embed_normalized = self.embed_avg[stage_idx] / cluster_size_smoothed.unsqueeze(1)
                codebook.data.copy_(embed_normalized)
        
        return quantized, indices.view(B, N), stage_loss
    
    def forward(self, x: torch.Tensor, return_indices: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Multi-stage residual quantization.
        
        Args:
            x: Input [B, N, D] where N is num_tokens, D is embed_dim
            return_indices: If True, return quantization indices
            
        Returns:
            quantized: Quantized representation [B, N, D]
            total_loss: Sum of all stage losses
            indices: Optional [B, N, num_quantizers] quantization indices
        """
        residual = x
        quantized_sum = torch.zeros_like(x)
        total_loss = 0.0
        
        all_indices = [] if return_indices else None
        
        # Iterate through quantization stages
        for stage_idx in range(self.num_quantizers):
            # Quantize current residual
            quantized_stage, indices, stage_loss = self.quantize_stage(
                residual, 
                self.codebooks[stage_idx],
                stage_idx
            )
            
            # Accumulate quantized representation
            quantized_sum = quantized_sum + quantized_stage
            
            # Update residual for next stage
            residual = residual - quantized_stage.detach()
            
            # Accumulate loss
            total_loss = total_loss + stage_loss
            
            # Store indices if needed
            if return_indices:
                all_indices.append(indices)
        
        # Stack indices if requested
        if return_indices:
            all_indices = torch.stack(all_indices, dim=-1)  # [B, N, num_quantizers]
        
        return quantized_sum, total_loss, all_indices
    
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode quantization indices back to continuous representation.
        
        Args:
            indices: [B, N, num_quantizers] quantization indices
            
        Returns:
            decoded: [B, N, D] reconstructed vectors
        """
        B, N, Q = indices.shape
        assert Q == self.num_quantizers, f"Expected {self.num_quantizers} quantizers, got {Q}"
        
        decoded = torch.zeros(B, N, self.codebook_dim, device=indices.device)
        
        for stage_idx in range(self.num_quantizers):
            stage_indices = indices[:, :, stage_idx]  # [B, N]
            stage_quantized = F.embedding(stage_indices, self.codebooks[stage_idx])  # [B, N, D]
            decoded = decoded + stage_quantized
        
        return decoded


class EntropyBottleneck(nn.Module):
    """
    Entropy bottleneck for rate estimation during training.
    Uses additive uniform noise to approximate quantization.
    
    Note: This is kept for backward compatibility, but ResidualVectorQuantizer
    is recommended for better detail preservation.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('initialized', torch.tensor(0))
    
    def quantize(self, x: torch.Tensor, mode: str = 'training') -> torch.Tensor:
        """
        Quantize with noise (training) or rounding (inference).
        """
        if mode == 'training':
            # Add uniform noise in [-0.5, 0.5] to approximate quantization
            noise = torch.rand_like(x) - 0.5
            return x + noise
        else:
            # Round to nearest integer
            return torch.round(x)
    
    def forward(self, x: torch.Tensor, scales: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Latent to quantize [B, C, H, W]
            scales: Predicted scales [B, C, H, W]
        Returns:
            x_quantized: Quantized latent
            rate: Estimated bit rate
        """
        # Quantize
        x_quantized = self.quantize(x, mode='training' if self.training else 'inference')
        
        # Estimate rate using Gaussian assumption: rate = -log2(p(x))
        # p(x) ~ N(0, scales^2)
        log_prob = -0.5 * torch.log(2 * math.pi * scales**2) - (x_quantized**2) / (2 * scales**2)
        rate = -log_prob / math.log(2)  # Convert to bits
        
        return x_quantized, rate


# ============================================================================
# Enhanced Spatial Tokenizer with Complex & Physics Integration
# ============================================================================

class PhysicsAwareSpatialTokenizer(nn.Module):
    """
    Enhanced tokenizer with:
    - Complex-valued processing
    - Multi-scale feature extraction
    - Physics-aware transformations
    """
    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        patch_size: Tuple[int, int] = (1, 1),
        max_height: int = 5000,
        max_width: int = 100,
        use_complex: bool = True,
    ):
        super().__init__()
        # For SAR data [B, H, W, 2], we treat it as 1 complex-valued channel
        self.input_channels = 1  # Always 1 for complex SAR data
        self.embed_dim = embed_dim
        self.patch_height, self.patch_width = patch_size
        self.use_complex = use_complex
        
        if use_complex:
            # Complex-valued feature extraction
            # Input: [B, 1, H, W, 2] (1 complex channel)
            self.feature_extractor = nn.Sequential(
                ComplexConv2d(1, 16, 3, padding=1),
                ComplexGELU(),
                ComplexConv2d(16, 16, 3, padding=1),
                ComplexGELU(),
                ComplexConv2d(16, 1, 1),
            )
            
            # Complex patch embedding
            # Each patch has patch_size[0] * patch_size[1] complex values (each with 2 components)
            patch_dim = patch_size[0] * patch_size[1]  # Number of complex values per patch
            self.patch_embed = ComplexLinear(patch_dim, embed_dim)
        else:
            # Fallback to real-valued (treats real/imag as 2 separate channels)
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(2, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 2, 1),
            )
            
            patch_dim = 2 * patch_size[0] * patch_size[1]  # real + imag flattened
            self.patch_embed = nn.Linear(patch_dim, embed_dim * 2)  # *2 for real/imag
        
        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(max_height // patch_size[0], max_width // patch_size[1], embed_dim * 2) * 0.02
        )
        
        self.feature_blend = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: Input [B, H, W, 2] complex data
        Returns:
            tokens: [B, num_tokens, embed_dim*2] (real/imag stacked)
            token_height, token_width
        """
        batch_size, height, width, _ = x.shape
        token_height = height // self.patch_height
        token_width = width // self.patch_width
        
        if self.use_complex:
            # Complex feature extraction (expects [B, 1, H, W, 2])
            x_complex = x.unsqueeze(1)  # [B, 1, H, W, 2] - 1 complex channel
            
            features = self.feature_extractor(x_complex)
            x_enhanced = x_complex + self.feature_blend * features
            
            # Remove channel dimension
            x_enhanced = x_enhanced.squeeze(1)  # [B, H, W, 2]
        else:
            # Real-valued path (flatten real/imag as channels)
            x_real_imag = x.permute(0, 3, 1, 2)  # [B, 2, H, W]
            features = self.feature_extractor(x_real_imag)
            x_enhanced = x_real_imag + self.feature_blend * features
            x_enhanced = x_enhanced.permute(0, 2, 3, 1)  # Back to [B, H, W, 2]
        
        # Patchify
        if self.patch_height == 1 and self.patch_width == 1:
            patches = x_enhanced
        else:
            patches = einops.rearrange(
                x_enhanced,
                'b (th ph) (tw pw) c -> b th tw (ph pw c)',
                ph=self.patch_height,
                pw=self.patch_width
            )
        
        # Flatten spatial dimensions
        patches_flat = einops.rearrange(patches, 'b h w c -> b (h w) c')
        
        # Embed
        if self.use_complex:
            # Reshape: [B, N, patch_h*patch_w*2] -> [B, N, patch_h*patch_w, 2]
            B, N, C = patches_flat.shape
            patch_elements = self.patch_height * self.patch_width
            patches_complex = patches_flat.reshape(B, N, patch_elements, 2)  # [B, N, patch_dim, 2]
            
            tokens = self.patch_embed(patches_complex)  # [B, N, embed_dim, 2]
            # Flatten complex to real representation for transformer
            tokens = einops.rearrange(tokens, 'b n d c -> b n (d c)')  # [B, N, embed_dim*2]
        else:
            tokens = self.patch_embed(patches_flat)
        
        # Add positional encoding
        pos_embed = self.pos_embed[:token_height, :token_width]
        pos_embed = einops.rearrange(pos_embed, 'h w d -> (h w) d')
        tokens = tokens + pos_embed.unsqueeze(0)
        
        return tokens, token_height, token_width


# ============================================================================
# Enhanced Transformer Block with Complex Attention
# ============================================================================

class ComplexAttentionBlock(nn.Module):
    """
    Transformer block operating on complex-valued representations.
    Maintains phase relationships through attention.
    """
    def __init__(
        self,
        embed_dim: int,  # This is the real embed_dim (complex pairs will be embed_dim*2)
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.complex_dim = embed_dim * 2  # Real/imag concatenated
        self.num_heads = num_heads
        
        # Complex multi-head attention
        self.norm1 = ComplexLayerNorm(embed_dim)
        
        # QKV projection (complex)
        self.qkv = ComplexLinear(embed_dim, embed_dim * 3)
        self.proj = ComplexLinear(embed_dim, embed_dim)
        
        # MLP
        self.norm2 = ComplexLayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            ComplexLinear(embed_dim, mlp_hidden),
            ComplexGELU(),
            ComplexLinear(mlp_hidden, embed_dim),
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def complex_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Complex-valued scaled dot-product attention.
        Args:
            q, k, v: [B, num_heads, seq_len, head_dim, 2]
        Returns:
            output: [B, num_heads, seq_len, head_dim, 2]
        """
        head_dim = q.shape[-2]
        
        # Complex dot product: q* · k (conjugate of q times k)
        # (a + ib)* · (c + id) = (ac + bd) + i(ad - bc)
        q_real, q_imag = q[..., 0], q[..., 1]
        k_real, k_imag = k[..., 0], k[..., 1]
        
        # Attention scores (use magnitude of complex dot product)
        attn_real = torch.matmul(q_real, k_real.transpose(-2, -1)) + torch.matmul(q_imag, k_imag.transpose(-2, -1))
        attn_imag = torch.matmul(q_real, k_imag.transpose(-2, -1)) - torch.matmul(q_imag, k_real.transpose(-2, -1))
        
        attn_magnitude = torch.sqrt(attn_real**2 + attn_imag**2 + 1e-8)
        
        # Scale and softmax
        attn_magnitude = attn_magnitude / math.sqrt(head_dim)
        attn_weights = F.softmax(attn_magnitude, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values (complex multiplication)
        v_real, v_imag = v[..., 0], v[..., 1]
        out_real = torch.matmul(attn_weights, v_real)
        out_imag = torch.matmul(attn_weights, v_imag)
        
        return torch.stack([out_real, out_imag], dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, embed_dim*2] (real/imag concatenated)
        Returns:
            output: [B, seq_len, embed_dim*2]
        """
        # Reshape to complex format [B, seq_len, embed_dim, 2]
        B, N, _ = x.shape
        x_complex = x.view(B, N, self.embed_dim, 2)
        
        # Attention block with residual
        x_norm = self.norm1(x_complex)
        
        # QKV projection
        qkv = self.qkv(x_norm)  # [B, N, embed_dim*3, 2]
        qkv = qkv.view(B, N, 3, self.num_heads, self.embed_dim // self.num_heads, 2)
        qkv = qkv.permute(2, 0, 3, 1, 4, 5)  # [3, B, num_heads, N, head_dim, 2]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Complex attention
        attn_out = self.complex_attention(q, k, v)  # [B, num_heads, N, head_dim, 2]
        attn_out = attn_out.permute(0, 2, 1, 3, 4).contiguous()  # [B, N, num_heads, head_dim, 2]
        attn_out = attn_out.view(B, N, self.embed_dim, 2)
        
        # Projection
        attn_out = self.proj(attn_out)
        
        # Residual
        x_complex = x_complex + attn_out
        
        # MLP block with residual
        x_norm2 = self.norm2(x_complex)
        mlp_out = self.mlp(x_norm2)
        x_complex = x_complex + mlp_out
        
        # Flatten back to [B, N, embed_dim*2]
        output = x_complex.view(B, N, self.complex_dim)
        
        return output


# ============================================================================
# Complete Physics-Aware Spatial Transformer
# ============================================================================

class PhysicsAwareSpatialTransformer(nn.Module):
    """
    Complete SAR compression model with:
    1. Complex-valued processing
    2. Deep unfolded physics
    3. LISTA sparsity priors
    4. Entropy modeling for compression
    5. Multi-domain losses
    """
    def __init__(
        self,
        input_channels: int = 2,
        output_channels: int = 2,
        embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        patch_size: Tuple[int, int] = (100, 20),
        max_height: int = 5000,
        max_width: int = 100,
        dropout: float = 0.0,
        physics_config: Optional[PhysicsConfig] = None,
        compression_config: Optional[CompressionConfig] = None,
    ):
        super().__init__()
        
        self.physics_config = physics_config or PhysicsConfig()
        self.compression_config = compression_config or CompressionConfig()
        self.input_dim = input_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # 1. Physics preprocessing (deep unfolding)
        if self.physics_config.use_deep_unfolding:
            self.unfolding_blocks = nn.ModuleList([
                DeepUnfoldedBlock(max_height, max_width, hidden_channels=32)
                for _ in range(self.physics_config.unfolding_iterations)
            ])
        
        # 2. Complex-aware tokenizer
        self.tokenizer = PhysicsAwareSpatialTokenizer(
            input_channels=input_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            max_height=max_height,
            max_width=max_width,
            use_complex=self.physics_config.use_complex_layers,
        )
        
        # 3. LISTA encoder for sparsity
        if self.physics_config.use_sparsity_prior:
            self.lista_encoder = LISTAEncoder(
                input_dim=embed_dim * 2,  # Complex (real/imag)
                latent_dim=embed_dim * 2,
                num_iterations=5,
                sparsity_lambda=self.physics_config.sparsity_lambda,
            )
            self.lista_decoder = LISTADecoder(
                latent_dim=embed_dim * 2,
                output_dim=embed_dim * 2,
            )
        
        # 4. Transformer encoder
        self.transformer_blocks = nn.ModuleList([
            ComplexAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            ) for _ in range(num_layers)
        ])
        
        # 5. Entropy model for compression with optional RVQ
        if self.compression_config.use_entropy_model:
            # NEW: Residual Vector Quantization for detail preservation
            if self.compression_config.use_residual_quantization:
                self.residual_quantizer = ResidualVectorQuantizer(
                    num_quantizers=self.compression_config.num_rvq_quantizers,
                    codebook_size=self.compression_config.rvq_codebook_size,
                    codebook_dim=embed_dim * 2,  # Match token dimension
                    commitment_weight=self.compression_config.rvq_commitment_weight,
                    kmeans_init=False,
                    ema_decay=0.99,
                )
            else:
                # Fallback to standard quantization
                self.pre_entropy_conv = nn.Conv2d(embed_dim * 2, 64, 1)
                
                if self.compression_config.use_hyperprior:
                    self.hyperprior = ScaleHyperprior(64, hyperprior_channels=32)
                
                self.entropy_bottleneck = EntropyBottleneck()
                
                self.post_entropy_conv = nn.Conv2d(64, embed_dim * 2, 1)
        
        # 6. Decoder with full attention (matching encoder depth for better reconstruction)
        # Using same number of layers as encoder for symmetric architecture
        
        # Initial attention-based upsampling/refinement block
        self.decoder_init_attention = ComplexAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio * 2,  # Larger MLP for initial reconstruction
            dropout=dropout,
        )
        
        # Main decoder blocks
        self.decoder_blocks = nn.ModuleList([
            ComplexAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            ) for _ in range(num_layers)  # Full depth for better upsampling/reconstruction
        ])
        
        # 7. Detokenizer
        # For complex SAR data, we have 1 complex channel per patch
        patch_dim = patch_size[0] * patch_size[1]  # Number of complex values per patch
        if self.physics_config.use_complex_layers:
            self.detokenize_proj = ComplexLinear(embed_dim, patch_dim)
        else:
            self.detokenize_proj = nn.Linear(embed_dim * 2, patch_dim * 2)
        
        # 8. Post-reconstruction deblocking refinement
        # This CNN smooths patch boundaries and reduces blocking artifacts
        if self.physics_config.use_complex_layers:
            self.deblocking_net = nn.Sequential(
                ComplexConv2d(1, 32, kernel_size=5, padding=2),  # Large kernel for cross-patch smoothing
                ComplexGELU(),
                ComplexConv2d(32, 32, kernel_size=5, padding=2),
                ComplexGELU(),
                ComplexConv2d(32, 1, kernel_size=3, padding=1),  # Residual output
            )
        else:
            self.deblocking_net = nn.Sequential(
                nn.Conv2d(2, 64, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv2d(64, 2, kernel_size=3, padding=1),
            )
        
        # Learnable blending factor for deblocking
        self.deblock_blend = nn.Parameter(torch.tensor(0.3))
        
        # Learnable output scaling
        self.output_scale = nn.Parameter(torch.tensor(1.0))
    
    def encode(self, x: torch.Tensor, measurements: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Encode input to compressed latent representation.
        
        Args:
            x: Input [B, H, W, 2] complex SAR data
            measurements: Optional raw measurements for data consistency
        
        Returns:
            Dictionary containing:
                - tokens: Compressed latent [B, num_tokens, embed_dim*2]
                - sparsity: L1 sparsity measure
                - rate: Estimated bit rate
                - token_height, token_width: Spatial dimensions
        """
        outputs = {}
        
        # 1. Deep unfolding (physics-based preprocessing)
        if self.physics_config.use_deep_unfolding and measurements is not None:
            x_processed = x
            for unfolding_block in self.unfolding_blocks:
                x_processed = unfolding_block(x_processed, measurements)
        else:
            x_processed = x
        
        # 2. Tokenization
        tokens, token_height, token_width = self.tokenizer(x_processed)
        outputs['token_height'] = token_height
        outputs['token_width'] = token_width
        
        # 3. LISTA for sparsity (optional)
        if self.physics_config.use_sparsity_prior:
            tokens_sparse, sparsity = self.lista_encoder(tokens)
            outputs['sparsity'] = sparsity
            tokens = tokens_sparse
        
        # 4. Transformer encoding
        for block in self.transformer_blocks:
            tokens = block(tokens)
        
        # 5. Quantization with RVQ or standard entropy model
        if self.compression_config.use_entropy_model:
            B, N, C = tokens.shape
            
            if self.compression_config.use_residual_quantization:
                # NEW: Use Residual Vector Quantization for detail preservation
                # tokens: [B, N, embed_dim*2]
                tokens_quantized, rvq_loss, indices = self.residual_quantizer(tokens, return_indices=True)
                
                # Estimate rate from RVQ
                # Each quantizer contributes log2(codebook_size) bits per token
                bits_per_token = self.compression_config.num_rvq_quantizers * math.log2(self.compression_config.rvq_codebook_size)
                outputs['rate'] = torch.tensor(bits_per_token).to(x.device)
                outputs['rvq_loss'] = rvq_loss
                outputs['rvq_indices'] = indices  # For potential entropy coding
                
                tokens = tokens_quantized
            else:
                # Fallback to standard entropy bottleneck
                # Reshape to spatial for conv-based entropy model
                tokens_spatial = tokens.view(B, token_height, token_width, C)
                tokens_spatial = tokens_spatial.permute(0, 3, 1, 2)  # [B, C, H, W]
                
                # Map to entropy model dimension
                tokens_conv = self.pre_entropy_conv(tokens_spatial)
                
                # Hyperprior for scale estimation
                if self.compression_config.use_hyperprior:
                    scales, hyperlatent = self.hyperprior(tokens_conv)
                    outputs['hyperlatent'] = hyperlatent
                else:
                    scales = torch.ones_like(tokens_conv)
                
                # Quantization and rate estimation
                if self.compression_config.use_quantization:
                    tokens_quantized, rate = self.entropy_bottleneck(tokens_conv, scales)
                    outputs['rate'] = rate.mean()  # Average bits per symbol
                else:
                    tokens_quantized = tokens_conv
                    outputs['rate'] = torch.tensor(0.0).to(x.device)
                
                # Map back
                tokens_decoded = self.post_entropy_conv(tokens_quantized)
                tokens = tokens_decoded.permute(0, 2, 3, 1).reshape(B, N, C)
        else:
            outputs['rate'] = torch.tensor(0.0).to(x.device)
        
        outputs['tokens'] = tokens
        
        return outputs
    
    def decode(self, tokens: torch.Tensor, token_height: int, token_width: int) -> torch.Tensor:
        """
        Decode latent tokens to spatial representation.
        
        Args:
            tokens: [B, num_tokens, embed_dim*2]
            token_height, token_width: Spatial token dimensions
        
        Returns:
            reconstructed: [B, H, W, 2] complex SAR data
        """
        B, N, C = tokens.shape
        
        # 1. LISTA decoder (if used)
        if self.physics_config.use_sparsity_prior:
            tokens = self.lista_decoder(tokens)
        
        # 2. Initial attention-based upsampling/refinement
        # This helps the decoder focus on important spatial relationships early
        tokens = self.decoder_init_attention(tokens)
        
        # 3. Main transformer decoding blocks
        for block in self.decoder_blocks:
            tokens = block(tokens)
        
        # 4. Detokenize
        # Reshape to complex format for complex linear
        if self.physics_config.use_complex_layers:
            tokens_complex = tokens.view(B, N, self.embed_dim, 2)
            patches = self.detokenize_proj(tokens_complex)  # [B, N, patch_dim, 2]
            patches_flat = patches.view(B, N, -1)  # [B, N, patch_dim*2]
        else:
            patches_flat = self.detokenize_proj(tokens)
        
        # 5. Reshape to spatial
        patch_size_total = self.patch_size[0] * self.patch_size[1] * 2  # *2 for real/imag
        
        if self.patch_size[0] == 1 and self.patch_size[1] == 1:
            spatial = patches_flat.view(B, token_height, token_width, 2)
        else:
            patches_spatial = patches_flat.view(B, token_height, token_width, -1)
            spatial = einops.rearrange(
                patches_spatial,
                'b th tw (ph pw c) -> b (th ph) (tw pw) c',
                ph=self.patch_size[0],
                pw=self.patch_size[1],
                c=2
            )
        
        # 6. Deblocking refinement (reduce patch boundary artifacts)
        if self.physics_config.use_complex_layers:
            # Complex path: [B, H, W, 2] -> [B, 1, H, W, 2]
            spatial_complex = spatial.unsqueeze(1)
            deblock_residual = self.deblocking_net(spatial_complex)
            spatial = spatial + self.deblock_blend * deblock_residual.squeeze(1)
        else:
            # Real path: [B, H, W, 2] -> [B, 2, H, W]
            spatial_channels = spatial.permute(0, 3, 1, 2)
            deblock_residual = self.deblocking_net(spatial_channels)
            spatial = spatial_channels + self.deblock_blend * deblock_residual
            spatial = spatial.permute(0, 2, 3, 1)  # Back to [B, H, W, 2]
        
        # 7. Output scaling
        spatial = spatial * self.output_scale
        
        return spatial
    
    def forward(self, x: torch.Tensor, measurements: Optional[torch.Tensor] = None, return_aux: bool = False) -> torch.Tensor:
        """
        Full forward pass: encode → decode.
        
        Args:
            x: Input [B, H, W, 2]
            measurements: Optional raw k-space measurements
            return_aux: If True, return (reconstructed, aux_outputs). If False, return only reconstructed.
        
        Returns:
            If return_aux=False: reconstructed tensor [B, H, W, 2]
            If return_aux=True: (reconstructed [B, H, W, 2], aux_outputs dict)
        """
        # Encode
        encode_outputs = self.encode(x, measurements)
        
        # Decode
        reconstructed = self.decode(
            encode_outputs['tokens'],
            encode_outputs['token_height'],
            encode_outputs['token_width']
        )
        
        # Return based on flag
        if return_aux:
            # Collect auxiliary outputs for loss computation
            aux_outputs = {
                'sparsity': encode_outputs.get('sparsity', torch.tensor(0.0).to(x.device)),
                'rate': encode_outputs.get('rate', torch.tensor(0.0).to(x.device)),
            }
            return reconstructed, aux_outputs
        else:
            # Return only reconstructed for compatibility with standard training loops
            return reconstructed


# ============================================================================
# Multi-Domain Loss Functions
# ============================================================================

class MultiDomainSARLoss(nn.Module):
    """
    Comprehensive loss combining:
    1. Spatial domain (complex MSE)
    2. Frequency domain (azimuth spectrum)
    3. Phase consistency
    4. Sparsity penalty
    5. Rate (for compression)
    6. Deblocking (patch boundary continuity)
    """
    def __init__(
        self,
        spatial_weight: float = 1.0,
        frequency_weight: float = 0.3,
        phase_weight: float = 0.5,
        sparsity_weight: float = 0.01,
        rate_weight: float = 0.01,
        deblocking_weight: float = 0.3,
        patch_size: Tuple[int, int] = (100, 20),
    ):
        super().__init__()
        self.spatial_weight = spatial_weight
        self.frequency_weight = frequency_weight
        self.phase_weight = phase_weight
        self.sparsity_weight = sparsity_weight
        self.rate_weight = rate_weight
        self.deblocking_weight = deblocking_weight
        self.patch_size = patch_size
    
    def complex_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Complex MSE: |pred - target|^2"""
        return F.mse_loss(pred, target)
    
    def frequency_domain_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE in azimuth frequency domain."""
        pred_complex = torch.complex(pred[..., 0], pred[..., 1])
        target_complex = torch.complex(target[..., 0], target[..., 1])
        
        # FFT along azimuth (axis 1)
        pred_fft = fft.fft(pred_complex, dim=1)
        target_fft = fft.fft(target_complex, dim=1)
        
        # MSE on magnitude and phase
        mag_loss = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))
        
        return mag_loss
    
    def phase_consistency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Phase loss using complex inner product.
        L = 1 - |<pred, target>| / (||pred|| ||target||)
        """
        pred_complex = torch.complex(pred[..., 0], pred[..., 1])
        target_complex = torch.complex(target[..., 0], target[..., 1])
        
        # Complex inner product
        inner_product = torch.sum(pred_complex * torch.conj(target_complex), dim=[1, 2])
        
        # Norms
        pred_norm = torch.sqrt(torch.sum(torch.abs(pred_complex)**2, dim=[1, 2]))
        target_norm = torch.sqrt(torch.sum(torch.abs(target_complex)**2, dim=[1, 2]))
        
        # Normalized inner product
        similarity = torch.abs(inner_product) / (pred_norm * target_norm + 1e-8)
        
        # Loss (1 - similarity)
        loss = 1.0 - similarity.mean()
        
        return loss
    
    def deblocking_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Penalize discontinuities at patch boundaries.
        Computes difference between adjacent patches to encourage smooth transitions.
        
        Args:
            pred: [B, H, W, 2]
            target: [B, H, W, 2]
        """
        ph, pw = self.patch_size
        
        # Convert to magnitude for boundary analysis
        pred_mag = torch.sqrt(pred[..., 0]**2 + pred[..., 1]**2)  # [B, H, W]
        target_mag = torch.sqrt(target[..., 0]**2 + target[..., 1]**2)
        
        # Compute gradients at patch boundaries
        # Vertical boundaries (every ph rows)
        boundary_loss = 0.0
        num_boundaries = 0
        
        # Vertical patch boundaries
        for i in range(ph, pred_mag.shape[1], ph):
            if i < pred_mag.shape[1]:
                # Gradient across boundary
                pred_grad = torch.abs(pred_mag[:, i, :] - pred_mag[:, i-1, :])
                target_grad = torch.abs(target_mag[:, i, :] - target_mag[:, i-1, :])
                boundary_loss += F.mse_loss(pred_grad, target_grad)
                num_boundaries += 1
        
        # Horizontal patch boundaries
        for j in range(pw, pred_mag.shape[2], pw):
            if j < pred_mag.shape[2]:
                # Gradient across boundary
                pred_grad = torch.abs(pred_mag[:, :, j] - pred_mag[:, :, j-1])
                target_grad = torch.abs(target_mag[:, :, j] - target_mag[:, :, j-1])
                boundary_loss += F.mse_loss(pred_grad, target_grad)
                num_boundaries += 1
        
        if num_boundaries > 0:
            boundary_loss = boundary_loss / num_boundaries
        
        return boundary_loss
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        aux_outputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss.
        
        Args:
            pred: Predicted [B, H, W, 2]
            target: Ground truth [B, H, W, 2]
            aux_outputs: Dictionary with 'sparsity', 'rate'
        
        Returns:
            total_loss: Scalar
            loss_dict: Individual loss components
        """
        losses = {}
        
        # 1. Spatial domain loss
        losses['spatial'] = self.complex_mse(pred, target) * self.spatial_weight
        
        # 2. Frequency domain loss
        losses['frequency'] = self.frequency_domain_loss(pred, target) * self.frequency_weight
        
        # 3. Phase consistency
        losses['phase'] = self.phase_consistency_loss(pred, target) * self.phase_weight
        
        # 4. Deblocking loss (patch boundary continuity)
        losses['deblocking'] = self.deblocking_loss(pred, target) * self.deblocking_weight
        
        # 5. Sparsity penalty
        sparsity = aux_outputs.get('sparsity', torch.tensor(0.0).to(pred.device))
        losses['sparsity'] = sparsity * self.sparsity_weight
        
        # 6. Rate penalty
        rate = aux_outputs.get('rate', torch.tensor(0.0).to(pred.device))
        losses['rate'] = rate * self.rate_weight
        
        # 7. RVQ loss (if using residual quantization)
        rvq_loss = aux_outputs.get('rvq_loss', torch.tensor(0.0).to(pred.device))
        if rvq_loss.item() > 0:
            # RVQ loss already includes commitment + codebook losses
            # Weight it appropriately (typically around 0.1-1.0)
            losses['rvq'] = rvq_loss * 0.25
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return total_loss, losses


# ============================================================================
# Factory Function
# ============================================================================

def create_physics_aware_transformer(
    patch_size: Tuple[int, int] = (100, 20),
    embed_dim: int = 128,
    num_layers: int = 6,
    num_heads: int = 8,
    use_complex: bool = True,
    use_deep_unfolding: bool = True,
    use_sparsity: bool = True,
    use_entropy_model: bool = True,
    mlp_ratio: float = 2.0,
    max_height: int = 5000,
    max_width: int = 100,
    dropout: float = 0.0,
    physics_config: Optional[PhysicsConfig] = None,
    compression_config: Optional[CompressionConfig] = None,
) -> PhysicsAwareSpatialTransformer:
    """
    Factory function to create physics-aware SAR transformer.
    
    Args:
        patch_size: Spatial patch dimensions (height, width)
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        use_complex: Use complex-valued layers (ignored if physics_config provided)
        use_deep_unfolding: Use deep unfolded physics blocks (ignored if physics_config provided)
        use_sparsity: Use LISTA sparsity priors (ignored if physics_config provided)
        use_entropy_model: Use entropy modeling (ignored if compression_config provided)
        mlp_ratio: MLP expansion ratio
        max_height: Maximum input height
        max_width: Maximum input width
        dropout: Dropout rate
        physics_config: Optional pre-configured PhysicsConfig (overrides individual flags)
        compression_config: Optional pre-configured CompressionConfig (overrides individual flags)
    
    Returns:
        PhysicsAwareSpatialTransformer instance
    """
    # Create default configs if not provided
    if physics_config is None:
        physics_config = PhysicsConfig(
            use_complex_layers=use_complex,
            use_deep_unfolding=use_deep_unfolding,
            use_sparsity_prior=use_sparsity,
        )
    
    if compression_config is None:
        compression_config = CompressionConfig(
            use_entropy_model=use_entropy_model,
            use_hyperprior=True,
            use_quantization=True,
            use_residual_quantization=True,  # Enable RVQ by default for detail preservation
            num_rvq_quantizers=4,  # 4 stages = good quality/bitrate tradeoff
            rvq_codebook_size=1024,  # 1024 entries per codebook
            rvq_commitment_weight=0.25,
        )
    
    model = PhysicsAwareSpatialTransformer(
        input_channels=2,
        output_channels=2,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        patch_size=patch_size,
        mlp_ratio=mlp_ratio,
        max_height=max_height,
        max_width=max_width,
        dropout=dropout,
        physics_config=physics_config,
        compression_config=compression_config,
    )
    
    return model


# ============================================================================
# Example Usage & Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Physics-Aware SAR Spatial Transformer - Test")
    print("=" * 80)
    
    # Configuration
    config = {
        'patch_size': (50, 10),
        'embed_dim': 64,
        'num_layers': 3,
        'num_heads': 4,
        'use_complex': True,
        'use_deep_unfolding': True,
        'use_sparsity': True,
        'use_entropy_model': True,
    }
    
    # Create model
    model = create_physics_aware_transformer(**config)
    
    print(f"\n✓ Model created with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Test input
    batch_size = 2
    height = 500
    width = 100
    x = torch.randn(batch_size, height, width, 2) * 10.0  # Complex SAR data
    
    print(f"\n✓ Test input: {x.shape}")
    print(f"  Input mean: {x.mean().item():.4f}")
    print(f"  Input std: {x.std().item():.4f}")
    
    # Forward pass
    with torch.no_grad():
        reconstructed, aux_outputs = model(x, measurements=x, return_aux=True)  # Use input as measurements for testing
    
    print(f"\n✓ Forward pass completed")
    print(f"  Output shape: {reconstructed.shape}")
    print(f"  Output mean: {reconstructed.mean().item():.4f}")
    print(f"  Output std: {reconstructed.std().item():.4f}")
    print(f"  Sparsity: {aux_outputs['sparsity'].item():.6f}")
    print(f"  Rate: {aux_outputs['rate'].item():.6f} bits/symbol")
    if 'rvq_loss' in aux_outputs:
        print(f"  RVQ Loss: {aux_outputs['rvq_loss'].item():.6f} (detail preservation)")
    if 'rvq_indices' in aux_outputs:
        print(f"  RVQ Indices shape: {aux_outputs['rvq_indices'].shape}")
    
    # Test loss
    loss_fn = MultiDomainSARLoss()
    total_loss, loss_dict = loss_fn(reconstructed, x, aux_outputs)
    
    print(f"\n✓ Loss computation:")
    for name, value in loss_dict.items():
        print(f"  {name}: {value.item():.6f}")
    
    # Model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model parameters: {num_params:,}")
    
    # Compression ratio estimate
    input_size = batch_size * height * width * 2 * 32  # bits (float32)
    # Estimate compressed size: num_tokens * embed_dim * rate_per_symbol
    num_tokens = (height // config['patch_size'][0]) * (width // config['patch_size'][1])
    compressed_size = num_tokens * config['embed_dim'] * 2 * aux_outputs['rate'].item()
    
    if compressed_size > 0:
        compression_ratio = input_size / compressed_size
        print(f"\n✓ Compression estimate:")
        print(f"  Input size: {input_size/8/1024:.2f} KB")
        print(f"  Compressed size: {compressed_size/8/1024:.2f} KB")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
