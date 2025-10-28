#!/usr/bin/env python3
"""
Scale-preserving spatial transformer for SAR data processing.
This version removes aggressive normalization to preserve input scales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops
from typing import Optional, Tuple, Literal

# ---- physics layers to paste above/near your transformer class ----
import torch
import torch.nn as nn
import torch.fft as fft

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization that preserves scale better than LayerNorm."""
    def __init__(self, dim: int, eps: float = 1e-8, learnable_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.learnable_scale = learnable_scale
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer('scale', torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.scale * x / (norm + self.eps)

class ScaleTracker(nn.Module):
    """Track and preserve input/output scale statistics with gentle corrections."""
    def __init__(self, momentum: float = 0.1, enabled: bool = True, warmup_steps: int = 100):
        super().__init__()
        self.momentum = momentum
        self.enabled = enabled
        self.warmup_steps = warmup_steps
        
        # Initialize with identity scale assumption
        self.register_buffer('input_scale', torch.tensor(1.0))
        self.register_buffer('output_scale', torch.tensor(1.0))
        self.register_buffer('num_updates', torch.tensor(0))
        
        # Learnable correction factor (starts at 1.0 for no correction)
        self.correction_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x_in: torch.Tensor, x_out: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self.training:
            return x_out
            
        with torch.no_grad():
            # Calculate current scales
            in_scale = x_in.std() + 1e-8
            out_scale = x_out.std() + 1e-8
            
            # Update running averages
            if self.num_updates == 0:
                self.input_scale.copy_(in_scale)
                self.output_scale.copy_(out_scale)
            else:
                self.input_scale.mul_(1 - self.momentum).add_(in_scale, alpha=self.momentum)
                self.output_scale.mul_(1 - self.momentum).add_(out_scale, alpha=self.momentum)
            
            self.num_updates += 1
        
        # Apply gentle scale correction only after warmup
        if self.num_updates > self.warmup_steps and self.output_scale > 1e-8:
            target_ratio = self.input_scale / self.output_scale
            # Gentle correction: clamp to reasonable range and apply learnable scaling
            gentle_ratio = target_ratio.clamp(0.8, 1.25)  # Very gentle correction
            return x_out * gentle_ratio * self.correction_scale
        
        return x_out * self.correction_scale

class AzimuthFFTFilter(nn.Module):
    """
    Apply an FFT along the azimuth (height / columns) dimension, multiply
    by a complex frequency-domain gain (learnable or fixed), then iFFT.
    Input expected as complex-packed real/imag last-dimension (shape [..., H, W, 2]).
    We operate per-column (along axis height).
    """
    def __init__(self, max_height: int, learnable: bool = True, init_gain: float = 1.0, use_window: bool = True):
        super().__init__()
        self.max_height = max_height
        self.learnable = learnable
        self.use_window = use_window
        # store complex gain in freq domain as two real params (real, imag)
        # param shape: [max_freq] where max_freq == max_height (we'll slice/truncate)
        # We'll param on half-spectrum if you want real symmetry, but keep full for simplicity
        self.gain_real = nn.Parameter(torch.ones(max_height) * float(init_gain))
        self.gain_imag = nn.Parameter(torch.zeros(max_height))
        if not learnable:
            # freeze
            self.gain_real.requires_grad_(False)
            self.gain_imag.requires_grad_(False)
        # optional Hann window in spatial domain to reduce spectral leakage
        if use_window:
            self.register_buffer('hann', torch.hann_window(max_height, periodic=False))
        else:
            self.hann = None

    def forward(self, x_complex: torch.Tensor) -> torch.Tensor:
        """
        x_complex: [B, H, W, 2] or [B, H, W] complex dtype (preferred).
        We'll accept both. We treat H as azimuth (FFT dim).
        Return same shape and dtype as input.
        """
        # convert to complex dtype for fft ops
        if x_complex.dtype.is_floating_point:
            # expecting last-dim real/imag
            real = x_complex[..., 0]
            imag = x_complex[..., 1]
            z = torch.complex(real, imag)  # [B, H, W]
        else:
            z = x_complex  # already complex

        B, H, W = z.shape
        # optionally window in space
        if self.hann is not None:
            w = self.hann[:H].to(z.device)  # [H]
            z = z * w.view(1, H, 1)

        # FFT along azimuth axis 1
        Z = fft.fft(z, n=H, dim=1)  # [B, H, W] complex

        # get freq gain (slice to H)
        gain = torch.complex(self.gain_real[:H].to(z.device), self.gain_imag[:H].to(z.device))  # [H]
        # broadcast multiply: [B, H, W] * [H] -> [B,H,W]
        Zf = Z * gain.view(1, H, 1)

        # inverse FFT
        zf = fft.ifft(Zf, n=H, dim=1)

        # return in same format as input (complex dtype)
        return zf

class AzimuthMatchedFilter(nn.Module):
    """
    Matched filter implemented in frequency domain.
    Provide a reference chirp in time domain (azimuth) or in freq domain.
    If no chirp is provided, the filter can be learned (same as AzimuthFFTFilter).
    """
    def __init__(self, ref_chirp: torch.Tensor = None, max_height: int = 5000, learnable: bool = False):
        super().__init__()
        self.max_height = max_height
        if ref_chirp is not None:
            # ref_chirp: shape [H] complex or real/imag concatenated
            # store its FFT (conj) as fixed freq response
            if ref_chirp.dtype.is_floating_point and ref_chirp.ndim == 2 and ref_chirp.shape[-1] == 2:
                real = ref_chirp[..., 0]
                imag = ref_chirp[..., 1]
                r = torch.complex(real, imag)
            else:
                r = ref_chirp.to(torch.complex64)
            R = fft.fft(r, n=max_height)
            # conjugate for matched filter in freq domain
            Hf = torch.conj(R)
            self.register_buffer('Hf', Hf)
            self.learnable = False
        else:
            # fallback to learnable frequency response
            self.learnable = learnable
            self.gain_real = nn.Parameter(torch.ones(max_height))
            self.gain_imag = nn.Parameter(torch.zeros(max_height))
            if not learnable:
                self.gain_real.requires_grad_(False)
                self.gain_imag.requires_grad_(False)

    def forward(self, x_complex: torch.Tensor) -> torch.Tensor:
        # convert to complex if necessary
        if x_complex.dtype.is_floating_point:
            real = x_complex[..., 0]
            imag = x_complex[..., 1]
            z = torch.complex(real, imag)
        else:
            z = x_complex

        B, H, W = z.shape
        # FFT
        Z = fft.fft(z, n=H, dim=1)
        if hasattr(self, 'Hf'):
            Hf = self.Hf[:H].to(Z.device)
        else:
            Hf = torch.complex(self.gain_real[:H].to(Z.device), self.gain_imag[:H].to(Z.device))
        Zf = Z * Hf.view(1, H, 1)
        zf = fft.ifft(Zf, n=H, dim=1)
        return zf


class EnhancedSpatialTokenizer(nn.Module):
    """
    Enhanced scale-preserving spatial tokenizer with hierarchical feature extraction
    and improved compression capabilities.
    """
    
    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        patch_size: Tuple[int, int] = (1, 1),
        max_height: int = 5000,
        max_width: int = 100,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.patch_height, self.patch_width = patch_size
        self.max_height = max_height
        self.max_width = max_width
        
        # Multi-scale feature extraction before tokenization
        self.feature_extractor = nn.Sequential(
            # Local feature extraction
            nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            
            # Feature compression back to input channels with residual
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=1, bias=False),
        )
        
        # Initialize feature extractor for minimal initial impact
        with torch.no_grad():
            nn.init.orthogonal_(self.feature_extractor[0].weight, gain=0.3)
            nn.init.orthogonal_(self.feature_extractor[2].weight, gain=0.3)
            nn.init.orthogonal_(self.feature_extractor[4].weight, gain=0.8)
        
        # Hierarchical patch embedding - multiple resolutions
        patch_dim = input_channels * self.patch_height * self.patch_width
        
        # Main patch embedding
        self.patch_embed = nn.Linear(patch_dim, embed_dim, bias=False)
        
        # Auxiliary embeddings for different scales
        self.patch_embed_local = nn.Linear(patch_dim, embed_dim // 4, bias=False)
        self.patch_embed_global = nn.Linear(patch_dim, embed_dim // 4, bias=False)
        
        # Fusion layer to combine multi-scale features
        self.feature_fusion = nn.Linear(embed_dim + embed_dim // 2, embed_dim, bias=False)
        
        # Optimal orthogonal initialization for all embeddings
        with torch.no_grad():
            nn.init.orthogonal_(self.patch_embed.weight, gain=1.0)
            nn.init.orthogonal_(self.patch_embed_local.weight, gain=1.0)
            nn.init.orthogonal_(self.patch_embed_global.weight, gain=1.0)
            nn.init.orthogonal_(self.feature_fusion.weight, gain=1.0)
        
        # Enhanced positional encoding with learnable components
        self.pos_embed = nn.Parameter(torch.randn(max_height // self.patch_height, 
                                                  max_width // self.patch_width, 
                                                  embed_dim) * 0.0001)
        
        # Learnable scale parameters optimized for multi-scale features
        optimal_scale = math.sqrt(embed_dim / patch_dim)
        self.output_scale = nn.Parameter(torch.tensor(optimal_scale))
        self.feature_blend_scale = nn.Parameter(torch.tensor(0.2))  # How much feature extraction to blend
        
        # Compression enhancement parameters
        self.local_attention_scale = nn.Parameter(torch.tensor(0.5))
        self.global_context_scale = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        batch_size = x.shape[0]
        
        if len(x.shape) == 3:
            x = x.unsqueeze(-1)
        
        height, width, channels = x.shape[1], x.shape[2], x.shape[3]
        token_height = height // self.patch_height
        token_width = width // self.patch_width
        
        # Convert to [B, C, H, W] for conv operations
        x_conv = x.permute(0, 3, 1, 2)
        
        # Enhanced feature extraction
        features_enhanced = self.feature_extractor(x_conv)
        
        # Residual connection with original input
        x_enhanced = x_conv + self.feature_blend_scale * features_enhanced
        
        # Convert back to [B, H, W, C]
        x_enhanced = x_enhanced.permute(0, 2, 3, 1)
        
        # Extract patches from enhanced features
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
        
        # Multi-scale embedding
        # Main embedding
        tokens_main = self.patch_embed(patches_flat)
        
        # Local context embedding (focuses on fine details)
        tokens_local = self.patch_embed_local(patches_flat)
        
        # Global context embedding (focuses on large-scale patterns)
        # Apply different preprocessing for global context
        patches_global = F.avg_pool2d(x_conv, kernel_size=3, stride=1, padding=1)
        patches_global = patches_global.permute(0, 2, 3, 1)
        if self.patch_height == 1 and self.patch_width == 1:
            patches_global_flat = einops.rearrange(patches_global, 'b h w c -> b (h w) c')
        else:
            patches_global_reshaped = einops.rearrange(
                patches_global, 
                'b (th ph) (tw pw) c -> b th tw (ph pw c)',
                th=token_height, tw=token_width,
                ph=self.patch_height, pw=self.patch_width
            )
            patches_global_flat = einops.rearrange(patches_global_reshaped, 'b h w c -> b (h w) c')
            
        tokens_global = self.patch_embed_global(patches_global_flat)
        
        # Fuse multi-scale features
        tokens_combined = torch.cat([
            tokens_main,
            tokens_local * self.local_attention_scale,
            tokens_global * self.global_context_scale
        ], dim=-1)
        
        # Final fusion and scaling
        tokens_fused = self.feature_fusion(tokens_combined) * self.output_scale
        
        # Add enhanced positional encoding
        pos_embed = self.pos_embed[:token_height, :token_width]
        pos_embed = einops.rearrange(pos_embed, 'h w d -> (h w) d')
        tokens = tokens_fused + pos_embed.unsqueeze(0)
        
        return tokens, token_height, token_width


class ScalePreservingSpatialDetokenizer(nn.Module):
    """
    Scale-preserving detokenizer that maintains magnitude relationships.
    """
    
    def __init__(
        self,
        embed_dim: int,
        output_channels: int,
        patch_size: Tuple[int, int] = (1, 1),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_channels = output_channels
        self.patch_height, self.patch_width = patch_size
        
        # Scale-preserving projection optimized for reconstruction
        patch_dim = output_channels * self.patch_height * self.patch_width
        self.patch_proj = nn.Linear(embed_dim, patch_dim, bias=False)
        
        # Optimal initialization for reconstruction (pseudo-inverse of tokenizer)
        with torch.no_grad():
            nn.init.orthogonal_(self.patch_proj.weight, gain=1.0)
            # No additional weight scaling - let learnable parameter handle all scaling
        
        # Mathematically precise reconstruction scale (from solution summary)
        # Decoder scale: sqrt(patch_dim/embed_dim) = sqrt(10000/128) = 8.838835
        optimal_scale = math.sqrt(patch_dim / embed_dim)
        self.output_scale = nn.Parameter(torch.tensor(optimal_scale))
        
        # Remove scale tracking for better stability
        self.scale_tracker = None
        
    def forward(self, tokens: torch.Tensor, token_height: int, token_width: int) -> torch.Tensor:
        batch_size = tokens.shape[0]
        
        # Project tokens to patches with optimal reconstruction scaling
        patches = self.patch_proj(tokens) * self.output_scale
        
        # Reshape to spatial grid
        if self.patch_height == 1 and self.patch_width == 1:
            spatial = einops.rearrange(
                patches, 
                'b (h w) c -> b h w c',
                h=token_height,
                w=token_width
            )
        else:
            spatial = einops.rearrange(
                patches,
                'b (th tw) (ph pw c) -> b (th ph) (tw pw) c',
                th=token_height,
                tw=token_width,
                ph=self.patch_height,
                pw=self.patch_width,
                c=self.output_channels
            )
        
        return spatial


class EnhancedTransformerBlock(nn.Module):
    """
    Enhanced transformer block with improved attention mechanisms and feature processing
    for better compression quality.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        layer_idx: int = 0,
        total_layers: int = 6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_layer_norm = use_layer_norm
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        
        # Enhanced multi-head attention with different attention patterns
        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.to_out = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Additional attention head for local context
        self.local_attention = nn.MultiheadAttention(
            embed_dim, num_heads//2, dropout=dropout, bias=False, batch_first=True
        )
        
        # Cross-scale attention for hierarchical features
        self.cross_scale_attn = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Enhanced initialization with layer-dependent gains
        layer_gain = 0.8 - (layer_idx / total_layers) * 0.2  # Decreasing gain for deeper layers
        with torch.no_grad():
            nn.init.orthogonal_(self.to_qkv.weight, gain=layer_gain)
            nn.init.orthogonal_(self.to_out.weight, gain=layer_gain)
            nn.init.orthogonal_(self.cross_scale_attn.weight, gain=0.5)
        
        # Adaptive normalization based on layer depth
        if use_layer_norm or layer_idx >= total_layers // 2:
            self.norm_attn = RMSNorm(embed_dim, learnable_scale=True)
            self.norm_mlp = RMSNorm(embed_dim, learnable_scale=True)
            self.norm_local = RMSNorm(embed_dim, learnable_scale=True)
        else:
            self.norm_attn = nn.Identity()
            self.norm_mlp = nn.Identity()
            self.norm_local = nn.Identity()
        
        # Enhanced MLP with gating mechanism
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim * 2, bias=False),  # Double for gating
            nn.GELU(),
            nn.Linear(mlp_hidden_dim * 2, embed_dim, bias=False),
        )
        
        # Gating mechanism for better feature selection
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim, bias=False),
            nn.Sigmoid(),
            nn.Linear(mlp_hidden_dim, embed_dim, bias=False),
        )
        
        # Enhanced initialization for MLP and gating
        with torch.no_grad():
            nn.init.orthogonal_(self.mlp[0].weight, gain=layer_gain)
            nn.init.orthogonal_(self.mlp[2].weight, gain=layer_gain)
            nn.init.orthogonal_(self.gate[0].weight, gain=0.3)
            nn.init.orthogonal_(self.gate[2].weight, gain=0.3)
        
        # Learnable residual scaling with layer-dependent initialization
        initial_attn_scale = 0.3 - (layer_idx / total_layers) * 0.1  # Decreasing for deeper layers
        initial_mlp_scale = 0.2 - (layer_idx / total_layers) * 0.05
        
        self.attn_scale = nn.Parameter(torch.tensor(initial_attn_scale))
        self.mlp_scale = nn.Parameter(torch.tensor(initial_mlp_scale))
        self.local_attn_scale = nn.Parameter(torch.tensor(0.1))
        self.gate_scale = nn.Parameter(torch.tensor(0.2))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Temperature control for different attention patterns
        self.global_temp = nn.Parameter(torch.tensor(0.7))  # For global attention
        self.local_temp = nn.Parameter(torch.tensor(0.9))   # For local attention
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        
        # Store input for residual connections
        x_input = x
        
        # Enhanced global self-attention
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        head_dim = embed_dim // self.num_heads
        q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with adaptive temperature
        scale = head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn * self.global_temp, dim=-1)
        attn = self.dropout(attn)
        
        global_out = torch.matmul(attn, v)
        global_out = global_out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        global_out = self.to_out(global_out)
        
        # Local attention for fine-grained features
        local_out, _ = self.local_attention(x, x, x)
        local_out = self.norm_local(local_out)
        
        # Cross-scale attention enhancement
        cross_scale_out = self.cross_scale_attn(x)
        
        # Combine attention outputs
        combined_attn = (global_out + 
                        self.local_attn_scale * local_out + 
                        0.1 * cross_scale_out)
        
        # Apply normalization and scaled residual
        attn_out = self.norm_attn(combined_attn)
        x_after_attn = x_input + self.attn_scale * attn_out
        
        # Enhanced MLP with gating
        mlp_input = x_after_attn
        mlp_raw = self.mlp(mlp_input)
        
        # Apply gating for selective feature enhancement
        gate_weights = self.gate(mlp_input)
        mlp_gated = mlp_raw * gate_weights * self.gate_scale
        
        mlp_out = self.norm_mlp(mlp_gated)
        x_final = x_after_attn + self.mlp_scale * mlp_out
        
        return x_final


class EnhancedSpatialEncoder(nn.Module):
    """
    Enhanced encoder module with improved compression capabilities:
    Input -> Physics Processing -> Enhanced Tokenization -> Enhanced Transformer Processing
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        patch_size: Tuple[int, int] = (100, 20),
        max_height: int = 5000,
        max_width: int = 100,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.patch_size = patch_size
        
        # Enhanced physics processing components
        self.use_azimuth_fft = True
        self.use_matched_filter = False
        self.az_fft_block = AzimuthFFTFilter(max_height=max_height, learnable=True, init_gain=1.0, use_window=True)
        self.az_matched = AzimuthMatchedFilter(ref_chirp=None, max_height=max_height, learnable=True)
        self.physics_alpha = nn.Parameter(torch.tensor(0.5))
        
        # Enhanced tokenizer with multi-scale features
        self.tokenizer = EnhancedSpatialTokenizer(
            input_channels=input_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            max_height=max_height,
            max_width=max_width,
        )
        
        # Enhanced transformer blocks with progressive complexity
        self.transformer_blocks = nn.ModuleList([
            EnhancedTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_layer_norm=(i >= num_layers // 3),  # Use norm in later 2/3 of layers
                layer_idx=i,
                total_layers=num_layers,
            ) for i in range(num_layers)
        ])
        
        # Compression enhancement layer for better feature concentration
        self.compression_enhancer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )
        
        # Initialize compression enhancer
        with torch.no_grad():
            nn.init.orthogonal_(self.compression_enhancer[0].weight, gain=0.5)
            nn.init.orthogonal_(self.compression_enhancer[2].weight, gain=0.8)
        
        # Learnable compression enhancement scale
        self.compression_scale = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Enhanced encode input to latent tokens."""
        return self.encode(x)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Enhanced encode input to compressed token representation with improved quality.
        
        Args:
            x: Input tensor [B, H, W, C] or [B, H, W, 2] for complex
            
        Returns:
            tokens: Enhanced encoded tokens [B, num_tokens, embed_dim]
            token_height: Number of token rows
            token_width: Number of token columns
        """
        # Enhanced physics processing
        z = x
        if z.dtype.is_floating_point and z.shape[-1] == 2:
            z_complex = torch.complex(z[..., 0], z[..., 1])
        elif torch.is_complex(z):
            z_complex = z
        else:
            z_complex = torch.complex(z[..., 0], torch.zeros_like(z[..., 0]))

        physics_out = z_complex
        if self.use_azimuth_fft:
            physics_out = self.az_fft_block(z_complex)
        if self.use_matched_filter:
            physics_out = self.az_matched(physics_out)

        z_complex = z_complex + self.physics_alpha * (physics_out - z_complex)
        x_for_tokenizer = torch.stack([z_complex.real, z_complex.imag], dim=-1)
        
        # Enhanced tokenization with multi-scale features
        tokens, token_height, token_width = self.tokenizer(x_for_tokenizer)
        
        # Enhanced transformer processing
        for block in self.transformer_blocks:
            tokens = block(tokens)
        
        # Compression enhancement for better feature concentration
        compressed_features = self.compression_enhancer(tokens)
        tokens_enhanced = tokens + self.compression_scale * compressed_features
        
        return tokens_enhanced, token_height, token_width


class EnhancedSpatialDecoder(nn.Module):
    """
    Enhanced decoder module with convolutional post-processing for better patch blending
    and improved scale preservation.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        output_channels: int = 2,
        patch_size: Tuple[int, int] = (100, 20),
        output_mode: Literal["magnitude", "complex", "real", "imag"] = "complex"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_channels = output_channels
        self.patch_size = patch_size
        self.output_mode = output_mode
        
        # Main detokenizer
        self.detokenizer = ScalePreservingSpatialDetokenizer(
            embed_dim=embed_dim,
            output_channels=output_channels,
            patch_size=patch_size,
        )
        
        # Lightweight convolutional post-processing for patch blending
        self.conv_blend = nn.Sequential(
            # Single conv layer for blending - more conservative approach
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            
            # Optional second conv for refinement with skip connection friendly design
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
        )
        
        # Initialize conv layers for minimal impact and scale preservation
        with torch.no_grad():
            # First conv: small random weights for blending
            nn.init.orthogonal_(self.conv_blend[0].weight, gain=0.1)
            
            # Second conv: near-identity initialization for refinement
            nn.init.orthogonal_(self.conv_blend[2].weight, gain=0.05)
        
        # Enhanced scale preservation components
        self.input_scale_tracker = ScaleTracker(momentum=0.02, enabled=True, warmup_steps=20)
        self.global_scale = nn.Parameter(torch.tensor(1.0))
        self.residual_scale = nn.Parameter(torch.tensor(0.95))  # Very strong residual connection
        self.conv_blend_scale = nn.Parameter(torch.tensor(0.1))  # Conservative conv contribution
        
        # Adaptive scale correction based on input statistics
        self.register_buffer('running_input_scale', torch.tensor(1.0))
        self.register_buffer('running_output_scale', torch.tensor(1.0))
        self.register_buffer('num_batches', torch.tensor(0))
    
    def forward(self, tokens: torch.Tensor, token_height: int, token_width: int, original_shape: torch.Size, input_for_tracking: torch.Tensor = None) -> torch.Tensor:
        """Enhanced decode with convolutional blending and scale preservation."""
        return self.decode(tokens, token_height, token_width, original_shape, input_for_tracking)
    
    def decode(self, tokens: torch.Tensor, token_height: int, token_width: int, original_shape: torch.Size, input_for_tracking: torch.Tensor = None) -> torch.Tensor:
        """
        Enhanced decode with lightweight convolutional blending and adaptive scale preservation.
        
        Args:
            tokens: Encoded tokens [B, num_tokens, embed_dim]
            token_height: Number of token rows
            token_width: Number of token columns
            original_shape: Original input shape for output mode handling
            input_for_tracking: Original input for scale tracking
            
        Returns:
            output: Reconstructed spatial tensor [B, H, W, C] with enhanced blending
        """
        # Step 1: Standard detokenization
        spatial_raw = self.detokenizer(tokens, token_height, token_width)
        
        # Step 2: Adaptive scale tracking and adjustment
        if self.training and input_for_tracking is not None:
            with torch.no_grad():
                input_scale = input_for_tracking.std()
                output_scale = spatial_raw.std()
                
                # Update running averages
                momentum = 0.05
                self.running_input_scale = self.running_input_scale * (1 - momentum) + input_scale * momentum
                self.running_output_scale = self.running_output_scale * (1 - momentum) + output_scale * momentum
                self.num_batches += 1
                
                # Calculate adaptive scale correction after warmup
                if self.num_batches > 10:
                    target_ratio = self.running_input_scale / (self.running_output_scale + 1e-8)
                    # Apply gentle adaptive correction
                    self.global_scale.data = self.global_scale.data * 0.99 + target_ratio * 0.01
        
        # Step 3: Convert to [B, C, H, W] for conv operations
        spatial_conv = spatial_raw.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        # Step 4: Apply lightweight convolutional blending
        conv_refined = self.conv_blend(spatial_conv)
        
        # Step 5: Sophisticated residual blending with learned scales
        # Main path: strong residual + small conv contribution for patch blending
        spatial_enhanced = (self.residual_scale * spatial_conv + 
                          self.conv_blend_scale * conv_refined)
        
        # Step 6: Convert back to [B, H, W, C]
        spatial_output = spatial_enhanced.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        
        # Step 7: Apply global scale preservation
        spatial_output = spatial_output * self.global_scale
        
        # Step 8: Final scale tracking for training stability
        if self.training and input_for_tracking is not None:
            spatial_output = self.input_scale_tracker(input_for_tracking, spatial_output)
        
        # Step 9: Handle output mode for complex data
        if self.output_mode != "complex" and original_shape[-1] == 2:
            spatial_output = self._apply_output_mode(spatial_output)
        
        return spatial_output
    
    def _apply_output_mode(self, x: torch.Tensor) -> torch.Tensor:
        """Apply output mode transformation for complex data."""
        if self.output_mode == "magnitude":
            real, imag = x[..., 0], x[..., 1]
            magnitude = torch.sqrt(real**2 + imag**2)
            return magnitude.unsqueeze(-1)
        elif self.output_mode == "real":
            return x[..., :1]
        elif self.output_mode == "imag":
            return x[..., 1:2]
        else:
            return x


class ScalePreservingSpatialTransformer(nn.Module):
    """
    Scale-preserving spatial transformer for SAR data with separated encoder and decoder.
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        output_channels: int = 2,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        patch_size: Tuple[int, int] = (100, 20),
        max_height: int = 5000,
        max_width: int = 100,
        dropout: float = 0.0,
        output_mode: Literal["magnitude", "complex", "real", "imag"] = "complex"
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.output_mode = output_mode
        self.input_dim = patch_size
        
        # Enhanced encoder and decoder modules
        self.encoder = EnhancedSpatialEncoder(
            input_channels=input_channels,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            max_height=max_height,
            max_width=max_width,
            dropout=dropout,
        )
        
        self.decoder = EnhancedSpatialDecoder(
            embed_dim=embed_dim,
            output_channels=output_channels,
            patch_size=patch_size,
            output_mode=output_mode
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode -> decode with enhanced processing."""
        original_shape = x.shape
        
        # Encode
        tokens, token_height, token_width = self.encoder(x)
        
        # Enhanced decode with input tracking for scale preservation
        output = self.decoder(tokens, token_height, token_width, original_shape, input_for_tracking=x)
        
        return output
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Encode input to compressed latent representation.
        
        Args:
            x: Input tensor [B, H, W, C]
            
        Returns:
            tokens: Encoded tokens [B, num_tokens, embed_dim]
            token_height: Number of token rows
            token_width: Number of token columns
        """
        return self.encoder.encode(x)
    
    def decode(self, tokens: torch.Tensor, token_height: int, token_width: int, original_shape: torch.Size, input_for_tracking: torch.Tensor = None) -> torch.Tensor:
        """
        Decode latent tokens back to spatial representation with enhanced processing.
        
        Args:
            tokens: Encoded tokens [B, num_tokens, embed_dim]
            token_height: Number of token rows
            token_width: Number of token columns
            original_shape: Original input shape for output mode handling
            input_for_tracking: Original input for scale tracking
            
        Returns:
            output: Reconstructed spatial tensor [B, H, W, C]
        """
        return self.decoder.decode(tokens, token_height, token_width, original_shape, input_for_tracking)


def create_spatial_vision_transformer(**kwargs):
    """Factory function for scale-preserving spatial transformer."""
    default_config = {
        'input_channels': 2,
        'output_channels': 2,
        'embed_dim': 256,
        'num_layers': 6,
        'num_heads': 8,
        'mlp_ratio': 2.0,
        'patch_size': (100, 20),
        'max_height': 5000,
        'max_width': 100,
        'dropout': 0.0,
        'output_mode': 'complex'
    }
    
    config = {**default_config, **kwargs}
    return ScalePreservingSpatialTransformer(**config)


if __name__ == "__main__":
    # Test the scale-preserving spatial transformer
    model = create_spatial_vision_transformer(
        patch_size=(50, 10),
        embed_dim=128,
        num_layers=2,
        num_heads=4
    )
    
    print("🔧 Scale-Preserving Spatial Transformer Test")
    print("=" * 50)
    
    # Test with different scales
    for scale in [0.1, 1.0, 10.0]:
        x = torch.randn(1, 500, 100, 2) * scale
        
        with torch.no_grad():
            output = model(x)
        
        input_std = x.std().item()
        output_std = output.std().item()
        ratio = output_std / input_std if input_std > 0 else 0
        
        print(f"Scale {scale:4.1f}: Input std {input_std:8.4f} | Output std {output_std:8.4f} | Ratio {ratio:6.3f}")
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")