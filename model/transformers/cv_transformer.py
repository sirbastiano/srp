from typing import Optional, Tuple, Dict, Any
from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from model.transformers.attend import Attend  # your flash/causal attention backend

def exists(x):
    """Check if a value is not None."""
    return x is not None

def default(x, d):
    """Return x if it exists, otherwise return default value d."""
    return x if exists(x) else d

# Complex activation and dropout classes
class ComplexActivation(nn.Module):
    """Apply activation function to both real and imaginary parts."""
    def __init__(self, activation: nn.Module):
        super().__init__()
        self.activation = activation

    def forward(self, x):
        if torch.is_complex(x):
            return torch.complex(self.activation(x.real), self.activation(x.imag))
        else:
            return self.activation(x)

class ComplexDropout(nn.Module):
    """Apply dropout to both real and imaginary parts."""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        if torch.is_complex(x):
            return torch.complex(self.dropout(x.real), self.dropout(x.imag))
        else:
            return self.dropout(x)

def modulate_with_rotation(x: Tensor, rot_emb: Tensor) -> Tensor:
    """
    Apply a complex rotation embedding to a complex64 tensor.
    
    Args:
        x: Complex tensor to rotate
        rot_emb: Rotation embedding (real angles or complex phases)
        
    Returns:
        Complex tensor with rotation applied
    """
    # Ensure complex type
    if not rot_emb.is_complex():
        # assume rot_emb real angles
        rot_emb = rot_emb.abs()
    phase = torch.cos(rot_emb) + 1j * torch.sin(rot_emb)
    return x * phase

def complex_attention_real(
    q: Tensor, k: Tensor, v: Tensor,
    attend_fn: Attend,
    mask: Optional[Tensor] = None
) -> Tensor:
    """
    Complex attention using real-channel flattening approach.
    
    Implements Equation 8 from https://arxiv.org/abs/2306.09827
    Flattens real & imaginary channels into one real attention operation.
    """
    assert q.is_complex() and k.is_complex() and v.is_complex(), "Inputs must be complex"
    # Convert to real representation [b, h, n, d, 2]
    qr, qi = torch.view_as_real(q).unbind(-1)
    kr, ki = torch.view_as_real(k).unbind(-1)
    vr, vi = torch.view_as_real(v).unbind(-1)

    # Stack so that we can compute the four MHAs in one go
    # shapes: (r=4, b, h, n, d)
    q_stack = torch.stack([qr, qr, qi, qi], dim=0)
    k_stack = torch.stack([kr, -ki, kr, -ki], dim=0)
    v_stack = torch.stack([vr, vr, vi, vi], dim=0)

    # Flatten batch dimension for Attend call
    r, b, h, n, d = q_stack.shape
    qf = rearrange(q_stack, 'r b h n d -> (r b) h n d')
    kf = rearrange(k_stack, 'r b h n d -> (r b) h n d')
    vf = rearrange(v_stack, 'r b h n d -> (r b) h n d')

    if exists(mask):
        mask = repeat(mask, 'b ... -> (r b) ...', r=r)

    # compute all four
    outf = attend_fn(qf, kf, vf, mask=mask)
    outf = rearrange(outf, '(r b) h n d -> r b h n d', r=r, b=b)

    # reassemble real & imag
    # real part = A_rr - A_ii
    # imag part = A_ri + A_ir
    Att = outf  # [4, b, h, n, d]
    rr, ri, ir, ii = Att  # unpack
    real = rr - ii
    imag = ri + ir

    stacked = torch.stack([real, imag], dim=-1)  # [b, h, n, d, 2]
    return torch.view_as_complex(stacked)


# ——— Complex Multihead Attention ———————————————————————————————————————————————————————

class ComplexMultiheadAttention(nn.Module):
    """
    Complex-valued multi-head attention module.
    
    Supports both real-channel flattening and full complex attention approaches.
    All projections are complex-valued and attention preserves complex structure.
    """
    
    def __init__(
        self,
        dim: int,
        *,
        causal: bool = False,
        dim_head: int = 32,
        heads: int = 8,
        complete_complex: bool = False,
        flash: bool = True
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.complete_complex = complete_complex
        
        inner_dim = dim_head * heads
        
        # Complex-valued projections
        self.to_q = nn.Linear(dim, inner_dim, bias=False, dtype=torch.complex64)
        self.to_k = nn.Linear(dim, inner_dim, bias=False, dtype=torch.complex64)
        self.to_v = nn.Linear(dim, inner_dim, bias=False, dtype=torch.complex64)
        self.to_out = nn.Linear(inner_dim, dim, bias=False, dtype=torch.complex64)
        
        # Attention function
        self.attend = Attend(
            causal=causal,
            flash=flash,
            dropout=0.1
        )

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        rotary_emb: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass with complex attention.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            context: Optional context for cross-attention
            mask: Optional attention mask
            rotary_emb: Optional rotary embeddings
            
        Returns:
            Attention output [batch, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        h = self.heads
        
        # Default to self-attention
        kv_input = default(context, x)
        
        # Project to q, k, v
        q = self.to_q(x)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        
        # Apply rotary embeddings if provided
        if exists(rotary_emb):
            q = modulate_with_rotation(q, rotary_emb)
            k = modulate_with_rotation(k, rotary_emb)
        
        # Compute attention
        if self.complete_complex:
            # Full complex attention (not implemented here for brevity)
            # Would require complex-valued attention computation
            out = complex_attention_real(q, k, v, self.attend, mask)
        else:
            # Real-channel flattening approach
            out = complex_attention_real(q, k, v, self.attend, mask)
        
        # Reshape back
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Final projection
        return self.to_out(out)


# ——— Complex Feed Forward ————————————————————————————————————————————————————————————

class ComplexFeedForward(nn.Module):
    """Complex-valued feed-forward network."""
    
    def __init__(
        self,
        dim: int,
        mult: int = 4,
        relu_squared: bool = True
    ):
        super().__init__()
        inner_dim = dim * mult
        
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim, dtype=torch.complex64),
            ComplexActivation(nn.ReLU() if not relu_squared else nn.ReLU()),
            nn.Linear(inner_dim, dim, dtype=torch.complex64)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ——— Complex RMS Norm —————————————————————————————————————————————————————————————————

class ComplexRMSNorm(nn.Module):
    """Complex-valued RMS normalization."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim, dtype=torch.complex64))

    def forward(self, x: Tensor) -> Tensor:
        # Compute RMS for complex numbers
        rms = torch.sqrt(torch.mean(torch.abs(x) ** 2, dim=-1, keepdim=True))
        return x / (rms + 1e-8) * self.scale


# ——— Rotary Embeddings ————————————————————————————————————————————————————————————————

class RotaryEmbedding(nn.Module):
    """Rotary position embeddings for complex attention."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len: int) -> Tensor:
        """Generate rotary embeddings for given sequence length."""
        device = self.inv_freq.device
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        return torch.complex(torch.cos(freqs), torch.sin(freqs))

    def rotate_queries_or_keys(self, x: Tensor) -> Tensor:
        """Apply rotary embeddings to queries or keys."""
        seq_len = x.shape[-2]
        rot_emb = self.forward(seq_len)
        return modulate_with_rotation(x, rot_emb)


# ——— Main Complex Transformer —————————————————————————————————————————————————————————

class ComplexTransformer(nn.Module):
    """
    Complex-Valued Transformer for SAR focusing that works with row/column sequences.
    Optimized for compressing raw SAR data and focusing it with a decoder.
    
    Args:
        input_dim: Dimensionality of input features (typically 1 for complex data)
        model_dim: Dimensionality of the transformer model (embedding size)
        num_layers: Number of layers in both encoder and decoder
        num_heads: Number of attention heads in each layer
        ff_dim: Dimensionality of the feedforward network
        dropout: Dropout rate applied in transformer layers
        mode: Processing mode ("parallel" or "autoregressive")
        max_seq_len: Maximum sequence length for positional encoding
        compression_ratio: Ratio for compressing the sequence length
        causal: Whether to use causal masking in attention
        complete_complex: Whether to use full complex attention vs real-channel flattening
        flash_attn: Whether to use flash attention
    """
    def __init__(
        self,
        input_dim: int = 1,
        model_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 256,
        dropout: float = 0.1,
        mode: str = "parallel",
        max_seq_len: int = 5000,
        compression_ratio: float = 0.1,
        causal: bool = False,
        complete_complex: bool = False,
        flash_attn: bool = True,
        verbose: bool = False
    ):
        super().__init__()
        
        # Store configuration
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.mode = mode
        self.verbose = verbose
        self.compression_ratio = compression_ratio
        self.compressed_dim = max(int(model_dim * compression_ratio), 16)
        
        # Input/output projections - complex-valued
        self.input_proj = nn.Linear(input_dim, model_dim, dtype=torch.complex64)
        self.output_proj = nn.Linear(model_dim, input_dim, dtype=torch.complex64)
        
        # Positional encoding for complex numbers
        self.register_buffer('pos_encoding', self._create_complex_pos_encoding(max_seq_len, model_dim))
        
        # Encoder layers for compression
        self.encoder_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.encoder_layers.append(nn.ModuleList([
                ComplexRMSNorm(model_dim),
                ComplexMultiheadAttention(
                    dim=model_dim,
                    causal=False,  # Encoder is not causal
                    dim_head=model_dim // num_heads,
                    heads=num_heads,
                    complete_complex=complete_complex,
                    flash=flash_attn
                ),
                ComplexRMSNorm(model_dim),
                ComplexFeedForward(
                    dim=model_dim,
                    mult=ff_dim // model_dim,
                    relu_squared=True
                )
            ]))
        
        # Compression layer - reduces sequence length
        self.compressor = nn.Sequential(
            nn.Linear(model_dim, self.compressed_dim, dtype=torch.complex64),
            ComplexActivation(nn.GELU()),
            ComplexDropout(dropout)
        )
        
        # Decoder layers for reconstruction
        self.decoder_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.decoder_layers.append(nn.ModuleList([
                ComplexRMSNorm(model_dim),
                ComplexMultiheadAttention(
                    dim=model_dim,
                    causal=causal,
                    dim_head=model_dim // num_heads,
                    heads=num_heads,
                    complete_complex=complete_complex,
                    flash=flash_attn
                ),
                ComplexRMSNorm(model_dim),
                ComplexFeedForward(
                    dim=model_dim,
                    mult=ff_dim // model_dim,
                    relu_squared=True
                )
            ]))
        
        # Learnable query embeddings for decoding
        self.register_parameter('query_embedding', 
                              nn.Parameter(torch.randn(max_seq_len, model_dim, dtype=torch.complex64) * 0.02))
        
        # Expansion layer - restores from compressed representation
        self.expander = nn.Sequential(
            nn.Linear(self.compressed_dim, model_dim, dtype=torch.complex64),
            ComplexActivation(nn.GELU()),
            ComplexDropout(dropout)
        )
        
        self.final_norm = ComplexRMSNorm(model_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _create_complex_pos_encoding(self, max_len: int, dim: int) -> torch.Tensor:
        """Create complex positional encoding."""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        
        # Create complex positional encoding
        pe_real = torch.sin(position * div_term)
        pe_imag = torch.cos(position * div_term)
        
        if dim % 2 == 1:
            # Handle odd dimensions
            pe_real = F.pad(pe_real, (0, 1))
            pe_imag = F.pad(pe_imag, (0, 1))
        
        pe = torch.complex(pe_real, pe_imag)  # [max_len, dim]
        return pe.unsqueeze(0)  # [1, max_len, dim]
    
    def _init_weights(self):
        """Initialize weights with proper scaling for complex networks."""
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.weight.dtype == torch.complex64:
                # Xavier initialization for complex weights
                fan_in = module.weight.shape[1]
                scale = math.sqrt(2.0 / fan_in)
                with torch.no_grad():
                    module.weight.real.normal_(0, scale)
                    module.weight.imag.normal_(0, scale)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def get_parameter_count(self):
        """Calculate total parameters and provide breakdown."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        breakdown = {
            "input_proj": sum(p.numel() for p in self.input_proj.parameters()),
            "output_proj": sum(p.numel() for p in self.output_proj.parameters()),
            "encoder_layers": sum(p.numel() for p in self.encoder_layers.parameters()),
            "decoder_layers": sum(p.numel() for p in self.decoder_layers.parameters()),
            "compressor": sum(p.numel() for p in self.compressor.parameters()),
            "expander": sum(p.numel() for p in self.expander.parameters()),
            "query_embedding": self.query_embedding.numel(),
            "pos_encoding": self.pos_encoding.numel(),
        }
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "breakdown": breakdown,
            "memory_mb": total_params * 8 / (1024 * 1024),  # Complex64 = 8 bytes per param
        }

    def preprocess_input(self, x):
        """
        Preprocess input to handle SAR complex data format.
        Expected: [batch_size, seq_len, input_dim] where input_dim=1 for complex
        """
        if self.verbose:
            print(f"Input shape: {x.shape}, dtype: {x.dtype}")
        
        # Convert to complex if needed
        if not torch.is_complex(x):
            if x.shape[-1] == 2:
                # Real/imag format -> complex
                x = torch.complex(x[..., 0], x[..., 1])
                x = x.unsqueeze(-1)  # Add channel dimension
            elif x.shape[-1] == 1:
                # Already single channel, assume real part only
                x = torch.complex(x[..., 0], torch.zeros_like(x[..., 0]))
                x = x.unsqueeze(-1)
        
        # Handle different input shapes
        if len(x.shape) == 4:
            # [B, seq_len, 1, channels] -> [B, seq_len, channels]
            if x.shape[2] == 1:
                x = x.squeeze(2)
            else:
                # [B, rows, cols, channels] -> [B, rows, cols*channels] (flatten spatial)
                batch_size, rows, cols, channels = x.shape
                x = x.view(batch_size, rows, cols * channels)
        
        # Ensure we have the right input dimension
        if x.shape[-1] != self.input_dim:
            x = x[..., :self.input_dim]  # Take only required dimensions
        
        return x

    def forward(self, x, y=None):
        """
        Forward pass for SAR focusing.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim] (complex)
            y: Optional target for teacher forcing (not used in parallel mode)
            
        Returns:
            Output tensor [batch_size, seq_len, input_dim] (complex)
        """
        # Preprocess input
        x = self.preprocess_input(x)
        batch_size, seq_len, _ = x.shape
        
        if self.verbose:
            print(f"Processing complex sequence of length {seq_len}")
        
        # 1. Project to model dimension
        x_proj = self.input_proj(x)  # [B, seq_len, model_dim]
        
        # 2. Add positional encoding
        pos_enc = self.pos_encoding[:, :seq_len, :]
        x_pos = x_proj + pos_enc
        
        # 3. Encode through transformer layers
        encoded = x_pos
        for norm1, attention, norm2, ffn in self.encoder_layers:
            # Pre-norm transformer block
            attn_out = attention(norm1(encoded))
            encoded = encoded + attn_out
            
            ffn_out = ffn(norm2(encoded))
            encoded = encoded + ffn_out
        
        # 4. Compress sequence (create bottleneck)
        compressed = self.compressor(encoded)  # [B, seq_len, compressed_dim]
        
        # 5. Take mean pooling to create fixed-size representation
        # This creates the "tiny embedding" you requested
        pooled = torch.mean(compressed, dim=1, keepdim=True)  # [B, 1, compressed_dim]
        
        # 6. Expand back to model dimension
        expanded = self.expander(pooled)  # [B, 1, model_dim]
        
        # 7. Decode using learned queries
        queries = self.query_embedding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)  # [B, seq_len, model_dim]
        
        # Use compressed representation as memory for decoder
        memory = expanded.expand(-1, seq_len, -1)  # [B, seq_len, model_dim]
        decoded = queries
        
        for norm1, attention, norm2, ffn in self.decoder_layers:
            # Cross-attention with compressed representation
            attn_out = attention(norm1(decoded), memory)
            decoded = decoded + attn_out
            
            ffn_out = ffn(norm2(decoded))
            decoded = decoded + ffn_out
        
        # 8. Final normalization and projection
        decoded = self.final_norm(decoded)
        output = self.output_proj(decoded)  # [B, seq_len, input_dim]
        
        if self.verbose:
            print(f"Output shape: {output.shape}")
            print(f"Compression ratio achieved: {pooled.shape[1] / seq_len:.3f}")
        
        return output