from typing import Optional, Tuple, Dict, Any
from functools import partial

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
    
    Args:
        q: Query tensor (complex)
        k: Key tensor (complex)
        v: Value tensor (complex)
        attend_fn: Attention function to use
        mask: Optional attention mask
        
    Returns:
        Complex attention output
    """
    assert q.is_complex() and k.is_complex() and v.is_complex(), "Inputs must be complex"
    # Split into real channels, flatten
    qr, qi = torch.view_as_real(q).unbind(-1)
    kr, ki = torch.view_as_real(k).unbind(-1)
    vr, vi = torch.view_as_real(v).unbind(-1)

    # stack [r, i] in feature dim: shape (..., 2*d) real
    q_ = torch.cat([qr, qi], dim=-1)
    k_ = torch.cat([kr, ki], dim=-1)
    v_ = torch.cat([vr, vi], dim=-1)

    # attend on real representations
    out = attend_fn(q_, k_, v_, mask=mask)

    # un-flatten and re-combine
    d = out.shape[-1] // 2
    outr, outi = out[..., :d], out[..., d:]
    stacked = torch.stack([outr, outi], dim=-1)
    return torch.view_as_complex(stacked)


def complex_attention_complete(
    q: Tensor, k: Tensor, v: Tensor,
    attend_fn: Attend,
    mask: Optional[Tensor] = None
) -> Tensor:
    """
    Full complex attention implementation.
    
    Implements the complete complex attention from Yang et al. (Equation 3 of https://arxiv.org/abs/1910.10202)
    Uses pre-registered buffers for reordering and sign patterns.
    Computes four different attention combinations: A_rr, A_ri, A_ir, A_ii
    
    Args:
        q: Query tensor (complex)
        k: Key tensor (complex) 
        v: Value tensor (complex)
        attend_fn: Attention function to use
        mask: Optional attention mask
        
    Returns:
        Complex attention output with proper complex arithmetic
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
    
    Args:
        dim: Input/output dimension
        causal: Whether to use causal (autoregressive) attention
        dim_head: Dimension per attention head
        heads: Number of attention heads
        complete_complex: Whether to use full complex attention (vs real-channel flattening)
        flash: Whether to use flash attention
    """
    def __init__(
        self,
        dim: int,
        *,
        causal: bool = False,
        dim_head: int = 32,
        heads: int = 8,
        complete_complex: bool = False,
        flash: bool = False
    ):
        super().__init__()
        inner = dim_head * heads

        # projectors (complex64)
        self.to_q  = nn.Linear(dim, inner, bias=False, dtype=torch.complex64)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False, dtype=torch.complex64)
        self.to_out= nn.Linear(inner, dim, bias=False, dtype=torch.complex64)

        # attention backend
        attend_fn = Attend(causal=causal, heads=heads, flash=flash)

        # choose real‑flatten or full complex
        self.attend_fn = attend_fn
        self.complete_complex = complete_complex

        # head splitting/merging
        self.split = Rearrange('b n (h d) -> b h n d',  h=heads)
        self.merge = Rearrange('b h n d -> b n (h d)')

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        rotary_emb: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass of complex multi-head attention.
        
        Args:
            x: Input tensor (complex)
            context: Optional context for cross-attention
            mask: Optional attention mask
            rotary_emb: Optional rotary position embeddings
            
        Returns:
            Attention output (complex)
        """
        # prepare q,k,v
        #print(f"Input tensor: shape={x.shape}, type={x.dtype}")
        #print(f"Context tensor: shape={context.shape if context is not None else 'None'}, type={context.dtype if context is not None else 'None'}")
        context = default(context, x)
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # split heads
        q, k, v = map(self.split, (q, k, v))

        # apply rotary if given
        if exists(rotary_emb) and rotary_emb is not None:
            q = modulate_with_rotation(q, rotary_emb)
            k = modulate_with_rotation(k, rotary_emb)

        # attend
        if self.complete_complex:
            out = complex_attention_complete(q, k, v, self.attend_fn, mask)
        else:
            out = complex_attention_real(q, k, v, self.attend_fn, mask)

        # merge & project out
        out = self.merge(out)
        return self.to_out(out)


# ——— Complex RMSNorm ————————————————————————————————————————————————————————————————

class ComplexRMSNorm(nn.Module):
    """
    Complex-valued Root Mean Square Layer Normalization.
    
    Normalizes using the magnitude of complex values while preserving phase.
    Uses learnable complex scaling parameters.
    
    Args:
        dim: Feature dimension
        eps: Small value to prevent division by zero
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # one complex weight per feature
        self.gamma = nn.Parameter(torch.ones(dim, dtype=torch.complex64))

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply complex RMSNorm.
        
        Args:
            x: Input tensor (complex) of shape [..., dim]
            
        Returns:
            Normalized tensor (complex)
        """
        # x: [..., dim] complex
        # compute RMS over |x|^2
        mag_sq = (x.abs() ** 2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mag_sq + self.eps)
        return x / rms * self.gamma


# ——— ModReLU ————————————————————————————————————————————————————————————————————————

class ModReLU(nn.Module):
    """
    Modulus ReLU activation for complex numbers.
    
    Applies ReLU to the magnitude while preserving the phase.
    Optionally squares the output (ReLU²).
    
    Args:
        bias_init: Initial value for the learnable bias
        relu_squared: Whether to square the ReLU output
    """
    def __init__(self, bias_init: float = 0.0, relu_squared: bool = False):
        super().__init__()
        self.pow = 2 if relu_squared else 1
        # scalar bias; you could also use per-dim bias of shape (dim,)
        self.bias = nn.Parameter(torch.tensor(bias_init, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply ModReLU activation.
        
        Args:
            x: Input complex tensor
            
        Returns:
            Complex tensor with ReLU applied to magnitude
        """
        # x: complex; magnitude rectified with bias
        mag = x.abs()
        clipped = F.relu(mag + self.bias) ** self.pow
        # keep phase
        phase = torch.exp(1j * torch.angle(x))
        return clipped * phase


# ——— CReLU ————————————————————————————————————————————————————————————————

class CReLU(nn.Module):
    """
    Complex ReLU activation that applies ReLU separately to real and imaginary parts.
    
    Args:
        dim: Feature dimension for per-dimension bias
    """
    def __init__(self, dim: int):
        super().__init__()
        self.relu = nn.ReLU()
        # one bias per dimension
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply CReLU activation.
        
        Args:
            x: Input complex tensor of shape [..., dim]
            
        Returns:
            Complex tensor with ReLU applied to real and imaginary parts separately
        """
        # x: complex of shape [..., dim]
        real = self.relu(x.real + self.bias)
        imag = self.relu(x.imag + self.bias)
        return torch.complex(real, imag)


# ——— Feed‑Forward ——————————————————————————————————————————————————————————————————

def ComplexFeedForward(
    dim: int,
    mult: int = 4,
    relu_squared: bool = True
) -> nn.Module:
    """
    Complex-valued feed-forward network.
    
    Two-layer MLP with ModReLU activation in between.
    All operations preserve complex structure.
    
    Args:
        dim: Input/output dimension
        mult: Hidden dimension multiplier
        relu_squared: Whether to use squared ReLU in ModReLU
        
    Returns:
        Sequential module implementing complex feed-forward network
    """
    inner = dim * mult
    return nn.Sequential(
        nn.Linear(dim, inner, dtype=torch.complex64),
        ModReLU(relu_squared=relu_squared),
        nn.Linear(inner, dim, dtype=torch.complex64)
    )


# ——— Rotary Embedding ————————————————————————————————————————————————————————————————

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for transformers.
    
    Generates sinusoidal position encodings that can be applied as rotations
    to query and key vectors in attention.
    
    Args:
        dim_head: Dimension of each attention head
        base: Base frequency for the sinusoidal embeddings
    """
    def __init__(self, dim_head: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim_head).float() / dim_head))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len: int) -> Tensor:
        """
        Generate rotary embeddings for given sequence length.
        
        Args:
            seq_len: Length of the sequence
            
        Returns:
            Real-valued rotation angles of shape [seq_len, dim_head]
        """
        device = next(iter(self.parameters())).device if list(self.parameters()) else 'cpu'
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = einsum('i,j->ij', t, self.inv_freq)
        return freqs  # real angles; use modulate_with_rotation to apply


# ——— Complex Transformer ——————————————————————————————————————————————————————————————

class ComplexTransformer(nn.Module):
    """
    Complex-Valued Transformer for SAR data processing.
    Processes vertical patches as sequence tokens with positional encoding.
    
    Args:
        dim: Model dimension
        depth: Number of transformer layers
        num_tokens: Vocabulary size for token embeddings (None for continuous inputs)
        causal: Whether to use causal masking in attention
        dim_head: Dimension per attention head
        heads: Number of attention heads
        ff_mult: Feed-forward hidden dimension multiplier
        relu_squared: Whether to use squared ReLU in activations
        complete_complex: Whether to use full complex attention vs real-channel flattening
        rotary: Whether to use rotary position embeddings
        flash_attn: Whether to use flash attention
        use_data_positions: Whether to use positional encoding from input data
        pos_encoding_type: Type of positional encoding ('complex', 'concat', 'add')
    """
    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        num_tokens: Optional[int] = None,
        causal: bool = False,
        dim_head: int = 32,
        heads: int = 8,
        ff_mult: int = 4,
        relu_squared: bool = True,
        complete_complex: bool = False,
        rotary: bool = False,
        flash_attn: bool = True,
        use_data_positions: bool = True,
        pos_encoding_type: str = 'concat',
        verbose: bool = False
    ):
        super().__init__()
        self.verbose = verbose
        self.use_data_positions = use_data_positions
        self.pos_encoding_type = pos_encoding_type
        self.token_dim = dim
        self.seq_embed = None
        if num_tokens is not None:
            # token‑embedding table
            self.seq_embed = nn.Parameter(
                torch.randn(num_tokens, dim, dtype=torch.complex64)
            )
        if self.verbose:
            print(f"Model dimension: {dim}, Depth: {depth}, Num tokens: {num_tokens}")
        # Use rotary embeddings only if not using data positions
        self.rotary = RotaryEmbedding(dim_head) if (rotary and not use_data_positions) else None
        
        # Positional encoding projection layers
        if use_data_positions:
            if pos_encoding_type == 'complex':
                # Project 2D positions to complex position encoding
                self.pos_proj = nn.Linear(dim, dim, dtype=torch.complex64)  # 1 complex input -> dim complex output
            elif pos_encoding_type == 'concat':
                # Concatenate positions and project
                self.pos_proj = nn.Linear(2 * dim, dim, dtype=torch.complex64)  # 2 real inputs -> dim complex output
            elif pos_encoding_type == 'add':
                # Separate projection for each position dimension
                self.pos_proj_h = nn.Linear(dim, dim, dtype=torch.complex64)
                self.pos_proj_v = nn.Linear(dim, dim, dtype=torch.complex64)
            else:
                raise ValueError(f"Unknown pos_encoding_type: {pos_encoding_type}")
        else:
            self.pos_proj = None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ComplexRMSNorm(dim),
                ComplexMultiheadAttention(
                    dim=dim,
                    causal=causal,
                    dim_head=dim_head,
                    heads=heads,
                    complete_complex=complete_complex,
                    flash=flash_attn
                ),
                ComplexRMSNorm(dim),
                ComplexFeedForward(
                    dim=dim,
                    mult=ff_mult,
                    relu_squared=relu_squared
                )
            ]))

        self.final_norm = ComplexRMSNorm(dim)
        self.to_logits = None
        if num_tokens is not None:
            self.to_logits = nn.Linear(dim, num_tokens, dtype=torch.complex64)

    def preprocess_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Preprocess input tensor and extract positional information.
        Ensures output is always (batch_size, seq_dim, token_dim) for transformer.
        Stores enough metadata to restore original shape in postprocessing.
        """
        original_shape = x.shape
        metadata: Dict[str, Any] = {'original_shape': original_shape}

        # If input is 4D, try to infer which axes are sequence, token, and channel
        if self.verbose:
            print(f"Preprocessing input tensor shape: {x.shape}, dtype: {x.dtype}")
        if len(x.shape) == 4:
            batch_size, d1, d2, d3 = x.shape
            if d1 > d2:
                seq_dim, token_dim = d1, d2
                x = x.transpose(1, 2)
            else:
                seq_dim, token_dim = d2, d1
            metadata['seq_dim'] = seq_dim
            metadata['token_dim'] = token_dim
            metadata['channel_dim'] = d3
            metadata['input_was_4d'] = True

        elif len(x.shape) == 3:
            # Already (batch, seq, token) or (batch, token, seq)
            batch_size, d1, d2 = x.shape
            if d1 > d2:
                seq_dim, token_dim = d1, d2
                x = x.transpose(1, 2)
            else:
                seq_dim, token_dim = d2, d1
            metadata['seq_dim'] = seq_dim
            metadata['token_dim'] = token_dim
            metadata['input_was_4d'] = False
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        if x.shape[-1] > 2:
            positions = x[..., -2:]
            x = x[..., 0] 
        # Extract positional information if using data positions
        pos_h = pos_v = None
        if self.use_data_positions:
            if positions is not None:
                # Use explicitly provided positions
                pos_h, pos_v = positions[..., 0], positions[..., 1]  # [batch, seq_len]
            else:
                # Extract from input data - assume positions encoded in the complex data
                if x.shape[-1] == 2 and x.is_complex():
                    # Extract positions from imaginary and real parts of second channel
                    pos_h = x[..., -1].real  # Horizontal position from real part
                    pos_v = x[..., -1].imag  # Vertical position from imaginary part
                    x = x[..., :-1]  # Remove position channel, keep only data
                    if self.verbose:
                        print(f"Extracted positions: pos_h={pos_h.shape}, pos_v={pos_v.shape}")
                        print(f"Data after position extraction: {x.shape}")
            pos_h, pos_v = pos_h.float(), pos_v.float()
            # Store positions in metadata for postprocessing
            metadata['pos_h'] = pos_h
            metadata['pos_v'] = pos_v
            
            # Apply positional encoding
            if pos_h is not None and pos_v is not None:
                if self.pos_encoding_type == 'complex':
                    # Combine positions as complex number: pos_h + 1j * pos_v
                    pos_complex = torch.complex(pos_h, pos_v)  # [batch, seq_len]
                    
                    # Ensure positional encoding matches data dimensions
                    if len(pos_complex.shape) == 2:  # [batch, seq_len]
                        pos_complex = pos_complex.unsqueeze(-1)  # [batch, seq_len, 1]
                    if self.verbose:
                        print(f"Complex positional encoding shape: {pos_complex.shape}")
                    
                    if self.pos_proj is not None:
                        pos_encoding = self.pos_proj(pos_complex)  # [batch, seq_len, dim]
                        if self.verbose:
                            print(f"Projected positional encoding shape: {pos_encoding.shape}")
                        # Ensure broadcasting compatibility
                        if pos_encoding.shape[-1] != x.shape[-1]:
                            # Expand pos_encoding to match x's last dimension
                            pos_encoding = pos_encoding.expand(-1, -1, x.shape[-1])
                        
                        x = x + pos_encoding
                        
                elif self.pos_encoding_type == 'concat':
                    # Concatenate positions and project
                    pos_concat = torch.stack([pos_h, pos_v], dim=-1)  # [batch, seq_len, 2]
                    if self.pos_proj is not None:
                        pos_encoding = self.pos_proj(pos_concat)  # [batch, seq_len, dim]
                        x = x + pos_encoding
                
        
        if self.verbose:
            print(f"Preprocessed tensor shape: {x.shape}, dtype: {x.dtype}")
        
        return x, metadata
    
    def postprocess_output(self, x: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Postprocess transformer output to restore original shape and add back positional information.
        """
        original_shape = metadata['original_shape']
        
        # Get stored positional information
        pos_h = metadata.get('pos_h', None)
        pos_v = metadata.get('pos_v', None)
        
        # If we used data positions, we need to add them back
        if self.use_data_positions and pos_h is not None and pos_v is not None:
            # # Create position channel from stored positions
            # pos_channel = torch.complex(pos_h, pos_v)  # [batch, seq_len]
            
            # # Expand to match transformer output dimensions
            # if len(pos_channel.shape) == 2:  # [batch, seq_len]
            #     pos_channel = pos_channel.unsqueeze(-1)  # [batch, seq_len, 1]
            
            # # Concatenate position channel back to the data
            # x = torch.cat([x, pos_channel], dim=-1)  # [batch, seq_len, token_len + 1]

            if len(x.shape) == 3:
                x = x.unsqueeze(-1)
            if len(pos_h.shape) == 3:
                pos_h = pos_h.unsqueeze(-1)
            if len(pos_v.shape) == 3:
                pos_v = pos_v.unsqueeze(-1)
            x = torch.cat([x, pos_h], dim=-1)
            x = torch.cat([x, pos_v], dim=-1)
        if self.verbose:
            print(f"Postprocessed tensor shape before restoring original: {x.shape}, dtype: {x.dtype}")
        # Restore original shape if needed
        if len(original_shape) == 3:
            original_batch, original_dim1, original_dim2 = original_shape
            current_batch, current_seq, current_token = x.shape
            
            # Check if we need to transpose back to original layout
            # If original was (batch, token_len, seq_len) but we have (batch, seq_len, token_len)
            if (original_dim1 < original_dim2 and current_seq > current_token) or \
               (original_dim1 > original_dim2 and current_seq < current_token):
                x = x.transpose(1, 2)  # Swap back to original dimension order
        elif metadata.get('input_was_4d', False):
            # Restore to (batch, orig_seq, orig_token, orig_channel)
            seq_dim = metadata['seq_dim']
            token_dim = metadata['token_dim']
            channel_dim = metadata['channel_dim']
            original_batch, original_dim1, original_dim2, _ = original_shape
            current_batch, current_seq, current_token, _ = x.shape
            #x = x.reshape(batch_size, seq_dim, token_dim, channel_dim)
            if (original_dim1 < original_dim2 and current_seq > current_token) or \
               (original_dim1 > original_dim2 and current_seq < current_token):
                x = x.transpose(1, 2)  # Swap back to original dimension order
            # Optionally permute back to original order if needed (not tracked here)
        else:
            # For 3D, restore to (batch, seq, token) or (batch, token, seq) as needed
            orig_shape = metadata['original_shape']
            if len(orig_shape) == 3:
                batch_size, d1, d2 = orig_shape
                if d1 > d2 and x.shape[1] != d1:
                    x = x.transpose(1, 2)
                elif d2 > d1 and x.shape[2] != d2:
                    x = x.transpose(1, 2)
        # ...existing code after shape restoration...
        if self.verbose:
            print(f"Postprocessed tensor shape: {x.shape}, original: {metadata['original_shape']}")
        return x
    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        return_abs_logits: bool = False,
        return_real_logits: bool = False
    ) -> Tensor:
        """
        Forward pass of the complex transformer.
        
        Args:
            x: Input tensor. Can be:
                - Token indices (Long) if seq_embed is used
                - Complex tensor of shape [batch, seq_len, dim] for continuous inputs
                - Complex tensor where positions are encoded in data
            positions: Optional explicit position tensor of shape [batch, seq_len, 2] 
                      where last dim is [horizontal_pos, vertical_pos]
            context: Optional context for cross-attention
            mask: Optional attention mask
            return_abs_logits: Whether to return absolute values of logits
            return_real_logits: Whether to return real parts of logits
            
        Returns:
            Transformer output tensor with original shape restored
        """        
        # Preprocess input and extract metadata
        if self.verbose:
            print(f"Input tensor shape before preprocessing: {src.shape}")
        # _, d1, d2, _ = src.shape
        # if d1 == self.token_dim: 
        #     src = src.transpose(1, 2)
        #     transp_d1_d2 = True
        # elif d2 == self.token_dim:
        #     transp_d1_d2 = False
        # else:
        #     raise ValueError("Expected input vector to have shape: (batch_size, seq_len, token_dim, )")

        x, metadata = self.preprocess_input(src)
        if self.verbose:
            print(f"Input tensor shape after preprocessing: {x.shape}")
        # Get rotary embeddings if needed
        seq_len = x.shape[-2]
        rot_emb = self.rotary(seq_len) if self.rotary else None

        # Apply transformer layers
        for i in range(len(self.layers)):
            norm1 = self.layers[i][0]  # type: ignore
            attn = self.layers[i][1]   # type: ignore  
            norm2 = self.layers[i][2]  # type: ignore
            ff = self.layers[i][3]     # type: ignore
            x = attn(norm1(x), context=context, mask=mask, rotary_emb=rot_emb) + x
            x = ff(norm2(x)) + x

        x = self.final_norm(x)

        # Apply output projection if available
        if self.to_logits is not None:
            logits = self.to_logits(x)
            
            # Handle different return types
            if return_abs_logits and return_real_logits:
                raise ValueError("Only one of return_abs_logits / return_real_logits may be True")
            
            if return_abs_logits:
                x = logits.abs()
            elif return_real_logits:
                x = logits.real
            else:
                x = logits
        if self.verbose:
            print(f"Output tensor shape before postprocessing: {x.shape}, dtype: {x.dtype}")
        # Postprocess output to restore original shape
        x = self.postprocess_output(x, metadata)
        if self.verbose:
            print(f"Final output tensor shape: {x.shape}")
        
        return x