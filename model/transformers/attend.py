from functools import partial

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F

from collections import namedtuple
from functools import wraps
from packaging import version

from einops import rearrange, repeat

# constants

EfficientAttentionConfig = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# tensor functions

def create_causal_mask(i, j, device):
    return torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

# main class

class Attend(nn.Module):
    """
    Attend is a PyTorch module implementing attention mechanisms with support for flash attention, 
    causal masking, and dropout for efficient transformer models.
    
    This module provides a unified interface for different attention implementations, automatically
    selecting the most efficient variant based on hardware capabilities and configuration. It supports
    both standard scaled dot-product attention and optimized flash attention for improved performance.
    
    Args:
        dropout (float, optional): Dropout probability applied to attention weights. Default is 0.0.
        causal (bool, optional): Whether to apply causal (lower triangular) masking for autoregressive 
            attention. Default is False.
        heads (int, optional): Number of attention heads. Currently unused but kept for compatibility. 
            Default is None.
        scale (float, optional): Scaling factor for attention logits. If None, uses 1/sqrt(d_k) where 
            d_k is the key dimension. Default is None.
        flash (bool, optional): Whether to use flash attention when available. Requires PyTorch >= 2.0 
            and compatible hardware. Default is False.
    
    Attributes:
        scale (float): Attention scaling factor.
        causal (bool): Whether causal masking is enabled.
        dropout (float): Dropout probability.
        attn_dropout (nn.Dropout): Dropout layer for attention weights.
        flash (bool): Whether flash attention is enabled.
        cpu_config (EfficientAttentionConfig): Configuration for CPU attention.
        cuda_config (EfficientAttentionConfig): Configuration for CUDA attention.
    
    Methods:
        flash_attn(q, k, v, mask=None): 
            Optimized flash attention implementation using PyTorch's scaled_dot_product_attention.
        forward(q, k, v, mask=None):
            Main attention computation with automatic backend selection.
    
    Notes:
        - Flash attention is automatically enabled for A100 (compute 8.0) and H100 (compute 9.0) GPUs.
        - For other GPUs, falls back to memory-efficient or math attention implementations.
        - Causal masking is automatically disabled for single-token queries (q_len == 1) to support KV caching.
        - Supports both 3D (b, n, d) and 4D (b, h, n, d) key-value tensors for flexibility.
    
    Example:
        >>> attend = Attend(dropout=0.1, causal=True, flash=True)
        >>> q = torch.randn(2, 8, 128, 64)  # (batch, heads, seq_len, head_dim)
        >>> k = torch.randn(2, 8, 128, 64)
        >>> v = torch.randn(2, 8, 128, 64)
        >>> output = attend(q, k, v)
        >>> print(output.shape)  # torch.Size([2, 8, 128, 64])
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        heads = None,
        scale = None,
        flash = False,
    ):
        super().__init__()
        self.scale = scale

        self.causal = causal
        self.create_causal_mask = create_causal_mask

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        # flash attention

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = EfficientAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        major, minor = device_properties.major, device_properties.minor

        if (major, minor) == (8, 0):
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        elif (major, minor) == (9, 0):
            print_once('H100 GPU detected, using flash attention')
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(False, True, True)

    def flash_attn(
        self,
        q, k, v,
        mask = None
    ):
        """
        Optimized flash attention implementation using PyTorch's scaled_dot_product_attention.
        
        Flash attention reduces memory usage from O(N²) to O(N) by using block-wise computation
        and avoiding materialization of the full attention matrix. This enables processing of
        much longer sequences efficiently.
        
        Key optimizations:
        - Tiled computation to reduce memory footprint
        - Fused operations to minimize memory transfers
        - Hardware-specific optimizations for A100/H100 GPUs
        - Automatic fallback to memory-efficient or math implementations
        
        Args:
            q (Tensor): Query tensor of shape (batch, heads, seq_len, head_dim)
            k (Tensor): Key tensor of shape (batch, heads, seq_len, head_dim) or (batch, seq_len, head_dim)
            v (Tensor): Value tensor of shape (batch, heads, seq_len, head_dim) or (batch, seq_len, head_dim)
            mask (Tensor, optional): Attention mask of shape (batch, heads, seq_len, seq_len) or broadcastable.
                True values indicate positions to attend to, False values are masked out.
        
        Returns:
            Tensor: Attention output of shape (batch, heads, seq_len, head_dim)
        
        Notes:
            - Automatically disables causal masking for single-token queries (KV caching optimization)
            - Handles mixed causal and padding masks correctly
            - Supports different key/value sequence lengths for cross-attention
            - Uses hardware-specific configurations (flash/math/memory-efficient) based on GPU type
            - Protects against entirely masked rows by ensuring at least one valid attention position
        
        Memory Complexity:
            - Standard attention: O(N²) where N is sequence length
            - Flash attention: O(N) with block size B, requiring O(N²/B) passes
        """
        
        batch, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        causal = self.causal

        # in the case of kv caching with one token (q_len == 1), just turn off causal masking
        # in speculative decoding, this may go up to 5-6, so right aligned causal mask will be needed there

        if q_len == 1 and causal:
            causal = False

        # expand key padding mask

        if exists(mask):
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

        # handle kv cache - this should be bypassable in updated flash attention 2

        if k_len > q_len and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            if not exists(mask):
                mask = ~causal_mask
            else:
                mask = mask & ~causal_mask
            causal = False

        # manually handle causal mask, if another mask was given

        row_is_entirely_masked = None

        if exists(mask) and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            mask = mask & ~causal_mask

            # protect against an entire row being masked out

            row_is_entirely_masked = ~mask.any(dim = -1)
            mask[..., 0] = mask[..., 0] | row_is_entirely_masked

            causal = False

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = causal
            )

        # for a row that is entirely masked out, should zero out the output of that row token

        if exists(row_is_entirely_masked):
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out

    def forward(
        self,
        q, k, v,
        mask = None
    ):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, heads, kv_heads, device = q.shape[-2], q.shape[1], k.shape[1], q.device

        scale = default(self.scale, q.shape[-1] ** -0.5)

        if self.flash:
            return self.flash_attn(q, k, v, mask = mask)

        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        sim = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale

        i, j, dtype = *sim.shape[-2:], sim.dtype

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal and n > 1:
            causal_mask = self.create_causal_mask(i, j, device = device)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1)
        attn = attn.type(dtype)

        attn = self.attn_dropout(attn)

        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        return out