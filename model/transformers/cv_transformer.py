from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from model.transformers.attend import Attend

# Type alias for torch.Tensor
Tensor = torch.Tensor

# Complex building blocks (defined below or imported where necessary)

# ---------------------------------------------------------------------------
# Complex building block classes
# ---------------------------------------------------------------------------

class ComplexActivation(nn.Module):
    """Applies a real-valued activation function to both real and imaginary parts."""
    def __init__(self, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(
            self.activation_fn(x.real),
            self.activation_fn(x.imag)
        )

class ComplexDropout(nn.Module):
    """Applies dropout to complex tensors by applying same mask to real and imaginary parts."""
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        # Generate dropout mask for real part and apply to both real and imaginary
        mask = torch.rand_like(x.real) > self.p
        scale = 1.0 / (1.0 - self.p)
        return torch.complex(
            x.real * mask * scale,
            x.imag * mask * scale
        )

class ComplexRMSNorm(nn.Module):
    """RMS normalization for complex tensors."""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS over last dimension
        rms = torch.sqrt(torch.mean(x.real**2 + x.imag**2, dim=-1, keepdim=True) + self.eps)
        # Scale both real and imaginary parts
        return x / rms * self.scale

class ComplexFeedForward(nn.Module):
    """Complex feedforward network with complex linear layers."""
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * mult
        
        # Complex linear layers
        self.w1_real = nn.Linear(dim, inner_dim)
        self.w1_imag = nn.Linear(dim, inner_dim)
        self.w2_real = nn.Linear(inner_dim, dim)
        self.w2_imag = nn.Linear(inner_dim, dim)
        
        self.activation = ComplexActivation(nn.GELU())
        self.dropout = ComplexDropout(dropout)
    
    def complex_linear(self, x: torch.Tensor, w_real: nn.Linear, w_imag: nn.Linear) -> torch.Tensor:
        """Apply complex linear transformation: (a+bi) * (c+di) = (ac-bd) + (ad+bc)i"""
        real_part = w_real(x.real) - w_imag(x.imag)
        imag_part = w_real(x.imag) + w_imag(x.real)
        return torch.complex(real_part, imag_part)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.complex_linear(x, self.w1_real, self.w1_imag)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.complex_linear(x, self.w2_real, self.w2_imag)
        return x

class RotaryEmbedding(nn.Module):
    """Rotary positional embedding for complex tensors."""
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Create frequency tensor
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('freqs', freqs)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        if seq_len is None:
            seq_len = x.shape[-2]
        
        # Generate position indices
        pos = torch.arange(seq_len, device=x.device).float()
        
        # Calculate angles
        angles = pos[:, None] * self.freqs[None, :]
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)
        
        # Apply rotation
        # For complex input, we rotate each complex number
        x_real, x_imag = x.real, x.imag
        
        # Split into pairs for rotation
        x_real_pairs = x_real.view(*x_real.shape[:-1], -1, 2)
        x_imag_pairs = x_imag.view(*x_imag.shape[:-1], -1, 2)
        
        # Apply rotation matrix
        rotated_real = x_real_pairs * cos_vals[..., None] - x_imag_pairs * sin_vals[..., None]
        rotated_imag = x_real_pairs * sin_vals[..., None] + x_imag_pairs * cos_vals[..., None]
        
        # Reshape back
        rotated_real = rotated_real.view(*x.shape)
        rotated_imag = rotated_imag.view(*x.shape)
        
        return torch.complex(rotated_real, rotated_imag)

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def quantize_complex_ste(z: torch.Tensor, step: float = 1.0) -> torch.Tensor:
    """Straight-Through estimator quantization applied separately to real/imag.
    Returns z + (q - z).detach() so gradients flow through z.
    """
    qr = torch.round(z.real / step) * step
    qi = torch.round(z.imag / step) * step
    q = torch.complex(qr, qi)
    return z + (q - z).detach()


def complex_to_real_embed(z: torch.Tensor) -> torch.Tensor:
    """Map complex tensor [..., D] -> real tensor [..., 2*D] by concatenating real/imag.
    Useful for RVQ and kmeans-like ops.
    """
    return torch.cat([z.real, z.imag], dim=-1)


# ---------------------------------------------------------------------------
# Complex azimuth denoiser (slightly enhanced)
# ---------------------------------------------------------------------------
class ComplexAzimuthDenoiser(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 7, padding: Optional[int] = None, res_scale: float = 0.12):
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding
        # Use depthwise separable conv style: use grouped convs per channel
        self.conv_r = nn.Conv1d(ch, ch, kernel_size, padding=padding, groups=1, bias=True)
        self.conv_i = nn.Conv1d(ch, ch, kernel_size, padding=padding, groups=1, bias=True)
        self.res_scale = nn.Parameter(torch.tensor(res_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq, ch] complex
        x_perm = x.permute(0, 2, 1)
        xr, xi = x_perm.real, x_perm.imag
        yr = self.conv_r(xr) - self.conv_i(xi)
        yi = self.conv_r(xi) + self.conv_i(xr)
        y = torch.complex(yr, yi).permute(0, 2, 1)
        return (1.0 - self.res_scale) * x + self.res_scale * y


# ---------------------------------------------------------------------------
# Residual Vector Quantizer for complex latents (simple, CPU/GPU-friendly)
# ---------------------------------------------------------------------------
class ResidualVectorQuantizer(nn.Module):
    """A small RVQ that quantizes real embeddings formed by concatenating real/imag.
    This is a practical option for compressing complex latents to integer indices.
    """
    def __init__(self, dim: int, n_codes: int = 256, n_stages: int = 2):
        super().__init__()
        self.dim = dim
        self.n_codes = n_codes
        self.n_stages = n_stages
        # codebooks are real-valued (we quantize concatenated real/imag)
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(n_codes, dim) * 0.02)
            for _ in range(n_stages)
        ])

    def forward(self, x_real: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """x_real: [..., dim]
        Returns: quantized_real [..., dim], list of indices per stage
        """
        Bshape = x_real.shape
        x_flat = x_real.view(-1, self.dim)  # [N, dim]
        residual = x_flat
        all_indices = []
        quantized = torch.zeros_like(x_flat)
        for s in range(self.n_stages):
            codes = self.codebooks[s]  # [n_codes, dim]
            # compute distances and pick nearest
            # d(x,c) = ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.c
            # use efficient matmul
            dots = residual @ codes.t()  # [N, n_codes]
            codes_norm = (codes ** 2).sum(dim=1).unsqueeze(0)  # [1, n_codes]
            residual_norm = (residual ** 2).sum(dim=1, keepdim=True)  # [N,1]
            dists = residual_norm + codes_norm - 2.0 * dots
            indices = torch.argmin(dists, dim=1)  # [N]
            chosen = codes[indices]  # [N, dim]
            quantized = quantized + chosen
            residual = residual - chosen
            all_indices.append(indices.view(*Bshape[:-1]))
        q = quantized.view(*Bshape)
        return q, all_indices

    def codebook_loss(self):
        # Simple regularizer to keep codebooks small
        return sum((cb ** 2).mean() for cb in self.codebooks)


# ---------------------------------------------------------------------------
# Enhanced local attention (relative bias + overlap windows)
# ---------------------------------------------------------------------------
class EnhancedLocalComplexAttention(nn.Module):
    def __init__(self, dim: int, window_size: int = 128, heads: int = 8, dim_head: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.window_size = window_size
        self.heads = heads
        self.dim = dim
        dim_head = dim // heads if dim_head is None else dim_head
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False, dtype=torch.complex64)
        self.to_k = nn.Linear(dim, inner_dim, bias=False, dtype=torch.complex64)
        self.to_v = nn.Linear(dim, inner_dim, bias=False, dtype=torch.complex64)
        self.to_out = nn.Linear(inner_dim, dim, bias=False, dtype=torch.complex64)

        self.dropout = ComplexDropout(dropout)

        # Relative bias for positions in [-window_size, window_size]
        self.rel_bias = nn.Parameter(torch.zeros(2 * window_size + 1, dtype=torch.float32))
        nn.init.normal_(self.rel_bias, std=0.02)
        self.scale = dim_head ** -0.5

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, seq, dim]
        b, n, _ = x.shape
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        if n <= self.window_size:
            # global attention case
            k_conj = torch.conj(k)
            dots = torch.einsum('bhid,bhjd->bhij', q, k_conj).real * self.scale
            # add relative bias centered
            bias = self._relative_bias(n, n, q.device)
            dots = dots + bias
            if mask is not None:
                dots = dots.masked_fill(mask.unsqueeze(1).expand(-1, h, -1, -1), float('-inf'))
            attn = F.softmax(dots, dim=-1)
            attn = self.dropout(attn.to(torch.complex64))
            out = torch.einsum('bhij,bhjd->bhid', attn, v)
        else:
            out = self._windowed_attention(q, k, v, n, mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def _relative_bias(self, q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
        # produce [1, h, q_len, k_len] bias tensor broadcastable to dots
        # bias depends on relative distance i-j -> index shift by +window_size
        idx = torch.arange(k_len, device=device).unsqueeze(0) - torch.arange(q_len, device=device).unsqueeze(1)  # [q,k]
        idx = idx.clamp(-self.window_size, self.window_size) + self.window_size
        bias = self.rel_bias[idx]  # [q,k]
        return bias.unsqueeze(0).unsqueeze(0)  # [1,1,q,k]

    def _windowed_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: int, mask: Optional[torch.Tensor]):
        b, h, n, d = q.shape
        ws = self.window_size
        # pad to multiple of window
        pad_len = (ws - n % ws) % ws
        if pad_len > 0:
            q = F.pad(q, (0,0,0,pad_len))
            k = F.pad(k, (0,0,0,pad_len))
            v = F.pad(v, (0,0,0,pad_len))
            if mask is not None:
                mask = F.pad(mask, (0, pad_len), value=True)
        n_padded = n + pad_len
        num_windows = n_padded // ws
        # reshape as windows
        q_win = q.view(b, h, num_windows, ws, d)
        k_win = k.view(b, h, num_windows, ws, d)
        v_win = v.view(b, h, num_windows, ws, d)
        # compute dots in window
        k_conj = torch.conj(k_win)
        dots = torch.einsum('bhnid,bhnjd->bhnij', q_win, k_conj).real * self.scale
        # add relative bias within windows
        # create a small bias matrix for ws x ws
        bias = self._relative_bias(ws, ws, q.device).squeeze(0).squeeze(0)  # [ws,ws]
        dots = dots + bias.unsqueeze(0).unsqueeze(0)
        if mask is not None:
            mask_win = mask.view(b, 1, num_windows, ws)
            mask_win = mask_win.unsqueeze(-1).expand(-1, -1, -1, -1, ws)
            dots = dots.masked_fill(mask_win, float('-inf'))
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn.to(torch.complex64))
        out = torch.einsum('bhnij,bhnjd->bhnid', attn, v_win)
        out = out.view(b, h, -1, d)
        if pad_len > 0:
            out = out[:, :, :seq_len, :]
        return out


# ---------------------------------------------------------------------------
# Enhanced compressor and transformer classes
# ---------------------------------------------------------------------------
class CVTransformer(nn.Module):
    """Encoder-decoder compressor with anchor pooling, RVQ support and improved attention.

    Usage:
        model = CVTransformer(input_dim=1, model_dim=256, ...)
        enc = model.encode(x)  # returns dict with quantized latents and meta
        out = model.decode(enc['quantized_latents'], enc['meta'])
    """
    def __init__(
        self,
        input_dim: int = 1,
        model_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_mult: int = 4,
        window_size: int = 128,
        compressed_dim: int = 32,
        latent_dim: int = 16,
        register_bank_size: int = 32,
        quant_step: float = 0.25,
        rvq_codes: int = 512,
        rvq_stages: int = 2,
        use_rvq: bool = True,
        hann_len: int = 5000,
        output_mode: str = "complex",  # "complex", "magnitude", "real", "imag"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_mode = output_mode
        self.model_dim = model_dim
        self.window_size = window_size
        self.compressed_dim = compressed_dim
        self.latent_dim = latent_dim
        self.quant_step = quant_step
        self.use_rvq = use_rvq

        # projections
        self.input_proj = nn.Linear(input_dim, model_dim, dtype=torch.complex64)
        self.output_proj = nn.Linear(model_dim, input_dim, dtype=torch.complex64)
        
        # Add a dynamic projection layer for handling variable input dimensions
        self.adaptive_proj = None
        self.adaptive_proj_out = None
        self._original_feature_dim = input_dim

        # stacks
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        dim_head = model_dim // num_heads
        for _ in range(num_layers):
            self.encoder_layers.append(nn.ModuleList([
                ComplexRMSNorm(model_dim),
                EnhancedLocalComplexAttention(dim=model_dim, window_size=window_size, heads=num_heads, dim_head=dim_head),
                ComplexRMSNorm(model_dim),
                ComplexFeedForward(model_dim, mult=ff_mult)
            ]))
        for _ in range(num_layers):
            self.decoder_layers.append(nn.ModuleList([
                ComplexRMSNorm(model_dim),
                EnhancedLocalComplexAttention(dim=model_dim, window_size=window_size, heads=num_heads, dim_head=dim_head),
                ComplexRMSNorm(model_dim),
                ComplexFeedForward(model_dim, mult=ff_mult)
            ]))

        # compressor / expander
        self.spatial_compressor = nn.Sequential(
            nn.Linear(model_dim, compressed_dim, dtype=torch.complex64),
            ComplexActivation(nn.GELU()),
            ComplexDropout(0.07)
        )
        self.az_denoiser = ComplexAzimuthDenoiser(compressed_dim, kernel_size=5)
        self.latent_proj = nn.Linear(compressed_dim, latent_dim, dtype=torch.complex64)
        self.latent_inv = nn.Linear(latent_dim, compressed_dim, dtype=torch.complex64)
        self.spatial_expander = nn.Sequential(
            nn.Linear(compressed_dim, model_dim, dtype=torch.complex64),
            ComplexActivation(nn.GELU()),
            ComplexDropout(0.07)
        )

        # registers and context
        hann = torch.hann_window(hann_len, periodic=True)
        self.register_buffer('hann_window', hann)
        self.register_parameter('register_bank', nn.Parameter(torch.randn(register_bank_size, model_dim, dtype=torch.complex64) * 0.02))
        self.register_buffer('pos_encoding', self._make_pos_encoding(hann_len, model_dim))

        # final norm
        self.final_norm = ComplexRMSNorm(model_dim)

        # RVQ (optional)
        if self.use_rvq:
            self.rvq = ResidualVectorQuantizer(dim=latent_dim * 2, n_codes=rvq_codes, n_stages=rvq_stages)
        else:
            self.rvq = None

        # init
        self._init_weights()

    def _make_pos_encoding(self, max_len: int, dim: int) -> torch.Tensor:
        pos = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim).float() * -(math.log(10000.0) / dim))
        angles = pos * div_term
        pe = torch.complex(torch.sin(angles), torch.cos(angles))
        return pe.unsqueeze(0)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, 'weight', None) is not None:
                if module.weight.dtype == torch.complex64:
                    fan_in = module.weight.shape[1]
                    fan_out = module.weight.shape[0]
                    scale = math.sqrt(2.0 / (fan_in + fan_out))
                    with torch.no_grad():
                        module.weight.real.normal_(0, scale * 0.7)
                        module.weight.imag.normal_(0, scale * 0.7)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def apply_hann(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, feature]
        L = x.shape[1]
        w = self.hann_window[:L].to(x.device)
        return x * w.unsqueeze(0).unsqueeze(-1)

    def encode(self, x: torch.Tensor) -> dict:
        # Store original feature dimension FIRST
        input_was_real = not torch.is_complex(x)  # Track if input was real
        
        if len(x.shape) == 2:
            original_feature_dim = 1  # Will be unsqueezed to [B, seq, 1]
        elif len(x.shape) > 3:
            # Calculate flattened feature dimension
            original_feature_dim = torch.prod(torch.tensor(x.shape[2:])).item()
        else:
            original_feature_dim = x.shape[-1]
        
        # convert to complex if necessary
        if not torch.is_complex(x):
            if x.shape[-1] >= 2:
                x = torch.complex(x[..., 0], x[..., 1])
            else:
                x = torch.complex(x[..., 0], torch.zeros_like(x[..., 0]))
        
        # Ensure 3D shape [B, seq_len, feature_dim]
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)  # [B, seq_len] -> [B, seq_len, 1]
        elif len(x.shape) > 3:
            # Flatten extra dimensions into the last dimension
            B = x.shape[0]
            seq_len = x.shape[1]
            feature_dim = torch.prod(torch.tensor(x.shape[2:]))
            x = x.view(B, seq_len, feature_dim)
        
        B, seq_len, feature_dim = x.shape
        
        # Handle dynamic input dimensions
        if feature_dim != self.input_dim:
            if self.adaptive_proj is None or self.adaptive_proj.in_features != feature_dim:
                self.adaptive_proj = nn.Linear(feature_dim, self.input_dim, dtype=torch.complex64).to(x.device)
            x = self.adaptive_proj(x)
        
        x_proj = self.input_proj(x)
        # add pos enc if same length
        if seq_len <= self.pos_encoding.shape[1]:
            x_proj = x_proj + self.pos_encoding[:, :seq_len, :].to(x_proj.device)

        encoded = x_proj
        for norm1, attn, norm2, ffn in self.encoder_layers:
            attn_out = attn(norm1(encoded))
            encoded = encoded + attn_out
            ffn_out = ffn(norm2(encoded))
            encoded = encoded + ffn_out

        compressed = self.spatial_compressor(encoded)
        compressed = self.az_denoiser(compressed)
        compressed = self.apply_hann(compressed)

        latents = self.latent_proj(compressed)  # [B, seq, latent_dim]

        # anchor pooling -> produce a compact spatial_repr per sample
        spatial_repr = self._anchor_pool(latents)
        # spatial_repr: [B, 1, latent_dim]

        # quantize latents before aggregation optionally, here quantize spatial_repr
        if self.use_rvq and self.rvq is not None:
            real_embed = complex_to_real_embed(spatial_repr.squeeze(1))  # [B, 2*latent_dim]
            q_real, indices = self.rvq(real_embed)
            q_complex = torch.complex(q_real[:, :self.latent_dim], q_real[:, self.latent_dim:])
            quantized = q_complex.unsqueeze(1)
            rvq_meta = {'rvq_indices': indices}
        else:
            # simple STE per-component quantization
            quantized = quantize_complex_ste(spatial_repr, step=self.quant_step)
            rvq_meta = {}

        meta = {'seq_len': seq_len, 'quant_step': float(self.quant_step), 'original_feature_dim': original_feature_dim, 'input_was_real': input_was_real}
        meta.update(rvq_meta)

        return {'quantized_latents': quantized, 'meta': meta}

    def _anchor_pool(self, latents: torch.Tensor, max_anchors: int = 16) -> torch.Tensor:
        # latents: [B, seq_len, latent_dim]
        B, seq, D = latents.shape
        # choose number of anchors depending on seq
        num_anchors = min(max_anchors, max(1, seq // (self.window_size // 2)))
        anchor_size = max(1, seq // num_anchors)
        overlap = anchor_size // 4
        anchors = []
        for i in range(num_anchors):
            start = max(0, i * anchor_size - overlap)
            end = min(seq, (i + 1) * anchor_size + overlap)
            seg = latents[:, start:end, :]
            weights = torch.softmax(torch.abs(seg).mean(dim=-1, keepdim=True), dim=1)
            anchor = torch.sum(seg * weights.to(torch.complex64), dim=1, keepdim=True)  # [B,1,D]
            anchors.append(anchor)
        anchor_stack = torch.cat(anchors, dim=1)  # [B, num_anchors, D]
        weights = torch.softmax(torch.abs(anchor_stack).mean(dim=-1, keepdim=True), dim=1)
        pooled = torch.sum(anchor_stack * weights.to(torch.complex64), dim=1, keepdim=True)  # [B,1,D]
        return pooled

    def decode(self, quantized_latents: torch.Tensor, meta: Optional[dict] = None) -> torch.Tensor:
        # quantized_latents: [B, 1, latent_dim]
        B = quantized_latents.shape[0]
        seq_len = meta['seq_len'] if meta is not None and 'seq_len' in meta else self.hann_window.shape[0]
        original_feature_dim = meta.get('original_feature_dim', self.input_dim) if meta else self.input_dim
        input_was_real = meta.get('input_was_real', True) if meta else True  # Track if input was real

        denoised_back = self.latent_inv(quantized_latents)  # [B,1,compressed_dim]
        # expand to full sequence with learned upsampling (repeat + small positional bias)
        expanded = self.spatial_expander(denoised_back)  # [B,1,model_dim]
        expanded = expanded.expand(B, seq_len, self.model_dim)

        # add small positional modulation to break symmetry
        if seq_len <= self.pos_encoding.shape[1]:
            pos = self.pos_encoding[:, :seq_len, :].to(expanded.device)
            expanded = expanded + pos

        decoded = expanded
        for norm1, attn, norm2, ffn in self.decoder_layers:
            attn_out = attn(norm1(decoded))
            decoded = decoded + attn_out
            ffn_out = ffn(norm2(decoded))
            decoded = decoded + ffn_out

        decoded = self.final_norm(decoded)
        out = self.output_proj(decoded)  # [B, seq, input_dim]
        
        # If input was originally real, take only real part to avoid complex output
        if input_was_real and torch.is_complex(out):
            out = out.real
        
        # If we need to project back to original feature dimension
        if original_feature_dim != self.input_dim:
            if not hasattr(self, 'adaptive_proj_out') or self.adaptive_proj_out is None or self.adaptive_proj_out.out_features != original_feature_dim:
                # Use real-valued projection if input was real
                dtype = torch.float32 if input_was_real else torch.complex64
                self.adaptive_proj_out = nn.Linear(self.input_dim, original_feature_dim, dtype=dtype).to(out.device)
            out = self.adaptive_proj_out(out)
            
            # Ensure output is real if input was real
            if input_was_real and torch.is_complex(out):
                out = out.real
        
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Remember if input was real and store original shape
        input_was_real = not torch.is_complex(x)
        original_shape = x.shape
        
        enc = self.encode(x)
        out = self.decode(enc['quantized_latents'], enc['meta'])
        
        # Handle different output modes for complex data
        if len(original_shape) == 4 and original_shape[-1] == 2 and hasattr(self, 'output_mode'):
            # Input shape is [B, H, W, 2] with real/imag stacked
            if self.output_mode == 'magnitude':
                # For magnitude mode, output should be [B, H, W] - the magnitude of the reconstructed complex data
                if torch.is_complex(out):
                    # If output is complex, take magnitude
                    out = torch.abs(out)  # [B, H, W]
                else:
                    # If output is real with shape [B, H, W, 2], compute magnitude
                    if len(out.shape) == 4 and out.shape[-1] == 2:
                        real_part = out[..., 0]  # [B, H, W]
                        imag_part = out[..., 1]  # [B, H, W]
                        out = torch.sqrt(real_part**2 + imag_part**2)  # [B, H, W]
                    # If shape is already [B, H, W], keep as is
            elif self.output_mode == 'complex':
                # For complex mode, output should be [B, H, W, 2] - preserve the real/imag structure
                if torch.is_complex(out):
                    # Convert complex tensor to real/imag stacked
                    out = torch.stack([out.real, out.imag], dim=-1)  # [B, H, W, 2]
                else:
                    # If already real with proper shape, keep as is
                    if len(out.shape) == 3:
                        # Add dummy imaginary part
                        out = torch.stack([out, torch.zeros_like(out)], dim=-1)  # [B, H, W, 2]
            elif self.output_mode == 'real':
                # For real mode, output should be [B, H, W] - real part only
                if torch.is_complex(out):
                    out = out.real  # [B, H, W]
                else:
                    if len(out.shape) == 4 and out.shape[-1] == 2:
                        out = out[..., 0]  # [B, H, W] - take real part
            elif self.output_mode == 'imag':
                # For imag mode, output should be [B, H, W] - imaginary part only
                if torch.is_complex(out):
                    out = out.imag  # [B, H, W]
                else:
                    if len(out.shape) == 4 and out.shape[-1] == 2:
                        out = out[..., 1]  # [B, H, W] - take imaginary part
        else:
            # For other cases, convert back to real if input was real
            if input_was_real and torch.is_complex(out):
                out = out.real
            
            # Try to match original shape
            if out.shape != original_shape:
                if out.numel() == torch.prod(torch.tensor(original_shape)):
                    out = out.view(original_shape)
        
        return out

    # Utility: compress/decompress tile helpers
    def compress_tile(self, x: torch.Tensor) -> Tuple[dict, torch.Tensor]:
        """Compress tile x and return metadata + quantized latent.
        x: [B, seq, input_dim]
        returns (meta_dict, latent_tensor)
        """
        enc = self.encode(x)
        return enc['meta'], enc['quantized_latents']

    def decompress_tile(self, quantized_latents: torch.Tensor, meta: dict) -> torch.Tensor:
        return self.decode(quantized_latents, meta)


# ---------------------------------------------------------------------------
# Example quick instantiation function (not executed at import)
# ---------------------------------------------------------------------------
def make_default_compressor():
    model = EnhancedSpatialCompressor(
        input_dim=1,
        model_dim=256,
        num_layers=6,
        num_heads=8,
        ff_mult=4,
        window_size=128,
        compressed_dim=32,
        latent_dim=16,
        register_bank_size=32,
        quant_step=0.25,
        rvq_codes=512,
        rvq_stages=2,
        use_rvq=True,
        hann_len=5000,
    )
    return model


if __name__ == '__main__':
    # quick smoke test (small sizes) to ensure shapes work
    model = make_default_compressor()
    B = 2
    seq = 256
    x = torch.randn(B, seq, 1)
    out = model(x)
    print('in', x.shape, 'out', out.shape)


# ---------------------------------------------------------------------------
# Helper functions for complex attention
# ---------------------------------------------------------------------------

def exists(val):
    return val is not None

def modulate_with_rotation(x, rotary_emb):
    """Apply rotary embedding to complex tensor"""
    return rotary_emb(x)

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
    
    # Concatenate real/imag to create real tensor
    q_flat = torch.cat([qr, qi], dim=-1)
    k_flat = torch.cat([kr, ki], dim=-1)
    v_flat = torch.cat([vr, vi], dim=-1)
    
    # Apply attention
    out_flat = attend_fn(q_flat, k_flat, v_flat, mask=mask)
    
    # Split back and reconstruct complex
    d = out_flat.shape[-1] // 2
    out_r, out_i = out_flat[..., :d], out_flat[..., d:]
    return torch.complex(out_r, out_i)

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
    
    b, h, n, d = qr.shape
    
    # Stack all combinations for batch processing
    # [qr, qi, qr, qi] attending to [kr, ki, ki, kr] with values [vr, vi, vi, vr]
    q_stack = torch.stack([qr, qi, qr, qi], dim=0)  # [4, b, h, n, d]
    k_stack = torch.stack([kr, ki, ki, kr], dim=0)  # [4, b, h, n, d]
    v_stack = torch.stack([vr, vi, vi, vr], dim=0)  # [4, b, h, n, d]
    
    # Flatten for batch attention
    qf = rearrange(q_stack, 'r b h n d -> (r b) h n d')
    kf = rearrange(k_stack, 'r b h n d -> (r b) h n d')
    vf = rearrange(v_stack, 'r b h n d -> (r b) h n d')

    if exists(mask):
        mask = repeat(mask, 'b ... -> (r b) ...', r=4)

    # compute all four
    outf = attend_fn(qf, kf, vf, mask=mask)
    outf = rearrange(outf, '(r b) h n d -> r b h n d', r=4, b=b)

    # reassemble real & imag
    # real part = A_rr - A_ii
    # imag part = A_ri + A_ir
    Att = outf  # [4, b, h, n, d]
    rr, ri, ir, ii = Att  # unpack
    
    real = rr - ii
    imag = ri + ir

    stacked = torch.stack([real, imag], dim=-1)  # [b, h, n, d, 2]
    return torch.view_as_complex(stacked)


# ---------------------------------------------------------------------------
# Complex MultiHead Attention Module
# ---------------------------------------------------------------------------

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
        self.heads = heads  # Store heads value directly

        # head splitting/merging
        self.split = rearrange
        self.merge = rearrange

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        rotary_emb: Optional[Tensor] = None
    ) -> Tensor:
        h = self.heads
        kv_input = context if exists(context) else x

        q = self.to_q(x)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

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
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# ---------------------------------------------------------------------------
# Complex Transformer Class
# ---------------------------------------------------------------------------
