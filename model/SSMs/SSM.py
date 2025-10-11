""" Standalone version of sarSSM including the code for S4D """


import logging
from functools import partial
import math
import numpy as np
from scipy import special as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only
from einops import rearrange, repeat
import opt_einsum as oe

contract = oe.contract
contract_expression = oe.contract_expression


_c2r = torch.view_as_real
_r2c = torch.view_as_complex
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()



""" simple nn.Module components """
class ComplexActivation(nn.Module):
    def __init__(self, activation: nn.Module):
        super().__init__()
        self.activation = activation

    def forward(self, x):
        if torch.is_complex(x):
            return torch.complex(self.activation(x.real), self.activation(x.imag))
        else:
            return self.activation(x)
class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        if torch.is_complex(x):
            return torch.complex(self.dropout(x.real), self.dropout(x.imag))
        else:
            return self.dropout(x)
        
class ComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.real_norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
        self.imag_norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        if torch.is_complex(x):
            return torch.complex(self.real_norm(x.real), self.imag_norm(x.imag))
        else:
            return self.real_norm(x)
def Activation(activation=None, dim=-1, complex:bool=False):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        act = nn.Identity()
    elif activation == 'tanh':
        act = nn.Tanh()
    elif activation == 'relu':
        act = nn.ReLU()
    elif activation == 'gelu':
        act = nn.GELU()
    elif activation in ['swish', 'silu']:
        act = nn.SiLU()
    elif activation == 'glu':
        act = nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        act = nn.Sigmoid()
    elif activation == 'leakyrelu':
        act = nn.LeakyReLU(negative_slope=0.125)
    elif activation == 'hardswish':
        act = nn.Hardswish()
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))
    if complex:
        return ComplexActivation(act)
    else:
        return act


class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, **kwargs):
        super().__init__()
        self.real = nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias, **kwargs)
        self.imag = nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias, **kwargs)

    def forward(self, x):
        # x: (B, in_channels, L)
        return torch.complex(self.real(x.real), self.imag(x.imag))

def LinearActivation(
        d_input, d_output, bias=True,
        transposed=False,
        activation=None,
        activate=False, # Apply activation as part of this module
        complex=False,
        **kwargs,
    ):
    """ Returns a linear nn.Module with control over axes order, initialization, and activation """

    # Use ComplexConv1d if complex and transposed, else standard
    if complex and transposed:
        linear_cls = partial(ComplexConv1d, kernel_size=1)
    elif transposed:
        linear_cls = partial(nn.Conv1d, kernel_size=1)
    elif complex:
        linear_cls = ComplexLinear
    else:
        linear_cls = nn.Linear

    if activation == 'glu': d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    if activate and activation is not None:
        activation = Activation(activation, dim=-2 if transposed else -1, complex=complex)
        linear = nn.Sequential(linear, activation)
    return linear

""" HiPPO utilities """

def random_dplr(N, H=1, scaling='inverse', real_scale=1.0, imag_scale=1.0):
    dtype = torch.cfloat

    pi = torch.tensor(np.pi)
    real_part = .5 * torch.ones(H, N//2)
    imag_part = repeat(torch.arange(N//2), 'n -> h n', h=H)

    real_part = real_scale * real_part
    if scaling == 'random':
        imag_part = torch.randn(H, N//2)
    elif scaling == 'linear':
        imag_part = pi * imag_part
    elif scaling == 'inverse': # Based on asymptotics of the default HiPPO matrix
        imag_part = 1/pi * N * (N/(1+2*imag_part)-1)
    else: raise NotImplementedError
    imag_part = imag_scale * imag_part
    w = -real_part + 1j * imag_part


    B = torch.randn(H, N//2, dtype=dtype)

    norm = -B/w # (H, N) # Result if you integrate the kernel with constant 1 function
    zeta = 2*torch.sum(torch.abs(norm)**2, dim=-1, keepdim=True) # Variance with a random C vector
    B = B / zeta**.5

    return w, B


class SSKernelDiag(nn.Module):
    """ 
        Version using (complex) diagonal state matrix. Note that it is slower and less memory efficient than the NPLR kernel because of lack of kernel support.
    """

    def __init__(
        self,
        w, C, log_dt,
        lr=None,
    ):

        super().__init__()

        # Rank of low-rank correction
        assert w.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = w.size(-1)
        assert self.H % w.size(0) == 0
        self.copies = self.H // w.size(0)


        # Broadcast everything to correct shapes
        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N))) # (H, C, N)

        # Register parameters
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))
        self.register("log_dt", log_dt, True, lr, 0.0)

        log_w_real = torch.log(-w.real + 1e-4)
        w_imag = w.imag
        self.register("log_w_real", log_w_real, True, lr, 0.0)
        self.register("w_imag", w_imag, True, lr, 0.0)


    def _w(self):
        # Get the internal w (diagonal) parameter
        w_real = -torch.exp(self.log_w_real)
        w_imag = self.w_imag
        w = w_real + 1j * w_imag
        w = repeat(w, 't n -> (v t) n', v=self.copies) # (H N)
        return w

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        dt = torch.exp(self.log_dt) # (H)
        C = _r2c(self.C) # (C H N)
        w = self._w() # (H N)

        # Incorporate dt into A
        dtA = w * dt.unsqueeze(-1)  # (H N)

        # Power up
        K = dtA.unsqueeze(-1) * torch.arange(L, device=w.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / w
        K = contract('chn, hnl -> chl', C, torch.exp(K))
        K = 2*K.real

        return K

    def setup_step(self):
        dt = torch.exp(self.log_dt) # (H)
        C = _r2c(self.C) # (C H N)
        w = self._w() # (H N)

        # Incorporate dt into A
        dtA = w * dt.unsqueeze(-1)  # (H N)
        self.dA = torch.exp(dtA) # (H N)
        self.dC = C * (torch.exp(dtA)-1.) / w # (C H N)
        self.dB = self.dC.new_ones(self.H, self.N) # (H N)

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
       
        # print(f"SSKernelDiag step u: {u.shape}, dA shape: {self.dA.shape}, dB shape: {self.dB.shape},state: {state.shape}")
        u = u.squeeze()
        
        next_state = contract("h n, b h n -> b h n", self.dA, state) \
                + contract("h n, b h -> b h n", self.dB, u)
        y = contract("c h n, b h n -> b c h", self.dC, next_state)
        # print(f"SSKernelDiag step y: {y.shape}, next_state: {next_state.shape}, dC: {self.dC.shape}")
        return 2*y.real, next_state


    def register(self, name, tensor, trainable=False, lr=None, wd=None):
        """Utility method: register a tensor as a buffer or trainable parameter"""

        if trainable:
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            self.register_buffer(name, tensor)

        optim = {}
        if trainable and lr is not None:
            optim["lr"] = lr
        if trainable and wd is not None:
            optim["weight_decay"] = wd
        if len(optim) > 0:
            setattr(getattr(self, name), "_optim", optim)

class S4DKernel(nn.Module):
    """
        Wrapper around SSKernelDiag that generates the diagonal SSM parameters
    """

    def __init__(
        self,
        H,
        N=64,
        scaling="inverse",
        channels=1, # 1-dim to C-dim map; can think of C as having separate "heads"
        dt_min=0.001,
        dt_max=0.1,
        lr=None, # Hook to set LR of SSM parameters differently
        n_ssm=1, # Copies of the ODE parameters A and B. Must divide H
        **kernel_args,
    ):
        super().__init__()
        self.N = N
        self.H = H
        dtype = torch.float
        cdtype = torch.cfloat
        self.channels = channels
        self.n_ssm = n_ssm

        # Generate dt
        log_dt = torch.rand(self.H, dtype=dtype) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        # Compute the preprocessed representation
        # Generate low rank correction p for the measure
        w, B = random_dplr(self.N, H=n_ssm, scaling=scaling)

        C = torch.randn(channels, self.H, self.N // 2, dtype=cdtype)

        # Broadcast tensors to n_ssm copies
        # These will be the parameters, so make sure tensors are materialized and contiguous
        B = repeat(B, 't n -> (v t) n', v=self.n_ssm // B.size(-2)).clone().contiguous()
        w = repeat(w, 't n -> (v t) n', v=self.n_ssm // w.size(-2)).clone().contiguous()

        # Combine B and C using structure of diagonal SSM
        C = C * repeat(B, 't n -> (v t) n', v=H//self.n_ssm)
        self.kernel = SSKernelDiag(
            w, C, log_dt,
            lr=lr
            #,**kernel_args,
        )

    def forward(self, L=None):
        k = self.kernel(L=L)
        return k.float()

    def setup_step(self):
        self.kernel.setup_step()

    def step(self, u, state, **kwargs):
        u, state = self.kernel.step(u, state, **kwargs)
        return u.float(), state

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)


class S4D(nn.Module):

    def __init__(
            self,
            d_model,
            d_state=64,
            channels=1, # maps 1-dim to C-dim
            bidirectional=False,
            # Arguments for FF
            activation='gelu', # activation in between SS and FF
            postact=None, # activation after FF
            dropout=0.0,
            transposed=False, # axis ordering (B, L, D) or (B, D, L)
            complex:bool=True,
            # SSM Kernel arguments
            **kernel_args,
        ):
        """
        d_state: the dimension of the state, also denoted by N
        channels: can be interpreted as a number of "heads"
        bidirectional: bidirectional
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, H) or (B, H, L) [B=batch size, L=sequence length, H=hidden dimension]

        Other options are all experimental and should not need to be configured
        """

        super().__init__()

        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed
        self.complex = complex
        self.D = nn.Parameter(torch.randn(channels, self.h))

        if self.bidirectional:
            channels *= 2

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, channels=channels, **kernel_args)

        # Pointwise
        self.activation = Activation(activation, complex=complex)
        if complex:
            self.dropout = ComplexDropout(dropout) if dropout > 0.0 else nn.Identity()
        else:
            dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout
            self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = LinearActivation(
            self.h*self.channels,
            self.h,
            transposed=self.transposed,
            activation=postact,
            activate=True,
            complex=complex
        )


    def forward(self, u):
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SS Kernel
        k = self.kernel(L=L) # (C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, L)) \
                    + F.pad(k1.flip(-1), (L, 0))
        if self.complex:
            k_f = torch.fft.fft(k, n=2*L) # (C H 2L)
            u_f = torch.fft.fft(u, n=2*L) # (B H 2L)
            y_f = contract('bhl,chl->bchl', u_f, k_f) # (B C H 2L)
            y = torch.fft.ifft(y_f, n=2*L)[..., :L] # (B C H L)
        else:
            k_f = torch.fft.rfft(k, n=2*L) # (C H L)
            u_f = torch.fft.rfft(u, n=2*L) # (B H L)
            # print(f"u_f: {u_f.shape}, k_f: {k_f.shape}, k: {k.shape}, u: {u.shape}")
            y_f = contract('bhl,chl->bchl', u_f, k_f) # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)
            y = torch.fft.irfft(y_f, n=2*L)[..., :L] # (B C H L)

        # # Ensure output length matches input
        # if y.shape[-1] < L:
        #     # Pad at the end
        #     pad_size = L - y.shape[-1]
        #     y = F.pad(y, (0, pad_size))
        # elif y.shape[-1] > L:
        #     # Crop to input length
        #     y = y[..., :L]

        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', u, self.D)

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        y = self.dropout(self.activation(y))

        if not self.transposed: y = y.transpose(-1, -2)

        y = self.output_linear(y)

        return y, None

    def setup_step(self):
        self.kernel.setup_step()

    def step(self, u, state):
        """ Step one time step as a recurrent model. Intended to be used during validation.

        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """

        y, next_state = self.kernel.step(u, state) # (B C H)
        # print(f"S4D step y before D of shape {self.D.shape}: {y.shape}, next_state: {next_state.shape}, state: {state.shape}, u: {u.shape}")
        # y = y + u.unsqueeze(-2) * self.D
        # y = y + u * self.D
        y = y + contract('b l h, c h -> b c h', u, self.D)
        # print(f"S4D step y after D: {y.shape}, next_state: {next_state.shape}, state: {state.shape}, u: {u.shape}")
        y = rearrange(y, '... c h -> ... (c h)')
        # print(f"S4D step y after rearrange: {y.shape}, next_state: {next_state.shape}, state: {state.shape}, u: {u.shape}")
        y = self.activation(y)
        if self.transposed:
            y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
        else:
            y = self.output_linear(y)
        y = y.unsqueeze(1) 
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        return self.kernel.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)
class ComplexLinear(nn.Module):
    """A simple complex linear layer using two real-valued linear layers."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.W_real = nn.Linear(in_features, out_features, bias=bias)
        self.W_imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        xr, xi = x.real, x.imag
        yr = self.W_real(xr) - self.W_imag(xi)
        yi = self.W_real(xi) + self.W_imag(xr)
        return torch.complex(yr, yi)
    # def __init__(self, in_features, out_features, bias=True):
    #     super().__init__()
    #     self.real = nn.Linear(in_features, out_features, bias=bias)
    #     self.imag = nn.Linear(in_features, out_features, bias=bias)
    # def forward(self, x):
    #     return torch.complex(self.real(x.real), self.imag(x.imag))


class sarSSM(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim=3,
        state_dim=16,
        output_dim=2,
        model_dim=2,
        dropout=0.1,
        use_pos_encoding=True,
        complex_valued=True,
        use_positional_as_token=False,
        preprocess=True,
        activation_function='relu',
        use_selectivity=True
    ):
        """
        sarSSM: Sequence State Model for Synthetic Aperture Radar (SAR) Data

        This model implements a stack of S4D (Structured State Space for Deep Learning) layers
        designed for processing SAR time-series or sequential data. It supports both real and complex-valued inputs,
        optional positional encoding, and flexible input/output projections.

        Args:
            input_dim (int): Number of input features per time step.
            state_dim (int): Dimension of the internal state in S4D layers.
            output_dim (int): Number of output features per time step.
            model_dim (int): Hidden dimension for S4D layers.
            num_layers (int): Number of stacked S4D layers.
            dropout (float, optional): Dropout probability. Default is 0.1.
            use_pos_encoding (bool, optional): If True, adds positional encoding. Default is True.
            complex_valued (bool, optional): If True, processes complex-valued inputs. Default is True.
            use_positional_as_token (bool, optional): If True, treats positional encoding as a token. Default is False.
            preprocess (bool, optional): If True, applies preprocessing to inputs. Default is True.
            use_selectivity (bool, optional): If True, applies selective gating mechanism. Default is True.
            **kwargs: Additional arguments for customization.

        Inputs:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim, seq_len, [2] if complex).

        Outputs:
            out (torch.Tensor): Output tensor of shape (batch_size, output_dim, seq_len, [2] if complex).

        Features:
            - Flexible support for real and complex-valued data.
            - Optional positional encoding for sequence modeling.
            - Modular design for easy extension and integration.
            - Suitable for SAR and other sequential signal processing tasks.

        Example:
            model = sarSSM(input_dim=2, state_dim=16, output_dim=2, model_dim=64, num_layers=4)
            out = model(x)
        """
        super().__init__()
        self.complex_valued = complex_valued
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_positional_as_token = use_positional_as_token
        self.model_dim = model_dim
        self.preprocess = preprocess
        self.use_selectivity = use_selectivity
        # Input projection: project input_dim to state_dim
        if complex_valued:
            self.input_proj = ComplexLinear(input_dim, model_dim)
            self.output_proj = ComplexLinear(model_dim, output_dim)
        else:
            self.input_proj = nn.Linear(input_dim, model_dim)
            self.output_proj = nn.Linear(model_dim, output_dim)

        self.layers = nn.ModuleList([
            S4D(d_model=model_dim, d_state=state_dim, dropout=dropout, transposed=True, activation=activation_function, complex=complex_valued)
            for _ in range(num_layers)
        ])
        if complex_valued:
            self.layer_output_projs = nn.ModuleList([
                ComplexLinear(model_dim*1, model_dim) for _ in range(num_layers)
            ])
        else:
            self.layer_output_projs = nn.ModuleList([
                nn.Linear(model_dim*1, model_dim) for _ in range(num_layers)
            ])
        if complex_valued:
            self.norm = ComplexLayerNorm(model_dim)
        else:
            self.norm = nn.LayerNorm(model_dim)
        if complex_valued:
            self.dropout = ComplexDropout(dropout)
        else:
            self.dropout = nn.Dropout(dropout)
        self.use_pos_encoding = use_pos_encoding
        if self.use_selectivity:
            if complex_valued:
                # gate based on magnitude and pos: use a small real module on magnitudes
                self.gate_mlp = nn.Sequential(
                    nn.Linear(2*model_dim, model_dim),
                    nn.GELU(),
                    nn.Linear(model_dim, 1),
                )
            else:
                self.gate_mlp = nn.Sequential(
                    nn.Linear(model_dim, model_dim),
                    nn.GELU(),
                    nn.Linear(model_dim, 1),
                )
    def _preprocess_input(self, x):
        # x: (B, 1000, seq_len, 2)
        # Optionally concatenate positional embedding as next token
        if self.use_positional_as_token:
            # x[..., 0] is backscatter, x[..., 1] is positional embedding
            backscatter = x[..., 0]  # (B, 1000, seq_len)
            pos_embed = x[..., 1]    # (B, 1000, seq_len)
            # Concatenate positional embedding as next token along seq_len
            x = torch.cat([backscatter, pos_embed[:, :, :1]], dim=-1)  # (B, 1000, seq_len+1)
            x = x.permute(0, 2, 1)  # (B, seq_len+1, 1000)
        else:
            # Keep as separate channel, flatten last two dims
            x = x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)  # (B, seq_len, 1000*2)
        return x
    def _postprocess_output(self, out):
        if len(out.shape) == 3:
            out = out.unsqueeze(-1)
        return out.permute(0, 2, 1, 3)
    def extract_features(self, x):
        return self.forward(x, extract_features=True)
    def forward(self, x, extract_features=False):
        # x: (B, L, F)
        x = x.squeeze()
        #print(f"sarSSM input shape: {x.shape}")
        B, L, feat = x.shape
        feats = []
        assert feat == self.input_dim, f"Expected input feature dim {self.input_dim}, got {feat}"
        if self.complex_valued:
            if x.is_complex():
                h = self.input_proj(x)  # expects complex input
            else:
                # minimal handling: if first two dims are real/imag, stack them
                if feat >= 2:
                    ri = torch.stack([x[...,0], x[...,1]], dim=-1).reshape(B, L, -1)
                    h = self.input_proj(ri)
                else:
                    h = self.input_proj(x.float())
        else:
            h = self.input_proj(x.float())
        # to (B,H,L)
        h = h.transpose(1,2).contiguous()
        gate = None
        if self.use_selectivity:
            gate_in = h.transpose(1,2).contiguous()
            if self.complex_valued:
                mag = gate_in.abs(); ph = torch.angle(gate_in)
                gate_feat = torch.cat([mag, ph], dim=-1)
            else:
                gate_feat = gate_in
            gate_logits = self.gate_mlp(gate_feat)
            gate = torch.sigmoid(gate_logits).squeeze(-1)  # (B,L)
        for layer, proj in zip(self.layers, self.layer_output_projs):
            y, _ = layer(h)  # (B, C*H, L)
            y_t = y.transpose(1,2).contiguous()  # (B,L,C*H)
            y_p = proj(y_t)  # (B,L,H)
            y_p = y_p.transpose(1,2).contiguous()
            if gate is not None:
                g = gate.unsqueeze(1)
                h = h + y_p * g
            else:
                h = h + y_p
            
            # if self.complex_valued:
            #     h = torch.complex(torch.tanh(h.real), torch.tanh(h.imag))
            # else:
            #     h = F.gelu(h)
            if extract_features:
                feats.append(h.transpose(1,2).contiguous())
        h = h.transpose(1,2).contiguous()
        # h = self.norm(h)
        # h = self.dropout(h)
        out = self.output_proj(h)  # (B,L,output_dim)
        if extract_features:
            return out, feats
        else:
            return out

    def step(self, u, states=None):
        # single-time-step recurrent fallback (not optimized)
        if states is None:
            states = [None]*len(self.layers)
        h = self.input_proj(u)  # (B,H)
        next_states = []
        for i, layer in enumerate(self.layers):
            y, ns = layer.step(h, states[i])
            y_p = self.layer_output_projs[i](y)
            if self.use_selectivity:
                if self.complex_valued:
                    g = torch.sigmoid(h.abs().mean(-1, keepdim=True))
                else:
                    g = torch.sigmoid(h.mean(-1, keepdim=True))
                h = h + y_p * g
            else:
                h = h + y_p
            h = torch.complex(torch.tanh(h.real), torch.tanh(h.imag)) if self.complex_valued else F.gelu(h)
            next_states.append(ns)
        out = self.output_proj(h.unsqueeze(1)).squeeze(1)
        return out, next_states
    
class sarSSMFinal(nn.Module):
    def __init__(self,
                num_layers: int,
                input_dim: int = 3,
                model_dim: int = 2,
                state_dim: int = 16,
                activation_function: str = 'relu',
                output_dim: int = 2,
                transposed: bool = False,
                ):
        super(sarSSMFinal, self).__init__()
        # print(f" SARSSMFINAL: Initializing sarSSM with input_dim={input_dim}, model_dim={model_dim}, state_dim={state_dim}, output_dim={output_dim}, num_layers={num_layers}, activation_function={activation_function}")
        self.num_layers = num_layers
        self.ssm = nn.ModuleList()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.output_dim = output_dim

        # position embedding mixing
        self.fc1 = nn.Linear(input_dim, model_dim)
        
        # ssm layers
        for _ in range(num_layers):
            self.ssm.append(
                S4D(d_model=model_dim, d_state=state_dim, transposed=transposed, activation=activation_function, complex=False),                
            )

        self.fc2 = nn.Linear(model_dim, output_dim)
    def extract_features(self, x):
        return self.forward(x, extract_features=True)
    def forward(self, x, extract_features=False):    
  
        # position embedding
        # print(f"shape of x before input into first layer is: {x.shape}")
        feats = []
        x = self.fc1(x)

        for ssm in self.ssm:
            x, _ = ssm(x)
            if extract_features:
                feats.append(x)

        # print(f"shape of x before input into last layer is: {x.shape}")
        
        x = self.fc2(x)
        if extract_features:
            return x, feats
        else: 
            return x
    
    def setup_step(self):
        for ssm in self.ssm:
            ssm.setup_step()
    
    def step(self, u, state):
        '''
        this is for reference from the layer below this function:
        step one time step as recurrent model. intended to be used during validation
        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        '''
        
        if state == None:
            # create a list of proper states for each of the ssm layers
            batch_size = u.shape[0] if len(u.shape) > 1 else 1
            state = []
            for ssm in self.ssm:
                # Each SSM layer needs state with shape (batch_size, H, N)
                layer_state = ssm.default_state(batch_size)
                state.append(layer_state)
            
        u = self.fc1(u)   
        
        for i, ssm in enumerate(self.ssm):
            u, state[i] = ssm.step(u, state[i])
            
        u = self.fc2(u)
        
        return u, state

# ---------------- Overlap-save wrapper ----------------
class OverlapSaveWrapper(nn.Module):
    def __init__(self, model:nn.Module, chunk_size:int=2048, overlap:int=0, step_mode: bool=False):
        
        super().__init__()
        assert chunk_size > overlap
        self.model = model
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step_mode = step_mode

    def forward(self, x):
        # x: (B,L,F)
        if x.shape[1] == 1:
            squeeze_dim = 1
            x = x.squeeze()
        elif x.shape[2] == 1:
            squeeze_dim = 2
            x = x.squeeze()
        B, L, feat = x.shape
        step = self.chunk_size - self.overlap
        outputs = []
        starts = list(range(0, L, step))
        for start in starts:
            end = start + self.chunk_size
            if end <= L:
                chunk = x[:, start:end, :]
            else:
                pad_len = end - L
                tail = x[:, start:L, :]
                pad = torch.zeros(B, pad_len, feat, device=x.device, dtype=x.dtype)
                chunk = torch.cat([tail, pad], dim=1)
            if self.step_mode:
                assert self.overlap == 0, "Overlap not supported in step_mode"
                state = None
                for t in range(chunk.shape[1]):
                    xt = chunk[:, t:t+1, :]  # (B, 1, C)
                    out_step, state = self.model.step(xt, state)
                    if t == 0:
                        out = out_step
                    else:
                        out = torch.cat((out, out_step), dim=1)  # Concatenate along sequence dimension

            else:
                y = self.model(chunk)
            left = 0 if start==0 else self.overlap//2
            seg_start = start + left
            seg_end = min(end - (self.overlap - left), L)
            seg_len = max(0, seg_end - seg_start)
            seg = y[:, left:left+seg_len, :]
            outputs.append((seg_start, seg_end, seg))
        outF = outputs[0][2].shape[-1]
        out = x.new_zeros(B, L, outF)
        for seg_start, seg_end, seg in outputs:
            out[:, seg_start:seg_end, :] = seg
        #out = out.unsqueeze(squeeze_dim)
        return out




def hann_window(n):
    return torch.hann_window(n, periodic=False, device='cpu')

class WindowedOverlapWrapper(nn.Module):
    def __init__(self, model, chunk_size=2048, overlap=512, device=None):
        super().__init__()
        assert chunk_size > overlap
        self.model = model
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step = chunk_size - overlap
        self.win = torch.hann_window(chunk_size, periodic=False)
        self.device = device

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.squeeze(1)
        elif x.shape[2] == 1:
            x = x.squeeze(2)
        # x: (B,L,F)
        if self.device is not None:
            win = self.win.to(self.device)
        else:
            win = self.win.to(x.device)
        B, L, feat = x.shape
        if L <= self.chunk_size:
            out = self.model(x)
            return out
        starts = list(range(0, L, self.step))
        outF = None
        acc = None
        weight = None
        for start in starts:
            end = min(start + self.chunk_size, L)
            chunk = x[:, start:end, :]
            # if last chunk shorter, pad to chunk_size
            pad_right = self.chunk_size - (end - start)
            if pad_right > 0:
                chunk = F.pad(chunk, (0,0,0,pad_right))  # pad time dimension to right
            y = self.model(chunk)  # (B, chunk_size, outF)
            if pad_right > 0:
                y = y[:, :end-start, :]  # remove padding in output
            # apply window (for overlap-add we need full-size window; for shorter last frames, use partial window)
            w = win
            if y.shape[1] != self.chunk_size:
                w = win[:y.shape[1]].to(y.device)
            yw = y * w.unsqueeze(0).unsqueeze(-1)  # (B, T, F)
            if acc is None:
                outF = y.shape[-1]
                acc = torch.zeros(B, L, outF, device=y.device, dtype=y.dtype)
                weight = torch.zeros(B, L, outF, device=y.device, dtype=y.dtype)
            acc[:, start:start+y.shape[1], :] += yw
            weight[:, start:start+y.shape[1], :] += w.unsqueeze(0).unsqueeze(-1)
        # avoid division by zero
        eps = 1e-8
        out = acc / (weight + eps)
        #print(f"Wrapper output dimension: {out.shape}")
        return out
