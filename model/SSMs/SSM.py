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
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

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
        next_state = contract("h n, b h n -> b h n", self.dA, state) \
                + contract("h n, b h -> b h n", self.dB, u)
        y = contract("c h n, b h n -> b c h", self.dC, next_state)
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
            transposed=True, # axis ordering (B, L, D) or (B, D, L)
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


# ...existing code...
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
            # print(f"u_f: {u_f.shape}, k_f: {k_f.shape}")
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
        # assert not self.training

        y, next_state = self.kernel.step(u, state) # (B C H)
        y = y + u.unsqueeze(-2) * self.D
        y = rearrange(y, '... c h -> ... (c h)')
        y = self.activation(y)
        if self.transposed:
            y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
        else:
            y = self.output_linear(y)
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
        self.real = nn.Linear(in_features, out_features, bias=bias)
        self.imag = nn.Linear(in_features, out_features, bias=bias)
    def forward(self, x):
        return torch.complex(self.real(x.real), self.imag(x.imag))

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
        activation_function='gelu',
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
        # Input projection: project input_dim to state_dim
        if complex_valued:
            self.input_proj = ComplexLinear(input_dim, model_dim)
            self.output_proj = ComplexLinear(model_dim, output_dim)
        else:
            self.input_proj = nn.Linear(input_dim, model_dim)
            self.output_proj = nn.Linear(model_dim, output_dim)

        self.layers = nn.ModuleList([
            S4D(d_model=model_dim, d_state=state_dim, dropout=dropout, transposed=False, activation=activation_function, complex=complex_valued)
            for _ in range(num_layers)
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
    def forward(self, x):
        if self.preprocess:
            x = self._preprocess_input(x)  # (B, L, input_dim)
        else:
            # If preprocessing is disabled, remove one before the last dimension (vertical position)
            x = torch.cat([x[..., :-2], x[..., -1:]], dim=-1)
            if x.shape[1] == 1:
                squeeze_dim = 1
            elif x.shape[2] == 1:
                squeeze_dim = 2
            else:
                squeeze_dim= None
            x = x.squeeze()
        # Project to state_dim
        # print(f"[SSM] Input shape after preprocessing: {x.shape}")
        h = self.input_proj(x)  # (B, state_dim, L)
        # print(f"[SSM] Input shape after input projection: {h.shape}")
        # Pass through S4D layers
        for i, layer in enumerate(self.layers):
            h, _ = layer(h)  # S4D expects (B, state_dim, L)
        # Normalize and dropout
        h = self.norm(h)
        h = self.dropout(h)

        out = self.output_proj(h)  # (B, output_dim, L)
        # Return in (B, L, output_dim) format for consistency
        if self.preprocess:
            return self._postprocess_output(out)
        else:
            if squeeze_dim is not None:
                out = out.unsqueeze(squeeze_dim)
            return out
    
    def step(self, x, state):
        '''
        this is for reference from the layer below this function:
        step one time step as recurrent model. intended to be used during validation
        x: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        '''
        
        if state == None:
            # create a list of 0 states for each of the ssm layers
            state = [np.zeros(self.d_state) for _ in self.layers]
        
        x = self._preprocess_input(x)  # (B, L, input_dim)
        # Project to state_dim
        h = self.input_proj(x)  # (B, state_dim, L)
        # Pass through S4D layers
        for i, layer in enumerate(self.layers):
            h, _ = layer.step(h, state[i])  # S4D expects (B, state_dim, L)
        # Normalize and dropout
        h = self.norm(h)
        h = self.dropout(h)

        out = self.output_proj(h)  # (B, output_dim, L)
        # Return in (B, L, output_dim) format for consistency
        return self._postprocess_output(out), state



def hippo_initializer(N, dt_min=0.001, dt_max=0.1):
    """
    HiPPO-based initialization for better long-range dependencies
    """
    # Generate optimized eigenvalues based on HiPPO theory
    pi = torch.tensor(np.pi)
    real_part = 0.5 * torch.ones(N//2)
    imag_part = torch.arange(N//2)
    
    # Use inverse scaling for better long-range modeling
    imag_part = 1/pi * N * (N/(1+2*imag_part)-1)
    w = -real_part + 1j * imag_part
    
    # Generate random dt in log space
    log_dt = torch.rand(1) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
    
    return w, log_dt

def register_parameter_with_lr(module, name, tensor, trainable=True, lr=None, wd=None):
    """Utility method: register a tensor as a buffer or trainable parameter with custom LR"""
    if trainable:
        module.register_parameter(name, nn.Parameter(tensor))
    else:
        module.register_buffer(name, tensor)

    if trainable and lr is not None:
        optim = {"lr": lr}
        if wd is not None:
            optim["weight_decay"] = wd
        setattr(getattr(module, name), "_optim", optim)

class SimpleSSM(nn.Module):
    """
    Simple State Space Model (SSM) with HiPPO initialization and complex state handling.

    This class implements a simple SSM for sequence modeling, using HiPPO-based initialization for
    long-range dependencies and complex-valued parameters for improved expressivity. It supports
    both FFT-based and standard convolution for efficient computation.

    Args:
        state_dim (int): Hidden state dimension of the SSM (must be even for complex parameterization).
        L (int): Length of the learned impulse response (kernel size).
        channel_dim (int): Number of parallel SSMs (channels).
        dt_min (float): Minimum timestep for HiPPO initialization.
        dt_max (float): Maximum timestep for HiPPO initialization.
        lr (float, optional): Custom learning rate for SSM parameters.
        use_fft (bool): Whether to use FFT-based convolution for efficiency.
    """
    def __init__(self,
                 state_dim: int = 64,
                 L: int = 128,
                 channel_dim: int = 1,
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 lr: float = None,
                 use_fft: bool = True):
        """
        Initialize the EnhancedSimpleSSM module.

        Args:
            state_dim (int): Hidden state dimension of the SSM (must be even for complex parameterization).
            L (int): Length of the learned impulse response (kernel size).
            channel_dim (int): Number of parallel SSMs (channels).
            dt_min (float): Minimum timestep for HiPPO initialization.
            dt_max (float): Maximum timestep for HiPPO initialization.
            lr (float, optional): Custom learning rate for SSM parameters.
            use_fft (bool): Whether to use FFT-based convolution for efficiency.
        """
        super().__init__()
        assert state_dim % 2 == 0, "state_dim must be even for complex parameterization"
        
        self.state_dim = state_dim
        self.L = L
        self.channel_dim = channel_dim
        self.use_fft = use_fft
        
        # Initialize using HiPPO theory for each channel
        self._init_ssm_parameters(dt_min, dt_max, lr)
        
        # Learnable skip connection (D term)
        self.D = nn.Parameter(torch.randn(channel_dim) * 0.1)

    def _init_ssm_parameters(self, dt_min, dt_max, lr):
        """
        Initialize SSM parameters using HiPPO methodology for long-range memory.
        Registers complex-valued parameters for eigenvalues, input/output matrices, and timestep.
        """
        
        # Generate HiPPO-based eigenvalues for each channel
        w_list = []
        log_dt_list = []
        C_list = []
        
        for c in range(self.channel_dim):
            w, log_dt = hippo_initializer(self.state_dim, dt_min, dt_max)
            w_list.append(w)
            log_dt_list.append(log_dt)
            
            # Initialize C matrix with proper scaling
            C = torch.randn(1, self.state_dim//2, dtype=torch.cfloat) * 0.1
            C_list.append(C)
        
        # Stack and register parameters
        w_stacked = torch.stack(w_list, dim=0)  # (channel_dim, state_dim//2)
        log_dt_stacked = torch.stack(log_dt_list, dim=0)  # (channel_dim,)
        C_stacked = torch.stack(C_list, dim=0)  # (channel_dim, 1, state_dim//2)
        
        # Register complex parameters using real view
        register_parameter_with_lr(self, "w_real", w_stacked.real, True, lr)
        register_parameter_with_lr(self, "w_imag", w_stacked.imag, True, lr)
        register_parameter_with_lr(self, "log_dt", log_dt_stacked, True, lr)
        register_parameter_with_lr(self, "C", _c2r(_resolve_conj(C_stacked)), True, lr)
        
        # B parameter (input matrix) - normalized for stability
        B = torch.randn(self.channel_dim, self.state_dim//2, dtype=torch.cfloat)
        B = B / torch.norm(B, dim=-1, keepdim=True)  # Normalize for stability
        register_parameter_with_lr(self, "B", _c2r(_resolve_conj(B)), True, lr)

    def _get_ssm_params(self):
        """
        Retrieve SSM parameters in complex form for computation.
        Returns:
            w (Tensor): Complex eigenvalues.
            dt (Tensor): Discretization timesteps.
            C (Tensor): Output matrix (complex).
            B (Tensor): Input matrix (complex).
        """
        w = self.w_real + 1j * self.w_imag  # (channel_dim, state_dim//2)
        dt = torch.exp(self.log_dt)  # (channel_dim,)
        C = _r2c(self.C)  # (channel_dim, 1, state_dim//2)
        B = _r2c(self.B)  # (channel_dim, state_dim//2)
        return w, dt, C, B

    def _compute_kernel_fft(self, L, device):
        """
        Compute the SSM kernel using optimized complex arithmetic and FFT for fast convolution.
        Args:
            L (int): Kernel length.
            device: Device for computation.
        Returns:
            kernel (Tensor): Convolution kernel for sequence modeling.
        """
        w, dt, C, B = self._get_ssm_params()
        
        # Discretize: A_discrete = exp(dt * w)
        dtA = dt.unsqueeze(-1) * w  # (channel_dim, state_dim//2)
        A_discrete = torch.exp(dtA)
        
        # Compute kernel using matrix powers
        # h[k] = C * A^k * B for k = 0, 1, ..., L-1
        kernels = []
        Ak = torch.ones_like(A_discrete)  # A^0 = I
        
        for k in range(L):
            # h[k] = C @ (A^k @ B)
            AkB = Ak * B  # Broadcasting: (channel_dim, state_dim//2)
            hk = torch.sum(C.squeeze(1) * AkB, dim=-1)  # (channel_dim,)
            
            # Add D term at k=0
            if k == 0:
                hk = hk + self.D
                
            kernels.append(hk.real)  # Take real part for output
            Ak = Ak * A_discrete  # Update A^k
        
        kernel = torch.stack(kernels, dim=-1)  # (channel_dim, L)
        return kernel.unsqueeze(1)  # (channel_dim, 1, L) for conv1d

    def forward(self, azimuth_data: torch.Tensor):
        """
        Forward pass for azimuth focusing using the SSM kernel.
        Args:
            azimuth_data (Tensor): Input tensor of shape (B, channel_dim, T).
        Returns:
            focused (Tensor): Output tensor of shape (B, channel_dim, T) after SSM convolution.
        """
        B, C, T = azimuth_data.shape
        assert C == self.channel_dim, f"Expected {self.channel_dim} channels, got {C}"
        
        device = azimuth_data.device
        
        if self.use_fft and T > self.L:
            # Use FFT-based convolution for long sequences
            kernel = self._compute_kernel_fft(self.L, device)
            
            # Pad for causal convolution
            pad = self.L - 1
            x_padded = F.pad(azimuth_data, (pad, 0))
            
            # FFT-based convolution
            L_conv = x_padded.size(-1)
            x_f = torch.fft.rfft(x_padded, n=L_conv)
            k_f = torch.fft.rfft(kernel, n=L_conv)
            y_f = x_f * k_f
            focused = torch.fft.irfft(y_f, n=L_conv)[..., :T]
        else:
            # Standard convolution for shorter sequences
            kernel = self._compute_kernel_fft(min(self.L, T), device)
            pad = kernel.size(-1) - 1
            x_padded = F.pad(azimuth_data, (pad, 0))
            focused = F.conv1d(x_padded, weight=kernel, groups=self.channel_dim)
        
        return focused

class MambaModel(nn.Module):
    """
    Mamba model with S4D-inspired initialization and complex handling.

    This class implements a selective state-space model (Mamba) with HiPPO-inspired initialization,
    complex parameter support, and multiple layers for deep sequence modeling. Each range bin is
    processed independently, and the model supports residual connections and dropout.

    Args:
        input_dim (int): Input feature dimension.
        state_dim (int): State dimension for SSM.
        delta_rank (int): Rank for delta projection in selective scan.
        num_layers (int): Number of stacked Mamba blocks.
        dropout (float): Dropout rate.
        dt_min (float): Minimum timestep for initialization.
        dt_max (float): Maximum timestep for initialization.
        lr (float, optional): Custom learning rate for SSM parameters.
    """
    def __init__(self, 
                 input_dim: int = 64,
                 state_dim: int = 16, 
                 delta_rank: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 bidirectional: bool = False):
        """
        Initialize the EnhancedMambaModel module.

        Args:
            input_dim (int): Input feature dimension.
            state_dim (int): State dimension for SSM.
            delta_rank (int): Rank for delta projection in selective scan.
            num_layers (int): Number of stacked Mamba blocks.
            dropout (float): Dropout rate.
            dt_min (float): Minimum timestep for initialization.
            dt_max (float): Maximum timestep for initialization.
            bidirectional (bool): Whether to use bidirectional processing.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.delta_rank = delta_rank
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Input/output projections
        self.input_proj = nn.Linear(1, input_dim)
        self.output_proj = nn.Linear(input_dim, 1)
        
        # Enhanced Mamba blocks with proper initialization
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(input_dim),
                EnhancedMambaBlock(input_dim, state_dim, delta_rank, dt_min, dt_max),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        Forward pass for the Enhanced Mamba model.
        Processes each range bin independently through stacked Mamba blocks.
        Args:
            x (Tensor): Input tensor of shape (B, range_bins, T).
        Returns:
            output (Tensor): Output tensor of shape (B, range_bins, T).
        """
        B, R, T = x.shape
        out = []
        
        for r in range(R):
            seq = x[:, r, :].unsqueeze(-1)  # (B, T, 1)
            h = self.input_proj(seq)  # (B, T, input_dim)
            if self.bidirectional:
                h_fwd = h
                h_bwd = torch.flip(h, dims=[1])
                for block in self.layers:
                    h_fwd = block(h_fwd) + h_fwd
                    h_bwd = block(h_bwd) + h_bwd
                h_bwd = torch.flip(h_bwd, dims=[1])
                h = h_fwd + h_bwd  # or torch.cat([h_fwd, h_bwd], dim=-1)
            else:
                for block in self.layers:
                    h = block(h) + h
            out_r = self.output_proj(h).squeeze(-1)
            out.append(out_r)
        return torch.stack(out, dim=1)

class EnhancedMambaBlock(nn.Module):
    """
    Enhanced Mamba block with S4D-inspired initialization and selective scan.

    This block implements input projection, causal convolution, selective scan with HiPPO-inspired
    initialization, gating, and output projection. It is designed for use in deep Mamba models for
    sequence modeling.

    Args:
        d_model (int): Model feature dimension.
        d_state (int): State dimension for SSM.
        delta_rank (int): Rank for delta projection.
        dt_min (float): Minimum timestep for initialization.
        dt_max (float): Maximum timestep for initialization.
        lr (float, optional): Custom learning rate for SSM parameters.
    """
    def __init__(self, 
                 d_model: int,
                 d_state: int, 
                 delta_rank: int,
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 lr: float = None):
        """
        Initialize the EnhancedMambaBlock module.

        Args:
            d_model (int): Model feature dimension.
            d_state (int): State dimension for SSM.
            delta_rank (int): Rank for delta projection.
            dt_min (float): Minimum timestep for initialization.
            dt_max (float): Maximum timestep for initialization.
            lr (float, optional): Custom learning rate for SSM parameters.
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.delta_rank = delta_rank
        
        # Projections
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, groups=d_model)
        self.x_proj = nn.Linear(d_model, delta_rank + 2 * d_state, bias=False)
        self.delta_proj = nn.Linear(delta_rank, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Initialize SSM parameters using HiPPO methodology
        self._init_ssm_params(dt_min, dt_max, lr)

    def _init_ssm_params(self, dt_min, dt_max, lr):
        """
        Initialize SSM parameters for the Mamba block using HiPPO-inspired eigenvalue placement and scaling.
        Registers A matrix and D parameter for skip connection.
        """
        
        # A matrix: use HiPPO-inspired initialization
        # Instead of simple arange, use optimized eigenvalue placement
        A_real = -0.5 * torch.ones(self.d_model, self.d_state)
        A_imag = torch.arange(1, self.d_state + 1).unsqueeze(0).expand(self.d_model, -1)
        
        # Apply inverse scaling for better long-range dependencies  
        A_imag = A_imag / (1 + 2 * torch.arange(self.d_state))
        
        A_log = torch.log(A_real.abs() + 1j * A_imag)
        register_parameter_with_lr(self, "A_log", A_log.real, True, lr)
        
        # D parameter: skip connection with proper initialization
        D = torch.ones(self.d_model) * 0.1
        register_parameter_with_lr(self, "D", D, True, lr)
        
        # Initialize delta projection with proper scale
        with torch.no_grad():
            dt = torch.exp(torch.rand(self.d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
            self.delta_proj.bias.copy_(torch.log(dt))

    def forward(self, x):
        """
        Forward pass for the Enhanced Mamba block.
        Args:
            x (Tensor): Input tensor of shape (B, L, d_model).
        Returns:
            output (Tensor): Output tensor of shape (B, L, d_model).
        """
        B, L, d = x.shape
        
        # Input projection and gating
        x_res = self.in_proj(x)  # (B, L, 2*d_model)
        x, res = x_res.chunk(2, dim=-1)  # Each: (B, L, d_model)
        
        # Convolution (causal)
        x = self.conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x = F.silu(x)
        
        # SSM computation
        y = self._run_ssm(x)
        
        # Gating and output projection
        y = y * F.silu(res)
        return self.out_proj(y)

    def _run_ssm(self, u):
        """
        Run the selective scan SSM computation for the input sequence.
        Args:
            u (Tensor): Input tensor of shape (B, L, d_model).
        Returns:
            output (Tensor): Output tensor of shape (B, L, d_model).
        """
        B, L, d = u.shape
        
        # Project to get delta, B, C
        delta_BC = self.x_proj(u)  # (B, L, delta_rank + 2*d_state)
        delta, B, C = delta_BC.split([self.delta_rank, self.d_state, self.d_state], dim=-1)
        
        # Compute delta with proper activation and bias
        delta = F.softplus(self.delta_proj(delta))  # (B, L, d_model)
        
        # Enhanced selective scan with better stability
        return selective_scan_fn(u, delta, self.A_log, B, C, self.D) #self._selective_scan_enhanced(u, delta, self.A_log, B, C, self.D)


# Usage example and model configurations
class ModelArgs:
    """
    Enhanced model arguments container with better defaults for SSM and Mamba models.
    """
    def __init__(self):
        self.input_dim = 64
        self.state_dim = 16 
        self.delta_rank = 8
        self.num_layers = 4
        self.dropout = 0.1
        self.seq_length = 128
        self.dt_min = 0.001
        self.dt_max = 0.1
        self.lr_ssm = 1e-3  # Custom LR for SSM parameters