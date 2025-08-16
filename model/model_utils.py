from model.transformers.rv_transformer import RealValuedTransformer  # your import
from model.transformers.cv_transformer import ComplexTransformer
from model.SSMs.SSM import MambaModel, SimpleSSM, sarSSM, S4D
import torch
import torch.nn as nn

class SARTransformerFactory:
    @staticmethod
    def create(
        real: bool = False,
        seq_len: int = 512,
        input_dim: int = 1,
        model_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1,
        depth: int = 4,
        heads: int = 8,
        dim_head: int = 32,
        ff_mult: int = 4,
        relu_squared: bool = True,
        complete_complex: bool = False,
        rotary: bool = True,
        flash_attn: bool = True,
        num_tokens: int = None,
        causal: bool = False,
        mode: str = "parallel"
    ):
        if real:
            model = RealValuedTransformer(
                input_dim=input_dim,
                model_dim=model_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            )
            #model.mode = mode  # Set mode if needed
            return model
        else:
            return ComplexTransformer(
                dim=model_dim,
                depth=depth,
                num_tokens=num_tokens,
                causal=causal,
                dim_head=dim_head,
                heads=heads,
                ff_mult=ff_mult,
                complete_complex=complete_complex,
                rotary=rotary,
                flash_attn=flash_attn,
                use_data_positions=False,  
                pos_encoding_type='complex' 
            )

class SARSSMFactory:
    """
    Factory class for creating different types of State Space Model (SSM) architectures
    for SAR focusing applications.
    """
    
    @staticmethod
    def create_simple_ssm(
        input_dim: int = 4,
        state_dim: int = 64,
        output_dim: int = 2,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_pos_encoding: bool = True,
        **kwargs
    ) -> nn.Module:
        """
        Create a SimpleSSM model for column-wise SAR focusing.
        
        Args:
            input_dim: Input feature dimension (typically 4 for real, imag, pos_y, pos_x)
            state_dim: Hidden state dimension for the SSM
            output_dim: Output dimension (typically 2 for real, imag)
            num_layers: Number of SSM layers to stack
            dropout: Dropout rate
            use_pos_encoding: Whether to use positional encoding
            
        Returns:
            Configured SimpleSSM model
        """
        # Create wrapper model with proper input/output projections
        class SimpleSSMWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.use_pos_encoding = use_pos_encoding
                
                # Input projection
                if use_pos_encoding:
                    self.input_proj = nn.Linear(input_dim, 2)  # Project to 2D for real/imag
                else:
                    self.input_proj = nn.Linear(input_dim - 2, 2)  # Exclude pos encoding
                
                # SSM layers
                self.ssm_layers = nn.ModuleList([
                    SimpleSSM(
                        state_dim=state_dim,
                        L=kwargs.get('seq_length', 1000),
                        channel_dim=2,  # For real and imaginary parts
                        dt_min=kwargs.get('dt_min', 0.001),
                        dt_max=kwargs.get('dt_max', 0.1),
                        lr=kwargs.get('lr_ssm', None),
                        use_fft=kwargs.get('use_fft', True)
                    )
                    for _ in range(num_layers)
                ])
                
                # Output projection
                self.output_proj = nn.Linear(2, output_dim)
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                # x shape: (B, T, input_dim) for column-wise processing
                if self.use_pos_encoding:
                    # Use all features including positional encoding
                    h = self.input_proj(x)  # (B, T, 2)
                else:
                    # Exclude positional encoding (last 2 dims)
                    h = self.input_proj(x[..., :-2])  # (B, T, 2)
                
                # Transpose for SSM: (B, 2, T)
                h = h.transpose(1, 2)
                
                # Apply SSM layers
                for ssm in self.ssm_layers:
                    h_new = ssm(h)
                    h = h + h_new  # Residual connection
                    h = self.dropout(h)
                
                # Transpose back and project output
                h = h.transpose(1, 2)  # (B, T, 2)
                return self.output_proj(h)  # (B, T, output_dim)
        
        return SimpleSSMWrapper()
    
    @staticmethod
    def create_mamba_ssm(
        input_dim: int = 4,
        state_dim: int = 256,
        output_dim: int = 2,
        num_layers: int = 8,
        dropout: float = 0.1,
        expansion_factor: int = 2,
        use_pos_encoding: bool = True,
        **kwargs
    ) -> nn.Module:
        """
        Create a Mamba SSM model for selective state space modeling.
        
        Args:
            input_dim: Input feature dimension
            state_dim: Hidden state dimension
            output_dim: Output dimension
            num_layers: Number of Mamba layers
            dropout: Dropout rate
            expansion_factor: Expansion factor for the model
            use_pos_encoding: Whether to use positional encoding
            
        Returns:
            Configured MambaModel
        """
        # Calculate model dimension based on expansion factor
        model_dim = state_dim * expansion_factor
        
        # Create wrapper for input/output handling
        class MambaSSMWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.use_pos_encoding = use_pos_encoding
                
                # Input projection
                proj_input_dim = input_dim if use_pos_encoding else input_dim - 2
                self.input_proj = nn.Linear(proj_input_dim, model_dim)
                
                # Mamba model
                self.mamba = MambaModel(
                    input_dim=model_dim,
                    state_dim=state_dim,
                    delta_rank=kwargs.get('delta_rank', state_dim // 16),
                    num_layers=num_layers,
                    dropout=dropout,
                    dt_min=kwargs.get('dt_min', 0.001),
                    dt_max=kwargs.get('dt_max', 0.1),
                    lr=kwargs.get('lr_ssm', None)
                )
                
                # Output projection
                self.output_proj = nn.Linear(model_dim, output_dim)
                
            def forward(self, x):
                # Handle input projection
                if self.use_pos_encoding:
                    h = self.input_proj(x)
                else:
                    h = self.input_proj(x[..., :-2])
                
                # Process through Mamba (expects (B, range_bins, T))
                # For column-wise processing, we treat each column as a range bin
                B, T, D = h.shape
                h = h.unsqueeze(1)  # (B, 1, T, D) - single range bin
                h = h.transpose(2, 3)  # (B, 1, D, T)
                h = h.squeeze(1)  # (B, D, T)
                
                # Transpose for Mamba input format (B, T, D)
                h = h.transpose(1, 2)  # (B, T, D)
                
                # Apply Mamba (note: need to adapt for single range bin)
                # Create temporary expanded format for Mamba
                h_expanded = h.unsqueeze(1)  # (B, 1, T, D)
                h_expanded = h_expanded.transpose(1, 2)  # (B, T, 1, D)
                h_expanded = h_expanded.squeeze(-1)  # (B, T, D)
                
                # Process each sequence
                outputs = []
                for i in range(h.shape[0]):  # Batch dimension
                    seq = h[i:i+1]  # (1, T, D)
                    out = self.mamba.input_proj(seq)  # (1, T, model_dim)
                    
                    for block in self.mamba.layers:
                        out = block(out) + out  # Residual connection
                    
                    out = self.mamba.output_proj(out)  # (1, T, 1)
                    outputs.append(out.squeeze(-1))  # (1, T)
                
                result = torch.stack(outputs, dim=0)  # (B, T)
                return self.output_proj(h)  # Use original projection
        
        return MambaSSMWrapper()
    
    @staticmethod
    def create_s4_ssm(
        input_dim: int = 4,
        state_dim: int = 512,
        output_dim: int = 2,
        num_layers: int = 12,
        dropout: float = 0.2,
        use_pos_encoding: bool = True,
        **kwargs
    ) -> nn.Module:
        """
        Create an S4D-based SSM model for long-range sequence modeling.
        
        Args:
            input_dim: Input feature dimension
            state_dim: Hidden state dimension (N in S4D terminology)
            output_dim: Output dimension
            num_layers: Number of S4D layers
            dropout: Dropout rate
            use_pos_encoding: Whether to use positional encoding
            
        Returns:
            Configured S4D-based model
        """
        class S4SSMWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.use_pos_encoding = use_pos_encoding
                
                # Model dimensions
                proj_input_dim = input_dim if use_pos_encoding else input_dim - 2
                d_model = kwargs.get('d_model', 64)
                
                # Input projection
                self.input_proj = nn.Linear(proj_input_dim, d_model)
                
                # S4D layers
                self.s4_layers = nn.ModuleList([
                    S4D(
                        d_model=d_model,
                        d_state=state_dim,
                        channels=kwargs.get('channels', 1),
                        bidirectional=kwargs.get('bidirectional', False),
                        activation=kwargs.get('activation', 'gelu'),
                        dropout=dropout,
                        transposed=False,  # Use (B, L, H) format
                        **{k: v for k, v in kwargs.items() if k.startswith('kernel_')}
                    )
                    for _ in range(num_layers)
                ])
                
                # Layer normalization for each S4D layer
                self.layer_norms = nn.ModuleList([
                    nn.LayerNorm(d_model) for _ in range(num_layers)
                ])
                
                # Output projection
                self.output_proj = nn.Linear(d_model, output_dim)
                self.final_dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                # Input projection
                if self.use_pos_encoding:
                    h = self.input_proj(x)  # (B, T, d_model)
                else:
                    h = self.input_proj(x[..., :-2])  # (B, T, d_model)
                
                # Apply S4D layers with residual connections
                for s4_layer, norm in zip(self.s4_layers, self.layer_norms):
                    # Pre-norm residual connection
                    h_norm = norm(h)
                    h_out, _ = s4_layer(h_norm)
                    h = h + h_out  # Residual connection
                    h = self.final_dropout(h)
                
                # Output projection
                return self.output_proj(h)  # (B, T, output_dim)
        
        return S4SSMWrapper()

def create_ssm_model(
    model_type: str,
    input_dim: int = 4,
    state_dim: int = 64,
    output_dim: int = 2,
    num_layers: int = 6,
    dropout: float = 0.1,
    use_pos_encoding: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create SSM models based on configuration.
    
    Args:
        model_type: Type of SSM model ('simple', 'mamba', 's4')
        input_dim: Input feature dimension
        state_dim: Hidden state dimension
        output_dim: Output dimension
        num_layers: Number of layers
        dropout: Dropout rate
        use_pos_encoding: Whether to use positional encoding
        **kwargs: Additional model-specific parameters
        
    Returns:
        Configured SSM model
        
    Raises:
        ValueError: If model_type is not supported
    """
    if model_type == "simple":
        return SARSSMFactory.create_simple_ssm(
            input_dim=input_dim,
            state_dim=state_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_pos_encoding=use_pos_encoding,
            **kwargs
        )
    elif model_type == "mamba":
        return SARSSMFactory.create_mamba_ssm(
            input_dim=input_dim,
            state_dim=state_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_pos_encoding=use_pos_encoding,
            **kwargs
        )
    elif model_type == "s4":
        return SARSSMFactory.create_s4_ssm(
            input_dim=input_dim,
            state_dim=state_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_pos_encoding=use_pos_encoding,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported SSM model type: {model_type}. Supported types: 'simple', 'mamba', 's4'")
            
models = {
    "rv_transformer": RealValuedTransformer,
    "cv_transformer": ComplexTransformer, 
    "ssm": MambaModel,
    "simple_ssm": lambda **kwargs: create_ssm_model("simple", **kwargs),
    "mamba_ssm": lambda **kwargs: create_ssm_model("mamba", **kwargs),
    "s4_ssm": lambda **kwargs: create_ssm_model("s4", **kwargs)
}

def create_sar_complex_transformer(
    input_dim: int = 4,
    model_dim: int = 64,
    num_layers: int = 4,
    num_heads: int = 8,
    dim_head: int = 32,
    ff_dim: int = 256,
    dropout: float = 0.1,
    use_data_positions: bool = True,
    pos_encoding_type: str = 'complex',
    complete_complex: bool = False,
    mode: str = "parallel",
    **kwargs
):
    """
    Factory function to create a ComplexTransformer optimized for SAR data.
    
    Args:
        input_dim: Input feature dimension (typically 4 for real, imag, pos_y, pos_x)
        model_dim: Model hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dim_head: Dimension per attention head
        ff_dim: Feed-forward hidden dimension
        dropout: Dropout rate
        use_data_positions: Whether to use positional encoding from input data
        pos_encoding_type: Type of positional encoding ('complex', 'concat', 'add')
        complete_complex: Whether to use full complex attention vs real-channel flattening
        mode: Processing mode (for compatibility)
        **kwargs: Additional arguments
        
    Returns:
        Configured ComplexTransformer for SAR data processing
    """
    # Adjust transformer dimensions based on input configuration
    if use_data_positions and input_dim >= 3:
        # If we're using positional encoding from data, the actual data dimension
        # is reduced (positions are handled separately)
        data_dim = max(1, input_dim - 2) if input_dim > 2 else 1
        transformer_dim = max(data_dim, model_dim // 4)  # Ensure reasonable minimum
    else:
        transformer_dim = input_dim if input_dim > 0 else model_dim
    
    return ComplexTransformer(
        dim=transformer_dim,
        depth=num_layers,
        heads=num_heads,
        dim_head=dim_head,
        ff_mult=max(1, ff_dim // transformer_dim),
        complete_complex=complete_complex,
        rotary=(not use_data_positions),  # Use rotary only if not using data positions
        flash_attn=kwargs.get('flash_attn', True),
        causal=(mode == "autoregressive"),
        use_data_positions=use_data_positions,
        pos_encoding_type=pos_encoding_type
    )

def get_model_from_configs(
        name: str,
        dim_head: int,
        seq_len: int = 256,
        input_dim: int = 1, 
        model_dim: int = 64, 
        num_layers: int = 4, 
        num_heads: int = 4, 
        ff_dim: int = 128, 
        dropout: float = .1, 
        lr: float = 1e-4, 
        mode: str = "parallel",
        **kwargs
    ):
    if name == "cv_transformer":
        # For complex-valued transformer, configure dimensions properly
        # input_dim=4 (real, imag, pos_y, pos_x) maps to 2 complex features
        # complex_dim = max(1, input_dim // 2) if input_dim > 2 else model_dim
        
        # Check if we have positional data in input
        use_data_positions = kwargs.get('use_data_positions', True)  # Default to True for SAR data
        pos_encoding_type = kwargs.get('pos_encoding_type', 'complex')  # Default encoding type
        
        # Adjust input dimension if using positional encoding from data
        # if use_data_positions and input_dim >= 3:
        #     # Assume last 2 dimensions are positions, so adjust dim for the complex transformer
        #     actual_dim = input_dim - 2 if input_dim > 2 else input_dim
        #     # For complex data, we typically have 1 complex value + 2 position dims
        #     # So the transformer dim should be set appropriately
        #     transformer_dim = max(1, actual_dim) if actual_dim > 0 else model_dim
        # else:
        #     transformer_dim = input_dim
        transformer_dim = input_dim
        
        model = ComplexTransformer(
            dim=transformer_dim,
            depth=num_layers,
            heads=num_heads,
            dim_head=dim_head,
            ff_mult=max(1, ff_dim // transformer_dim),
            complete_complex=kwargs.get('complete_complex', False),  # Use real-channel flattening approach
            rotary=(not use_data_positions),  # Use rotary only if not using data positions
            flash_attn=kwargs.get('flash_attn', True),  # Use flash attention if available
            causal=(mode == "autoregressive"),
            use_data_positions=use_data_positions,
            pos_encoding_type=pos_encoding_type,
            #**{k: v for k, v in kwargs.items() if k not in ['complete_complex', 'flash_attn', 'use_data_positions', 'pos_encoding_type', 'dim']}
        )
    elif name=="rv_transformer":
        model = RealValuedTransformer(
            input_dim=input_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            mode=mode  
        )
    elif name in ["simple_ssm", "mamba_ssm", "s4_ssm"]:
        # Extract SSM type from name
        ssm_type = name.split("_")[0]  # "simple", "mamba", or "s4"
        
        model = create_ssm_model(
            model_type=ssm_type,
            input_dim=input_dim,
            state_dim=model_dim,
            output_dim=kwargs.get('output_dim', 2),
            num_layers=num_layers,
            dropout=dropout,
            use_pos_encoding=kwargs.get('use_pos_encoding', True),
            **kwargs
        )
    elif name=="ssm":
        # Legacy support - defaults to simple SSM
        model = create_ssm_model(
            model_type="simple",
            input_dim=input_dim,
            state_dim=model_dim,
            output_dim=kwargs.get('output_dim', 2),
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        raise ValueError(f"Invalid model name: {name}")
    return model


