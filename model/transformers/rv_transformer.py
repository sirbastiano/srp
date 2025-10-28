import torch
import torch.nn as nn
import math
import numpy as np

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, dim]

    def forward(self, x):
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1)]
        return x

class RealValuedTransformer(nn.Module):
    """
    A Transformer-based model for SAR focusing that works with row/column sequences.
    Optimized for compressing raw SAR data and focusing it with a decoder.
    
    Args:
        input_dim (int): Dimensionality of input features (typically 2 for real/imag or 3 with pos encoding)
        model_dim (int): Dimensionality of the transformer model (embedding size)
        num_layers (int): Number of layers in both encoder and decoder
        num_heads (int): Number of attention heads in each layer
        ff_dim (int): Dimensionality of the feedforward network
        dropout (float): Dropout rate applied in transformer layers
        mode (str): Processing mode ("parallel" or "autoregressive")
        max_seq_len (int): Maximum sequence length for positional encoding
        compression_ratio (float): Ratio for compressing the sequence length
    """

    def __init__(
            self, 
            input_dim: int = 2, 
            model_dim: int = 64, 
            num_layers: int = 4, 
            num_heads: int = 8, 
            ff_dim: int = 256, 
            dropout: float = 0.1, 
            mode: str = "parallel", 
            max_seq_len: int = 5000,
            compression_ratio: float = 0.1,
            verbose: bool = False
        ):
        super().__init__()
        
        # Store configuration
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.mode = mode
        self.verbose = verbose
        self.compression_ratio = compression_ratio
        self.compressed_dim = max(int(model_dim * compression_ratio), 32)
        
        # Input/output projections
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.output_proj = nn.Linear(model_dim, input_dim)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(model_dim, max_len=max_seq_len)
        
        # Encoder layers for compression
        self.encoder_layers = nn.ModuleList([])
        for _ in range(num_layers):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            )
            self.encoder_layers.append(encoder_layer)
        
        # Compression layer - reduces sequence length
        self.compressor = nn.Sequential(
            nn.Linear(model_dim, self.compressed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Decoder layers for reconstruction
        self.decoder_layers = nn.ModuleList([])
        for _ in range(num_layers):
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            )
            self.decoder_layers.append(decoder_layer)
        
        # Learnable query embeddings for decoding
        self.register_parameter('query_embedding', nn.Parameter(torch.randn(max_seq_len, model_dim) * 0.02))
        
        # Expansion layer - restores from compressed representation
        self.expander = nn.Sequential(
            nn.Linear(self.compressed_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(model_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
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
            "pos_encoding": self.pos_enc.pe.numel(),
        }
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "breakdown": breakdown,
            "memory_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        }
    
    def extract_features(self, x, layer_idx=None):
        """
        Extract intermediate features for knowledge distillation.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            layer_idx: Which layer to extract features from (None = middle layer)
            
        Returns:
            Tuple of (output, intermediate_features)
        """
        # Get the full forward pass output
        output = self.forward(x)
        
        # For feature extraction, we'll extract from encoder layers
        x = self.preprocess_input(x)
        x_proj = self.input_proj(x)
        x_pos = self.pos_enc(x_proj)
        
        # Extract features from specified layer (or middle layer if not specified)
        if layer_idx is None:
            layer_idx = len(self.encoder_layers) // 2
        
        encoded = x_pos
        for i, layer in enumerate(self.encoder_layers):
            encoded = layer(encoded)
            if i == layer_idx:
                features = encoded.clone()
                break
        else:
            # If layer_idx is out of range, use the last layer
            features = encoded
        
        return output, features

    def preprocess_input(self, x):
        """
        Preprocess input to handle SAR data format.
        Expected: [batch_size, seq_len, input_dim] where seq_len=1000 for rows
        """
        if self.verbose:
            print(f"Input shape: {x.shape}")
        
        # Handle different input shapes
        if len(x.shape) == 4:
            # [B, seq_len, 1, channels] -> [B, seq_len, channels]
            if x.shape[2] == 1:
                x = x.squeeze(2)
            else:
                # [B, rows, cols, channels] -> [B, rows, cols*channels] (flatten spatial)
                batch_size, rows, cols, channels = x.shape
                x = x.view(batch_size, rows, cols * channels)
        
        # Handle input dimension mismatch - take only the required channels
        if x.shape[-1] > self.input_dim:
            # Take only the first input_dim channels (e.g., real/imag from real/imag/pos_y/pos_x)
            x = x[..., :self.input_dim]
            if self.verbose:
                print(f"Truncated input from {x.shape[-1] + (x.shape[-1] - self.input_dim)} to {self.input_dim} channels")
        elif x.shape[-1] < self.input_dim:
            # Pad if needed
            padding = torch.zeros(*x.shape[:-1], self.input_dim - x.shape[-1], device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=-1)
            if self.verbose:
                print(f"Padded input from {x.shape[-1] - (self.input_dim - x.shape[-1])} to {self.input_dim} channels")
        
        return x

    def encode_sequence(self, x):
        """
        Encode input sequence through transformer encoder layers.
        
        Args:
            x: Input tensor [batch_size, seq_len, model_dim]
            
        Returns:
            Encoded tensor [batch_size, seq_len, model_dim]
        """
        encoded = x
        for layer in self.encoder_layers:
            encoded = layer(encoded)
        return encoded
    
    def decode_sequence(self, queries, memory):
        """
        Decode queries using memory through transformer decoder layers.
        
        Args:
            queries: Query tensor [batch_size, seq_len, model_dim]
            memory: Memory tensor [batch_size, seq_len, model_dim]
            
        Returns:
            Decoded tensor [batch_size, seq_len, model_dim]
        """
        decoded = queries
        for layer in self.decoder_layers:
            decoded = layer(decoded, memory)
        return decoded    
    def _autoregressive_inference(self, memory):
        """
        Autoregressive inference: generate target sequence step by step.
        
        Args:
            memory: Encoder output [B, seq_len, model_dim]
            
        Returns:
            Generated sequence [B, seq_len, model_dim]
        """
        batch_size, seq_len, model_dim = memory.shape
        device = memory.device
        
        # Start with zeros (could be learned start token)
        tgt = torch.zeros(batch_size, 1, self.input_dim, device=device)
        
        generated = []
        
        for i in range(seq_len):
            # Project current target
            tgt_proj = self.input_proj(tgt)  # [B, current_len, model_dim]
            tgt_pos = self.pos_enc(tgt_proj)
            
            # Generate causal mask for autoregressive decoding
            tgt_len = tgt_pos.size(1)
            causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device) * float('-inf'), diagonal=1)
            
            # Decode with causal attention using our decoder layers
            decoded = tgt_pos
            for layer in self.decoder_layers:
                # Note: PyTorch transformer decoder layers expect (tgt, memory, tgt_mask)
                decoded = layer(decoded, memory, tgt_mask=causal_mask)
            
            # Get the last token prediction
            next_token = decoded[:, -1:, :]  # [B, 1, model_dim]
            generated.append(next_token)
            
            # Project back to input space for next iteration
            next_input = self.output_proj(next_token)  # [B, 1, input_dim]
            
            # Concatenate for next iteration
            tgt = torch.cat([tgt, next_input], dim=1)  # [B, current_len+1, input_dim]
        
        # Concatenate all generated tokens
        return torch.cat(generated, dim=1)  # [B, seq_len, model_dim]

    def forward(self, x, y=None):
        """
        Forward pass for SAR focusing.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim] (real)
            y: Optional target for teacher forcing (not used in parallel mode)
            
        Returns:
            Output tensor [batch_size, seq_len, input_dim] (real)
        """
        # Preprocess input
        x = self.preprocess_input(x)
        batch_size, seq_len, _ = x.shape
        
        if self.verbose:
            print(f"Processing real sequence of length {seq_len} in {self.mode} mode")
        
        # 1. Project to model dimension
        x_proj = self.input_proj(x)  # [B, seq_len, model_dim]
        
        # 2. Add positional encoding
        x_pos = self.pos_enc(x_proj)
        
        # 3. Encode through transformer layers
        encoded = self.encode_sequence(x_pos)  # [B, seq_len, model_dim]
        
        if self.mode == "autoregressive":
            # Use autoregressive decoding
            output_features = self._autoregressive_inference(encoded)
            decoded = self.final_norm(output_features)
            output = self.output_proj(decoded)  # [B, seq_len, input_dim]
        else:
            # Parallel mode (default)
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
            decoded = self.decode_sequence(queries, memory)  # [B, seq_len, model_dim]
            
            # 8. Final normalization and projection
            decoded = self.final_norm(decoded)
            output = self.output_proj(decoded)  # [B, seq_len, input_dim]
        
        if self.verbose:
            print(f"Output shape: {output.shape}")
            if self.mode == "parallel":
                print(f"Compression ratio achieved: {pooled.shape[1] / seq_len:.3f}")
        
        return output