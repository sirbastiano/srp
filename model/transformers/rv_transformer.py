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
    A Transformer-based model for processing real-valued sequential data.
    Args:
        seq_len (int): Length of input sequences.
        input_dim (int, optional): Dimensionality of input features. Default is 1.
        model_dim (int, optional): Dimensionality of the transformer model. Default is 64.
        num_layers (int, optional): Number of layers in both encoder and decoder. Default is 4.
        num_heads (int, optional): Number of attention heads in each layer. Default is 4.
        ff_dim (int, optional): Dimensionality of the feedforward network. Default is 128.
        dropout (float, optional): Dropout rate applied in transformer layers. Default is 0.1.
    Attributes:
        input_proj (nn.Linear): Linear layer to project input to model dimension.
        output_proj (nn.Linear): Linear layer to project model output back to input dimension.
        encoder (nn.TransformerEncoder): Transformer encoder module.
        decoder (nn.TransformerDecoder): Transformer decoder module.
    Methods:
        forward(src, tgt=None):
            Forward pass through the model.
            Args:
                src (Tensor): Source input tensor of shape [batch_size, input_dim, seq_len].
                tgt (Tensor, optional): Target input tensor for autoregressive decoding.
            Returns:
                Tensor: Output tensor of shape [batch_size, input_dim, seq_len].
            Notes:
                - In "parallel" mode, decoding is performed in one shot.
                - In autoregressive mode, target input is required for step-wise decoding.
    """

    def __init__(
            self, 
            input_dim:int=2, 
            model_dim:int=1000, 
            num_layers:int=4, 
            num_heads:int=4, 
            ff_dim:int=128, 
            dropout:float=0.1, 
            mode: str = "parallel", 
            verbose: bool = False
        ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.output_proj = nn.Linear(model_dim, input_dim)
        self.mode = mode
        self.verbose = verbose

        #self.pos_enc = PositionalEncoding(model_dim, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.register_parameter('query_embedding', nn.Parameter(torch.randn(5000, model_dim) * 0.1))

    def preprocess_input_shape(self, x):
        """
        Preprocess input tensor to handle various input shapes.
        
        Cases handled:
        1. [B, T, 1, C] -> [B, T, C] (remove singleton dimension)
        2. [B, T, N, C] -> [B, N, T*C] (treat each of N vectors as sequence elements)
        3. [B, T, C] -> [B, T, C] (no change needed)
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor of shape [B, seq_len, features]
        """
        if self.verbose:
            print(f"Original input shape: {x.shape}")
        
        if len(x.shape) == 4:
            batch_size, vector_len, num_vectors, channels = x.shape
            
            if num_vectors == 1:
                # Case 1: [B, T, 1, C] -> [B, T, C]
                x = x.squeeze(2)
                if self.verbose:
                    print(f"Removed singleton dimension: {x.shape}")
            else:
                # Case 2: [B, 1000, 79, 3] -> [B, 79, 1000*3]
                # Treat each of the 79 vectors as a sequence element
                # Each element has 1000*3=3000 features
                x = x.permute(0, 2, 1, 3)  # [B, 79, 1000, 3]
                pos_embedding = x[..., -2:]
                x = x[..., :-2]
                
                x = x + pos_embedding
                x = x.contiguous().view(batch_size, num_vectors, vector_len * (channels - 2))
                if self.verbose:
                    print(f"Reshaped to sequence format: {x.shape}")
        
        elif len(x.shape) == 3:
            # Case 3: [B, T, C] - already correct
            if self.verbose:
                print(f"Shape already correct: {x.shape}")

        else:
            raise ValueError(f"Unsupported input shape: {x.shape}. Expected 3D or 4D tensor.")
        return x

    def process_output_shape(self, output, original_shape):
        """
        Process output tensor to restore original structure.
        
        Args:
            output: Output tensor of shape [B, seq_len, features]
            original_shape: Original input shape for reference
            
        Returns:
            Processed tensor matching original structure
        """
        if self.verbose:
            print(f"Processing output shape: {output.shape}")
        
        if len(original_shape) == 4:
            batch_size, vector_len, num_vectors, channels = original_shape
            current_batch, seq_len, features = output.shape
            
            if seq_len == num_vectors and features == vector_len * channels:
                # Reshape back to [B, 1000, 79, 3]
                output = output.view(batch_size, num_vectors, vector_len, channels)
                output = output.permute(0, 2, 1, 3)  # [B, 1000, 79, 3]
                #output = output[..., :2]
                if self.verbose:
                    print(f"Restored original structure: {output.shape}")
            else:
                # Fallback: add singleton dimension
                output = output.unsqueeze(2)
                if features == vector_len * (channels - 2):
                    output = output.view(batch_size, num_vectors, vector_len, (channels-2))
                else:
                    output = output.view(batch_size, num_vectors, vector_len, -1) 
                output = output.permute(0, 2, 1, 3)  # [B, 1000, 79, 2]
                #output = output[..., :2]
                if self.verbose:
                    print(f"Added singleton dimension: {output.shape}")

        elif len(original_shape) == 3:
            # Keep as is for 3D inputs
            if self.verbose:
                print(f"Keeping 3D output: {output.shape}")

        return output

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
        tgt = torch.zeros(batch_size, 1, self.input_proj.in_features, device=device)
        
        generated = []
        
        for i in range(seq_len):
            # Project current target
            tgt_proj = self.input_proj(tgt)  # [B, current_len, model_dim]
            
            # Generate causal mask for autoregressive decoding
            tgt_len = tgt_proj.size(1)
            causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device) * float('-inf'), diagonal=1)
            
            # Decode with causal attention
            decoder_output = self.decoder(
                tgt_proj, 
                memory, 
                tgt_mask=causal_mask
            )  # [B, current_len, model_dim]
            
            # Get the last token prediction
            next_token = decoder_output[:, -1:, :]  # [B, 1, model_dim]
            generated.append(next_token)
            
            # Project back to input space for next iteration
            next_input = self.output_proj(next_token)  # [B, 1, input_dim]
            
            # Concatenate for next iteration
            tgt = torch.cat([tgt, next_input], dim=1)  # [B, current_len+1, input_dim]
        
        # Concatenate all generated tokens
        return torch.cat(generated, dim=1)  # [B, seq_len, model_dim]

    def forward(self, src, tgt=None):
        # Store original shape for output processing
        original_src_shape = src.shape
        
        # Preprocess input shapes: [B, 1000, 79, 4] -> [B, 79, 4000]
        src = self.preprocess_input_shape(src)
        
        # Project and encode: [B, 79, 4000] -> [B, 79, model_dim]
        src_proj = self.input_proj(src)
        memory = self.encoder(src_proj)        # [B, 79, model_dim]

        if self.mode == "encoder_only":
            # Encoder-only mode: direct transformation
            output = memory
            
        elif self.mode == "autoregressive":
            # Autoregressive encoder-decoder mode
            if tgt is not None:
                # TRAINING: Teacher forcing with ground truth
                tgt = self.preprocess_input_shape(tgt)
                tgt_proj = self.input_proj(tgt)
                
                # Create causal mask for autoregressive training
                seq_len = tgt_proj.size(1)
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=tgt_proj.device) * float('-inf'), diagonal=1)
                
                output = self.decoder(tgt_proj, memory, tgt_mask=causal_mask)
            else:
                # INFERENCE: Autoregressive generation step by step
                output = self._autoregressive_inference(memory)
                
        elif self.mode in  ["encoder_decoder", "parallel"]:
            # Non-autoregressive encoder-decoder mode
            if tgt is not None:
                # TRAINING: Teacher forcing with ground truth (no causal mask)
                tgt = self.preprocess_input_shape(tgt)
                tgt_proj = self.input_proj(tgt)
                output = self.decoder(tgt_proj, memory)  # No causal mask - parallel decoding
            else:
                # INFERENCE: Use learned query vectors
                batch_size, seq_len = memory.shape[:2]
                queries = self.query_embedding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
                output = self.decoder(queries, memory)
                
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Project back to original feature space: [B, 79, model_dim] -> [B, 79, 4000]
        transf_output = self.output_proj(output)
        
        # Restore original shape: [B, 79, 4000] -> [B, 1000, 79, 4] -> [B, 1000, 79, 2]
        return self.process_output_shape(transf_output, original_src_shape)