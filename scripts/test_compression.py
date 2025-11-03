#!/usr/bin/env python3
"""
Test script to demonstrate encoder/decoder compression with mock data from YAML config.
Usage:
  python scripts/test_compression.py training/training_configs/spatial_transformer.yaml
"""
import sys
import yaml
import torch
from pathlib import Path

def test_encoder_decoder_compression(config_path: Path):
    """Test the separated encoder and decoder with compression analysis."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    dataloader_config = config['dataloader']
    
    # Import here to avoid path issues
    sys.path.append(str(Path(__file__).parent.parent))
    from model.transformers.spatial_transformer import ScalePreservingSpatialTransformer
    
    # Create model from config
    model_params = {
        'input_channels': int(model_config.get('input_channels', 2)),
        'output_channels': int(model_config.get('output_channels', 2)),
        'embed_dim': int(model_config.get('embed_dim', 256)),
        'num_layers': int(model_config.get('num_layers', 6)),
        'num_heads': int(model_config.get('num_heads', 8)),
        'mlp_ratio': float(model_config.get('mlp_ratio', 2.0)),
        'patch_size': tuple(model_config.get('patch_size', [100, 20])),
        'max_height': int(model_config.get('max_height', 5000)),
        'max_width': int(model_config.get('max_width', 100)),
        'dropout': float(model_config.get('dropout', 0.0)),
        'output_mode': model_config.get('output_mode', 'complex')
    }
    
    model = ScalePreservingSpatialTransformer(**model_params)
    
    # Create mock data based on config
    input_patch_size = tuple(dataloader_config.get('patch_size', [5000, 100]))
    batch_size = 2  # Small batch for testing
    input_channels = model_params['input_channels']
    
    # Use smaller size for testing if too large
    H, W = input_patch_size
    if H > 1000:
        H = 1000  # Reduce for faster testing
    if W > 100:
        W = 100
    
    print(f"🧪 Testing Encoder/Decoder Compression")
    print(f"=" * 50)
    print(f"Config: {config_path}")
    print(f"Input size: {H}×{W}×{input_channels}")
    print(f"Patch size: {model_params['patch_size']}")
    print(f"Embed dim: {model_params['embed_dim']}")
    
    # Generate mock SAR data (complex-valued)
    mock_data = torch.randn(batch_size, H, W, input_channels)
    print(f"\n📊 Mock Data:")
    print(f"  Shape: {mock_data.shape}")
    print(f"  Memory: {mock_data.numel() * 4 / (1024**2):.2f} MB")
    print(f"  Std: {mock_data.std().item():.4f}")
    
    with torch.no_grad():
        # Test encoder
        print(f"\n🔄 Encoding...")
        tokens, token_h, token_w = model.encode(mock_data)
        
        print(f"  Encoded tokens shape: {tokens.shape}")
        print(f"  Token grid: {token_h}×{token_w} = {token_h * token_w} tokens")
        print(f"  Token memory: {tokens.numel() * 4 / (1024**2):.2f} MB")
        print(f"  Token std: {tokens.std().item():.4f}")
        
        # Calculate compression metrics
        input_elements = mock_data.numel()
        token_elements = tokens.numel()
        compression_ratio = input_elements / token_elements
        spatial_compression = (H * W) / (token_h * token_w)
        
        print(f"\n📈 Compression Metrics:")
        print(f"  Input elements: {input_elements:,}")
        print(f"  Token elements: {token_elements:,}")
        print(f"  Overall compression: {compression_ratio:.2f}x")
        print(f"  Spatial compression: {spatial_compression:.1f}x")
        print(f"  Memory reduction: {(mock_data.numel() * 4) / (tokens.numel() * 4):.2f}x")
        
        # Test decoder
        print(f"\n🔄 Decoding...")
        reconstructed = model.decode(tokens, token_h, token_w, mock_data.shape)
        
        print(f"  Reconstructed shape: {reconstructed.shape}")
        print(f"  Shape preservation: {reconstructed.shape == mock_data.shape}")
        print(f"  Reconstructed std: {reconstructed.std().item():.4f}")
        
        # Test full forward pass
        print(f"\n🔄 Full Forward Pass...")
        full_output = model(mock_data)
        print(f"  Full output shape: {full_output.shape}")
        print(f"  Output matches decode: {torch.allclose(full_output, reconstructed, atol=1e-6)}")
        
        # Reconstruction quality
        mse = torch.nn.functional.mse_loss(reconstructed, mock_data)
        print(f"\n🎯 Reconstruction Quality:")
        print(f"  MSE Loss: {mse.item():.6f}")
        print(f"  Input/Output std ratio: {reconstructed.std().item() / mock_data.std().item():.3f}")
        
    print(f"\n✅ Test completed successfully!")
    print(f"   Encoder: {H}×{W}×{input_channels} → {token_h}×{token_w}×{model_params['embed_dim']}")
    print(f"   Decoder: {token_h}×{token_w}×{model_params['embed_dim']} → {H}×{W}×{input_channels}")
    print(f"   Compression: {compression_ratio:.2f}x ({spatial_compression:.1f}x spatial)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/test_compression.py <config.yaml>')
        print('Example: python scripts/test_compression.py training/training_configs/spatial_transformer.yaml')
        sys.exit(2)
    
    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f'Config file not found: {config_path}')
        sys.exit(2)
    
    test_encoder_decoder_compression(config_path)