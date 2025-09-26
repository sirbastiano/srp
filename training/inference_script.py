import sys 
import os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import time
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

from dataloader.dataloader import get_sar_dataloader, SARTransform
from model.model_utils import get_model_from_configs
from training.training_loops import TrainRVTransformer, TrainCVTransformer, TrainSSM
from sarpyx.utils.losses import get_loss_function
import numpy as np
def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('visualization.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: Path, args):
    """Load and merge configuration from YAML file with command line arguments."""
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Add data_dir to dataloader config if it exists at top level
    if 'data_dir' in config and 'dataloader' in config:
        config['dataloader']['data_dir'] = config['data_dir']
    
    # Add transforms to dataloader if it exists at top level
    if 'transforms' in config and 'dataloader' in config:
        config['dataloader']['transforms'] = config['transforms']
    
    # Apply command line overrides
    if args.device:
        config.setdefault('training', {})['device'] = args.device
    if args.batch_size:
        config.setdefault('training', {})['batch_size'] = args.batch_size
        # Also update dataloader batch sizes
        for split in ['train', 'validation', 'test']:
            if 'dataloader' in config and split in config['dataloader']:
                config['dataloader'][split]['batch_size'] = args.batch_size
    if args.save_dir:
        config.setdefault('training', {})['save_dir'] = args.save_dir
    
    # Ensure required sections exist
    config.setdefault('model', {})
    config.setdefault('training', {})
    config.setdefault('dataloader', {})
    
    # Handle legacy configuration structure - map top-level to training section
    legacy_mappings = {
        'device': 'device',
        'base_save_dir': 'save_dir',
        'batch_size': 'batch_size'
    }
    
    for old_key, new_key in legacy_mappings.items():
        if old_key in config:
            config['training'].setdefault(new_key, config[old_key])
    
    # Map model configuration properly
    model_mappings = {
        'seq_len': 'seq_len',
        'input_dim': 'input_dim', 
        'model_dim': 'dim',  # Map model_dim to dim for ComplexTransformer
        'num_layers': 'depth',  # Map num_layers to depth for ComplexTransformer
        'num_heads': 'heads',
        'ff_dim': 'ff_mult',  # This will need conversion: ff_mult = ff_dim // model_dim
        'dropout': 'dropout',
        'dim_head': 'dim_head',
        'pos_encoding_type': 'pos_encoding_type'
    }
    
    # Apply model mappings and conversions
    for yaml_key, model_key in model_mappings.items():
        if yaml_key in config['model']:
            if yaml_key == 'ff_dim':
                # Convert ff_dim to ff_mult ratio
                ff_dim = config['model'][yaml_key]
                model_dim = config['model'].get('model_dim', config['model'].get('dim', 512))
                config['model'][model_key] = max(1, ff_dim // model_dim)
            else:
                config['model'][model_key] = config['model'][yaml_key]
    
    # Ensure dataloader has proper structure
    if 'dataloader' not in config:
        config['dataloader'] = {}
    
    # Set dataloader defaults if missing
    dataloader_defaults = {
        'level_from': 'rcmc',
        'level_to': 'az', 
        'num_workers': 0,
        'patch_mode': 'rectangular',
        'patch_size': [1000, 1],
        'buffer': [1000, 1000],
        'stride': [300, 1],
        'shuffle_files': False,
        'complex_valued': True,
        'save_samples': False,
        'backend': 'zarr',
        'verbose': False,
        'cache_size': 1000,
        'online': False,
        'concatenate_patches': True,
        'concat_axis': 0,
        'positional_encoding': True
    }
    
    for key, default_value in dataloader_defaults.items():
        config['dataloader'].setdefault(key, default_value)
    
    # Ensure splits exist with defaults
    split_defaults = {
        'train': {
            'batch_size': config['training'].get('batch_size', 16),
            'samples_per_prod': 100,
            'patch_order': 'row',
            'max_products': 1,
            'pattern': '*2023*.zarr'
        },
        'validation': {
            'batch_size': config['training'].get('batch_size', 16),
            'samples_per_prod': 100,
            'patch_order': 'row',
            'max_products': 1,
            'pattern': '*2024*.zarr'
        },
        'test': {
            'batch_size': max(1, config['training'].get('batch_size', 16) // 2),
            'samples_per_prod': 50,
            'patch_order': 'row',
            'max_products': 1,
            'pattern': '*2025*.zarr'
        }
    }
    
    for split_name, split_config in split_defaults.items():
        config['dataloader'].setdefault(split_name, {})
        for key, default_value in split_config.items():
            config['dataloader'][split_name].setdefault(key, default_value)
    
    # Add transforms configuration if missing
    if 'transforms' not in config['dataloader']:
        config['dataloader']['transforms'] = config.get('transforms', {
            'normalize': False,
            'complex_valued': True
        })
    
    print(f"Loaded configuration: {config}")
    return config


def create_transforms_from_config(transforms_cfg):
    """Create SARTransform from configuration."""
    try:
        from dataloader.utils import RC_MIN, RC_MAX, GT_MIN, GT_MAX
        
        if transforms_cfg.get('normalize', True):
            # Use config values if provided, otherwise use defaults from utils
            rc_min = transforms_cfg.get('rc_min', RC_MIN)
            rc_max = transforms_cfg.get('rc_max', RC_MAX)
            gt_min = transforms_cfg.get('gt_min', GT_MIN)
            gt_max = transforms_cfg.get('gt_max', GT_MAX)
            complex_valued = transforms_cfg.get('complex_valued', True)
            
            transforms = SARTransform.create_minmax_normalized_transform(
                normalize=True,
                rc_min=rc_min,
                rc_max=rc_max,
                gt_min=gt_min,
                gt_max=gt_max,
                complex_valued=complex_valued
            )
        else:
            transforms = SARTransform()  # Uses identity transforms
    except ImportError:
        # Fallback if utils are not available
        transforms = SARTransform()
    
    return transforms


def create_dataloader_from_config(data_dir, dataloader_cfg, split_cfg, transforms):
    """Create a dataloader from base config and split-specific config."""
    # Base dataloader configuration
    base_config = {
        'data_dir': data_dir,
        'level_from': dataloader_cfg.get('level_from', 'rcmc'),
        'level_to': dataloader_cfg.get('level_to', 'az'),
        'num_workers': dataloader_cfg.get('num_workers', 0),
        'patch_mode': dataloader_cfg.get('patch_mode', 'rectangular'),
        'patch_size': tuple(dataloader_cfg.get('patch_size', [1000, 1])),
        'buffer': tuple(dataloader_cfg.get('buffer', [0, 0])),
        'stride': tuple(dataloader_cfg.get('stride', [300, 1])),
        'max_base_sample_size': dataloader_cfg.get('max_base_sample_size', (50000, 10000)),
        'shuffle_files': dataloader_cfg.get('shuffle_files', False),
        'complex_valued': dataloader_cfg.get('complex_valued', False),
        'save_samples': dataloader_cfg.get('save_samples', False),
        'backend': dataloader_cfg.get('backend', 'zarr'),
        'verbose': dataloader_cfg.get('verbose', True),
        'cache_size': dataloader_cfg.get('cache_size', 1000),
        'online': dataloader_cfg.get('online', True),
        'concatenate_patches': dataloader_cfg.get('concatenate_patches', True),
        'concat_axis': dataloader_cfg.get('concat_axis', 0),
        'positional_encoding': dataloader_cfg.get('positional_encoding', True),
        'transform': transforms
    }
    
    # Split-specific configuration
    split_config = {
        'block_pattern': split_cfg.get('block_pattern', None),
        'batch_size': split_cfg.get('batch_size', 16),
        'samples_per_prod': split_cfg.get('samples_per_prod', 100),
        'patch_order': split_cfg.get('patch_order', 'row'),
        'max_products': split_cfg.get('max_products', 1)
    }
    
    # Merge configurations
    final_config = {**base_config, **split_config}
    print(f"Creating dataloader with config: {final_config}")
    return get_sar_dataloader(**final_config)


def create_test_dataloader(dataloader_cfg):
    """Create test dataloader for visualization."""
    # Get data directory from dataloader config or use default
    data_dir = dataloader_cfg.get('data_dir', '/Data/sar_focusing')
    
    # Create transforms
    transforms_cfg = dataloader_cfg.get('transforms', {})
    transforms = create_transforms_from_config(transforms_cfg)
    
    # Create test loader with minimal transforms
    test_cfg = dataloader_cfg.get('inference', {}) #'test', {})
    print(f"Test config: {test_cfg}")
    test_loader = create_dataloader_from_config(
        data_dir=data_dir,
        dataloader_cfg=dataloader_cfg,
        split_cfg=test_cfg,
        transforms=transforms
    )
    
    return test_loader

def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize SAR data samples from test dataloader')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cpu', help='Device override (cpu/cuda)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size override')
    parser.add_argument('--save_dir', type=str, default='./visualizations', help='Save directory for visualizations')
    parser.add_argument('--max_batches', type=int, default=5, help='Maximum number of batches to visualize')
    parser.add_argument('--max_samples_per_batch', type=int, default=4, help='Maximum samples per batch')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting visualization with config: {args.config}")
    
    # Load configuration
    config = load_config(Path(args.config), args)
    
    # Extract configurations
    dataloader_cfg = config['dataloader']
    training_cfg = config.get('training', {})
    
    # Override save directory
    save_dir = args.save_dir or training_cfg.get('save_dir', './visualizations')
    
    # Log configuration summary
    logger.info("Configuration Summary:")
    logger.info(f"  Data directory: {dataloader_cfg.get('data_dir', 'Not specified')}")
    logger.info(f"  Level from: {dataloader_cfg.get('level_from', 'rcmc')}")
    logger.info(f"  Level to: {dataloader_cfg.get('level_to', 'az')}")
    logger.info(f"  Patch size: {dataloader_cfg.get('patch_size', [1000, 1])}")
    logger.info(f"  Batch size: {dataloader_cfg.get('test', {}).get('batch_size', 'Not specified')}")
    logger.info(f"  Save directory: {save_dir}")
    
    # Create test dataloader
    logger.info("Creating test dataloader...")
    try:
        print(dataloader_cfg)
        test_loader = create_test_dataloader(dataloader_cfg)
        logger.info(f"Created test dataloader with {len(test_loader)} batches")
        logger.info(f"Dataset contains {len(test_loader.dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to create test dataloader: {str(e)}")
        raise
    
    # Visualize samples
    logger.info("Starting sample visualization...")
    try:
        visualize_batch_samples(
            test_loader=test_loader,
            save_dir=save_dir,
            max_batches=args.max_batches,
            max_samples_per_batch=args.max_samples_per_batch
        )
        
        logger.info("Visualization completed successfully!")
        logger.info(f"Check the visualizations in: {save_dir}")
        
    except Exception as e:
        logger.error(f"Visualization failed with error: {str(e)}")
        raise


if __name__ == '__main__':
    main()