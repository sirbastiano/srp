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
from typing import Dict, Any
import os
from pytorch_lightning.loggers import TensorBoardLogger

from dataloader.dataloader import SampleFilter, get_sar_dataloader, SARTransform
from model.model_utils import get_model_from_configs
from training.training_loops import get_training_loop_by_model_name
from training.visualize import save_results_and_metrics

def setup_logging(out_file : str = 'training.log', model_name: str = 'model', exp_dir: str = './results'):
    tb_logger = TensorBoardLogger(save_dir=exp_dir, name=model_name)
    log_file = os.path.join(exp_dir, out_file)
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    return tb_logger, logging.getLogger(__name__)


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
    if args.mode:
        config.setdefault('training', {})['mode'] = args.mode
    if args.device:
        config.setdefault('training', {})['device'] = args.device
    if args.batch_size:
        config.setdefault('training', {})['batch_size'] = args.batch_size
        # Also update dataloader batch sizes
        for split in ['train', 'validation', 'test']:
            if 'dataloader' in config and split in config['dataloader']:
                config['dataloader'][split]['batch_size'] = args.batch_size
    if args.learning_rate:
        config.setdefault('training', {})['learning_rate'] = args.learning_rate
    if args.num_epochs:
        config.setdefault('training', {})['num_epochs'] = args.num_epochs
    if args.save_dir:
        config.setdefault('training', {})['save_dir'] = args.save_dir
    
    # Ensure required sections exist
    config.setdefault('model', {})
    config.setdefault('training', {})
    config.setdefault('dataloader', {})
    
    # Handle legacy configuration structure - map top-level to training section
    legacy_mappings = {
        'epochs': 'num_epochs',
        'lr': 'learning_rate', 
        'mode': 'mode',
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
        'stride': [1000, 1],
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
            'samples_per_prod': 1000,
            'patch_order': 'row',
            'max_products': 1,
            'filters': {"years": [2023]}
        },
        'validation': {
            'batch_size': config['training'].get('batch_size', 16),
            'samples_per_prod': 500,
            'patch_order': 'row',
            'max_products': 1,
            'filters': {"years": [2024]}
        },
        'test': {
            'batch_size': max(1, config['training'].get('batch_size', 16) // 2),
            'samples_per_prod': 200,
            'patch_order': 'row',
            'max_products': 1,
            'filters': {"years": [2025]}
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
    print(config)
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
            adaptive = transforms_cfg.get('adaptive', False)
            complex_valued = transforms_cfg.get('complex_valued', True)
            
            transforms = SARTransform.create_minmax_normalized_transform(
                normalize=True,
                adaptive=adaptive,
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
        'batch_size': split_cfg.get('batch_size', 16),
        'samples_per_prod': split_cfg.get('samples_per_prod', 1000),
        'patch_order': split_cfg.get('patch_order', 'row'),
        'max_products': split_cfg.get('max_products', 1),
        'filters': split_cfg.get('filters', {})
    }
    
    # Merge configurations
    final_config = {**base_config, **split_config}
    print(final_config)
    return get_sar_dataloader(**final_config)


def create_dataloaders(dataloader_cfg):
    """Create train, validation, and test dataloaders from configuration."""
    # Get data directory from dataloader config or use default
    data_dir = dataloader_cfg.get('data_dir', '/Data/sar_focusing_new')
    # Create transforms
    transforms_cfg = dataloader_cfg.get('transforms', {})
    transforms = create_transforms_from_config(transforms_cfg)
    
    # Create train loader
    train_cfg = dataloader_cfg.get('train', {})
    train_filters = train_cfg.get('filters', {})
    train_cfg['filters'] = SampleFilter(**train_filters) if train_filters else None
    train_loader = create_dataloader_from_config(
        data_dir=data_dir,
        dataloader_cfg=dataloader_cfg,
        split_cfg=train_cfg,
        transforms=transforms
    )
    
    # Create validation loader
    val_cfg = dataloader_cfg.get('validation', {})
    val_filters = val_cfg.get('filters', {})
    val_cfg['filters'] = SampleFilter(**val_filters) if val_filters else None
    val_loader = create_dataloader_from_config(
        data_dir=data_dir,
        dataloader_cfg=dataloader_cfg,
        split_cfg=val_cfg,
        transforms=transforms
    )
    
    # Create test loader (no transforms for test as in original code)
    test_cfg = dataloader_cfg.get('test', {})
    test_filters = test_cfg.get('filters', {})
    test_cfg['filters'] = SampleFilter(**test_filters) if test_filters else None
    test_loader = create_dataloader_from_config(
        data_dir=data_dir,
        dataloader_cfg=dataloader_cfg,
        split_cfg=test_cfg,
        transforms=transforms
    )
    
    if 'inference' not in dataloader_cfg:
        return train_loader, val_loader, test_loader, None
    else: 
        inference_cfg = dataloader_cfg.get('inference', {})
        inference_filters = inference_cfg.get('filters', {})
        inference_cfg['filters'] = SampleFilter(**inference_filters) if inference_filters else None
        inference_loader = create_dataloader_from_config(
            data_dir=data_dir,
            dataloader_cfg=dataloader_cfg,
            split_cfg=inference_cfg,
            transforms=transforms
        )
        return train_loader, val_loader, test_loader, inference_loader



def main():
    """Main training function with enhanced SAR patch processing support."""
    parser = argparse.ArgumentParser(description='Train models with optional SAR patch preprocessing')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--experiment_name', type=str, default='experiment', help='Experiment name to use')
    parser.add_argument('--model_name', type=str, default='model', help='Model name to use')
    parser.add_argument('--mode', type=str, default=None, help='Training mode override')
    parser.add_argument('--device', type=str, default=None, help='Device override')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size override')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate override')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of epochs override')
    parser.add_argument('--save_dir', type=str, default="./results", help='Save directory override')
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['plateau', 'cosine'], help='LR scheduler type')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(Path(args.config), args)
        
    # Setup logging
    tb_logger, text_logger = setup_logging(model_name=args.experiment_name, exp_dir=args.save_dir)
    text_logger.info(f"Starting training with config: {args.config}")

    # Extract configurations
    model_cfg = config['model']
    dataloader_cfg = config['dataloader']
    training_cfg = config['training']
    
    # Log configuration summary
    text_logger.info("Full Configuration:")
    for section, section_cfg in config.items():
        text_logger.info(f"  [{section}]")
        if isinstance(section_cfg, dict):
            for key, value in section_cfg.items():
                text_logger.info(f"    {key}: {value}")
        else:
            text_logger.info(f"    {section_cfg}")
    # text_logger.info("Configuration Summary:")
    # text_logger.info(f"  Model: {model_cfg.get('name', 'Unknown')}")
    # text_logger.info(f"  Mode: {training_cfg.get('mode', 'parallel')}")
    # text_logger.info(f"  Device: {training_cfg.get('device', 'cuda')}")
    # text_logger.info(f"  Batch size: {training_cfg.get('batch_size', 32)}")
    # text_logger.info(f"  Learning rate: {training_cfg.get('learning_rate', 1e-4)}")
    # text_logger.info(f"  Epochs: {training_cfg.get('num_epochs', 100)}")
    # text_logger.info(f"  Save directory: {training_cfg.get('save_dir', './results')}")
    # text_logger.info(f"  Scheduler type: {training_cfg.get('scheduler_type', 'cosine')}")
    # text_logger.info(f"  Patience: {training_cfg.get('patience', 10)}")
    # ###### TODO: FINISH LOGGING ALL CONFIGS ######
    
    # Check if patch preprocessing is needed
    
    model = get_model_from_configs(**model_cfg)
    text_logger.info(f"Created standard model: {model_cfg.get('name', 'Unknown')}")
    
    # Create dataloaders
    text_logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader, inference_loader = create_dataloaders(dataloader_cfg)
    text_logger.info(f"Created dataloaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}, Inference: {len(inference_loader)}")

    # Determine trainer class and create trainer
    lightning_model, trainer = get_training_loop_by_model_name(
        model_cfg.get('name', ''), 
        model=model, 
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        inference_loader=inference_loader,
        save_dir=training_cfg.get('save_dir', './results'), 
        loss_fn_name=training_cfg.get('loss_fn', 'mse'),
        mode=training_cfg.get('mode', 'parallel'),
        scheduler_type=training_cfg.get('scheduler_type', 'cosine'), 
        logger=tb_logger,
    )
    # Start training
    text_logger.info("Starting training process...")
    try:
        trainer.fit(lightning_model)
        
    except Exception as e:
        text_logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == '__main__':
    main()