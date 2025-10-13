#!/usr/bin/env python3
"""
Enhanced training script with support for:
1. Flexible parameter passing using **kwargs
2. WandB parameter sweeps 
3. Parallel training execution
4. Configuration file saving for each run
"""

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
import json
import itertools
import subprocess
import concurrent.futures
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import wandb
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from dataloader.dataloader import SampleFilter, get_sar_dataloader, SARTransform
from model.model_utils import get_model_from_configs, create_model_with_pretrained
from training.training_loops import get_training_loop_by_model_name


def generate_sweep_configs(base_config: Dict[str, Any], sweep_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate all possible combinations of sweep parameters.
    
    Args:
        base_config: Base configuration dictionary
        sweep_params: Dictionary with parameter sweeps in format:
                     {"training.lr": [1e-4, 1e-3, 1e-2], 
                      "model.dim": [64, 128, 256]}
    
    Returns:
        List of configuration dictionaries for each sweep combination
    """
    if not sweep_params:
        return [base_config]
    
    # Parse sweep parameters
    param_keys = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    
    # Generate all combinations
    combinations = list(itertools.product(*param_values))
    
    configs = []
    for combo in combinations:
        config = json.loads(json.dumps(base_config))  # Deep copy
        
        for param_key, param_value in zip(param_keys, combo):
            # Handle nested parameter paths like "training.lr" or "model.dim"
            keys = param_key.split('.')
            current_dict = config
            
            # Navigate to the correct nested dictionary
            for key in keys[:-1]:
                if key not in current_dict:
                    current_dict[key] = {}
                current_dict = current_dict[key]
            
            # Set the final value
            current_dict[keys[-1]] = param_value
        
        configs.append(config)
    
    return configs


def create_wandb_sweep_config(sweep_params: Dict[str, Any], method: str = "grid") -> Dict[str, Any]:
    """
    Create WandB sweep configuration from parameter sweep dictionary.
    
    Args:
        sweep_params: Parameter sweep configuration
        method: Sweep method ('grid', 'random', 'bayes')
    
    Returns:
        WandB sweep configuration
    """
    wandb_params = {}
    
    for param_key, param_values in sweep_params.items():
        if isinstance(param_values, list):
            wandb_params[param_key] = {"values": param_values}
        elif isinstance(param_values, dict):
            # Support for distribution-based sweeps
            wandb_params[param_key] = param_values
        else:
            wandb_params[param_key] = {"value": param_values}
    
    sweep_config = {
        "method": method,
        "metric": {
            "name": "val_loss",
            "goal": "minimize"
        },
        "parameters": wandb_params
    }
    
    return sweep_config


def setup_logging(
    out_file: str = 'training.log', 
    model_name: str = 'model', 
    exp_dir: str = './results', 
    use_wandb: bool = True, 
    wandb_project: str = 'ssm4sar', 
    wandb_entity: Optional[str] = None, 
    wandb_tags: List[str] = ['training'], 
    config: Optional[Dict[str, Any]] = None,
    sweep_id: Optional[str] = None
) -> tuple:
    """Enhanced logging setup with sweep support."""
    
    if use_wandb:
        if sweep_id:
            # If we're in a sweep, wandb.init will be called by the sweep agent
            run_name = f'{model_name}_{exp_dir}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{sweep_id}'
        else:
            run_name = f'{model_name}_{exp_dir}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        tb_logger = WandbLogger(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            tags=wandb_tags,
            config=config
        )
    else: 
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


def load_config(config_path: Path, args) -> Dict[str, Any]:
    """Load configuration with sweep support."""
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Apply the same configuration processing as before
    # Add data_dir to dataloader config if it exists at top level
    if 'data_dir' in config and 'dataloader' in config:
        config['dataloader']['data_dir'] = config['data_dir']
    
    # Add transforms to dataloader if it exists at top level
    if 'transforms' in config and 'dataloader' in config:
        config['dataloader']['transforms'] = config['transforms']
    
    # Ensure required sections exist
    config.setdefault('model', {})
    config.setdefault('training', {})
    config.setdefault('dataloader', {})
    
    # Handle sweep configuration
    if 'sweep' in config:
        sweep_config = config.pop('sweep')
        config['_sweep_params'] = sweep_config
    
    # Set dataloader defaults (same as original)
    dataloader_defaults = {
        'level_from': 'rc',
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
    
    # Set split defaults (same as original)
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
    
    return config


def create_transforms_from_config(transforms_cfg):
    """Create SARTransform from configuration (same as original)."""
    try:
        from dataloader.utils import RC_MIN, RC_MAX, GT_MIN, GT_MAX
        
        if transforms_cfg.get('normalize', True):
            normalization_type = transforms_cfg.get('normalization_type', 'minmax')
            complex_valued = transforms_cfg.get('complex_valued', True)
            adaptive = transforms_cfg.get('adaptive', False)
            
            if normalization_type == 'minmax':
                rc_min = transforms_cfg.get('rc_min', RC_MIN)
                rc_max = transforms_cfg.get('rc_max', RC_MAX)
                gt_min = transforms_cfg.get('gt_min', GT_MIN)
                gt_max = transforms_cfg.get('gt_max', GT_MAX)
                
                transforms = SARTransform.create_minmax_normalized_transform(
                    normalize=True,
                    adaptive=adaptive,
                    rc_min=rc_min,
                    rc_max=rc_max,
                    gt_min=gt_min,
                    gt_max=gt_max,
                    complex_valued=complex_valued
                )
                
            elif normalization_type in ['zscore', 'standardize']:
                if adaptive:
                    transforms = SARTransform.create_zscore_normalized_transform(
                        normalize=True,
                        adaptive=True,
                        complex_valued=complex_valued
                    )
                else:
                    rc_mean = transforms_cfg.get('rc_mean', 0.0)
                    rc_std = transforms_cfg.get('rc_std', 1.0)
                    gt_mean = transforms_cfg.get('gt_mean', 0.0)
                    gt_std = transforms_cfg.get('gt_std', 1.0)
                    
                    transforms = SARTransform.create_zscore_normalized_transform(
                        normalize=True,
                        adaptive=False,
                        rc_mean=rc_mean,
                        rc_std=rc_std,
                        gt_mean=gt_mean,
                        gt_std=gt_std,
                        complex_valued=complex_valued
                    )
            
            elif normalization_type == 'robust':
                transforms = SARTransform.create_robust_normalized_transform(
                    normalize=True,
                    adaptive=adaptive,
                    complex_valued=complex_valued
                )
            
            else:
                raise ValueError(f"Unsupported normalization_type: {normalization_type}")
        else:
            transforms = SARTransform()
            
    except ImportError as e:
        print(f"Warning: Could not import normalization utilities: {e}")
        transforms = SARTransform()
    
    return transforms


def create_dataloader_from_config(data_dir, dataloader_cfg, split_cfg, transforms):
    """Create a dataloader from base config and split-specific config (same as original)."""
    base_config = {
        'data_dir': data_dir,
        'level_from': dataloader_cfg.get('level_from', 'rcmc'),
        'level_to': dataloader_cfg.get('level_to', 'az'),
        'num_workers': dataloader_cfg.get('num_workers', 0),  # Default to 0 to prevent worker crashes
        'patch_mode': dataloader_cfg.get('patch_mode', 'rectangular'),
        'patch_size': tuple(dataloader_cfg.get('patch_size', [1000, 1])),
        'buffer': tuple(dataloader_cfg.get('buffer', [0, 0])),
        'stride': tuple(dataloader_cfg.get('stride', [300, 1])),
        'shuffle_files': dataloader_cfg.get('shuffle_files', False),
        'complex_valued': dataloader_cfg.get('complex_valued', False),
        'save_samples': dataloader_cfg.get('save_samples', False),
        'backend': dataloader_cfg.get('backend', 'zarr'),
        'verbose': dataloader_cfg.get('verbose', True),
        'cache_size': dataloader_cfg.get('cache_size', 100),  # Reduced cache size
        'online': dataloader_cfg.get('online', True),
        'concatenate_patches': dataloader_cfg.get('concatenate_patches', True),
        'concat_axis': dataloader_cfg.get('concat_axis', 0),
        'positional_encoding': dataloader_cfg.get('positional_encoding', True),
        'transform': transforms,
        'block_pattern': split_cfg.get('block_pattern', None)
    }
    
    split_config = {
        'batch_size': split_cfg.get('batch_size', 16),
        'samples_per_prod': split_cfg.get('samples_per_prod', 1000),
        'patch_order': split_cfg.get('patch_order', 'row'),
        'max_products': split_cfg.get('max_products', 1),
        'filters': split_cfg.get('filters', {})
    }
    
    final_config = {**base_config, **split_config}
    return get_sar_dataloader(**final_config)


def create_dataloaders(dataloader_cfg):
    """Create train, validation, and test dataloaders (same as original)."""
    data_dir = dataloader_cfg.get('data_dir', '/Data/sar_focusing_new')
    transforms_cfg = dataloader_cfg.get('transforms', {})
    transforms = create_transforms_from_config(transforms_cfg)
    
    train_cfg = dataloader_cfg.get('train', {})
    train_filters = train_cfg.get('filters', {})
    train_cfg['filters'] = SampleFilter(**train_filters) if train_filters else None
    train_loader = create_dataloader_from_config(data_dir, dataloader_cfg, train_cfg, transforms)
    
    val_cfg = dataloader_cfg.get('validation', {})
    val_filters = val_cfg.get('filters', {})
    val_cfg['filters'] = SampleFilter(**val_filters) if val_filters else None
    val_loader = create_dataloader_from_config(data_dir, dataloader_cfg, val_cfg, transforms)
    
    test_cfg = dataloader_cfg.get('test', {})
    test_filters = test_cfg.get('filters', {})
    test_cfg['filters'] = SampleFilter(**test_filters) if test_filters else None
    test_loader = create_dataloader_from_config(data_dir, dataloader_cfg, test_cfg, transforms)
    
    if 'inference' not in dataloader_cfg:
        return train_loader, val_loader, test_loader, None
    else: 
        inference_cfg = dataloader_cfg.get('inference', {})
        inference_filters = inference_cfg.get('filters', {})
        inference_cfg['filters'] = SampleFilter(**inference_filters) if inference_filters else None
        inference_loader = create_dataloader_from_config(data_dir, dataloader_cfg, inference_cfg, transforms)
        return train_loader, val_loader, test_loader, inference_loader


def save_config_to_results(config: Dict[str, Any], save_dir: str, run_id: Optional[str] = None) -> str:
    """
    Save the configuration used for training to the results directory.
    
    Args:
        config: Configuration dictionary
        save_dir: Directory to save results
        run_id: Optional run identifier
    
    Returns:
        Path to saved config file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_id:
        config_filename = f"config_{run_id}_{timestamp}.yaml"
    else:
        config_filename = f"config_{timestamp}.yaml"
    
    config_path = os.path.join(save_dir, config_filename)
    
    # Remove sweep parameters from saved config as they're not needed
    config_to_save = {k: v for k, v in config.items() if not k.startswith('_')}
    
    with open(config_path, 'w') as f:
        yaml.dump(config_to_save, f, default_flow_style=False, indent=2)
    
    return config_path


def train_single_config(
    config: Dict[str, Any], 
    base_save_dir: str,
    run_id: Optional[str] = None,
    sweep_id: Optional[str] = None,
    use_wandb: bool = True
) -> Dict[str, Any]:
    """
    Train a model with a specific configuration.
    
    Args:
        config: Full configuration dictionary
        base_save_dir: Base directory for saving results
        run_id: Unique run identifier
        sweep_id: WandB sweep ID if running in sweep mode
        use_wandb: Whether to use WandB logging
        
    Returns:
        Dictionary with training results
    """
    # Extract configurations
    model_cfg = config['model']
    dataloader_cfg = config['dataloader']
    training_cfg = config['training']
    
    # Create unique save directory for this run
    if run_id:
        save_dir = os.path.join(base_save_dir, f"run_{run_id}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(base_save_dir, f"run_{timestamp}")
    
    training_cfg['save_dir'] = save_dir
    
    # Save configuration for this run
    config_path = save_config_to_results(config, save_dir, run_id)
    
    # Setup logging
    tb_logger, text_logger = setup_logging(
        model_name=training_cfg.get('save_dir', 'model'),
        exp_dir=save_dir,
        use_wandb=use_wandb,
        wandb_project=training_cfg.get('wandb_project', 'ssm4sar'),
        wandb_entity=training_cfg.get('wandb_entity', None),
        wandb_tags=training_cfg.get('wandb_tags', ['training']),
        config=config,
        sweep_id=sweep_id
    )
    
    text_logger.info(f"Starting training with run_id: {run_id}")
    text_logger.info(f"Configuration saved to: {config_path}")
    
    # try:
    # Create model with all model config parameters
    if 'pretrained_path' in model_cfg and model_cfg['pretrained_path'] is not None and os.path.exists(model_cfg['pretrained_path']):
        model = create_model_with_pretrained(model_cfg, pretrained_path=model_cfg['pretrained_path'], device=config['device'], start_key=model_cfg.get('start_key', 'model.'))
    else:
        model = get_model_from_configs(**model_cfg)
    text_logger.info(f"Created model: {model_cfg.get('name', 'Unknown')}")
    
    # Watch gradients if using WandB
    if hasattr(tb_logger, 'experiment') and hasattr(tb_logger.experiment, 'watch'):
        tb_logger.experiment.watch(model, log='all', log_freq=100)
        text_logger.info("✅ Enabled WandB gradient and parameter tracking")

    # Create dataloaders
    text_logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader, inference_loader = create_dataloaders(dataloader_cfg)
    text_logger.info(f"Created dataloaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    # Handle multiple loss functions
    loss_functions = training_cfg.get('loss_fn', 'mse')
    if not isinstance(loss_functions, list):
        loss_functions = [loss_functions]
    
    results = {}
    
    for loss_fn in loss_functions:
        text_logger.info(f"Training with loss function: {loss_fn}")
        
        # Create training loop with flexible parameter passing
        lightning_model, trainer = get_training_loop_by_model_name(
            model_cfg.get('name', ''), 
            model=model, 
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            inference_loader=inference_loader,
            loss_fn_name=loss_fn,
            logger=tb_logger,
            input_dim=model.input_dim,
            **training_cfg  # Pass all training config parameters as kwargs
        )
        
        # Watch model with logger
        try:
            if tb_logger and hasattr(tb_logger, 'experiment') and hasattr(tb_logger.experiment, 'watch'):
                tb_logger.experiment.watch(lightning_model, log='all', log_freq=10, log_graph=True)
            elif tb_logger and isinstance(tb_logger, WandbLogger):
                tb_logger.watch(lightning_model, log='all', log_freq=10, log_graph=True)
        except Exception as e:
            text_logger.warning(f'Logger watch failed (non-fatal): {e}')
        
        # Start training
        text_logger.info("Starting training process...")
        trainer.fit(lightning_model)
        
        # Test the model
        test_results = trainer.test(lightning_model)
        results[loss_fn] = test_results
        
        text_logger.info(f"Training completed for loss function: {loss_fn}")
    
    text_logger.info("All training completed successfully")
    return {'status': 'success', 'results': results, 'save_dir': save_dir}
        
    # except Exception as e:
    #     text_logger.error(f"Training failed with error: {str(e)}")
    #     return {'status': 'failed', 'error': str(e), 'save_dir': save_dir}


def run_wandb_sweep_agent(sweep_id: str, project: str, entity: Optional[str] = None, base_config_path: Optional[str] = None):
    """Run WandB sweep agent for parameter sweeps."""
    def train_sweep():
        """Training function for WandB sweep."""
        # Initialize wandb run (this gets the swept parameters)
        wandb.init()
        
        # Load base configuration if provided
        if base_config_path and os.path.exists(base_config_path):
            with open(base_config_path, 'r') as f:
                base_config = yaml.safe_load(f)
        else:
            # Create minimal base config
            base_config = {
                'model': {},
                'training': {},
                'dataloader': {}
            }
        
        # Get swept parameters from wandb.config
        swept_params = dict(wandb.config)
        
        # Convert flat wandb config back to nested structure and merge with base config
        for key, value in swept_params.items():
            keys = key.split('.')
            current_dict = base_config
            for k in keys[:-1]:
                if k not in current_dict:
                    current_dict[k] = {}
                current_dict = current_dict[k]
            current_dict[keys[-1]] = value
        
        # Run training with this configuration
        run_id = wandb.run.id if wandb.run else "unknown"
        result = train_single_config(
            config=base_config,
            base_save_dir="./sweep_results",
            run_id=run_id,
            sweep_id=sweep_id,
            use_wandb=True
        )
        
        # Log results back to wandb
        if result['status'] == 'success':
            for loss_fn, test_results in result['results'].items():
                for metric_dict in test_results:
                    for metric_name, metric_value in metric_dict.items():
                        wandb.log({f"{loss_fn}_{metric_name}": metric_value})
        
        wandb.finish()
    
    # Run the sweep agent
    wandb.agent(sweep_id, train_sweep, project=project, entity=entity)


def run_parallel_training(
    configs: List[Dict[str, Any]], 
    base_save_dir: str, 
    max_workers: Optional[int] = None,
    use_wandb: bool = True
) -> List[Dict[str, Any]]:
    """
    Run training for multiple configurations in parallel.
    
    Args:
        configs: List of configuration dictionaries
        base_save_dir: Base directory for saving results
        max_workers: Maximum number of parallel workers
        use_wandb: Whether to use WandB logging
        
    Returns:
        List of training results
    """
    if max_workers is None:
        max_workers = min(len(configs), os.cpu_count() or 1)
    
    print(f"Running {len(configs)} training configurations with {max_workers} parallel workers")
    
    # Add warning about WandB parallel execution
    if use_wandb and max_workers > 1:
        print("⚠️  Warning: Running WandB with parallel workers. Each run will create separate WandB runs.")
        print("   Consider using WandB sweeps (--wandb_sweep) for better organization.")
    
    # Add warning about high parallelism with data downloads
    if max_workers > 4:
        print("⚠️  Warning: High parallelism (>4 workers) may cause:")
        print("   • File system race conditions during chunk downloads")
        print("   • Memory usage multiplication (8x cache size)")
        print("   • Division by zero errors from corrupted downloads")
        print("   Consider pre-downloading data or using --max_workers 2-4")
        print("   See parallel_training_analysis.py for detailed information")
    
    # For small numbers of configs or single worker, run sequentially
    if len(configs) <= 1 or max_workers == 1:
        results = []
        for i, config in enumerate(configs):
            print(f"Running configuration {i+1}/{len(configs)}")
            result = train_single_config(
                config=config,
                base_save_dir=base_save_dir,
                run_id=f"parallel_{i:04d}",
                sweep_id=None,
                use_wandb=use_wandb  # Enable WandB for sequential runs too
            )
            results.append(result)
        return results
    
    # Use ProcessPoolExecutor for better daemon process handling
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_config = {}
        for i, config in enumerate(configs):
            future = executor.submit(
                train_single_config,
                config,
                base_save_dir,
                f"parallel_{i:04d}",
                None,  # sweep_id
                use_wandb  # Enable WandB for parallel runs
            )
            future_to_config[future] = i
        
        # Collect results as they complete
        results: List[Dict[str, Any]] = []
        for _ in range(len(configs)):
            results.append({})  # Initialize with empty dicts
            
        for future in concurrent.futures.as_completed(future_to_config):
            config_idx = future_to_config[future]
            #try:
            result = future.result()
            results[config_idx] = result
            print(f"Completed configuration {config_idx + 1}/{len(configs)}")
            # except Exception as e:
            #     print(f"Configuration {config_idx + 1} failed with error: {e}")
            #     results[config_idx] = {
            #         'status': 'failed', 
            #         'error': str(e), 
            #         'save_dir': os.path.join(base_save_dir, f"run_parallel_{config_idx:04d}")
            #     }
    
    return results


def main():
    """Enhanced main function with sweep and parallel support."""
    parser = argparse.ArgumentParser(description='Enhanced training with sweep and parallel support')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--save_dir', type=str, default="./results", help='Save directory override')
    
    # Sweep options
    parser.add_argument('--sweep', action='store_true', help='Run parameter sweep')
    parser.add_argument('--sweep_method', type=str, default='grid', choices=['grid', 'random', 'bayes'], help='Sweep method')
    parser.add_argument('--wandb_sweep', action='store_true', help='Use WandB sweep instead of local sweep')
    parser.add_argument('--sweep_id', type=str, help='WandB sweep ID to join')
    parser.add_argument('--run_sweep_agent', action='store_true', help='Create WandB sweep AND run agent immediately')
    
    # Parallel options
    parser.add_argument('--parallel', action='store_true', help='Run configurations in parallel')
    parser.add_argument('--max_workers', type=int, help='Maximum parallel workers')
    
    # WandB options
    parser.add_argument('--wandb_project', type=str, default='ssm4sar', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, help='WandB entity name')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(Path(args.config), args)
    
    # Check if we're joining an existing WandB sweep
    if args.sweep_id:
        print(f"Joining WandB sweep: {args.sweep_id}")
        run_wandb_sweep_agent(
            sweep_id=args.sweep_id,
            project=args.wandb_project,
            entity=args.wandb_entity,
            base_config_path=args.config
        )
        return
    
    # Check if we need to run sweeps
    sweep_params = config.get('_sweep_params', {})
    
    if args.sweep and sweep_params:
        if args.wandb_sweep:
            # Create WandB sweep
            sweep_config = create_wandb_sweep_config(sweep_params, args.sweep_method)
            sweep_id = wandb.sweep(
                sweep_config,
                project=args.wandb_project,
                entity=args.wandb_entity
            )
            print(f"Created WandB sweep: {sweep_id}")
            print(f"Run agents with: python {__file__} --config {args.config} --sweep_id {sweep_id}")
            
            # If --run_sweep_agent is specified, immediately run an agent
            if args.run_sweep_agent:
                print(f"Starting sweep agent for sweep: {sweep_id}")
                run_wandb_sweep_agent(
                    sweep_id=sweep_id,
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    base_config_path=args.config
                )
            
        else:
            # Generate sweep configurations
            sweep_configs = generate_sweep_configs(config, sweep_params)
            print(f"Generated {len(sweep_configs)} sweep configurations")
            
            if args.parallel:
                # Run sweep configurations in parallel
                results = run_parallel_training(
                    configs=sweep_configs,
                    base_save_dir=args.save_dir,
                    max_workers=args.max_workers
                )
                
                # Save sweep summary
                sweep_summary = {
                    'total_runs': len(results),
                    'successful_runs': sum(1 for r in results if r['status'] == 'success'),
                    'failed_runs': sum(1 for r in results if r['status'] == 'failed'),
                    'results': results
                }
                
                summary_path = os.path.join(args.save_dir, 'sweep_summary.json')
                with open(summary_path, 'w') as f:
                    json.dump(sweep_summary, f, indent=2)
                
                print(f"Parallel sweep completed. Summary saved to: {summary_path}")
                
            else:
                # Run sweep configurations sequentially
                for i, sweep_config in enumerate(sweep_configs):
                    print(f"Running sweep configuration {i+1}/{len(sweep_configs)}")
                    result = train_single_config(
                        config=sweep_config,
                        base_save_dir=args.save_dir,
                        run_id=f"sweep_{i:04d}",
                        use_wandb=True
                    )
                    print(f"Sweep {i+1} result: {result['status']}")
    
    else:
        # Single configuration training
        result = train_single_config(
            config=config,
            base_save_dir=args.save_dir,
            use_wandb=True
        )
        print(f"Training result: {result['status']}")


if __name__ == '__main__':
    main()