#!/usr/bin/env python3
"""
Simplified training script for sarSSMFinal model with knowledge distillation.
This script combines dataloader creation, model instantiation, and training execution in a single file.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import logging
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
import wandb
from typing import Dict, Any, Optional

# Import dataloader components
from dataloader.dataloader import SampleFilter, get_sar_dataloader, SARTransform

# Import model utilities
from model.model_utils import get_model_from_configs
from model.SSMs.SSM import sarSSMFinal
# Import training components
from training.training_loops import TrainSSM

# Import loss functions
from sarpyx.utils.losses import get_loss_function


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_sample_filters(dataloader_config: Dict[str, Any]) -> Dict[str, SampleFilter]:
    """Create SampleFilter objects for train/val/test splits."""
    filters = {}
    
    for split in ['train', 'validation', 'test']:
        if split in dataloader_config:
            split_config = dataloader_config[split]
            if 'filters' in split_config:
                filter_dict = split_config['filters']
                filters[split] = SampleFilter(**filter_dict)
            else:
                filters[split] = None
        else:
            filters[split] = None
    
    return filters


def create_dataloaders(config: Dict[str, Any]) -> tuple:
    """Create train, validation, and test dataloaders from configuration."""
    dataloader_config = config['dataloader']
    transform_config = config.get('transforms', {})
    data_dir = config.get('data_dir', '/tmp/sar_data')
    
    # Create sample filters for each split
    filters = create_sample_filters(dataloader_config)
    
    # Create transform
    transform = SARTransform.create_minmax_normalized_transform(
        normalize=transform_config.get('normalize', True),
        complex_valued=transform_config.get('complex_valued', True),
        rc_min=transform_config.get('rc_min', -2000),
        rc_max=transform_config.get('rc_max', 2000),
        gt_min=transform_config.get('gt_min', -2000),
        gt_max=transform_config.get('gt_max', 2000)
    )
    
    # Common dataloader parameters
    common_params = {
        'data_dir': data_dir,
        'transform': transform,
        'level_from': dataloader_config.get('level_from', 'rc'),
        'level_to': dataloader_config.get('level_to', 'az'),
        'num_workers': dataloader_config.get('num_workers', 0),
        'patch_mode': dataloader_config.get('patch_mode', 'rectangular'),
        'patch_size': dataloader_config.get('patch_size', [1000, 1]),
        'buffer': dataloader_config.get('buffer', [1000, 1000]),
        'stride': dataloader_config.get('stride', [1000, 1]),
        'max_base_sample_size': dataloader_config.get('max_base_sample_size', [50000, 50000]),
        'shuffle_files': dataloader_config.get('shuffle_files', False),
        'complex_valued': dataloader_config.get('complex_valued', True),
        'save_samples': dataloader_config.get('save_samples', False),
        'backend': dataloader_config.get('backend', 'zarr'),
        'verbose': dataloader_config.get('verbose', False),
        'cache_size': dataloader_config.get('cache_size', 1000),
        'online': dataloader_config.get('online', True),
        'concatenate_patches': dataloader_config.get('concatenate_patches', False),
        'concat_axis': dataloader_config.get('concat_axis', 0),
        'positional_encoding': dataloader_config.get('positional_encoding', True),
    }
    
    # Create train dataloader
    train_config = dataloader_config.get('train', {})
    train_loader = get_sar_dataloader(
        filters=filters['train'],
        batch_size=train_config.get('batch_size', 32),
        samples_per_prod=train_config.get('samples_per_prod', 10000),
        patch_order=train_config.get('patch_order', 'row'),
        max_products=train_config.get('max_products', 10),
        **common_params
    )
    
    # Create validation dataloader
    val_config = dataloader_config.get('validation', {})
    val_loader = get_sar_dataloader(
        filters=filters['validation'],
        batch_size=val_config.get('batch_size', 32),
        samples_per_prod=val_config.get('samples_per_prod', 1000),
        patch_order=val_config.get('patch_order', 'row'),
        max_products=val_config.get('max_products', 1),
        **common_params
    )
    
    # Create test dataloader
    test_config = dataloader_config.get('test', {})
    test_loader = get_sar_dataloader(
        filters=filters['test'],
        batch_size=test_config.get('batch_size', 32),
        samples_per_prod=test_config.get('samples_per_prod', 5000),
        patch_order=test_config.get('patch_order', 'row'),
        max_products=test_config.get('max_products', 10),
        **common_params
    )
    
    return train_loader, val_loader, test_loader




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


def main():
    """Main training function."""
    # Configuration
    config_path = '/home/gdaga/sarpyx_new/sarpyx/training/training_configs/s4_ssm_student.yaml'
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)
    
    # Setup logging
    save_dir = config.get('save_dir', './results/sarSSMFinal_training')
    tb_logger, logger = setup_logging(save_dir, 'sarSSMFinal')
    
    logger.info("Starting simplified training script")
    logger.info(f"Configuration loaded from: {config_path}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Validation loader: {len(val_loader)} batches") 
    logger.info(f"Test loader: {len(test_loader)} batches")
    
    # Create model
    logger.info("Creating sarSSMFinal model...")
    model_config = config['model']
    
    # Create model using the factory function
    model = sarSSMFinal(#get_model_from_configs(
        input_dim=model_config['input_dim'],
        model_dim=model_config['model_dim'],
        state_dim=model_config['state_dim'],
        output_dim=model_config['output_dim'],
        num_layers=model_config['num_layers'],
        activation_function=model_config.get('activation_function', 'relu'),
        transposed=False
    )
    
    print(f"Created model: {model.__class__.__name__}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training configuration
    training_config = config.get('training', {})
    num_epochs = training_config.get('num_epochs', 50)
    device_no = 0 if torch.cuda.is_available() else None
    
    # Create Lightning training module
    logger.info("Setting up training module...")
    lightning_model = TrainSSM(
        base_save_dir=save_dir,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=get_loss_function(training_config.get('loss_fn', 'mse')),
        mode=training_config.get('mode', 'parallel'),
        scheduler_type=training_config.get('scheduler_type', 'cosine'),
        lr=training_config.get('lr', 1e-4),
        real=True,
        step_mode=True, # Enable step mode for quantization-aware training,
        input_dim=model.input_dim
    )
    
    # Create PyTorch Lightning trainer
    logger.info("Creating PyTorch Lightning trainer...")
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=tb_logger,
        devices=[device_no] if device_no is not None else 'auto',
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        enable_progress_bar=True,
        log_every_n_steps=1,
        fast_dev_run=False,
        # Add checkpointing
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=save_dir,
                filename='sarSSMFinal_{epoch:02d}_{val_loss:.4f}',
                monitor='val_loss',
                save_top_k=3,
                mode='min'
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min'
            )
        ]
    )
    
    # Start training
    logger.info("Starting training...")
    logger.info(f"Device: {trainer.device_ids if hasattr(trainer, 'device_ids') else 'auto'}")
    logger.info(f"Max epochs: {num_epochs}")
    
    try:
        trainer.fit(lightning_model)
        logger.info("Training completed successfully!")
        
        # Test the model
        logger.info("Starting testing...")
        trainer.test(lightning_model)
        logger.info("Testing completed!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    logger.info(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    main()