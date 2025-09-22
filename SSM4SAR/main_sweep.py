"""
Enhanced main script with Weights & Biases integration for parameter sweeps
"""

from trainer import azimuthModelTrainer
from datamodule import SARDataModule
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from argparse import ArgumentParser
from sarSSM import sarSSM

import lightning as pl
import os
import torch
import warnings
import pprint
import json
import wandb
from datetime import datetime

torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore", category=UserWarning)


def parse_arguments():
    """Parse command line arguments for training with sweep support."""
    BASE_DIR = "/Data_large/marine/PythonProjects/SAR/sarpyx/SSM4SAR"
    parser = ArgumentParser(description='sarSSM experiment script with sweep support')
    
    # Experiment identification
    parser.add_argument('--experiment_id', type=str, default=None,
                        help='Unique experiment identifier for sweeps')
    parser.add_argument('-dn', '--directory', type=str, default='experiment_1',
                        help='Experiment directory name')
    parser.add_argument('-mn', '--model_name', type=str, default='model_1',
                        help='Model name for logging')
    
    # Hardware configuration
    parser.add_argument('-gpu', '--gpu_no', type=int, default=0,
                        help='Number of GPUs to use')
    
    # Model architecture
    parser.add_argument('-nl', '--num_layers', type=int, default=4,
                        help='Number of SSM layers')
    parser.add_argument('-hs', '--hidden_state_size', type=int, default=8,
                        help='Hidden state size of SSM layers')
    parser.add_argument('-af', '--act_fun', type=str, default='leakyrelu',
                        choices=['relu', 'hardtanh', 'hardsigmoid', 'hardshrink', 
                                'gelu', 'leakyrelu', 'hardswish', 'prelu'],
                        help='Activation function')
    
    # Training parameters
    parser.add_argument('-ep', '--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('-bs', '--batch_size', type=int, default=10,
                        help='Training batch size')
    parser.add_argument('-vb', '--valid_batch_size', type=int, default=10,
                        help='Validation batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('-sp', '--ssim', type=float, default=0.5,
                        help='SSIM loss proportion')
    
    # Data paths
    parser.add_argument('-td', '--train_dir', type=str,
                        default=f'{BASE_DIR}/maya4_data/training',
                        help='Training data directory')
    parser.add_argument('-vd', '--val_dir', type=str,
                        default=f'{BASE_DIR}/maya4_data/validation',
                        help='Validation data directory')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output directory for sweep results')
    
    # Weights & Biases configuration
    parser.add_argument('--use_wandb', type=bool, default=False,
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='ssm4sar',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity (username or team)')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=[],
                        help='W&B tags for the run')
    
    # Early stopping and checkpointing
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--save_top_k', type=int, default=3,
                        help='Number of best models to save')
    
    return vars(parser.parse_args())


def setup_logging(args):
    """Set up logging with TensorBoard and optionally W&B."""
    loggers = []
    
    # TensorBoard logger
    exp_dir = args.get('outdir', args['directory'])
    model_name = args['model_name']
    if args.get('experiment_id'):
        model_name = f"{model_name}_{args['experiment_id']}"
    
    tb_logger = TensorBoardLogger(exp_dir, name=model_name)
    loggers.append(tb_logger)
    
    # Weights & Biases logger
    if args['use_wandb']:
        # Create run name
        run_name = args.get('experiment_id', f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Prepare config for W&B
        wandb_config = {
            'model': {
                'num_layers': args['num_layers'],
                'hidden_state_size': args['hidden_state_size'],
                'activation_function': args['act_fun']
            },
            'training': {
                'epochs': args['epochs'],
                'batch_size': args['batch_size'],
                'valid_batch_size': args['valid_batch_size'],
                'learning_rate': args['learning_rate'],
                'weight_decay': args['weight_decay'],
                'ssim_proportion': args['ssim']
            },
            'data': {
                'train_dir': args['train_dir'],
                'val_dir': args['val_dir']
            }
        }
        
        wandb_logger = WandbLogger(
            project=args['wandb_project'],
            entity=args['wandb_entity'],
            name=run_name,
            tags=args['wandb_tags'] + ['ssm4sar'],
            config=wandb_config
        )
        loggers.append(wandb_logger)
    
    return loggers


def setup_callbacks(args):
    """Set up training callbacks."""
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=None,  # Will use logger's log_dir
        filename='best-{epoch:02d}-{val_loss:.3f}',
        save_top_k=args['save_top_k'],
        mode='min',
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=args['early_stopping_patience'],
        verbose=True,
        mode='min'
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    return callbacks


def save_experiment_info(args, trainer, model, data_module):
    """Save experiment configuration and results."""
    exp_dir = args.get('outdir', args['directory'])
    model_name = args['model_name']
    if args.get('experiment_id'):
        model_name = f"{model_name}_{args['experiment_id']}"
    
    # Create output directory
    output_dir = os.path.join(exp_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(args, f, indent=2, default=str)
    
    # Save model and weights
    torch.save(model, os.path.join(output_dir, "model"))
    torch.save(model.state_dict(), os.path.join(output_dir, "model_weights"))
    
    # Save final metrics if available
    if hasattr(trainer, 'logged_metrics'):
        metrics_path = os.path.join(output_dir, 'final_metrics.json')
        metrics = {k: float(v) if torch.is_tensor(v) else v 
                  for k, v in trainer.logged_metrics.items()}
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Save script backup
    with open(__file__, 'r') as script_file:
        contents = script_file.read()
    with open(os.path.join(output_dir, 'training_script.py'), 'w') as new_file:
        new_file.write(contents)
    
    return output_dir


def main():
    """Main training function."""
    args = parse_arguments()
    
    print("="*80)
    print("SSM4SAR Training Configuration")
    print("="*80)
    pprint.pprint(args, width=80)
    print("="*80)
    
    # Create data module
    data_module = SARDataModule(
        train_dir=args['train_dir'],
        val_dir=args['val_dir'],
        train_batch_size=args['batch_size'],
        val_batch_size=args['valid_batch_size'],
        level_from="rc",
        level_to="az",
        num_workers=8,
        patch_mode="rectangular",
        patch_size=(10000, 1),
        buffer=(1000, 1000),
        stride=(1, 300),
        shuffle_files=False,
        patch_order="col",
        complex_valued=True,
        positional_encoding=True,
        save_samples=False,
        backend="zarr",
        verbose=False,
        samples_per_prod=20000,
        cache_size=100,
        online=True,
        max_products=1
    )
    
    # Create model
    model = sarSSM(
        num_layers=args['num_layers'],
        d_state=args['hidden_state_size'],
        activation_function=args['act_fun']
    )
    
    # Create Lightning module
    lightning_model = azimuthModelTrainer(
        model=model,
        ssim_proportion=args['ssim'],
        lr=args['learning_rate'],
        weight_decay=args['weight_decay']
    )
    
    # Setup logging
    loggers = setup_logging(args)
    
    # Setup callbacks
    callbacks = setup_callbacks(args)
    
    # Create trainer
    # GPU validation and device configuration
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        print(f"CUDA available: {available_gpus} GPU(s) detected")
        
        # Validate requested GPU
        gpu_no = args['gpu_no']
        if gpu_no >= available_gpus:
            print(f"⚠ Requested GPU {gpu_no} not available. Machine has GPUs: {list(range(available_gpus))}")
            print(f"Falling back to GPU 0")
            gpu_no = 0
        
        accelerator = 'gpu'
        devices = [gpu_no]
        precision = '32'  # Use 32-bit precision to avoid cuFFT issues with non-power-of-2 signal sizes
        print(f"✓ Using GPU {gpu_no} for training")
        
        # Test CUDA capability before using
        try:
            torch.cuda.get_device_capability(gpu_no)
            print(f"✓ GPU {gpu_no} capability check passed")
        except RuntimeError as e:
            print(f"⚠ CUDA capability check failed: {e}")
            print("Falling back to CPU training...")
            accelerator = 'cpu'
            devices = 'auto'
            precision = '32'
    else:
        print("⚠ CUDA not available, using CPU training")
        accelerator = 'cpu'
        devices = 'auto'
        precision = '32'
    
    trainer = pl.Trainer(
        max_epochs=args['epochs'],
        logger=loggers,
        callbacks=callbacks,
        devices=devices,
        accelerator=accelerator,
        precision=precision,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,  # For performance
        benchmark=True,  # For performance with fixed input sizes
    )
    
    print(f"\nEstimated stepping batches: {trainer.estimated_stepping_batches}")
    print(f"Training on: {trainer.device_ids if hasattr(trainer, 'device_ids') else 'CPU'}")
    
    # Train the model
    try:
        trainer.fit(lightning_model, datamodule=data_module)
        
        # Save experiment information
        output_dir = save_experiment_info(args, trainer, model, data_module)
        
        print(f"\nTraining completed successfully!")
        print(f"Results saved to: {output_dir}")
        
        # Log final metrics to W&B if enabled
        if args['use_wandb'] and trainer.logged_metrics:
            wandb.log({"final_metrics": trainer.logged_metrics})
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        if args['use_wandb']:
            wandb.log({"training_failed": True, "error": str(e)})
        raise
    
    finally:
        # Clean up W&B
        if args['use_wandb']:
            wandb.finish()


if __name__ == '__main__':
    main()
