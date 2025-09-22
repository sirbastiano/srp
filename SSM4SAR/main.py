'''
Main function for sarSSM training - Updated to    parser.add_argument('-gn',
                        '--gpu_no',
                        type=int,
                        default=0,
                        help='GPU number to run on'
    )ataModule
'''

from trainer import azimuthModelTrainer
from datamodule import SARDataModule
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from argparse import ArgumentParser
from sarSSM import sarSSM
from utils import load_model_safely

import lightning as pl
import os
import torch
import wandb
torch.set_float32_matmul_precision('medium')
import warnings 
import pprint
warnings.filterwarnings("ignore", category=UserWarning)
import secrets

def parse_arguments():
    """Parse command line arguments for sarSSM training.
    
    Returns:
        dict: Dictionary of parsed arguments.
    """
    BASE_DIR = '/Data_large/marine/PythonProjects/SAR/sarpyx/SSM4SAR'
    parser = ArgumentParser(description='sarSSM experiment script')
    
    # - Parameters
    
    # -- Experiment Directory Name
    parser.add_argument('-dn',
                        '--directory',
                        type=str,
                        default='experiment_1',
                        help='Experiment directory name, can be any valid string'
    )
    
    # -- Model Name
    parser.add_argument('-mn',
                        '--model_name',
                        type=str,
                        default='model_1',
                        help='Experiment model name, can be any valid? string'
    )
    
    # -- GPU Number
    parser.add_argument('-gpu',
                        '--gpu_no',
                        type=int,
                        default=1,
                        help='GPU number to run on'
    )
    
    
    # -- Model Number of Layers
    parser.add_argument('-nl',
                         '--num_layers',
                         type=int,
                         default=4,
                         help='Number of ssm layers after the embedding layer'
    )
    
    # -- Model hidden state size
    parser.add_argument('-hs',
                        '--hidden_state_size',
                        type=int,
                        default=8,
                        help='Size of the hidden state of the SSM layer'
    )
    
    # -- Number of training epochs
    parser.add_argument('-ep',
                        '--epochs',
                        type=int,
                        default=10,
                        help='Number of training epochs for the experiment (used if --max_steps not provided)'
    )
    
    # -- Maximum training steps (iteration-based training)
    parser.add_argument('--max_steps',
                        type=int,
                        default=None,
                        help='Maximum number of training steps (overrides epochs for iteration-based training)'
    )
    
    # -- Validation interval
    parser.add_argument('--val_check_interval',
                        type=int,
                        default=50,
                        help='Check validation every N training steps'
    )
    
    # -- Maximum validation steps
    parser.add_argument('--max_steps_validation',
                        type=int,
                        default=None,
                        help='Maximum number of validation steps to run (limits validation set size)'
    )
    
    # -- Training Batch Size
    parser.add_argument('-bs',
                        '--batch_size',
                        type=int,
                        default=10,
                        help='Batch size for the training loop'
    )
    
    # -- Validation Batch Size
    parser.add_argument('-vb',
                        '--valid_batch_size',
                        type=int,
                        default=32,
                        help='Batch size for the validation set'
    )
    # -- Learning Rate
    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        default=0.0001,
                        help='Learning rate for the non-SSM layers'
    )
    
    # -- Weight Decay
    parser.add_argument('-wd',
                        '--weight_decay',
                        type=float,
                        default=0.001,
                        help='Weight decay for the training optimizer'
    )
    
    # -- ssim_proportion
    parser.add_argument('-sp',
                        '--ssim',
                        type=float,
                        default=0.25,
                        help='ssim proportion'
    )

    # -- activation function
    parser.add_argument('-af',
                        '--act_fun',
                        type=str,
                        default='gelu',
                        choices=['relu', 'hardtanh', 'hardsigmoid', 'hardshrink', 'gelu', 'leakyrelu', 'hardswish', 'prelu'],
                        help='activation function for sarSSM'
    )
    
    # -- Training data directory
    parser.add_argument('-td',
                        '--train_dir',
                        type=str,
                        default=f'{BASE_DIR}/maya4_data/training',
                        help='Directory containing the training data'
    )
    
    # -- Validation data directory
    parser.add_argument('-vd',
                        '--val_dir',
                        type=str,
                        default=f'{BASE_DIR}/maya4_data/validation',
                        help='Directory containing the validation data'
    )

    # -- Weights & Biases configuration
    parser.add_argument('--use_wandb',
                        action='store_true',
                        help='Enable Weights & Biases logging'
    )
    
    parser.add_argument('--wandb_project',
                        type=str,
                        default='ssm4sar',
                        help='W&B project name'
    )
    
    parser.add_argument('--wandb_entity',
                        type=str,
                        default=None,
                        help='W&B entity (username or team)'
    )
    
    parser.add_argument('--wandb_tags',
                        type=str,
                        nargs='+',
                        default=['training'],
                        help='W&B tags for the run'
    )

    # -- Normalization arguments
    parser.add_argument(
        '--normalization_scheme',
        type=str,
        default='minmax',
        choices=['minmax', 'standard', 'robust', 'log', 'adaptive'],
        help='Default normalization scheme for both input and output'
    )
    
    parser.add_argument(
        '--input_norm_scheme',
        type=str,
        default=None,
        choices=['minmax', 'standard', 'robust', 'log', 'adaptive'],
        help='Normalization scheme for input levels (rc, rcmc, raw). Overrides default.'
    )
    
    parser.add_argument(
        '--output_norm_scheme',
        type=str,
        default=None,
        choices=['minmax', 'standard', 'robust', 'log', 'adaptive'],
        help='Normalization scheme for output level (az). Overrides default.'
    )
    
    # -- Normalization parameters for standard scheme
    parser.add_argument(
        '--input_mean',
        type=float,
        default=0.0,
        help='Mean for standard normalization of input data'
    )
    
    parser.add_argument(
        '--input_std',
        type=float,
        default=1000.0,
        help='Standard deviation for standard normalization of input data'
    )
    
    parser.add_argument(
        '--output_mean',
        type=float,
        default=0.0,
        help='Mean for standard normalization of output data'
    )
    
    parser.add_argument(
        '--output_std',
        type=float,
        default=5000.0,
        help='Standard deviation for standard normalization of output data'
    )
    
    # -- Normalization parameters for robust scheme
    parser.add_argument(
        '--input_median',
        type=float,
        default=0.0,
        help='Median for robust normalization of input data'
    )
    
    parser.add_argument(
        '--input_iqr',
        type=float,
        default=2000.0,
        help='IQR for robust normalization of input data'
    )
    
    parser.add_argument(
        '--output_median',
        type=float,
        default=0.0,
        help='Median for robust normalization of output data'
    )
    
    parser.add_argument(
        '--output_iqr',
        type=float,
        default=8000.0,
        help='IQR for robust normalization of output data'
    )
    
    # -- Normalization parameters for log scheme
    parser.add_argument(
        '--log_offset',
        type=float,
        default=1e-8,
        help='Offset for log normalization to avoid log(0)'
    )
    
    parser.add_argument(
        '--log_scale',
        type=float,
        default=1.0,
        help='Scale factor for log normalization'
    )
    
    # -- Normalization parameters for adaptive scheme
    parser.add_argument(
        '--adaptive_percentile_low',
        type=float,
        default=1.0,
        help='Lower percentile for adaptive normalization'
    )
    
    parser.add_argument(
        '--adaptive_percentile_high',
        type=float,
        default=99.0,
        help='Upper percentile for adaptive normalization'
    )
    
    # -- Complex data normalization options
    parser.add_argument(
        '--separate_real_imag',
        action='store_true',
        help='Normalize real and imaginary parts separately for complex data'
    )
    
    # -- Min-max normalization custom ranges
    parser.add_argument(
        '--custom_input_min',
        type=float,
        default=None,
        help='Custom minimum value for input min-max normalization'
    )
    
    parser.add_argument(
        '--custom_input_max',
        type=float,
        default=None,
        help='Custom maximum value for input min-max normalization'
    )
    
    parser.add_argument(
        '--custom_output_min',
        type=float,
        default=None,
        help='Custom minimum value for output min-max normalization'
    )
    
    parser.add_argument(
        '--custom_output_max',
        type=float,
        default=None,
        help='Custom maximum value for output min-max normalization'
    )

    arguments = vars(parser.parse_args())
    return arguments


def build_normalization_kwargs(arguments: dict) -> dict:
    """Build normalization kwargs from parsed arguments.
    
    Args:
        arguments: Parsed command line arguments.
        
    Returns:
        dict: Normalization kwargs for SARDataModule.
    """
    normalization_kwargs = {}
    
    # Standard normalization parameters
    if arguments.get('input_mean') is not None or arguments.get('input_std') is not None:
        normalization_kwargs['input_kwargs'] = {
            'mean': arguments.get('input_mean', 0.0),
            'std': arguments.get('input_std', 1000.0)
        }
    
    if arguments.get('output_mean') is not None or arguments.get('output_std') is not None:
        if 'output_kwargs' not in normalization_kwargs:
            normalization_kwargs['output_kwargs'] = {}
        normalization_kwargs['output_kwargs'].update({
            'mean': arguments.get('output_mean', 0.0),
            'std': arguments.get('output_std', 5000.0)
        })
    
    # Robust normalization parameters
    if arguments.get('input_median') is not None or arguments.get('input_iqr') is not None:
        if 'input_kwargs' not in normalization_kwargs:
            normalization_kwargs['input_kwargs'] = {}
        normalization_kwargs['input_kwargs'].update({
            'median': arguments.get('input_median', 0.0),
            'iqr': arguments.get('input_iqr', 2000.0)
        })
    
    if arguments.get('output_median') is not None or arguments.get('output_iqr') is not None:
        if 'output_kwargs' not in normalization_kwargs:
            normalization_kwargs['output_kwargs'] = {}
        normalization_kwargs['output_kwargs'].update({
            'median': arguments.get('output_median', 0.0),
            'iqr': arguments.get('output_iqr', 8000.0)
        })
    
    # Log normalization parameters
    if arguments.get('log_offset') is not None or arguments.get('log_scale') is not None:
        log_kwargs = {
            'offset': arguments.get('log_offset', 1e-8),
            'scale': arguments.get('log_scale', 1.0)
        }
        if 'input_kwargs' not in normalization_kwargs:
            normalization_kwargs['input_kwargs'] = {}
        if 'output_kwargs' not in normalization_kwargs:
            normalization_kwargs['output_kwargs'] = {}
        normalization_kwargs['input_kwargs'].update(log_kwargs)
        normalization_kwargs['output_kwargs'].update(log_kwargs)
    
    # Adaptive normalization parameters
    if arguments.get('adaptive_percentile_low') is not None or arguments.get('adaptive_percentile_high') is not None:
        adaptive_kwargs = {
            'percentile_low': arguments.get('adaptive_percentile_low', 1.0),
            'percentile_high': arguments.get('adaptive_percentile_high', 99.0)
        }
        if 'input_kwargs' not in normalization_kwargs:
            normalization_kwargs['input_kwargs'] = {}
        if 'output_kwargs' not in normalization_kwargs:
            normalization_kwargs['output_kwargs'] = {}
        normalization_kwargs['input_kwargs'].update(adaptive_kwargs)
        normalization_kwargs['output_kwargs'].update(adaptive_kwargs)
    
    # Custom min-max ranges
    if any(arguments.get(k) is not None for k in ['custom_input_min', 'custom_input_max']):
        if 'input_kwargs' not in normalization_kwargs:
            normalization_kwargs['input_kwargs'] = {}
        normalization_kwargs['input_kwargs'].update({
            'data_min': arguments.get('custom_input_min', -3000),
            'data_max': arguments.get('custom_input_max', 3000)
        })
    
    if any(arguments.get(k) is not None for k in ['custom_output_min', 'custom_output_max']):
        if 'output_kwargs' not in normalization_kwargs:
            normalization_kwargs['output_kwargs'] = {}
        normalization_kwargs['output_kwargs'].update({
            'data_min': arguments.get('custom_output_min', -12000),
            'data_max': arguments.get('custom_output_max', 12000)
        })
    
    # Complex data options
    normalization_kwargs['separate_real_imag'] = arguments.get('separate_real_imag', False)
    
    return normalization_kwargs


def save_script(exp_dir: str, model_name: str) -> None:
    # Create directories if they don't exist
    os.makedirs(os.path.join(exp_dir, model_name), exist_ok=True)
    
    # open the current script
    with open(__file__, 'r') as script_file:
        contents = script_file.read()
        
    # Write the contents to a new file
    with open(os.path.join(exp_dir, model_name, 'original_script.py'), 'w') as new_file:
        new_file.write(contents)     


if __name__ == '__main__': 
    arguments = parse_arguments()
    
    # Seed everything with a fresh random seed each run
    seed = secrets.randbits(32)
    pl.seed_everything(seed, workers=True)
    print(f'Using random seed: {seed}')
    
    # Extract parameters
    exp_dir = arguments['directory']
    model_name = arguments['model_name']
    train_dir = arguments['train_dir']
    val_dir = arguments['val_dir']
    
    # Training parameters
    num_epochs = arguments['epochs']
    max_steps = arguments.get('max_steps')
    val_check_interval = arguments['val_check_interval']
    max_steps_validation = arguments.get('max_steps_validation')
    train_batch = arguments['batch_size']
    valid_batch = arguments['valid_batch_size']
    lr = arguments['learning_rate']
    weight_decay = arguments['weight_decay']
    ssim_proportion = arguments['ssim']
    
    # Device parameters
    gpu_no = arguments['gpu_no']
    
    # Model parameters
    d_state = arguments['hidden_state_size']
    num_layers = arguments['num_layers']
    activation_function = arguments['act_fun']
    
    # Normalization parameters
    normalization_scheme = arguments['normalization_scheme']
    input_norm_scheme = arguments.get('input_norm_scheme')
    output_norm_scheme = arguments.get('output_norm_scheme')
    normalization_kwargs = build_normalization_kwargs(arguments)
    
    print(f'Normalization configuration:')
    print(f'  Default scheme: {normalization_scheme}')
    print(f'  Input scheme: {input_norm_scheme or normalization_scheme}')
    print(f'  Output scheme: {output_norm_scheme or normalization_scheme}')
    print(f'  Additional kwargs: {normalization_kwargs}')
    
    print(f'\nTraining configuration:')
    if max_steps is not None:
        print(f'  Mode: Iteration-based')
        print(f'  Max steps: {max_steps}')
        print(f'  Validation every: {val_check_interval} steps')
    else:
        print(f'  Mode: Epoch-based') 
        print(f'  Epochs: {num_epochs}')
    
    if max_steps_validation is not None:
        print(f'  Validation steps limit: {max_steps_validation}')
        
    print(f'  Batch size: {train_batch}')
    print(f'  SSIM proportion: {ssim_proportion}')
    print(f'  Model checkpointing: Best model based on validation SSIM')
    
    # Create the DataModule with normalization configuration
    data_module = SARDataModule(
        train_dir=train_dir,
        val_dir=val_dir,
        train_batch_size=train_batch,
        val_batch_size=valid_batch,
        level_from='rc',
        level_to='az',
        num_workers=4,  # Reduced from 12 to prevent memory issues
        patch_mode='rectangular',
        patch_size=(10_000, 1), # COLUMN
        buffer=(1, 1), # NO BUFFER
        stride=(10_000, 1), # SHIFT DOWN
        shuffle_files=False,
        patch_order='row', # GO RIGHT
        complex_valued=True,
        positional_encoding=True,
        save_samples=False,
        backend='zarr',
        verbose=False,
        samples_per_prod=20_000,
        cache_size=300, # preserve chunks in memory
        online=True,
        max_products=1,
        normalization_scheme=normalization_scheme,
        input_norm_scheme=input_norm_scheme,
        output_norm_scheme=output_norm_scheme,
        normalization_kwargs=normalization_kwargs,
    )
    
    # Create the model
    model = sarSSM(num_layers=num_layers, d_state=d_state, activation_function=activation_function)
    
    # Create the Lightning module (no longer needs batch sizes)
    lightning_model = azimuthModelTrainer(
        model=model,
        ssim_proportion=ssim_proportion,
        lr=lr,
        weight_decay=weight_decay
    )

    # Create logger
    if arguments.get('use_wandb', False):
        # Setup W&B logger with normalization info
        run_name = f'{model_name}_{exp_dir}'
        logger = WandbLogger(
            project=arguments.get('wandb_project', 'ssm4sar'),
            entity=arguments.get('wandb_entity', None),
            name=run_name,
            tags=arguments.get('wandb_tags', ['training']),
            config={
                'seed': seed,
                'model': {
                    'num_layers': num_layers,
                    'hidden_state_size': d_state,
                    'activation_function': activation_function
                },
                'training': {
                    'mode': 'iteration-based' if max_steps else 'epoch-based',
                    'epochs': num_epochs,
                    'max_steps': max_steps,
                    'val_check_interval': val_check_interval,
                    'max_steps_validation': max_steps_validation,
                    'batch_size': train_batch,
                    'valid_batch_size': valid_batch,
                    'learning_rate': lr,
                    'weight_decay': weight_decay,
                    'ssim_proportion': ssim_proportion
                },
                'normalization': {
                    'default_scheme': normalization_scheme,
                    'input_scheme': input_norm_scheme or normalization_scheme,
                    'output_scheme': output_norm_scheme or normalization_scheme,
                    'kwargs': normalization_kwargs
                }
            }
        )
        # Watch gradients/parameters and graph
        try:
            logger.watch(lightning_model, log='all', log_freq=200, log_graph=True)
        except Exception as e:
            print(f'W&B watch failed (non-fatal): {e}')
    else:
        # Use TensorBoard logger
        logger = TensorBoardLogger(exp_dir, name=model_name)

    # Create trainer
    # GPU validation and device configuration
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        print(f"CUDA available: {available_gpus} GPU(s) detected")
        
        # Validate requested GPU
        if gpu_no >= available_gpus:
            print(f"⚠ Requested GPU {gpu_no} not available. Machine has GPUs: {list(range(available_gpus))}")
            print(f"Falling back to GPU 0")
            gpu_no = 0
        
        accelerator = 'gpu'
        devices = [gpu_no]
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
    else:
        print("⚠ CUDA not available, using CPU training")
        accelerator = 'cpu'
        devices = 'auto'
    
    # Callbacks: LR monitor and model checkpointing based on validation SSIM
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # ModelCheckpoint callback to save best model based on validation SSIM
    from lightning.pytorch.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(exp_dir, model_name, 'checkpoints'),
        filename='best-model-{step}-{val_ssim_avg:.4f}',
        monitor='val_ssim_avg',
        mode='max',  # Higher SSIM is better
        save_top_k=1,
        save_last=True,
        verbose=True,
        auto_insert_metric_name=False
    )
    
    # Early stopping based on validation SSIM (optional, with patience for steps)
    early_stop = EarlyStopping(
        monitor='val_ssim_avg', 
        mode='max', 
        patience=10,  # Wait 10 validation checks before stopping
        verbose=True,
        check_finite=True
    )

    # Determine training configuration (iteration-based vs epoch-based)
    if max_steps is not None:
        print(f"Using iteration-based training with max_steps={max_steps}")
        trainer_config = {
            'max_steps': max_steps,
            'val_check_interval': val_check_interval,  # Validate every N steps
        }
        
        # Set validation batch limit
        if max_steps_validation is not None:
            trainer_config['limit_val_batches'] = max_steps_validation
            print(f"Validation will be limited to {max_steps_validation} steps per validation run")
        else:
            # Default behavior: limit based on training frequency  
            trainer_config['limit_val_batches'] = max_steps // val_check_interval + 1
            print(f"Validation will be limited to {trainer_config['limit_val_batches']} batches per validation run")
    else:
        print(f"Using epoch-based training with max_epochs={num_epochs}")
        trainer_config = {
            'max_epochs': num_epochs,
            'check_val_every_n_epoch': 1,  # Validate every epoch
        }
        
        # Set validation batch limit for epoch-based training too
        if max_steps_validation is not None:
            trainer_config['limit_val_batches'] = max_steps_validation
            print(f"Validation will be limited to {max_steps_validation} steps per validation run")

    trainer = pl.Trainer(
        logger=logger,
        devices=devices,
        accelerator=accelerator,
        fast_dev_run=False,
        log_every_n_steps=1,  # Log every step to see all metrics
        enable_progress_bar=True,
        enable_model_summary=True,
        callbacks=[lr_monitor, checkpoint_callback, early_stop],
        **trainer_config  # Apply either max_steps or max_epochs configuration
    )
    
    print(f"\n\nEstimated number of stepping batches : {trainer.estimated_stepping_batches} \n\n")
    
    # Fit the model using the DataModule
    trainer.fit(lightning_model, datamodule=data_module)
    
    # Print trainer attributes for debugging in a more readable format
    print("\n--- Trainer Attributes (Debug) ---")
    pprint.pprint(trainer.__dict__, indent=4)
    print("----------------------------------\n")
    
    # save the script to the same directory as the tensorboard logging
    save_script(exp_dir, model_name)

    # Save model state dict (always works and is more portable)
    torch.save(model.state_dict(), os.path.join(exp_dir, model_name, "model_weights"))
    
    # Save full model (handle W&B hooks issue)
    try:
        torch.save(model, os.path.join(exp_dir, model_name, "model"))
        print("✓ Full model saved successfully")
    except Exception as e:
        print(f"⚠ Could not save full model due to hooks (W&B): {e}")
        print("✓ Model weights saved successfully - use these for loading the model")
        
        # Save model architecture info for reconstruction
        model_info = {
            'model_class': type(model).__name__,
            'model_params': {
                'num_layers': num_layers,
                'd_state': d_state,
                'activation_function': activation_function
            },
            'state_dict_path': 'model_weights'
        }
        torch.save(model_info, os.path.join(exp_dir, model_name, "model_info"))
        print("✓ Model architecture info saved for reconstruction")
        print(f"  To load later: from utils import load_model_safely; model = load_model_safely('{os.path.join(exp_dir, model_name)}')")
