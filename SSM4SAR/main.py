'''
Main function for sarSSM training - Updated to use DataModule
'''

from trainer import azimuthModelTrainer
from datamodule import SARDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from argparse import ArgumentParser
from sarSSM import sarSSM

import lightning as pl
import os
import torch
torch.set_float32_matmul_precision('medium')
import warnings 
import pprint
warnings.filterwarnings("ignore", category=UserWarning)

def parse_arguments():
    """
    Parses arguments
    
    Parameters
    ----------
    arguments List
        Arguments for parsing
        
    Returns
    -------
    arguments : dict
        Dictionary of parsed arguments
    
    """
    BASE_DIR = "/Data_large/marine/PythonProjects/SAR/sarpyx/SSM4SAR"
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
                        default=3,
                        help='Number of training epochs for the experiment'
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
                        default=10,
                        help='Batch size for the validation set'
    )
    # -- Learning Rate
    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        default=0.005,
                        help='Learning rate for the non-SSM layers'
    )
    
    # -- Weight Decay
    parser.add_argument('-wd',
                        '--weight_decay',
                        type=float,
                        default=0.01,
                        help='Weight decay for the training optimizer'
    )
    
    # -- ssim_proportion
    parser.add_argument('-sp',
                        '--ssim',
                        type=float,
                        default=0.5,
                        help='ssim proportion'
    )

    # -- activation function
    parser.add_argument('-af',
                        '--act_fun',
                        type=str,
                        default='leakyrelu',
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

    arguments = vars(parser.parse_args())
    
    return arguments


def save_script(exp_dir, model_name):
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
    
    # directory parameters
    exp_dir = arguments['directory']
    model_name = arguments['model_name']
    train_dir = arguments['train_dir']
    val_dir = arguments['val_dir']
    
    # training parameters
    num_epochs = arguments['epochs']
    train_batch = arguments['batch_size']
    valid_batch = arguments['valid_batch_size']
    lr = arguments['learning_rate']
    weight_decay = arguments['weight_decay']
    ssim_proportion = arguments['ssim']
    
    # device parameters
    gpu_no = arguments['gpu_no']
    
    # model parameters
    d_state = arguments['hidden_state_size']
    num_layers = arguments['num_layers']
    activation_function = arguments['act_fun']
    
    # Create the DataModule
    data_module = SARDataModule(
        train_dir=train_dir,
        val_dir=val_dir,
        train_batch_size=train_batch,
        val_batch_size=valid_batch,
        level_from="rc",
        level_to="az",
        num_workers=8,  # Reduced for stability
        patch_mode="rectangular",
        patch_size=(10000, 1),
        buffer=(1000, 1000),
        stride=(1, 300),  # Original stride that works
        shuffle_files=False,
        patch_order="col",
        complex_valued=True,  # Use complex values
        positional_encoding=True,  # With positional encoding for 3 channels
        save_samples=False,
        backend="zarr",
        verbose=False,  # Disable verbose output for faster loading
        samples_per_prod=20000,  
        cache_size=100,
        online=True,
        max_products=1
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
    logger = TensorBoardLogger(exp_dir, name=model_name)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=logger,
        devices=gpu_no,
        fast_dev_run=False,
        log_every_n_steps=1,  # Log every step to see all metrics
        check_val_every_n_epoch=1,  # Validate every epoch
        enable_progress_bar=True,
        enable_model_summary=True,
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

    torch.save(model, os.path.join(exp_dir, model_name, "model"))
    torch.save(model.state_dict(), os.path.join(exp_dir, model_name, "model_weights"))
