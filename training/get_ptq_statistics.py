import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import yaml
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import time
from typing import Dict, Any
import matplotlib.pyplot as plt

from model.model_utils import get_model_from_configs, create_model_with_pretrained
from training.training_loops import get_training_loop_by_model_name
from training.visualize import get_full_image_and_prediction, compute_metrics, plot_intensity_histograms, display_inference_results
from sarpyx.utils.losses import get_loss_function
from training_script import load_config
from training_script import create_dataloaders
import matplotlib.pyplot as plt
import numpy as np
import pickle

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("visualization.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


##############################
## Section 1
#####################################################

args = argparse.Namespace(
    config="training_configs/s4_ssm_stepssm.yaml", 
    pretrained_path=False,
    #pretrained_path=os.path.join(os.getcwd(), '..', 'results/progressive_layer_distillation/checkpoints/best.ckpt'), #'results', 'enhanced_distillation','checkpoints', 'last-v3.ckpt'),
    # config="training_configs/s4_ssm_complex_sweep.yaml", #"training_configs/s4_ssm_student.yaml",
    # pretrained_path=os.path.join(os.getcwd(), '..', 'results/s4_ssm_smaller_cols/last.ckpt'), #'results', 'enhanced_distillation','checkpoints', 'last-v3.ckpt'),
    # config="training_configs/s4_ssm_student.yaml", 
    # pretrained_path=os.path.join(os.getcwd(), '..', 'results', 'enhanced_distillation','checkpoints', 'last-v3.ckpt'), 
    device="cuda", 
    batch_size=16,
    save_dir="./visualizations",
    mode="parallel",
    learning_rate=1e-4, 
    num_epochs=50
)

# # Setup logging
logger = setup_logging()
#logger.info(f"Starting visualization with config: {args.config}")

# Load configuration
config = load_config(Path(args.config), args)

# Extract configurations
dataloader_cfg = config['dataloader']
training_cfg = config.get('training', {})

# Override save directory
save_dir = args.save_dir or training_cfg.get('save_dir', './visualizations')

# Create test dataloader
dataloader_cfg['patch_size'] = [5000, 1]
dataloader_cfg['stride'] = [4000, 1]
logger.info("Configuration Summary:")
logger.info(f"  Data directory: {dataloader_cfg.get('data_dir', 'Not specified')}")
logger.info(f"  Level from: {dataloader_cfg.get('level_from', 'rc')}")
logger.info(f"  Level to: {dataloader_cfg.get('level_to', 'az')}")
logger.info(f"  Patch size: {dataloader_cfg.get('patch_size', [1000, 1])}")
logger.info(f"  Batch size: {dataloader_cfg.get('test', {}).get('batch_size', 'Not specified')}")
logger.info(f"  Save directory: {save_dir}")
logger.info("Creating test dataloader...")
try:
    _, _, _, inference_loader = create_dataloaders(dataloader_cfg)
    logger.info(f"Created test dataloader with {len(inference_loader)} batches")
    logger.info(f"Dataset contains {len(inference_loader.dataset)} samples")
except Exception as e:
    logger.error(f"Failed to create test dataloader: {str(e)}")
    raise

try:
    model = create_model_with_pretrained(config['model'], pretrained_path=args.pretrained_path, device=args.device, start_key='model.')
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# load the model weights
with open('quant_weights/stepssm_weights.pkl', 'rb') as f:
    weights = pickle.load(f)

model.load_state_dict(weights)



# Visualize samples
logger.info("Starting sample visualization...")
model_name = config['model']['name']
print(f"Using model: {model}")
inference_fn = get_training_loop_by_model_name(model_name, train_loader=inference_loader, val_loader=inference_loader, test_loader=inference_loader, model=model, save_dir=save_dir, mode=args.mode, loss_fn_name="mse", input_dim=model.input_dim)[0].forward
gts = []
preds = []
inputs = []
orig_gts = []
orig_preds = []
for i in range(5):
    gt, pred, input, orig_gt, orig_pred = get_full_image_and_prediction(
        dataloader=inference_loader,
        show_window=((1000, 1000), (10000, 5000)),
        zfile=i,
        inference_fn=inference_fn,
        return_input=True, 
        return_original=True,
        device="cuda", 
        vminmax='auto' #(2000, 6000)
    )
    gts.append(gt)
    preds.append(pred)
    inputs.append(input)
    orig_gts.append(orig_gt)
    orig_preds.append(orig_pred)
    # print(compute_metrics(gt, pred))
    # display_inference_results(
    #     input_data=input,
    #     gt_data=gt,
    #     pred_data=pred,
    #     figsize=(20, 6),
    #     vminmax="auto",  # Adjust this range based on your data, 
    #     show=True, 
    #     save=False
    # )
    # plot_intensity_histograms(orig_gt[..., 0], orig_pred, gt, pred, figsize=(20, 12), bins=100)
    # logger.info("Visualization completed successfully!")
    # logger.info(f"Check the visualizations in: {save_dir}")




######################
## Section 2  - plot histograms
##########################################
for i in range(5):

    print(compute_metrics(gts[i], preds[i]))
    display_inference_results(
        input_data=inputs[i],
        gt_data=gts[i],
        pred_data=preds[i],
        figsize=(20, 6),
        vminmax="auto",  # Adjust this range based on your data, 
        show=True, 
        save=False
    )
    plot_intensity_histograms(orig_gts[i][..., 0], orig_preds[i], gts[i], preds[i], figsize=(20, 12), bins=100)
    logger.info("Visualization completed successfully!")
    logger.info(f"Check the visualizations in: {save_dir}")