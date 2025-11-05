#!/usr/bin/env python3
"""
Inference Visualization Script for SAR Focusing Models

This script loads a trained model and visualizes predictions on test data,
saving the results as PNG files.
"""

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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving to file
import matplotlib.pyplot as plt
import numpy as np

from model.model_utils import get_model_from_configs, create_model_with_pretrained
from training.training_loops import get_training_loop_by_model_name
from training.visualize import get_full_image_and_prediction, compute_metrics
from sarpyx.utils.losses import get_loss_function
from training_script import load_config
from inference_script import create_test_dataloader


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("visualization.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


def plot_intensity_histograms(orig_gt, orig_pred, gt, pred, save_path='intensity_histograms.png',
                              figsize=(20, 12), bins=100):
    """
    Plot histograms of pixel intensities for original and processed data.
    Separate plots for real and imaginary components.

    Args:
        orig_gt: Original ground truth data
        orig_pred: Original prediction data
        gt: Processed ground truth data
        pred: Processed prediction data
        save_path: Path to save the plot
        figsize: Figure size
        bins: Number of histogram bins
    """
    # Convert to numpy if needed
    if hasattr(orig_gt, 'numpy'):
        orig_gt = orig_gt.cpu().numpy()
    if hasattr(orig_pred, 'numpy'):
        orig_pred = orig_pred.cpu().numpy()
    if hasattr(gt, 'numpy'):
        gt = gt.cpu().numpy()
    if hasattr(pred, 'numpy'):
        pred = pred.cpu().numpy()

    # Create subplots: 2 rows (original vs processed) x 2 cols (real vs imaginary)
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Original data histograms - Real component
    axes[0, 0].hist(orig_gt.real.flatten(), bins=bins, alpha=0.7, label='Original GT (Real)',
                    color='blue', density=True)
    axes[0, 0].hist(orig_pred.real.flatten(), bins=bins, alpha=0.7, label='Original Pred (Real)',
                    color='red', density=True)
    axes[0, 0].set_title('Original Data - Real Component')
    axes[0, 0].set_xlabel('Pixel Intensity')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Original data histograms - Imaginary component
    axes[0, 1].hist(orig_gt.imag.flatten(), bins=bins, alpha=0.7, label='Original GT (Imag)',
                    color='blue', density=True)
    axes[0, 1].hist(orig_pred.imag.flatten(), bins=bins, alpha=0.7, label='Original Pred (Imag)',
                    color='red', density=True)
    axes[0, 1].set_title('Original Data - Imaginary Component')
    axes[0, 1].set_xlabel('Pixel Intensity')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Processed data histograms - Real component
    axes[1, 0].hist(gt.real.flatten(), bins=bins, alpha=0.7, label='Processed GT (Real)',
                    color='green', density=True)
    axes[1, 0].hist(pred.real.flatten(), bins=bins, alpha=0.7, label='Processed Pred (Real)',
                    color='orange', density=True)
    axes[1, 0].set_title('Processed Data - Real Component')
    axes[1, 0].set_xlabel('Pixel Intensity')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Processed data histograms - Imaginary component
    axes[1, 1].hist(gt.imag.flatten(), bins=bins, alpha=0.7, label='Processed GT (Imag)',
                    color='green', density=True)
    axes[1, 1].hist(pred.imag.flatten(), bins=bins, alpha=0.7, label='Processed Pred (Imag)',
                    color='orange', density=True)
    axes[1, 1].set_title('Processed Data - Imaginary Component')
    axes[1, 1].set_xlabel('Pixel Intensity')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Add statistics text
    for i, (data_gt, data_pred, title_suffix) in enumerate([
        (orig_gt, orig_pred, "Original"),
        (gt, pred, "Processed")
    ]):
        for j, component in enumerate(['real', 'imag']):
            if component == 'real':
                gt_comp, pred_comp = data_gt.real, data_pred.real
            else:
                gt_comp, pred_comp = data_gt.imag, data_pred.imag

            # Calculate statistics
            gt_mean, gt_std = np.mean(gt_comp), np.std(gt_comp)
            pred_mean, pred_std = np.mean(pred_comp), np.std(pred_comp)

            # Add stats text box
            stats_text = f'GT: μ={gt_mean:.2f}, σ={gt_std:.2f}\nPred: μ={pred_mean:.2f}, σ={pred_std:.2f}'
            axes[i, j].text(0.02, 0.98, stats_text, transform=axes[i, j].transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved intensity histograms to: {save_path}")

    # Print detailed statistics
    print("\n" + "="*60)
    print("DETAILED STATISTICS")
    print("="*60)

    for data_name, data_gt, data_pred in [("Original", orig_gt, orig_pred), ("Processed", gt, pred)]:
        print(f"\n{data_name} Data:")
        print("-" * 30)

        for comp_name, gt_comp, pred_comp in [("Real", data_gt.real, data_pred.real),
                                              ("Imaginary", data_gt.imag, data_pred.imag)]:
            print(f"\n{comp_name} Component:")
            print(f"  GT    - Mean: {np.mean(gt_comp):8.4f}, Std: {np.std(gt_comp):8.4f}, "
                  f"Min: {np.min(gt_comp):8.4f}, Max: {np.max(gt_comp):8.4f}")
            print(f"  Pred  - Mean: {np.mean(pred_comp):8.4f}, Std: {np.std(pred_comp):8.4f}, "
                  f"Min: {np.min(pred_comp):8.4f}, Max: {np.max(pred_comp):8.4f}")
            print(f"  Diff  - Mean: {np.mean(gt_comp - pred_comp):8.4f}, "
                  f"Std: {np.std(gt_comp - pred_comp):8.4f}")


def display_inference_results(input_data, gt_data, pred_data, save_path='inference_results.png',
                             figsize=(20, 6), vminmax=(0, 1000)):
    """
    Display input, ground truth, and prediction in a 3-column grid.

    Args:
        input_data: Input data from the dataset
        gt_data: Ground truth data
        pred_data: Model prediction
        save_path: Path to save the plot
        figsize: Figure size
        vminmax: Value range for visualization
    """
    # Convert tensors to numpy if needed
    if hasattr(input_data, 'numpy'):
        input_data = input_data.cpu().numpy()
    if hasattr(gt_data, 'numpy'):
        gt_data = gt_data.cpu().numpy()
    if hasattr(pred_data, 'numpy'):
        pred_data = pred_data.cpu().numpy()

    # Function to get magnitude visualization
    def get_magnitude_vis(data, vminmax):
        if np.iscomplexobj(data):
            magnitude = np.abs(data)
        else:
            magnitude = data

        if vminmax == 'auto':
            vmin, vmax = np.percentile(magnitude, [2, 98])
        elif isinstance(vminmax, tuple):
            vmin, vmax = vminmax
        else:
            vmin, vmax = np.min(magnitude), np.max(magnitude)

        return magnitude, vmin, vmax

    # Prepare visualizations
    imgs = []

    # Input data
    img, vmin, vmax = get_magnitude_vis(input_data, vminmax)
    imgs.append({'name': 'Input (RC)', 'img': img, 'vmin': vmin, 'vmax': vmax})

    # Ground truth
    img, vmin, vmax = get_magnitude_vis(gt_data, vminmax)
    imgs.append({'name': 'Ground Truth (AZ)', 'img': img, 'vmin': vmin, 'vmax': vmax})

    # Prediction
    img, vmin, vmax = get_magnitude_vis(pred_data, vminmax)
    imgs.append({'name': 'Prediction (AZ)', 'img': img, 'vmin': vmin, 'vmax': vmax})

    # Create the plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for i in range(3):
        im = axes[i].imshow(
            imgs[i]['img'],
            aspect='auto',
            cmap='viridis',
            vmin=imgs[i]['vmin'],
            vmax=imgs[i]['vmax']
        )

        axes[i].set_title(f"{imgs[i]['name']}")
        axes[i].set_xlabel('Range')
        axes[i].set_ylabel('Azimuth')

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        # Set equal aspect ratio
        axes[i].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved inference results to: {save_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize SAR model inference results')
    parser.add_argument('--config', type=str, default='s4_ssm_real.yaml',
                       help='Path to configuration YAML file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for inference')
    parser.add_argument('--save-dir', type=str, default='.',
                       help='Directory to save visualization outputs')
    parser.add_argument('--mode', type=str, default='parallel',
                       help='Processing mode (parallel or sequential)')
    parser.add_argument('--pretrained-path', type=str,
                       default='./results/best_model',
                       help='Path to pretrained model checkpoint')
    parser.add_argument('--vmin', type=float, default=2000,
                       help='Minimum value for visualization color scale')
    parser.add_argument('--vmax', type=float, default=6000,
                       help='Maximum value for visualization color scale')
    parser.add_argument('--show-window', type=str, default='1000,1000,10000,5000',
                       help='Visualization window as: y_start,y_size,x_start,x_size')
    parser.add_argument('--zfile', type=int, default=0,
                       help='Index of zarr file to visualize')

    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()

    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting visualization with config: {args.config}")

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Load configuration
    config = load_config(Path(args.config), args)

    # Extract configurations
    dataloader_cfg = config['dataloader']
    training_cfg = config.get('training', {})

    # Log configuration summary
    logger.info("Configuration Summary:")
    logger.info(f"  Data directory: {dataloader_cfg.get('data_dir', 'Not specified')}")
    logger.info(f"  Level from: {dataloader_cfg.get('level_from', 'rc')}")
    logger.info(f"  Level to: {dataloader_cfg.get('level_to', 'az')}")
    logger.info(f"  Patch size: {dataloader_cfg.get('patch_size', [1000, 1])}")
    logger.info(f"  Batch size: {dataloader_cfg.get('test', {}).get('batch_size', 'Not specified')}")
    logger.info(f"  Save directory: {args.save_dir}")

    # Create test dataloader
    logger.info("Creating test dataloader...")
    try:
        test_loader = create_test_dataloader(dataloader_cfg)
        logger.info(f"Created test dataloader with {len(test_loader)} batches")
        logger.info(f"Dataset contains {len(test_loader.dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to create test dataloader: {str(e)}")
        raise

    # Load model
    logger.info(f"Loading model from: {args.pretrained_path}")
    try:
        model = create_model_with_pretrained(
            config['model'],
            pretrained_path=args.pretrained_path,
            device=args.device
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

    # Parse show_window argument
    window_parts = [int(x) for x in args.show_window.split(',')]
    show_window = ((window_parts[0], window_parts[1]), (window_parts[2], window_parts[3]))

    # Visualize samples
    logger.info("Starting sample visualization...")
    try:
        inference_fn = get_training_loop_by_model_name(
            "ssm",
            train_loader=test_loader,
            val_loader=test_loader,
            test_loader=test_loader,
            model=model,
            save_dir=args.save_dir,
            mode=args.mode,
            loss_fn_name="mse"
        )[0].forward

        gt, pred, input_data, orig_gt, orig_pred = get_full_image_and_prediction(
            dataloader=test_loader,
            show_window=show_window,
            zfile=args.zfile,
            inference_fn=inference_fn,
            return_input=True,
            return_original=True,
            device=args.device,
            vminmax=(args.vmin, args.vmax)
        )

        # Compute and print metrics
        metrics = compute_metrics(gt, pred)
        logger.info("Computed metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

        # Save inference results
        inference_path = os.path.join(args.save_dir, 'inference_results.png')
        display_inference_results(
            input_data=input_data,
            gt_data=gt,
            pred_data=pred,
            save_path=inference_path,
            figsize=(20, 6),
            vminmax=(args.vmin, args.vmax)
        )

        # Save histograms
        histogram_path = os.path.join(args.save_dir, 'intensity_histograms.png')
        plot_intensity_histograms(
            orig_gt, orig_pred, gt, pred,
            save_path=histogram_path,
            figsize=(20, 12),
            bins=100
        )

        logger.info("Visualization completed successfully!")
        logger.info(f"Check the visualizations in: {args.save_dir}")

    except Exception as e:
        logger.error(f"Visualization failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
