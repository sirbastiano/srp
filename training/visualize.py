import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
import json
from torch.utils.data import DataLoader
from dataloader.dataloader import SARZarrDataset
from tqdm import tqdm
from torch import nn
from typing import Union, Optional, Tuple, Dict
import torch
import logging

logging.basicConfig(level=logging.INFO)

def visualize_batch_samples(
    model, 
    test_loader, 
    device: str = 'cuda',
    save_dir: str = './visualizations', 
    max_batches: int = 5, 
    max_samples_per_batch: int = 4
):
    """
    Visualize samples from the test dataloader with model predictions.
    
    Args:
        model: Trained model for inference
        test_loader: Test dataloader
        device: Device to run model inference on
        save_dir: Directory to save visualizations
        max_batches: Maximum number of batches to visualize
        max_samples_per_batch: Maximum samples to show per batch
    """
    os.makedirs(save_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting visualization of test samples with model predictions")
    logger.info(f"Saving visualizations to: {save_dir}")
    
    # Setup model for inference
    model.to(device)
    model.eval()
    
    try:
        batch_count = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if batch_count >= max_batches:
                    break
                    
                logger.info(f"Processing batch {batch_idx + 1}/{max_batches}")
                logger.info(f"Batch input shape: {inputs.shape}, target shape: {targets.shape}")
                
                # Move data to device and run inference
                inputs_device = inputs.to(device)
                targets_device = targets.to(device)
                
                # Get model predictions
                try:
                    # # Try complex transformer interface first
                    # if hasattr(model, 'preprocess_input'):
                    #     predictions = model(src=inputs_device, tgt=targets_device)
                    # else:
                    #     # Fallback to standard forward pass
                    predictions = model(inputs_device)
                except Exception as e:
                    logger.warning(f"Model inference failed with {type(model).__name__}: {str(e)}")
                    # Try alternative interfaces
                    try:
                        predictions = model(inputs_device, targets_device)
                    except Exception as e2:
                        logger.error(f"All inference attempts failed: {str(e2)}")
                        continue
                
                # Move back to CPU for visualization
                inputs_cpu = inputs.detach().cpu().numpy()
                targets_cpu = targets.detach().cpu().numpy()
                predictions_cpu = predictions.detach().cpu().numpy()
                
                # Get dataset reference
                dataset = test_loader.dataset
                
                try:
                    # Determine number of samples to visualize
                    num_samples = min(max_samples_per_batch, inputs_cpu.shape[0])
                    
                    logger.info(f"Visualizing {num_samples} samples from batch {batch_idx + 1}")
                    logger.info(f"Input shape: {inputs_cpu.shape}, Prediction shape: {predictions_cpu.shape}, Target shape: {targets_cpu.shape}")
                    
                    # Use dataset's batch visualization method
                    if hasattr(test_loader, 'get_batch_visualization'):
                        # Create comprehensive batch visualization using dataset method
                        save_path = os.path.join(save_dir, f"batch_{batch_idx}_inputs_targets.png")
                        
                        # Visualize inputs and targets together
                        test_loader.get_batch_visualization(
                            inputs_batch=inputs_cpu,
                            targets_batch=targets_cpu,
                            batch_indices=list(range(num_samples)),
                            max_samples=num_samples,
                            titles=None,  # Use default titles
                            show=False,
                            save_path=save_path,
                            figsize=(20, 6 * num_samples),
                            vminmax=(0, 1000),
                            ncols=2
                        )
                        logger.info(f"Saved input/target visualization to: {save_path}")
                        
                        # Create model prediction comparison visualization
                        save_path_pred = os.path.join(save_dir, f"batch_{batch_idx}_predictions.png")   
                        logger.info(f"Saved manual prediction comparison to: {save_path_pred}")
                    
                    else:
                        # Fallback if dataset doesn't have batch visualization
                        logger.warning("Dataset doesn't have get_batch_visualization method, using manual approach")
                        
                        # Create manual visualization
                        fig, axes = plt.subplots(num_samples, 3, figsize=(18, 6 * num_samples))
                        if num_samples == 1:
                            axes = axes.reshape(1, -1)
                        
                        for sample_idx in range(num_samples):
                            # Get individual samples and process them
                            input_sample = inputs_cpu[sample_idx]
                            pred_sample = predictions_cpu[sample_idx]
                            target_sample = targets_cpu[sample_idx]
                            
                            # Simple processing - remove positional encoding and take magnitude
                            if input_sample.shape[-1] > 2:
                                input_sample = input_sample[..., :-2]
                            if pred_sample.shape[-1] > 2:
                                pred_sample = pred_sample[..., :-2]
                            if target_sample.shape[-1] > 2:
                                target_sample = target_sample[..., :-2]
                            
                            # Convert to magnitude if complex
                            input_data = np.abs(input_sample.squeeze()) if np.iscomplexobj(input_sample) else input_sample.squeeze()
                            pred_data = np.abs(pred_sample.squeeze()) if np.iscomplexobj(pred_sample) else pred_sample.squeeze()
                            target_data = np.abs(target_sample.squeeze()) if np.iscomplexobj(target_sample) else target_sample.squeeze()
                            
                            # Ensure 2D for imshow
                            if input_data.ndim == 1:
                                input_data = input_data.reshape(-1, 1)
                            if pred_data.ndim == 1:
                                pred_data = pred_data.reshape(-1, 1)
                            if target_data.ndim == 1:
                                target_data = target_data.reshape(-1, 1)
                            
                            # Plot
                            im1 = axes[sample_idx, 0].imshow(input_data, aspect='auto', cmap='viridis')
                            axes[sample_idx, 0].set_title(f'Input {sample_idx} ({dataset.level_from.upper()})')
                            plt.colorbar(im1, ax=axes[sample_idx, 0], fraction=0.046, pad=0.04)
                            
                            im2 = axes[sample_idx, 1].imshow(pred_data, aspect='auto', cmap='viridis')
                            axes[sample_idx, 1].set_title(f'Prediction {sample_idx} ({dataset.level_to.upper()})')
                            plt.colorbar(im2, ax=axes[sample_idx, 1], fraction=0.046, pad=0.04)
                            
                            im3 = axes[sample_idx, 2].imshow(target_data, aspect='auto', cmap='viridis')
                            axes[sample_idx, 2].set_title(f'Target {sample_idx} ({dataset.level_to.upper()})')
                            plt.colorbar(im3, ax=axes[sample_idx, 2], fraction=0.046, pad=0.04)
                        
                        plt.tight_layout()
                        save_path = os.path.join(save_dir, f"batch_{batch_idx}_manual_comparison.png")
                        plt.savefig(save_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        logger.info(f"Saved manual comparison to: {save_path}")
                    
                    # Create error analysis if desired
                    try:
                        error_save_path = os.path.join(save_dir, f"batch_{batch_idx}_prediction_error.png")
                        
                        fig_error, axes_error = plt.subplots(num_samples, 2, figsize=(12, 6 * num_samples))
                        if num_samples == 1:
                            axes_error = axes_error.reshape(1, -1)
                        
                        for sample_idx in range(num_samples):
                            pred_sample = predictions_cpu[sample_idx]
                            target_sample = targets_cpu[sample_idx]
                            
                            # Remove positional encoding if present
                            if pred_sample.shape[-1] > 2:
                                pred_sample = pred_sample[..., :-2]
                            if target_sample.shape[-1] > 2:
                                target_sample = target_sample[..., :-2]
                            
                            # Compute error (handle complex data)
                            if np.iscomplexobj(pred_sample) or np.iscomplexobj(target_sample):
                                pred_mag = np.abs(pred_sample.squeeze())
                                target_mag = np.abs(target_sample.squeeze())
                            else:
                                pred_mag = pred_sample.squeeze()
                                target_mag = target_sample.squeeze()
                            
                            # Ensure 2D for visualization
                            if pred_mag.ndim == 1:
                                pred_mag = pred_mag.reshape(-1, 1)
                                target_mag = target_mag.reshape(-1, 1)
                            
                            # Compute absolute and relative errors
                            abs_error = np.abs(pred_mag - target_mag)
                            rel_error = abs_error / (target_mag + 1e-8)  # Avoid division by zero
                            
                            # Plot absolute error
                            im1 = axes_error[sample_idx, 0].imshow(abs_error, aspect='auto', cmap='hot')
                            axes_error[sample_idx, 0].set_title(f'Absolute Error {sample_idx}')
                            axes_error[sample_idx, 0].set_xlabel('Range')
                            axes_error[sample_idx, 0].set_ylabel('Azimuth')
                            plt.colorbar(im1, ax=axes_error[sample_idx, 0], fraction=0.046, pad=0.04)
                            
                            # Plot relative error
                            im2 = axes_error[sample_idx, 1].imshow(rel_error, aspect='auto', cmap='hot')
                            axes_error[sample_idx, 1].set_title(f'Relative Error {sample_idx}')
                            axes_error[sample_idx, 1].set_xlabel('Range') 
                            axes_error[sample_idx, 1].set_ylabel('Azimuth')
                            plt.colorbar(im2, ax=axes_error[sample_idx, 1], fraction=0.046, pad=0.04)
                        
                        plt.tight_layout()
                        plt.savefig(error_save_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        logger.info(f"Saved error analysis to: {error_save_path}")
                        
                    except Exception as e:
                        logger.warning(f"Error analysis failed: {str(e)}")
                    
                except Exception as e:
                    logger.error(f"Error creating visualization for batch {batch_idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                batch_count += 1
                
    except Exception as e:
        logger.error(f"Error during model visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    logger.info(f"Model visualization complete! Saved {batch_count} batch comparisons to {save_dir}")

def visualize_pair(raw_image: np.ndarray, focused_image: np.ndarray, save_path: str, cmap: str = 'gray') -> None:
    """
    Plot raw and focused SAR images side by side and save the figure.

    Args:
        raw_image (np.ndarray): The input raw SAR image.
        focused_image (np.ndarray): The output focused SAR image.
        save_path (str): File path to save the plotted figure (including filename and extension, e.g., .png).
        cmap (str): Matplotlib colormap to use. Defaults to 'gray'.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(raw_image, cmap=cmap)
    axes[0].set_title('Raw SAR Image')
    axes[0].axis('off')

    axes[1].imshow(focused_image, cmap=cmap)
    axes[1].set_title('Focused SAR Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def compute_sidelobe_ratio(image: np.ndarray, mainlobe_size: int = 5) -> float:
    """
    Estimate the sidelobe ratio: ratio between mainlobe peak and max sidelobe.
    """
    peak_idx = np.unravel_index(np.argmax(image), image.shape)
    peak_value = image[peak_idx]
    mask = np.ones_like(image, dtype=bool)
    r, c = peak_idx
    r_min, r_max = max(0, r - mainlobe_size), min(image.shape[0], r + mainlobe_size + 1)
    c_min, c_max = max(0, c - mainlobe_size), min(image.shape[1], c + mainlobe_size + 1)
    mask[r_min:r_max, c_min:c_max] = False
    sidelobe_max = np.max(image[mask])
    pslr = 20 * np.log10(peak_value / (sidelobe_max + 1e-12))
    return pslr


def compute_metrics(raw_image: np.ndarray, focused_image: np.ndarray, mainlobe_size: int = 5) -> dict:
    """
    Compute image quality metrics between raw and focused SAR images.
    """
    try:
        psnr_value = peak_signal_noise_ratio(raw_image, focused_image,
                                            data_range=focused_image.max() - focused_image.min())
    except Exception:
        psnr_value = None

    try:
        pslr_value = compute_sidelobe_ratio(focused_image, mainlobe_size=mainlobe_size)
    except Exception:
        pslr_value = None

    return {
        'psnr_raw_vs_focused': psnr_value,
        'pslr_focused': pslr_value
    }


def save_metrics(metrics: dict, save_path: str) -> None:
    """
    Save computed metrics to a JSON file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def save_results_and_metrics(
    test_loader,
    model,
    device: str,
    save_dir: str = '.',
    num_samples: int = 10,
    mainlobe_size: int = 5
) -> None:
    """
    Run model on test_loader, save side-by-side visualizations and aggregate metrics.

    Args:
        test_loader: DataLoader yielding (raw, focused) pairs or (raw, target) inputs.
        model: Trained SAR focusing model with signature model(src, tgt).
        device: Compute device, e.g. 'cpu' or 'cuda'.
        save_dir: Directory to save outputs.
        num_samples: Number of samples to visualize.
        mainlobe_size: Half-size parameter for sidelobe ratio.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.eval()

    aggregated = []
    with torch.no_grad():
        for idx, (raw, target) in enumerate(test_loader):
            raw = raw.to(device).float()
            target = target.to(device).float()

            # Prediction: for parallel mode assume full-target pass
            output = model(src=raw, tgt=target)

            raw_np = raw.cpu().numpy().squeeze()
            out_np = output.cpu().numpy().squeeze()

            # Visualize first num_samples
            if idx < num_samples:
                fig_path = os.path.join(save_dir, f'sample_{idx:02d}.png')
                visualize_pair(raw_np, out_np, fig_path)

            # Compute metrics
            m = compute_metrics(raw_np, out_np, mainlobe_size=mainlobe_size)
            aggregated.append(m)

    # Aggregate metrics
    valid = [m for m in aggregated if m['psnr_raw_vs_focused'] is not None]
    avg_psnr = float(np.mean([m['psnr_raw_vs_focused'] for m in valid])) if valid else None
    avg_pslr = float(np.mean([m['pslr_focused'] for m in aggregated if m['pslr_focused'] is not None]))
    summary = {
        'avg_psnr_raw_vs_focused': avg_psnr,
        'avg_pslr_focused': avg_pslr,
        'num_samples_evaluated': len(aggregated)
    }
    metrics_path = os.path.join(save_dir, 'aggregated_metrics.json')
    save_metrics(summary, metrics_path)

    print(f"Saved {min(num_samples, len(aggregated))} visual samples to {save_dir}")
    print(f"Saved aggregated metrics to {metrics_path}")


if __name__ == '__main__':
    # Example usage
    import argparse
    from imageio import imread
    import torch

    parser = argparse.ArgumentParser(description="Visualize SAR focusing results and compute metrics.")
    parser.add_argument('--raw_path', type=str, required=True)
    parser.add_argument('--focused_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--mainlobe_size', type=int, default=5)
    args = parser.parse_args()

    raw = imread(args.raw_path)
    focused = imread(args.focused_path)

    fig_path = os.path.join(args.output_dir, 'comparison.png')
    visualize_pair(raw, focused, fig_path)

    metrics = compute_metrics(raw, focused, mainlobe_size=args.mainlobe_size)
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    save_metrics(metrics, metrics_path)

    print(f"Saved comparison figure to: {fig_path}")
    print(f"Saved metrics to: {metrics_path}")
