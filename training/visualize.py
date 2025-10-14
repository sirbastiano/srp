import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
import json
from torch.utils.data import DataLoader
import dataloader
from dataloader.dataloader import SARDataloader
from tqdm import tqdm
from torch import nn
from typing import Union, Optional, Tuple, Dict, Callable, List
import torch
import logging
from pathlib import Path
import wandb
import io
from PIL import Image
import time
import psutil
import gc


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dict with memory usage in MB for RAM and GPU memory (if available)
    """
    # RAM memory usage
    process = psutil.Process()
    ram_usage_mb = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB
    
    memory_stats = {
        'ram_usage_mb': ram_usage_mb
    }
    
    # GPU memory usage (if CUDA is available)
    if torch.cuda.is_available():
        gpu_memory_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_memory_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
        memory_stats.update({
            'gpu_memory_allocated_mb': gpu_memory_allocated_mb,
            'gpu_memory_reserved_mb': gpu_memory_reserved_mb
        })
    
    return memory_stats


def format_memory_stats(stats: Dict[str, float]) -> str:
    """Format memory statistics for logging."""
    parts = [f"RAM: {stats['ram_usage_mb']:.1f}MB"]
    if 'gpu_memory_allocated_mb' in stats:
        parts.append(f"GPU Allocated: {stats['gpu_memory_allocated_mb']:.1f}MB")
        parts.append(f"GPU Reserved: {stats['gpu_memory_reserved_mb']:.1f}MB")
    return " | ".join(parts)


        
def log_inference_to_wandb(input_data, gt_data, pred_data, logger, step_or_epoch, figsize=(20, 6), vminmax=(0, 1000), save_path: str="./visualizations/inference.png"):
    """
    Create inference visualization and log it to W&B.
    
    Args:
        input_data: Input data from the dataset
        gt_data: Ground truth data  
        pred_data: Model prediction
        logger: W&B logger from PyTorch Lightning
        step_or_epoch: Current step or epoch for captioning
        figsize: Figure size
        vminmax: Value range for visualization
    """
    # Create the visualization figure
    fig = display_inference_results(
        input_data=input_data,
        gt_data=gt_data, 
        pred_data=pred_data,
        figsize=figsize,
        vminmax=vminmax,
        show=False,  # Don't show in notebook
        save=True,  # Don't save to file  
        return_figure=True,  # Return the figure for W&B
        save_path=save_path
    )
    
    # Convert matplotlib figure to W&B Image
    # Method 1: Save to buffer and convert
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    
    # Log to W&B
    if hasattr(logger, 'experiment'):  # WandbLogger
        logger.experiment.log({
            "inference_comparison": wandb.Image(img, caption=f"Inference at step {step_or_epoch}"),
            "global_step": step_or_epoch
        })
    
    # Clean up
    plt.close(fig)
    buf.close()

        
def display_inference_results(input_data, gt_data, pred_data, figsize=(20, 6), vminmax=(0, 1000), show: bool=True, save: bool=True, save_path: str="./visualizations/", return_figure: bool=False):
    """
    Display input, ground truth, and prediction in a 3-column grid.
    
    Args:
        input_data: Input data from the dataset
        gt_data: Ground truth data
        pred_data: Model prediction
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
    
    # Function to get magnitude visualization (similar to get_sample_visualization)
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
    imgs.append({'name': 'Input (RCMC)', 'img': img, 'vmin': vmin, 'vmax': vmax})
    
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
    if show:
        plt.show()
    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logging.info(f"Saved inference results to {save_path}")
    if return_figure:
        return fig
        
def calculate_reconstruction_dimensions(
    coordinates: List[Tuple[int, int]], 
    patch_height: int, 
    patch_width: int, 
    stride_height: int = None, 
    stride_width: int = None,
    concatenate_patches: bool = False,
    concat_axis: int = 0, 
    batch_size: int = 16
) -> Tuple[int, int, int, int, int, int]:
    """
    Calculate optimal reconstruction dimensions based on actual patch coordinates.
    
    Args:
        coordinates: List of (y, x) patch coordinates that will be used
        patch_height: Height of each patch
        patch_width: Width of each patch  
        stride_height: Vertical stride between patches (optional, for validation)
        stride_width: Horizontal stride between patches (optional, for validation)
        concatenate_patches: Whether patches are concatenated
        concat_axis: Concatenation axis (0=vertical, 1=horizontal)
        
    Returns:
        Tuple[int, int, int, int, int, int]: 
        (final_height, final_width, min_y, max_y, min_x, max_x)
    """
    if not coordinates:
        return patch_height, patch_width, 0, patch_height, 0, patch_width
    
    # Extract coordinate ranges
    y_coords = [y for y, x in coordinates]
    x_coords = [x for y, x in coordinates]
    
    min_y, max_y = min(y_coords), max(y_coords)
    min_x, max_x = min(x_coords), max(x_coords)
    print(f"Found patch coordinates: (min_x, min_y)=({min_x}, {min_y}), (max_x, max_y)=({max_x}, {max_y})")

    # Calculate final dimensions
    if concatenate_patches:
        if concat_axis == 0:
            # Vertical concatenation: patches are stacked as columns
            # Height determined by patch height, width by coordinate spread
            final_height = patch_height
            final_width = max_x - min_x + patch_width #max_x - min_x + patch_width
        elif concat_axis == 1:
            # Horizontal concatenation: patches are stacked as rows  
            # Width determined by patch width, height by coordinate spread
            final_height = max_y - min_y + patch_height
            final_width = patch_width
        else:
            raise ValueError(f"Invalid concat_axis: {concat_axis}")
    else:
        # Standard patches: simple bounding box calculation
        final_height = max_y - min_y + patch_height
        final_width = max_x - min_x + patch_width
    
    return final_height, final_width, min_y, max_y, min_x, max_x
def get_full_image_and_prediction(
    dataloader: SARDataloader,
    zfile: Union[str, int, os.PathLike],
    inference_fn: Callable[[np.ndarray], np.ndarray],
    show_window: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    return_input: bool = False,
    return_original: bool = False,
    vminmax: Union[Tuple[int, int], str] = 'auto', 
    device: Union[str, torch.device] = "cuda", 
    verbose: bool = True
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Given a file name or index, runs inference on the first max_samples_per_prod patches,
    and reconstructs the full image (at level_to) and the corresponding prediction.

    Args:
        zfile: File name, index, or Path to the Zarr file.
        inference_fn: Callable that takes a batch of input patches and returns predictions.
        max_samples_per_prod: Maximum number of patches to use (default: self._samples_per_prod).
        batch_size: Batch size for inference.
        return_input: If True, also returns the reconstructed input image.
        verbose: If True, prints progress.

    Returns:
        (gt_full, pred_full, [input_full]): Tuple of ground truth image, prediction image,
        and optionally the reconstructed input image (all as numpy arrays).
    """
    
    dataset = dataloader.dataset
    batch_size = dataloader.batch_size
    # Resolve zfile from index if needed
    if isinstance(zfile, int):
        zfile = dataset.get_files()[zfile]
    else:
        zfile = Path(zfile)

    # Ensure patches are calculated
    #dataset.buffer = show_window[0] if show_window is not None else dataset.buffer
    # dataset.calculate_patches_from_store(zfile, patch_order='chunk') #"row")
    # coords = dataset.get_samples_by_file(zfile) #_samples_by_file[zfile]
    # max_samples_per_prod = dataset._samples_per_prod
    # coords = coords[:max_samples_per_prod]
    coords = dataloader.get_coords_from_zfile(zfile, window=show_window)
    # print(f"Calculated coordinates for zfile {zfile}: {coords}")
    # Get patch and image shapes
    ph, pw = dataset.get_patch_size(zfile)
    sh, sw = dataset.get_whole_sample_shape(zfile)
    if dataset.concatenate_patches:
        if dataset.concat_axis == 0:
            ph = sh
        elif dataset.concat_axis == 1:
            pw = sw
        else:
            raise ValueError(f"Concatenation axis must be either 0 or 1, but is: {dataset.concat_axis}")
    stride_x, stride_y = dataset.stride
    if show_window is None:
        h, w, _, _, _, _= calculate_reconstruction_dimensions(
            coords,
            patch_height=ph,
            patch_width=pw,
            stride_height=stride_y,
            stride_width=stride_x,
            concatenate_patches=True,
            concat_axis=dataset.concat_axis,
            batch_size =batch_size
        )
    else:
        h, w = show_window[1][0] - show_window[0][0], show_window[1][1] - show_window[0][1]
    #print(f"Total patch reconstructed dimensions: ({h}, {w})")
    #dataset.get_whole_sample_shape(zfile)
    
    stride_y, stride_x = dataset.stride

    # Prepare empty arrays for reconstruction
    # gt_full = np.zeros((h, w), dtype=np.complex64)
    # pred_full = np.zeros((h, w), dtype=np.complex64)
    # input_full = np.zeros((h, w), dtype=np.complex64) if return_input else None
    # count_map = np.zeros((h, w), dtype=np.int32)

    # Collect all input patches for inference
    gt_full = np.zeros((h, w), dtype=np.complex64)
    pred_full = np.zeros((h, w), dtype=np.complex64)
    input_full = np.zeros((h, w), dtype=np.complex64) if return_input else None
    count_map = np.ones((h, w), dtype=np.int32)
    #positions = dataloader.get_coords_from_zfile(zfile, window=show_window)
    
    #print(f"Dataloader dimension: {len(dataloader)}")
    out_batches = 0
    tot_inference_time = 0.0
    
    # Memory tracking for inference only
    memory_stats = {
        'initial_memory': get_memory_usage(),
        'peak_memory': get_memory_usage(),
        'batch_memory_usage': [],
        'inference_memory_deltas': []
    }
    dataloader.filter_by_zfiles(zfile) #Make sure that only the selected file is plotted
    with torch.no_grad():
        removed_positions = 0
        for batch_idx, (input_batch, output_batch) in enumerate(dataloader):
            # Memory before inference
            pre_inference_memory = get_memory_usage()
            
            stop = True
            batch_size = input_batch.shape[0]
            max_patch_idx = 0
            for patch_idx in range(batch_size+1):
                if batch_idx * batch_size + patch_idx - removed_positions == len(coords):
                    stop = True
                    break
                # print(f"Batch index={batch_idx}, patch index={patch_idx}, removed positions={removed_positions}")
                x, y = coords[batch_idx * batch_size + patch_idx - removed_positions]
                x_to = x - show_window[0][0] #dataset.buffer[1]
                y_to = y - show_window[0][1] #dataset.buffer[0]
                if x_to > h or y_to > w:
                    # Remove the corresponding patch from the batch by masking out this index
                    input_batch = torch.cat([input_batch[:patch_idx], input_batch[patch_idx+1:]], dim=0)
                    output_batch = torch.cat([output_batch[:patch_idx], output_batch[patch_idx+1:]], dim=0)
                    (x_rem, y_rem) = coords.pop(batch_idx * batch_size + patch_idx - removed_positions)
                    removed_positions +=1
                    #print(f"Position ({x}, {y}): Removed position ({x_rem}, {y_rem}) mapped to ({x_to}, {y_to}) -- out of bounds for array of shape {gt_full.shape}")
                else:
                    #print(f"Position ({x}, {y}): Keeping position mapped to ({x_to}, {y_to})")
                    stop = False
                    max_patch_idx = patch_idx
                
            # print(f"Processing batch {batch_idx}")
            
            # Clear cache before inference to get accurate memory measurement
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # INFERENCE - this is what we're measuring
            t0 = time.time()
            pred_batch = inference_fn(x=input_batch, device=device)  # Should return (B, ph, pw) or (B, ph, pw, ...)
            dt = time.time() - t0
            tot_inference_time = tot_inference_time + dt
            
            # Memory after inference
            post_inference_memory = get_memory_usage()
            
            # Calculate memory delta for this inference
            inference_memory_delta = {
                'ram_delta_mb': post_inference_memory['ram_usage_mb'] - pre_inference_memory['ram_usage_mb']
            }
            if 'gpu_memory_allocated_mb' in post_inference_memory:
                inference_memory_delta['gpu_allocated_delta_mb'] = (
                    post_inference_memory['gpu_memory_allocated_mb'] - 
                    pre_inference_memory['gpu_memory_allocated_mb']
                )
                inference_memory_delta['gpu_reserved_delta_mb'] = (
                    post_inference_memory['gpu_memory_reserved_mb'] - 
                    pre_inference_memory['gpu_memory_reserved_mb']
                )
            
            memory_stats['inference_memory_deltas'].append(inference_memory_delta)
            memory_stats['batch_memory_usage'].append(post_inference_memory)
            
            # Update peak memory
            for key in post_inference_memory:
                if post_inference_memory[key] > memory_stats['peak_memory'].get(key, 0):
                    memory_stats['peak_memory'][key] = post_inference_memory[key]
            
            if isinstance(pred_batch, torch.Tensor):
                pred_batch = pred_batch.detach().cpu().numpy()
            
            batch_size = input_batch.shape[0]
            for patch_idx in range(max_patch_idx):
                x, y = coords[batch_idx * batch_size + patch_idx]
                x_to = x - dataset.buffer[1]
                y_to = y - dataset.buffer[0]
                gt_patch = dataset.get_patch_visualization(output_batch[patch_idx], dataset.level_to, vminmax=vminmax, restore_complex=True, remove_positional_encoding=False)
                if len(gt_patch.shape) > 2:
                    gt_patch = np.squeeze(gt_patch, -1)
                #print(f"Ground truth patch with index {idx} has shape: {gt_patch.shape}, while reconstructed ground truth patch has dimension {gt_patch.shape}")

                pred_patch = dataset.get_patch_visualization(pred_batch[patch_idx], dataset.level_to, vminmax=vminmax, restore_complex=True, remove_positional_encoding=False)
                if len(pred_patch.shape) > 2:
                    pred_patch = np.squeeze(pred_patch, -1)
                #print(f"Prediction with index {idx} has shape: {pred_patch.shape}, while reconstructed prediction patch has dimension {pred_patch.shape}")
                if return_input:
                    input_patch = dataset.get_patch_visualization(input_batch[patch_idx], dataset.level_from, vminmax=vminmax, restore_complex=True)
                    if input_patch.ndim > 2:
                        if input_patch.shape[-1] == 1:
                            input_patch = np.squeeze(input_patch, -1)
                        elif input_patch.shape[-1] == 0:
                            input_patch = input_patch.reshape(input_patch.shape[:-1])
                        else:
                            raise ValueError(f"Input patch has unexpected shape {input_patch.shape}, cannot squeeze last dimension")

                assert gt_patch.shape == pred_patch.shape, f"Prediction patch has a different size than original patch. Original patch shape: {gt_patch.shape}, prediction patch shape: {pred_patch.shape}"

                ph, pw = gt_patch.shape
                # print(f"Patch shape: (ph, pw)=({ph}, {pw})")
                if h - x_to < 0 and w - y_to < 0:
                    print(f"Stopping further processing -- patch at (x, y)=({x_to}, {y_to}) is out of bounds for array of shape {gt_full.shape}")
                    stop = True
                    break
                actual_ph = min(ph, h - x_to) 
                actual_pw = min(pw, w - y_to)
                # Place patch in the correct location
                # print(f"Trying to put sample from (x, y)=({x}, {y}) to (x, y)=({x_to}, {y_to}) with shapes (h, w)=({actual_ph}, {actual_pw}) to array with full shape={gt_full.shape}") 
                if actual_ph > 0 and actual_pw > 0 and x_to + actual_ph <= gt_full.shape[0] and y_to + actual_pw <= gt_full.shape[1]:
                    gt_full[x_to:x_to+actual_ph, y_to:y_to+actual_pw] += gt_patch[:actual_ph, :actual_pw] * count_map[x_to:x_to+actual_ph, y_to:y_to+actual_pw] 
                    pred_full[x_to:x_to+actual_ph, y_to:y_to+actual_pw] += pred_patch[:actual_ph, :actual_pw] * count_map[x_to:x_to+actual_ph, y_to:y_to+actual_pw] 
                    if return_input:
                        input_full[x_to:x_to+actual_ph, y_to:y_to+actual_pw] += input_patch[:actual_ph, :actual_pw] * count_map[x_to:x_to+actual_ph, y_to:y_to+actual_pw]
                # else:
                #     print(f"Skipping patch at (x, y)=({x_to}, {y_to}) with shape ({actual_ph}, {actual_pw}) -- out of bounds for array of shape {gt_full.shape}")
                # gt_full[x_to:x_to+actual_ph, y_to:y_to+actual_pw] = gt_patch[:actual_ph, :actual_pw]
                # pred_full[x_to:x_to+actual_ph, y_to:y_to+actual_pw] = pred_patch[:actual_ph, :actual_pw]
                count_map[x_to:x_to+actual_ph, y_to:y_to+actual_pw] = 0
                
            if stop:
                #print(f"Stopping further processing -- all remaining patches are out of bounds for array of shape {gt_full.shape}")
                break
    if verbose:
        print(f"Total inference time for {out_batches} batches: {tot_inference_time:.2f} seconds")
        if out_batches > 0:
            print(f"Average inference time per batch: {tot_inference_time / out_batches:.2f} seconds")
        else:
            print(f"Only one batch processed, total time: {tot_inference_time:.2f} seconds")
        
        # Memory usage statistics (inference only)
        print("\n=== INFERENCE MEMORY STATISTICS ===")
        print(f"Initial memory: {format_memory_stats(memory_stats['initial_memory'])}")
        print(f"Peak memory: {format_memory_stats(memory_stats['peak_memory'])}")
        
        # Calculate memory usage deltas
        if memory_stats['inference_memory_deltas']:
            ram_deltas = [delta['ram_delta_mb'] for delta in memory_stats['inference_memory_deltas']]
            avg_ram_delta = np.mean(ram_deltas)
            max_ram_delta = np.max(ram_deltas)
            print(f"Average RAM increase per inference: {avg_ram_delta:.1f}MB")
            print(f"Maximum RAM increase per inference: {max_ram_delta:.1f}MB")
            
            if 'gpu_allocated_delta_mb' in memory_stats['inference_memory_deltas'][0]:
                gpu_alloc_deltas = [delta['gpu_allocated_delta_mb'] for delta in memory_stats['inference_memory_deltas']]
                gpu_reserved_deltas = [delta['gpu_reserved_delta_mb'] for delta in memory_stats['inference_memory_deltas']]
                
                avg_gpu_alloc_delta = np.mean(gpu_alloc_deltas)
                max_gpu_alloc_delta = np.max(gpu_alloc_deltas)
                avg_gpu_reserved_delta = np.mean(gpu_reserved_deltas)
                max_gpu_reserved_delta = np.max(gpu_reserved_deltas)
                
                print(f"Average GPU allocated increase per inference: {avg_gpu_alloc_delta:.1f}MB")
                print(f"Maximum GPU allocated increase per inference: {max_gpu_alloc_delta:.1f}MB")
                print(f"Average GPU reserved increase per inference: {avg_gpu_reserved_delta:.1f}MB")
                print(f"Maximum GPU reserved increase per inference: {max_gpu_reserved_delta:.1f}MB")
        
        # Total memory increase from start to finish
        final_memory = memory_stats['batch_memory_usage'][-1] if memory_stats['batch_memory_usage'] else memory_stats['initial_memory']
        total_ram_increase = final_memory['ram_usage_mb'] - memory_stats['initial_memory']['ram_usage_mb']
        print(f"Total RAM increase during inference: {total_ram_increase:.1f}MB")
        
        if 'gpu_memory_allocated_mb' in final_memory:
            total_gpu_increase = final_memory['gpu_memory_allocated_mb'] - memory_stats['initial_memory']['gpu_memory_allocated_mb']
            print(f"Total GPU allocated increase during inference: {total_gpu_increase:.1f}MB")
        
        print("=====================================\n")

    # Average overlapping regions
    # mask = count_map > 0
    # gt_full[mask] /= count_map[mask]
    # pred_full[mask] /= count_map[mask]
    # if return_input:
    #     input_full[mask] /= count_map[mask]

    if dataset.verbose:
        print(f"Reconstructed image shape: {gt_full.shape}, prediction shape: {pred_full.shape}")

    if return_input:
        if return_original:
            return gt_full, pred_full, input_full, output_batch, pred_batch
        return gt_full, pred_full, input_full
    else:
        if return_original:
            return gt_full, pred_full, output_batch, pred_batch

        return gt_full, pred_full




from sarpyx.utils.metrics import (
    evaluate_sar_metrics,
    mse_complex,
    rmse_complex,
    psnr_amplitude,
    ssim_amplitude,
    amplitude_correlation,
    phase_error_stats,
    complex_coherence,
    phase_coherence,
    enl,
    resolution_gain,
)
def average_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Average a list of metric dictionaries.

    Args:
        metrics (List[Dict[str, float]]): List of metric dictionaries.

    Returns:
        Dict[str, float]: Dictionary with averaged metrics.
    """
    if not metrics:
        return {}
    
    avg_metrics = {}
    keys = metrics[0].keys()
    
    for key in keys:
        values = [m[key] for m in metrics if key in m]
        if values:
            avg_metrics[key] = float(np.mean(values))
    
    return avg_metrics
def compute_metrics(gt_image: np.ndarray, focused_image: np.ndarray) -> dict:
    """
    Compute comprehensive SAR image quality metrics between raw and focused SAR images.
    Uses the metrics from sarpyx.utils.metrics.
    """
    # Evaluate all SAR metrics (amplitude, phase, SAR-specific)
    try:
        sar_metrics = evaluate_sar_metrics(gt_image, focused_image)
    except Exception as e:
        print(f"Error computing SAR metrics: {e}")
        sar_metrics = {}

    # Combine all metrics
    metrics = dict(sar_metrics)
    return metrics

def save_metrics(metrics: dict, save_path: str) -> None:
    """
    Save computed metrics to a JSON file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)

