import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
import json
from torch.utils.data import DataLoader
import dataloader
from dataloader.dataloader import SARZarrDataset, SARDataloader
from tqdm import tqdm
from torch import nn
from typing import Union, Optional, Tuple, Dict
import torch
import logging
from pathlib import Path
from typing import Callable, List

logging.basicConfig(level=logging.INFO)
def display_inference_results(input_data, gt_data, pred_data, figsize=(20, 6), vminmax=(0, 1000), show: bool=True, save: bool=True, save_path: str="./visualizationtest.pngs/"):
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
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path)
        logging.info(f"Saved inference results to {save_path}")
        
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
    vminmax: Union[Tuple[int, int], str] = 'auto', 
    device: Union[str, torch.device] = "cuda"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    count_map = np.zeros((h, w), dtype=np.int32)
    #positions = dataloader.get_coords_from_zfile(zfile, window=show_window)
    
    print(f"Dataloader dimension: {len(dataloader)}")
    out_batches = 0
    with torch.no_grad():
        removed_positions = 0
        for batch_idx, (input_batch, output_batch) in enumerate(dataloader):
            stop = True
            batch_size = input_batch.shape[0]
            max_patch_idx = 0
            for patch_idx in range(batch_size):
                if batch_idx * batch_size + patch_idx - removed_positions == len(coords):
                    stop = True
                    break
                #print(f"Batch index={batch_idx}, patch index={patch_idx}, removed positions={removed_positions}")
                x, y = coords[batch_idx * batch_size + patch_idx - removed_positions]
                x_to = x - show_window[0][0] #dataset.buffer[1]
                y_to = y - show_window[0][1] #dataset.buffer[0]
                if x_to >= h or y_to >= w:
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
                
            print(f"Processing batch {batch_idx}")
            
            pred_batch = inference_fn(x=input_batch, device=device)  # Should return (B, ph, pw) or (B, ph, pw, ...)
            if isinstance(pred_batch, torch.Tensor):
                pred_batch = pred_batch.detach().cpu().numpy()
            
            batch_size = input_batch.shape[0]
            for patch_idx in range(max_patch_idx):
                x, y = coords[batch_idx * batch_size + patch_idx]
                x_to = x - dataset.buffer[1]
                y_to = y - dataset.buffer[0]
                gt_patch = dataset.get_patch_visualization(output_batch[patch_idx], dataset.level_to, vminmax=vminmax, restore_complex=True, remove_positional_encoding=True)
                #print(f"Ground truth patch with index {idx} has shape: {gt_patch.shape}, while reconstructed ground truth patch has dimension {gt_patch.shape}")
                pred_patch = dataset.get_patch_visualization(pred_batch[patch_idx], dataset.level_to, vminmax=vminmax, restore_complex=True, remove_positional_encoding=False)
                #print(f"Prediction with index {idx} has shape: {pred_patch.shape}, while reconstructed prediction patch has dimension {pred_patch.shape}")
                if return_input:
                    input_patch = dataset.get_patch_visualization(input_batch[patch_idx], dataset.level_from, vminmax=vminmax, restore_complex=True)

                assert gt_patch.shape == pred_patch.shape, f"Prediction patch has a different size than original patch. Original patch shape: {gt_patch.shape}, prediction patch shape: {pred_patch.shape}"
                ph, pw = gt_patch.shape
                # if h - x_to < 0 and w - y_to < 0:
                #     print(f"Stopping further processing -- patch at (x, y)=({x_to}, {y_to}) is out of bounds for array of shape {gt_full.shape}")
                #     stop = True
                #     break
                actual_ph = min(ph, h- x_to)
                actual_pw = min(pw, w - y_to)
                # Place patch in the correct location
                # print(f"Trying to put sample from (x, y)=({x_to}, {y_to}) with shapes (h, w)=({actual_ph}, {actual_pw}) to array with full shape={gt_full.shape}") 
                if actual_ph > 0 and actual_pw > 0 and x_to + actual_ph <= gt_full.shape[0] and y_to + actual_pw <= gt_full.shape[1]:
                    gt_full[x_to:x_to+actual_ph, y_to:y_to+actual_pw] += gt_patch[:actual_ph, :actual_pw]
                    pred_full[x_to:x_to+actual_ph, y_to:y_to+actual_pw] += pred_patch[:actual_ph, :actual_pw]
                    if return_input:
                        input_full[x_to:x_to+actual_ph, y_to:y_to+actual_pw] += input_patch[:actual_ph, :actual_pw]
                # else:
                #     print(f"Skipping patch at (x, y)=({x_to}, {y_to}) with shape ({actual_ph}, {actual_pw}) -- out of bounds for array of shape {gt_full.shape}")
                # gt_full[x_to:x_to+actual_ph, y_to:y_to+actual_pw] = gt_patch[:actual_ph, :actual_pw]
                # pred_full[x_to:x_to+actual_ph, y_to:y_to+actual_pw] = pred_patch[:actual_ph, :actual_pw]
                count_map[x_to:x_to+actual_ph, y_to:y_to+actual_pw] += 1
                
            if stop:
                print(f"Stopping further processing -- all remaining patches are out of bounds for array of shape {gt_full.shape}")
                break


    # Average overlapping regions
    mask = count_map > 0
    gt_full[mask] /= count_map[mask]
    pred_full[mask] /= count_map[mask]
    if return_input:
        input_full[mask] /= count_map[mask]

    if dataset.verbose:
        print(f"Reconstructed image shape: {gt_full.shape}, prediction shape: {pred_full.shape}")

    if return_input:
        return gt_full, pred_full, input_full
    else:
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

            # Compute comprehensive SAR metrics
            m = compute_metrics(raw_np, out_np, mainlobe_size=mainlobe_size)
            aggregated.append(m)

    # Aggregate metrics (example: average PSNR, PSLR, and others)
    summary = {}
    metric_keys = aggregated[0].keys() if aggregated else []
    for key in metric_keys:
        values = [m[key] for m in aggregated if m.get(key) is not None]
        if values:
            summary[f'avg_{key}'] = float(np.mean(values))
    summary['num_samples_evaluated'] = len(aggregated)

    metrics_path = os.path.join(save_dir, 'aggregated_metrics.json')
    save_metrics(summary, metrics_path)

    print(f"Saved {min(num_samples, len(aggregated))} visual samples to {save_dir}")
    print(f"Saved aggregated metrics to {metrics_path}")
