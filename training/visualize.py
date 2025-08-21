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
from pathlib import Path
from typing import Callable, List

logging.basicConfig(level=logging.INFO)

def calculate_reconstruction_dimensions(
    coordinates: List[Tuple[int, int]], 
    patch_height: int, 
    patch_width: int, 
    stride_height: int = None, 
    stride_width: int = None,
    concatenate_patches: bool = False,
    concat_axis: int = 0
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
    
    # Calculate final dimensions
    if concatenate_patches:
        if concat_axis == 0:
            # Vertical concatenation: patches are stacked as columns
            # Height determined by patch height, width by coordinate spread
            final_height = patch_height
            final_width = max_x - min_x + patch_width
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
    dataset: SARZarrDataset,
    zfile: Union[str, int, os.PathLike],
    inference_fn: Callable[[np.ndarray], np.ndarray],
    max_samples_per_prod: Optional[int] = None,
    batch_size: int = 16,
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
    # Resolve zfile from index if needed
    if isinstance(zfile, int):
        zfile = dataset.files[zfile]
    else:
        zfile = Path(zfile)

    # Ensure patches are calculated
    dataset.calculate_patches_from_store(zfile, patch_order="row")
    coords = dataset._samples_by_file[zfile]
    if max_samples_per_prod is None:
        max_samples_per_prod = dataset._samples_per_prod
    coords = coords[:max_samples_per_prod]

    # Get patch and image shapes
    ph, pw = dataset.get_patch_size(zfile)
    stride_x, stride_y = dataset.stride
    h, w, _, _, _, _= calculate_reconstruction_dimensions(
        coords,
        patch_height=ph,
        patch_width=pw,
        stride_height=stride_y,
        stride_width=stride_x,
        concatenate_patches=True,
        concat_axis=0
    )
    print(f"Total patch reconstructed dimensions: ({h}, {w})")
    #dataset.get_whole_sample_shape(zfile)
    
    stride_y, stride_x = dataset.stride

    # Prepare empty arrays for reconstruction
    # gt_full = np.zeros((h, w), dtype=np.complex64)
    # pred_full = np.zeros((h, w), dtype=np.complex64)
    # input_full = np.zeros((h, w), dtype=np.complex64) if return_input else None
    # count_map = np.zeros((h, w), dtype=np.int32)

    # Collect all input patches for inference
    input_patches = []
    gt_patches = []
    positions = []
    for (y, x) in coords:
        patch_from, patch_to = dataset[(str(zfile), y, x)]
        input_patches.append(patch_from)
        gt_patches.append(patch_to)
        positions.append((y, x))

    input_patches = np.stack(input_patches, axis=0)  # (N, ph, pw)
    gt_patches = np.stack(gt_patches, axis=0)        # (N, ph, pw)

    # Run inference in batches
    preds = []
    for i in range(0, len(input_patches), batch_size):
        batch = input_patches[i:i+batch_size]
        
        pred = inference_fn(x=batch, device=device)  # Should return (B, ph, pw) or (B, ph, pw, ...)
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        preds.append(pred)
    preds = np.concatenate(preds, axis=0)
    
    gt_full = np.zeros((h, w), dtype=np.complex64)
    pred_full = np.zeros((h, w), dtype=np.complex64)
    input_full = np.zeros((h, w), dtype=np.complex64) if return_input else None
    count_map = np.zeros((h, w), dtype=np.int32)


    # Place patches into the full image arrays
    for idx, (y, x) in enumerate(positions):
        gt_patch = dataset.get_patch_visualization(gt_patches[idx], dataset.level_to, vminmax=vminmax, restore_complex=True,)
        print(f"Ground truth patch with index {idx} has shape: {gt_patches[idx].shape}, while reconstructed ground truth patch has dimension {gt_patch.shape}")
        pred_patch = dataset.get_patch_visualization(preds[idx], dataset.level_to, vminmax=vminmax, restore_complex=True, remove_positional_encoding=False)
        print(f"Prediction with index {idx} has shape: {preds[idx].shape}, while reconstructed prediction patch has dimension {pred_patch.shape}")
        if return_input:
            input_patch = dataset.get_patch_visualization(input_patches[idx], dataset.level_from, vminmax=vminmax, restore_complex=True)

        assert gt_patch.shape == pred_patch.shape, f"Prediction patch has a different size than original patch. Original patch shape: {gt_patch.shape}, prediction patch shape: {pred_patch.shape}"
        ph, pw = gt_patch.shape
        # Place patch in the correct location
        gt_full[y:y+ph, x:x+pw] += gt_patch
        pred_full[y:y+ph, x:x+pw] += pred_patch
        if return_input:
            input_full[y:y+ph, x:x+pw] += input_patch
        count_map[y:y+ph, x:x+pw] += 1

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
