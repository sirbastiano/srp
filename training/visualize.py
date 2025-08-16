import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
import json
from torch.utils.data import DataLoader
from typing import Union, Optional, Tuple, Dict

def visualize_image_portion(dataloader: DataLoader, 
                            zfile: Union[str, os.PathLike],
                            start_y: int, 
                            start_x: int,
                            portion_height: int,
                            portion_width: int,
                            plot_type: str = 'magnitude',
                            show: bool = True,
                            vminmax: Optional[Union[Tuple[float, float], str]] = 'auto',
                            figsize: Tuple[int, int] = (15, 8),
                            save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Visualize a portion of an image by merging multiple patches, handling stride and overlap.
    Only works with square/rectangular patch modes.

    Args:
        dataloader (DataLoader): DataLoader instance for the dataset.
        zfile (Union[str, os.PathLike]): Path to the Zarr file.
        start_y (int): Starting y-coordinate.
        start_x (int): Starting x-coordinate.
        portion_height (int): Height of the portion.
        portion_width (int): Width of the portion.
        plot_type (str, optional): Visualization type ('magnitude', 'phase', etc.).
        show (bool, optional): Whether to display the plot.
        vminmax (Optional[Union[Tuple[float, float], str]], optional): Color scale limits or 'auto'.
        figsize (Tuple[int, int], optional): Figure size.
        save_path (Optional[str], optional): Path to save the visualization.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Merged arrays for input and target portions.
    """

    import matplotlib.pyplot as plt
    
    if patch_mode == "parabolic":
        raise ValueError("Image portion visualization is not supported for parabolic patch mode.")
        
    if zfile not in _samples_by_file:
        raise ValueError(f"File {zfile} not found in dataset.")
    
    # Get patch and stride information
    patch_h, patch_w = patch_size
    stride_h, stride_w = stride
    
    # Calculate how many patches we need in each dimension
    patches_needed_h = (portion_height + stride_h - 1) // stride_h
    patches_needed_w = (portion_width + stride_w - 1) // stride_w
    
    # Initialize merged arrays
    merged_input = np.zeros((portion_height, portion_width), dtype=np.complex64)
    merged_target = np.zeros((portion_height, portion_width), dtype=np.complex64)
    
    # Track coverage for proper merging
    coverage_mask = np.zeros((portion_height, portion_width), dtype=bool)
    
    patches_found = 0
    patches_used = 0
    
    # Iterate through the grid of patches needed
    for i in range(patches_needed_h):
        for j in range(patches_needed_w):
            # Calculate patch coordinates in the original image
            patch_y = start_y + i * stride_h
            patch_x = start_x + j * stride_w
            
            # Check if this patch exists in our dataset
            patch_coord = (patch_y, patch_x)
            if patch_coord in _samples_by_file[zfile]:
                patches_found += 1
                
                # Get the patch data using __getitem__
                try:
                    input_patch, target_patch = __getitem__((files.index(str(zfile)), patch_y, patch_x))
                    
                    # Convert back to numpy and handle complex conversion
                    if not complex_valued:
                        # Data is stored as [real, imag] channels, convert back to complex
                        if input_patch.dim() == 3:  # Check if it's a 3D tensor
                            input_patch_np = input_patch[0].numpy() + 1j * input_patch[1].numpy()
                            target_patch_np = target_patch[0].numpy() + 1j * target_patch[1].numpy()
                        else:
                            # Handle 2D case
                            input_patch_np = input_patch.numpy().astype(np.complex64)
                            target_patch_np = target_patch.numpy().astype(np.complex64)
                    else:
                        input_patch_np = input_patch.numpy()
                        target_patch_np = target_patch.numpy()
                    
                    # Calculate where this patch should go in the merged array
                    merge_start_y = i * stride_h
                    merge_start_x = j * stride_w
                    merge_end_y = min(merge_start_y + stride_h, portion_height)
                    merge_end_x = min(merge_start_x + stride_w, portion_width)
                    
                    # Calculate the actual patch region to copy (respecting stride, not full patch)
                    patch_copy_h = merge_end_y - merge_start_y
                    patch_copy_w = merge_end_x - merge_start_x
                    
                    # Copy the stride-sized portion from the patch to merged array
                    merged_input[merge_start_y:merge_end_y, merge_start_x:merge_end_x] = \
                        input_patch_np[:patch_copy_h, :patch_copy_w]
                    merged_target[merge_start_y:merge_end_y, merge_start_x:merge_end_x] = \
                        target_patch_np[:patch_copy_h, :patch_copy_w]
                    
                    coverage_mask[merge_start_y:merge_end_y, merge_start_x:merge_end_x] = True
                    patches_used += 1
                        
                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not load patch at ({patch_y}, {patch_x}): {e}")
                    continue
    
    if patches_used == 0:
        raise ValueError(f"No valid patches found for the requested portion starting at ({start_y}, {start_x})")
    
    if verbose:
        coverage_percent = np.sum(coverage_mask) / coverage_mask.size * 100
        print(f"Merged {patches_used}/{patches_found} available patches. Coverage: {coverage_percent:.1f}%")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Process data for visualization
    input_vis, input_vmin, input_vmax = get_sample_visualization(
        merged_input, plot_type=plot_type, vminmax=vminmax
    )
    target_vis, target_vmin, target_vmax = get_sample_visualization(
        merged_target, plot_type=plot_type, vminmax=vminmax
    )
    
    # Plot input (level_from)
    im1 = axes[0].imshow(input_vis, aspect='auto', cmap='viridis', 
                        vmin=input_vmin, vmax=input_vmax)
    axes[0].set_title(f'{level_from.upper()} - {plot_type.title()}')
    axes[0].set_xlabel('Range')
    axes[0].set_ylabel('Azimuth')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=8)
    
    # Plot target (level_to)
    im2 = axes[1].imshow(target_vis, aspect='auto', cmap='viridis',
                        vmin=target_vmin, vmax=target_vmax)
    axes[1].set_title(f'{level_to.upper()} - {plot_type.title()}')
    axes[1].set_xlabel('Range')
    axes[1].set_ylabel('Azimuth')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=8)
    
    # Add coverage information
    fig.suptitle(f'Image Portion: File {file_idx}, Start ({start_y}, {start_x}), '
                f'Size ({portion_height}, {portion_width})\n'
                f'Patches Used: {patches_used}, Coverage: {np.sum(coverage_mask) / coverage_mask.size * 100:.1f}%',
                fontsize=12)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Visualization saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return merged_input, merged_target

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
