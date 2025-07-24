import os
import random
import matplotlib.pyplot as plt
import numpy as np
from sarpyx.utils.zarr_utils import ProductHandler


def generate_random_window(size: int, max_value: int) -> tuple[int, int]:
    """
    Generate a random window (start, end) of given size within the range [0, max_value).

    Args:
        size (int): Size of the window.
        max_value (int): Maximum value (exclusive) for indices.

    Returns:
        tuple[int, int]: (start, end) indices of the window.
    """
    assert size > 0, f'size ({size}) must be positive'
    assert size <= max_value, f'size ({size}) must be <= max_value ({max_value})'
    start: int = random.randint(0, max_value - size)
    end: int = start + size
    return start, end


def main():
    # Create output directory
    output_dir = "./tmp"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find zarr files
    data_dir = "/Data_large/marine/PythonProjects/SAR/sarpyx/focused_data"
    zarr_files = [f for f in os.listdir(data_dir) if f.endswith('.zarr')]
    print(f"Found {len(zarr_files)} zarr files:")
    for zarr_file in zarr_files:
        print(f"- {zarr_file}")
    
    # Process each zarr file
    for zarr_file in zarr_files:
        filepath = os.path.join(data_dir, zarr_file)
        pHandler = ProductHandler(filepath)
        
        # Generate indices generically
        size_w = 2000
        size_h = 2000
        H, W = pHandler.array_shapes['raw']
        
        row_start, row_end = generate_random_window(size_h, H)
        col_start, col_end = generate_random_window(size_w, W)
        
        print(f"PROCESSED: {zarr_file} with rows: {row_start}-{row_end} and cols: {col_start}-{col_end}")
        
        # Save visualization instead of displaying
        output_filename = os.path.join(output_dir, f"{os.path.splitext(zarr_file)[0]}_visualization.png")
        
        # Assuming ProductHandler has a method to save visualizations
        # If not, you may need to modify this part based on the actual API
        fig = pHandler.visualize_arrays(
            array_names=['raw', 'rc', 'az'],
            rows=(row_start, row_end),
            cols=(col_start, col_end),
            plot_type='magnitude',
            show=False  # Don't display, just return figure
        )
        
        if fig:
            fig.savefig(output_filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {output_filename}")


if __name__ == "__main__":
    main()