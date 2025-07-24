import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sarpyx.utils.zarr_utils import ProductHandler
import os
import csv
import gc

data_dir = "/Data_large/marine/PythonProjects/SAR/sarpyx/focused_data"
save_dir = Path("/Data_large/marine/PythonProjects/SAR/sarpyx/data/focused_prev")

# Create save directory if it doesn't exist
save_dir.mkdir(parents=True, exist_ok=True)

# Initialize CSV file for array information
csv_path = save_dir / 'array_info.csv'
csv_data = []

zarr_files = [f for f in os.listdir(data_dir) if f.endswith('.zarr')]
print(f"Found {len(zarr_files)} zarr files:")


for idx, zarr_file in enumerate(zarr_files):
    if idx in [0,1]:
        continue
    try:
        print(f"Processing file {idx+1}/{len(zarr_files)}: {zarr_file}")
        filepath = os.path.join('/Data_large/marine/PythonProjects/SAR/sarpyx/focused_data', zarr_file)
        
        # Initialize ProductHandler with error checking
        pHandler = ProductHandler(filepath)
        
        # Check if arrays are available
        if not hasattr(pHandler, 'array_shapes') or not pHandler.array_shapes:
            print(f"Warning: No arrays found in {zarr_file}, skipping...")
            continue
            
        shapes = pHandler.array_shapes
        
        # Check if 'raw' array exists
        if 'raw' not in shapes:
            print(f"Warning: 'raw' array not found in {zarr_file}, skipping...")
            continue
            
        H, W = shapes['raw']
        print(f"Shapes of arrays in {zarr_file}: {H}x{W}")

        # Use complete image size
        row_start, row_end = 0, H
        col_start, col_end = 0, W

        print(f"DISPLAYED: {zarr_file} with full size: {H}x{W}")

        # Store array information for CSV
        for array_name, shape in shapes.items():
            csv_data.append({
                'file_name': zarr_file,
                'array_name': array_name,
                'height': shape[0],
                'width': shape[1],
                'total_size': shape[0] * shape[1]
            })

        # Create visualization
        pHandler.visualize_arrays(
            array_names=['raw','rc','az'], 
            rows=(row_start, row_end), 
            cols=(col_start, col_end),
            plot_type='magnitude',
            vminmax='auto',
        )
        
        # Save the current figure
        save_filename = f"{zarr_file.replace('.zarr', '')}_full_size.png"
        save_path = save_dir / save_filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
        
        # Close the plot to free memory
        plt.close()
        
    except Exception as e:
        print(f"Error processing {zarr_file}: {str(e)}")
        # Close any open plots in case of error
        plt.close('all')
        # Continue with next file instead of terminating
        continue
    
    finally:
        # Clean up variables and force garbage collection
        locals_to_delete = ['filepath', 'pHandler', 'shapes', 'H', 'W', 
                           'row_start', 'row_end', 'col_start', 'col_end',
                           'save_filename', 'save_path']
        
        for var_name in locals_to_delete:
            if var_name in locals():
                del locals()[var_name]
        
        # Force garbage collection
        gc.collect()

# Write CSV file with array information
with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['file_name', 'array_name', 'height', 'width', 'total_size']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_data)

print(f"Array information saved to: {csv_path}")
