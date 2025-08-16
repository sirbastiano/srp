import os
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
import zarr

RC_MIN = -3000
RC_MAX = 3000

GT_MIN = -12000
GT_MAX = 12000

def minmax_normalize(array, array_min, array_max):
    """
    Normalizes the input array to the range [0, 1].

    Args:
        array (np.ndarray): Input array.
        array_min (float): Minimum value for normalization.
        array_max (float): Maximum value for normalization.

    Returns:
        np.ndarray: Normalized array.
    """
    normalized_array = (array - array_min) / (array_max - array_min)
    normalized_array = np.clip(normalized_array, 0, 1)
    return normalized_array

def get_sample_visualization( 
                    data: np.ndarray,
                    plot_type: str = 'magnitude',
                    show: bool = True,
                    vminmax: Optional[Union[Tuple[float, float], str]] = (0, 1000),
                    figsize: Tuple[int, int] = (15, 15)) -> Tuple[np.ndarray, float, float]:
    """
    Visualize multiple arrays side by side.
    
    Args:
        idx (Tuple(file_idx, y, x)): Tuple containing file index, y, and x coordinates
        plot_type (str): Type of plot ('magnitude', 'phase', 'real', 'imag')
        show (bool): Whether to show the plot or not
        vminmax (Optional[Union[Tuple[float, float], str]]): Min/max values for colorbar or 'auto'
        figsize (Tuple[int, int]): Figure size for matplotlib
    """    

        
    if plot_type == 'magnitude' and np.iscomplexobj(data):
        plot_data = np.abs(data)
        title_suffix = 'Magnitude'
    elif plot_type == 'phase' and np.iscomplexobj(data):
        plot_data = np.angle(data)
        title_suffix = 'Phase'
    elif plot_type == 'real':
        plot_data = np.real(data)
        title_suffix = 'Real'
    elif plot_type == 'imag':
        plot_data = np.imag(data)
        title_suffix = 'Imaginary'
    else:
        plot_data = data
        title_suffix = 'Data'

    
    
    # Set vmin/vmax for 'raw' array, otherwise use phase or provided/default
    if vminmax == 'raw':
        vmin, vmax = 0, 10
    elif plot_type == 'phase':
        vmin, vmax = -np.pi, np.pi
    elif vminmax == 'auto':
        mean_val = np.mean(plot_data)
        std_val = np.std(plot_data)
        vmin, vmax = mean_val - std_val, mean_val + std_val
    elif vminmax is not None and isinstance(vminmax, tuple):
        vmin, vmax = vminmax
    else:
        vmin, vmax = 0, 1000
        
    return plot_data, vmin, vmax

            
def get_zarr_version(store_path: os.PathLike) -> int:
    import os
    import json
    if os.path.exists(store_path / 'zarr.json'):
        return 3
    elif os.path.exists(store_path / '.zgroup'):
            return 2
    else:
        raise ValueError("No .zgroup or zarr.json found")

def get_chunk_name_from_coords(
    y: int, x: int, zarr_file_name: str, level:str, chunks: Tuple[int, int] = (256, 256), version: int = 3
) -> str:
    """
    Generate a chunk name from zarr archive and coordinates.
    Args:
        y (int): Y coordinate.
        x (int): X coordinate.
        ch (int): Chunk height.
        cw (int): Chunk width.
        zarr_file_name (str): Name of the Zarr file.
        level (str): Level of the Zarr archive.
    Returns:
        str: Chunk name.
    """

    # Compute chunk indices for (y, x)
    cy, cx = y // chunks[0], x // chunks[1]
    if version == 3:
        chunk_fname = f"{zarr_file_name}/{level}/c/{cy}/{cx}"
    else:
        chunk_fname = f"{zarr_file_name}/{level}/{cy}.{cx}"
    return chunk_fname

def extract_stripmap_mode_from_filename(filename: Union[os.PathLike, str]) -> Optional[int]:
    """
    Extract the stripmap mode from a filename formatted as ...-s{number}.zarr.

    Args:
        filename (str): The filename to parse.

    Returns:
        Optional[int]: The stride number if found, else None.
    """
    if isinstance(filename, os.PathLike):
        filename = str(filename)
    import re
    match = re.search(r's(\d)a-s(\d)-', filename)
    if match:
        return int(match.group(2))
    return None