import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
from location_utils import get_products_spatial_mapping
from create_balanced_dataset_splits import create_balanced_splits
import torch
import zarr

LOCATIONS_CSV_FILENAME = 'sar_products_locations.csv'

try:
    # Attempt to import when package initialization order allows it
    from dataloader import SARTransform, get_sar_dataloader  # type: ignore[attr-defined]
except ImportError:
    # During package initialization this may fail due to circular imports.
    # Defer the import until runtime when these helpers are first needed.
    SARTransform = None  # type: ignore[assignment]
    get_sar_dataloader = None  # type: ignore[assignment]


def _ensure_dataloader_dependencies() -> None:
    """Lazily import helpers that depend on the dataloader module.

    This avoids circular import issues when the package is initialized.
    """

    global SARTransform, get_sar_dataloader

    if SARTransform is not None and get_sar_dataloader is not None:
        return

    try:
        from dataloader.normalization import SARTransform as _SARTransform  # type: ignore
        from dataloader.dataloader import get_sar_dataloader as _get_sar_dataloader  # type: ignore
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImportError(
            "Unable to import dataloader helpers (SARTransform, get_sar_dataloader). "
            "Ensure the dataloader package is installed and accessible."
        ) from exc

    SARTransform = _SARTransform  # type: ignore[assignment]
    get_sar_dataloader = _get_sar_dataloader  # type: ignore[assignment]

class SampleFilter:
    def __init__(self, parts: Optional[List[str]]=None, years: Optional[List[int]] = None, months: Optional[List[int]] = None, polarizations: Optional[List[str]] = None, stripmap_modes: Optional[List[int]] = None):
        """
        Initialize a filter for SAR dataset samples.

        Args:
            parts (List[str], optional): List of part names to include.
            years (List[int], optional): List of years to include.
            months (List[int], optional): List of months to include.
            polarizations (List[str], optional): List of polarizations to include.
            stripmap_modes (List[int], optional): List of stripmap modes to include.
        """
        self.parts = parts if parts is not None else []
        self.years = years if years is not None else []
        self.months = months if months is not None else []
        self.polarizations = polarizations if polarizations is not None else []
        self.stripmap_modes = stripmap_modes if stripmap_modes is not None else []
    def get_filter_dict(self) -> Dict[str, List[Union[int, str]]]:
        filter_dict = {}
        if self.parts:
            filter_dict['part'] = self.parts
        if self.years:
            filter_dict['year'] = self.years
        if self.months:
            filter_dict['month'] = self.months
        if self.polarizations:
            filter_dict['polarization'] = self.polarizations
        if self.stripmap_modes:
            filter_dict['stripmap_mode'] = self.stripmap_modes
        return filter_dict
    def matches(self, record: dict) -> bool:
        if self.years and record.get('year') not in self.years:
            return False
        if self.months and record.get('month') not in self.months:
            return False
        if self.polarizations and record.get('polarization') not in self.polarizations:
            return False
        if self.stripmap_modes and str(record.get('stripmap_mode')) not in self.stripmap_modes:
            return False
        if self.parts and record.get('part') not in self.parts:
            return False
        return True
    def _filter_products(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = pd.Series([True] * len(df))
        if len(self.years) > 0:
            mask &= df["acquisition_date"].dt.year.isin(self.years)
        if len(self.months) > 0:
            mask &= df["acquisition_date"].dt.month.isin(self.months)
        if len(self.stripmap_modes) > 0:
            mask &= df["stripmap_mode"].isin(self.stripmap_modes)
        if len(self.polarizations) > 0:
            mask &= df["polarization"].isin(self.polarizations)
        if len(self.parts) > 0:
            mask &= df["part"].isin(self.parts)
        return df[mask]


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
    #normalized_array = np.clip(normalized_array, 0, 1)
    return normalized_array
def minmax_inverse(array, array_min, array_max):
    """
    Inverse normalization for the input array.

    Args:
        array (np.ndarray): Input array.
        array_min (float): Minimum value for normalization.
        array_max (float): Maximum value for normalization.

    Returns:
        np.ndarray: Inverse normalized array.
    """
    # Apply inverse normalization
    return array * (array_max - array_min) + array_min

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
        plot_type (str): Type of plot ('magnitude', 'phase', 'real', 'imag', 'complex')
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
    else: # 'complex' or other
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
def get_part_from_filename(filename: Union[os.PathLike, str]) -> Optional[str]:
    """
    Extract the part (e.g., PT1, PT2) from a filename formatted as .../{part}/s1a-s{number}-raw-s-{polarization}-...zarr.

    Args:
        filename (str): The filename to parse.

    Returns:
        Optional[str]: The part if found, else None.
    """
    try:
        part = str(filename).split(os.path.sep)[-2]
    except IndexError:
        print(f"Warning: could not extract part from filename '{filename}', using PT1 as part.")
        part = "PT1"
    return part

def parse_product_filename(filename: Union[str, os.PathLike]) -> dict:
    """
    Parse the product filename to extract metadata.
    
    Args:
        filename (str): The product filename.
    Returns:
        dict: A dictionary with extracted metadata.
    """
    # Example: s1a-s6-raw-s-vv-20210614t121147-20210614t121217-051942-0646ac.zarr
    sep = re.escape(os.sep)
    pattern = (
        rf"(?P<part>[a-zA-Z0-9]+){sep}s1a-s(?P<stripmap_mode>[a-zA-Z0-9]+)-raw-s-(?P<polarization>[a-zA-Z0-9]+)-"
        r"(?P<start_date>\d{8})t\d+-\d{8}t\d+-\d+-[a-zA-Z0-9]+\.zarr"
    )
    f_name = str(os.path.join(str(filename).split(os.path.sep)[-2], str(filename).split(os.path.sep)[-1]))
    match = re.match(pattern, f_name)
    if not match:
        return None
    stripmap_mode = match.group("stripmap_mode")
    polarization = match.group("polarization")
    start_date = match.group("start_date")
    part = match.group("part")
    # Remove -s{stripmap_mode} and -{polarization} from product name
    product_name = re.sub(r"-s\d+-raw-s-\w+-", "-", str(os.path.basename(filename)))
    product_name = product_name.split(".zarr")[0]
    acquisition_date = datetime.strptime(start_date, "%Y%m%d")
    return {
        "product_name": product_name,
        "stripmap_mode": int(stripmap_mode),
        "polarization": polarization,
        "acquisition_date": acquisition_date,
        "full_name": Path(filename), 
        "part": part,
        "store": None,
        "lat": None, 
        "lon": None, 
        "samples": [] 
    }

def display_inference_results(input_data, gt_data, pred_data = None, figsize=(20, 6), vminmax=(0, 1000), show: bool=True, save: bool=True, save_path: str="./visualizations/", return_figure: bool=False):
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
    if pred_data is not None and hasattr(pred_data, 'numpy'):
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
    if pred_data is not None:
        img, vmin, vmax = get_magnitude_vis(pred_data, vminmax)
        imgs.append({'name': 'Prediction (AZ)', 'img': img, 'vmin': vmin, 'vmax': vmax})
    
    # Create the plot
    n_axes = 3 if pred_data is not None else 2
    fig, axes = plt.subplots(1, n_axes, figsize=figsize)

    for i in range(n_axes):
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


def get_balanced_sample_files(
    max_samples: int,
    data_dir: Union[str, os.PathLike],
    sample_filter: Optional[SampleFilter] = None,
    config_path: Optional[str] = None,
    split_type: str = "train",
    ensure_representation: bool = True,
    min_samples_per_cluster: int = 1,
    verbose: bool = False, 
    n_clusters: int = 20,
    repo_author: str = 'Maya4', 
    repos: List[str] = ['PT1', 'PT2', 'PT3', 'PT4']
) -> List[str]:
    """
    Get a balanced sample of SAR product files ensuring equal representation across
    geographic clusters and scene types.
    
    Args:
        max_samples (int): Maximum number of samples to return
        sample_filter (Optional[SampleFilter]): SampleFilter object for additional filtering
        config_path (str): Path to dataset splits configuration directory
        split_type (str): Type of split to use ('train', 'validation', 'test')
        ensure_representation (bool): Whether to ensure geographic/scene representation
        min_samples_per_cluster (int): Minimum samples per geographic cluster
        verbose (bool): Whether to print detailed information
        
    Returns:
        List[str]: List of balanced SAR product filenames
        
    Raises:
        FileNotFoundError: If configuration files are not found
        ValueError: If max_samples is too small for balanced representation
    """
    
    if verbose:
        print(f"Getting balanced sample of {max_samples} files from {split_type} set...")
    
    # Determine the CSV file path
    if config_path is None: 
        config_path = os.path.join(os.path.dirname(__file__), '../dataset_splits')
    csv_path = os.path.join(config_path, f"{split_type}_products.csv")
    
    if not os.path.exists(csv_path):
        if verbose:
            print(f"  Split file not found: {csv_path}")
            print(f"  Creating balanced splits...")
        
        # Find the locations CSV file
        locations_csv = os.path.join(os.path.dirname(csv_path), LOCATIONS_CSV_FILENAME)
        if not os.path.exists(locations_csv):
            df = get_products_spatial_mapping(
                author=repo_author,
                repos=repos,
                data_dir=data_dir,
                output_csv_file_path=locations_csv,
                overwrite_csv=False, 
                verbose=False
            )
            locations_csv = os.path.join(data_dir, LOCATIONS_CSV_FILENAME)
            
        
        # Create the splits
        splits = create_balanced_splits(
            csv_file=locations_csv,
            output_dir=config_path,
            n_clusters=n_clusters,
            use_geopy=False,
            force_recreate=False
        )
        df = splits[split_type]
    else: 
        if verbose:
            print(f"  Loading existing split file: {csv_path}")
        # Try loading with semicolon delimiter first (new format)
        try:
            df = pd.read_csv(csv_path, sep=';')
        except:
            # Fall back to comma delimiter (old format)
            df = pd.read_csv(csv_path)
    # Load the split data
    
    files = df['filename'].tolist()
    parts = df['part'].tolist()
    
    # Create full paths for filtering
    full_paths = [os.path.join(data_dir, part, f) for part, f in zip(parts, files)]
    
    # Parse product filenames for filtering
    records = [r for r in (parse_product_filename(fp) for fp in full_paths) if r is not None]
    parsed_df = pd.DataFrame(records)

    if verbose:
        print(f"Loaded {len(parsed_df)} products from {split_type} split")
    
    # Apply SampleFilter if provided
    if sample_filter is not None:
        filtered_files = sample_filter._filter_products(parsed_df)
        filtered_files = pd.DataFrame(filtered_files)
        filtered_files.sort_values(by=['full_name'], inplace=True)
        
        if verbose:
            print(f"After SampleFilter: {len(filtered_files)} products remain")
            if len(filtered_files) > 0:
                print(f"  Sample filenames: {filtered_files['full_name'].head(3).tolist()}")
        
        files_to_use = filtered_files
    else:
        files_to_use = parsed_df
    
    # Check if we have enough samples
    if len(files_to_use) == 0:
        if verbose:
            print("No products match the filter criteria")
        return []

    if len(files_to_use) <= max_samples:
        if verbose:
            print(f"Found {len(files_to_use)} products (requested {max_samples})")
            print(f"Returning all {len(files_to_use)} available products (less than max_samples)")
        return [str(f) for f in files_to_use['full_name'].tolist()]

    # Balanced sampling with scene type representation and overlap avoidance
    if ensure_representation and 'scene_type' in df.columns and 'coordinates' in df.columns:
        # Merge scene type and coordinate info from original df with the filtered results
        # Match by filename to preserve this information
        merged_df = files_to_use.copy()
        
        # Create a mapping from filename to scene type and coordinates
        filename_to_scene = dict(zip(df['filename'], df.get('scene_type', [None]*len(df))))
        filename_to_coords = dict(zip(df['filename'], df.get('coordinates', [None]*len(df))))
        
        # Extract just the filename from full_name path for matching
        merged_df['filename_only'] = merged_df['full_name'].apply(lambda x: Path(x).name)
        merged_df['scene_type'] = merged_df['filename_only'].map(filename_to_scene)
        merged_df['coordinates'] = merged_df['filename_only'].map(filename_to_coords)
        
        # Remove rows where scene type or coordinates are missing
        merged_df = merged_df.dropna(subset=['scene_type', 'coordinates'])
        
        if len(merged_df) == 0:
            if verbose:
                print("Warning: No scene type or coordinate information found for filtered products, using random sampling")
            sampled_df = files_to_use.sample(n=min(max_samples, len(files_to_use)), random_state=42)
            return [str(f) for f in sampled_df['full_name'].tolist()]
        
        return _get_balanced_representation_samples(
            merged_df, max_samples, min_samples_per_cluster, verbose
        )
    else:
        # Simple random sampling if no scene type or coordinate information
        if verbose:
            print("No scene type or coordinate information available, using random sampling")
        sampled_df = files_to_use.sample(n=max_samples, random_state=42)
        return [str(f) for f in sampled_df['full_name'].tolist()]


def _get_balanced_representation_samples(
    df: pd.DataFrame, 
    max_samples: int, 
    min_samples_per_cluster: int = 1,
    verbose: bool = False
) -> List[str]:
    """
    Internal method to get balanced samples ensuring EQUAL scene type representation.
    
    Strategy:
    1. Drop duplicate positions (same geographic location)
    2. Aim for samples_number/3 samples from each scene type (land, sea, coast)
    3. If one scene type lacks enough samples, fill the gap with ROUND-ROBIN sampling
    4. Maintain spatial diversity by avoiding overlapping samples
    
    IMPORTANT FIX (2024):
    Previous version sorted deficit filling by available_for_filling (descending),
    which caused the sea class (most samples) to always fill the entire deficit first,
    even when land/coast had valid non-overlapping samples available.
    
    New approach uses ROUND-ROBIN: cycles through all scene types one sample at a time,
    ensuring fair representation even when categories have different sample counts.
    
    Args:
        df (pd.DataFrame): DataFrame with product information including 'coordinates' and 'scene_type'
        max_samples (int): Maximum number of samples to select
        min_samples_per_cluster (int): Not used in new implementation (kept for compatibility)
        verbose (bool): Whether to print detailed information
        
    Returns:
        List[str]: List of balanced filenames with paths
    """
    import ast
    from shapely.geometry import Polygon
    
    if 'scene_type' not in df.columns:
        if verbose:
            print("Warning: Missing 'scene_type' column, using random sampling")
        sampled_df = df.sample(n=min(max_samples, len(df)), random_state=42)
        return [str(f) for f in sampled_df['full_name'].tolist()]
    
    # Parse coordinates if they're strings
    def parse_coords(coord_str):
        if pd.isna(coord_str) or coord_str == '':
            return None
        try:
            coords = ast.literal_eval(coord_str) if isinstance(coord_str, str) else coord_str
            # Handle nested structure: [[[lon, lat], [lon, lat], ...]]
            if isinstance(coords[0][0], list):
                coords = coords[0]
            return Polygon(coords)
        except:
            return None
    
    # Step 0: Parse polygons and drop duplicates by position
    if verbose:
        print(f"\nBalanced sampling with EQUAL scene type representation:")
        print(f"  Total available products (after filters): {len(df)}")
    
    df = df.copy()
    df['polygon'] = df['coordinates'].apply(parse_coords)
    df = df[df['polygon'].notna()]  # Remove products without valid polygons
    
    if len(df) == 0:
        if verbose:
            print("Warning: No valid polygon coordinates found, using random sampling")
        sampled_df = df.sample(n=min(max_samples, len(df)), random_state=42)
        return [str(f) for f in sampled_df['full_name'].tolist()]
    
    # Drop duplicates by centroid position (keep first occurrence)
    df['centroid_str'] = df['polygon'].apply(lambda p: f"{p.centroid.x:.4f},{p.centroid.y:.4f}")
    original_count = len(df)
    df = df.drop_duplicates(subset='centroid_str', keep='first')
    
    if verbose:
        duplicates_removed = original_count - len(df)
        print(f"  Dropped {duplicates_removed} duplicate positions")
        print(f"  Remaining unique positions: {len(df)}")
    
    scene_counts = df['scene_type'].value_counts()
    
    if verbose:
        print(f"\n  Scene type distribution after deduplication:")
        for scene, count in scene_counts.items():
            print(f"    {scene}: {count} products ({count/len(df)*100:.1f}%)")
    
    # Step 1: Calculate EQUAL allocation (samples_number/3 for each scene type)
    required_scene_types = ['land', 'sea', 'coast']
    target_per_scene = max_samples // 3
    
    if verbose:
        print(f"\n  Target: {max_samples} total samples")
        print(f"  Equal allocation: {target_per_scene} samples per scene type")
    
    # Step 2: Sample from each scene type (up to target)
    selected_samples = {}
    available_for_filling = {}
    
    for scene_type in required_scene_types:
        scene_df = df[df['scene_type'] == scene_type]
        available = len(scene_df)
        
        if available == 0:
            if verbose:
                print(f"    {scene_type}: 0 samples available - SKIPPING")
            selected_samples[scene_type] = pd.DataFrame()
            available_for_filling[scene_type] = 0
            continue
        
        if available >= target_per_scene:
            # Enough samples - select target amount with overlap avoidance
            selected = _select_non_overlapping_samples(scene_df, target_per_scene, verbose=False)
            selected_samples[scene_type] = selected
            available_for_filling[scene_type] = available - len(selected)
            
            if verbose:
                print(f"    {scene_type}: {len(selected)}/{target_per_scene} samples selected ({available} available)")
        else:
            # Not enough samples - take all available
            selected_samples[scene_type] = scene_df.copy()
            available_for_filling[scene_type] = 0
            
            if verbose:
                print(f"    {scene_type}: {available}/{target_per_scene} samples selected (LIMITED - using all available)")
    
    # Step 3: Calculate deficit and fill gaps from other scene types
    total_selected = sum(len(samples) for samples in selected_samples.values())
    deficit = max_samples - total_selected
    
    if deficit > 0:
        if verbose:
            print(f"\n  Deficit: {deficit} samples needed to reach {max_samples}")
            print(f"  Filling from scene types with extra samples...")
        
        # Get already selected polygons for overlap checking
        already_selected = pd.concat([s for s in selected_samples.values() if len(s) > 0], ignore_index=True)
        
        # Try to fill from scene types that have extra samples
        # First pass: ROUND-ROBIN strict overlap avoidance
        # This ensures fair distribution across all scene types
        max_iterations = deficit * 3  # Prevent infinite loops
        iteration = 0
        
        while deficit > 0 and iteration < max_iterations:
            made_progress = False
            
            # Round-robin: cycle through scene types in fixed order
            for scene_type in required_scene_types:  # ['land', 'sea', 'coast']
                if deficit <= 0:
                    break
                
                if available_for_filling[scene_type] <= 0:
                    continue
                
                scene_df = df[df['scene_type'] == scene_type]
                # Exclude already selected samples
                already_selected_indices = selected_samples[scene_type].index if len(selected_samples[scene_type]) > 0 else []
                remaining_df = scene_df[~scene_df.index.isin(already_selected_indices)]
                
                if len(remaining_df) == 0:
                    available_for_filling[scene_type] = 0
                    continue
                
                # Try to select ONE sample at a time (round-robin)
                additional = _select_non_overlapping_samples(
                    remaining_df, 
                    1,  # ← Only 1 sample per iteration for fairness
                    already_selected=already_selected,
                    verbose=False,
                    strict_overlap=True
                )
                
                if len(additional) > 0:
                    selected_samples[scene_type] = pd.concat([selected_samples[scene_type], additional], ignore_index=True)
                    already_selected = pd.concat([already_selected, additional], ignore_index=True)
                    deficit -= 1
                    available_for_filling[scene_type] -= 1
                    made_progress = True
                    
                    if verbose:
                        print(f"    Added 1 sample from {scene_type} (deficit now: {deficit})")
                else:
                    # No non-overlapping samples found - mark as exhausted
                    available_for_filling[scene_type] = 0
            
            if not made_progress:
                break  # No scene type could add samples - stop trying
            
            iteration += 1
        
        # Second pass: ROUND-ROBIN relaxed overlap if still deficit
        if deficit > 0:
            if verbose:
                print(f"\n  Still {deficit} deficit after strict filtering - relaxing overlap constraints...")
            
            max_iterations = deficit * 3
            iteration = 0
            
            while deficit > 0 and iteration < max_iterations:
                made_progress = False
                
                # Round-robin through scene types
                for scene_type in required_scene_types:
                    if deficit <= 0:
                        break
                    
                    scene_df = df[df['scene_type'] == scene_type]
                    already_selected_indices = selected_samples[scene_type].index if len(selected_samples[scene_type]) > 0 else []
                    remaining_df = scene_df[~scene_df.index.isin(already_selected_indices)]
                    
                    if len(remaining_df) == 0:
                        continue
                    
                    # Relaxed selection - take ONE sample at a time (round-robin)
                    additional = remaining_df.sample(n=1, random_state=42+iteration)
                    
                    if len(additional) > 0:
                        selected_samples[scene_type] = pd.concat([selected_samples[scene_type], additional], ignore_index=True)
                        deficit -= 1
                        made_progress = True
                        
                        if verbose:
                            print(f"    Added 1 sample from {scene_type} (RELAXED - deficit now: {deficit})")
                
                if not made_progress:
                    break
                
                iteration += 1
    
    # Combine all selected samples
    all_selected = [s for s in selected_samples.values() if len(s) > 0]
    
    if all_selected:
        balanced_df = pd.concat(all_selected, ignore_index=True)
    else:
        if verbose:
            print("\n  Warning: No samples selected, falling back to random sampling")
        balanced_df = df.sample(n=min(max_samples, len(df)), random_state=42)
    
    # Final verification and summary
    if verbose:
        print(f"\n  Final distribution:")
        final_scene_counts = balanced_df['scene_type'].value_counts()
        for scene in required_scene_types:
            count = final_scene_counts.get(scene, 0)
            pct = count / len(balanced_df) * 100 if len(balanced_df) > 0 else 0
            ideal_pct = 100.0 / 3
            diff = pct - ideal_pct
            print(f"    {scene}: {count} samples ({pct:.1f}%, target: {ideal_pct:.1f}%, Δ{diff:+.1f}%)")
        print(f"\n  Total selected: {len(balanced_df)} samples")
    
    return [str(f) for f in balanced_df['full_name'].tolist()]


def _select_non_overlapping_samples(
    scene_df: pd.DataFrame,
    target_count: int,
    already_selected: Optional[pd.DataFrame] = None,
    verbose: bool = False,
    strict_overlap: bool = True
) -> pd.DataFrame:
    """
    Select samples from a scene type DataFrame while avoiding overlaps.
    
    Args:
        scene_df: DataFrame of samples to select from
        target_count: Number of samples to select
        already_selected: DataFrame of already selected samples to avoid overlapping with
        verbose: Whether to print progress
        strict_overlap: If True, enforces strict overlap avoidance. If False, just basic dedup.
        
    Returns:
        DataFrame of selected non-overlapping samples
    """
    selected = []
    selected_indices = set()
    
    # Shuffle candidates for randomness
    candidates = scene_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    for idx, candidate in candidates.iterrows():
        if len(selected) >= target_count:
            break
        
        candidate_poly = candidate['polygon']
        
        if strict_overlap:
            # Check overlap with already selected samples from this batch
            overlaps = False
            for sel_sample in selected:
                if candidate_poly.intersects(sel_sample['polygon']):
                    intersection = candidate_poly.intersection(sel_sample['polygon'])
                    if intersection.area > 0.0001:  # Small threshold
                        overlaps = True
                        break
            
            # Check overlap with previously selected samples from other scene types
            if not overlaps and already_selected is not None and len(already_selected) > 0:
                for _, already_sel in already_selected.iterrows():
                    if candidate_poly.intersects(already_sel['polygon']):
                        intersection = candidate_poly.intersection(already_sel['polygon'])
                        if intersection.area > 0.0001:
                            overlaps = True
                            break
            
            if not overlaps:
                selected.append(candidate)
        else:
            # Relaxed mode - just add it
            selected.append(candidate)
    
    if len(selected) > 0:
        return pd.DataFrame(selected)
    else:
        return pd.DataFrame()  # Empty DataFrame with same columns


def validate_balanced_sampling(
    selected_files: List[str],
    config_path: str = "/Data_large/marine/PythonProjects/SAR/sarpyx/dataset_splits",
    split_type: str = "train"
) -> dict:
    """
    Validate that the selected files maintain balanced representation.
    
    Args:
        selected_files (List[str]): List of selected filenames
        config_path (str): Path to dataset splits configuration
        split_type (str): Type of split ('train', 'validation', 'test')
        
    Returns:
        dict: Statistics about the selected samples
    """
    
    csv_path = os.path.join(config_path, f"{split_type}_products.csv")
    if not os.path.exists(csv_path):
        return {"error": f"Configuration file not found: {csv_path}"}
    
    # Try loading with semicolon delimiter first (new format)
    try:
        df = pd.read_csv(csv_path, sep=';')
    except:
        # Fall back to comma delimiter (old format)
        df = pd.read_csv(csv_path)
    selected_df = df[df['filename'].isin(selected_files)]
    
    stats = {
        "total_selected": len(selected_files),
        "total_available": len(df),
        "selection_rate": len(selected_files) / len(df) if len(df) > 0 else 0
    }
    
    if 'geo_cluster' in selected_df.columns:
        stats["cluster_distribution"] = selected_df['geo_cluster'].value_counts().to_dict()
        stats["clusters_represented"] = len(selected_df['geo_cluster'].unique())
        stats["total_clusters"] = len(df['geo_cluster'].unique())
    
    if 'scene_type' in selected_df.columns:
        stats["scene_type_distribution"] = selected_df['scene_type'].value_counts().to_dict()
        stats["scene_types_represented"] = len(selected_df['scene_type'].unique())
        stats["total_scene_types"] = len(df['scene_type'].unique())
    
    if 'year' in selected_df.columns:
        stats["year_distribution"] = selected_df['year'].value_counts().to_dict()
    
    return stats


def create_transforms_from_config(transforms_cfg):
    """Create SARTransform from configuration (same as original)."""
    _ensure_dataloader_dependencies()

    transform_cls = SARTransform
    if transform_cls is None:  # pragma: no cover - defensive safeguard
        raise RuntimeError("SARTransform is unavailable even after lazy import.")

    if transforms_cfg.get('normalize', True):
        normalization_type = transforms_cfg.get('normalization_type', 'minmax')
        complex_valued = transforms_cfg.get('complex_valued', True)
        adaptive = transforms_cfg.get('adaptive', False)

        if normalization_type == 'minmax':
            rc_min = transforms_cfg.get('rc_min', RC_MIN)
            rc_max = transforms_cfg.get('rc_max', RC_MAX)
            gt_min = transforms_cfg.get('gt_min', GT_MIN)
            gt_max = transforms_cfg.get('gt_max', GT_MAX)

            transforms = transform_cls.create_minmax_normalized_transform(
                normalize=True,
                adaptive=adaptive,
                rc_min=rc_min,
                rc_max=rc_max,
                gt_min=gt_min,
                gt_max=gt_max,
                complex_valued=complex_valued
            )

        elif normalization_type in ['zscore', 'standardize']:
            if adaptive:
                transforms = transform_cls.create_zscore_normalized_transform(
                    normalize=True,
                    adaptive=True,
                    complex_valued=complex_valued
                )
            else:
                rc_mean = transforms_cfg.get('rc_mean', 0.0)
                rc_std = transforms_cfg.get('rc_std', 1.0)
                gt_mean = transforms_cfg.get('gt_mean', 0.0)
                gt_std = transforms_cfg.get('gt_std', 1.0)

                transforms = transform_cls.create_zscore_normalized_transform(
                    normalize=True,
                    adaptive=False,
                    rc_mean=rc_mean,
                    rc_std=rc_std,
                    gt_mean=gt_mean,
                    gt_std=gt_std,
                    complex_valued=complex_valued
                )

        elif normalization_type == 'robust':
            transforms = transform_cls.create_robust_normalized_transform(
                normalize=True,
                adaptive=adaptive,
                complex_valued=complex_valued
            )

        else:
            raise ValueError(f"Unsupported normalization_type: {normalization_type}")
    else:
        transforms = transform_cls()

    return transforms


def create_dataloader_from_config(data_dir, dataloader_cfg, split_cfg, transforms):
    """Create a dataloader from base config and split-specific config (same as original)."""
    _ensure_dataloader_dependencies()

    base_config = {
        'data_dir': data_dir,
        'level_from': dataloader_cfg.get('level_from', 'rcmc'),
        'level_to': dataloader_cfg.get('level_to', 'az'),
        'num_workers': dataloader_cfg.get('num_workers', 0),  # Default to 0 to prevent worker crashes
        'patch_mode': dataloader_cfg.get('patch_mode', 'rectangular'),
        'patch_size': tuple(dataloader_cfg.get('patch_size', [1000, 1])),
        'buffer': tuple(dataloader_cfg.get('buffer', [0, 0])),
        'stride': tuple(dataloader_cfg.get('stride', [300, 1])),
        'shuffle_files': dataloader_cfg.get('shuffle_files', False),
        'complex_valued': dataloader_cfg.get('complex_valued', False),
        'save_samples': dataloader_cfg.get('save_samples', False),
        'backend': dataloader_cfg.get('backend', 'zarr'),
        'verbose': dataloader_cfg.get('verbose', True),
        'cache_size': dataloader_cfg.get('cache_size', 100),  # Reduced cache size
        'online': dataloader_cfg.get('online', True),
        'concatenate_patches': dataloader_cfg.get('concatenate_patches', True),
        'concat_axis': dataloader_cfg.get('concat_axis', 0),
        'positional_encoding': dataloader_cfg.get('positional_encoding', True),
        'transform': transforms,
        'block_pattern': split_cfg.get('block_pattern', None)
    }
    
    split_config = {
        'batch_size': split_cfg.get('batch_size', 16),
        'samples_per_prod': split_cfg.get('samples_per_prod', 1000),
        'patch_order': split_cfg.get('patch_order', 'row'),
        'max_products': split_cfg.get('max_products', 1),
        'filters': split_cfg.get('filters', {})
    }
    
    final_config = {**base_config, **split_config}

    dataloader_fn = get_sar_dataloader
    if dataloader_fn is None:  # pragma: no cover - defensive safeguard
        raise RuntimeError("get_sar_dataloader is unavailable even after lazy import.")

    return dataloader_fn(**final_config)


def create_dataloaders(dataloader_cfg):
    """Create train, validation, and test dataloaders (same as original)."""
    # Import here to avoid circular import
    from location_utils import get_products_spatial_mapping
    
    data_dir = dataloader_cfg.get('data_dir', '/Data/sar_focusing_new')

    # Only create the locations CSV if it doesn't exist
    locations_csv_path = os.path.join(data_dir, LOCATIONS_CSV_FILENAME)
    if not os.path.exists(locations_csv_path):
        print(f"📍 Creating product locations CSV at {locations_csv_path}...")
        get_products_spatial_mapping(
            author=dataloader_cfg.get('author', 'Maya4'), 
            repos=dataloader_cfg.get('repos', ['PT1', 'PT2', 'PT3', 'PT4']), 
            data_dir=data_dir, 
            output_csv_file_path=locations_csv_path,
            overwrite_csv=False
        )
    else:
        print(f"✓ Using existing product locations CSV: {locations_csv_path}")
    
    transforms_cfg = dataloader_cfg.get('transforms', {})
    transforms = create_transforms_from_config(transforms_cfg)
    
    train_cfg = dataloader_cfg.get('train', {})
    train_cfg['split'] = 'train'
    train_filters = train_cfg.get('filters', {})
    train_cfg['filters'] = SampleFilter(**train_filters) if train_filters else None
    train_loader = create_dataloader_from_config(data_dir, dataloader_cfg, train_cfg, transforms)
    
    val_cfg = dataloader_cfg.get('validation', {})
    val_cfg['split'] = 'validation'
    val_filters = val_cfg.get('filters', {})
    val_cfg['filters'] = SampleFilter(**val_filters) if val_filters else None
    val_loader = create_dataloader_from_config(data_dir, dataloader_cfg, val_cfg, transforms)
    
    test_cfg = dataloader_cfg.get('test', {})
    test_cfg['split'] = 'test'
    test_filters = test_cfg.get('filters', {})
    test_cfg['filters'] = SampleFilter(**test_filters) if test_filters else None
    test_loader = create_dataloader_from_config(data_dir, dataloader_cfg, test_cfg, transforms)
    
    if 'inference' not in dataloader_cfg:
        return train_loader, val_loader, test_loader, None
    else: 
        inference_cfg = dataloader_cfg.get('inference', {})
        inference_filters = inference_cfg.get('filters', {})
        inference_cfg['filters'] = SampleFilter(**inference_filters) if inference_filters else None
        inference_loader = create_dataloader_from_config(data_dir, dataloader_cfg, inference_cfg, transforms)
        return train_loader, val_loader, test_loader, inference_loader
