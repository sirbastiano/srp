"""
Location Extraction Utilities Using Phidown

This module provides utilities to extract geographic location information
from Sentinel-1 zarr product filenames using the phidown tool.

Instead of reading ephemeris data from zarr metadata and converting ECEF coordinates
to lat/lon, this approach queries the Copernicus Data Space catalog directly
using the product name derived from the filename.

Author: SAR Processing Team
Date: October 2025
"""

from phidown.search import CopernicusDataSearcher
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
import time
from tqdm import tqdm
import glob
import os
from datetime import datetime

def create_valid_datetime_string(date_str: str) -> Optional[str]:
    """
    Convert a date string to a valid datetime format (YYYY-MM-DDTHH:MM:SS).
    
    Args:
        date_str: Input date string (e.g., '20230331t123129')
        
    Returns:
        Valid datetime string or None if invalid
    """
    try:
        dt = datetime.strptime(date_str, "%Y%m%dT%H%M%S")
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
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

def extract_location_from_zarr_filename_with_phidown(zarr_filename: str) -> Optional[Dict]:
    """
    Extract geographic location information from a zarr filename using phidown.
    
    Args:
        zarr_filename: Name of the zarr file
        
    Returns:
        Dictionary with location information
    """
    from phidown.search import CopernicusDataSearcher
    
    result = {
        'coordinates': None,
        'footprint': None,
        'product_name': None,
        'success': False,
        'error': None
    }
    
    # Remove .zarr extension if present
    result['part'] = get_part_from_filename(zarr_filename)
    zarr_filename = os.path.basename(zarr_filename)
    result['filename'] = zarr_filename
    product_name = zarr_filename.replace('.zarr', '') if zarr_filename.endswith('.zarr') else zarr_filename
    
    # Parse zarr filename components
    # Format: s1a-s1-raw-s-vv-20230331t123129-20230331t123154-047888-05c119
    parts = product_name.split('-')
    
    if len(parts) < 8:
        result['error'] = f"Invalid filename format: expected at least 8 parts, got {len(parts)}"
        return result
    
    # Extract components
    satellite = parts[0].upper()         # s1a -> S1A
    swath_mode = parts[1].upper()        # s1, s3, s6 (stripmap modes)
    polarization = parts[4].upper()      # vv, hh, vh, hv
    acquisition_date = create_valid_datetime_string(parts[5])  # 20230331t123129
    end_date = create_valid_datetime_string(parts[6])          # 20230331t123154
    orbit_num = parts[7]                 # 047888
    data_take = parts[8]                 # 05c119
    
    result['acquisition_date'] = acquisition_date
    result['end_date'] = end_date
    result['satellite'] = satellite
    result['swath_mode'] = swath_mode
    result['polarization'] = polarization
    result['orbit_num'] = orbit_num
    result['data_take'] = data_take
    
    # Construct SAFE product name pattern
    # Format: S1A_S3_RAW__0SSH_20230331T123129_20230331T123154_047888_05C119
    pol_code = f"0S{polarization[0:2].upper()}" if len(polarization) >= 2 else "0SSH"
    safe_pattern = f"{satellite}_S{swath_mode[1:]}_RAW__{pol_code}_{acquisition_date.upper()}_{end_date.upper()}_{orbit_num}_{data_take.upper()}"
    result['product_name'] = safe_pattern
    
    # Initialize Copernicus searcher
    searcher = CopernicusDataSearcher()
    
    # Search for the product
    searcher.query_by_filter(
        collection_name='SENTINEL-1',
        product_type=None,
        orbit_direction=None,
        cloud_cover_threshold=None,
        aoi_wkt=None,
        start_date=acquisition_date,
        end_date=end_date,
        top=50,  # Get more results to find the right one
        attributes={'processingLevel':'LEVEL0',  # RAW data is Level 0
                'operationalMode': 'SM'}      # Stripmap mode from 's' in filename
    )

    df_search = searcher.execute_query()
    print(f"\nFound {len(df_search)} products in the time range:")

    if not df_search.empty:
        # Look for products that might match our zarr file
        print("\nSearching for matching products...")
        
        # Display products to find the one that matches
        display_columns = ['Name', 'GeoFootprint', 'ContentDate', 'S3Path']
        available_columns = [col for col in display_columns if col in df_search.columns]
        
        print("\nTop 10 products in the time range:")
        print(df_search[available_columns].head(10).to_string())
    else:
        print("No products found in the specified time range.")
    
    if len(df_search) > 0:
        # Get the first result (should be exact match)
        product = df_search.iloc[0]
        # Extract footprint (WKT polygon)
        result['coordinates'] = product['GeoFootprint']['coordinates']
    else:
        print("No matching products found for product .")

    return result

def find_sar_products(root_dir: str, extensions: List[str] = ['.zarr']) -> List[Path]:
    """
    Recursively scan directory for SAR product files.
    
    Args:
        root_dir (str): Root directory to scan
        extensions (List[str]): File extensions to look for
        
    Returns:
        List[Path]: List of SAR product file paths
    """
    print(f"Scanning for SAR products in: {root_dir}")
    product_files = []
    
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Warning: Directory {root_dir} does not exist")
        return product_files
    
    for ext in extensions:
        # Use glob to find all files with the extension
        pattern = f"**/*{ext}"
        matches = list(root_path.glob(pattern))
        product_files.extend(matches)
        print(f"  Found {len(matches)} files with extension {ext}")
    
    # Remove duplicates and sort
    product_files = sorted(list(set(product_files)))
    print(f"Total unique SAR products found: {len(product_files)}")
    
    return product_files

def get_products_spatial_mapping(author: str, repos: List[str], data_dir: Union[str, os.PathLike], verbose: bool = True, extensions: List[str] = ['.zarr'], output_csv_file_path: Optional[Union[str, os.PathLike]] = None, overwrite_csv: bool = False) -> pd.DataFrame:
    """
    Get spatial mapping (latitude, longitude) for a list of products.
    
    Args:
        products (List[str]): List of product names
        mapping_df (pd.DataFrame): DataFrame containing product to spatial mapping  with columns ['product_name', 'latitude', 'longitude']
    """
    all_products = []

    # try:
    from api import list_repos_by_author, list_base_files_in_repo

    if not overwrite_csv and output_csv_file_path is not None and os.path.exists(Path(output_csv_file_path)):
        df = pd.read_csv(Path(output_csv_file_path))
        return df
    
    for part in repos:

        repos = list_repos_by_author(author)
        
        part_repos = [repo for repo in repos if part.lower() in repo.lower()]

        for repo in part_repos:
            print(f"    Scanning repository: {repo}")
            repo_files = list_base_files_in_repo(f"{repo}")
            
            for filename in repo_files:
                # filename = file_info.get('filename', '')
                if any(filename.endswith(ext) for ext in extensions):
                    # Create a remote product entry
                    remote_product = {
                        'source': 'remote_hf',
                        'repo': repo,
                        'part': part,
                        'filename': filename,
                        'full_path': f"hf://{repo}/{filename}",
                    }
                    all_products.append(remote_product)
    if verbose:
        print(f"Remote products found: {len(all_products)}")
        print("="*60)
        print("EXTRACTING LOCATION INFORMATION FROM ALL PRODUCTS")
        print("="*60)

    if not all_products:
        if verbose:
            print("No valid products found. Creating empty DataFrame.")
        products_df = pd.DataFrame(columns=[
            'source', 'file_path', 'filename', 'latitude', 'longitude', 'num_samples',
            'file_size_mb', 'shape', 'metadata_available', 'ephemeris_available', 'error'
        ])
    else:
        # Process all products
        product_data = []
        
        # Process local products
        if verbose:
            print(f"\n🔍 Processing { len(all_products) } products in total...")

        for i, remote_product in enumerate(all_products):
            # current_index += 1
            # if current_index % 5 == 0:  # More frequent progress for remote (slower)
            #     print(f"  Progress: {current_index}/{total_products} ({current_index/total_products*100:.1f}%)")
            
            # Extract location info from remote product using phidown
            filename = remote_product.get('filename', '') if isinstance(remote_product, dict) else str(remote_product)
            part = remote_product.get('part', '') if isinstance(remote_product, dict) else 'PT1'
            product_info = extract_location_from_zarr_filename_with_phidown(f"{data_dir}/{part}/{filename}")
            product_data.append(product_info)
        
        # Create DataFrame
        products_df = pd.DataFrame(product_data)
        if verbose:
            print(f"DataFrame created with {len(products_df)} products")
            print("="*60)
            print("SAVING DATA TO CSV")
            print("="*60)

    # Create output directory if it doesn't exist
    output_dir = Path(str(output_csv_file_path)).parent
    output_dir.mkdir(exist_ok=True)
    from datetime import datetime
    # Generate timestamp for unique filename

    try:
        # Save the complete DataFrame to CSV
        products_df.to_csv(output_csv_file_path, index=False)
        if verbose: 
            print(f"✅ Successfully saved {len(products_df)} products to: {output_csv_file_path}")

    except Exception as e:
        print(f"❌ Error saving to CSV: {e}")
    
    return products_df


def get_sar_product_locations(
    author: str,
    repos: List[str],
    data_dir: Union[str, os.PathLike],
    verbose: bool = True,
    extensions: List[str] = ['.zarr'],
    output_csv_file_path: Optional[Union[str, os.PathLike]] = None,
    overwrite_csv: bool = False
) -> pd.DataFrame:
    """Backward-compatible wrapper for legacy notebooks.

    Delegates to :func:`get_products_spatial_mapping` while preserving the
    original function signature used throughout older notebooks.
    """
    return get_products_spatial_mapping(
        author=author,
        repos=repos,
        data_dir=data_dir,
        verbose=verbose,
        extensions=extensions,
        output_csv_file_path=output_csv_file_path,
        overwrite_csv=overwrite_csv
    )


def get_location_for_zarr_file(zarr_path: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Get latitude and longitude for a zarr file using phidown.
    
    This is a simplified interface that returns just the coordinates.
    
    Args:
        zarr_path: Full or relative path to zarr file
        
    Returns:
        Tuple of (latitude, longitude) or (None, None) if extraction fails
        
    Example:
        >>> lat, lon = get_location_for_zarr_file("/path/to/s1a-s1-raw-s-vv-20230331t123129-20230331t123154-047888-05c119.zarr")
        >>> if lat and lon:
        ...     print(f"Location: {lat:.4f}°N, {lon:.4f}°E")
    """
    # Extract filename from path
    filename = Path(zarr_path).name
    
    # Use phidown to get location
    location_info = _extract_location(filename)
    
    if location_info and location_info['success']:
        return location_info['latitude'], location_info['longitude']
    else:
        return None, None


def extract_locations_for_all_zarr_files(
    data_dir: str, 
    max_files: Optional[int] = None,
    recursive: bool = True,
    save_csv: bool = True,
    output_csv: str = 'zarr_locations_phidown.csv',
    api_delay: float = 0.1
) -> pd.DataFrame:
    """
    Extract geographic locations for all zarr files in a directory using phidown.
    
    Args:
        data_dir: Root directory containing zarr files
        max_files: Maximum number of files to process (None for all)
        recursive: Whether to search recursively in subdirectories
        save_csv: Whether to save results to CSV
        output_csv: Path to output CSV file
        api_delay: Delay in seconds between API calls to avoid rate limiting
        
    Returns:
        DataFrame with columns: filename, filepath, latitude, longitude, 
                                footprint, product_name, success, error
                                
    Example:
        >>> df = extract_locations_for_all_zarr_files(
        ...     data_dir="/Data_large/marine/SAR/sarpyx/data",
        ...     max_files=10,
        ...     save_csv=True
        ... )
        >>> print(f"Successfully extracted {df['success'].sum()} locations")
    """
    print(f"Scanning for zarr files in: {data_dir}")
    print(f"Recursive search: {recursive}")
    
    # Find all zarr files
    if recursive:
        zarr_pattern = f"{data_dir}/**/*.zarr"
        zarr_files = glob.glob(zarr_pattern, recursive=True)
    else:
        zarr_pattern = f"{data_dir}/*.zarr"
        zarr_files = glob.glob(zarr_pattern)
    
    print(f"Found {len(zarr_files)} zarr files")
    
    if max_files and len(zarr_files) > max_files:
        zarr_files = zarr_files[:max_files]
        print(f"Processing first {max_files} files")
    
    # Extract locations for each file
    results = []
    
    for zarr_path in tqdm(zarr_files, desc="Extracting locations"):
        filepath = zarr_path
        filename = Path(zarr_path).name
        
        # Extract location using phidown
        location_info = extract_location_from_zarr_filename_with_phidown(filename)
        
        # Create result entry
        result = {
            'filename': filename,
            'filepath': filepath,
            'latitude': location_info.get('latitude') if location_info else None,
            'longitude': location_info.get('longitude') if location_info else None,
            'footprint': location_info.get('footprint') if location_info else None,
            'product_name': location_info.get('product_name') if location_info else None,
            'success': location_info.get('success', False) if location_info else False,
            'error': location_info.get('error') if location_info else 'Unknown error'
        }
        
        results.append(result)
        
        # Add delay to avoid overwhelming the Copernicus API
        if api_delay > 0:
            time.sleep(api_delay)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    successful = df['success'].sum()
    failed = len(df) - successful
    
    print(f"\n{'='*60}")
    print(f"LOCATION EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {len(df)}")
    print(f"Successful extractions: {successful} ({successful/len(df)*100:.1f}%)")
    print(f"Failed extractions: {failed} ({failed/len(df)*100:.1f}%)")
    
    if failed > 0:
        print(f"\nCommon errors:")
        error_counts = df[~df['success']]['error'].value_counts()
        for error, count in error_counts.head(3).items():
            print(f"  - {error}: {count} files")
    
    # Show sample of successful extractions
    if successful > 0:
        print(f"\nSample successful extractions:")
        sample = df[df['success']].head(3)
        for idx, row in sample.iterrows():
            print(f"  - {row['filename']}")
            print(f"    Lat: {row['latitude']:.4f}, Lon: {row['longitude']:.4f}")
    
    # Save to CSV if requested
    if save_csv and len(df) > 0:
        df.to_csv(output_csv, index=False)
        print(f"\n💾 Saved results to: {output_csv}")
    
    return df


def batch_get_locations_from_filelist(
    filepaths: List[str],
    show_progress: bool = True,
    api_delay: float = 0.1
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """
    Get locations for a list of zarr file paths.
    
    Args:
        filepaths: List of zarr file paths
        show_progress: Whether to show progress bar
        api_delay: Delay between API calls
        
    Returns:
        Dictionary mapping filepath to (latitude, longitude) tuple
        
    Example:
        >>> files = ["file1.zarr", "file2.zarr", "file3.zarr"]
        >>> locations = batch_get_locations_from_filelist(files)
        >>> for filepath, (lat, lon) in locations.items():
        ...     if lat and lon:
        ...         print(f"{filepath}: {lat:.4f}°N, {lon:.4f}°E")
    """
    locations = {}
    
    iterator = tqdm(filepaths, desc="Extracting locations") if show_progress else filepaths
    
    for filepath in iterator:
        lat, lon = get_location_for_zarr_file(filepath)
        locations[filepath] = (lat, lon)
        
        if api_delay > 0:
            time.sleep(api_delay)
    
    return locations


# Example usage
if __name__ == "__main__":
    print("Phidown Location Extraction Utilities")
    print("=" * 60)
    
    # Example 1: Single file
    print("\nExample 1: Extract location from single file")
    sample_filename = "s1a-s1-raw-s-vv-20230331t123129-20230331t123154-047888-05c119.zarr"
    location = extract_location_from_zarr_filename_with_phidown(sample_filename)
    
    if location and location['success']:
        print(f"✓ {sample_filename}")
        print(f"  Lat: {location['latitude']:.6f}, Lon: {location['longitude']:.6f}")
    else:
        print(f"✗ Failed to extract location")
        if location and location.get('error'):
            print(f"  Error: {location['error']}")
    
    # Example 2: Simplified interface
    print("\nExample 2: Simplified interface")
    lat, lon = get_location_for_zarr_file(sample_filename)
    if lat and lon:
        print(f"✓ Location: {lat:.4f}°N, {lon:.4f}°E")
    
    print("\n" + "=" * 60)
    print("Ready to use! Import this module in your scripts:")
    print("  from location_utils_phidown import get_location_for_zarr_file")
    print("  lat, lon = get_location_for_zarr_file('your_file.zarr')")
