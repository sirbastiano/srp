"""
Utility functions for SARPyX CLI tools.

This module provides common utility functions used across multiple CLI commands.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union


def validate_path(path: Union[str, Path], must_exist: bool = True) -> Path:
    """
    Validate and convert a path string to a Path object.
    
    Args:
        path: Path string or Path object to validate.
        must_exist: Whether the path must exist.
        
    Returns:
        Validated Path object.
        
    Raises:
        SystemExit: If validation fails.
    """
    path_obj = Path(path)
    
    if must_exist and not path_obj.exists():
        print(f'Error: Path does not exist: {path_obj}', file=sys.stderr)
        sys.exit(1)
    
    return path_obj


def create_output_directory(output_dir: Union[str, Path]) -> Path:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Output directory path.
        
    Returns:
        Path object for the output directory.
        
    Raises:
        SystemExit: If directory creation fails.
    """
    output_path = Path(output_dir)
    
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    except Exception as e:
        print(f'Error: Cannot create output directory {output_path}: {e}', file=sys.stderr)
        sys.exit(1)


def check_snap_installation(gpt_path: Optional[str] = None) -> bool:
    """
    Check if SNAP GPT is available.
    
    Args:
        gpt_path: Optional path to GPT executable.
        
    Returns:
        True if SNAP GPT is available, False otherwise.
    """
    import subprocess
    
    if gpt_path:
        gpt_executable = gpt_path
    else:
        # Try common locations
        common_paths = [
            '/Applications/snap/bin/gpt',  # macOS
            '/home/*/ESA-STEP/snap/bin/gpt',  # Linux
            'gpt',  # System PATH
            'gpt.exe'  # Windows
        ]
        
        gpt_executable = None
        for path in common_paths:
            if path.startswith('/home/*/'):
                # Expand wildcard for Linux
                import glob
                matches = glob.glob(path)
                if matches:
                    gpt_executable = matches[0]
                    break
            else:
                if Path(path).exists() or path in ['gpt', 'gpt.exe']:
                    gpt_executable = path
                    break
    
    if not gpt_executable:
        return False
    
    try:
        result = subprocess.run(
            [gpt_executable, '--help'],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def print_processing_summary(
    product_path: str,
    output_path: str,
    parameters: Dict[str, Union[str, float, bool, List[str]]]
) -> None:
    """
    Print a summary of processing parameters.
    
    Args:
        product_path: Input product path.
        output_path: Output path.
        parameters: Dictionary of processing parameters.
    """
    print('='*60)
    print('PROCESSING SUMMARY')
    print('='*60)
    print(f'Input product: {product_path}')
    print(f'Output: {output_path}')
    print('')
    
    print('Processing parameters:')
    for key, value in parameters.items():
        if isinstance(value, list):
            value_str = ', '.join(str(v) for v in value)
        else:
            value_str = str(value)
        print(f'  {key}: {value_str}')
    
    print('='*60)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes.
        
    Returns:
        Formatted size string.
    """
    if size_bytes == 0:
        return '0 B'
    
    size_names = ['B', 'KB', 'MB', 'GB', 'TB']
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f'{size_bytes:.1f} {size_names[i]}'


def get_product_info(product_path: Union[str, Path]) -> Dict[str, str]:
    """
    Extract basic information from a SAR product path.
    
    Args:
        product_path: Path to SAR product.
        
    Returns:
        Dictionary with product information.
    """
    path_obj = Path(product_path)
    
    info = {
        'name': path_obj.name,
        'size': 'Unknown',
        'type': 'Unknown'
    }
    
    # Get size if it exists
    try:
        if path_obj.is_file():
            info['size'] = format_file_size(path_obj.stat().st_size)
        elif path_obj.is_dir():
            total_size = sum(f.stat().st_size for f in path_obj.rglob('*') if f.is_file())
            info['size'] = format_file_size(total_size)
    except Exception:
        pass
    
    # Determine product type from name
    name_lower = path_obj.name.lower()
    if 's1a_' in name_lower or 's1b_' in name_lower:
        info['type'] = 'Sentinel-1'
    elif 'csk_' in name_lower or 'cosmo' in name_lower:
        info['type'] = 'COSMO-SkyMed'
    elif 'tsx_' in name_lower or 'terrasar' in name_lower:
        info['type'] = 'TerraSAR-X'
    elif name_lower.endswith('.safe'):
        info['type'] = 'SAR Product (.SAFE)'
    elif name_lower.endswith('.zip'):
        info['type'] = 'Compressed SAR Product'
    
    return info


def validate_window_sizes(
    target_window: float,
    guard_window: float,
    background_window: float
) -> None:
    """
    Validate that window sizes are in correct relationship.
    
    Args:
        target_window: Target window size.
        guard_window: Guard window size.
        background_window: Background window size.
        
    Raises:
        SystemExit: If validation fails.
    """
    if target_window >= guard_window:
        print('Error: Target window size should be smaller than guard window size', file=sys.stderr)
        sys.exit(1)
    
    if guard_window >= background_window:
        print('Error: Guard window size should be smaller than background window size', file=sys.stderr)
        sys.exit(1)
    
    # Additional validation for reasonable sizes
    if target_window < 1.0:
        print('Warning: Target window size is very small (< 1m). This may affect detection performance.')
    
    if background_window > 2000.0:
        print('Warning: Background window size is very large (> 2km). This may affect processing speed.')


def estimate_processing_time(product_path: Union[str, Path]) -> str:
    """
    Provide a rough estimate of processing time based on product size.
    
    Args:
        product_path: Path to SAR product.
        
    Returns:
        Estimated processing time string.
    """
    try:
        path_obj = Path(product_path)
        
        if path_obj.is_file():
            size_mb = path_obj.stat().st_size / (1024 * 1024)
        elif path_obj.is_dir():
            total_size = sum(f.stat().st_size for f in path_obj.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
        else:
            return 'Unknown'
        
        # Rough estimates based on typical processing times
        if size_mb < 100:
            return '< 5 minutes'
        elif size_mb < 500:
            return '5-15 minutes'
        elif size_mb < 1000:
            return '15-30 minutes'
        elif size_mb < 2000:
            return '30-60 minutes'
        else:
            return '> 1 hour'
    
    except Exception:
        return 'Unknown'
