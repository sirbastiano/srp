"""
Constants module for SAR autofocus processing.

This module provides functions to load and compute constants required for SAR
autofocus algorithms, supporting both metadata-driven and default configurations.
"""

from typing import Dict, Any, Tuple, Optional, Union
import math

try:
    import torch
    TensorType = torch.Tensor
except ImportError:
    torch = None
    TensorType = Any
    print("Warning: PyTorch not available. Some functionality will be limited.")

try:
    import sentinel1decoder
except ImportError:
    sentinel1decoder = None
    print("Warning: sentinel1decoder not available. Using fallback constants.")

# Physical constants (fallback values when sentinel1decoder is not available)
SPEED_OF_LIGHT_MPS = 299792458.0  # m/s
TX_WAVELENGTH_M = 0.055465764662349676  # C-band wavelength in meters
F_REF = 37.53472224e6  # Reference frequency in Hz
SUPPRESSED_DATA_FACTOR = 320 / 8  # Factor for suppressed data time calculation


def _get_physical_constants() -> Dict[str, float]:
    """Get physical constants from sentinel1decoder or use fallbacks."""
    if sentinel1decoder is not None:
        return {
            'wavelength': sentinel1decoder.constants.TX_WAVELENGTH_M,
            'c': sentinel1decoder.constants.SPEED_OF_LIGHT_MPS,
            'f_ref': sentinel1decoder.constants.F_REF
        }
    return {
        'wavelength': TX_WAVELENGTH_M,
        'c': SPEED_OF_LIGHT_MPS,
        'f_ref': F_REF
    }


def _compute_time_vectors(constants: Dict[str, Any]) -> Dict[str, Any]:
    """Compute time-related vectors for SAR processing."""
    if torch is None:
        raise ImportError("PyTorch is required for tensor operations")
    
    # Range sampling vectors
    sample_indices = torch.arange(0, constants['len_range_line'], dtype=torch.float32)
    constants['sample_num_along_range_line'] = sample_indices
    
    # Fast time vector
    fast_time_vec = (constants['range_start_time'] + 
                     constants['range_sample_period'] * sample_indices)
    constants['fast_time_vec'] = fast_time_vec
    
    # Azimuth frequency vector
    az_freq_start = -constants['az_sample_freq'] / 2
    az_freq_end = constants['az_sample_freq'] / 2
    az_freq_step = 1 / (constants['PRI'] * constants['len_az_line'])
    constants['f_eta'] = torch.arange(
        start=az_freq_start, 
        end=az_freq_end, 
        step=az_freq_step, 
        dtype=torch.float32
    )
    
    return constants


def _compute_tx_replica(constants: Dict[str, Any], meta: Any) -> TensorType:
    """Compute transmit pulse replica for range compression."""
    if torch is None:
        raise ImportError("PyTorch is required for tensor operations")
    
    # Extract pulse parameters from metadata
    tx_pulse_start_freq = meta["Tx Pulse Start Frequency"].unique()[0]
    tx_ramp_rate = meta["Tx Ramp Rate"].unique()[0] 
    tx_pulse_length = meta["Tx Pulse Length"].unique()[0]
    
    # Generate time vector for pulse
    num_tx_vals = int(tx_pulse_length * constants['range_sample_freq'])
    tx_time_vals = torch.linspace(-tx_pulse_length/2, tx_pulse_length/2, 
                                  steps=num_tx_vals, dtype=torch.float32)
    
    # Compute phase components
    phi1 = tx_pulse_start_freq + tx_ramp_rate * tx_pulse_length / 2
    phi2 = tx_ramp_rate / 2
    
    # Generate complex exponential replica
    phase = 2j * math.pi * (phi1 * tx_time_vals + phi2 * tx_time_vals**2)
    tx_replica = torch.exp(phase)
    
    return tx_replica


def load_constants_from_meta(
    meta: Any, 
    patch_dim: Tuple[int, int] = (4096, 4096),
    use_torch: bool = True
) -> Dict[str, Any]:
    """
    Load SAR processing constants from metadata.
    
    This function extracts and computes all necessary constants for SAR processing
    from the provided metadata, including physical constants, timing parameters,
    and signal processing parameters.
    
    Args:
        meta: SAR metadata containing acquisition parameters
        patch_dim: Dimensions of the processing patch (azimuth, range)
        use_torch: Whether to use PyTorch tensors (if available)
        
    Returns:
        Dictionary containing all processing constants
        
    Raises:
        ValueError: If required metadata fields are missing
        ImportError: If required dependencies are not available
    """
    if torch is None and use_torch:
        raise ImportError("PyTorch is required but not available")
    
    # Validate input
    required_fields = ["PRI", "Range Decimation", "Rank", "SWST",
                       "Tx Pulse Start Frequency", "Tx Ramp Rate", "Tx Pulse Length"]
    for field in required_fields:
        if field not in meta.columns:
            raise ValueError(f"Required metadata field '{field}' not found")
    
    constants = {}
    
    # Physical constants
    phys_constants = _get_physical_constants()
    constants.update(phys_constants)
    constants['pi'] = math.pi
    
    # Geometry and timing
    constants['PRI'] = float(meta["PRI"].unique()[0])
    constants['len_az_line'] = patch_dim[0]
    constants['len_range_line'] = patch_dim[1] 
    constants['rank'] = int(meta["Rank"].unique()[0])
    
    # Sampling frequencies and periods
    constants['az_sample_freq'] = 1.0 / constants['PRI']
    
    if sentinel1decoder is not None:
        rgdec = meta["Range Decimation"].unique()[0]
        constants['range_sample_freq'] = float(
            sentinel1decoder.utilities.range_dec_to_sample_rate(rgdec)
        )
    else:
        # Fallback calculation based on decimation code
        rgdec = meta["Range Decimation"].unique()[0]
        constants['range_sample_freq'] = 100092592.64  # Common S1 value
        print(f"Warning: Using fallback range sampling frequency for RGDEC={rgdec}")
    
    constants['range_sample_period'] = 1.0 / constants['range_sample_freq']
    
    # Range timing
    suppressed_data_time = SUPPRESSED_DATA_FACTOR / phys_constants['f_ref']
    constants['range_start_time'] = (float(meta["SWST"].unique()[0]) + 
                                     suppressed_data_time)
    
    # Compute derived vectors and parameters
    if use_torch:
        constants = _compute_time_vectors(constants)
        constants['tx_replica'] = _compute_tx_replica(constants, meta)
    
    return constants


def load_constants(
    patch_dim: Tuple[int, int] = (18710, 25780),
    use_torch: bool = True
) -> Dict[str, Any]:
    """
    Load default SAR processing constants.
    
    This function provides a set of default constants suitable for Sentinel-1
    SAR processing. These values are based on typical S1 parameters and can
    be used when metadata is not available.
    
    Args:
        patch_dim: Dimensions of the processing patch (azimuth, range)
        use_torch: Whether to use PyTorch tensors (if available)
        
    Returns:
        Dictionary containing default processing constants
        
    Raises:
        ImportError: If PyTorch is required but not available
    """
    if torch is None and use_torch:
        raise ImportError("PyTorch is required but not available")
    
    constants = {}
    
    # Physical constants
    constants['wavelength'] = TX_WAVELENGTH_M
    constants['c'] = SPEED_OF_LIGHT_MPS
    constants['pi'] = math.pi
    
    # Default Sentinel-1 parameters
    constants['PRI'] = 0.0005345716926237736  # Pulse Repetition Interval (s)
    constants['len_az_line'] = patch_dim[0]
    constants['len_range_line'] = patch_dim[1]
    constants['rank'] = 9  # Default rank value
    constants['range_sample_freq'] = 100092592.64  # Hz
    constants['range_start_time'] = 0.00011360148005720262  # s
    
    # Derived parameters
    constants['az_sample_freq'] = 1.0 / constants['PRI']
    constants['range_sample_period'] = 1.0 / constants['range_sample_freq']
    
    # Compute vectors if PyTorch is available
    if use_torch and torch is not None:
        # Convert key parameters to tensors for consistency
        constants['wavelength'] = torch.tensor(constants['wavelength'], dtype=torch.float32)
        constants['c'] = torch.tensor(constants['c'], dtype=torch.float32)
        constants['PRI'] = torch.tensor(constants['PRI'], dtype=torch.float32)
        
        # Sample indices and fast time vector
        sample_indices = torch.arange(0, constants['len_range_line'], dtype=torch.float32)
        constants['sample_num_along_range_line'] = sample_indices
        
        fast_time_vec = (constants['range_start_time'] + 
                         constants['range_sample_period'] * sample_indices)
        constants['fast_time_vec'] = fast_time_vec
        
        # Azimuth frequency vector 
        az_freq_start = -constants['az_sample_freq'] / 2
        az_freq_end = constants['az_sample_freq'] / 2
        az_freq_step = 1 / (constants['PRI'] * constants['len_az_line'])
        constants['f_eta'] = torch.arange(
            start=az_freq_start, 
            end=az_freq_end, 
            step=az_freq_step, 
            dtype=torch.float32
        )
    
    return constants


def validate_constants(constants: Dict[str, Any]) -> bool:
    """
    Validate that all required constants are present and reasonable.
    
    Args:
        constants: Dictionary of processing constants
        
    Returns:
        True if constants are valid, False otherwise
    """
    required_keys = [
        'wavelength', 'c', 'PRI', 'len_az_line', 'len_range_line',
        'az_sample_freq', 'range_sample_freq', 'range_sample_period'
    ]
    
    # Check all required keys are present
    for key in required_keys:
        if key not in constants:
            print(f"Warning: Missing required constant '{key}'")
            return False
    
    # Basic sanity checks
    if constants['wavelength'] <= 0 or constants['wavelength'] > 1:
        print("Warning: Wavelength value seems unreasonable")
        return False
        
    if constants['c'] < 2.9e8 or constants['c'] > 3.1e8:
        print("Warning: Speed of light value seems unreasonable")
        return False
        
    if constants['PRI'] <= 0 or constants['PRI'] > 0.01:
        print("Warning: PRI value seems unreasonable")
        return False
        
    return True


def get_processing_info(constants: Dict[str, Any]) -> Dict[str, str]:
    """
    Get human-readable information about the processing parameters.
    
    Args:
        constants: Dictionary of processing constants
        
    Returns:
        Dictionary with formatted parameter information
    """
    info = {}
    
    if 'wavelength' in constants:
        wavelength = constants['wavelength']
        if torch is not None and isinstance(wavelength, torch.Tensor):
            wavelength = wavelength.item()
        info['Wavelength'] = f"{wavelength*100:.2f} cm"
        info['Frequency'] = f"{constants['c']/wavelength/1e9:.2f} GHz"
    
    if 'PRI' in constants:
        pri = constants['PRI']
        if torch is not None and isinstance(pri, torch.Tensor):
            pri = pri.item()
        info['PRF'] = f"{1/pri:.1f} Hz"
        info['PRI'] = f"{pri*1000:.3f} ms"
    
    if 'range_sample_freq' in constants:
        info['Range Sampling'] = f"{constants['range_sample_freq']/1e6:.1f} MHz"
    
    if 'len_az_line' in constants and 'len_range_line' in constants:
        info['Patch Size'] = f"{constants['len_az_line']} x {constants['len_range_line']}"
    
    return info
