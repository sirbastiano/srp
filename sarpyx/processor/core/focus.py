import numpy as np
import argparse
try:
    import torch
except:
    print('Unable to import torch module')
import pickle
# import sentinel1decoder
import pandas as pd
from scipy.interpolate import interp1d
import math
from pathlib import Path 
import copy 
import gc
from functools import wraps

def auto_gc(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        gc.collect()
        return result
    return wrapper

# check ram usage:
import psutil
import time

from .transforms import perform_fft_custom 
from . import constants as cnst

from os import environ
environ['OMP_NUM_THREADS'] = '8'

from ..utils.io import dump

def timing_decorator(func):
    """
    A decorator to measure the execution time of a function and print the elapsed time.
    
    Parameters:
        func (callable): The function to measure.
        
    Returns:
        callable: The wrapped function with timing measurement.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Elapsed time for {func.__name__}: {elapsed_time:.4f} seconds")
        return result
    return wrapper

def printmemory():
    print(f'RAM memory usage: {psutil.virtual_memory().percent}%')
    return


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_params(device=None, slant_range_vec=None, D=None, c=None, len_range_line=None, range_sample_freq=None, wavelength=None):
    params = {key: value for key, value in locals().items()}
    return params

def range_dec_to_sample_rate(rgdec_code: int) -> float:
    """
    Convert range decimation code to sample rate.

    Args:
        rgdec_code: Range decimation code

    Returns:
        Sample rate for this range decimation code.

    """
    if rgdec_code == 0:
        return 3 * cnst.F_REF
    elif rgdec_code == 1:
        return (8/3) * cnst.F_REF
    elif rgdec_code == 3:
        return (20/9) * cnst.F_REF
    elif rgdec_code == 4:
        return (16/9) * cnst.F_REF
    elif rgdec_code == 5:
        return (3/2) * cnst.F_REF
    elif rgdec_code == 6:
        return (4/3) * cnst.F_REF
    elif rgdec_code == 7:
        return (2/3) * cnst.F_REF
    elif rgdec_code == 8:
        return (12/7) * cnst.F_REF
    elif rgdec_code == 9:
        return (5/4) * cnst.F_REF
    elif rgdec_code == 10:
        return (6/13) * cnst.F_REF
    elif rgdec_code == 11:
        return (16/11) * cnst.F_REF
    else:
        raise Exception(f"Invalid range decimation code {rgdec_code} supplied - valid codes are 0-11")


class coarseRDA:

    def __init__(self, raw_data=None, verbose=False, backend='numpy'):
        ######### Settings Private Variables
        self._backend = backend
        self._verbose = verbose
        ######### Extraction of data
        self.radar_data = raw_data['echo']
        self.ephemeris = raw_data['ephemeris']
        self.ephemeris['time_stamp'] /= 2**24
        self.metadata = raw_data['metadata']
        ######### Preliminary estimations
        self.len_range_line = self.radar_data.shape[1]
        self.len_az_line = self.radar_data.shape[0]
        if self._backend == 'torch':
            self.device = self.radar_data.device
            print('Selected device:', self.device)
        
        ######### Prompting Replica:    
        self._prompt_tx_replica()
        

    @timing_decorator
    @auto_gc
    def fft2D(self, w_pad=None, executors=12):
        # TODO: Test this function
        """
        Perform 2D FFT on a radar data array in range and azimuth dimensions.

        Args:
            raw_data (dict): Dict containing: 'echo':2D numpy array of radar data; 'metadata'; 'ephemeris';
            backend (str, optional): Backend to use for FFT. Defaults to 'numpy'.

        Returns:
            np.array: 2D numpy array of radar data after 2D FFT.
        """
        
        if self._backend == 'numpy':
            self.radar_data = np.ascontiguousarray(self.radar_data)
            # FFT each range line
            self.radar_data = np.fft.fft(self.radar_data, axis=1, n=w_pad)
            # FFT each azimuth line
            self.radar_data = np.fft.fftshift(np.fft.fft(self.radar_data, axis=0), axes=0)
        
        elif self._backend == 'custom':
            self.radar_data = perform_fft_custom(self.radar_data, num_slices=executors)
            
        elif self._backend == 'torch':
            def fft2D_in_chunks(radar_data, chunk_size):
                radar_data = torch.tensor(radar_data, dtype=torch.complex64, device='cuda')
                num_chunks = (radar_data.shape[0] + chunk_size - 1) // chunk_size
                result = []
                
                for i in range(num_chunks):
                    chunk = radar_data[i * chunk_size:(i + 1) * chunk_size, :]
                    chunk_fft = torch.fft.fft(chunk, dim=1)
                    chunk_fft = torch.fft.fftshift(torch.fft.fft(chunk_fft, dim=0), dim=0)
                    result.append(chunk_fft)
                
                return torch.cat(result, dim=0)
        
            # Convert radar_data to a PyTorch tensor and move to device
            # self.radar_data = torch.tensor(self.radar_data, dtype=torch.complex64, device=self.device)
            # FFT each range line
            if w_pad is not None:
                self.radar_data = torch.fft.fft(self.radar_data, dim=1, n=self.radar_data.shape[1]+w_pad)
            else:
                self.radar_data = torch.fft.fft(self.radar_data, dim=1)
                
            # FFT each azimuth line
            self.radar_data = torch.fft.fftshift(torch.fft.fft(self.radar_data, dim=0), dim=0)
            
        else:
            raise ValueError('Backend not supported.')
        
        if self._verbose:
            print('- FFT performed successfully!')


    @timing_decorator
    @auto_gc
    def _prompt_tx_replica(self):
        RGDEC = self.metadata["Range Decimation"].unique()[0]
        self.range_sample_freq = range_dec_to_sample_rate(RGDEC)

        # Nominal Replica Parameters
        TXPSF = self.metadata["Tx Pulse Start Frequency"].unique()[0]
        TXPRR = self.metadata["Tx Ramp Rate"].unique()[0]
        TXPL = self.metadata["Tx Pulse Length"].unique()[0]
        num_tx_vals = int(TXPL*self.range_sample_freq)
        tx_replica_time_vals = np.linspace(-TXPL/2, TXPL/2, num=num_tx_vals)
        phi1 = TXPSF + TXPRR*TXPL/2
        phi2 = TXPRR/2
        self.num_tx_vals = num_tx_vals
        self.tx_replica = np.exp(2j * np.pi * (phi1*tx_replica_time_vals + phi2*tx_replica_time_vals**2))
        self.replica_len = len(self.tx_replica)


    @timing_decorator
    @auto_gc
    def get_range_filter(self, pad_W = 0) -> np.ndarray:
        """
        Computes a range filter for radar data, specifically tailored to Sentinel-1 radar parameters.

        Notes:
            1. This function assumes that the Sentinel-1 specific constants are available through the 'sentinel1decoder.constants' module.
            2. The function makes use of the scipy `interp1d` function to interpolate spacecraft velocities.
            3. It assumes that the metadata DataFrame has specific columns like 'Range Decimation', 'PRI', 'Rank', and 'SWST'.

        """        
        tx_replica = self.tx_replica
        
        # Create range filter from replica pulse
        range_filter = np.zeros(self.len_range_line + pad_W, dtype=complex)
        index_start = np.ceil((self.len_range_line-self.num_tx_vals)/2)-1
        index_end = self.num_tx_vals+np.ceil((self.len_range_line-self.num_tx_vals)/2)-2
        range_filter[int(index_start):int(index_end+1)] = tx_replica
        
        # zero-pad the replica:
        range_filter = np.conjugate(np.fft.fft(range_filter))
        return range_filter


    @timing_decorator
    @auto_gc
    def _compute_effective_velocities(self):
        # Tx pulse parameters
        self.c = cnst.SPEED_OF_LIGHT_MPS
        self.PRI = self.metadata["PRI"].unique()[0]
        rank = self.metadata["Rank"].unique()[0]
        suppressed_data_time = 320/(8*cnst.F_REF)
        range_start_time = self.metadata["SWST"].unique()[0] + suppressed_data_time
        

        # Sample rates along azimuth and range
        range_sample_period = 1/self.range_sample_freq
        self.az_sample_freq = 1 / self.PRI
        az_sample_period = self.PRI

        # Fast time vector - defines the time axis along the fast time direction
        sample_num_along_range_line = np.arange(0, self.len_range_line, 1)
        fast_time_vec = range_start_time + (range_sample_period * sample_num_along_range_line)

        # Slant range vector - defines R0, the range of closest approach, for each range cell
        self.slant_range_vec =((rank * self.PRI) + fast_time_vec) * self.c/2
        
        # Spacecraft velocity - numerical calculation of the effective spacecraft velocity
        ecef_vels = self.ephemeris.apply(lambda x: math.sqrt(x["vx"]**2 + x["vy"]**2 +x["vz"]**2), axis=1)
        velocity_interp = interp1d(self.ephemeris["time_stamp"].unique(), ecef_vels.unique(), fill_value="extrapolate")
        x_interp = interp1d(self.ephemeris["time_stamp"].unique(), self.ephemeris["x"].unique(), fill_value="extrapolate")
        y_interp = interp1d(self.ephemeris["time_stamp"].unique(), self.ephemeris["y"].unique(), fill_value="extrapolate")
        z_interp = interp1d(self.ephemeris["time_stamp"].unique(), self.ephemeris["z"].unique(), fill_value="extrapolate")
        space_velocities = self.metadata.apply(lambda x: velocity_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float)

        x_positions = self.metadata.apply(lambda x: x_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float)
        y_positions = self.metadata.apply(lambda x: y_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float)
        z_positions = self.metadata.apply(lambda x: z_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float)

        position_array = np.transpose(np.vstack((x_positions, y_positions, z_positions)))

        a = cnst.WGS84_SEMI_MAJOR_AXIS_M
        b = cnst.WGS84_SEMI_MINOR_AXIS_M
        H = np.linalg.norm(position_array, axis=1)
        W = np.divide(space_velocities, H)
        lat = np.arctan(np.divide(position_array[:, 2], position_array[:, 0]))
        local_earth_rad = np.sqrt(
            np.divide(
                (np.square(a**2 * np.cos(lat)) + np.square(b**2 * np.sin(lat))),
                (np.square(a * np.cos(lat)) + np.square(b * np.sin(lat)))
            )
        )
        cos_beta = (np.divide(np.square(local_earth_rad) + np.square(H) - np.square(self.slant_range_vec[:, np.newaxis]) , 2 * local_earth_rad * H))
        ground_velocities = local_earth_rad * W * cos_beta

        self.effective_velocities = np.sqrt(space_velocities * ground_velocities)


    @timing_decorator
    @auto_gc
    def get_RCMC(self):
        """
        Calculate and return the RCMC filter for a given radar dataset.

        Args:
            metadata (pd.DataFrame): Pandas DataFrame containing metadata for the radar dataset.

        Returns:
            np.array: 1D numpy array representing the RCMC filter.
        """
        self._compute_effective_velocities()
        
        self.wavelength = cnst.TX_WAVELENGTH_M
        self.az_freq_vals = np.arange(-self.az_sample_freq/2, self.az_sample_freq/2, 1/(self.PRI*self.len_az_line))
        
        # Cosine of the instantanoeus squint angle
        self.D = np.sqrt(
            1 - np.divide(
                self.wavelength**2 * np.square(self.az_freq_vals),
                4 * np.square(self.effective_velocities)
            )
        ).T
        
        # Create RCMC filter
        range_freq_vals = np.linspace(-self.range_sample_freq/2, self.range_sample_freq/2, num=self.len_range_line)
        rcmc_shift = self.slant_range_vec[0] * (np.divide(1, self.D) - 1)
        rcmc_filter = np.exp(4j * np.pi * range_freq_vals * rcmc_shift / self.c)
        return rcmc_filter
    
    
    @timing_decorator
    @auto_gc
    def ifft_rg(self):
        if self._backend == 'numpy':
            self.radar_data = np.fft.ifftshift(np.fft.ifft(self.radar_data, axis=1), axes=1)
        elif self._backend == 'torch':
                self.radar_data = torch.fft.ifft(self.radar_data, dim=1)
                self.radar_data = torch.fft.ifftshift(self.radar_data, dim=1)
        else:
            raise ValueError("Unsupported backend. Choose 'numpy' or 'torch'.")
    
    
    @timing_decorator
    @auto_gc
    def ifft_az(self):
        if self._backend == 'numpy':
            self.radar_data = np.fft.ifft(self.radar_data, axis=0)
        elif self._backend == 'torch':
                self.radar_data = torch.fft.ifft(self.radar_data, dim=0)
        else:
            raise ValueError("Unsupported backend. Choose 'numpy' or 'torch'.")
    
    
    @timing_decorator
    @auto_gc 
    def get_azimuth_filter(self):
        az_filter = np.exp(4j * np.pi * self.slant_range_vec * self.D / self.wavelength)
        return az_filter
    
    
    @timing_decorator
    @auto_gc
    def data_focus(self):
        """
        Performs the focusing of the raw data.
        """
        # Init
        W_PAD = self.replica_len
        H, original_W = self.radar_data.shape
        
        # Start Processing:
        self.fft2D(w_pad=W_PAD)
        # RG compression
        self.radar_data = multiply(self.radar_data, 
                                        self.get_range_filter(pad_W=W_PAD))
        ########################
        ## Remove padding
        start_index = W_PAD // 2
        end_index = start_index + original_W
        self.radar_data = self.radar_data[:, start_index:end_index]
        ########################
        # RCMC
        self.radar_data = multiply(self.radar_data, 
                                        self.get_RCMC())
        
        # IFFT Range
        self.ifft_rg()
        
        # Az Compression
        self.radar_data = multiply(self.radar_data, 
                                        self.get_azimuth_filter())
        
        # IFFT
        self.ifft_az()
        printmemory()

    @timing_decorator
    def savefile(self, savepath):
        dump(self.radar_data, savepath)

def multiply(a, b):
    """
    Multiplies two arrays a and b using the specified backend.
    """
    return a * b

        # if not isinstance(a, np.ndarray):
        #     raise ValueError("Array 'a' must be a numpy ndarray")
        # if not isinstance(b, np.ndarray):
        #     raise ValueError("Array 'b' must be a numpy ndarray")
        # Ensure shapes are compatible for element-wise multiplication

        
        # elif backend == 'torch':
        #     if not isinstance(a, torch.Tensor):
        #         raise ValueError("Array 'a' must be a torch Tensor")
        #     if not isinstance(b, torch.Tensor):
        #         raise ValueError("Array 'b' must be a torch Tensor")
        #     # Ensure shapes are compatible for element-wise multiplication
        #     if a.shape != b.shape:
        #         raise ValueError("Shapes of arrays 'a' and 'b' must be the same")
        #     return torch.mul(a, b)
        # else:
        #     raise ValueError("Unsupported backend. Use 'numpy' or 'torch'.")




if __name__ == '__main__':
    pass
"""
    parser = argparse.ArgumentParser(description='SAR Processor')
    parser.add_argument('--data', type=str, default='radar_data.npy', help='path to the radar data')
    parser.add_argument('--meta', type=str, default='/path/to/ephemeris.pkl', help='Path to the ephemeris file')
    parser.add_argument('--ephemeris', type=str, default='radar_data.npy', help='path to the radar data')
    parser.add_argument('--output', type=str, default='outputdir', help='path to the focused radar data')
    parser.add_argument('--backend', type=str, default='numpy', help='backend used to process data')
    parser.add_argument('--num_chunks', type=int, default=15, help='Number of chunks to parse the SAR data')
    parser.add_argument('--idx_chunk', type=int, default=0, help='Index of the chunk to parse the SAR data')
    
    print('\n\n***   Starting SAR Processor   ***')
    args = parser.parse_args()
    # Load data:
    name = Path(args.data).stem
    idx = args.idx_chunk
    print(f'Processing chunk {idx+1}/{args.num_chunks}')
    printmemory()
    radar_data, meta, ephemeris = get_partition(data_path=args.data, ephem_path=args.ephemeris, meta_path=args.meta, num_chunks = args.num_chunks, idx_chunk=idx)

"""