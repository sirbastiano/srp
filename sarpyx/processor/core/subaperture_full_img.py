import os
import re
import glob
from typing import List, Optional, Union
import warnings
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

from .meta import Handler
from . import utilis as ut
from .dim_updater import update_dim_add_bands_from_data_dir

gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.UseExceptions()
warnings.filterwarnings("ignore")


def DeHammWin(signal,coeff):
     """
     Applies a modified Hamming window to the input signal and adjusts the imaginary part of the result.
     
     Args: 
        signal (array-like) : input signal to be processed, expected as array of complex numbers
        coeff (float): coefficient for the Hamming window, typically between 0 and 1
     
    Returns:
        numpy.ndarray : processed signal after applying the Hamming window
        
    Notes:
        - The Hamming window is computed as: w[it] = coeff - (1 - coeff) * cos(2 * pi * it / nSample)
        - The function modifies the imaginary part of the input signal by negating it.
        - The input signal is divided element-wise by the computed Hamming window.
     """

     def changeImag(arr):
               arrAdj = np.array([complex(x.real, -y.imag) for x,y in zip(arr, arr)])
               return arrAdj
     
     nSample=len(signal)

     alpha = np.empty(nSample)
     w = np.empty(nSample)

     for it in range(nSample):
          alpha[it]=2*np.pi*it/nSample
          w[it]=coeff-(1-coeff)*np.cos(alpha[it])

     return changeImag(signal/w)

def write_envi_bsq_float32(path_img, path_hdr, arr2d, band_name, byte_order=1):
    """
    Write a single-band ENVI BSQ float32 image + header.

    Parameters
    ----------
    path_img : str
        Output binary file path (.img).
    path_hdr : str
        Output ENVI header path (.hdr).
    arr2d : array-like
        2D array (lines, samples).
    band_name : str
        Band name to write into the ENVI header (must match your naming convention).
    byte_order : int
        ENVI byte order: 0 = little endian, 1 = big endian.
    """
    arr = np.asarray(arr2d, dtype=np.float32)

    # Force explicit endianness
    arr_out = arr.astype(">f4", copy=False) if byte_order == 1 else arr.astype("<f4", copy=False)

    # Write binary (BSQ, 1 band)
    arr_out.tofile(path_img)

    lines, samples = arr.shape
    hdr = f"""ENVI
description = {{Sentinel-1 SM Level-1 SLC Product - Unit: real}}
samples = {samples}
lines = {lines}
bands = 1
header offset = 0
file type = ENVI Standard
data type = 4
interleave = bsq
byte order = {byte_order}
band names = {{ {band_name} }}
data gain values = {{1.0}}
data offset values = {{0.0}}
"""
    with open(path_hdr, "w", encoding="ascii") as f:
        f.write(hdr)

class CombinedSublooking:
    
    def __init__(self, metadata_pointer_safe: str, numberofLooks =3, i_image = None, q_image = None, DownSample = True, assetMetadata = None):
        # MODE is the modality of the analysis: CSK, SEN, or SAO
        ##### PARAMETER SELECTION #####
        self.choice=1  # Range == 0 | Azimuth == 1
        self.numberOfLooks=numberofLooks
        self.CentralFreqRange = 0
        self.CentralFreqAzim = 0
        self.AzimRes = 5
        self.RangeRes = 5
        self.WeightFunctAzim = 'HAMMING'
        self.WeightFunctRange = 'HAMMING'
        self.DownSample = DownSample
        
        self.metadata_pointer_safe = metadata_pointer_safe
        self.i_img = i_image
        self.q_img = q_image
        
        ##### DeWeighting #####
        self.choiceDeWe=0  # Ancillary: 0 (using theoretical weighting) |  Average : 1 (Compute average spectrum)
        
        # Create complex image according
        if type(self.i_img) is str:  
            self.Box = ut.create_complex_image_from_file(self.i_img, self.q_img) 
        elif type(self.i_img) is np.ndarray:
            self.Box = ut.create_complex_image_from_array(self.i_img, self.q_img)
        else:
            raise ValueError("Complex image creation unsuccessful.")

        # Determine metatdata format and handling
        if assetMetadata is None:
            metadata_pointer = self.metadata_pointer_safe  
        elif isinstance(assetMetadata, dict):
            metadata_pointer = assetMetadata
        else:
            raise ValueError("Supplied asset metadata in unrecognized format.")

        meta = Handler(metadata_pointer)
        meta.chain()

        #### METADATA EXTRACTED ####
        self.PRF = meta.PRF
        self.AzimBand = meta.AzimBand
        self.ChirpBand = meta.ChirpBand
        self.RangeBand = meta.RangeBand
        self.WeightFunctRangeParams = meta.WeightFunctRangeParams
        self.WeightFunctAzimParams = meta.WeightFunctAzimParams
        self.AzimSpacing = meta.AzimSpacing
        self.RangeSpacing = meta.RangeSpacing
        
        #Calculate derivative parameters
        self.centroidSeparations=(self.AzimBand-1)/self.numberOfLooks       #No overlap
        self.subLookBandwidth=(self.AzimBand-1)/self.numberOfLooks          #No overlap
        
        ##### Central Frequency Def #####
        if self.choice == 0: # Range
            self.centralFreq=self.CentralFreqRange
        else:                # Azimuth
            self.centralFreq=self.CentralFreqAzim

        # Definition of empty matrices:
        self.nRows, self.nCols = np.shape(self.Box)
        self.SpectrumOneDim=np.empty(shape=(self.nRows, self.nCols), dtype = np.complex64) # np.clongdouble)
        self.SpectrumOneDimNormDeWe = np.zeros(shape=(self.nRows, self.nCols), dtype = np.complex64) # np.clongdouble)
        self.SpectrumOneDimDeWe = np.zeros(shape=(self.nRows, self.nCols), dtype = np.complex64) # np.clongdouble)
            
    
    def FrequencyComputation(self, VERBOSE=False):
        """
        Calculate the min, max and central frequencies of each sublook.
        
        Returns:
            List: central frequencies for each sublook
            List: minimum frequencies for each sublook
            List: maximum frequencies for each sublook
        """
        self.freqCentr = np.empty(self.numberOfLooks)
        self.freqMin = np.empty(self.numberOfLooks)
        self.freqMax = np.empty(self.numberOfLooks)
        
        # Calculate central frequencies based on number of looks (odd will be zero-centered)
        if self.numberOfLooks % 2 == 0:  
            for k in range(self.numberOfLooks):
                if k < 2:
                    self.freqCentr[k] = (-1)**k * self.centroidSeparations/2
                else:
                    # Simplified pattern for higher indices
                    multiple = 0.5 + (k // 2)
                    self.freqCentr[k] = (-1)**k * multiple * self.centroidSeparations
        else: 
            for k in range(self.numberOfLooks):
                multiple = (k+1) // 2
                self.freqCentr[k] = (-1)**k * multiple * self.centroidSeparations
        
        # Calculate min/max frequencies for all sublooks
        for k in range(self.numberOfLooks):
            self.freqMin[k] = round((self.freqCentr[k] - self.subLookBandwidth/2), 4)
            self.freqMax[k] = round((self.freqCentr[k] + self.subLookBandwidth/2), 4)
        
        if VERBOSE:
            # Check for overlapping sublooks
            if self.centroidSeparations < self.subLookBandwidth:
                print("Sub-looking with spectral overlap")
            else:
                print("Execution without overlapping sublooks \n")
            
            # Display frequency bands
            band_range = f"Available spectral range:[{-self.AzimBand/2}, {self.AzimBand/2}] \n"
            print(band_range)
            
            for idx in range(self.numberOfLooks):
                print(f"Sub{idx+1}: [{self.freqMin[idx]}, {self.freqMax[idx]}]")
            
        # Validate frequency bands are within available bandwidth
        min_freq = np.min(self.freqMin)
        max_freq = np.max(self.freqMax)
        
        if self.choice == 0:
            band = self.ChirpBand
            band_limit = band/2 * 1e-6 # el factor de 1e-6 se utiliza para escalar la frecuencia y manejar valores uniformes.
        else:
            band = self.AzimBand
            band_limit = band/2
        
        if min_freq < -band_limit or max_freq > band_limit:
            raise ValueError("Sub-look spectrum outside the available bandwidth")
        
        if VERBOSE:   
            print("\n Frequency computation successfully ended.")
     
        
    def CalcFrequencyVectors(self, VERBOSE=False):
        """
        Define a zero-centred frequency vector from the number of sampled points along the axis of analysis and the specified bandwidth or PRF.
        
        """

        if self.choice ==0: 
            self.nSample=self.nCols       # Number of data points along the chosen axis
            seq = np.arange(-self.nSample/2, self.nSample/2) 
            self.freqVect=self.RangeBand/self.nSample*seq
            
        else:
            self.nSample=self.nRows
            seq = np.arange(-self.nSample/2, self.nSample/2) 
            self.freqVect=self.PRF/self.nSample*seq
        
        if VERBOSE:
            print(f"Frequency vector: {self.freqVect[:3]} ... {self.freqVect[-3:]}") 
    
        
    def SpectrumComputation(self, VERBOSE=False):
        """"
        Calculate frequency spectrum of the image by applying a Fast Fourier Transform in one dimension (range or azimuth).
        
        Returns:
        Spectrum array of complex numbers.
        """
        ################ Spectrum Computed in Range ################
        if self.choice==0:
            self.SpectrumOneDim = np.fft.fftshift(np.fft.fft(self.Box, axis=1))
            """ Array of complex numbers, zero-shifted"""
        else: 
            self.SpectrumOneDim = np.fft.fftshift(np.fft.fft(self.Box, axis=0))
            """ Array of complex numbers, zero-shifted"""
        # if self.choice==0:
        #     self.SpectrumOneDim = np.fft.fft(self.Box, axis=1)
        #     """ Array of complex numbers, zero-shifted"""
        # else: 
        #     self.SpectrumOneDim = np.fft.fft(self.Box, axis=0)
        #     """ Array of complex numbers, zero-shifted"""
        
        if VERBOSE:
            plt.figure(figsize=(5,5))
            plt.plot(self.SpectrumOneDim[:,300])
            if self.choice == 0:
                plt.title("Spectrum Computed in Range")
            else:
                plt.title("Spectrum Computed in Azimuth")
            plt.show()
            
    def SpectrumNormalization(self, VERBOSE=False):
        """
        Address singal attenuation by normalizing frequency spectrum to the average value by row or column and scale by average intensity of the image.
        
        Returns:
        SpectrumOneDimNorm: Normalized spectrum.
        """
        assert self.SpectrumOneDim is not None, "Error: Spectrum must be computed before normalizing"
        if self.choice==0: # RANGE
            target_average = np.sum(self.SpectrumOneDim)/len(self.SpectrumOneDim[0])
            dim_average_int = np.mean(np.abs(self.SpectrumOneDim), axis=1)
            self.SpectrumOneDimNorm = self.SpectrumOneDim/ np.outer(dim_average_int, np.zeros(self.nRows)+1.) * target_average
            
        else: # AZIMUTH
            # target_average = np.sum(self.SpectrumOneDim)/len(self.SpectrumOneDim[1])
            # dim_average_int = np.mean(np.abs(self.SpectrumOneDim), axis=1) # In original code: axis=0 but when processing non-square inputs it fails...
            # self.SpectrumOneDimNorm = self.SpectrumOneDim/ np.outer(dim_average_int, np.zeros(self.nCols)+1.) * target_average
            
            target_average = np.mean(np.abs(self.SpectrumOneDim))
            dim_average_int = np.mean(np.abs(self.SpectrumOneDim), axis=1)  # mean magnitude profile
            dim_average_int[dim_average_int == 0] = 1e-6  
            
            # Normalización más suave, con balance de escala global
            self.SpectrumOneDimNorm = (self.SpectrumOneDim / dim_average_int[:, None]) * np.median(dim_average_int)
                        
        if VERBOSE:
            plt.subplot(1,2,1)
            plt.plot(self.SpectrumOneDim[:,300])
            plt.title("Raw frequency spectrum")
            plt.subplot(1,2,2)
            plt.plot(self.SpectrumOneDimNorm[:,300])
            plt.title("Normalized frequency Spectrum")
            plt.show() 

                 
    def SpectrumDeWeighting(self):
        """
        Call the appropiate deweighting function
        """
        if self.choiceDeWe==0:
            self.AncillaryDeWe()
        else:
            self.AverageDeWe()

    def AverageDeWe(self, VERBOSE=False):
        # compute average spectrum 
        if self.choice==0: # RANGE.
            AverSpectrNorm=np.sum(np.abs(self.SpectrumOneDimNorm))
            for jt in range(self.nRows):
                curr=self.SpectrumOneDimNorm[jt,:]
                deWe=curr/AverSpectrNorm
                self.SpectrumOneDimNormDeWe[jt,:]=deWe
        else:              # AZIMUTH.
            AverSpectrNorm=np.sum(np.abs(self.SpectrumOneDimNorm))
            for jt in range(self.nCols):
                curr=self.SpectrumOneDimNorm[:,jt]
                deWe=curr/AverSpectrNorm
                self.SpectrumOneDimNormDeWe[:,jt]=deWe
        # Second normalization
        self.SpectrumOneDimNormDeWe=1/np.max(self.SpectrumOneDimNormDeWe)*self.SpectrumOneDimNormDeWe
        
        if VERBOSE:
            plt.subplot(1,2,1)
            plt.plot(self.SpectrumOneDimNorm[:,300])
            plt.title("Normalized spectrum")
            plt.subplot(1,2,2)
            plt.plot(self.SpectrumOneDimNormDeWe[:,300])
            plt.title("Average Deweighted Spectrum")
            plt.show() 


    def AncillaryDeWe(self, VERBOSE=False):
        """
        Reverse the effect of a previously applied Hamming window.
        """

        if self.choice==0:
            bHamm=self.RangeBand
        else:
            bHamm=self.AzimBand
        
        indexGoodHamm=np.where(np.abs(self.freqVect)<=bHamm/2)[0]
        nSampleGoodHamm=len(indexGoodHamm)
        if nSampleGoodHamm % 2 !=0:
            nSampleGoodHamm-=1
            indexGoodHamm=indexGoodHamm[:-1]

        indexCentroid = np.where(np.abs(self.freqVect-self.centralFreq) == np.min(np.abs(self.freqVect-self.centralFreq)))
        indexCentroid=indexCentroid[0][0]+1
        indexGoodOrdHamm = np.arange(indexCentroid-nSampleGoodHamm/2, indexCentroid+nSampleGoodHamm/2,1).astype('int')-1

        for jt in range(len(indexGoodOrdHamm)):
            if indexGoodOrdHamm[jt]>self.nSample-2:
                indexGoodOrdHamm[jt]-=self.nSample-1
            if indexGoodOrdHamm[jt]<=0:
                indexGoodOrdHamm[jt]+=self.nSample-1
            
        if self.choice==0:
            DeHammEnd=self.nRows
        else:
            DeHammEnd=self.nCols

        for kt in range(DeHammEnd):
            if self.choice==0:
                curr=self.SpectrumOneDimNorm[kt,:]
            else:
                curr=self.SpectrumOneDimNorm[:,kt]
            
            currApp=np.array([curr[x] for x in indexGoodOrdHamm])
            currApp=np.roll(currApp,int(self.nSample/2-indexCentroid))
            
            if self.choice==0:
                DeHammSign=DeHammWin(currApp, self.WeightFunctRangeParams)
            else:
                DeHammSign=DeHammWin(currApp, self.WeightFunctAzimParams)

            DeHammSignOrd = np.roll(DeHammSign, -int(self.nSample/2-indexCentroid))

            if self.choice==0:
                self.SpectrumOneDimNormDeWe[kt,:]=DeHammSignOrd
            else:
                self.SpectrumOneDimNormDeWe[indexGoodHamm,kt]=DeHammSignOrd
        
        if VERBOSE:
            plt.subplot(1,2,1)
            plt.plot(self.SpectrumOneDimNorm[:,300])
            plt.title("Normalized spectrum")
            plt.subplot(1,2,2)
            plt.plot(self.SpectrumOneDimNormDeWe[:,300])
            plt.title("Ancillary Deweighted Spectrum")
            plt.show()    
    
    def Generation(self, VERBOSE=False):
        """
        Generate the sublooks
        
        Returns:
            numpy.array: 3D array of complex numbers of size [nLooks, x, y]. 
        """
        indexLooks = np.empty((self.numberOfLooks, 2), dtype=int)
        LookSpectr = np.empty((self.numberOfLooks, self.nRows, self.nCols), dtype=np.complex64)
        LookSpectrCentered = np.empty((self.numberOfLooks, self.nRows, self.nCols), dtype=np.complex64)
        
        for it in range(self.numberOfLooks):
            startIndex = np.abs(self.freqVect - self.freqMin[it]).argmin()
            endIndex = np.abs(self.freqVect - self.freqMax[it]).argmin() - 1
            indexLooks[it] = [startIndex, endIndex]
            for jt in range(self.nCols):
                slice_indices = slice(startIndex, endIndex + 1)
                LookSpectr[it, slice_indices, jt] = self.SpectrumOneDimNormDeWe[slice_indices, jt][::-1]
                
        nPixLook = np.min(indexLooks[:, 1] - indexLooks[:, 0] + 1)
        self.Looks = np.empty((self.numberOfLooks, self.nRows, self.nCols),dtype = np.complex64)
        
        for it in range(self.numberOfLooks):
            startIndex = indexLooks[it, 0]        
            # Process each column
            for jt in range(self.nCols):
                # Extract window of spectral data and shift it to first rows
                data_window = LookSpectr[it, startIndex:startIndex + nPixLook+1, jt]
                LookSpectrCentered[it, :len(data_window), jt] = data_window
                # Apply IFFT to get spatial domain representation
                self.Looks[it, :, jt] = np.fft.ifft(LookSpectrCentered[it, :, jt])
                
            # Recentre the image
            self.Looks[it] = np.roll(self.Looks[it], self.Looks[it].shape[self.choice]//2 , axis=self.choice)
            
        # Verify and report success
        if self.Looks.shape[0] == self.numberOfLooks:
            print(f"{self.numberOfLooks} sublooks created successfully.")  

    def save_sublooks_envi(self, out_dir, pol, prefix="", byte_order=1):
        """
        Save i/q of each sublook as ENVI BSQ float32, forcing endianness.
    
        out_dir: output folder (typically the DIMAP .data directory)
        pol: polarization label ('VV', 'VH', etc.)
        prefix: optional output prefix (e.g. "L2_" or "L3_")
        """
        os.makedirs(out_dir, exist_ok=True)
    
        # Normalize prefix: allow "" or "L2_" etc.
        # If user passes "L2" without underscore, fix it.
        if prefix and not prefix.endswith("_"):
            prefix = prefix + "_"
    
        for idx_slook in range(self.numberOfLooks):
            sa = idx_slook + 1
            z = self.Looks[idx_slook]  # (lines, samples) complex
    
            # Ensure float32
            i = np.asarray(z.real, dtype=np.float32)
            q = np.asarray(z.imag, dtype=np.float32)
    
            # Filenames
            img_i = os.path.join(out_dir, f"{prefix}i_{pol}_SA{sa}.img")
            hdr_i = os.path.join(out_dir, f"{prefix}i_{pol}_SA{sa}.hdr")
            img_q = os.path.join(out_dir, f"{prefix}q_{pol}_SA{sa}.img")
            hdr_q = os.path.join(out_dir, f"{prefix}q_{pol}_SA{sa}.hdr")
    
            # ENVI band names MUST match the naming convention used in the DIM updater regex
            band_name_i = f"{prefix}i_{pol}_SA{sa}" if prefix else f"i_{pol}_SA{sa}"
            band_name_q = f"{prefix}q_{pol}_SA{sa}" if prefix else f"q_{pol}_SA{sa}"
    
            write_envi_bsq_float32(img_i, hdr_i, i, band_name=band_name_i, byte_order=byte_order)
            write_envi_bsq_float32(img_q, hdr_q, q, band_name=band_name_q, byte_order=byte_order)
            
    def chain(self, VERBOSE=False):
        self.FrequencyComputation(VERBOSE=VERBOSE)
        self.CalcFrequencyVectors(VERBOSE=VERBOSE)
        self.SpectrumComputation(VERBOSE=VERBOSE)
        self.SpectrumNormalization(VERBOSE=VERBOSE)
        self.SpectrumDeWeighting()
        self.Generation(VERBOSE=VERBOSE)
        return self.Looks

    def run_and_save(self, out_dir, pol, prefix="", byte_order=1, VERBOSE=False):
        self.chain(VERBOSE=VERBOSE)
        self.save_sublooks_envi(
            out_dir=out_dir,
            pol=pol,
            prefix=prefix,
            byte_order=byte_order,
        )
        return self.Looks

def find_pols_in_dim_data(data_dir: str):
    """
    Search for base polarization pairs i_<POL>.img and q_<POL>.img in the product .data folder.

    Notes
    -----
    - Ignores previously generated subaperture outputs that contain "_SA".
    - Returns a sorted list of detected polarizations (e.g. ['VV', 'VH']).
    """

    # Match: i_VV.img, i_VH.img, etc. (sin _SA)
    pat = re.compile(r"^i_([A-Z]{2})\.img$", re.IGNORECASE)

    pols = []
    for fp in glob.glob(os.path.join(data_dir, "i_*.img")):
        base = os.path.basename(fp)
        if "_SA" in base.upper():
            continue
        m = pat.match(base)
        if not m:
            continue
        pol = m.group(1).upper()

        q_fp = os.path.join(data_dir, f"q_{pol}.img")
        if os.path.exists(q_fp):
            pols.append(pol)

    pols = sorted(set(pols))
    return pols

def do_subaps(
    dim_path: str,
    safe_path: str,
    numberofLooks: int = 3,
    n_decompositions: Optional[Union[int, List[int]]] = None,
    DownSample: bool = True,
    byte_order: int = 1,
    prefix: str = "",
    VERBOSE: bool = False,
):
    """
    Orchestrator:
    - Takes a DIMAP product (.dim) to locate the .data folder (i/q bands)
    - Takes the original .SAFE product to extract metadata (PRF, bandwidths, window coeffs, etc.)
    - Detects available base polarizations in .data (i_<POL>.img + q_<POL>.img)
    - Generates subapertures for each pol and writes them into the same .data folder

    New parameters:
    - n_decompositions:
        * None -> uses numberofLooks (backward compatible)
        * int  -> runs a single decomposition with that N
        * list -> runs all decompositions in the list (e.g. [2,3,5,7])

    Output naming note:
    - If multiple decompositions are requested, this function automatically adds
      a prefix like "L{N}_" to avoid overwriting SA1/SA2/... outputs.
    """

    # -------------------------
    # Validate input paths
    # -------------------------
    base, ext = os.path.splitext(dim_path)
    if ext.lower() != ".dim":
        raise ValueError(f"Expected a .dim file, got: {dim_path}")

    data_dir = base + ".data"
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Expected .data directory not found: {data_dir}")

    if not os.path.isdir(safe_path):
        raise FileNotFoundError(f"SAFE directory not found: {safe_path}")

    # -------------------------
    # Normalize decompositions list
    # -------------------------
    if n_decompositions is None:
        decomps = [int(numberofLooks)]
    else:
        if isinstance(n_decompositions, int):
            decomps = [int(n_decompositions)]
        else:
            decomps = [int(x) for x in n_decompositions]

        # Remove duplicates and sort
        decomps = sorted(set(decomps))

    if not decomps:
        raise ValueError("n_decompositions ended up empty.")
    if any(n < 2 for n in decomps):
        raise ValueError(f"Invalid n_decompositions: {decomps}. Use values >= 2.")

    # -------------------------
    # Detect base polarizations from i_<POL>.img / q_<POL>.img
    # (ignore any previously generated *_SA* files)
    # -------------------------
    pat = re.compile(r"^i_([A-Z]{2})\.img$", re.IGNORECASE)
    pols = []
    for fp in glob.glob(os.path.join(data_dir, "i_*.img")):
        fname = os.path.basename(fp)
        if "_SA" in fname.upper():
            continue

        m = pat.match(fname)
        if not m:
            continue

        pol = m.group(1).upper()
        q_fp = os.path.join(data_dir, f"q_{pol}.img")
        if os.path.exists(q_fp):
            pols.append(pol)

    pols = sorted(set(pols))

    if not pols:
        raise RuntimeError(f"No i_<POL>.img / q_<POL>.img pairs found in: {data_dir}")

    if VERBOSE:
        print(f"Base polarizations detected in {data_dir}: {pols}")
        print(f"Metadata source SAFE: {safe_path}")
        print(f"Decompositions to run: {decomps}")

    # -------------------------
    # Run one or multiple decompositions
    # -------------------------
    for nlooks in decomps:
        # If multiple decompositions, add an N-based prefix to avoid overwriting files
        prefix_n = f"{prefix}L{nlooks}_" if len(decomps) > 1 else prefix

        if VERBOSE:
            print(f"\n=== Running decomposition N={nlooks} (prefix='{prefix_n}') ===")

        for pol in pols:
            i_fp = os.path.join(data_dir, f"i_{pol}.img")
            q_fp = os.path.join(data_dir, f"q_{pol}.img")

            if not (os.path.exists(i_fp) and os.path.exists(q_fp)):
                if VERBOSE:
                    print(f"Skipping POL={pol} (missing i/q): i={i_fp}, q={q_fp}")
                continue

            if VERBOSE:
                print(f"\nProcessing subapertures for POL={pol} (N={nlooks})")
                print(f"  i: {i_fp}")
                print(f"  q: {q_fp}")

            sub = CombinedSublooking(
                metadata_pointer_safe=safe_path,  # SAFE used for metadata
                numberofLooks=nlooks,
                i_image=i_fp,
                q_image=q_fp,
                DownSample=DownSample,
                assetMetadata=None,
            )

            sub.run_and_save(
                out_dir=data_dir,
                pol=pol,
                prefix=prefix_n,
                byte_order=byte_order,
                VERBOSE=VERBOSE,
            )

    # -------------------------
    # Update the .dim ONCE at the end
    # (at this point all new bands exist in .data)
    # -------------------------
    try:
        dim_updated = update_dim_add_bands_from_data_dir(
            dim_in=dim_path,
            dim_out=None,  # writes "<original>_subaps.dim"
            verbose=VERBOSE,
        )
        if VERBOSE:
            print(f"\nDIM updated: {dim_updated}")
    except Exception as e:
        # Do not fail the pipeline if the DIM update fails
        if VERBOSE:
            print(f"\nWARNING: could not auto-update the .dim file: {e}")

    # return pols

