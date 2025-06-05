import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
import warnings
from meta import Handler
import utilis as ut


# Stop GDAL printing both warnings and errors to STDERR
gdal.PushErrorHandler('CPLQuietErrorHandler')
# Make GDAL raise python exceptions for errors (warnings won't raise an exception)
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


########################################################################################################

class CombinedSublooking:
    
    def __init__(self, productPath: str, numberofLooks =3, i_image = None, q_image = None, DownSample = True, assetMetadata = None):
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
        
        self.filepath=productPath
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
            metadata_pointer = self.filepath
        elif type(assetMetadata) is dict:
            metadata_pointer = assetMetadata 
        else:
            raise ValueError( "Supplied asset metadata in unrecognized format. Pass nested dictionary or None to use asset .xml file.")
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
        self.SpectrumOneDim=np.empty(shape=(self.nRows, self.nCols), dtype = np.clongdouble)
        self.SpectrumOneDimNormDeWe = np.zeros(shape=(self.nRows, self.nCols), dtype = np.clongdouble)
        self.SpectrumOneDimDeWe = np.zeros(shape=(self.nRows, self.nCols), dtype = np.clongdouble)
            
    
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
            band_limit = band/2 * 1e-6
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
        else:
            target_average = np.sum(self.SpectrumOneDim)/len(self.SpectrumOneDim[1])
            dim_average_int = np.mean(np.abs(self.SpectrumOneDim), axis=0)
            self.SpectrumOneDimNorm = self.SpectrumOneDim/ np.outer(dim_average_int, np.zeros(self.nCols)+1.) * target_average
        
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
    
        
    def Generation(self, VERBOSE = True):
        """
        Generate the sublooks
        
        Returns:
            numpy.array: 3D array of complex numbers of size [nLooks, x, y]. 
        """
        # Extraction of Sub-bands, Weigthing
        indexLooks=np.empty((self.numberOfLooks,2), dtype='int')
        LookSpectr=np.empty((self.numberOfLooks, self.nRows, self.nCols),dtype = np.clongdouble)
        LookSpectrCentered=np.empty((self.numberOfLooks,self.nRows,self.nCols), dtype = np.clongdouble)
        
        # TODO Add range downsampling functionality
        
        if self.choice==0: # RANGE COMPUTATION
            for it in range(self.numberOfLooks):
                # Find index of nearest frequency to window min and max
                        startIndex = np.abs(self.freqVect-1e6*self.freqMin[it]).argmin()
                        endIndex = np.abs(self.freqVect-1e6*self.freqMax[it]).argmin()
                        indexLooks[it]=[startIndex, endIndex-1]

            for it in range(self.numberOfLooks):
                        for jt in range(self.nRows):
                            DeWe=self.SpectrumOneDimNormDeWe[jt, indexLooks[it,0]:indexLooks[it,1]+1]
                            LookSpectr[it, jt, indexLooks[it,0]:indexLooks[it,1]+1]=DeWe
            
            nPixLook=np.min(indexLooks[:,1]-indexLooks[:,0]+1)
            for it in range(self.numberOfLooks):
                startIndex=indexLooks[it,0]
                endIndex=startIndex+nPixLook-1
                for jt in range(self.nRows):
                        # Shifting
                        curr=LookSpectr[it,jt,startIndex:endIndex+1]
                        LookSpectrCentered[it,jt, 0:len(curr)]=curr
                        # IFFT and SL gen
                        currLong=LookSpectrCentered[it,jt,:]
                        # currImage=eng.ifft(matlab.double(currLong.tolist(), is_complex=True))
                        currImage=np.fft.ifft(currLong)
                        self.Looks[it,jt,:]=currImage
        
        else: # AZIMUTH COMPUTATION
            
        # Find frequency window indices for all looks at once
            for it in range(self.numberOfLooks):
                # Use more efficient argmin() for finding closest frequency indices
                startIndex = np.abs(self.freqVect - self.freqMin[it]).argmin()
                endIndex = np.abs(self.freqVect - self.freqMax[it]).argmin() - 1  
                indexLooks[it] = [startIndex, endIndex]
                
                # TODO: Aggiungere supporto per frequenze maggiori (FDC non centrato)
                
                # Extract and flip spectral data for each column in current look
                for jt in range(self.nCols):
                    slice_indices = slice(startIndex, endIndex + 1)
                    LookSpectr[it, slice_indices, jt] = self.SpectrumOneDimNormDeWe[slice_indices, jt][::-1]  # Flip to prepare for subsequent IFFT
                
                # if VERBOSE: 
                #     plt.imshow(np.real(LookSpectr[it, :, :]), cmap='gray')
                #     plt.title(f"Look {it+1} - Frequency Window {indexLooks[it][0]} to {indexLooks[it][1]}")
                #     plt.show()

            # Calculate minimum width across all frequency windows
            nPixLook = np.min(indexLooks[:, 1] - indexLooks[:, 0] + 1)
  
            if self.DownSample:
                self.Looks = np.empty((self.numberOfLooks, nPixLook, self.nRows),dtype = np.clongdouble)
            else:
                self.Looks = np.empty((self.numberOfLooks, self.nRows, self.nRows),dtype = np.clongdouble)
                    
            # Process each look
            for it in range(self.numberOfLooks):
                startIndex = indexLooks[it, 0]
                #window_indices = slice(startIndex, startIndex + nPixLook)
                
                # Process each column
                for jt in range(self.nCols):
                    # Extract window of spectral data and shift it to  first rows
                    data_window = LookSpectr[it, startIndex:startIndex + nPixLook+1, jt]
                    LookSpectrCentered[it, :len(data_window), jt] = data_window
                    
                    # Apply IFFT to get spatial domain representation
                    if self.DownSample:
                        self.Looks[it, :, jt] = np.fft.ifft(LookSpectrCentered[it, :nPixLook, jt])
                    else:                        
                        self.Looks[it, :, jt] = np.fft.ifft(LookSpectrCentered[it, :, jt])

                               
                # Recentre the image
                self.Looks[it] = np.roll(self.Looks[it], self.nSample // 2, axis=self.choice)

            # Verify and report success
            if self.Looks.shape[0] == self.numberOfLooks:
                print(f"{self.numberOfLooks} sublooks created successfully.")  


    def chain(self):
        self.FrequencyComputation()
        self.CalcFrequencyVectors()
        self.SpectrumComputation()
        self.SpectrumNormalization()
        self.SpectrumDeWeighting()
        self.Generation()            