from pathlib import Path
import subprocess
import lxml.etree as ET2
import pandas as pd
import os
import matplotlib.pyplot as plt
import rasterio
from shapely.geometry import Point
import numpy as np
from scipy import io
import h5py
from zipfile import ZipFile
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import XML, fromstring
# from numba import njit, jit
import shutil
import argparse
import copy
import warnings



from. meta import Handler 


# Stop printing warnings and errors
import logging
logging.getLogger('rasterio').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

#######################################|  SETUP |###################################################
# In command prompt:
# cmd = "conda create -n py39 python=3.9 scipy pandas numpy=1.20 rasterio matplotlib lxml openpyxl h5py numba jupyter -c conda-forge"
####################################################################################################

def center_frequency(array, freq_c, printout=False):
     """
     Centers a frequency vector around the specified central frequency of a sensor.

     This function shifts the elements of the input frequency array such that the specified
     central frequency (`freq_c`) is aligned with the zero frequency. The shifting is performed
     using a circular rotation.

     Args:
          array (numpy.ndarray): The input frequency vector to be centered.
          freq_c (float): The target central frequency to align the array around.
          printout (bool, optional): If True, prints the original vector, the rotated vector,
                                           and the new center frequency. Defaults to False.

     Returns:
          numpy.ndarray: The rotated frequency vector with the specified central frequency aligned.
     """
     """Center the frequency vector around the correct central frequency of the sensor"""
     # find index
     def find_nearest(array, value):
          """
          Find the index of the nearest value in an array.

          Args:
               array (numpy.ndarray): The input array to search.
               value (float): The value to find the nearest to.

          Returns:
               int: The index of the nearest value in the array.
          """
          array = np.asarray(array)
          return np.abs(array - value).argmin()

     idx = find_nearest(array, freq_c)
     zero_index = find_nearest(array, 0)
     rotated_array = np.roll(array, shift=zero_index - idx)

     if printout:
          print('Original vector:', array)
          print('Rotated vector:', rotated_array)
          print('New center:', rotated_array[len(rotated_array) // 2])

     return rotated_array


def DeHammWin(Isign,coeff):
     """
     Applies a modified Hamming window to the input signal and adjusts the imaginary part of the result.
     Parameters:
     -----------
     Isign : array-like
          The input signal to be processed. It is expected to be a sequence of complex numbers.
     coeff : float
          The coefficient for the Hamming window. Typically, this value is between 0 and 1.
     Returns:
     --------
     numpy.ndarray
          The processed signal after applying the Hamming window and adjusting the imaginary part.
     Notes:
     ------
     - The Hamming window is computed as: w[it] = coeff - (1 - coeff) * cos(2 * pi * it / nSample)
     - The function modifies the imaginary part of the input signal by negating it.
     - The input signal is divided element-wise by the computed Hamming window.
     Example:
     --------
     >>> import numpy as np
     >>> signal = np.array([complex(1, 2), complex(3, 4), complex(5, 6)])
     >>> coeff = 0.54
     >>> result = DeHammWin(signal, coeff)
     >>> print(result)
     """

     def changeImag(arr):
               arrAdj = np.array([complex(x.real, -y.imag) for x,y in zip(arr, arr)])
               return arrAdj
     
     nSample=len(Isign)

     alpha = np.empty(nSample)
     w = np.empty(nSample)

     for it in range(nSample):
          alpha[it]=2*np.pi*it/nSample
          w[it]=coeff-(1-coeff)*np.cos(alpha[it])

     return changeImag(Isign/w)


############ DISCRIMINATION PIPELINE #################################
class SubLookAnalysis:

     def __init__(self, productPath: str):
          """
          Initialize SubLookAnalysis with the given product path.
          
          Args:
               productPath (str): Path to the product file.
          """
          # MODE is the modality of the analysis: CSK, SEN, or SAO
          ##### PARAMETER SELECTION #####
          self.choice=1  # Range == 0 | Azimuth == 1
          self.numberOfLooks=3
          self.centroidSeparations=700
          self.subLookBandwidth=700

          self.filepath=productPath
          ##### DeWeighting #####
          # self.choiceDeWe=0 # Ancillary Data': 0 | Perform de-weighting using therorical weigthing function
          self.choiceDeWe=0 # Average Spectrum': 1 | Compute average spectrum 
          # TODO: modify the function to go to the tif file and open it
          with rasterio.open(subsetPath) as src:
               sub_i = src.read(1)
               sub_q = src.read(2)
          Box = sub_i + 1j * sub_q
          ##### METADATA #####
          self.Box = Box

          
          # TODO: modify handler class to point to the xml file
          meta = Handler(filepath=ParentPath)
          # TODO: modify the handler meta class to execute correctly the chain function
          meta.Chain()

          #### METADATA EXTRACTED ####
          self.PRF = meta.PRF
          self.AzimBand = meta.AzimBand
          self.ChirpBand = meta.ChirpBand
          self.RangeBand = meta.RangeBand
          self.CentralFreqRange = meta.CentralFreqRange
          self.CentralFreqAzim = meta.CentralFreqAzim
          self.WeightFunctRange = meta.WeightFunctRange
          self.WeightFunctRangeParams = meta.WeightFunctRangeParams
          self.WeightFunctAzim = meta.WeightFunctAzim
          self.WeightFunctAzimParams = meta.WeightFunctAzimParams
          self.AzimRes = meta.AzimRes
          self.AzimSpacing = meta.AzimSpacing
          self.RangeRes = meta.RangeRes
          self.RangeSpacing = meta.RangeSpacing
          # A subset of the image is created with Box dimension (ODD). Example: 30k x 30k, for computational efficiency.
          a,b = self.Box.shape

          assert ( a % 2 == 0 | b % 2 == 0), "Error: Box dimensions do not must be odd."

          ##### Central Frequence Def #####
          if self.choice == 0: # Range
               self.centralFreq=self.CentralFreqRange
          else:                # Azimuth
               self.centralFreq=self.CentralFreqAzim

          # Definition of empty matrices:
          self.nRows, self.nCols = np.shape(self.Box)
          self.SpectrumOneDim=np.empty(shape=(self.nRows, self.nCols), dtype = np.clongdouble)
          self.SpectrumOneDimNormDeWe = np.zeros(shape=(self.nRows, self.nCols), dtype = np.clongdouble)
          self.Looks=np.empty((self.numberOfLooks, self.nRows, self.nRows),dtype = np.clongdouble)

          


     def frequencyComputation(self):
          
          self.freqCentr=np.empty(self.numberOfLooks)
          self.freqMin=np.empty(self.numberOfLooks)
          self.freqMax=np.empty(self.numberOfLooks)          

          # fDC = self.centralFreq

          # CASE 1:
          if self.numberOfLooks % 2 == 0: # even number of looks

               COUNTER = 1
               FLAG = True

               for k in range(int(self.numberOfLooks)):
                    if k < 2:
                         self.freqCentr[k] = (-1)**k * self.centroidSeparations/2
                         self.freqMin[k]=self.freqCentr[k]-self.subLookBandwidth/2
                         self.freqMax[k]=self.freqCentr[k]+self.subLookBandwidth/2 
                    else:

                         if FLAG:
                              self.freqCentr[k] = (-1)**k * (0.5 + COUNTER) * self.centroidSeparations
                              FLAG = False
                         else:
                              self.freqCentr[k] = (-1)**k * (0.5 + COUNTER) * self.centroidSeparations
                              FLAG = True
                              COUNTER += 1
                         
                         self.freqMin[k]=self.freqCentr[k]-self.subLookBandwidth/2
                         self.freqMax[k]=self.freqCentr[k]+self.subLookBandwidth/2 

          else: # odd number of looks
               COUNTER = 0
               FLAG = True

               for k in range(self.numberOfLooks):

                         if FLAG:
                              self.freqCentr[k] = (-1)**k * COUNTER * self.centroidSeparations
                              COUNTER+=1
                              FLAG = False
                         
                         else: # Ora COUNTER è 1
                              self.freqCentr[k] = (-1)**k * COUNTER * self.centroidSeparations
                              FLAG = True


                         self.freqMin[k]=self.freqCentr[k]-self.subLookBandwidth/2
                         self.freqMax[k]=self.freqCentr[k]+self.subLookBandwidth/2 


          if self.centroidSeparations < self.subLookBandwidth:
               assert self.centroidSeparations<self.subLookBandwidth, "Overlapped sub-looks \n"
          else:
               print("Execution without overlapping sublooks \n")

          print(f"Available Band:[{-self.AzimBand/2}, {self.AzimBand/2}]")
          for idx, elem in enumerate(self.freqMin):
               print(f"Sub{idx+1}: [{self.freqMin[idx]}, {self.freqMax[idx]}]")

          # Error Handling sub-band outside the processed range:
          MINFREQ=np.min(self.freqMin)
          MAXFREQ=np.max(self.freqMax)

          if self.choice == 0:
               Band = self.ChirpBand
               assert MINFREQ > -Band/2*1e-6, 'sub-look spectrum outside the available bandwidth'
               assert MAXFREQ <  Band/2*1e-6, 'sub-look spectrum outside the available bandwidth'

          else:
               Band = self.AzimBand
               assert MINFREQ > -Band/2, 'sub-look spectrum outside the available bandwidth'
               assert MAXFREQ <  Band/2, 'sub-look spectrum outside the available bandwidth'
          print("Frequency computation successfully ended.")
          


     def SpectrumComputation(self, VERBOSE=False):
          ################ Spectrum Computed in Range ################
          if self.choice==0:
               for it in range(self.nRows):
                    currRow=self.Box[it,:]
                    currRowFour=np.fft.fft(currRow)
                    currRowFourShift=np.fft.fftshift(currRowFour)
                    self.SpectrumOneDim[it,:]=currRowFourShift
               # Frequency Vector
               self.nSample=self.nCols
               seq=[-self.nSample/2+x for x in range(self.nSample)]
               seq=np.array(seq)
               freqVect=self.RangeBand/self.nSample*seq
               self.freqVect = center_frequency(freqVect, self.centralFreq)
          ################ END Spectrum Computed in Range ################

          ################ Spectrum Computed in Azimuth ################
          else: 
               for it in range(self.nCols):
                    currCol=self.Box[:,it]
                    currColFour=np.fft.fft(currCol)
                    currColFourShift=np.fft.fftshift(currColFour)
                    self.SpectrumOneDim[:,it]=currColFourShift
               # Frequency Vector
               self.nSample=self.nRows
               seq=[-self.nSample/2+x for x in range(self.nSample)]
               seq=np.array(seq)
               freqVect=self.PRF/self.nSample*seq
               self.freqVect = freqVect
               self.freqVect = center_frequency(freqVect, self.centralFreq)
               if VERBOSE:
                    plt.figure(figsize=(5,5))
                    plt.plot(self.SpectrumOneDim[:,300])
                    plt.show()
          ################ END Spectrum Computed in Azimuth ################

          # Spectrum Normalization
          maxSpectr=np.max(np.abs(self.SpectrumOneDim))
          self.SpectrumOneDimNorm=1/maxSpectr*self.SpectrumOneDim
          assert self.SpectrumOneDimNorm is not None, "Error: Spectrum Computation Aborted"
          print("Spectrum Computation successfully ended.")
          if VERBOSE:
               print(seq)
               # plt.figure(figsize=(5,5))
               # plt.plot(self.SpectrumOneDimNorm[:,512])
               # plt.show()


     def SpectrumDeWeighting(self):
          if self.choiceDeWe==0:
               self.AncillaryDeWe()
          else:
               self.AverageSpectrumDeWe()


     def AverageSpectrumDeWe(self, VERBOSE=False):
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


     def AncillaryDeWe(self, VERBOSE=False):

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
          
          # Second Normalization:
          self.SpectrumOneDimNormDeWe=1/np.max(np.abs(self.SpectrumOneDimNormDeWe))*self.SpectrumOneDimNormDeWe
          print("Ancillary DeWeighting completed.")
          
          if VERBOSE:
               print(f"DeHammSign:{DeHammSign}")
               print(f"IndexGoodOrdHamm:{indexGoodOrdHamm}")
               print(f"IndexGoodHamm:{indexGoodHamm}")
               plt.figure(figsize=(5,5))
               plt.plot(self.SpectrumOneDimNormDeWe[:,128])
               plt.show()
          ############################    TUTTO CORRETTO   ##################################

    
     def Generation(self, VERBOSE=False):
          # Extraction of Sub-bands, Weigthing
          indexLooks=np.empty((self.numberOfLooks,2), dtype='int')
          LookSpectr=np.empty((self.numberOfLooks, self.nRows, self.nCols),dtype = np.clongdouble)
          # LookSpectrDeWe=np.empty((self.numberOfLooks, self.nRows, self.nCols),dtype = np.clongdouble)
          LookSpectrCentered=np.zeros((self.numberOfLooks,self.nRows,self.nCols), dtype = np.clongdouble)

          if self.choice==0: # RANGE COMPUTATION
               for it in range(self.numberOfLooks):
                         startIndex=np.where(np.abs(self.freqVect-1e6*self.freqMin[it]) == np.min(np.abs(self.freqVect-1e6*self.freqMin[it])))[0][0]
                         endIndex=np.where(np.abs(self.freqVect-1e6*self.freqMax[it]) == np.min(np.abs(self.freqVect-1e6*self.freqMax[it])))[0][0]
                         indexLooks[it,:]=np.array([startIndex, endIndex-1])

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
               for it in range(self.numberOfLooks):
                         startIndex=np.where(np.abs(self.freqVect-self.freqMin[it]) == np.min(np.abs(self.freqVect-self.freqMin[it])))[0][0]
                         endIndex=np.where(np.abs(self.freqVect-self.freqMax[it]) == np.min(np.abs(self.freqVect-self.freqMax[it])))[0][0]
                         indexLooks[it,:]=np.array([startIndex, endIndex-1])
                         
                         # print(f"startIndex:{startIndex}, endIndex:{endIndex}")
                         # print(f"FreqVector:{startIndex}")
                         # TODO: Aggiungere supporto per frequenze maggiori (FDC non centrato)
                         
                         for jt in range(self.nCols):
                              DeWe=self.SpectrumOneDimNormDeWe[indexLooks[it,0]:indexLooks[it,1]+1,jt]
                              LookSpectr[it,indexLooks[it,0]:indexLooks[it,1]+1,jt]=DeWe[::-1] #flipud
               
               nPixLook=np.min(indexLooks[:,1]-indexLooks[:,0]+1)
               for it in range(self.numberOfLooks):
                    startIndex=indexLooks[it,0]
                    endIndex=startIndex+nPixLook-1
                    for jt in range(self.nCols):
                         # Shifting
                         curr=LookSpectr[it,startIndex:endIndex+1,jt]
                         LookSpectrCentered[it,0:len(curr),jt]=curr
                         # IFFT and SL gen
                         currLong=LookSpectrCentered[it,:,jt]
                         # currImage=eng.ifft(matlab.double(currLong.tolist(), is_complex=True))
                         currImage=np.fft.ifft(currLong)
                         self.Looks[it,:,jt]=currImage
          
          ############################    TUTTO CORRETTO   ##################################
          if VERBOSE:
               plt.figure(dpi=140)
               plt.imshow(abs(self.Looks[0,:,:]))
               plt.show()

               plt.figure(dpi=140)
               plt.imshow(abs(self.Looks[1,:,:]))
               plt.show()

               plt.figure(dpi=140)
               plt.imshow(abs(self.Looks[2,:,:]))
               plt.show()
               
               print(f"SubLook Generation successfully ended.")




