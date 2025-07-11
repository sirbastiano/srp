from zipfile import ZipFile
import xml.etree.ElementTree as ET
from ..utilis import iterNodes

class Handler:
    """"
    Handler class for processing metadata from a zip file containing product information.

    Attributes:
        filepath (str): Path to the zip file containing the product data.
        PRF (float): Pulse repetition frequency extracted from the metadata.
        AzimBand (float): Total azimuth bandwidth extracted from the metadata.
        RangeBand (float): Total range bandwidth extracted from the metadata.
        ChirpBand (float): Chirp bandwidth calculated from pulse length and ramp rate.
        AzimRes (int): Azimuth resolution (default is 5).
        RangeRes (int): Range resolution (default is 20).
        RangeSpacing (float): Range pixel spacing extracted from the metadata.
        AzimSpacing (float): Azimuth pixel spacing extracted from the metadata.
        WeightFunctRangeParams (float): Range window coefficient for weighting function.
        WeightFunctRange (str): Type of weighting function used for range processing (default is 'HAMMING').
        WeightFunctAzimParams (float): Azimuth window coefficient for weighting function.
        WeightFunctAzim (str): Type of weighting function used for azimuth processing (default is 'HAMMING').
        CentralFreqRange (int): Central frequency for range processing (default is 0).
        CentralFreqAzim (int): Central frequency for azimuth processing (default is 0).

    Methods:
        __init__(filepath):
            Initializes the Handler object with the path to the zip file.

        Chain():
            Processes the metadata from the zip file, extracting relevant parameters
            such as PRF, bandwidths, resolutions, and spacing. The method reads the
            XML annotation file within the zip archive and parses the required data.
    """
    def __init__(self, filepath):
        self.filepath = filepath

    def chain(self):
        """
        Main method to process metadata from the zip file.
        """
        try:
            archive = self._open_archive()
            annotations = self._get_annotations(archive)
            xml = self._read_annotation(archive, annotations)
            root = ET.fromstring(xml)
            self._extract_metadata(root)
        except Exception as e:
            raise RuntimeError(f"Error processing metadata: {e}")

    def _open_archive(self):
        """
        Opens the zip file and returns the archive object.
        """
        try:
            return ZipFile(self.filepath, mode='r')
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.filepath}")
        except Exception as e:
            raise RuntimeError(f"Failed to open zip file: {e}")

    def _get_annotations(self, archive):
        """
        Retrieves the list of annotation files from the archive.
        """
        namelist = archive.namelist()
        annotations = [
            x for x in namelist
            if ("annotation" in x and ".xml" in x and "calibration" not in x and "rfi" not in x)
        ]
        if not annotations:
            raise ValueError("No valid annotation files found in the archive.")
        return annotations

    def _read_annotation(self, archive, annotations):
        """
        Reads the first annotation file from the archive.
        """
        try:
            return archive.read(annotations[0])
        except Exception as e:
            raise RuntimeError(f"Failed to read annotation file: {e}")

    def _extract_metadata(self, root):
        """
        Extracts metadata from the XML root and assigns it to class attributes.
        """
        try:
            ValDict = {}
            Ancillary = iterNodes(root, Val_Dict=ValDict)

            self.PRF = float(Ancillary.get('prf', 0))
            self.AzimBand = float(root.find('.//imageAnnotation//processingInformation//swathProcParams//azimuthProcessing//totalBandwidth').text or 0)
            self.RangeBand = float(root.find('.//imageAnnotation//processingInformation//swathProcParams//rangeProcessing//totalBandwidth').text or 0)
            self.ChirpBand = float(Ancillary.get('txPulseLength', 0)) * float(Ancillary.get('txPulseRampRate', 0))
            self.AzimRes = 5
            self.RangeRes = 20
            self.RangeSpacing = float(Ancillary.get('rangePixelSpacing', 0))
            self.AzimSpacing = float(Ancillary.get('azimuthPixelSpacing', 0))
            self.WeightFunctRangeParams = float(root.find('.//imageAnnotation//processingInformation//swathProcParams//rangeProcessing//windowCoefficient').text or 0)
            self.WeightFunctRange = 'HAMMING'
            self.WeightFunctAzimParams = float(root.find('.//imageAnnotation//processingInformation//swathProcParams//azimuthProcessing//windowCoefficient').text or 0)
            self.WeightFunctAzim = 'HAMMING'
            self.CentralFreqRange = 0
            self.CentralFreqAzim = 0
        except AttributeError as e:
            raise ValueError(f"Missing or invalid XML elements: {e}")
        except Exception as e:
            raise RuntimeError(f"Error extracting metadata: {e}")
