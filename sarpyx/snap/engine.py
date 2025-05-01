import argparse
import copy
import os
import shutil
import subprocess
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import h5py
import lxml.etree as ET2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit, njit
from osgeo import gdal
from scipy import io
from shapely.geometry import Point

# Import utility functions
from sarpyx.utils.io import delProd, mode_identifier

# Stop GDAL printing both warnings and errors to STDERR
gdal.PushErrorHandler('CPLQuietErrorHandler')
# Make GDAL raise python exceptions for errors (warnings won't raise an exception)
gdal.UseExceptions()
warnings.filterwarnings("ignore")


class GPT:
    """ A wrapper class for executing SNAP Graph Processing Tool (GPT) commands. """

    # Define constants for product types
    SAO = 'SAO'
    CSK = 'CSK'
    SEN = 'SEN'

    # Define default GPT paths for different OS
    GPT_PATHS = {
        'Ubuntu': '/home/<username>/ESA-STEP/snap/bin/gpt',
        'MacOS': '/Applications/snap/bin/gpt',
        'Windows': 'gpt.exe'  # Assuming gpt.exe is in PATH for Windows
    }
    DEFAULT_PARALLELISM = {
        'Ubuntu': 24,
        'MacOS': 8,
        'Windows': 8  # Default parallelism for Windows or unspecified OS
    }

    def __init__(self, product: str | Path, outdir: str | Path, format: str = 'BEAM-DIMAP', mode: str | None = None):
        self.prod_path = Path(product)
        self.name = self.prod_path.stem
        self.format = format
        self.outdir = Path(outdir)
        self.mode = mode
        self.gpt_executable = self._get_gpt_executable()
        self.parallelism = self._get_parallelism()

        self.current_cmd = []  # Store command as a list of arguments

    def _get_gpt_executable(self) -> str:
        """Determines the correct GPT executable path based on the mode."""
        if self.mode and self.mode in self.GPT_PATHS:
            return self.GPT_PATHS[self.mode]
        # Attempt to guess based on os or default to 'gpt'
        # This part might need refinement based on actual OS detection if needed
        if os.name == 'posix':  # Covers Linux, MacOS
            if Path(self.GPT_PATHS['MacOS']).exists():
                return self.GPT_PATHS['MacOS']
            if Path(self.GPT_PATHS['Ubuntu']).exists():
                return self.GPT_PATHS['Ubuntu']  # Check Ubuntu path as well
            return 'gpt'  # Default for other posix
        elif os.name == 'nt':  # Windows
            return self.GPT_PATHS['Windows']
        else:
            return 'gpt'  # Default if OS is unknown

    def _get_parallelism(self) -> int:
        """Determines the parallelism level."""
        if self.mode and self.mode in self.DEFAULT_PARALLELISM:
            return self.DEFAULT_PARALLELISM[self.mode]
        # Guess based on OS or default
        if os.name == 'posix':
            if Path(self.GPT_PATHS['MacOS']).exists():
                return self.DEFAULT_PARALLELISM['MacOS']
            return self.DEFAULT_PARALLELISM.get('Ubuntu', 6)  # Default to Ubuntu or 6
        elif os.name == 'nt':
            return self.DEFAULT_PARALLELISM['Windows']
        else:
            return 6  # Default

    def _reset_command(self):
        """ Resets the command list for a new GPT operation. """
        self.current_cmd = [
            self.gpt_executable,
            f'-q {self.parallelism}',  # Number of threads
            '-x',  # Disable JVM exit after execution
            '-e',  # Enable SNAP extension points
            f'-Ssource={self.prod_path.as_posix()}'  # Source product
        ]

    def _build_output_path(self, suffix: str) -> Path:
        """Builds the output path for a processing step."""
        base_name = self.outdir / f"{self.name}_{suffix}"
        if self.format == 'GEOTIFF':
            return base_name.with_suffix('.tif')  # Use .tif for GeoTIFF
        else:  # Default to BEAM-DIMAP
            return base_name.with_suffix('.dim')

    def _execute_command(self) -> bool:
        """ Executes the currently built GPT command. """
        cmd_str = ' '.join(self.current_cmd)
        print(f"Executing: {cmd_str}")
        try:
            # Using shell=True because GPT commands can be complex strings
            process = subprocess.run(cmd_str, shell=True, check=True, capture_output=True, text=True, universal_newlines=True)
            print("GPT Output:\n", process.stdout)
            if process.stderr:
                print("GPT Errors/Warnings:\n", process.stderr)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error executing GPT command: {cmd_str}")
            print(f"Return code: {e.returncode}")
            print(f"Stderr: {e.stderr}")
            return False
        except FileNotFoundError:
            print(f"Error: GPT command '{self.gpt_executable}' not found. Ensure SNAP is installed and configured correctly.")
            return False

    def _call(self, suffix: str) -> str | None:
        """ Finalizes and executes the GPT command. """
        output_path = self._build_output_path(suffix)
        self.current_cmd.extend([
            f'-t {output_path.as_posix()}',
            f'-f {self.format}'
        ])

        if self._execute_command():
            self.prod_path = output_path  # Update product path for chaining
            return output_path.as_posix()
        else:
            return None  # Indicate failure

    def ImportVector(self, vector_data: str | Path):
        """ Imports vector data into the product. """
        self._reset_command()
        self.current_cmd.append(f'Import-Vector -PseparateShapes=false -PvectorFile={Path(vector_data).as_posix()}')
        return self._call(suffix='SHP')

    def LandMask(self, shoreline_extension: int = 300, geometry_name: str = "Buff_750", use_srtm: bool = True, invert_geometry: bool = True, land_mask: bool = False):
        """
        Applies Land-Sea Masking using a predefined XML graph structure.
        Allows customization of key parameters.
        """
        self._reset_command()  # Reset command for XML execution
        suffix = 'LM'
        output_path = self._build_output_path(suffix)
        xml_path = self.outdir / f"{self.name}_landmask_graph.xml"  # Unique XML per run

        # Determine product type if not already set (e.g., during init)
        if not hasattr(self, 'prod_type'):
            try:
                self.prod_type = mode_identifier(self.prod_path.name)
                print(f"Inferred product type: {self.prod_type}")
            except Exception as e:
                print(f"Warning: Could not automatically determine product type: {e}. Defaulting source band.")
                self.prod_type = None  # Indicate unknown type

        # Determine source band based on product type
        if self.prod_type == self.CSK:
            source_band = 'Intensity_null'  # Adjust if specific CSK polarization is known
        elif self.prod_type == self.SEN:
            # Defaulting to VH for Sentinel, adjust if VV or other is needed
            source_band = 'Intensity_VH'
        else:  # SAO or unknown
            # Defaulting, adjust as needed for SAO or if type is unknown
            print(f"Warning: Product type is '{self.prod_type if self.prod_type else 'Unknown'}'. Using default source band 'Intensity_VH'.")
            source_band = 'Intensity_VH'

        # XML Graph Template
        graph_xml = f"""<graph id="Graph">
                  <version>1.0</version>
                  <node id="Read">
                  <operator>Read</operator>
                  <sources/>
                  <parameters class="com.bc.ceres.binding.dom.XppDomElement">
                       <file>{self.prod_path.as_posix()}</file>
                  </parameters>
                  </node>
                  <node id="Land-Sea-Mask">
                  <operator>Land-Sea-Mask</operator>
                  <sources>
                       <sourceProduct refid="Read"/>
                  </sources>
                  <parameters class="com.bc.ceres.binding.dom.XppDomElement">
                       <sourceBands>{source_band}</sourceBands>
                       <landMask>{str(land_mask).lower()}</landMask>
                       <useSRTM>{str(use_srtm).lower()}</useSRTM>
                       <geometry>{geometry_name}</geometry>
                       <invertGeometry>{str(invert_geometry).lower()}</invertGeometry>
                       <shorelineExtension>{shoreline_extension}</shorelineExtension>
                  </parameters>
                  </node>
                  <node id="Write">
                  <operator>Write</operator>
                  <sources>
                       <sourceProduct refid="Land-Sea-Mask"/>
                  </sources>
                  <parameters class="com.bc.ceres.binding.dom.XppDomElement">
                       <file>{output_path.as_posix()}</file>
                       <formatName>{self.format}</formatName>
                  </parameters>
                  </node>
                  </graph>
                  """

        try:
            # Write the generated XML to a file
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(graph_xml)

            # Update command to execute the XML graph
            self.current_cmd = [self.gpt_executable, xml_path.as_posix()]  # Overwrite command for graph execution

            if self._execute_command():
                self.prod_path = output_path  # Update product path
                os.remove(xml_path)  # Clean up the temporary XML file
                return output_path.as_posix()
            else:
                return None

        except Exception as e:
            print(f"Error generating or writing LandMask XML graph: {e}")
            if xml_path.exists():
                os.remove(xml_path)  # Clean up if writing failed mid-way
            return None

    def Calibration(self, Pols: list[str] = ['VH'], output_complex: bool = True):
        """ Applies radiometric calibration. """
        self._reset_command()
        pol_str = ','.join(Pols)  # More pythonic way to join strings
        self.current_cmd.append(f'Calibration -PoutputImageInComplex={str(output_complex).lower()} -PselectedPolarisations={pol_str}')
        return self._call(suffix='CAL')

    def Deburst(self, Pols: list[str] = ['VH']):
        """ Applies TOPSAR Debursting. """
        self._reset_command()
        pol_str = ','.join(Pols)
        self.current_cmd.append(f'TOPSAR-Deburst -PselectedPolarisations={pol_str}')
        return self._call(suffix='DEB')

    def Multilook(self, nRgLooks: int, nAzLooks: int):
        """ Applies Multilooking. """
        self._reset_command()
        self.current_cmd.append(f'Multilook -PnRgLooks={nRgLooks} -PnAzLooks={nAzLooks}')
        return self._call(suffix='ML')

    def AdaptiveThresholding(self, background_window_m: float = 800, guard_window_m: float = 500, target_window_m: float = 50, pfa: float = 6.5):
        """ Applies Adaptive Thresholding for object detection. """
        self._reset_command()
        self.current_cmd.append(f'AdaptiveThresholding -PbackgroundWindowSizeInMeter={background_window_m} -PguardWindowSizeInMeter={guard_window_m} -Ppfa={pfa} -PtargetWindowSizeInMeter={target_window_m}')
        return self._call(suffix='AT')

    def ObjectDiscrimination(self, min_target_m: float, max_target_m: float):
        """ Discriminates objects based on size. """
        self._reset_command()
        self.current_cmd.append(f'Object-Discrimination -PminTargetSizeInMeter={min_target_m} -PmaxTargetSizeInMeter={max_target_m}')
        return self._call(suffix='OD')

    def Subset(self, loc: list[float], sourceBands: list[str], idx: str, winSize: int = 128, GeoCoords: bool = False, copy_metadata: bool = True):
        """ Creates a subset of the product. """
        original_format = self.format
        self.format = 'GeoTIFF'
        self._reset_command()

        source_bands_str = ','.join(sourceBands)

        if not GeoCoords:
            x = int(loc[0]) - winSize // 2
            y = int(loc[1]) - winSize // 2
            region = f'{x},{y},{winSize},{winSize}'
            self.current_cmd.append(f'Subset -PcopyMetadata={str(copy_metadata).lower()} -Pregion={region} -PsourceBands={source_bands_str}')
        else:
            lon, lat = loc[0], loc[1]
            half_size_deg = winSize * 0.0001
            min_lon, max_lon = lon - half_size_deg, lon + half_size_deg
            min_lat, max_lat = lat - half_size_deg, lat + half_size_deg
            wkt_roi = f'POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))'
            print(f"Using WKT ROI: {wkt_roi}")
            self.current_cmd.append(f"Subset -PcopyMetadata={str(copy_metadata).lower()} -PgeoRegion='{wkt_roi}' -PsourceBands={source_bands_str}")

        result = self._call(suffix=f'SUB{idx}')
        self.format = original_format
        return result


def _process_product_cfar(product_path: Path, mask_shp_path: Path, gpt_mode: str | None, delete_intermediate: bool, pfa_thresholds: list[float]):
    """Helper function to process a single product through the CFAR chain."""
    out_dir = product_path.parent
    try:
        prod_type = mode_identifier(product_path.name)
    except Exception as e:
        print(f"Error determining product type for {product_path}: {e}. Cannot proceed with CFAR.")
        return None, None

    op = GPT(product=product_path.as_posix(), outdir=out_dir.as_posix(), mode=gpt_mode)
    op.prod_type = prod_type

    processed_products = []

    prod_start_cfar = product_path.as_posix()

    if prod_type == GPT.SEN:
        prod_deb = op.Deburst()
        if not prod_deb:
            return None, None
        processed_products.append(Path(prod_deb))
        prod_cal = op.Calibration(Pols=['VH'])
        if not prod_cal:
            return None, None
        processed_products.append(Path(prod_cal))
        prod_shp = op.ImportVector(vector_data=mask_shp_path)
        if not prod_shp:
            return None, None
        processed_products.append(Path(prod_shp))
        prod_lm = op.LandMask()
        if not prod_lm:
            return None, None
        processed_products.append(Path(prod_lm))
        prod_start_cfar = prod_lm
        if delete_intermediate:
            delProd(processed_products[1])
            delProd(processed_products[2])

    elif prod_type == GPT.CSK:
        prod_ml = op.Multilook(nRgLooks=2, nAzLooks=2)
        if not prod_ml:
            return None, None
        processed_products.append(Path(prod_ml))
        prod_cal = op.Calibration(Pols=['HH'])
        if not prod_cal:
            return None, None
        processed_products.append(Path(prod_cal))
        prod_shp = op.ImportVector(vector_data=mask_shp_path)
        if not prod_shp:
            return None, None
        processed_products.append(Path(prod_shp))
        prod_lm = op.LandMask()
        if not prod_lm:
            return None, None
        processed_products.append(Path(prod_lm))
        prod_start_cfar = prod_lm
        if delete_intermediate:
            delProd(processed_products[0])
            delProd(processed_products[1])
            delProd(processed_products[2])

    elif prod_type == GPT.SAO:
        prod_shp = op.ImportVector(vector_data=mask_shp_path)
        if not prod_shp:
            return None, None
        processed_products.append(Path(prod_shp))
        prod_lm = op.LandMask()
        if not prod_lm:
            return None, None
        processed_products.append(Path(prod_lm))
        prod_start_cfar = prod_lm
        if delete_intermediate:
            delProd(processed_products[0])

    last_successful_excel = None
    for pfa in pfa_thresholds:
        op_cfar = GPT(product=prod_start_cfar, outdir=out_dir.as_posix(), mode=gpt_mode)

        at_params = {'pfa': pfa}
        if prod_type == GPT.CSK:
            at_params.update({'background_window_m': 650, 'guard_window_m': 400, 'target_window_m': 25})

        prod_at = op_cfar.AdaptiveThresholding(**at_params)
        if not prod_at:
            continue

        prod_od = op_cfar.ObjectDiscrimination(min_target_m=35, max_target_m=500)
        if not prod_od:
            if delete_intermediate:
                delProd(prod_at)
            continue

        prod_od_path = Path(prod_od)
        prod_od_data_dir = prod_od_path.with_suffix('.data')

        try:
            csv_files = list(prod_od_data_dir.glob('*.csv'))
            ship_csv_path = None
            for f in csv_files:
                if f.stem.lower().startswith('ship'):
                    ship_csv_path = f
                    break

            if ship_csv_path:
                ship_detections_df = pd.read_csv(ship_csv_path, header=1, sep='\t')
                out_excel_path = out_dir / f"{product_path.stem}_pfa_{pfa}.xlsx"
                ship_detections_df.to_excel(out_excel_path, index=False)
                print(f'Saved ExcelFile to: {out_excel_path}')
                last_successful_excel = out_excel_path.as_posix()
            else:
                print(f"No Ship detection CSV found in {prod_od_data_dir} for PFA {pfa}.")

        except FileNotFoundError:
            print(f".data directory not found: {prod_od_data_dir}")
        except pd.errors.EmptyDataError:
            print(f"Ship detection CSV file is empty: {ship_csv_path}")
        except Exception as e:
            print(f"Error processing detection results for PFA {pfa}: {e}")
        finally:
            if delete_intermediate:
                delProd(prod_at)
                delProd(prod_od)

    if delete_intermediate and Path(prod_start_cfar).exists() and prod_start_cfar != product_path.as_posix():
        delProd(prod_start_cfar)

    first_processed = processed_products[0].as_posix() if processed_products else product_path.as_posix()
    return first_processed, last_successful_excel


def CFAR(prod: str | Path, mask_shp_path: str | Path, mode: str | None = None, Thresh: list[float] | float = 12.5, DELETE: bool = False):
    """
    Performs Constant False Alarm Rate (CFAR) ship detection processing chain.

    Inputs:
         prod: Path to the input SAR product file.
         mask_shp_path: Path to the shapefile used for land masking.
         mode: OS mode ('MacOS', 'Ubuntu', None) for GPT configuration.
         Thresh: A single PFA threshold or a list of PFA thresholds to test.
         DELETE: If True, delete intermediate processing files.
    Returns:
        Tuple[str | None, str | None]: Path to the first major processed product (e.g., Deburst/Multilook)
                                       and path to the last generated Excel file, or None if processing fails.
    """
    product_path = Path(prod)
    mask_path = Path(mask_shp_path)

    if isinstance(Thresh, float):
        pfa_thresholds = [Thresh]
    elif isinstance(Thresh, list):
        pfa_thresholds = Thresh
    else:
        print("Warning: Invalid type for Thresh, using default [12.5].")
        pfa_thresholds = [12.5]

    return _process_product_cfar(
        product_path=product_path,
        mask_shp_path=mask_path,
        gpt_mode=mode,
        delete_intermediate=DELETE,
        pfa_thresholds=pfa_thresholds
    )


def _CFAR_Batch(prod_sen: str | Path, mask_shp_path: str | Path, mode: str, prod_type: str):
    """
    Internal batch processing function for CFAR (appears deprecated or specific use case).
    Prefer using the main CFAR function with a list of thresholds.

    Note: This function seems less flexible than CFAR and uses hardcoded PFA values
          and assumes delete_intermediate=True. Consider refactoring or removing if
          the main CFAR function covers the use case.
    """
    warnings.warn("_CFAR_Batch seems redundant with the main CFAR function. Consider refactoring.", DeprecationWarning)

    product_path = Path(prod_sen)
    mask_path = Path(mask_shp_path)

    pfa_thresholds = [6.5, 7.5, 9.5, 11.5, 12.5, 15.5, 16.5, 16.75]

    delete_intermediate = True

    _process_product_cfar(
        product_path=product_path,
        mask_shp_path=mask_path,
        gpt_mode=mode,
        delete_intermediate=delete_intermediate,
        pfa_thresholds=pfa_thresholds
    )
