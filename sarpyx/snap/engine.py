import os
import subprocess
import warnings
from pathlib import Path
import urllib.request
import zipfile

import pandas as pd

from ..utils.io import delProd

warnings.filterwarnings("ignore")


class GPT:
    """A wrapper class for executing SNAP Graph Processing Tool (GPT) commands."""

    # Default GPT paths and parallelism for different OS
    GPT_PATHS = {
        'Ubuntu': '/home/<username>/ESA-STEP/snap/bin/gpt',
        'MacOS': '/Applications/snap/bin/gpt',
        'Windows': 'gpt.exe'
    }
    
    DEFAULT_PARALLELISM = {
        'Ubuntu': 8,
        'MacOS': 8,
        'Windows': 8
    }

    def __init__(self, product: str | Path, 
        outdir: str | Path, 
        format: str = 'BEAM-DIMAP',
        gpt_path: str | None = "/usr/local/snap/bin/gpt", 
        mode: str | None = None):
        """
        SNAP GPT processing engine.
        
        Args:
            product (str | Path): Path to the input SAR product file or directory.
            outdir (str | Path): Output directory where processed results will be saved.
            format (str, optional): Output format for processed data. Defaults to 'BEAM-DIMAP'.
                Supported formats include 'BEAM-DIMAP' and 'GEOTIFF'.
            gpt_path (str | None, optional): Path to the SNAP GPT executable. 
                Defaults to "/usr/local/snap/bin/gpt".
            mode (str | None, optional): Processing mode configuration. Defaults to None.
        Attributes:
        
            prod_path (Path): Path object for the input product.
            name (str): Stem name of the input product file.
            format (str): Output format for processed data.
            outdir (Path): Path object for the output directory.
            mode (str | None): Processing mode configuration.
            gpt_executable: Path to the validated GPT executable.
            parallelism: Configured parallelism settings for processing.
            current_cmd (list): List to store current command components.
        """
        
        self.prod_path = Path(product)
        self.name = self.prod_path.stem
        self.format = format
        self.outdir = Path(outdir)
        self.mode = mode
        self.gpt_executable = self._get_gpt_executable(gpt_path)
        self.parallelism = self._get_parallelism()
        self.current_cmd = []

    def _get_gpt_executable(self, gpt_path: str | None = None) -> str:
        """Determines the correct GPT executable path."""
        if gpt_path:
            return gpt_path
        
        if self.mode and self.mode in self.GPT_PATHS:
            return self.GPT_PATHS[self.mode]
        
        # Auto-detect based on OS
        if os.name == 'posix':
            for path in [self.GPT_PATHS['MacOS'], self.GPT_PATHS['Ubuntu']]:
                if Path(path).exists():
                    return path
            return 'gpt'
        elif os.name == 'nt':
            return self.GPT_PATHS['Windows']
        else:
            return 'gpt'

    def _get_parallelism(self) -> int:
        """Determines the parallelism level."""
        if self.mode and self.mode in self.DEFAULT_PARALLELISM:
            return self.DEFAULT_PARALLELISM[self.mode]
        
        # Auto-detect based on OS
        if os.name == 'posix':
            if Path(self.GPT_PATHS['MacOS']).exists():
                return self.DEFAULT_PARALLELISM['MacOS']
            return self.DEFAULT_PARALLELISM.get('Ubuntu', 6)
        elif os.name == 'nt':
            return self.DEFAULT_PARALLELISM['Windows']
        else:
            return 6

    def _reset_command(self):
        """Resets the command list for a new GPT operation."""
        self.current_cmd = [
            self.gpt_executable,
            f'-q {self.parallelism}',
            '-x',
            '-e',
            f'-Ssource={self.prod_path.as_posix()}'
        ]

    def _build_output_path(self, suffix: str) -> Path:
        """Builds the output path for a processing step."""
        base_name = self.outdir / f"{self.name}_{suffix}"
        if self.format == 'GEOTIFF':
            return base_name.with_suffix('.tif')
        else:
            return base_name.with_suffix('.dim')

    def _execute_command(self) -> bool:
        """Executes the currently built GPT command."""
        cmd_str = ' '.join(self.current_cmd)
        print(f"Executing GPT command: {cmd_str}")
        
        try:
            process = subprocess.run(
                cmd_str, 
                shell=True, 
                check=True, 
                capture_output=True, 
                text=True, 
                timeout=3600
            )
            
            if process.stdout:
                print("GPT Output:", process.stdout)
            if process.stderr:
                print("GPT Warnings:", process.stderr)
            
            print("Command executed successfully!")
            return True
            
        except subprocess.TimeoutExpired:
            print("Error: GPT command timed out after 1 hour")
            return False
            
        except subprocess.CalledProcessError as e:
            print(f"Error executing GPT command: {cmd_str}")
            print(f"Return code: {e.returncode}")
            if e.stdout:
                print(f"Stdout: {e.stdout}")
            if e.stderr:
                print(f"Stderr: {e.stderr}")
            return False
            
        except FileNotFoundError:
            print(f"Error: GPT executable '{self.gpt_executable}' not found!")
            print("Ensure SNAP is installed and configured correctly.")
            return False
            
        except Exception as e:
            print(f"Unexpected error during GPT execution: {type(e).__name__}: {e}")
            return False

    def _call(self, suffix: str) -> str | None:
        """Finalizes and executes the GPT command."""
        output_path = self._build_output_path(suffix)
        self.current_cmd.extend([
            f'-t {output_path.as_posix()}',
            f'-f {self.format}'
        ])

        if self._execute_command():
            self.prod_path = output_path
            return output_path.as_posix()
        else:
            return None

    def ImportVector(self, vector_data: str | Path):
        """Imports vector data into the product."""
        vector_path = Path(vector_data)
        
        # Check if the shapefile exists
        if not vector_path.exists():
            print(f"Shapefile not found: {vector_path}")
            print("Downloading from Zenodo...")
            
            # Download and extract from Zenodo
            zenodo_url = "https://zenodo.org/api/records/6992586/files-archive"
            download_dir = Path.cwd() / "zenodo_download"
            archive_path = download_dir / "zenodo_archive.zip"
            
            try:
                # Download the archive
                urllib.request.urlretrieve(zenodo_url, archive_path)
                print(f"Downloaded archive to: {archive_path}")
                
                # Extract the archive
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(download_dir)
                print(f"Extracted archive to: {download_dir}")
                
                # Find shapefile in extracted contents
                shp_files = list(download_dir.rglob("*.shp"))
                if shp_files:
                    vector_path = shp_files[0]  # Use the first shapefile found
                    print(f"Using shapefile: {vector_path}")
                else:
                    raise FileNotFoundError("No shapefile found in downloaded archive")
                
                # Clean up the archive
                archive_path.unlink()
                
            except Exception as e:
                print(f"Error downloading or extracting shapefile: {e}")
                return None
        
        self._reset_command()
        self.current_cmd.append(f'Import-Vector -PseparateShapes=false -PvectorFile={vector_path.as_posix()}')
        return self._call(suffix='SHP')

    def LandMask(self, 
            shoreline_extension: int = 300, 
            geometry_name: str = "Buff_750", 
            use_srtm: bool = True, 
            invert_geometry: bool = True, 
            land_mask: bool = False):
        """Applies Land-Sea Masking using a predefined XML graph structure."""
        
        self._reset_command()
        suffix = 'LM'
        output_path = self._build_output_path(suffix)
        xml_path = self.outdir / f"{self.name}_landmask_graph.xml"

        # Determine product type if not already set
        if not hasattr(self, 'prod_type'):
            try:
                self.prod_type = mode_identifier(self.prod_path.name)
                print(f"Inferred product type: {self.prod_type}")
            except Exception as e:
                print(f"Warning: Could not determine product type: {e}")
                self.prod_type = None

        # Determine source band based on product type
        if self.prod_type == "COSMO-SkyMed":
            source_band = 'Intensity_null'
        elif self.prod_type == "Sentinel-1":
            source_band = 'Intensity_VH'
        else:
            print(f"Warning: Product type is '{self.prod_type}'. Using default source band 'Intensity_VH'.")
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
        </graph>"""

        try:
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(graph_xml)

            self.current_cmd = [self.gpt_executable, xml_path.as_posix()]

            if self._execute_command():
                self.prod_path = output_path
                os.remove(xml_path)
                return output_path.as_posix()
            else:
                return None

        except Exception as e:
            print(f"Error generating LandMask XML graph: {e}")
            if xml_path.exists():
                os.remove(xml_path)
            return None

    def Calibration(self, Pols: list[str] = ['VV'], output_complex: bool = True):
        """Applies radiometric calibration."""
        self._reset_command()
        pol_str = ','.join(Pols)
        self.current_cmd.append(f'Calibration -PoutputImageInComplex={str(output_complex).lower()} -PselectedPolarisations={pol_str}')
        return self._call(suffix='CAL')

    def Deburst(self, Pols: list[str] = ['VH']):
        """Applies TOPSAR Debursting."""
        self._reset_command()
        pol_str = ','.join(Pols)
        self.current_cmd.append(f'TOPSAR-Deburst -PselectedPolarisations={pol_str}')
        return self._call(suffix='DEB')

    def Multilook(self, nRgLooks: int, nAzLooks: int):
        """Applies Multilooking."""
        self._reset_command()
        self.current_cmd.append(f'Multilook -PnRgLooks={nRgLooks} -PnAzLooks={nAzLooks}')
        return self._call(suffix='ML')

    def AdaptiveThresholding(self, background_window_m: float = 800, guard_window_m: float = 500, 
                           target_window_m: float = 50, pfa: float = 6.5):
        """Applies Adaptive Thresholding for object detection."""
        self._reset_command()
        self.current_cmd.append(f'AdaptiveThresholding -PbackgroundWindowSizeInMeter={background_window_m} -PguardWindowSizeInMeter={guard_window_m} -Ppfa={pfa} -PtargetWindowSizeInMeter={target_window_m}')
        return self._call(suffix='AT')

    def ObjectDiscrimination(self, min_target_m: float, max_target_m: float):
        """Discriminates objects based on size."""
        self._reset_command()
        self.current_cmd.append(f'Object-Discrimination -PminTargetSizeInMeter={min_target_m} -PmaxTargetSizeInMeter={max_target_m}')
        return self._call(suffix='OD')

    def Subset(self, loc: list[float], sourceBands: list[str], idx: str, winSize: int = 128, 
               GeoCoords: bool = False, copy_metadata: bool = True):
        """Creates a subset of the product."""
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
            self.current_cmd.append(f"Subset -PcopyMetadata={str(copy_metadata).lower()} -PgeoRegion='{wkt_roi}' -PsourceBands={source_bands_str}")

        result = self._call(suffix=f'SUB{idx}')
        self.format = original_format
        return result

    def AatsrSST(self, dual: bool = True, dual_coefficients_file: str = 'AVERAGE_POLAR_DUAL_VIEW',
                dual_mask_expression: str = '!cloud_flags_nadir.LAND and !cloud_flags_nadir.CLOUDY and !cloud_flags_nadir.SUN_GLINT and !cloud_flags_fward.LAND and !cloud_flags_fward.CLOUDY and !cloud_flags_fward.SUN_GLINT',
                invalid_sst_value: float = -999.0, nadir: bool = True,
                nadir_coefficients_file: str = 'AVERAGE_POLAR_SINGLE_VIEW',
                nadir_mask_expression: str = '!cloud_flags_nadir.LAND and !cloud_flags_nadir.CLOUDY and !cloud_flags_nadir.SUN_GLINT'):
        """
        Computes sea surface temperature (SST) from (A)ATSR products.
        This method processes ATSR (Along Track Scanning Radiometer) data to derive
        sea surface temperature using both dual-view and nadir-view algorithms.
        Args:
            dual (bool, optional): Enable dual-view SST processing. Defaults to True.
            dual_coefficients_file (str, optional): Coefficients file for dual-view processing.
                Defaults to 'AVERAGE_POLAR_DUAL_VIEW'.
            dual_mask_expression (str, optional): Mask expression for dual-view processing
                to exclude land, clouds, and sun glint pixels. Defaults to expression
                excluding these conditions for both nadir and forward views.
            invalid_sst_value (float, optional): Value assigned to invalid SST pixels.
                Defaults to -999.0.
            nadir (bool, optional): Enable nadir-view SST processing. Defaults to True.
            nadir_coefficients_file (str, optional): Coefficients file for nadir-view processing.
                Defaults to 'AVERAGE_POLAR_SINGLE_VIEW'.
            nadir_mask_expression (str, optional): Mask expression for nadir-view processing
                to exclude land, clouds, and sun glint pixels. Defaults to expression
                excluding these conditions for nadir view only.
        Returns:
            The result of the SST computation operation.
        Note:
            The method builds command parameters for the SNAP Aatsr.SST operator and
            executes the processing chain with a 'SST' suffix.
        """
        self._reset_command()
        
        cmd_params = []
        cmd_params.append(f'-Pdual={str(dual).lower()}')
        cmd_params.append(f'-PdualCoefficientsFile={dual_coefficients_file}')
        cmd_params.append(f'-PdualMaskExpression="{dual_mask_expression}"')
        cmd_params.append(f'-PinvalidSstValue={invalid_sst_value}')
        cmd_params.append(f'-Pnadir={str(nadir).lower()}')
        cmd_params.append(f'-PnadirCoefficientsFile={nadir_coefficients_file}')
        cmd_params.append(f'-PnadirMaskExpression="{nadir_mask_expression}"')
        
        self.current_cmd.append(f'Aatsr.SST {" ".join(cmd_params)}')
        return self._call(suffix='SST')


def _process_product_cfar(product_path: Path, mask_shp_path: Path, gpt_mode: str | None, 
                         delete_intermediate: bool, pfa_thresholds: list[float]):
    """Helper function to process a single product through the CFAR chain."""
    out_dir = product_path.parent
    try:
        prod_type = mode_identifier(product_path.name)
    except Exception as e:
        print(f"Error determining product type for {product_path}: {e}")
        return None, None

    op = GPT(product=product_path.as_posix(), outdir=out_dir.as_posix(), mode=gpt_mode)
    op.prod_type = prod_type

    processed_products = []
    prod_start_cfar = product_path.as_posix()

    # Process based on product type
    if prod_type == "Sentinel-1":
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

    elif prod_type == "COSMO-SkyMed":
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

    elif prod_type == "SAOCOM":
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

    # Process CFAR for each PFA threshold
    last_successful_excel = None
    for pfa in pfa_thresholds:
        op_cfar = GPT(product=prod_start_cfar, outdir=out_dir.as_posix(), mode=gpt_mode)

        at_params = {'pfa': pfa}
        if prod_type == "COSMO-SkyMed":
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
            ship_csv_path = next((f for f in csv_files if f.stem.lower().startswith('ship')), None)

            if ship_csv_path:
                ship_detections_df = pd.read_csv(ship_csv_path, header=1, sep='\t')
                out_excel_path = out_dir / f"{product_path.stem}_pfa_{pfa}.xlsx"
                ship_detections_df.to_excel(out_excel_path, index=False)
                print(f'Saved ExcelFile to: {out_excel_path}')
                last_successful_excel = out_excel_path.as_posix()
            else:
                print(f"No Ship detection CSV found for PFA {pfa}")

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


def CFAR(prod: str | Path, mask_shp_path: str | Path, mode: str | None = None, 
         Thresh: list[float] | float = 12.5, DELETE: bool = False):
    """
    Performs Constant False Alarm Rate (CFAR) ship detection processing chain.

    Args:
        prod: Path to the input SAR product file
        mask_shp_path: Path to the shapefile used for land masking
        mode: OS mode ('MacOS', 'Ubuntu', None) for GPT configuration
        Thresh: A single PFA threshold or a list of PFA thresholds to test
        DELETE: If True, delete intermediate processing files

    Returns:
        Tuple[str | None, str | None]: Path to the first major processed product 
                                       and path to the last generated Excel file
    """
    product_path = Path(prod)
    mask_path = Path(mask_shp_path)

    if isinstance(Thresh, (int, float)):
        pfa_thresholds = [float(Thresh)]
    elif isinstance(Thresh, list):
        pfa_thresholds = Thresh
    else:
        print("Warning: Invalid type for Thresh, using default [12.5]")
        pfa_thresholds = [12.5]

    return _process_product_cfar(
        product_path=product_path,
        mask_shp_path=mask_path,
        gpt_mode=mode,
        delete_intermediate=DELETE,
        pfa_thresholds=pfa_thresholds
    )


def mode_identifier(filename: str) -> str:
    """Identifies the product type based on filename."""
    if 'S1' in filename:
        return "Sentinel-1"
    elif 'CSK' in filename:
        return "COSMO-SkyMed"
    elif 'SAO' in filename:
        return "SAOCOM"
    else:
        raise ValueError(f"Unknown product type for file: {filename}")
