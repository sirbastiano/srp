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
        'Ubuntu': 16,
        'MacOS': 8,
        'Windows': 8
    }
    
    # Supported output formats for SNAP GPT processing
    OUTPUT_FORMATS = [
        'PyRate export',                  # PyRate configuration format
        'GeoTIFF+XML',                   # GeoTIFF with XML metadata
        'JPEG2000',                      # JPEG2000 compressed format
        'GDAL-BMP-WRITER',               # Windows Bitmap format
        'NetCDF4-CF',                    # NetCDF4 Climate and Forecast conventions
        'PolSARPro',                     # Polarimetric SAR data analysis format
        'Snaphu',                        # Statistical-cost network-flow algorithm format
        'Generic Binary BSQ',            # Band Sequential binary format
        'CSV',                           # Comma-separated values format
        'GDAL-GS7BG-WRITER',            # Golden Software 7 Binary Grid format
        'GDAL-GTiff-WRITER',            # GDAL GeoTIFF writer
        'GDAL-BT-WRITER',               # VTP .bt terrain format
        'GeoTIFF-BigTIFF',              # BigTIFF format for large files
        'GDAL-RMF-WRITER',              # Raster Matrix Format
        'GDAL-KRO-WRITER',              # KOLOR Raw format
        'GDAL-PNM-WRITER',              # Portable Anymap format
        'Gamma',                         # Gamma Remote Sensing format
        'GDAL-MFF-WRITER',              # Vexcel MFF format
        'GeoTIFF',                       # Standard GeoTIFF format
        'NetCDF4-BEAM',                  # NetCDF4 BEAM format
        'GDAL-GTX-WRITER',              # NOAA .gtx vertical datum shift format
        'GDAL-RST-WRITER',              # Idrisi Raster format
        'GDAL-SGI-WRITER',              # SGI Image format
        'ZNAP',                          # SNAP compressed format
        'GDAL-GSBG-WRITER',             # Golden Software Binary Grid format
        'ENVI',                          # ENVI header labeled raster format
        'BEAM-DIMAP',                    # BEAM-DIMAP XML product format
        'GDAL-HFA-WRITER',              # Erdas Imagine format
        'GDAL-COG-WRITER',              # Cloud Optimized GeoTIFF format
        'HDF5',                          # Hierarchical Data Format version 5
        'GDAL-NITF-WRITER',             # National Imagery Transmission Format
        'GDAL-SAGA-WRITER',             # SAGA GIS Binary format
        'GDAL-ILWIS-WRITER',            # ILWIS Raster Map format
        'JP2,JPG,PNG,BMP,GIF',          # Common image formats
        'GDAL-PCIDSK-WRITER'            # PCI PCIDSK Database File format
        ]

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
        assert self.prod_path.exists(), f"Product path does not exist: {self.prod_path}"
        self.name = self.prod_path.stem
        assert format in self.OUTPUT_FORMATS, f"Unsupported format: {format}. Supported formats are: {self.OUTPUT_FORMATS}"
        self.format = format
        assert outdir, "Output directory must be specified"
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
            '-x ', # TODO: virtual memory, auto add with checks
            # '-c', # TODO: cache size, auto add with checks
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

    def Subset(self, 
               source_bands: list[str] | None = None,
               tie_point_grids: list[str] | None = None,
               region: str | None = None,
               reference_band: str | None = None,
               geo_region: str | None = None,
               sub_sampling_x: int = 1,
               sub_sampling_y: int = 1,
               full_swath: bool = False,
               vector_file: str | Path | None = None,
               polygon_region: str | None = None,
               copy_metadata: bool = False,
               suffix: str = 'SUB'):
        """
        Creates a spatial and/or spectral subset of a data product.
        
        This method allows extraction of specific bands, spatial regions, and subsetting
        based on pixel coordinates, geographical coordinates, or vector polygons.
        
        Args:
            source_bands (list[str] | None, optional): The list of source bands to include.
                If None, all bands are included. Defaults to None.
            tie_point_grids (list[str] | None, optional): The list of tie-point grid names 
                to include. Defaults to None.
            region (str | None, optional): The subset region in pixel coordinates.
                Use the format: 'x,y,width,height' (e.g., '100,200,500,500').
                If not given, the entire scene is used. The 'geo_region' parameter 
                has precedence. Defaults to None.
            reference_band (str | None, optional): The band used to indicate the pixel 
                coordinates. Defaults to None.
            geo_region (str | None, optional): The subset region in geographical coordinates 
                using WKT-format, e.g., 'POLYGON((lon1 lat1, lon2 lat2, ..., lon1 lat1))'.
                If not given, the entire scene is used. This parameter has precedence over 
                'region'. Defaults to None.
            sub_sampling_x (int, optional): The pixel sub-sampling step in X (horizontal 
                image direction). Defaults to 1 (no sub-sampling).
            sub_sampling_y (int, optional): The pixel sub-sampling step in Y (vertical 
                image direction). Defaults to 1 (no sub-sampling).
            full_swath (bool, optional): Forces the operator to extend the subset region 
                to the full swath. Defaults to False.
            vector_file (str | Path | None, optional): The file from which the polygon 
                is read. Defaults to None.
            polygon_region (str | None, optional): The subset region in geographical 
                coordinates using WKT-format. If not given, the geo_region or region 
                is used. Defaults to None.
            copy_metadata (bool, optional): Whether to copy the metadata of the source 
                product. Defaults to False.
            suffix (str, optional): Suffix for the output filename. Defaults to 'SUB'.
        
        Returns:
            str | None: Path to the subset output product, or None if failed.
        
        Examples:
            # Subset by pixel region
            op.Subset(region='100,200,500,500', source_bands=['Amplitude_VV'])
            
            # Subset by geographic region
            op.Subset(
                geo_region='POLYGON((10.0 53.0, 11.0 53.0, 11.0 54.0, 10.0 54.0, 10.0 53.0))',
                source_bands=['Intensity_VH', 'Intensity_VV']
            )
            
            # Subset using vector file
            op.Subset(vector_file='aoi.shp', source_bands=['Sigma0_VV'])
        """
        self._reset_command()
        
        cmd_params = []
        
        if source_bands:
            source_bands_str = ','.join(source_bands)
            cmd_params.append(f'-PsourceBands={source_bands_str}')
        
        if tie_point_grids:
            tie_point_grids_str = ','.join(tie_point_grids)
            cmd_params.append(f'-PtiePointGrids={tie_point_grids_str}')
        
        if region:
            cmd_params.append(f'-Pregion={region}')
        
        if reference_band:
            cmd_params.append(f'-PreferenceBand={reference_band}')
        
        if geo_region:
            # Ensure proper quoting for WKT geometry
            cmd_params.append(f"-PgeoRegion='{geo_region}'")
        
        if sub_sampling_x != 1:
            cmd_params.append(f'-PsubSamplingX={sub_sampling_x}')
        
        if sub_sampling_y != 1:
            cmd_params.append(f'-PsubSamplingY={sub_sampling_y}')
        
        if full_swath:
            cmd_params.append(f'-PfullSwath={str(full_swath).lower()}')
        
        if vector_file:
            vector_path = Path(vector_file)
            cmd_params.append(f'-PvectorFile={vector_path.as_posix()}')
        
        if polygon_region:
            # Ensure proper quoting for WKT polygon
            cmd_params.append(f"-PpolygonRegion='{polygon_region}'")
        
        cmd_params.append(f'-PcopyMetadata={str(copy_metadata).lower()}')
        
        self.current_cmd.append(f'Subset {" ".join(cmd_params)}')
        return self._call(suffix=suffix)

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

    def ApplyOrbitFile(self, orbit_type: str = 'Sentinel Precise (Auto Download)', 
                       poly_degree: int = 3, continue_on_fail: bool = False):
        """
        Applies orbit file correction to SAR products.
        
        This method updates the orbit state vectors in the product metadata using
        precise or restituted orbit files, improving geolocation accuracy.
        
        Args:
            orbit_type (str, optional): Type of orbit file to apply. 
                Defaults to 'Sentinel Precise (Auto Download)'.
                Valid options:
                - 'Sentinel Precise (Auto Download)'
                - 'Sentinel Restituted (Auto Download)'
                - 'DORIS Preliminary POR (ENVISAT)'
                - 'DORIS Precise VOR (ENVISAT) (Auto Download)'
                - 'DELFT Precise (ENVISAT, ERS1&2) (Auto Download)'
                - 'PRARE Precise (ERS1&2) (Auto Download)'
                - 'Kompsat5 Precise'
            poly_degree (int, optional): Degree of polynomial for orbit interpolation.
                Defaults to 3.
            continue_on_fail (bool, optional): Continue processing if orbit file application fails.
                Defaults to False.
        
        Returns:
            str | None: Path to the output product with applied orbit file, or None if failed.
        """
        self._reset_command()
        self.current_cmd.append(
            f'Apply-Orbit-File '
            f'-PorbitType="{orbit_type}" '
            f'-PpolyDegree={poly_degree} '
            f'-PcontinueOnFail={str(continue_on_fail).lower()}'
        )
        return self._call(suffix='ORB')

    def TerrainCorrection(self, 
                         source_bands: list[str] | None = None,
                         dem_name: str = 'SRTM 3Sec',
                         external_dem_file: str | Path | None = None,
                         external_dem_no_data_value: float = 0.0,
                         external_dem_apply_egm: bool = True,
                         dem_resampling_method: str = 'BILINEAR_INTERPOLATION',
                         img_resampling_method: str = 'BILINEAR_INTERPOLATION',
                         pixel_spacing_in_meter: float = 0.0,
                         pixel_spacing_in_degree: float = 0.0,
                         map_projection: str = 'WGS84(DD)', # 'GEOGCS["WGS84(DD)", DATUM["WGS84", SPHEROID["WGS84", 6378137.0, 298.257223563]], PRIMEM["Greenwich", 0.0], UNIT["degree", 0.017453292519943295], AXIS["Geodetic longitude", EAST], AXIS["Geodetic latitude", NORTH], AUTHORITY["EPSG","4326"]]',
                         align_to_standard_grid: bool = False,
                         standard_grid_origin_x: float = 0.0,
                         standard_grid_origin_y: float = 0.0,
                         nodata_value_at_sea: bool = False,
                         save_dem: bool = False,
                         save_lat_lon: bool = True,
                         save_incidence_angle_from_ellipsoid: bool = False,
                         save_local_incidence_angle: bool = True,
                         save_projected_local_incidence_angle: bool = False,
                         save_selected_source_band: bool = True,
                         save_layover_shadow_mask: bool = False,
                         output_complex: bool = True,
                         apply_radiometric_normalization: bool = False,
                         save_sigma_nought: bool = False,
                         save_gamma_nought: bool = False,
                         save_beta_nought: bool = False,
                         incidence_angle_for_sigma0: str = 'Use projected local incidence angle from DEM',
                         incidence_angle_for_gamma0: str = 'Use projected local incidence angle from DEM',
                         aux_file: str = 'Latest Auxiliary File',
                         external_aux_file: str | Path | None = None):
        """
        Applies terrain correction (orthorectification) to SAR products using Range-Doppler method.
        
        This method corrects geometric distortions caused by topography and sensor geometry,
        projecting the SAR image onto a cartographic coordinate system using a DEM.
        
        Args:
            source_bands (list[str] | None, optional): List of source bands to process. 
                Defaults to None (all bands).
            dem_name (str, optional): Digital elevation model name. Defaults to 'SRTM 3Sec'.
            external_dem_file (str | Path | None, optional): Path to external DEM file. 
                Defaults to None.
            external_dem_no_data_value (float, optional): No data value for external DEM. 
                Defaults to 0.0.
            external_dem_apply_egm (bool, optional): Apply EGM96 geoid to external DEM. 
                Defaults to True.
            dem_resampling_method (str, optional): DEM resampling method. 
                Defaults to 'BILINEAR_INTERPOLATION'.
            img_resampling_method (str, optional): Image resampling method. 
                Defaults to 'BILINEAR_INTERPOLATION'.
            pixel_spacing_in_meter (float, optional): Output pixel spacing in meters. 
                Defaults to 0.0 (automatic).
            pixel_spacing_in_degree (float, optional): Output pixel spacing in degrees. 
                Defaults to 0.0 (automatic).
            map_projection (str, optional): Map projection in WKT format. 
                Defaults to WGS84 geographic coordinates.
            align_to_standard_grid (bool, optional): Align output to standard grid. 
                Defaults to False.
            standard_grid_origin_x (float, optional): X-coordinate of standard grid origin. 
                Defaults to 0.0.
            standard_grid_origin_y (float, optional): Y-coordinate of standard grid origin. 
                Defaults to 0.0.
            nodata_value_at_sea (bool, optional): Mask sea areas with no data value. 
                Defaults to False.
            save_dem (bool, optional): Save DEM band in output. Defaults to False.
            save_lat_lon (bool, optional): Save latitude/longitude bands. Defaults to True.
            save_incidence_angle_from_ellipsoid (bool, optional): Save incidence angle from ellipsoid. 
                Defaults to False.
            save_local_incidence_angle (bool, optional): Save local incidence angle. 
                Defaults to True.
            save_projected_local_incidence_angle (bool, optional): Save projected local incidence angle. 
                Defaults to False.
            save_selected_source_band (bool, optional): Save selected source bands. 
                Defaults to True.
            save_layover_shadow_mask (bool, optional): Save layover/shadow mask. 
                Defaults to False.
            output_complex (bool, optional): Output complex data. Defaults to True.
            apply_radiometric_normalization (bool, optional): Apply radiometric normalization. 
                Defaults to False.
            save_sigma_nought (bool, optional): Save sigma nought band. Defaults to False.
            save_gamma_nought (bool, optional): Save gamma nought band. Defaults to False.
            save_beta_nought (bool, optional): Save beta nought band. Defaults to False.
            incidence_angle_for_sigma0 (str, optional): Incidence angle type for sigma0. 
                Defaults to 'Use projected local incidence angle from DEM'.
            incidence_angle_for_gamma0 (str, optional): Incidence angle type for gamma0. 
                Defaults to 'Use projected local incidence angle from DEM'.
            aux_file (str, optional): Auxiliary file selection. 
                Defaults to 'Latest Auxiliary File'.
            external_aux_file (str | Path | None, optional): Path to external auxiliary file. 
                Defaults to None.
        
        Returns:
            str | None: Path to the terrain-corrected output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = []
        
        if source_bands:
            source_bands_str = ','.join(source_bands)
            cmd_params.append(f'-PsourceBands={source_bands_str}')
        
        cmd_params.extend([
            f'-PdemName="{dem_name}"',
            f'-PexternalDEMNoDataValue={external_dem_no_data_value}',
            f'-PexternalDEMApplyEGM={str(external_dem_apply_egm).lower()}',
            f'-PdemResamplingMethod={dem_resampling_method}',
            f'-PimgResamplingMethod={img_resampling_method}',
            f'-PpixelSpacingInMeter={pixel_spacing_in_meter}',
            f'-PpixelSpacingInDegree={pixel_spacing_in_degree}',
            f'-PmapProjection="{map_projection}"',
            f'-PalignToStandardGrid={str(align_to_standard_grid).lower()}',
            f'-PstandardGridOriginX={standard_grid_origin_x}',
            f'-PstandardGridOriginY={standard_grid_origin_y}',
            f'-PnodataValueAtSea={str(nodata_value_at_sea).lower()}',
            f'-PsaveDEM={str(save_dem).lower()}',
            f'-PsaveLatLon={str(save_lat_lon).lower()}',
            f'-PsaveIncidenceAngleFromEllipsoid={str(save_incidence_angle_from_ellipsoid).lower()}',
            f'-PsaveLocalIncidenceAngle={str(save_local_incidence_angle).lower()}',
            f'-PsaveProjectedLocalIncidenceAngle={str(save_projected_local_incidence_angle).lower()}',
            f'-PsaveSelectedSourceBand={str(save_selected_source_band).lower()}',
            f'-PsaveLayoverShadowMask={str(save_layover_shadow_mask).lower()}',
            f'-PoutputComplex={str(output_complex).lower()}',
            f'-PapplyRadiometricNormalization={str(apply_radiometric_normalization).lower()}',
            f'-PsaveSigmaNought={str(save_sigma_nought).lower()}',
            f'-PsaveGammaNought={str(save_gamma_nought).lower()}',
            f'-PsaveBetaNought={str(save_beta_nought).lower()}',
            f'-PincidenceAngleForSigma0="{incidence_angle_for_sigma0}"',
            f'-PincidenceAngleForGamma0="{incidence_angle_for_gamma0}"',
            f'-PauxFile="{aux_file}"'
        ])
        
        if external_dem_file:
            cmd_params.append(f'-PexternalDEMFile={Path(external_dem_file).as_posix()}')
        
        if external_aux_file:
            cmd_params.append(f'-PexternalAuxFile={Path(external_aux_file).as_posix()}')
        
        self.current_cmd.append(f'Terrain-Correction {" ".join(cmd_params)}')
        return self._call(suffix='TC')

    def Demodulate(self):
        """
        Performs demodulation and deramping of SLC data.
        
        This method removes the modulation and ramping applied to Single Look Complex (SLC)
        data during SAR processing, preparing the data for further interferometric or
        analysis operations.
        
        Returns:
            str | None: Path to the demodulated output product, or None if failed.
        """
        self._reset_command()
        # Demodulate operator requires -SsourceProduct instead of -Ssource
        for i, cmd_part in enumerate(self.current_cmd):
            if cmd_part.startswith('-Ssource='):
                self.current_cmd[i] = cmd_part.replace('-Ssource=', '-SsourceProduct=')
                break
        self.current_cmd.append('Demodulate')
        return self._call(suffix='DEMOD')

    def Write(self, 
              output_file: str | Path | None = None,
              format_name: str | None = None,
              clear_cache_after_row_write: bool = False,
              delete_output_on_failure: bool = True,
              write_entire_tile_rows: bool = False):
        """
        Writes a data product to a file with explicit control over write parameters.
        
        This method provides direct access to the SNAP Write operator, allowing
        fine-grained control over the write process including caching behavior
        and tile row processing.
        
        Args:
            output_file (str | Path | None, optional): The output file path. 
                If None, uses the standard output path construction. Defaults to None.
            format_name (str | None, optional): The output file format name. 
                If None, uses the format specified in the GPT instance. Defaults to None.
            clear_cache_after_row_write (bool, optional): If True, the internal tile cache 
                is cleared after a tile row has been written. Only effective if 
                write_entire_tile_rows is True. Defaults to False.
            delete_output_on_failure (bool, optional): If True, all output files are 
                deleted after a failed write operation. Defaults to True.
            write_entire_tile_rows (bool, optional): If True, the write operation waits 
                until an entire tile row is computed before writing. Defaults to False.
        
        Returns:
            str | None: Path to the written output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = []
        cmd_params.append(
            f'-PclearCacheAfterRowWrite={str(clear_cache_after_row_write).lower()}'
        )
        cmd_params.append(
            f'-PdeleteOutputOnFailure={str(delete_output_on_failure).lower()}'
        )
        cmd_params.append(
            f'-PwriteEntireTileRows={str(write_entire_tile_rows).lower()}'
        )
        
        # If output file is specified, use it; otherwise let _call handle it
        if output_file:
            output_path = Path(output_file)
            cmd_params.append(f'-Pfile={output_path.as_posix()}')
        
        # If format is specified, use it; otherwise use instance format
        if format_name:
            cmd_params.append(f'-PformatName="{format_name}"')
        else:
            cmd_params.append(f'-PformatName="{self.format}"')
        
        self.current_cmd.append(f'Write {" ".join(cmd_params)}')
        return self._call(suffix='WRITE')









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
