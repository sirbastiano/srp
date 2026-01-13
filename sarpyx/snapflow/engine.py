"""SNAP GPT (Graph Processing Tool) wrapper for SAR data processing.

This module provides a Pythonic interface to ESA's SNAP GPT command-line tool,
enabling automated processing of Synthetic Aperture Radar (SAR) data through
various operators like calibration, terrain correction, and subsetting.
"""

import os
import subprocess
import warnings
import zipfile
from pathlib import Path
from typing import List, Optional
from urllib import request

warnings.filterwarnings('ignore')


class GPT:
    """Wrapper class for executing SNAP Graph Processing Tool (GPT) commands.
    
    This class provides a high-level interface to SNAP's GPT command-line tool,
    abstracting away the complexity of command construction and execution.
    """

    GPT_PATHS = {
        'Ubuntu': '/home/<username>/ESA-STEP/snap/bin/gpt',
        'MacOS': '/Applications/snap/bin/gpt',
        'Windows': 'gpt.exe'
    }
    
 
    
    OUTPUT_FORMATS = [
        'PyRate export',
        'GeoTIFF+XML',
        'JPEG2000',
        'GDAL-BMP-WRITER',
        'NetCDF4-CF',
        'PolSARPro',
        'Snaphu',
        'Generic Binary BSQ',
        'CSV',
        'GDAL-GS7BG-WRITER',
        'GDAL-GTiff-WRITER',
        'GDAL-BT-WRITER',
        'GeoTIFF-BigTIFF',
        'GDAL-RMF-WRITER',
        'GDAL-KRO-WRITER',
        'GDAL-PNM-WRITER',
        'Gamma',
        'GDAL-MFF-WRITER',
        'GeoTIFF',
        'NetCDF4-BEAM',
        'GDAL-GTX-WRITER',
        'GDAL-RST-WRITER',
        'GDAL-SGI-WRITER',
        'ZNAP',
        'GDAL-GSBG-WRITER',
        'ENVI',
        'BEAM-DIMAP',
        'GDAL-HFA-WRITER',
        'GDAL-COG-WRITER',
        'HDF5',
        'GDAL-NITF-WRITER',
        'GDAL-SAGA-WRITER',
        'GDAL-ILWIS-WRITER',
        'JP2,JPG,PNG,BMP,GIF',
        'GDAL-PCIDSK-WRITER'
    ]
    
    EXTENSIONS_MAP = {
        'PyRate export': '.pyr',
        'GeoTIFF+XML': '.tif',
        'JPEG2000': '.jp2',
        'GDAL-BMP-WRITER': '.bmp',
        'NetCDF4-CF': '.nc',
        'PolSARPro': '.psp',
        'Snaphu': '.snaphu',
        'Generic Binary BSQ': '.bsq',
        'CSV': '.csv',
        'GDAL-GS7BG-WRITER': '.gs7bg',
        'GDAL-GTiff-WRITER': '.tif',
        'GDAL-BT-WRITER': '.bt',
        'GeoTIFF-BigTIFF': '.tif',
        'GDAL-RMF-WRITER': '.rmf',
        'GDAL-KRO-WRITER': '.kro',
        'GDAL-PNM-WRITER': '.pnm',
        'Gamma': '.gamma',
        'GDAL-MFF-WRITER': '.mff',
        'GeoTIFF': '.tif',
        'NetCDF4-BEAM': '.nc',
        'GDAL-GTX-WRITER': '.gtx',
        'GDAL-RST-WRITER': '.rst',
        'GDAL-SGI-WRITER': '.sgi',
        'ZNAP': '.znap',
        'GDAL-GSBG-WRITER': '.gsbg',
        'ENVI': '.hdr',
        'BEAM-DIMAP': '.dim',
        'GDAL-HFA-WRITER': '.img',
        'GDAL-COG-WRITER': '.tif',
        'HDF5': '.h5',
        'GDAL-NITF-WRITER': '.ntf',
        'GDAL-SAGA-WRITER': '.sdat',
        'GDAL-ILWIS-WRITER': '.mpl',
        'JP2,JPG,PNG,BMP,GIF': '.jp2',
        'GDAL-PCIDSK-WRITER': '.pix'
    }

    def __init__(
        self,
        product: str | Path,
        outdir: str | Path,
        format: str = 'BEAM-DIMAP',
        gpt_path: Optional[str] = '/usr/local/snap/bin/gpt',
        memory: str = '64G',
        parallelism: Optional[int] = 16,
    ):
        """Initialize SNAP GPT processing engine.
        
        Args:
            product: Path to the input SAR product file or directory.
            outdir: Output directory where processed results will be saved.
            format: Output format for processed data. Defaults to 'BEAM-DIMAP'.
            gpt_path: Path to the SNAP GPT executable.
                Defaults to '/usr/local/snap/bin/gpt'.
            
        Raises:
            AssertionError: If product path doesn't exist, format is unsupported,
                or output directory is not specified.
        """
        self.prod_path = Path(product)
        if not self.prod_path.exists():
            raise FileNotFoundError(f'Product path does not exist: {self.prod_path}')
        
        self.name = self.prod_path.stem
        
        if format not in self.OUTPUT_FORMATS:
            raise ValueError(
                f'Unsupported format: {format}. '
                f'Supported formats: {", ".join(self.OUTPUT_FORMATS)}'
            )
        self.format = format
        
        if not outdir:
            raise ValueError('Output directory must be specified')
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        
        self.parallelism = parallelism
        self.memory = memory
        
        self.gpt_executable = self._get_gpt_executable(gpt_path)
        self.current_cmd: List[str] = []

    def _get_gpt_executable(self, gpt_path: Optional[str] = None) -> str:
        """Determine the correct GPT executable path.
        
        Args:
            gpt_path: Explicit path to GPT executable.
            
        Returns:
            Path to the GPT executable.
        """
        if gpt_path:
            return gpt_path
        
        if self.mode and self.mode in self.GPT_PATHS:
            return self.GPT_PATHS[self.mode]
        
        if os.name == 'posix':
            for path in [self.GPT_PATHS['MacOS'], self.GPT_PATHS['Ubuntu']]:
                if Path(path).exists():
                    return path
            return 'gpt'
        elif os.name == 'nt':
            return self.GPT_PATHS['Windows']
        else:
            return 'gpt'



    def _reset_command(self) -> None:
        """Reset the command list for a new GPT operation."""
        self.current_cmd = [
            self.gpt_executable,
            f'-q {self.parallelism}',
            f'-c {self.memory}',
            '-x',
            '-e',
            f'-Ssource={self.prod_path.as_posix()}'
        ]

    def _build_output_path(self, suffix: str, output_name: Optional[str] = None) -> Path:
        """Build the output path for a processing step.
        
        Args:
            suffix: Suffix to append to filename if output_name is not provided.
            output_name: Custom output filename (without extension).
                If None, uses '{self.name}_{suffix}' format.
        
        Returns:
            Complete output path with appropriate extension.
        """
        if output_name:
            base_name = self.outdir / output_name
        else:
            base_name = self.outdir / f'{self.name}_{suffix}'
        
        # TODO: verify this
        extension = self.EXTENSIONS_MAP.get(self.format, '')
        return base_name.with_suffix(extension)

    def _execute_command(self) -> bool:
        """Execute the currently built GPT command.
        
        Returns:
            True if command executed successfully, False otherwise.
        """
        cmd_str = ' '.join(self.current_cmd)
        print(f'Executing GPT command: {cmd_str}')
        
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
                print(f'GPT Output: {process.stdout}')
            if process.stderr:
                print(f'GPT Warnings: {process.stderr}')
            
            print('Command executed successfully!')
            return True
            
        except subprocess.TimeoutExpired:
            print('Error: GPT command timed out after 1 hour')
            return False
            
        except subprocess.CalledProcessError as e:
            print(f'Error executing GPT command: {cmd_str}')
            print(f'Return code: {e.returncode}')
            if e.stdout:
                print(f'Stdout: {e.stdout}')
            if e.stderr:
                print(f'Stderr: {e.stderr}')
            return False
            
        except FileNotFoundError:
            print(f"Error: GPT executable '{self.gpt_executable}' not found!")
            print('Ensure SNAP is installed and configured correctly.')
            return False
            
        except Exception as e:
            print(f'Unexpected error during GPT execution: {type(e).__name__}: {e}')
            return False

    def _call(self, suffix: str, output_name: Optional[str] = None) -> Optional[str]:
        """Finalize and execute the GPT command.
        
        Args:
            suffix: Suffix for auto-generated filename.
            output_name: Custom output filename (without extension).
                Overrides auto-generated name.
        
        Returns:
            Path to output file if successful, None otherwise.
        """
        output_path = self._build_output_path(suffix, output_name)
        self.current_cmd.extend([
            f'-t {output_path.as_posix()}',
            f'-f {self.format}'
        ])

        if self._execute_command():
            self.prod_path = output_path
            return output_path.as_posix()
        return None


    # =============== OPERATORS =================

    def import_vector(
        self,
        vector_data: str | Path,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Import vector data into the product.
        
        Args:
            vector_data: Path to the shapefile or vector data.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to output product with imported vector, or None if failed.
        """
        vector_path = Path(vector_data)
        
        if not vector_path.exists():
            print(f'Shapefile not found: {vector_path}')
            print('Downloading from Zenodo...')
            
            zenodo_url = 'https://zenodo.org/api/records/6992586/files-archive'
            download_dir = Path.cwd() / 'zenodo_download'
            download_dir.mkdir(parents=True, exist_ok=True)
            archive_path = download_dir / 'zenodo_archive.zip'
            
            try:
                request.urlretrieve(zenodo_url, archive_path)
                print(f'Downloaded archive to: {archive_path}')
                
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(download_dir)
                print(f'Extracted archive to: {download_dir}')
                
                shp_files = list(download_dir.rglob('*.shp'))
                if shp_files:
                    vector_path = shp_files[0]
                    print(f'Using shapefile: {vector_path}')
                else:
                    raise FileNotFoundError('No shapefile found in downloaded archive')
                
                archive_path.unlink()
                
            except Exception as e:
                print(f'Error downloading or extracting shapefile: {e}')
                return None
        
        self._reset_command()
        self.current_cmd.append(
            f'Import-Vector -PseparateShapes=false '
            f'-PvectorFile={vector_path.as_posix()}'
        )
        return self._call(suffix='SHP', output_name=output_name)

    def land_mask(
        self,
        shoreline_extension: int = 300,
        geometry_name: str = 'Buff_750',
        use_srtm: bool = True,
        invert_geometry: bool = True,
        land_mask: bool = False,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Apply Land-Sea Masking using a predefined XML graph structure.
        
        Args:
            shoreline_extension: Distance to extend shoreline in meters.
            geometry_name: Name of the geometry to use for masking.
            use_srtm: Use SRTM DEM for land/sea determination.
            invert_geometry: Invert the geometry mask.
            land_mask: Mask land (True) or sea (False) areas.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to masked output product, or None if failed.
        """
        self._reset_command()
        suffix = 'LM'
        output_path = self._build_output_path(suffix, output_name)
        xml_path = self.outdir / f'{self.name}_landmask_graph.xml'

        if not hasattr(self, 'prod_type'):
            try:
                self.prod_type = _identify_product_type(self.prod_path.name)
                print(f'Inferred product type: {self.prod_type}')
            except Exception as e:
                print(f'Warning: Could not determine product type: {e}')
                self.prod_type = None

        source_band_map = {
            'COSMO-SkyMed': 'Intensity_null',
            'Sentinel-1': 'Intensity_VH'
        }
        source_band = source_band_map.get(self.prod_type or '', 'Intensity_VH')
        
        if self.prod_type not in source_band_map:
            print(
                f"Warning: Product type is '{self.prod_type}'. "
                f"Using default source band 'Intensity_VH'."
            )

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
            xml_path.write_text(graph_xml, encoding='utf-8')
            self.current_cmd = [self.gpt_executable, xml_path.as_posix()]

            if self._execute_command():
                self.prod_path = output_path
                xml_path.unlink(missing_ok=True)
                return output_path.as_posix()
            return None

        except Exception as e:
            print(f'Error generating LandMask XML graph: {e}')
            xml_path.unlink(missing_ok=True)
            return None

    def terrain_mask(
        self,
        dem_name: str = 'SRTM 3Sec',
        dem_resampling_method: str = 'NEAREST_NEIGHBOUR',
        external_dem_file: Optional[str | Path] = None,
        external_dem_no_data_value: float = 0.0,
        threshold_in_meter: float = 40.0,
        window_size_str: str = '15x15',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Generate a terrain mask using a DEM.
        
        Args:
            dem_name: The digital elevation model to use.
            dem_resampling_method: DEM resampling method.
            external_dem_file: Path to an external DEM file.
            external_dem_no_data_value: No data value for the external DEM.
            threshold_in_meter: Elevation threshold for mask detection.
            window_size_str: Size of the window used for filtering.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to the terrain mask product, or None if failed.
        """
        self._reset_command()
        cmd_params = [
            f'-PdemName="{dem_name}"',
            f'-PdemResamplingMethod={dem_resampling_method}',
            f'-PexternalDEMNoDataValue={external_dem_no_data_value}',
            f'-PthresholdInMeter={threshold_in_meter}',
            f'-PwindowSizeStr="{window_size_str}"'
        ]

        if external_dem_file:
            cmd_params.append(f'-PexternalDEMFile={Path(external_dem_file).as_posix()}')

        self.current_cmd.append(f'Terrain-Mask {" ".join(cmd_params)}')
        return self._call(suffix='TMSK', output_name=output_name)

    def terrain_flattening(
        self,
        additional_overlap: float = 0.1,
        dem_name: str = 'SRTM 1Sec HGT',
        dem_resampling_method: str = 'BILINEAR_INTERPOLATION',
        external_dem_apply_egm: bool = False,
        external_dem_file: Optional[str | Path] = None,
        external_dem_no_data_value: float = 0.0,
        nodata_value_at_sea: bool = True,
        output_sigma0: bool = False,
        output_simulated_image: bool = False,
        oversampling_multiple: float = 1.0,
        source_bands: Optional[List[str]] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Perform terrain flattening with configurable DEM and output options."""
        self._reset_command()

        cmd_params = [
            f'-PadditionalOverlap={additional_overlap}',
            f'-PdemName="{dem_name}"',
            f'-PdemResamplingMethod={dem_resampling_method}',
            f'-PexternalDEMApplyEGM={str(external_dem_apply_egm).lower()}',
            f'-PexternalDEMNoDataValue={external_dem_no_data_value}',
            f'-PnodataValueAtSea={str(nodata_value_at_sea).lower()}',
            f'-PoutputSigma0={str(output_sigma0).lower()}',
            f'-PoutputSimulatedImage={str(output_simulated_image).lower()}',
            f'-PoversamplingMultiple={oversampling_multiple}'
        ]

        if external_dem_file:
            cmd_params.append(f'-PexternalDEMFile={Path(external_dem_file).as_posix()}')

        if source_bands:
            cmd_params.append(f'-PsourceBands={",".join(source_bands)}')

        self.current_cmd.append(f'Terrain-Flattening {" ".join(cmd_params)}')
        return self._call(suffix='TFLAT', output_name=output_name)

    def temporal_percentile(
        self,
        source_products: Optional[List[str | Path]] = None,
        source_product_paths: Optional[List[str]] = None,
        band_math_expression: Optional[str] = None,
        source_band_name: Optional[str] = None,
        crs: str = 'EPSG:4326',
        west_bound: float = -15.0,
        north_bound: float = 75.0,
        east_bound: float = 30.0,
        south_bound: float = 35.0,
        pixel_size_x: float = 0.05,
        pixel_size_y: float = 0.05,
        resampling: str = 'Nearest',
        percentiles: Optional[List[int]] = None,
        valid_pixel_expression: str = 'true',
        gap_filling_method: str = 'gapFillingLinearInterpolation',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        start_value_fallback: float = 0.0,
        end_value_fallback: float = 0.0,
        keep_intermediate_time_series_product: bool = True,
        time_series_output_dir: Optional[str | Path] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compute percentile statistics over time series products."""
        if not band_math_expression and not source_band_name:
            raise ValueError('Either band_math_expression or source_band_name must be provided')

        self._reset_command()

        products = source_products or [self.prod_path]
        if not products:
            raise ValueError('source_products must contain at least one product path')

        source_products_str = ','.join(Path(p).as_posix() for p in products)
        for i, cmd_part in enumerate(self.current_cmd):
            if cmd_part.startswith('-Ssource='):
                self.current_cmd[i] = f'-SsourceProducts={source_products_str}'
                break

        cmd_params = [
            f'-Pcrs="{crs}"',
            f'-PwestBound={west_bound}',
            f'-PnorthBound={north_bound}',
            f'-PeastBound={east_bound}',
            f'-PsouthBound={south_bound}',
            f'-PpixelSizeX={pixel_size_x}',
            f'-PpixelSizeY={pixel_size_y}',
            f'-Presampling={resampling}',
            f'-PkeepIntermediateTimeSeriesProduct={str(keep_intermediate_time_series_product).lower()}',
            f'-PvalidPixelExpression="{valid_pixel_expression}"',
            f'-PgapFillingMethod={gap_filling_method}',
            f'-PstartValueFallback={start_value_fallback}',
            f'-PendValueFallback={end_value_fallback}'
        ]

        if start_date:
            cmd_params.append(f'-PstartDate="{start_date}"')
        if end_date:
            cmd_params.append(f'-PendDate="{end_date}"')

        if percentiles:
            percentiles_str = ','.join(str(p) for p in percentiles)
        else:
            percentiles_str = '90'
        cmd_params.append(f'-Ppercentiles={percentiles_str}')

        if source_product_paths:
            cmd_params.append(f'-PsourceProductPaths={",".join(source_product_paths)}')

        if time_series_output_dir:
            cmd_params.append(f'-PtimeSeriesOutputDir={Path(time_series_output_dir).as_posix()}')

        if band_math_expression:
            cmd_params.append(f'-PbandMathsExpression="{band_math_expression}"')
        if source_band_name:
            cmd_params.append(f'-PsourceBandName="{source_band_name}"')

        self.current_cmd.append(f'TemporalPercentile {" ".join(cmd_params)}')
        return self._call(suffix='TMPP', output_name=output_name)

    def calibration(
        self,
        pols: Optional[List[str]] = None,
        output_complex: bool = True,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Apply radiometric calibration.
        
        Args:
            pols: Polarizations to calibrate. If None, all polarizations are used.
            output_complex: Output complex values.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to calibrated product, or None if failed.
        """
        self._reset_command()
        
        if pols is not None:
            pol_str = ','.join(pols)
            self.current_cmd.append(
                f'Calibration -PoutputImageInComplex={str(output_complex).lower()} '
                f'-PselectedPolarisations={pol_str}'
            )
        else:
            self.current_cmd.append(
                f'Calibration -PoutputImageInComplex={str(output_complex).lower()}'
            )
        
        return self._call(suffix='CAL', output_name=output_name)

    def thermal_noise_removal(
        self,
        pols: Optional[List[str]] = None,
        output_noise: bool = False,
        reintroduce_thermal_noise: bool = False,
        remove_thermal_noise: bool = True,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Remove thermal noise from products.
        
        Args:
            pols: Polarizations to process. If None, all polarizations are used.
            output_noise: Output the noise band.
            reintroduce_thermal_noise: Re-introduce thermal noise.
            remove_thermal_noise: Remove thermal noise.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to output product, or None if failed.
        """
        self._reset_command()
        for i, cmd_part in enumerate(self.current_cmd):
            if cmd_part.startswith('-Ssource='):
                self.current_cmd[i] = cmd_part.replace('-Ssource=', '-SsourceProduct=')
                break
        
        cmd_params = [
            f'-PoutputNoise={str(output_noise).lower()}',
            f'-PreIntroduceThermalNoise={str(reintroduce_thermal_noise).lower()}',
            f'-PremoveThermalNoise={str(remove_thermal_noise).lower()}'
        ]
        
        if pols:
            cmd_params.append(f'-PselectedPolarisations={",".join(pols)}')
        
        self.current_cmd.append(f'ThermalNoiseRemoval {" ".join(cmd_params)}')
        return self._call(suffix='TNR', output_name=output_name)

    def deburst(
        self,
        pols: Optional[List[str]] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Apply TOPSAR Debursting.
        
        Args:
            pols: Polarizations to deburst. If None, all channels are debursted.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to deburst product, or None if failed.
        """
        self._reset_command()
        
        if pols is not None:
            pol_str = ','.join(pols)
            self.current_cmd.append(f'TOPSAR-Deburst -PselectedPolarisations={pol_str}')
        else:
            self.current_cmd.append('TOPSAR-Deburst')
        
        return self._call(suffix='DEB', output_name=output_name)

    def multilook(
        self,
        n_rg_looks: int,
        n_az_looks: int,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Apply Multilooking.
        
        Args:
            n_rg_looks: Number of range looks.
            n_az_looks: Number of azimuth looks.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to multilooked product, or None if failed.
        """
        self._reset_command()
        self.current_cmd.append(
            f'Multilook -PnRgLooks={n_rg_looks} -PnAzLooks={n_az_looks}'
        )
        return self._call(suffix='ML', output_name=output_name)

    def adaptive_thresholding(
        self,
        background_window_m: float = 800,
        guard_window_m: float = 500,
        target_window_m: float = 50,
        pfa: float = 6.5,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Apply Adaptive Thresholding for object detection.
        
        Args:
            background_window_m: Background window size in meters.
            guard_window_m: Guard window size in meters.
            target_window_m: Target window size in meters.
            pfa: Probability of false alarm.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to thresholded product, or None if failed.
        """
        self._reset_command()
        self.current_cmd.append(
            f'AdaptiveThresholding '
            f'-PbackgroundWindowSizeInMeter={background_window_m} '
            f'-PguardWindowSizeInMeter={guard_window_m} '
            f'-Ppfa={pfa} '
            f'-PtargetWindowSizeInMeter={target_window_m}'
        )
        return self._call(suffix='AT', output_name=output_name)

    def object_discrimination(
        self,
        min_target_m: float,
        max_target_m: float,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Discriminate objects based on size.
        
        Args:
            min_target_m: Minimum target size in meters.
            max_target_m: Maximum target size in meters.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to discriminated product, or None if failed.
        """
        self._reset_command()
        self.current_cmd.append(
            f'Object-Discrimination '
            f'-PminTargetSizeInMeter={min_target_m} '
            f'-PmaxTargetSizeInMeter={max_target_m}'
        )
        return self._call(suffix='OD', output_name=output_name)

    def subset(
        self,
        source_bands: Optional[List[str]] = None,
        tie_point_grids: Optional[List[str]] = None,
        region: Optional[str] = None,
        reference_band: Optional[str] = None,
        geo_region: Optional[str] = None,
        sub_sampling_x: int = 1,
        sub_sampling_y: int = 1,
        full_swath: bool = False,
        vector_file: Optional[str | Path] = None,
        polygon_region: Optional[str] = None,
        copy_metadata: bool = False,
        suffix: str = 'SUB',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Create a spatial and/or spectral subset of a data product.
        
        This method allows extraction of specific bands, spatial regions, and subsetting
        based on pixel coordinates, geographical coordinates, or vector polygons.
        
        Args:
            source_bands: List of source bands to include. If None, all bands are included.
            tie_point_grids: List of tie-point grid names to include.
            region: Subset region in pixel coordinates.
                Format: 'x,y,width,height' (e.g., '100,200,500,500').
            reference_band: Band used to indicate the pixel coordinates.
            geo_region: Subset region in geographical coordinates using WKT-format.
                e.g., 'POLYGON((lon1 lat1, lon2 lat2, ..., lon1 lat1))'.
                This parameter has precedence over 'region'.
            sub_sampling_x: Pixel sub-sampling step in X (horizontal direction).
            sub_sampling_y: Pixel sub-sampling step in Y (vertical direction).
            full_swath: Forces the operator to extend the subset region to the full swath.
            vector_file: File from which the polygon is read.
            polygon_region: Subset region in geographical coordinates using WKT-format.
            copy_metadata: Whether to copy the metadata of the source product.
            suffix: Suffix for the output filename.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to the subset output product, or None if failed.
        
        Examples:
            # Subset by pixel region
            gpt.subset(region='100,200,500,500', source_bands=['Amplitude_VV'])
            
            # Subset by geographic region
            gpt.subset(
                geo_region='POLYGON((10.0 53.0, 11.0 53.0, 11.0 54.0, 10.0 54.0, 10.0 53.0))',
                source_bands=['Intensity_VH', 'Intensity_VV']
            )
            
            # Subset using vector file
            gpt.subset(vector_file='aoi.shp', source_bands=['Sigma0_VV'])
        """
        self._reset_command()
        
        cmd_params = []
        
        if source_bands:
            cmd_params.append(f'-PsourceBands={",".join(source_bands)}')
        
        if tie_point_grids:
            cmd_params.append(f'-PtiePointGrids={",".join(tie_point_grids)}')
        
        if region:
            cmd_params.append(f'-Pregion={region}')
        
        if reference_band:
            cmd_params.append(f'-PreferenceBand={reference_band}')
        
        if geo_region:
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
            cmd_params.append(f"-PpolygonRegion='{polygon_region}'")
        
        cmd_params.append(f'-PcopyMetadata={str(copy_metadata).lower()}')
        
        self.current_cmd.append(f'Subset {" ".join(cmd_params)}')
        return self._call(suffix=suffix, output_name=output_name)

    def aatsr_sst(
        self,
        dual: bool = True,
        dual_coefficients_file: str = 'AVERAGE_POLAR_DUAL_VIEW',
        dual_mask_expression: str = (
            '!cloud_flags_nadir.LAND and !cloud_flags_nadir.CLOUDY and '
            '!cloud_flags_nadir.SUN_GLINT and !cloud_flags_fward.LAND and '
            '!cloud_flags_fward.CLOUDY and !cloud_flags_fward.SUN_GLINT'
        ),
        invalid_sst_value: float = -999.0,
        nadir: bool = True,
        nadir_coefficients_file: str = 'AVERAGE_POLAR_SINGLE_VIEW',
        nadir_mask_expression: str = (
            '!cloud_flags_nadir.LAND and !cloud_flags_nadir.CLOUDY and '
            '!cloud_flags_nadir.SUN_GLINT'
        ),
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compute sea surface temperature (SST) from (A)ATSR products.
        
        This method processes ATSR (Along Track Scanning Radiometer) data to derive
        sea surface temperature using both dual-view and nadir-view algorithms.
        
        Args:
            dual: Enable dual-view SST processing.
            dual_coefficients_file: Coefficients file for dual-view processing.
            dual_mask_expression: Mask expression for dual-view processing.
            invalid_sst_value: Value assigned to invalid SST pixels.
            nadir: Enable nadir-view SST processing.
            nadir_coefficients_file: Coefficients file for nadir-view processing.
            nadir_mask_expression: Mask expression for nadir-view processing.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to SST output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-Pdual={str(dual).lower()}',
            f'-PdualCoefficientsFile={dual_coefficients_file}',
            f'-PdualMaskExpression="{dual_mask_expression}"',
            f'-PinvalidSstValue={invalid_sst_value}',
            f'-Pnadir={str(nadir).lower()}',
            f'-PnadirCoefficientsFile={nadir_coefficients_file}',
            f'-PnadirMaskExpression="{nadir_mask_expression}"'
        ]
        
        self.current_cmd.append(f'Aatsr.SST {" ".join(cmd_params)}')
        return self._call(suffix='SST', output_name=output_name)

    def aatsr_ungrid(
        self,
        l1b_characterisation_file: Optional[str | Path] = None,
        corner_reference_flag: bool = True,
        topographic_flag: bool = False,
        topography_homogenity: float = 0.05,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Ungrid (A)ATSR L1B products and extract geolocation and pixel field of view data.
        
        This method processes ATSR L1B products to extract geolocation information and
        pixel field-of-view data, with optional topographic corrections.
        
        Args:
            l1b_characterisation_file: L1B characterisation file needed to specify
                first forward pixel and first nadir pixel.
            corner_reference_flag: Choose pixel coordinate reference point for output file.
                True for corner (default), False for centre.
            topographic_flag: Apply topographic corrections to tie points.
            topography_homogenity: Distance (image coordinates) pixel can be from
                tie-point to have topographic correction applied.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to ungridded output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PcornerReferenceFlag={str(corner_reference_flag).lower()}',
            f'-PtopographicFlag={str(topographic_flag).lower()}',
            f'-PtopographyHomogenity={topography_homogenity}'
        ]
        
        if l1b_characterisation_file:
            char_file_path = Path(l1b_characterisation_file)
            cmd_params.append(f'-PL1BCharacterisationFile={char_file_path.as_posix()}')
        
        self.current_cmd.append(f'AATSR.Ungrid {" ".join(cmd_params)}')
        return self._call(suffix='UNGRID', output_name=output_name)

    def wind_field_estimation(
        self,
        source_bands: Optional[List[str]] = None,
        window_size_in_km: float = 20.0,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Estimate wind speed and direction from SAR imagery.
        
        This method analyzes SAR backscatter patterns to estimate surface wind
        fields over water bodies, particularly useful for oceanographic applications.
        
        Args:
            source_bands: List of source bands to use for wind estimation.
                If None, all available bands are used.
            window_size_in_km: Window size for wind estimation in kilometers.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to wind field output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [f'-PwindowSizeInKm={window_size_in_km}']
        
        if source_bands:
            cmd_params.insert(0, f'-PsourceBands={",".join(source_bands)}')
        
        self.current_cmd.append(f'Wind-Field-Estimation {" ".join(cmd_params)}')
        return self._call(suffix='WIND', output_name=output_name)

    def wdvi(
        self,
        red_source_band: Optional[str] = None,
        nir_source_band: Optional[str] = None,
        red_factor: float = 1.0,
        nir_factor: float = 1.0,
        slope_soil_line: float = 1.5,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compute Weighted Difference Vegetation Index (WDVI).
        
        This method retrieves isovegetation lines parallel to the soil line,
        where the soil line has an arbitrary slope and passes through the origin.
        WDVI is particularly useful for vegetation monitoring and analysis.
        
        Args:
            red_source_band: The red band for WDVI computation.
                If None, operator will try to find the best fitting band.
            nir_source_band: The near-infrared band for WDVI computation.
                If None, operator will try to find the best fitting band.
            red_factor: Multiplication factor for red band values.
            nir_factor: Multiplication factor for NIR band values.
            slope_soil_line: Slope of the soil line passing through origin.
            resample_type: Resample method if bands differ in size.
                Must be one of 'None', 'Lowest resolution', 'Highest resolution'.
            upsampling: Interpolation method for upsampling to finer resolution.
                Must be one of 'Nearest', 'Bilinear', 'Bicubic'.
            downsampling: Aggregation method for downsampling to coarser resolution.
                Must be one of 'First', 'Min', 'Max', 'Mean', 'Median'.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to WDVI output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PredFactor={red_factor}',
            f'-PnirFactor={nir_factor}',
            f'-PslopeSoilLine={slope_soil_line}'
        ]
        
        if red_source_band:
            cmd_params.append(f'-PredSourceBand={red_source_band}')
        
        if nir_source_band:
            cmd_params.append(f'-PnirSourceBand={nir_source_band}')
        
        self.current_cmd.append(f'WdviOp {" ".join(cmd_params)}')
        return self._call(suffix='WDVI', output_name=output_name)

    def warp(
        self,
        rms_threshold: float = 0.05,
        warp_polynomial_order: int = 2,
        interpolation_method: str = 'Cubic convolution (6 points)',
        dem_refinement: bool = False,
        dem_name: str = 'SRTM 3Sec',
        exclude_master: bool = False,
        open_residuals_file: bool = False,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Create warp function and get co-registered images.
        
        This method performs geometric co-registration of images using warp
        polynomial functions, enabling precise alignment of multi-temporal
        or multi-sensor SAR imagery.
        
        Args:
            rms_threshold: Confidence level for outlier detection procedure.
                Lower value accepts more outliers. Must be one of 0.001, 0.05, 0.1, 0.5, 1.0.
            warp_polynomial_order: The order of WARP polynomial function.
                Must be one of 1, 2, or 3.
            interpolation_method: Interpolation method for resampling.
                Must be one of 'Nearest-neighbor interpolation', 'Bilinear interpolation',
                'Bicubic interpolation', 'Bicubic2 interpolation', 'Linear interpolation',
                'Cubic convolution (4 points)', 'Cubic convolution (6 points)',
                'Truncated sinc (6 points)', 'Truncated sinc (8 points)',
                'Truncated sinc (16 points)'.
            dem_refinement: Refine estimated offsets using a-priori DEM.
            dem_name: The digital elevation model to use.
            exclude_master: Whether to exclude the master image from output.
            open_residuals_file: Show the residuals file in a text viewer.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to warped output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PrmsThreshold={rms_threshold}',
            f'-PwarpPolynomialOrder={warp_polynomial_order}',
            f'-PinterpolationMethod="{interpolation_method}"',
            f'-PdemRefinement={str(dem_refinement).lower()}',
            f'-PdemName="{dem_name}"',
            f'-PexcludeMaster={str(exclude_master).lower()}',
            f'-PopenResidualsFile={str(open_residuals_file).lower()}'
        ]
        
        self.current_cmd.append(f'Warp {" ".join(cmd_params)}')
        return self._call(suffix='WARP', output_name=output_name)

    def update_geo_reference(
        self,
        source_bands: Optional[List[str]] = None,
        dem_name: str = 'SRTM 3Sec',
        dem_resampling_method: str = 'BICUBIC_INTERPOLATION',
        external_dem_file: Optional[str | Path] = None,
        external_dem_no_data_value: float = 0.0,
        re_grid_method: bool = False,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Update geo-reference information in the product.
        
        This method updates the geolocation information of SAR products using
        a digital elevation model, improving the accuracy of geographic coordinates.
        
        Args:
            source_bands: List of source bands to process. If None, all bands are processed.
            dem_name: The digital elevation model to use.
                Must be one of 'ACE', 'ASTER 1sec GDEM', 'GETASSE30',
                'SRTM 1Sec HGT', 'SRTM 3Sec'.
            dem_resampling_method: DEM resampling method.
            external_dem_file: Path to external DEM file.
            external_dem_no_data_value: No data value for external DEM.
            re_grid_method: Apply re-gridding method.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to output product with updated geo-reference, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PdemName="{dem_name}"',
            f'-PdemResamplingMethod={dem_resampling_method}',
            f'-PexternalDEMNoDataValue={external_dem_no_data_value}',
            f'-PreGridMethod={str(re_grid_method).lower()}'
        ]
        
        if source_bands:
            cmd_params.insert(0, f'-PsourceBands={",".join(source_bands)}')
        
        if external_dem_file:
            cmd_params.append(f'-PexternalDEMFile={Path(external_dem_file).as_posix()}')
        
        self.current_cmd.append(f'Update-Geo-Reference {" ".join(cmd_params)}')
        return self._call(suffix='UGR', output_name=output_name)

    def add_elevation(
        self,
        dem_name: str = 'SRTM 3Sec',
        dem_resampling_method: str = 'BICUBIC_INTERPOLATION',
        elevation_band_name: str = 'elevation',
        external_dem_file: Optional[str | Path] = None,
        external_dem_no_data_value: float = 0.0,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Add a DEM elevation band to the product.
        
        This method creates an elevation band from a specified DEM source and
        appends it to the product for downstream processing.
        
        Args:
            dem_name: The digital elevation model to use.
                Must be one of 'ACE', 'ASTER 1sec GDEM', 'GETASSE30',
                'SRTM 1Sec HGT', 'SRTM 3Sec'.
            dem_resampling_method: DEM resampling method.
            elevation_band_name: Name of the elevation band.
            external_dem_file: Path to external DEM file.
            external_dem_no_data_value: No data value for external DEM.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to output product with elevation band, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PdemName="{dem_name}"',
            f'-PdemResamplingMethod={dem_resampling_method}',
            f'-PexternalDEMNoDataValue={external_dem_no_data_value}',
            f'-PelevationBandName="{elevation_band_name}"'
        ]
        
        if external_dem_file:
            cmd_params.append(f'-PexternalDEMFile={Path(external_dem_file).as_posix()}')
        
        self.current_cmd.append(f'AddElevation {" ".join(cmd_params)}')
        return self._call(suffix='ELEV', output_name=output_name)

    def three_pass_dinsar(
        self,
        source_products: Optional[List[str | Path]] = None,
        orbit_degree: int = 3,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Perform three-pass differential interferometry.
        
        This method runs the Three-passDInSAR operator using multiple source products.
        
        Args:
            source_products: Source product paths. If None, uses the current product.
            orbit_degree: Degree of orbit interpolation polynomial.
                Valid interval is (1, 10].
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to output product, or None if failed.
        """
        self._reset_command()
        
        products = source_products or [self.prod_path]
        if not products:
            raise ValueError('source_products must contain at least one product path')
        source_list = ','.join(Path(p).as_posix() for p in products)
        for i, cmd_part in enumerate(self.current_cmd):
            if cmd_part.startswith('-Ssource='):
                self.current_cmd[i] = f'-SsourceProducts={source_list}'
                break
        
        self.current_cmd.append(f'Three-passDInSAR -PorbitDegree={orbit_degree}')
        return self._call(suffix='3PASS', output_name=output_name)

    def unmix(
        self,
        source_bands: Optional[List[str]] = None,
        endmember_file: Optional[str | Path] = None,
        unmixing_model_name: str = 'Constrained LSU',
        abundance_band_name_suffix: str = '_abundance',
        error_band_name_suffix: str = '_error',
        compute_error_bands: bool = False,
        min_bandwidth: float = 10.0,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Perform linear spectral unmixing.
        
        This method decomposes mixed spectral signatures into constituent endmembers,
        useful for analyzing surface composition from multi-spectral or hyperspectral data.
        
        Args:
            source_bands: List of spectral bands providing the source spectrum.
                If None, all bands are used.
            endmember_file: Text file containing endmembers in a table.
                Wavelengths must be given in nanometers.
            unmixing_model_name: The unmixing model to use.
                Must be one of 'Unconstrained LSU', 'Constrained LSU',
                'Fully Constrained LSU'.
            abundance_band_name_suffix: Suffix for generated abundance band names
                (name = endmember + suffix). Must match pattern '[a-zA-Z_0-9]*'.
            error_band_name_suffix: Suffix for generated error band names
                (name = source + suffix). Must match pattern '[a-zA-Z_0-9]*'.
            compute_error_bands: Generate error bands for all source bands.
            min_bandwidth: Minimum spectral bandwidth for endmember wavelength
                matching in nanometers. Must be greater than 0.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to unmixed output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PunmixingModelName="{unmixing_model_name}"',
            f'-PabundanceBandNameSuffix={abundance_band_name_suffix}',
            f'-PerrorBandNameSuffix={error_band_name_suffix}',
            f'-PcomputeErrorBands={str(compute_error_bands).lower()}',
            f'-PminBandwidth={min_bandwidth}'
        ]
        
        if source_bands:
            cmd_params.insert(0, f'-PsourceBands={",".join(source_bands)}')
        
        if endmember_file:
            cmd_params.append(f'-PendmemberFile={Path(endmember_file).as_posix()}')
        
        self.current_cmd.append(f'Unmix {" ".join(cmd_params)}')
        return self._call(suffix='UNMIX', output_name=output_name)

    def undersample(
        self,
        source_bands: Optional[List[str]] = None,
        method: str = 'LowPass Filtering',
        filter_size: str = '3x3',
        sub_sampling_x: int = 2,
        sub_sampling_y: int = 2,
        output_image_by: str = 'Ratio',
        target_image_height: int = 1000,
        target_image_width: int = 1000,
        width_ratio: float = 0.5,
        height_ratio: float = 0.5,
        range_spacing: float = 12.5,
        azimuth_spacing: float = 12.5,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Undersample the dataset by reducing spatial resolution.
        
        This method reduces the spatial dimensions of SAR imagery through
        sub-sampling or low-pass filtering, useful for reducing data volume
        or matching different resolution products.
        
        Args:
            source_bands: List of source bands to undersample. If None, all bands are processed.
            method: Undersampling method to use.
                Must be one of 'Sub-Sampling', 'LowPass Filtering'.
            filter_size: Filter size for low-pass filtering.
                Must be one of '3x3', '5x5', '7x7'.
            sub_sampling_x: Sub-sampling factor in X direction.
            sub_sampling_y: Sub-sampling factor in Y direction.
            output_image_by: Method to determine output image size.
                Must be one of 'Image Size', 'Ratio', 'Pixel Spacing'.
            target_image_height: Row dimension of output image (pixels).
            target_image_width: Column dimension of output image (pixels).
            width_ratio: Width ratio of output/input images.
            height_ratio: Height ratio of output/input images.
            range_spacing: Range pixel spacing in meters.
            azimuth_spacing: Azimuth pixel spacing in meters.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to undersampled output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-Pmethod="{method}"',
            f'-PfilterSize={filter_size}',
            f'-PsubSamplingX={sub_sampling_x}',
            f'-PsubSamplingY={sub_sampling_y}',
            f'-PoutputImageBy="{output_image_by}"',
            f'-PtargetImageHeight={target_image_height}',
            f'-PtargetImageWidth={target_image_width}',
            f'-PwidthRatio={width_ratio}',
            f'-PheightRatio={height_ratio}',
            f'-PrangeSpacing={range_spacing}',
            f'-PazimuthSpacing={azimuth_spacing}'
        ]
        
        if source_bands:
            cmd_params.insert(0, f'-PsourceBands={",".join(source_bands)}')
        
        self.current_cmd.append(f'Undersample {" ".join(cmd_params)}')
        return self._call(suffix='UNDER', output_name=output_name)

    def tsavi(
        self,
        red_source_band: Optional[str] = None,
        nir_source_band: Optional[str] = None,
        red_factor: float = 1.0,
        nir_factor: float = 1.0,
        slope: float = 0.5,
        intercept: float = 0.5,
        adjustment: float = 0.08,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compute Transformed Soil Adjusted Vegetation Index (TSAVI).
        
        This method retrieves TSAVI which minimizes soil background effects
        using soil line parameters. It's particularly effective for areas
        with partial vegetation cover.
        
        Args:
            red_source_band: The red band for TSAVI computation.
                If None, operator will try to find the best fitting band.
            nir_source_band: The near-infrared band for TSAVI computation.
                If None, operator will try to find the best fitting band.
            red_factor: Multiplication factor for red band values.
            nir_factor: Multiplication factor for NIR band values.
            slope: The soil line slope.
            intercept: The soil line intercept.
            adjustment: Adjustment factor to minimize soil background.
            resample_type: Resample method if bands differ in size.
                Must be one of 'None', 'Lowest resolution', 'Highest resolution'.
            upsampling: Interpolation method for upsampling to finer resolution.
                Must be one of 'Nearest', 'Bilinear', 'Bicubic'.
            downsampling: Aggregation method for downsampling to coarser resolution.
                Must be one of 'First', 'Min', 'Max', 'Mean', 'Median'.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to TSAVI output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PredFactor={red_factor}',
            f'-PnirFactor={nir_factor}',
            f'-Pslope={slope}',
            f'-Pintercept={intercept}',
            f'-Padjustment={adjustment}'
        ]
        
        if red_source_band:
            cmd_params.append(f'-PredSourceBand={red_source_band}')
        
        if nir_source_band:
            cmd_params.append(f'-PnirSourceBand={nir_source_band}')
        
        self.current_cmd.append(f'TsaviOp {" ".join(cmd_params)}')
        return self._call(suffix='TSAVI', output_name=output_name)

    def tndvi(
        self,
        red_source_band: Optional[str] = None,
        nir_source_band: Optional[str] = None,
        red_factor: float = 1.0,
        nir_factor: float = 1.0,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compute Transformed Normalized Difference Vegetation Index (TNDVI).
        
        This method retrieves isovegetation lines parallel to the soil line,
        providing a transformed version of NDVI that accounts for soil background effects.
        
        Args:
            red_source_band: The red band for TNDVI computation.
                If None, operator will try to find the best fitting band.
            nir_source_band: The near-infrared band for TNDVI computation.
                If None, operator will try to find the best fitting band.
            red_factor: Multiplication factor for red band values.
            nir_factor: Multiplication factor for NIR band values.
            resample_type: Resample method if bands differ in size.
                Must be one of 'None', 'Lowest resolution', 'Highest resolution'.
            upsampling: Interpolation method for upsampling to finer resolution.
                Must be one of 'Nearest', 'Bilinear', 'Bicubic'.
            downsampling: Aggregation method for downsampling to coarser resolution.
                Must be one of 'First', 'Min', 'Max', 'Mean', 'Median'.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to TNDVI output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PredFactor={red_factor}',
            f'-PnirFactor={nir_factor}'
        ]
        
        if red_source_band:
            cmd_params.append(f'-PredSourceBand={red_source_band}')
        
        if nir_source_band:
            cmd_params.append(f'-PnirSourceBand={nir_source_band}')
        
        self.current_cmd.append(f'TndviOp {" ".join(cmd_params)}')
        return self._call(suffix='TNDVI', output_name=output_name)

    def topsar_split(
        self,
        subswath: Optional[str] = None,
        selected_polarisations: Optional[List[str]] = None,
        first_burst_index: int = 1,
        last_burst_index: int = 9999,
        wkt_aoi: Optional[str] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Create a new product with only the selected subswath.
        
        This method splits Sentinel-1 TOPS data by subswath and/or burst,
        useful for reducing processing time and focusing on specific areas of interest.
        
        Args:
            subswath: The subswath to select (e.g., 'IW1', 'IW2', 'IW3').
                If None, all subswaths are included.
            selected_polarisations: List of polarisations to include.
                If None, all polarisations are included.
            first_burst_index: The first burst index to include (1-based).
                Must be >= 1.
            last_burst_index: The last burst index to include (1-based).
                Must be >= 1.
            wkt_aoi: WKT polygon to be used for selecting bursts.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to split output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PfirstBurstIndex={first_burst_index}',
            f'-PlastBurstIndex={last_burst_index}'
        ]
        
        if subswath:
            cmd_params.insert(0, f'-Psubswath={subswath}')
        
        if selected_polarisations:
            cmd_params.append(f'-PselectedPolarisations={",".join(selected_polarisations)}')
        
        if wkt_aoi:
            cmd_params.append(f'-PwktAoi="{wkt_aoi}"')
        
        self.current_cmd.append(f'TOPSAR-Split {" ".join(cmd_params)}')
        return self._call(suffix='SPLIT', output_name=output_name)

    def topsar_merge(
        self,
        selected_polarisations: Optional[List[str]] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Merge subswaths of a Sentinel-1 TOPSAR product.
        
        This method merges multiple subswaths (e.g., IW1, IW2, IW3) into a single
        product, creating a seamless wide-swath image from split TOPS data.
        
        Args:
            selected_polarisations: List of polarisations to merge.
                If None, all polarisations are merged.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to merged output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = []
        
        if selected_polarisations:
            cmd_params.append(f'-PselectedPolarisations={",".join(selected_polarisations)}')
        
        self.current_cmd.append(f'TOPSAR-Merge {" ".join(cmd_params)}')
        return self._call(suffix='MERGE', output_name=output_name)

    def topsar_deramp_demod(
        self,
        output_deramp_demod_phase: bool = False,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Perform deramp and demodulation for TOPSAR burst co-registration.
        
        This method performs burst co-registration using orbit and DEM information,
        removing the azimuth phase ramp and demodulating the signal. This is essential
        for interferometric processing of Sentinel-1 TOPS data.
        
        Args:
            output_deramp_demod_phase: Output the deramp/demod phase.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to deramp/demod output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [f'-PoutputDerampDemodPhase={str(output_deramp_demod_phase).lower()}']
        
        self.current_cmd.append(f'TOPSAR-DerampDemod {" ".join(cmd_params)}')
        return self._call(suffix='DERAMP', output_name=output_name)

    def topo_phase_removal(
        self,
        orbit_degree: int = 3,
        dem_name: str = 'SRTM 3Sec',
        external_dem_file: Optional[str | Path] = None,
        external_dem_no_data_value: float = 0.0,
        tile_extension_percent: str = '100',
        output_topo_phase_band: bool = False,
        output_elevation_band: bool = False,
        output_lat_lon_bands: bool = False,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compute and subtract topographic phase from interferogram.
        
        This method removes the topographic phase contribution from an interferogram
        using a DEM, essential for differential interferometry to isolate deformation signals.
        
        Args:
            orbit_degree: Degree of orbit interpolation polynomial.
                Must be in range (1, 10].
            dem_name: The digital elevation model to use.
            external_dem_file: Path to external DEM file.
            external_dem_no_data_value: No data value for external DEM.
            tile_extension_percent: Extension of tile for DEM simulation
                (optimization parameter).
            output_topo_phase_band: Output topographic phase band.
            output_elevation_band: Output elevation band.
            output_lat_lon_bands: Output latitude/longitude bands.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to output product with topographic phase removed, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PorbitDegree={orbit_degree}',
            f'-PdemName="{dem_name}"',
            f'-PexternalDEMNoDataValue={external_dem_no_data_value}',
            f'-PtileExtensionPercent={tile_extension_percent}',
            f'-PoutputTopoPhaseBand={str(output_topo_phase_band).lower()}',
            f'-PoutputElevationBand={str(output_elevation_band).lower()}',
            f'-PoutputLatLonBands={str(output_lat_lon_bands).lower()}'
        ]
        
        if external_dem_file:
            cmd_params.append(f'-PexternalDEMFile={Path(external_dem_file).as_posix()}')
        
        self.current_cmd.append(f'TopoPhaseRemoval {" ".join(cmd_params)}')
        return self._call(suffix='TOPO', output_name=output_name)

    def tool_adapter(
        self,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Execute a custom tool adapter operator.
        
        This method provides a generic interface to SNAP's Tool Adapter framework,
        allowing execution of external tools and scripts integrated into SNAP.
        Tool adapters must be configured separately in SNAP before use.
        
        Args:
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to tool adapter output product, or None if failed.
        
        Note:
            This is a generic operator. Specific tool adapter parameters
            should be added to the command manually or through XML graphs.
        """
        self._reset_command()
        self.current_cmd.append('ToolAdapterOp')
        return self._call(suffix='TOOL', output_name=output_name)

    def apply_orbit_file(
        self,
        orbit_type: str = 'Sentinel Precise (Auto Download)',
        poly_degree: int = 3,
        continue_on_fail: bool = False,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Apply orbit file correction to SAR products.
        
        This method updates the orbit state vectors in the product metadata using
        precise or restituted orbit files, improving geolocation accuracy.
        
        Args:
            orbit_type: Type of orbit file to apply.
                Valid options include 'Sentinel Precise (Auto Download)',
                'Sentinel Restituted (Auto Download)', etc.
            poly_degree: Degree of polynomial for orbit interpolation.
            continue_on_fail: Continue processing if orbit file application fails.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to the output product with applied orbit file, or None if failed.
        """
        self._reset_command()
        self.current_cmd.append(
            f'Apply-Orbit-File '
            f'-PorbitType="{orbit_type}" '
            f'-PpolyDegree={poly_degree} '
            f'-PcontinueOnFail={str(continue_on_fail).lower()}'
        )
        return self._call(suffix='ORB', output_name=output_name)

    def terrain_correction(
        self,
        source_bands: Optional[List[str]] = None,
        dem_name: str = 'SRTM 3Sec',
        external_dem_file: Optional[str | Path] = None,
        external_dem_no_data_value: float = 0.0,
        external_dem_apply_egm: bool = True,
        dem_resampling_method: str = 'BILINEAR_INTERPOLATION',
        img_resampling_method: str = 'BISINC_21_POINT_INTERPOLATION',
        pixel_spacing_in_meter: float = 0.0,
        pixel_spacing_in_degree: float = 0.0,
        map_projection: str = 'WGS84(DD)',
        align_to_standard_grid: bool = False,
        standard_grid_origin_x: float = 0.0,
        standard_grid_origin_y: float = 0.0,
        nodata_value_at_sea: bool = False,
        save_dem: bool = True,
        save_lat_lon: bool = True,
        save_incidence_angle_from_ellipsoid: bool = False,
        save_local_incidence_angle: bool = True,
        save_projected_local_incidence_angle: bool = False,
        save_selected_source_band: bool = True,
        save_layover_shadow_mask: bool = False,
        output_complex: bool = True,
        apply_radiometric_normalization: bool = False,
        save_sigma_nought: bool = False,
        save_gamma_nought: bool = True,
        save_beta_nought: bool = False,
        incidence_angle_for_sigma0: str = 'Use projected local incidence angle from DEM',
        incidence_angle_for_gamma0: str = 'Use projected local incidence angle from DEM',
        aux_file: str = 'Latest Auxiliary File',
        external_aux_file: Optional[str | Path] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Apply terrain correction (orthorectification) using Range-Doppler method.
        
        This method corrects geometric distortions caused by topography and sensor geometry,
        projecting the SAR image onto a cartographic coordinate system using a DEM.
        
        Args:
            source_bands: List of source bands to process. If None, all bands are processed.
            dem_name: Digital elevation model name.
            external_dem_file: Path to external DEM file.
            external_dem_no_data_value: No data value for external DEM.
            external_dem_apply_egm: Apply EGM96 geoid to external DEM.
            dem_resampling_method: DEM resampling method.
            img_resampling_method: Image resampling method.
            pixel_spacing_in_meter: Output pixel spacing in meters (0 = automatic).
            pixel_spacing_in_degree: Output pixel spacing in degrees (0 = automatic).
            map_projection: Map projection in WKT format.
            align_to_standard_grid: Align output to standard grid.
            standard_grid_origin_x: X-coordinate of standard grid origin.
            standard_grid_origin_y: Y-coordinate of standard grid origin.
            nodata_value_at_sea: Mask sea areas with no data value.
            save_dem: Save DEM band in output.
            save_lat_lon: Save latitude/longitude bands.
            save_incidence_angle_from_ellipsoid: Save incidence angle from ellipsoid.
            save_local_incidence_angle: Save local incidence angle.
            save_projected_local_incidence_angle: Save projected local incidence angle.
            save_selected_source_band: Save selected source bands.
            save_layover_shadow_mask: Save layover/shadow mask.
            output_complex: Output complex data.
            apply_radiometric_normalization: Apply radiometric normalization.
            save_sigma_nought: Save sigma nought band.
            save_gamma_nought: Save gamma nought band.
            save_beta_nought: Save beta nought band.
            incidence_angle_for_sigma0: Incidence angle type for sigma0.
            incidence_angle_for_gamma0: Incidence angle type for gamma0.
            aux_file: Auxiliary file selection.
            external_aux_file: Path to external auxiliary file.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to the terrain-corrected output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
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
        ]
        
        if source_bands:
            cmd_params.insert(0, f'-PsourceBands={",".join(source_bands)}')
        
        if external_dem_file:
            cmd_params.append(f'-PexternalDEMFile={Path(external_dem_file).as_posix()}')
        
        if external_aux_file:
            cmd_params.append(f'-PexternalAuxFile={Path(external_aux_file).as_posix()}')
        
        self.current_cmd.append(f'Terrain-Correction {" ".join(cmd_params)}')
        return self._call(suffix='TC', output_name=output_name)

    def demodulate(self, output_name: Optional[str] = None) -> Optional[str]:
        """Perform demodulation and deramping of SLC data.
        
        This method removes the modulation and ramping applied to Single Look Complex (SLC)
        data during SAR processing, preparing the data for further interferometric or
        analysis operations.
        
        Args:
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to the demodulated output product, or None if failed.
        """
        self._reset_command()
        for i, cmd_part in enumerate(self.current_cmd):
            if cmd_part.startswith('-Ssource='):
                self.current_cmd[i] = cmd_part.replace('-Ssource=', '-SsourceProduct=')
                break
        self.current_cmd.append('Demodulate')
        return self._call(suffix='DEMOD', output_name=output_name)

    def write(
        self,
        output_file: Optional[str | Path] = None,
        format_name: Optional[str] = None,
        clear_cache_after_row_write: bool = False,
        delete_output_on_failure: bool = True,
        write_entire_tile_rows: bool = False
    ) -> Optional[str]:
        """Write a data product to a file with explicit control over write parameters.
        
        This method provides direct access to the SNAP Write operator, allowing
        fine-grained control over the write process including caching behavior
        and tile row processing.
        
        Args:
            output_file: The output file path. If None, uses standard path construction.
            format_name: The output file format name. If None, uses the instance format.
            clear_cache_after_row_write: Clear internal tile cache after a tile row
                has been written. Only effective if write_entire_tile_rows is True.
            delete_output_on_failure: Delete all output files after a failed write operation.
            write_entire_tile_rows: Wait until an entire tile row is computed before writing.
        
        Returns:
            Path to the written output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PclearCacheAfterRowWrite={str(clear_cache_after_row_write).lower()}',
            f'-PdeleteOutputOnFailure={str(delete_output_on_failure).lower()}',
            f'-PwriteEntireTileRows={str(write_entire_tile_rows).lower()}'
        ]
        
        if output_file:
            output_path = Path(output_file)
            cmd_params.append(f'-Pfile={output_path.as_posix()}')
        
        if format_name:
            cmd_params.append(f'-PformatName="{format_name}"')
        else:
            cmd_params.append(f'-PformatName="{self.format}"')
        
        self.current_cmd.append(f'Write {" ".join(cmd_params)}')
        return self._call(suffix='WRITE')

    def tile_writer(
        self,
        output_file: Optional[str | Path] = None,
        format_name: Optional[str] = None,
        division_by: str = 'Tiles',
        number_of_tiles: str = '4',
        pixel_size_x: int = 200,
        pixel_size_y: int = 200,
        overlap: int = 0,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Write a data product to tiles.
        
        This method splits the output into multiple tiles, useful for processing
        large datasets or creating tiled outputs for specific applications.
        
        Args:
            output_file: The output file path. If None, uses standard path construction.
            format_name: The output file format name. If None, uses the instance format.
            division_by: How to divide the tiles.
                Must be one of 'Tiles', 'Pixels'.
            number_of_tiles: The number of output tiles.
                Must be one of '2', '4', '9', '16', '36', '64', '100', '256'.
            pixel_size_x: Tile pixel width.
            pixel_size_y: Tile pixel height.
            overlap: Tile overlap in pixels.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to tiled output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PdivisionBy={division_by}',
            f'-PnumberOfTiles={number_of_tiles}',
            f'-PpixelSizeX={pixel_size_x}',
            f'-PpixelSizeY={pixel_size_y}',
            f'-Poverlap={overlap}'
        ]
        
        if output_file:
            output_path = Path(output_file)
            cmd_params.append(f'-Pfile={output_path.as_posix()}')
        
        if format_name:
            cmd_params.append(f'-PformatName="{format_name}"')
        else:
            cmd_params.append(f'-PformatName="{self.format}"')
        
        self.current_cmd.append(f'TileWriter {" ".join(cmd_params)}')
        return self._call(suffix='TILES', output_name=output_name)

    # Legacy method names for backward compatibility
    ImportVector = import_vector
    LandMask = land_mask
    TerrainMask = terrain_mask
    TerrainFlattening = terrain_flattening
    Calibration = calibration
    ThermalNoiseRemoval = thermal_noise_removal
    Deburst = deburst
    Multilook = multilook
    AdaptiveThresholding = adaptive_thresholding
    ObjectDiscrimination = object_discrimination
    Subset = subset
    AatsrSST = aatsr_sst
    AatsrUngrid = aatsr_ungrid
    WindFieldEstimation = wind_field_estimation
    Wdvi = wdvi
    Warp = warp
    UpdateGeoReference = update_geo_reference
    AddElevation = add_elevation
    ThreePassDInSAR = three_pass_dinsar
    TemporalPercentile = temporal_percentile
    Unmix = unmix
    Undersample = undersample
    Tsavi = tsavi
    Tndvi = tndvi
    TopsarSplit = topsar_split
    TopsarMerge = topsar_merge
    TopsarDerampDemod = topsar_deramp_demod
    TopoPhaseRemoval = topo_phase_removal
    ToolAdapter = tool_adapter
    ApplyOrbitFile = apply_orbit_file
    TerrainCorrection = terrain_correction
    Demodulate = demodulate
    Write = write
    TileWriter = tile_writer


def _identify_product_type(filename: str) -> str:
    """Identify the product type based on filename.
    
    Args:
        filename: Name of the product file.
    
    Returns:
        Product type string.
    
    Raises:
        ValueError: If product type cannot be determined.
    """
    if 'S1' in filename:
        return 'Sentinel-1'
    elif 'CSK' in filename:
        return 'COSMO-SkyMed'
    elif 'SAO' in filename:
        return 'SAOCOM'
    else:
        raise ValueError(f'Unknown product type for file: {filename}')
