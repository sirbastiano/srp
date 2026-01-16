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

    def set_no_data_value(
        self,
        no_data_value: float = 0.0,
        no_data_value_used: bool = True,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Set the NoDataValue and enable the NoDataValueUsed flag for all bands."""
        self._reset_command()
        self.current_cmd.append(
            f'SetNoDataValue '
            f'-PnoDataValue={no_data_value} '
            f'-PnoDataValueUsed={str(no_data_value_used).lower()}'
        )
        return self._call(suffix='NDV', output_name=output_name)

    def supervised_wishart_classification(
        self,
        training_dataset: str,
        window_size: int = 5,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Perform supervised Wishart classification using provided training data."""
        self._reset_command()
        training_path = Path(training_dataset).as_posix()
        self.current_cmd.append(
            f'Supervised-Wishart-Classification '
            f'-PtrainingDataSet={training_path} '
            f'-PwindowSize={window_size}'
        )
        return self._call(suffix='WISH', output_name=output_name)

    def subgraph(
        self,
        graph_file: str,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Encapsulate an existing graph via the SubGraph operator."""
        graph_path = Path(graph_file)
        if not graph_path.exists():
            raise FileNotFoundError(f'{graph_path} does not exist')

        self._reset_command()
        self.current_cmd.append(f'SubGraph -PgraphFile={graph_path.as_posix()}')
        return self._call(suffix='SUBG', output_name=output_name)

    def statistics_op(
        self,
        source_products: Optional[List[str | Path]] = None,
        source_product_paths: Optional[List[str]] = None,
        shapefile: Optional[str | Path] = None,
        feature_id: str = 'name',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_shapefile: Optional[str | Path] = None,
        output_ascii_file: Optional[str | Path] = None,
        percentiles: Optional[List[int]] = None,
        accuracy: int = 3,
        write_data_types_separately: bool = False,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compute statistics for the provided source products."""
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
            f'-Paccuracy={accuracy}',
            f'-PwriteDataTypesSeparately={str(write_data_types_separately).lower()}'
        ]

        if source_product_paths:
            cmd_params.append(f'-PsourceProductPaths={",".join(source_product_paths)}')

        if shapefile:
            cmd_params.append(f'-Pshapefile={Path(shapefile).as_posix()}')
            cmd_params.append(f'-PfeatureId={feature_id}')

        if start_date:
            cmd_params.append(f'-PstartDate="{start_date}"')
        if end_date:
            cmd_params.append(f'-PendDate="{end_date}"')

        if output_shapefile:
            cmd_params.append(f'-PoutputShapefile={Path(output_shapefile).as_posix()}')
        if output_ascii_file:
            cmd_params.append(f'-PoutputAsciiFile={Path(output_ascii_file).as_posix()}')

        if percentiles:
            percentiles_str = ','.join(str(p) for p in percentiles)
        else:
            percentiles_str = '90,95'
        cmd_params.append(f'-Ppercentiles={percentiles_str}')

        self.current_cmd.append(f'StatisticsOp {" ".join(cmd_params)}')
        return self._call(suffix='STAT', output_name=output_name)

    def stamps_export(
        self,
        target_folder: str,
        psi_format: bool = True,
        source_product: Optional[str | Path] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Export StaMPS-compatible data products."""
        self._reset_command()

        product_path = Path(source_product or self.prod_path).as_posix()
        for i, cmd_part in enumerate(self.current_cmd):
            if cmd_part.startswith('-Ssource='):
                self.current_cmd[i] = f'-SsourceProduct={product_path}'
                break

        target_folder_path = Path(target_folder)
        target_folder_path.mkdir(parents=True, exist_ok=True)

        self.current_cmd.append(
            f'StampsExport -PtargetFolder={target_folder_path.as_posix()} '
            f'-PpsiFormat={str(psi_format).lower()}'
        )
        return self._call(suffix='STMP', output_name=output_name)

    def stack_split(
        self,
        target_folder: str = 'target',
        format_name: str = 'BEAM-DIMAP',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Split the product into one file per band via Stack-Split."""
        self._reset_command()

        target_path = Path(target_folder)
        target_path.mkdir(parents=True, exist_ok=True)

        self.current_cmd.append(
            f'Stack-Split '
            f'-PtargetFolder={target_path.as_posix()} '
            f'-PformatName="{format_name}"'
        )
        return self._call(suffix='SSPL', output_name=output_name)

    def stack_averaging(
        self,
        statistic: str = 'Mean Average',
        source_product: Optional[str | Path] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Run Stack-Averaging on the product with the desired statistic."""
        self._reset_command()

        product_path = Path(source_product or self.prod_path).as_posix()
        for i, cmd_part in enumerate(self.current_cmd):
            if cmd_part.startswith('-Ssource='):
                self.current_cmd[i] = f'-SsourceProduct={product_path}'
                break

        self.current_cmd.append(
            f'Stack-Averaging -Pstatistic="{statistic}"'
        )
        return self._call(suffix='STKAVG', output_name=output_name)

    def s2_resampling(
        self,
        source_product: Optional[str | Path] = None,
        bands: Optional[List[str]] = None,
        downsampling: str = 'Mean',
        flag_downsampling: str = 'First',
        masks: Optional[List[str]] = None,
        resample_on_pyramid_levels: bool = True,
        resolution: str = '60',
        upsampling: str = 'Bilinear',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Run the S2Resampling operator with configurable options."""
        self._reset_command()

        product_path = Path(source_product or self.prod_path).as_posix()
        for i, cmd_part in enumerate(self.current_cmd):
            if cmd_part.startswith('-Ssource='):
                self.current_cmd[i] = f'-SsourceProduct={product_path}'
                break

        cmd_params = [
            f'-Presolution={resolution}',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PflagDownsampling={flag_downsampling}',
            f'-PresampleOnPyramidLevels={str(resample_on_pyramid_levels).lower()}'
        ]

        if bands:
            cmd_params.insert(0, f'-Pbands={",".join(bands)}')

        if masks:
            cmd_params.append(f'-Pmasks={",".join(masks)}')

        self.current_cmd.append(f'S2Resampling {" ".join(cmd_params)}')
        return self._call(suffix='S2R', output_name=output_name)

    def s2rep(
        self,
        downsampling: str = 'First',
        nir_factor: float = 1.0,
        nir_source_band: Optional[str] = None,
        red_b4_factor: float = 1.0,
        red_b5_factor: float = 1.0,
        red_b6_factor: float = 1.0,
        red_source_band4: Optional[str] = None,
        red_source_band5: Optional[str] = None,
        red_source_band6: Optional[str] = None,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compute the Sentinel-2 red-edge position index (S2rep)."""
        self._reset_command()

        cmd_params = [
            f'-PresampleType={resample_type}',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PredB4Factor={red_b4_factor}',
            f'-PredB5Factor={red_b5_factor}',
            f'-PredB6Factor={red_b6_factor}',
            f'-PnirFactor={nir_factor}'
        ]

        if red_source_band4:
            cmd_params.append(f'-PredSourceBand4={red_source_band4}')
        if red_source_band5:
            cmd_params.append(f'-PredSourceBand5={red_source_band5}')
        if red_source_band6:
            cmd_params.append(f'-PredSourceBand6={red_source_band6}')
        if nir_source_band:
            cmd_params.append(f'-PnirSourceBand={nir_source_band}')

        self.current_cmd.append(f'S2repOp {" ".join(cmd_params)}')
        return self._call(suffix='S2REP', output_name=output_name)

    def spectral_angle_mapper(
        self,
        reference_bands: Optional[List[str]] = None,
        thresholds: str = '0.0',
        spectra: Optional[List[str]] = None,
        hidden_spectra: Optional[List[str]] = None,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Classify the product using the Spectral Angle Mapper operator."""
        self._reset_command()

        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-Pthresholds="{thresholds}"'
        ]

        if reference_bands:
            cmd_params.append(f'-PreferenceBands={",".join(reference_bands)}')

        if spectra:
            cmd_params.append(f'-Pspectra={",".join(spectra)}')

        if hidden_spectra:
            cmd_params.append(f'-PhiddenSpectra={",".join(hidden_spectra)}')

        self.current_cmd.append(f'SpectralAngleMapperOp {" ".join(cmd_params)}')
        return self._call(suffix='SAM', output_name=output_name)

    def speckle_filter(
        self,
        source_bands: Optional[List[str]] = None,
        filter_type: str = 'Lee Sigma',
        filter_size_x: int = 3,
        filter_size_y: int = 3,
        damping_factor: int = 2,
        en_l: float = 1.0,
        estimate_enl: bool = False,
        num_looks_str: str = '1',
        sigma_str: str = '0.9',
        window_size: str = '7x7',
        target_window_size_str: str = '3x3',
        an_size: int = 50,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Apply the Speckle-Filter operator with configurable parameters."""
        self._reset_command()

        cmd_params = [
            f'-PanSize={an_size}',
            f'-PdampingFactor={damping_factor}',
            f'-Penl={en_l}',
            f'-PestimateENL={str(estimate_enl).lower()}',
            f'-Pfilter="{filter_type}"',
            f'-PfilterSizeX={filter_size_x}',
            f'-PfilterSizeY={filter_size_y}',
            f'-PnumLooksStr={num_looks_str}',
            f'-PsigmaStr={sigma_str}',
            f'-PtargetWindowSizeStr="{target_window_size_str}"',
            f'-PwindowSize="{window_size}"'
        ]

        if source_bands:
            cmd_params.insert(0, f'-PsourceBands={",".join(source_bands)}')

        self.current_cmd.append(f'Speckle-Filter {" ".join(cmd_params)}')
        return self._call(suffix='SPKL', output_name=output_name)

    def speckle_divergence(
        self,
        source_bands: Optional[List[str]] = None,
        window_size_str: str = '15x15',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Generate urban-area detection via the Speckle-Divergence operator."""
        self._reset_command()

        cmd_params = [
            f'-PwindowSizeStr="{window_size_str}"'
        ]

        if source_bands:
            cmd_params.insert(0, f'-PsourceBands={",".join(source_bands)}')

        self.current_cmd.append(f'Speckle-Divergence {" ".join(cmd_params)}')
        return self._call(suffix='SPKD', output_name=output_name)

    def snaphu_import(
        self,
        source_products: Optional[List[str | Path]] = None,
        do_not_keep_wrapped: bool = False,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Import Snaphu results into the product via SnaphuImport."""
        self._reset_command()

        products = source_products or [self.prod_path]
        if not products:
            raise ValueError('source_products must contain at least one product path')

        source_products_str = ','.join(Path(p).as_posix() for p in products)
        for i, cmd_part in enumerate(self.current_cmd):
            if cmd_part.startswith('-Ssource='):
                self.current_cmd[i] = f'-SsourceProducts={source_products_str}'
                break

        self.current_cmd.append(
            f'SnaphuImport -PdoNotKeepWrapped={str(do_not_keep_wrapped).lower()}'
        )
        return self._call(suffix='SNAP', output_name=output_name)

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

    def smac_op(
        self,
        aerosol_type: str,
        band_names: List[str],
        invalid_pixel: float = 0.0,
        mask_expression: Optional[str] = None,
        mask_expression_forward: Optional[str] = None,
        surf_press: float = 1013.0,
        tau_aero550: float = 0.2,
        u_h2o: float = 3.0,
        u_o3: float = 0.15,
        use_meris_ads: bool = False,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Run the SmacOp atmospheric correction operator."""
        if not band_names:
            raise ValueError('band_names must be provided for SmacOp')

        self._reset_command()

        cmd_params = [
            f'-PtauAero550={tau_aero550}',
            f'-PuH2o={u_h2o}',
            f'-PuO3={u_o3}',
            f'-PsurfPress={surf_press}',
            f'-PuseMerisADS={str(use_meris_ads).lower()}',
            f'-PaerosolType={aerosol_type}',
            f'-PinvalidPixel={invalid_pixel}',
            f'-PbandNames={",".join(band_names)}'
        ]

        if mask_expression:
            cmd_params.append(f'-PmaskExpression="{mask_expression}"')
        if mask_expression_forward:
            cmd_params.append(f'-PmaskExpressionForward="{mask_expression_forward}"')

        self.current_cmd.append(f'SmacOp {" ".join(cmd_params)}')
        return self._call(suffix='SMAC', output_name=output_name)

    def sm_dielectric_modeling(
        self,
        source_products: Optional[List[str | Path]] = None,
        effective_soil_temperature: float = 18.0,
        max_sm: float = 0.55,
        min_sm: float = 0.0,
        model_to_use: str = 'Hallikainen',
        output_land_cover: bool = True,
        output_rdc: bool = True,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Perform soil moisture inversion using the dielectric model."""
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
            f'-PeffectiveSoilTemperature={effective_soil_temperature}',
            f'-PmaxSM={max_sm}',
            f'-PminSM={min_sm}',
            f'-PmodelToUse={model_to_use}',
            f'-PoutputLandCover={str(output_land_cover).lower()}',
            f'-PoutputRDC={str(output_rdc).lower()}'
        ]

        self.current_cmd.append(f'SM-Dielectric-Modeling {" ".join(cmd_params)}')
        return self._call(suffix='SMOK', output_name=output_name)

    def slice_assembly(
        self,
        source_products: Optional[List[str | Path]] = None,
        selected_polarisations: Optional[List[str]] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Merge Sentinel-1 slice products using the SliceAssembly operator."""
        self._reset_command()

        products = source_products or [self.prod_path]
        if not products:
            raise ValueError('source_products must contain at least one product path')

        source_products_str = ','.join(Path(p).as_posix() for p in products)
        for i, cmd_part in enumerate(self.current_cmd):
            if cmd_part.startswith('-Ssource='):
                self.current_cmd[i] = f'-SsourceProducts={source_products_str}'
                break

        cmd_params = []
        if selected_polarisations:
            cmd_params.append(f'-PselectedPolarisations={",".join(selected_polarisations)}')

        self.current_cmd.append(f'SliceAssembly {" ".join(cmd_params)}')
        return self._call(suffix='SLICE', output_name=output_name)

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

    def savi(
        self,
        red_source_band: Optional[str] = None,
        nir_source_band: Optional[str] = None,
        red_factor: float = 1.0,
        nir_factor: float = 1.0,
        soil_correction_factor: float = 0.5,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compute the Soil Adjusted Vegetation Index (SAVI)."""
        self._reset_command()

        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PredFactor={red_factor}',
            f'-PnirFactor={nir_factor}',
            f'-PsoilCorrectionFactor={soil_correction_factor}'
        ]

        if red_source_band:
            cmd_params.append(f'-PredSourceBand={red_source_band}')
        if nir_source_band:
            cmd_params.append(f'-PnirSourceBand={nir_source_band}')

        self.current_cmd.append(f'SaviOp {" ".join(cmd_params)}')
        return self._call(suffix='SAVI', output_name=output_name)

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

    def srgr(
        self,
        source_bands: Optional[List[str]] = None,
        interpolation_method: str = 'Linear interpolation',
        warp_polynomial_order: int = 4,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Convert slant range to ground range using SRGR operator."""
        self._reset_command()

        cmd_params = [
            f'-PinterpolationMethod="{interpolation_method}"',
            f'-PwarpPolynomialOrder={warp_polynomial_order}'
        ]

        if source_bands:
            cmd_params.insert(0, f'-PsourceBands={",".join(source_bands)}')

        self.current_cmd.append(f'SRGR {" ".join(cmd_params)}')
        return self._call(suffix='SRGR', output_name=output_name)

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
        output_deramp_demod_phase: bool = True,
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
        dem_name: str = 'Copernicus 30m Global DEM',
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
            dem_name: Digital elevation model name. (CDEM, SRTM 3Sec, Copernicus 30m Global DEM, ...)
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

    def sar_sim_terrain_correction(
        self,
        map_projection: str = 'WGS84(DD)',
        img_resampling_method: str = 'BILINEAR_INTERPOLATION',
        pixel_spacing_in_meter: float = 0.0,
        pixel_spacing_in_degree: float = 0.0,
        align_to_standard_grid: bool = False,
        standard_grid_origin_x: float = 0.0,
        standard_grid_origin_y: float = 0.0,
        rms_threshold: float = 1.0,
        warp_polynomial_order: int = 1,
        apply_radiometric_normalization: bool = False,
        aux_file: str = 'Latest Auxiliary File',
        external_aux_file: Optional[str | Path] = None,
        open_shifts_file: bool = False,
        open_residuals_file: bool = False,
        output_complex: bool = False,
        save_dem: bool = False,
        save_lat_lon: bool = False,
        save_local_incidence_angle: bool = False,
        save_projected_local_incidence_angle: bool = False,
        save_selected_source_band: bool = True,
        save_sigma_nought: bool = False,
        save_gamma_nought: bool = False,
        save_beta_nought: bool = False,
        incidence_angle_for_sigma0: str = 'Use projected local incidence angle from DEM',
        incidence_angle_for_gamma0: str = 'Use projected local incidence angle from DEM',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Orthorectify the product using SARSim Terrain Correction."""
        self._reset_command()

        cmd_params = [
            f'-PrmsThreshold={rms_threshold}',
            f'-PwarpPolynomialOrder={warp_polynomial_order}',
            f'-PimgResamplingMethod={img_resampling_method}',
            f'-PpixelSpacingInMeter={pixel_spacing_in_meter}',
            f'-PpixelSpacingInDegree={pixel_spacing_in_degree}',
            f'-PmapProjection="{map_projection}"',
            f'-PalignToStandardGrid={str(align_to_standard_grid).lower()}',
            f'-PstandardGridOriginX={standard_grid_origin_x}',
            f'-PstandardGridOriginY={standard_grid_origin_y}',
            f'-PsaveDEM={str(save_dem).lower()}',
            f'-PsaveLatLon={str(save_lat_lon).lower()}',
            f'-PsaveLocalIncidenceAngle={str(save_local_incidence_angle).lower()}',
            f'-PsaveProjectedLocalIncidenceAngle={str(save_projected_local_incidence_angle).lower()}',
            f'-PsaveSelectedSourceBand={str(save_selected_source_band).lower()}',
            f'-PoutputComplex={str(output_complex).lower()}',
            f'-PapplyRadiometricNormalization={str(apply_radiometric_normalization).lower()}',
            f'-PsaveSigmaNought={str(save_sigma_nought).lower()}',
            f'-PsaveGammaNought={str(save_gamma_nought).lower()}',
            f'-PsaveBetaNought={str(save_beta_nought).lower()}',
            f'-PincidenceAngleForSigma0="{incidence_angle_for_sigma0}"',
            f'-PincidenceAngleForGamma0="{incidence_angle_for_gamma0}"',
            f'-PauxFile="{aux_file}"',
            f'-PopenResidualsFile={str(open_residuals_file).lower()}',
            f'-PopenShiftsFile={str(open_shifts_file).lower()}'
        ]

        if external_aux_file:
            cmd_params.append(f'-PexternalAuxFile={Path(external_aux_file).as_posix()}')

        self.current_cmd.append(f'SARSim-Terrain-Correction {" ".join(cmd_params)}')
        return self._call(suffix='SSTM', output_name=output_name)

    def sar_simulation(
        self,
        dem_name: str = 'SRTM 3Sec',
        dem_resampling_method: str = 'BICUBIC_INTERPOLATION',
        external_dem_apply_egm: bool = True,
        external_dem_file: Optional[str | Path] = None,
        external_dem_no_data_value: float = 0.0,
        save_layover_shadow_mask: bool = False,
        source_bands: Optional[List[str]] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Run the rigorous SAR Simulation operator."""
        self._reset_command()

        cmd_params = [
            f'-PdemName="{dem_name}"',
            f'-PdemResamplingMethod={dem_resampling_method}',
            f'-PexternalDEMApplyEGM={str(external_dem_apply_egm).lower()}',
            f'-PexternalDEMNoDataValue={external_dem_no_data_value}',
            f'-PsaveLayoverShadowMask={str(save_layover_shadow_mask).lower()}'
        ]

        if source_bands:
            cmd_params.insert(0, f'-PsourceBands={",".join(source_bands)}')

        if external_dem_file:
            cmd_params.append(f'-PexternalDEMFile={Path(external_dem_file).as_posix()}')

        self.current_cmd.append(f'SAR-Simulation {" ".join(cmd_params)}')
        return self._call(suffix='SRSIM', output_name=output_name)

    def sar_mosaic(
        self,
        source_bands: Optional[List[str]] = None,
        average: bool = True,
        convergence_threshold: float = 1e-4,
        feather: int = 0,
        gradient_domain_mosaic: bool = False,
        max_iterations: int = 5000,
        normalize_by_mean: bool = True,
        pixel_size: float = 0.0,
        presampling_method: str = 'NEAREST_NEIGHBOUR',
        scene_height: int = 0,
        scene_width: int = 0,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Mosaic multiple SAR products with adjustable blending parameters."""
        self._reset_command()

        cmd_params = [
            f'-Paverage={str(average).lower()}',
            f'-PconvergenceThreshold={convergence_threshold}',
            f'-Pfeather={feather}',
            f'-PgradientDomainMosaic={str(gradient_domain_mosaic).lower()}',
            f'-PmaxIterations={max_iterations}',
            f'-PnormalizeByMean={str(normalize_by_mean).lower()}',
            f'-PpixelSize={pixel_size}',
            f'-PresamplingMethod={presampling_method}',
            f'-PsceneHeight={scene_height}',
            f'-PsceneWidth={scene_width}'
        ]

        if source_bands:
            cmd_params.insert(0, f'-PsourceBands={",".join(source_bands)}')

        self.current_cmd.append(f'SAR-Mosaic {" ".join(cmd_params)}')
        return self._call(suffix='SMOS', output_name=output_name)
       
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

    def band_maths(
        self,
        target_bands: Optional[List[dict]] = None,
        variables: Optional[List[dict]] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Create a product with one or more bands using mathematical expressions.
        
        This method allows creating new bands based on mathematical expressions
        applied to existing bands, enabling complex band arithmetic and transformations.
        
        Args:
            target_bands: List of target band dictionaries. Each dict should contain:
                - name (str): Band name
                - type (str): Data type (e.g., 'float32')
                - expression (str): Mathematical expression
                - description (str, optional): Band description
                - unit (str, optional): Band unit
                - no_data_value (float, optional): NoData value
            variables: List of variable dictionaries for use in expressions.
                Each dict should contain:
                - name (str): Variable name
                - type (str): Data type
                - value (str): Variable value
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to output product with computed bands, or None if failed.
        """
        self._reset_command()
        
        cmd_params = []
        
        # Note: Complex parameters like targetBands and variables are typically 
        # better handled via XML graphs. For simple cases, consider using
        # BandSelect or other operators.
        
        self.current_cmd.append(f'BandMaths {" ".join(cmd_params)}')
        return self._call(suffix='BMTH', output_name=output_name)

    def band_select(
        self,
        source_bands: Optional[List[str]] = None,
        band_name_pattern: Optional[str] = None,
        selected_polarisations: Optional[List[str]] = None,
        selected_sub_images: Optional[List[str]] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Creates a new product with only selected bands.
        
        This method filters the product to include only specified bands,
        reducing data size and focusing on relevant information.
        
        Args:
            source_bands: List of source band names to include.
            band_name_pattern: Band name regular expression pattern.
            selected_polarisations: List of polarisations to select.
            selected_sub_images: List of imagettes or sub-images to select.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to output product with selected bands, or None if failed.
        """
        self._reset_command()
        
        cmd_params = []
        
        if source_bands:
            cmd_params.append(f'-PsourceBands={",".join(source_bands)}')
        
        if band_name_pattern:
            cmd_params.append(f'-PbandNamePattern={band_name_pattern}')
        
        if selected_polarisations:
            cmd_params.append(f'-PselectedPolarisations={",".join(selected_polarisations)}')
        
        if selected_sub_images:
            cmd_params.append(f'-PselectedSubImages={",".join(selected_sub_images)}')
        
        self.current_cmd.append(f'BandSelect {" ".join(cmd_params)}')
        return self._call(suffix='BSL', output_name=output_name)

    def band_merge(
        self,
        source_products: Optional[List[str | Path]] = None,
        source_bands: Optional[List[str]] = None,
        geographic_error: float = 1.0e-5,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Allows copying raster data from any number of source products to a specified 'master' product.
        
        This method merges bands from multiple products into a single product,
        useful for combining data from different sources.
        
        Args:
            source_products: List of source product paths.
            source_bands: List of source band names.
            geographic_error: Maximum lat/lon error in degrees between products.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to merged output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [f'-PgeographicError={geographic_error}']
        
        if source_bands:
            cmd_params.append(f'-PsourceBands={",".join(source_bands)}')
        
        self.current_cmd.append(f'BandMerge {" ".join(cmd_params)}')
        return self._call(suffix='BMRG', output_name=output_name)

    def coherence(
        self,
        coh_win_az: int = 10,
        coh_win_rg: int = 10,
        subtract_flat_earth_phase: bool = False,
        srp_polynomial_degree: int = 5,
        srp_number_points: int = 501,
        orbit_degree: int = 3,
        square_pixel: bool = True,
        subtract_topographic_phase: bool = False,
        dem_name: str = 'SRTM 3Sec',
        external_dem_file: Optional[str | Path] = None,
        external_dem_no_data_value: float = 0.0,
        external_dem_apply_egm: bool = True,
        tile_extension_percent: str = '100',
        single_master: bool = True,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Estimate coherence from stack of coregistered images.
        
        This method computes interferometric coherence, a measure of correlation
        between SAR images, essential for InSAR quality assessment.
        
        Args:
            coh_win_az: Size of coherence estimation window in Azimuth direction.
                Valid interval is (1, 90].
            coh_win_rg: Size of coherence estimation window in Range direction.
                Valid interval is (1, 90].
            subtract_flat_earth_phase: Subtract flat earth phase.
            srp_polynomial_degree: Order of 'Flat earth phase' polynomial.
                Must be one of 1, 2, 3, 4, 5, 6, 7, 8.
            srp_number_points: Number of points for the 'flat earth phase' polynomial estimation.
                Must be one of 301, 401, 501, 601, 701, 801, 901, 1001.
            orbit_degree: Degree of orbit (polynomial) interpolator.
                Must be one of 1, 2, 3, 4, 5.
            square_pixel: Use ground square pixel.
            subtract_topographic_phase: Subtract topographic phase.
            dem_name: The digital elevation model.
            external_dem_file: Path to external DEM file.
            external_dem_no_data_value: No data value for external DEM.
            external_dem_apply_egm: Apply EGM96 geoid to external DEM.
            tile_extension_percent: Define extension of tile for DEM simulation.
            single_master: Single master mode.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to coherence output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PcohWinAz={coh_win_az}',
            f'-PcohWinRg={coh_win_rg}',
            f'-PsubtractFlatEarthPhase={str(subtract_flat_earth_phase).lower()}',
            f'-PsrpPolynomialDegree={srp_polynomial_degree}',
            f'-PsrpNumberPoints={srp_number_points}',
            f'-PorbitDegree={orbit_degree}',
            f'-PsquarePixel={str(square_pixel).lower()}',
            f'-PsubtractTopographicPhase={str(subtract_topographic_phase).lower()}',
            f'-PdemName="{dem_name}"',
            f'-PexternalDEMNoDataValue={external_dem_no_data_value}',
            f'-PexternalDEMApplyEGM={str(external_dem_apply_egm).lower()}',
            f'-PtileExtensionPercent={tile_extension_percent}',
            f'-PsingleMaster={str(single_master).lower()}'
        ]
        
        if external_dem_file:
            cmd_params.append(f'-PexternalDEMFile={Path(external_dem_file).as_posix()}')
        
        self.current_cmd.append(f'Coherence {" ".join(cmd_params)}')
        return self._call(suffix='COH', output_name=output_name)

    def interferogram(
        self,
        subtract_flat_earth_phase: bool = True,
        srp_polynomial_degree: int = 5,
        srp_number_points: int = 501,
        orbit_degree: int = 3,
        include_coherence: bool = True,
        coh_win_az: int = 10,
        coh_win_rg: int = 10,
        square_pixel: bool = True,
        subtract_topographic_phase: bool = False,
        dem_name: str = 'SRTM 3Sec',
        external_dem_file: Optional[str | Path] = None,
        external_dem_no_data_value: float = 0.0,
        external_dem_apply_egm: bool = True,
        tile_extension_percent: str = '100',
        output_flat_earth_phase: bool = False,
        output_topo_phase: bool = False,
        output_elevation: bool = False,
        output_lat_lon: bool = False,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compute interferograms from stack of coregistered S-1 images.
        
        This method generates interferometric phase from coregistered SAR image pairs,
        essential for InSAR displacement and topography analysis.
        
        Args:
            subtract_flat_earth_phase: Subtract flat earth phase.
            srp_polynomial_degree: Order of 'Flat earth phase' polynomial.
            srp_number_points: Number of points for polynomial estimation.
            orbit_degree: Degree of orbit interpolator.
            include_coherence: Include coherence estimation.
            coh_win_az: Coherence window size in azimuth.
            coh_win_rg: Coherence window size in range.
            square_pixel: Use ground square pixel.
            subtract_topographic_phase: Subtract topographic phase.
            dem_name: The digital elevation model.
            external_dem_file: Path to external DEM file.
            external_dem_no_data_value: No data value for external DEM.
            external_dem_apply_egm: Apply EGM96 geoid to external DEM.
            tile_extension_percent: Tile extension for DEM simulation.
            output_flat_earth_phase: Output flat earth phase band.
            output_topo_phase: Output topographic phase band.
            output_elevation: Output elevation band.
            output_lat_lon: Output latitude/longitude bands.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to interferogram output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PsubtractFlatEarthPhase={str(subtract_flat_earth_phase).lower()}',
            f'-PsrpPolynomialDegree={srp_polynomial_degree}',
            f'-PsrpNumberPoints={srp_number_points}',
            f'-PorbitDegree={orbit_degree}',
            f'-PincludeCoherence={str(include_coherence).lower()}',
            f'-PcohWinAz={coh_win_az}',
            f'-PcohWinRg={coh_win_rg}',
            f'-PsquarePixel={str(square_pixel).lower()}',
            f'-PsubtractTopographicPhase={str(subtract_topographic_phase).lower()}',
            f'-PdemName="{dem_name}"',
            f'-PexternalDEMNoDataValue={external_dem_no_data_value}',
            f'-PexternalDEMApplyEGM={str(external_dem_apply_egm).lower()}',
            f'-PtileExtensionPercent={tile_extension_percent}',
            f'-PoutputFlatEarthPhase={str(output_flat_earth_phase).lower()}',
            f'-PoutputTopoPhase={str(output_topo_phase).lower()}',
            f'-PoutputElevation={str(output_elevation).lower()}',
            f'-PoutputLatLon={str(output_lat_lon).lower()}'
        ]
        
        if external_dem_file:
            cmd_params.append(f'-PexternalDEMFile={Path(external_dem_file).as_posix()}')
        
        self.current_cmd.append(f'Interferogram {" ".join(cmd_params)}')
        return self._call(suffix='IFG', output_name=output_name)

    def goldstein_phase_filtering(
        self,
        alpha: float = 1.0,
        fft_size_string: str = '64',
        window_size_string: str = '3',
        use_coherence_mask: bool = False,
        coherence_threshold: float = 0.2,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Phase filtering using Goldstein method.
        
        This method applies adaptive phase filtering to interferograms,
        reducing noise while preserving phase features.
        
        Args:
            alpha: Adaptive filter exponent. Valid interval is (0, 1].
            fft_size_string: FFT size. Must be one of '32', '64', '128', '256'.
            window_size_string: Window size. Must be one of '3', '5', '7'.
            use_coherence_mask: Use coherence mask.
            coherence_threshold: The coherence threshold. Valid interval is [0, 1].
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to filtered output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-Palpha={alpha}',
            f'-PFFTSizeString={fft_size_string}',
            f'-PwindowSizeString={window_size_string}',
            f'-PuseCoherenceMask={str(use_coherence_mask).lower()}',
            f'-PcoherenceThreshold={coherence_threshold}'
        ]
        
        self.current_cmd.append(f'GoldsteinPhaseFiltering {" ".join(cmd_params)}')
        return self._call(suffix='GPHS', output_name=output_name)

    def back_geocoding(
        self,
        dem_name: str = 'SRTM 3Sec',
        dem_resampling_method: str = 'BICUBIC_INTERPOLATION',
        external_dem_file: Optional[str | Path] = None,
        external_dem_no_data_value: float = 0.0,
        resampling_type: str = 'BISINC_5_POINT_INTERPOLATION',
        mask_out_area_without_elevation: bool = True,
        output_range_azimuth_offset: bool = False,
        output_deramp_demod_phase: bool = False,
        disable_reramp: bool = False,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Bursts co-registration using orbit and DEM.
        
        This method performs coregistration of burst SAR data using precise
        orbit information and DEM, essential for InSAR processing.
        
        Args:
            dem_name: The digital elevation model.
            dem_resampling_method: DEM resampling method.
            external_dem_file: Path to external DEM file.
            external_dem_no_data_value: No data value for external DEM.
            resampling_type: Method for resampling slave grid onto master grid.
            mask_out_area_without_elevation: Mask areas without elevation data.
            output_range_azimuth_offset: Output range/azimuth offset bands.
            output_deramp_demod_phase: Output deramp/demod phase band.
            disable_reramp: Disable reramp operation.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to coregistered output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PdemName="{dem_name}"',
            f'-PdemResamplingMethod={dem_resampling_method}',
            f'-PexternalDEMNoDataValue={external_dem_no_data_value}',
            f'-PresamplingType={resampling_type}',
            f'-PmaskOutAreaWithoutElevation={str(mask_out_area_without_elevation).lower()}',
            f'-PoutputRangeAzimuthOffset={str(output_range_azimuth_offset).lower()}',
            f'-PoutputDerampDemodPhase={str(output_deramp_demod_phase).lower()}',
            f'-PdisableReramp={str(disable_reramp).lower()}'
        ]
        
        if external_dem_file:
            cmd_params.append(f'-PexternalDEMFile={Path(external_dem_file).as_posix()}')
        
        self.current_cmd.append(f'Back-Geocoding {" ".join(cmd_params)}')
        return self._call(suffix='BGEO', output_name=output_name)

    def create_stack(
        self,
        master_bands: Optional[List[str]] = None,
        source_bands: Optional[List[str]] = None,
        resampling_type: str = 'NONE',
        extent: str = 'Master',
        initial_offset_method: str = 'Orbit',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Collocates two or more products based on their geo-codings.
        
        This method creates a stack by collocating multiple products,
        essential for multi-temporal analysis.
        
        Args:
            master_bands: List of master bands.
            source_bands: List of source bands.
            resampling_type: Method for resampling slave grid onto master grid.
            extent: The output image extents. Must be one of 'Master', 'Minimum', 'Maximum'.
            initial_offset_method: Method for computing initial offset.
                Must be one of 'Orbit', 'Product Geolocation'.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to stack output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PresamplingType={resampling_type}',
            f'-Pextent={extent}',
            f'-PinitialOffsetMethod="{initial_offset_method}"'
        ]
        
        if master_bands:
            cmd_params.insert(0, f'-PmasterBands={",".join(master_bands)}')
        
        if source_bands:
            cmd_params.insert(0, f'-PsourceBands={",".join(source_bands)}')
        
        self.current_cmd.append(f'CreateStack {" ".join(cmd_params)}')
        return self._call(suffix='STCK', output_name=output_name)

    def resample(
        self,
        reference_band: Optional[str] = None,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        target_resolution: Optional[int] = None,
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        flag_downsampling: str = 'First',
        resampling_preset: Optional[str] = None,
        band_resamplings: Optional[str] = None,
        resample_on_pyramid_levels: bool = True,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Resampling of a multi-size source product to a single-size target product.
        
        This method resamples bands to a common resolution, essential for
        multi-sensor or multi-resolution data integration.
        
        Args:
            reference_band: Name of reference band. All other bands resampled to match it.
            target_width: Target width in pixels.
            target_height: Target height in pixels.
            target_resolution: Target resolution.
            upsampling: Interpolation method for upsampling.
                Must be one of 'Nearest', 'Bilinear', 'Bicubic'.
            downsampling: Aggregation method for downsampling.
                Must be one of 'First', 'Min', 'Max', 'Mean', 'Median'.
            flag_downsampling: Aggregation method for flags.
                Must be one of 'First', 'FlagAnd', 'FlagOr', 'FlagMedianAnd', 'FlagMedianOr'.
            resampling_preset: Resampling preset name.
            band_resamplings: Band-specific resampling settings.
            resample_on_pyramid_levels: Increase performance for viewing.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to resampled output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PflagDownsampling={flag_downsampling}',
            f'-PresampleOnPyramidLevels={str(resample_on_pyramid_levels).lower()}'
        ]
        
        if reference_band:
            cmd_params.insert(0, f'-PreferenceBand={reference_band}')
        
        if target_width is not None:
            cmd_params.append(f'-PtargetWidth={target_width}')
        
        if target_height is not None:
            cmd_params.append(f'-PtargetHeight={target_height}')
        
        if target_resolution is not None:
            cmd_params.append(f'-PtargetResolution={target_resolution}')
        
        if resampling_preset:
            cmd_params.append(f'-PresamplingPreset={resampling_preset}')
        
        if band_resamplings:
            cmd_params.append(f'-PbandResamplings={band_resamplings}')
        
        self.current_cmd.append(f'Resample {" ".join(cmd_params)}')
        return self._call(suffix='RSMP', output_name=output_name)

    def linear_to_from_db(
        self,
        source_bands: Optional[List[str]] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Converts bands to/from dB.
        
        This method converts amplitude values between linear and decibel (dB) scale,
        commonly used in SAR data analysis.
        
        Args:
            source_bands: List of source bands to convert.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to converted output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = []
        
        if source_bands:
            cmd_params.append(f'-PsourceBands={",".join(source_bands)}')
        
        self.current_cmd.append(f'LinearToFromdB {" ".join(cmd_params)}')
        return self._call(suffix='DB', output_name=output_name)

    def remove_grd_border_noise(
        self,
        selected_polarisations: Optional[List[str]] = None,
        border_limit: int = 500,
        trim_threshold: float = 0.5,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Mask no-value pixels for GRD product.
        
        This method removes border noise artifacts from Sentinel-1 GRD products,
        improving data quality.
        
        Args:
            selected_polarisations: List of polarisations to process.
            border_limit: The border margin limit in pixels.
            trim_threshold: The trim threshold.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to noise-removed output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PborderLimit={border_limit}',
            f'-PtrimThreshold={trim_threshold}'
        ]
        
        if selected_polarisations:
            cmd_params.insert(0, f'-PselectedPolarisations={",".join(selected_polarisations)}')
        
        self.current_cmd.append(f'Remove-GRD-Border-Noise {" ".join(cmd_params)}')
        return self._call(suffix='RBNX', output_name=output_name)

    def collocate(
        self,
        source_product_paths: Optional[List[str]] = None,
        reference_product_name: Optional[str] = None,
        target_product_type: str = 'COLLOCATED',
        copy_secondary_metadata: bool = False,
        rename_reference_components: bool = True,
        rename_secondary_components: bool = True,
        reference_component_pattern: str = '${ORIGINAL_NAME}_M',
        secondary_component_pattern: str = '${ORIGINAL_NAME}_S${SLAVE_NUMBER_ID}',
        resampling_type: str = 'NEAREST_NEIGHBOUR',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Collocates two products based on their geo-codings.
        
        This method aligns multiple products to a common grid, enabling
        direct pixel-by-pixel comparison and analysis.
        
        Args:
            source_product_paths: Comma-separated list of source product paths.
            reference_product_name: Name of the reference product.
            target_product_type: Product type string for target product.
            copy_secondary_metadata: Copy metadata from secondary products.
            rename_reference_components: Rename reference components in target.
            rename_secondary_components: Rename secondary components in target.
            reference_component_pattern: Text pattern for renaming reference components.
            secondary_component_pattern: Text pattern for renaming secondary components.
            resampling_type: Method for resampling secondary grid onto reference grid.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to collocated output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PtargetProductType="{target_product_type}"',
            f'-PcopySecondaryMetadata={str(copy_secondary_metadata).lower()}',
            f'-PrenameReferenceComponents={str(rename_reference_components).lower()}',
            f'-PrenameSecondaryComponents={str(rename_secondary_components).lower()}',
            f'-PreferenceComponentPattern="{reference_component_pattern}"',
            f'-PsecondaryComponentPattern="{secondary_component_pattern}"',
            f'-PresamplingType={resampling_type}'
        ]
        
        if source_product_paths:
            cmd_params.insert(0, f'-PsourceProductPaths={",".join(source_product_paths)}')
        
        if reference_product_name:
            cmd_params.insert(0, f'-PreferenceProductName="{reference_product_name}"')
        
        self.current_cmd.append(f'Collocate {" ".join(cmd_params)}')
        return self._call(suffix='COLL', output_name=output_name)

    def polarimetric_decomposition(
        self,
        decomposition: str = 'Sinclair Decomposition',
        window_size: int = 5,
        output_ha_alpha: bool = False,
        output_beta_delta_gamma_lambda: bool = False,
        output_alpha123: bool = False,
        output_lambda123: bool = False,
        output_touzi_param_set0: bool = False,
        output_touzi_param_set1: bool = False,
        output_touzi_param_set2: bool = False,
        output_touzi_param_set3: bool = False,
        output_huynen_param_set0: bool = True,
        output_huynen_param_set1: bool = False,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Perform Polarimetric decomposition of a given product.
        
        This method applies various polarimetric decomposition techniques to
        extract physical scattering mechanisms from polarimetric SAR data.
        
        Args:
            decomposition: Decomposition method. Must be one of:
                'Sinclair Decomposition', 'Pauli Decomposition', 
                'Freeman-Durden Decomposition', 'Generalized Freeman-Durden Decomposition',
                'Yamaguchi Decomposition', 'van Zyl Decomposition',
                'H-A-Alpha Quad Pol Decomposition', 'H-Alpha Dual Pol Decomposition',
                'Cloude Decomposition', 'Touzi Decomposition', 'Huynen Decomposition',
                'Yang Decomposition', 'Krogager Decomposition', 'Cameron Decomposition',
                'Model-free 3-component Decomposition', 'Model-free 4-component Decomposition',
                'Model-Based Dual Pol Decomposition'.
            window_size: The sliding window size. Valid interval is [1, 100].
            output_ha_alpha: Output entropy, anisotropy, alpha.
            output_beta_delta_gamma_lambda: Output beta, delta, gamma, lambda.
            output_alpha123: Output alpha 1, 2, 3.
            output_lambda123: Output lambda 1, 2, 3.
            output_touzi_param_set0: Output psi, tau, alpha, phi.
            output_touzi_param_set1: Output psi1, tau1, alpha1, phi1.
            output_touzi_param_set2: Output psi2, tau2, alpha2, phi2.
            output_touzi_param_set3: Output psi3, tau3, alpha3, phi3.
            output_huynen_param_set0: Output 2A0_b, B0_plus_B, B0_minus_B.
            output_huynen_param_set1: Output A0, B0, B, C, D, E, F, G, H.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to decomposed output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-Pdecomposition="{decomposition}"',
            f'-PwindowSize={window_size}',
            f'-PoutputHAAlpha={str(output_ha_alpha).lower()}',
            f'-PoutputBetaDeltaGammaLambda={str(output_beta_delta_gamma_lambda).lower()}',
            f'-PoutputAlpha123={str(output_alpha123).lower()}',
            f'-PoutputLambda123={str(output_lambda123).lower()}',
            f'-PoutputTouziParamSet0={str(output_touzi_param_set0).lower()}',
            f'-PoutputTouziParamSet1={str(output_touzi_param_set1).lower()}',
            f'-PoutputTouziParamSet2={str(output_touzi_param_set2).lower()}',
            f'-PoutputTouziParamSet3={str(output_touzi_param_set3).lower()}',
            f'-PoutputHuynenParamSet0={str(output_huynen_param_set0).lower()}',
            f'-PoutputHuynenParamSet1={str(output_huynen_param_set1).lower()}'
        ]
        
        self.current_cmd.append(f'Polarimetric-Decomposition {" ".join(cmd_params)}')
        return self._call(suffix='PDEC', output_name=output_name)

    def polarimetric_parameters(
        self,
        use_mean_matrix: bool = True,
        window_size_x_str: str = '5',
        window_size_y_str: str = '5',
        output_span: bool = True,
        output_pedestal_height: bool = False,
        output_rvi: bool = False,
        output_rfdi: bool = False,
        output_csi: bool = False,
        output_vsi: bool = False,
        output_bmi: bool = False,
        output_iti: bool = False,
        output_hhvv_ratio: bool = False,
        output_hhhv_ratio: bool = False,
        output_vvvh_ratio: bool = False,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compute general polarimetric parameters.
        
        This method extracts various polarimetric parameters from quad-pol
        or dual-pol SAR data for analysis and interpretation.
        
        Args:
            use_mean_matrix: Use mean coherency or covariance matrix.
            window_size_x_str: Window size in X direction.
                Must be one of '3', '5', '7', '9', '11', '13', '15', '17', '19'.
            window_size_y_str: Window size in Y direction.
                Must be one of '3', '5', '7', '9', '11', '13', '15', '17', '19'.
            output_span: Output Span.
            output_pedestal_height: Output pedestal height.
            output_rvi: Output RVI.
            output_rfdi: Output RFDI.
            output_csi: Output CSI.
            output_vsi: Output VSI.
            output_bmi: Output BMI.
            output_iti: Output ITI.
            output_hhvv_ratio: Output Co-Pol HH/VV ratio.
            output_hhhv_ratio: Output Cross-Pol HH/HV ratio.
            output_vvvh_ratio: Output Cross-Pol VV/VH ratio.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to output product with polarimetric parameters, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PuseMeanMatrix={str(use_mean_matrix).lower()}',
            f'-PwindowSizeXStr={window_size_x_str}',
            f'-PwindowSizeYStr={window_size_y_str}',
            f'-PoutputSpan={str(output_span).lower()}',
            f'-PoutputPedestalHeight={str(output_pedestal_height).lower()}',
            f'-PoutputRVI={str(output_rvi).lower()}',
            f'-PoutputRFDI={str(output_rfdi).lower()}',
            f'-PoutputCSI={str(output_csi).lower()}',
            f'-PoutputVSI={str(output_vsi).lower()}',
            f'-PoutputBMI={str(output_bmi).lower()}',
            f'-PoutputITI={str(output_iti).lower()}',
            f'-PoutputHHVVRatio={str(output_hhvv_ratio).lower()}',
            f'-PoutputHHHVRatio={str(output_hhhv_ratio).lower()}',
            f'-PoutputVVVHRatio={str(output_vvvh_ratio).lower()}'
        ]
        
        self.current_cmd.append(f'Polarimetric-Parameters {" ".join(cmd_params)}')
        return self._call(suffix='PPAR', output_name=output_name)

    def offset_tracking(
        self,
        grid_azimuth_spacing: int = 40,
        grid_range_spacing: int = 40,
        registration_window_width: str = '128',
        registration_window_height: str = '128',
        x_corr_threshold: float = 0.1,
        registration_oversampling: str = '16',
        average_box_size: str = '5',
        max_velocity: float = 5.0,
        radius: int = 4,
        resampling_type: str = 'BICUBIC_INTERPOLATION',
        spatial_average: bool = True,
        fill_holes: bool = True,
        roi_vector: Optional[str] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Create velocity vectors from offset tracking.
        
        This method tracks pixel offsets between coregistered images to measure
        surface displacement, useful for glacier flow and landslide monitoring.
        
        Args:
            grid_azimuth_spacing: Output grid azimuth spacing in pixels.
            grid_range_spacing: Output grid range spacing in pixels.
            registration_window_width: Registration window width.
                Must be one of '32', '64', '128', '256', '512', '1024', '2048'.
            registration_window_height: Registration window height.
                Must be one of '32', '64', '128', '256', '512', '1024', '2048'.
            x_corr_threshold: The cross-correlation threshold.
            registration_oversampling: Registration oversampling factor.
                Must be one of '2', '4', '8', '16', '32', '64', '128', '256', '512'.
            average_box_size: Average box size.
                Must be one of '3', '5', '9', '11'.
            max_velocity: Threshold for eliminating invalid GCPs.
            radius: Radius for hole-filling.
            resampling_type: Method for velocity interpolation.
                Must be one of 'NEAREST_NEIGHBOUR', 'BILINEAR_INTERPOLATION',
                'BICUBIC_INTERPOLATION', 'BISINC_5_POINT_INTERPOLATION', 'CUBIC_CONVOLUTION'.
            spatial_average: Apply spatial averaging.
            fill_holes: Fill holes in velocity field.
            roi_vector: Region of interest vector.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to offset tracking output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PgridAzimuthSpacing={grid_azimuth_spacing}',
            f'-PgridRangeSpacing={grid_range_spacing}',
            f'-PregistrationWindowWidth={registration_window_width}',
            f'-PregistrationWindowHeight={registration_window_height}',
            f'-PxCorrThreshold={x_corr_threshold}',
            f'-PregistrationOversampling={registration_oversampling}',
            f'-PaverageBoxSize={average_box_size}',
            f'-PmaxVelocity={max_velocity}',
            f'-Pradius={radius}',
            f'-PresamplingType={resampling_type}',
            f'-PspatialAverage={str(spatial_average).lower()}',
            f'-PfillHoles={str(fill_holes).lower()}'
        ]
        
        if roi_vector:
            cmd_params.append(f'-ProiVector={roi_vector}')
        
        self.current_cmd.append(f'Offset-Tracking {" ".join(cmd_params)}')
        return self._call(suffix='OFST', output_name=output_name)

    def ndvi(
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
        """The retrieves the Normalized Difference Vegetation Index (NDVI).
        
        This method calculates NDVI from optical imagery, a widely used
        indicator of vegetation health and density.
        
        Args:
            red_source_band: The red band for NDVI computation.
                If not provided, the operator will try to find the best fitting band.
            nir_source_band: The near-infrared band for NDVI computation.
                If not provided, the operator will try to find the best fitting band.
            red_factor: The value of the red source band is multiplied by this value.
            nir_factor: The value of the NIR source band is multiplied by this value.
            resample_type: If selected bands differ in size, the resample method used.
                Must be one of 'None', 'Lowest resolution', 'Highest resolution'.
            upsampling: Method for interpolation (upsampling to finer resolution).
                Must be one of 'Nearest', 'Bilinear', 'Bicubic'.
            downsampling: Method for aggregation (downsampling to coarser resolution).
                Must be one of 'First', 'Min', 'Max', 'Mean', 'Median'.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to NDVI output product, or None if failed.
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
        
        self.current_cmd.append(f'NdviOp {" ".join(cmd_params)}')
        return self._call(suffix='NDVI', output_name=output_name)

    def reproject(
        self,
        crs: Optional[str] = None,
        wkt_file: Optional[str | Path] = None,
        resampling: str = 'Nearest',
        reference_pixel_x: Optional[float] = None,
        reference_pixel_y: Optional[float] = None,
        easting: Optional[float] = None,
        northing: Optional[float] = None,
        orientation: float = 0.0,
        pixel_size_x: Optional[float] = None,
        pixel_size_y: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        tile_size_x: Optional[int] = None,
        tile_size_y: Optional[int] = None,
        orthorectify: bool = False,
        elevation_model_name: Optional[str] = None,
        no_data_value: Optional[float] = None,
        include_tie_point_grids: bool = True,
        add_delta_bands: bool = False,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Reprojection of a source product to a target Coordinate Reference System.
        
        This method transforms geographic coordinates and resamples data to a
        new projection, essential for data integration and standardization.
        
        Args:
            crs: Target Coordinate Reference System in WKT or as authority code.
                Examples: 'EPSG:4326', 'AUTO:42001' (UTM).
            wkt_file: File containing target CRS in WKT format.
            resampling: Resampling method for floating-point raster data.
                Must be one of 'Nearest', 'Bilinear', 'Bicubic'.
            reference_pixel_x: X-position of reference pixel.
            reference_pixel_y: Y-position of reference pixel.
            easting: Easting of reference pixel.
            northing: Northing of reference pixel.
            orientation: Orientation of output product in degrees.
                Valid interval is [-360, 360].
            pixel_size_x: Pixel size in X direction in CRS units.
            pixel_size_y: Pixel size in Y direction in CRS units.
            width: Width of target product.
            height: Height of target product.
            tile_size_x: Tile size in X direction.
            tile_size_y: Tile size in Y direction.
            orthorectify: Whether to orthorectify the source product.
            elevation_model_name: Elevation model name for orthorectification.
            no_data_value: Value used to indicate no-data.
            include_tie_point_grids: Include tie-point grids in output.
            add_delta_bands: Whether to add delta longitude and latitude bands.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to reprojected output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-Presampling={resampling}',
            f'-Porientation={orientation}',
            f'-Porthorectify={str(orthorectify).lower()}',
            f'-PincludeTiePointGrids={str(include_tie_point_grids).lower()}',
            f'-PaddDeltaBands={str(add_delta_bands).lower()}'
        ]
        
        if crs:
            cmd_params.insert(0, f'-Pcrs="{crs}"')
        
        if wkt_file:
            cmd_params.insert(0, f'-PwktFile={Path(wkt_file).as_posix()}')
        
        if reference_pixel_x is not None:
            cmd_params.append(f'-PreferencePixelX={reference_pixel_x}')
        
        if reference_pixel_y is not None:
            cmd_params.append(f'-PreferencePixelY={reference_pixel_y}')
        
        if easting is not None:
            cmd_params.append(f'-Peasting={easting}')
        
        if northing is not None:
            cmd_params.append(f'-Pnorthing={northing}')
        
        if pixel_size_x is not None:
            cmd_params.append(f'-PpixelSizeX={pixel_size_x}')
        
        if pixel_size_y is not None:
            cmd_params.append(f'-PpixelSizeY={pixel_size_y}')
        
        if width is not None:
            cmd_params.append(f'-Pwidth={width}')
        
        if height is not None:
            cmd_params.append(f'-Pheight={height}')
        
        if tile_size_x is not None:
            cmd_params.append(f'-PtileSizeX={tile_size_x}')
        
        if tile_size_y is not None:
            cmd_params.append(f'-PtileSizeY={tile_size_y}')
        
        if elevation_model_name:
            cmd_params.append(f'-PelevationModelName={elevation_model_name}')
        
        if no_data_value is not None:
            cmd_params.append(f'-PnoDataValue={no_data_value}')
        
        self.current_cmd.append(f'Reproject {" ".join(cmd_params)}')
        return self._call(suffix='REPR', output_name=output_name)

    def polarimetric_matrices(
        self,
        matrix: str = 'T3',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Generate covariance or coherency matrix for given product.
        
        This operator creates polarimetric covariance or coherency matrices from
        SAR data, which are fundamental for polarimetric decomposition and
        classification operations.
        
        Args:
            matrix: The covariance or coherency matrix type to generate.
                Must be one of 'C2', 'C3', 'C4', 'T3', 'T4'.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to output product with generated matrix, or None if failed.
        """
        self._reset_command()
        cmd_params = [f'-Pmatrix={matrix}']
        self.current_cmd.append(f'Polarimetric-Matrices {" ".join(cmd_params)}')
        return self._call(suffix='POLMAT', output_name=output_name)

    def polarimetric_speckle_filter(
        self,
        filter: str = 'Refined Lee Filter',
        filter_size: int = 5,
        num_looks_str: str = '1',
        window_size: str = '7x7',
        target_window_size_str: str = '3x3',
        an_size: int = 50,
        sigma_str: str = '0.9',
        search_window_size_str: str = '15',
        patch_size_str: str = '5',
        scale_size_str: str = '1',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Apply polarimetric speckle filtering for noise reduction.
        
        This operator reduces speckle noise in polarimetric SAR data using various
        filtering methods while preserving polarimetric information and structural
        features.
        
        Args:
            filter: The speckle filter to apply.
                Must be one of 'Box Car Filter', 'IDAN Filter', 'Refined Lee Filter',
                'Improved Lee Sigma Filter'.
            filter_size: The filter size. Valid range: (1, 100].
            num_looks_str: Number of looks.
                Must be one of '1', '2', '3', '4'.
            window_size: The window size.
                Must be one of '5x5', '7x7', '9x9', '11x11', '13x13', '15x15', '17x17'.
            target_window_size_str: The target window size.
                Must be one of '3x3', '5x5'.
            an_size: The Adaptive Neighbourhood size. Valid range: (1, 200].
            sigma_str: Sigma parameter.
                Must be one of '0.5', '0.6', '0.7', '0.8', '0.9'.
            search_window_size_str: The search window size.
                Must be one of '3', '5', '7', '9', '11', '13', '15', '17', '19',
                '21', '23', '25'.
            patch_size_str: The patch size.
                Must be one of '3', '5', '7', '9', '11'.
            scale_size_str: The scale size.
                Must be one of '0', '1', '2'.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to filtered output product, or None if failed.
        """
        self._reset_command()
        cmd_params = [
            f'-Pfilter="{filter}"',
            f'-PfilterSize={filter_size}',
            f'-PnumLooksStr={num_looks_str}',
            f'-PwindowSize={window_size}',
            f'-PtargetWindowSizeStr={target_window_size_str}',
            f'-PanSize={an_size}',
            f'-PsigmaStr={sigma_str}',
            f'-PsearchWindowSizeStr={search_window_size_str}',
            f'-PpatchSizeStr={patch_size_str}',
            f'-PscaleSizeStr={scale_size_str}'
        ]
        self.current_cmd.append(f'Polarimetric-Speckle-Filter {" ".join(cmd_params)}')
        return self._call(suffix='POLSPK', output_name=output_name)

    def polarimetric_classification(
        self,
        classification: str = 'H Alpha Wishart',
        window_size: int = 5,
        max_iterations: int = 3,
        num_initial_classes: int = 90,
        num_final_classes: int = 15,
        mixed_category_threshold: float = 0.5,
        decomposition: str = 'Sinclair Decomposition',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Perform polarimetric classification of SAR data.
        
        This operator applies polarimetric classification algorithms to identify
        different scattering mechanisms and land cover types based on polarimetric
        properties.
        
        Args:
            classification: The classification method to use.
                Must be one of 'Cloude-Pottier', 'Cloude-Pottier Dual Pol',
                'H Alpha Wishart', 'H Alpha Wishart Dual Pol',
                'Freeman-Durden Wishart', 'General Wishart'.
            window_size: The sliding window size. Valid range: (1, 100].
            max_iterations: The maximum number of iterations. Valid range: [1, 100].
            num_initial_classes: The initial number of classes. Valid range: [9, 1000].
            num_final_classes: The desired number of classes. Valid range: [9, 100].
            mixed_category_threshold: The threshold for classifying pixels to mixed
                category. Valid range: (0, *).
            decomposition: The polarimetric decomposition method.
                Must be one of 'Sinclair Decomposition', 'Pauli Decomposition',
                'Freeman-Durden Decomposition', 'Generalized Freeman-Durden Decomposition',
                'Yamaguchi Decomposition', 'van Zyl Decomposition',
                'H-A-Alpha Quad Pol Decomposition', 'Cloude Decomposition',
                'Touzi Decomposition'.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to classified output product, or None if failed.
        """
        self._reset_command()
        cmd_params = [
            f'-Pclassification="{classification}"',
            f'-PwindowSize={window_size}',
            f'-PmaxIterations={max_iterations}',
            f'-PnumInitialClasses={num_initial_classes}',
            f'-PnumFinalClasses={num_final_classes}',
            f'-PmixedCategoryThreshold={mixed_category_threshold}',
            f'-Pdecomposition="{decomposition}"'
        ]
        self.current_cmd.append(f'Polarimetric-Classification {" ".join(cmd_params)}')
        return self._call(suffix='POLCLS', output_name=output_name)

    def read_product(
        self,
        file: str | Path,
        format_name: Optional[str] = None,
        source_bands: Optional[List[str]] = None,
        source_masks: Optional[List[str]] = None,
        pixel_region: Optional[str] = None,
        geometry_region: Optional[str] = None,
        vector_file: Optional[str | Path] = None,
        polygon_region: Optional[str] = None,
        use_advanced_options: bool = False,
        copy_metadata: bool = True,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Read a data product from a given file location.
        
        This operator loads SAR data products from various formats and allows
        subsetting during the read operation based on spatial regions or band selection.
        
        Args:
            file: The file from which the data product is read.
            format_name: An optional format name to specify the reader.
            source_bands: List of source band names to read.
            source_masks: List of source mask names to read.
            pixel_region: Subset region in pixel coordinates (format: 'x,y,width,height').
            geometry_region: Subset region in geographical coordinates using WKT-format.
            vector_file: File from which the polygon geometry is read.
            polygon_region: Subset region in geographical coordinates using WKT-format.
            use_advanced_options: Whether to use advanced options for reading.
            copy_metadata: Whether to copy the metadata of the source product.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to read output product, or None if failed.
        """
        self._reset_command()
        
        file_path = Path(file)
        cmd_params = [f'-Pfile={file_path.as_posix()}']
        
        if format_name:
            cmd_params.append(f'-PformatName={format_name}')
        
        if source_bands:
            cmd_params.append(f'-PsourceBands={",".join(source_bands)}')
        
        if source_masks:
            cmd_params.append(f'-PsourceMasks={",".join(source_masks)}')
        
        if pixel_region:
            cmd_params.append(f'-PpixelRegion={pixel_region}')
        
        if geometry_region:
            cmd_params.append(f"-PgeometryRegion='{geometry_region}'")
        
        if vector_file:
            vector_path = Path(vector_file)
            cmd_params.append(f'-PvectorFile={vector_path.as_posix()}')
        
        if polygon_region:
            cmd_params.append(f"-PpolygonRegion='{polygon_region}'")
        
        cmd_params.append(f'-PuseAdvancedOptions={str(use_advanced_options).lower()}')
        cmd_params.append(f'-PcopyMetadata={str(copy_metadata).lower()}')
        
        self.current_cmd.append(f'Read {" ".join(cmd_params)}')
        return self._call(suffix='READ', output_name=output_name)

    def merge_products(
        self,
        geographic_error: float = 1.0e-5,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Merge several source products using a master product as reference.
        
        This operator combines multiple products by aligning them to a master product
        that provides the reference geo-information.
        
        Args:
            geographic_error: Maximum lat/lon error in degrees between products.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to merged output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [f'-PgeographicError={geographic_error}']
        
        self.current_cmd.append(f'Merge {" ".join(cmd_params)}')
        return self._call(suffix='MERGE', output_name=output_name)

    def mosaic(
        self,
        combine: str = 'OR',
        crs: str = 'EPSG:4326',
        east_bound: float = 30.0,
        north_bound: float = 75.0,
        south_bound: float = 35.0,
        west_bound: float = -15.0,
        pixel_size_x: float = 0.05,
        pixel_size_y: float = 0.05,
        orthorectify: bool = False,
        elevation_model_name: Optional[str] = None,
        resampling: str = 'Nearest',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Create a mosaic from a set of source products.
        
        This operator creates a single mosaicked product from multiple input products,
        aligning them to a common coordinate system and spatial resolution.
        
        Args:
            combine: Specifies how conditions are combined. Must be 'OR' or 'AND'.
            crs: The CRS of the target product (WKT or authority code).
            east_bound: The eastern longitude. Valid range: [-180, 180].
            north_bound: The northern latitude. Valid range: [-90, 90].
            south_bound: The southern latitude. Valid range: [-90, 90].
            west_bound: The western longitude. Valid range: [-180, 180].
            pixel_size_x: Size of a pixel in X-direction in map units.
            pixel_size_y: Size of a pixel in Y-direction in map units.
            orthorectify: Whether the source product should be orthorectified.
            elevation_model_name: Elevation model name for orthorectification.
            resampling: Resampling method. Must be 'Nearest', 'Bilinear', or 'Bicubic'.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to mosaic output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-Pcombine={combine}',
            f'-Pcrs="{crs}"',
            f'-PeastBound={east_bound}',
            f'-PnorthBound={north_bound}',
            f'-PsouthBound={south_bound}',
            f'-PwestBound={west_bound}',
            f'-PpixelSizeX={pixel_size_x}',
            f'-PpixelSizeY={pixel_size_y}',
            f'-Porthorectify={str(orthorectify).lower()}',
            f'-Presampling={resampling}'
        ]
        
        if elevation_model_name:
            cmd_params.append(f'-PelevationModelName={elevation_model_name}')
        
        self.current_cmd.append(f'Mosaic {" ".join(cmd_params)}')
        return self._call(suffix='MOSAIC', output_name=output_name)

    def flip(
        self,
        flip_type: str = 'Vertical',
        source_bands: Optional[List[str]] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Flip a product horizontally, vertically, or both.
        
        This operator flips the spatial orientation of a product along the horizontal
        and/or vertical axes.
        
        Args:
            flip_type: The type of flip to apply.
                Must be 'Horizontal', 'Vertical', or 'Horizontal and Vertical'.
            source_bands: List of source bands to flip.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to flipped output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [f'-PflipType="{flip_type}"']
        
        if source_bands:
            cmd_params.append(f'-PsourceBands={",".join(source_bands)}')
        
        self.current_cmd.append(f'Flip {" ".join(cmd_params)}')
        return self._call(suffix='FLIP', output_name=output_name)

    def image_filter(
        self,
        selected_filter_name: Optional[str] = None,
        source_bands: Optional[List[str]] = None,
        user_defined_kernel_file: Optional[str | Path] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Apply common image processing filters.
        
        This operator applies various image filtering operations including edge detection,
        smoothing, sharpening, and custom kernel-based filters.
        
        Args:
            selected_filter_name: The name of the filter to apply.
            source_bands: List of source bands to filter.
            user_defined_kernel_file: Path to file containing a user-defined kernel.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to filtered output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = []
        
        if selected_filter_name:
            cmd_params.append(f'-PselectedFilterName={selected_filter_name}')
        
        if source_bands:
            cmd_params.append(f'-PsourceBands={",".join(source_bands)}')
        
        if user_defined_kernel_file:
            kernel_path = Path(user_defined_kernel_file)
            cmd_params.append(f'-PuserDefinedKernelFile={kernel_path.as_posix()}')
        
        self.current_cmd.append(f'Image-Filter {" ".join(cmd_params)}')
        return self._call(suffix='IMGFLT', output_name=output_name)

    def convert_datatype(
        self,
        target_data_type: str = 'uint8',
        target_scaling_str: str = 'Linear (between 95% clipped histogram)',
        target_no_data_value: float = 0.0,
        source_bands: Optional[List[str]] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Convert product data type.
        
        This operator converts the data type of bands in a product, with various
        scaling options to map the original value range to the target data type.
        
        Args:
            target_data_type: The target data type.
                Must be one of 'int8', 'int16', 'int32', 'uint8', 'uint16',
                'uint32', 'float32', 'float64'.
            target_scaling_str: The scaling method for data type conversion.
                Must be one of 'Truncate', 'Linear (slope and intercept)',
                'Linear (between 95% clipped histogram)',
                'Linear (peak clipped histogram)', 'Logarithmic'.
            target_no_data_value: The no-data value for the target product.
            source_bands: List of source bands to convert.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to converted output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PtargetDataType={target_data_type}',
            f'-PtargetScalingStr="{target_scaling_str}"',
            f'-PtargetNoDataValue={target_no_data_value}'
        ]
        
        if source_bands:
            cmd_params.append(f'-PsourceBands={",".join(source_bands)}')
        
        self.current_cmd.append(f'Convert-Datatype {" ".join(cmd_params)}')
        return self._call(suffix='CVTDT', output_name=output_name)

    def land_sea_mask(
        self,
        land_mask: bool = True,
        use_srtm: bool = True,
        geometry: Optional[str] = None,
        invert_geometry: bool = False,
        shoreline_extension: int = 0,
        source_bands: Optional[List[str]] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Create a bitmask defining land versus ocean areas.
        
        This operator generates a mask to distinguish between land and sea areas,
        using coastline data from SRTM or custom geometry.
        
        Args:
            land_mask: If True, masks land areas; if False, masks sea areas.
            use_srtm: Use SRTM water body data for land/sea determination.
            geometry: Name of geometry to use for masking.
            invert_geometry: Invert the geometry mask.
            shoreline_extension: Distance in pixels to extend the shoreline.
            source_bands: List of source bands to apply the mask to.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to masked output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PlandMask={str(land_mask).lower()}',
            f'-PuseSRTM={str(use_srtm).lower()}',
            f'-PinvertGeometry={str(invert_geometry).lower()}',
            f'-PshorelineExtension={shoreline_extension}'
        ]
        
        if geometry:
            cmd_params.append(f'-Pgeometry={geometry}')
        
        if source_bands:
            cmd_params.append(f'-PsourceBands={",".join(source_bands)}')
        
        self.current_cmd.append(f'Land-Sea-Mask {" ".join(cmd_params)}')
        return self._call(suffix='LSMSK', output_name=output_name)

    def enhanced_spectral_diversity(
        self,
        fine_win_width_str: str = '512',
        fine_win_height_str: str = '512',
        fine_win_acc_azimuth: str = '16',
        fine_win_acc_range: str = '16',
        fine_win_oversampling: str = '128',
        x_corr_threshold: float = 0.1,
        coh_threshold: float = 0.3,
        num_blocks_per_overlap: int = 10,
        esd_estimator: str = 'Periodogram',
        weight_func: str = 'Inv Quadratic',
        temporal_baseline_type: str = 'Number of images',
        max_temporal_baseline: int = 4,
        integration_method: str = 'L1 and L2',
        do_not_write_target_bands: bool = False,
        use_supplied_range_shift: bool = False,
        overall_range_shift: float = 0.0,
        use_supplied_azimuth_shift: bool = False,
        overall_azimuth_shift: float = 0.0,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Estimate constant range and azimuth offsets for a stack of images.
        
        This method performs Enhanced Spectral Diversity (ESD) analysis to compute
        precise azimuth and range offsets between overlapping bursts in TOPS mode data.
        
        Args:
            fine_win_width_str: Width of fine registration window.
            fine_win_height_str: Height of fine registration window.
            fine_win_acc_azimuth: Fine registration azimuth accuracy.
            fine_win_acc_range: Fine registration range accuracy.
            fine_win_oversampling: Fine registration oversampling factor.
            x_corr_threshold: Peak cross-correlation threshold.
            coh_threshold: Coherence threshold for outlier removal.
            num_blocks_per_overlap: Number of windows per overlap for ESD.
            esd_estimator: ESD estimator used for azimuth shift computation.
            weight_func: Weight function of coherence for azimuth shift estimation.
            temporal_baseline_type: Baseline type for building integration network.
            max_temporal_baseline: Maximum temporal baseline between image pairs.
            integration_method: Method used for integrating shifts network.
            do_not_write_target_bands: Do not write target bands.
            use_supplied_range_shift: Use user supplied range shift.
            overall_range_shift: The overall range shift value.
            use_supplied_azimuth_shift: Use user supplied azimuth shift.
            overall_azimuth_shift: The overall azimuth shift value.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to ESD-corrected output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PfineWinWidthStr={fine_win_width_str}',
            f'-PfineWinHeightStr={fine_win_height_str}',
            f'-PfineWinAccAzimuth={fine_win_acc_azimuth}',
            f'-PfineWinAccRange={fine_win_acc_range}',
            f'-PfineWinOversampling={fine_win_oversampling}',
            f'-PxCorrThreshold={x_corr_threshold}',
            f'-PcohThreshold={coh_threshold}',
            f'-PnumBlocksPerOverlap={num_blocks_per_overlap}',
            f'-PesdEstimator={esd_estimator}',
            f'-PweightFunc="{weight_func}"',
            f'-PtemporalBaselineType="{temporal_baseline_type}"',
            f'-PmaxTemporalBaseline={max_temporal_baseline}',
            f'-PintegrationMethod="{integration_method}"',
            f'-PdoNotWriteTargetBands={str(do_not_write_target_bands).lower()}',
            f'-PuseSuppliedRangeShift={str(use_supplied_range_shift).lower()}',
            f'-PoverallRangeShift={overall_range_shift}',
            f'-PuseSuppliedAzimuthShift={str(use_supplied_azimuth_shift).lower()}',
            f'-PoverallAzimuthShift={overall_azimuth_shift}'
        ]
        
        self.current_cmd.append(f'Enhanced-Spectral-Diversity {" ".join(cmd_params)}')
        return self._call(suffix='ESD', output_name=output_name)

    def phase_to_displacement(
        self,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Convert unwrapped phase to displacement along line of sight.
        
        This method performs phase-to-displacement conversion, translating interferometric
        phase measurements into physical displacement values.
        
        Args:
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to displacement output product, or None if failed.
        """
        self._reset_command()
        self.current_cmd.append('PhaseToDisplacement')
        return self._call(suffix='DISP', output_name=output_name)

    def phase_to_elevation(
        self,
        dem_name: str = 'SRTM 3Sec',
        dem_resampling_method: str = 'BILINEAR_INTERPOLATION',
        external_dem_file: Optional[str | Path] = None,
        external_dem_no_data_value: float = 0.0,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Convert interferometric phase to elevation (DEM generation).
        
        This method generates a digital elevation model from interferometric phase data.
        
        Args:
            dem_name: The digital elevation model to use as reference.
            dem_resampling_method: DEM resampling method.
            external_dem_file: Path to external DEM file.
            external_dem_no_data_value: No data value for external DEM.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to elevation output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PdemName="{dem_name}"',
            f'-PdemResamplingMethod={dem_resampling_method}',
            f'-PexternalDEMNoDataValue={external_dem_no_data_value}'
        ]
        
        if external_dem_file:
            cmd_params.append(f'-PexternalDEMFile={Path(external_dem_file).as_posix()}')
        
        self.current_cmd.append(f'PhaseToElevation {" ".join(cmd_params)}')
        return self._call(suffix='ELEV', output_name=output_name)

    def phase_to_height(
        self,
        n_points: int = 200,
        n_heights: int = 3,
        degree_1d: int = 2,
        degree_2d: int = 5,
        orbit_degree: int = 3,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Convert unwrapped phase to height.
        
        This method performs phase-to-height conversion using polynomial fitting.
        
        Args:
            n_points: Number of points for evaluation of flat earth phase.
            n_heights: Number of height samples in range [0,5000).
            degree_1d: Degree of 1D polynomial to fit reference phase through.
            degree_2d: Degree of 2D polynomial to fit reference phase through.
            orbit_degree: Degree of orbit (polynomial) interpolator.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to height output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PnPoints={n_points}',
            f'-PnHeights={n_heights}',
            f'-Pdegree1D={degree_1d}',
            f'-Pdegree2D={degree_2d}',
            f'-PorbitDegree={orbit_degree}'
        ]
        
        self.current_cmd.append(f'PhaseToHeight {" ".join(cmd_params)}')
        return self._call(suffix='HEIGHT', output_name=output_name)

    def snaphu_export(
        self,
        snaphu_processing_location: Optional[str | Path] = None,
        snaphu_install_location: Optional[str | Path] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Download and execute SNAPHU on interferograms.
        
        This method exports interferograms in SNAPHU format for phase unwrapping.
        
        Args:
            snaphu_processing_location: Directory for SNAPHU processing.
            snaphu_install_location: Directory to install SNAPHU binary.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to SNAPHU export output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = []
        
        if snaphu_processing_location:
            cmd_params.append(f'-PsnaphuProcessingLocation={Path(snaphu_processing_location).as_posix()}')
        
        if snaphu_install_location:
            cmd_params.append(f'-PsnaphuInstallLocation={Path(snaphu_install_location).as_posix()}')
        
        self.current_cmd.append(f'BatchSnaphuUnwrapOp {" ".join(cmd_params)}')
        return self._call(suffix='SNAPHU', output_name=output_name)

    def cross_correlation(
        self,
        num_gcp_to_generate: int = 2000,
        coarse_registration_window_width: str = '128',
        coarse_registration_window_height: str = '128',
        row_interp_factor: str = '2',
        column_interp_factor: str = '2',
        max_iteration: int = 10,
        gcp_tolerance: float = 0.5,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Automatic GCP selection using cross-correlation.
        
        This method performs automatic GCP selection and coregistration.
        
        Args:
            num_gcp_to_generate: Number of GCPs to use in a grid.
            coarse_registration_window_width: Coarse registration window width.
            coarse_registration_window_height: Coarse registration window height.
            row_interp_factor: Row interpolation factor.
            column_interp_factor: Column interpolation factor.
            max_iteration: Maximum number of iterations.
            gcp_tolerance: Tolerance in slave GCP validation check.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to coregistered output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PnumGCPtoGenerate={num_gcp_to_generate}',
            f'-PcoarseRegistrationWindowWidth={coarse_registration_window_width}',
            f'-PcoarseRegistrationWindowHeight={coarse_registration_window_height}',
            f'-ProwInterpFactor={row_interp_factor}',
            f'-PcolumnInterpFactor={column_interp_factor}',
            f'-PmaxIteration={max_iteration}',
            f'-PgcpTolerance={gcp_tolerance}'
        ]
        
        self.current_cmd.append(f'Cross-Correlation {" ".join(cmd_params)}')
        return self._call(suffix='XCORR', output_name=output_name)

    def dem_assisted_coregistration(
        self,
        dem_name: str = 'SRTM 3Sec',
        dem_resampling_method: str = 'BICUBIC_INTERPOLATION',
        external_dem_file: Optional[str | Path] = None,
        external_dem_no_data_value: float = 0.0,
        resampling_type: str = 'BISINC_5_POINT_INTERPOLATION',
        mask_out_area_without_elevation: bool = True,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Perform orbit and DEM based coregistration.
        
        This method uses orbit and DEM for geometric coregistration.
        
        Args:
            dem_name: The digital elevation model to use.
            dem_resampling_method: DEM resampling method.
            external_dem_file: Path to external DEM file.
            external_dem_no_data_value: No data value for external DEM.
            resampling_type: Method for resampling slave grid onto master grid.
            mask_out_area_without_elevation: Mask out areas without elevation.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to coregistered output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PdemName="{dem_name}"',
            f'-PdemResamplingMethod={dem_resampling_method}',
            f'-PexternalDEMNoDataValue={external_dem_no_data_value}',
            f'-PresamplingType={resampling_type}',
            f'-PmaskOutAreaWithoutElevation={str(mask_out_area_without_elevation).lower()}'
        ]
        
        if external_dem_file:
            cmd_params.append(f'-PexternalDEMFile={Path(external_dem_file).as_posix()}')
        
        self.current_cmd.append(f'DEM-Assisted-Coregistration {" ".join(cmd_params)}')
        return self._call(suffix='DEMCOREG', output_name=output_name)

    def eap_phase_correction(
        self,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Apply EAP (Equivalent Aperture Position) phase correction.
        
        This method corrects phase errors from platform motion.
        
        Args:
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to phase-corrected output product, or None if failed.
        """
        self._reset_command()
        self.current_cmd.append('EAP-Phase-Correction')
        return self._call(suffix='EAPCORR', output_name=output_name)

    def ellipsoid_correction_rd(
        self,
        source_bands: Optional[List[str]] = None,
        pixel_spacing_in_meter: float = 0.0,
        map_projection: str = 'WGS84(DD)',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Apply ellipsoid correction with Range-Doppler method.
        
        This method performs geometric terrain correction using ellipsoid model.
        
        Args:
            source_bands: List of source bands to process.
            pixel_spacing_in_meter: Pixel spacing in meters.
            map_projection: Coordinate reference system in WKT format.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to terrain-corrected output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PpixelSpacingInMeter={pixel_spacing_in_meter}',
            f'-PmapProjection="{map_projection}"'
        ]
        
        if source_bands:
            cmd_params.insert(0, f'-PsourceBands={",".join(source_bands)}')
        
        self.current_cmd.append(f'Ellipsoid-Correction-RD {" ".join(cmd_params)}')
        return self._call(suffix='EC', output_name=output_name)

    def range_filter(
        self,
        fft_length: int = 8,
        alpha_hamming: float = 0.75,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Apply range filtering for spectral analysis.
        
        This method performs range direction filtering in frequency domain.
        
        Args:
            fft_length: Length of filtering window.
            alpha_hamming: Weight for Hamming filter.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to range-filtered output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PfftLength={fft_length}',
            f'-PalphaHamming={alpha_hamming}'
        ]
        
        self.current_cmd.append(f'RangeFilter {" ".join(cmd_params)}')
        return self._call(suffix='RGFLT', output_name=output_name)

    def azimuth_filter(
        self,
        fft_length: int = 256,
        alpha_hamming: float = 0.75,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Apply azimuth filtering for spectral analysis.
        
        This method performs azimuth direction filtering in frequency domain.
        
        Args:
            fft_length: Length of filtering window.
            alpha_hamming: Weight for Hamming filter.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to azimuth-filtered output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PfftLength={fft_length}',
            f'-PalphaHamming={alpha_hamming}'
        ]
        
        self.current_cmd.append(f'AzimuthFilter {" ".join(cmd_params)}')
        return self._call(suffix='AZFLT', output_name=output_name)

    def band_pass_filter(
        self,
        subband: str = 'low',
        alpha: float = 1.0,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Create basebanded SLC based on a subband of 1/3 the original bandwidth.
        
        This method applies band-pass filter to extract specific frequency subband.
        
        Args:
            subband: Subband selection ('low' or 'high').
            alpha: Hamming alpha parameter.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to band-pass filtered output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-Psubband={subband}',
            f'-Palpha={alpha}'
        ]
        
        self.current_cmd.append(f'BandPassFilter {" ".join(cmd_params)}')
        return self._call(suffix='BPF', output_name=output_name)

    def oversample(
        self,
        source_bands: Optional[List[str]] = None,
        output_image_by: str = 'Ratio',
        width_ratio: float = 2.0,
        height_ratio: float = 2.0,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Oversample the dataset to increase spatial resolution.
        
        This method increases pixel density through interpolation.
        
        Args:
            source_bands: List of source bands to oversample.
            output_image_by: Method to specify output dimensions.
            width_ratio: Width ratio of output/input images.
            height_ratio: Height ratio of output/input images.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to oversampled output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PoutputImageBy="{output_image_by}"',
            f'-PwidthRatio={width_ratio}',
            f'-PheightRatio={height_ratio}'
        ]
        
        if source_bands:
            cmd_params.insert(0, f'-PsourceBands={",".join(source_bands)}')
        
        self.current_cmd.append(f'Oversample {" ".join(cmd_params)}')
        return self._call(suffix='OVER', output_name=output_name)

    def glcm(
        self,
        source_bands: Optional[List[str]] = None,
        window_size_str: str = '9x9',
        angle_str: str = 'ALL',
        quantization_levels_str: str = '32',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Extract texture features using Gray Level Co-occurrence Matrix.
        
        This method computes texture features from SAR imagery.
        
        Args:
            source_bands: List of source bands for texture analysis.
            window_size_str: Size of the analysis window.
            angle_str: Angle for co-occurrence matrix computation.
            quantization_levels_str: Number of quantization levels.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to GLCM output product with texture features, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PwindowSizeStr="{window_size_str}"',
            f'-PangleStr={angle_str}',
            f'-PquantizationLevelsStr={quantization_levels_str}'
        ]
        
        if source_bands:
            cmd_params.insert(0, f'-PsourceBands={",".join(source_bands)}')
        
        self.current_cmd.append(f'GLCM {" ".join(cmd_params)}')
        return self._call(suffix='GLCM', output_name=output_name)

    def pca(
        self,
        source_band_names: Optional[List[str]] = None,
        component_count: int = -1,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Perform Principal Component Analysis (PCA).
        
        This method transforms correlated bands into uncorrelated components.
        
        Args:
            source_band_names: Names of bands to use for analysis.
            component_count: Maximum number of components (-1 for all).
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to PCA output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [f'-PcomponentCount={component_count}']
        
        if source_band_names:
            cmd_params.insert(0, f'-PsourceBandNames={",".join(source_band_names)}')
        
        self.current_cmd.append(f'PCA {" ".join(cmd_params)}')
        return self._call(suffix='PCA', output_name=output_name)

    def k_means_cluster_analysis(
        self,
        source_band_names: Optional[List[str]] = None,
        cluster_count: int = 14,
        iteration_count: int = 30,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Perform K-Means cluster analysis.
        
        This method performs unsupervised classification using K-Means.
        
        Args:
            source_band_names: Names of bands to use for cluster analysis.
            cluster_count: Number of clusters.
            iteration_count: Number of iterations.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to clustered output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PclusterCount={cluster_count}',
            f'-PiterationCount={iteration_count}'
        ]
        
        if source_band_names:
            cmd_params.insert(0, f'-PsourceBandNames={",".join(source_band_names)}')
        
        self.current_cmd.append(f'KMeansClusterAnalysis {" ".join(cmd_params)}')
        return self._call(suffix='KMEANS', output_name=output_name)

    def em_cluster_analysis(
        self,
        source_band_names: Optional[List[str]] = None,
        cluster_count: int = 14,
        iteration_count: int = 30,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Perform Expectation-Maximization cluster analysis.
        
        This method performs probabilistic clustering using EM algorithm.
        
        Args:
            source_band_names: Names of bands to use for cluster analysis.
            cluster_count: Number of clusters.
            iteration_count: Number of iterations.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to clustered output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [
            f'-PclusterCount={cluster_count}',
            f'-PiterationCount={iteration_count}'
        ]
        
        if source_band_names:
            cmd_params.insert(0, f'-PsourceBandNames={",".join(source_band_names)}')
        
        self.current_cmd.append(f'EMClusterAnalysis {" ".join(cmd_params)}')
        return self._call(suffix='EM', output_name=output_name)

    def random_forest_classifier(
        self,
        tree_count: int = 10,
        feature_bands: Optional[List[str]] = None,
        training_vectors: Optional[List[str]] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Perform Random Forest classification.
        
        This method performs supervised classification using Random Forest.
        
        Args:
            tree_count: Number of trees in the forest.
            feature_bands: Names of bands to use as features.
            training_vectors: Vectors to train on.
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to classified output product, or None if failed.
        """
        self._reset_command()
        
        cmd_params = [f'-PtreeCount={tree_count}']
        
        if feature_bands:
            cmd_params.append(f'-PfeatureBands={",".join(feature_bands)}')
        
        if training_vectors:
            cmd_params.append(f'-PtrainingVectors={",".join(training_vectors)}')
        
        self.current_cmd.append(f'Random-Forest-Classifier {" ".join(cmd_params)}')
        return self._call(suffix='RF', output_name=output_name)

    def decision_tree(
        self,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Perform Decision Tree classification.
        
        This method performs supervised classification using decision trees.
        
        Args:
            output_name: Custom output filename (without extension).
        
        Returns:
            Path to classified output product, or None if failed.
        """
        self._reset_command()
        self.current_cmd.append('DecisionTree')
        return self._call(suffix='DT', output_name=output_name)

    def arvi(
        self,
        red_source_band: Optional[str] = None,
        nir_source_band: Optional[str] = None,
        blue_source_band: Optional[str] = None,
        red_factor: float = 1.0,
        nir_factor: float = 1.0,
        blue_factor: float = 1.0,
        gamma_parameter: float = 1.0,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compute Atmospherically Resistant Vegetation Index."""
        self._reset_command()
        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PredFactor={red_factor}',
            f'-PnirFactor={nir_factor}',
            f'-PblueFactor={blue_factor}',
            f'-PgammaParameter={gamma_parameter}'
        ]
        if red_source_band:
            cmd_params.append(f'-PredSourceBand={red_source_band}')
        if nir_source_band:
            cmd_params.append(f'-PnirSourceBand={nir_source_band}')
        if blue_source_band:
            cmd_params.append(f'-PblueSourceBand={blue_source_band}')
        self.current_cmd.append(f'ArviOp {" ".join(cmd_params)}')
        return self._call(suffix='ARVI', output_name=output_name)

    def dvi(
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
        """Compute Difference Vegetation Index."""
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
        self.current_cmd.append(f'DviOp {" ".join(cmd_params)}')
        return self._call(suffix='DVI', output_name=output_name)

    def gemi(
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
        """Compute Global Environmental Monitoring Index."""
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
        self.current_cmd.append(f'GemiOp {" ".join(cmd_params)}')
        return self._call(suffix='GEMI', output_name=output_name)

    def gndvi(
        self,
        green_source_band: Optional[str] = None,
        nir_source_band: Optional[str] = None,
        green_factor: float = 1.0,
        nir_factor: float = 1.0,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compute Green Normalized Difference Vegetation Index."""
        self._reset_command()
        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PgreenFactor={green_factor}',
            f'-PnirFactor={nir_factor}'
        ]
        if green_source_band:
            cmd_params.append(f'-PgreenSourceBand={green_source_band}')
        if nir_source_band:
            cmd_params.append(f'-PnirSourceBand={nir_source_band}')
        self.current_cmd.append(f'GndviOp {" ".join(cmd_params)}')
        return self._call(suffix='GNDVI', output_name=output_name)

    def ipvi(
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
        """Compute Infrared Percentage Vegetation Index."""
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
        self.current_cmd.append(f'IpviOp {" ".join(cmd_params)}')
        return self._call(suffix='IPVI', output_name=output_name)

    def mcari(
        self,
        red1_source_band: Optional[str] = None,
        red2_source_band: Optional[str] = None,
        green_source_band: Optional[str] = None,
        red1_factor: float = 1.0,
        red2_factor: float = 1.0,
        green_factor: float = 1.0,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compute Modified Chlorophyll Absorption Ratio Index."""
        self._reset_command()
        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-Pred1Factor={red1_factor}',
            f'-Pred2Factor={red2_factor}',
            f'-PgreenFactor={green_factor}'
        ]
        if red1_source_band:
            cmd_params.append(f'-Pred1SourceBand={red1_source_band}')
        if red2_source_band:
            cmd_params.append(f'-Pred2SourceBand={red2_source_band}')
        if green_source_band:
            cmd_params.append(f'-PgreenSourceBand={green_source_band}')
        self.current_cmd.append(f'McariOp {" ".join(cmd_params)}')
        return self._call(suffix='MCARI', output_name=output_name)

    def mndwi(
        self,
        green_source_band: Optional[str] = None,
        mir_source_band: Optional[str] = None,
        green_factor: float = 1.0,
        mir_factor: float = 1.0,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compute Modified Normalized Difference Water Index."""
        self._reset_command()
        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PgreenFactor={green_factor}',
            f'-PmirFactor={mir_factor}'
        ]
        if green_source_band:
            cmd_params.append(f'-PgreenSourceBand={green_source_band}')
        if mir_source_band:
            cmd_params.append(f'-PmirSourceBand={mir_source_band}')
        self.current_cmd.append(f'MndwiOp {" ".join(cmd_params)}')
        return self._call(suffix='MNDWI', output_name=output_name)

    def msavi2(
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
        """Compute second Modified Soil Adjusted Vegetation Index."""
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
        self.current_cmd.append(f'Msavi2Op {" ".join(cmd_params)}')
        return self._call(suffix='MSAVI2', output_name=output_name)

    def msavi(
        self,
        red_source_band: Optional[str] = None,
        nir_source_band: Optional[str] = None,
        red_factor: float = 1.0,
        nir_factor: float = 1.0,
        slope: float = 0.5,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compute Modified Soil Adjusted Vegetation Index."""
        self._reset_command()
        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PredFactor={red_factor}',
            f'-PnirFactor={nir_factor}',
            f'-Pslope={slope}'
        ]
        if red_source_band:
            cmd_params.append(f'-PredSourceBand={red_source_band}')
        if nir_source_band:
            cmd_params.append(f'-PnirSourceBand={nir_source_band}')
        self.current_cmd.append(f'MsaviOp {" ".join(cmd_params)}')
        return self._call(suffix='MSAVI', output_name=output_name)

    def mtci(
        self,
        red_source_band4: Optional[str] = None,
        red_source_band5: Optional[str] = None,
        nir_source_band: Optional[str] = None,
        red_b4_factor: float = 1.0,
        red_b5_factor: float = 1.0,
        nir_factor: float = 1.0,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compute MERIS Terrestrial Chlorophyll Index."""
        self._reset_command()
        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PredB4Factor={red_b4_factor}',
            f'-PredB5Factor={red_b5_factor}',
            f'-PnirFactor={nir_factor}'
        ]
        if red_source_band4:
            cmd_params.append(f'-PredSourceBand4={red_source_band4}')
        if red_source_band5:
            cmd_params.append(f'-PredSourceBand5={red_source_band5}')
        if nir_source_band:
            cmd_params.append(f'-PnirSourceBand={nir_source_band}')
        self.current_cmd.append(f'MtciOp {" ".join(cmd_params)}')
        return self._call(suffix='MTCI', output_name=output_name)

    def ndwi(
        self,
        nir_source_band: Optional[str] = None,
        mir_source_band: Optional[str] = None,
        nir_factor: float = 1.0,
        mir_factor: float = 1.0,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compute Normalized Difference Water Index."""
        self._reset_command()
        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PnirFactor={nir_factor}',
            f'-PmirFactor={mir_factor}'
        ]
        if nir_source_band:
            cmd_params.append(f'-PnirSourceBand={nir_source_band}')
        if mir_source_band:
            cmd_params.append(f'-PmirSourceBand={mir_source_band}')
        self.current_cmd.append(f'NdwiOp {" ".join(cmd_params)}')
        return self._call(suffix='NDWI', output_name=output_name)

    def rvi(
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
        """Compute Ratio Vegetation Index."""
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
        self.current_cmd.append(f'RviOp {" ".join(cmd_params)}')
        return self._call(suffix='RVI', output_name=output_name)

    def bi2(
        self,
        red_source_band: Optional[str] = None,
        green_source_band: Optional[str] = None,
        nir_source_band: Optional[str] = None,
        red_factor: float = 1.0,
        green_factor: float = 1.0,
        nir_factor: float = 1.0,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Second Brightness Index."""
        self._reset_command()
        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PredFactor={red_factor}',
            f'-PgreenFactor={green_factor}',
            f'-PnirFactor={nir_factor}'
        ]
        if red_source_band:
            cmd_params.append(f'-PredSourceBand={red_source_band}')
        if green_source_band:
            cmd_params.append(f'-PgreenSourceBand={green_source_band}')
        if nir_source_band:
            cmd_params.append(f'-PnirSourceBand={nir_source_band}')
        self.current_cmd.append(f'Bi2Op {" ".join(cmd_params)}')
        return self._call(suffix='BI2', output_name=output_name)

    def bi(
        self,
        red_source_band: Optional[str] = None,
        green_source_band: Optional[str] = None,
        red_factor: float = 1.0,
        green_factor: float = 1.0,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Brightness Index."""
        self._reset_command()
        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PredFactor={red_factor}',
            f'-PgreenFactor={green_factor}'
        ]
        if red_source_band:
            cmd_params.append(f'-PredSourceBand={red_source_band}')
        if green_source_band:
            cmd_params.append(f'-PgreenSourceBand={green_source_band}')
        self.current_cmd.append(f'BiOp {" ".join(cmd_params)}')
        return self._call(suffix='BI', output_name=output_name)

    def ci(
        self,
        red_source_band: Optional[str] = None,
        green_source_band: Optional[str] = None,
        red_factor: float = 1.0,
        green_factor: float = 1.0,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Colour Index."""
        self._reset_command()
        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PredFactor={red_factor}',
            f'-PgreenFactor={green_factor}'
        ]
        if red_source_band:
            cmd_params.append(f'-PredSourceBand={red_source_band}')
        if green_source_band:
            cmd_params.append(f'-PgreenSourceBand={green_source_band}')
        self.current_cmd.append(f'CiOp {" ".join(cmd_params)}')
        return self._call(suffix='CI', output_name=output_name)

    def ireci(
        self,
        red_source_band4: Optional[str] = None,
        red_source_band5: Optional[str] = None,
        red_source_band6: Optional[str] = None,
        nir_source_band: Optional[str] = None,
        red_b4_factor: float = 1.0,
        red_b5_factor: float = 1.0,
        red_b6_factor: float = 1.0,
        nir_factor: float = 1.0,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Inverted Red-Edge Chlorophyll."""
        self._reset_command()
        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PredB4Factor={red_b4_factor}',
            f'-PredB5Factor={red_b5_factor}',
            f'-PredB6Factor={red_b6_factor}',
            f'-PnirFactor={nir_factor}'
        ]
        if red_source_band4:
            cmd_params.append(f'-PredSourceBand4={red_source_band4}')
        if red_source_band5:
            cmd_params.append(f'-PredSourceBand5={red_source_band5}')
        if red_source_band6:
            cmd_params.append(f'-PredSourceBand6={red_source_band6}')
        if nir_source_band:
            cmd_params.append(f'-PnirSourceBand={nir_source_band}')
        self.current_cmd.append(f'IreciOp {" ".join(cmd_params)}')
        return self._call(suffix='IRECI', output_name=output_name)

    def ndi45(
        self,
        red_source_band4: Optional[str] = None,
        red_source_band5: Optional[str] = None,
        red_b4_factor: float = 1.0,
        red_b5_factor: float = 1.0,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Normalized Difference Index."""
        self._reset_command()
        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PredB4Factor={red_b4_factor}',
            f'-PredB5Factor={red_b5_factor}'
        ]
        if red_source_band4:
            cmd_params.append(f'-PredSourceBand4={red_source_band4}')
        if red_source_band5:
            cmd_params.append(f'-PredSourceBand5={red_source_band5}')
        self.current_cmd.append(f'Ndi45Op {" ".join(cmd_params)}')
        return self._call(suffix='NDI45', output_name=output_name)

    def ndpi(
        self,
        green_source_band: Optional[str] = None,
        mir_source_band: Optional[str] = None,
        green_factor: float = 1.0,
        mir_factor: float = 1.0,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Normalized Differential Pond."""
        self._reset_command()
        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PgreenFactor={green_factor}',
            f'-PmirFactor={mir_factor}'
        ]
        if green_source_band:
            cmd_params.append(f'-PgreenSourceBand={green_source_band}')
        if mir_source_band:
            cmd_params.append(f'-PmirSourceBand={mir_source_band}')
        self.current_cmd.append(f'NdpiOp {" ".join(cmd_params)}')
        return self._call(suffix='NDPI', output_name=output_name)

    def ndti(
        self,
        red_source_band: Optional[str] = None,
        green_source_band: Optional[str] = None,
        red_factor: float = 1.0,
        green_factor: float = 1.0,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Normalized Difference Turbidity."""
        self._reset_command()
        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PredFactor={red_factor}',
            f'-PgreenFactor={green_factor}'
        ]
        if red_source_band:
            cmd_params.append(f'-PredSourceBand={red_source_band}')
        if green_source_band:
            cmd_params.append(f'-PgreenSourceBand={green_source_band}')
        self.current_cmd.append(f'NdtiOp {" ".join(cmd_params)}')
        return self._call(suffix='NDTI', output_name=output_name)

    def ndwi2(
        self,
        green_source_band: Optional[str] = None,
        nir_source_band: Optional[str] = None,
        green_factor: float = 1.0,
        nir_factor: float = 1.0,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Normalized Difference Water."""
        self._reset_command()
        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PgreenFactor={green_factor}',
            f'-PnirFactor={nir_factor}'
        ]
        if green_source_band:
            cmd_params.append(f'-PgreenSourceBand={green_source_band}')
        if nir_source_band:
            cmd_params.append(f'-PnirSourceBand={nir_source_band}')
        self.current_cmd.append(f'Ndwi2Op {" ".join(cmd_params)}')
        return self._call(suffix='NDWI2', output_name=output_name)

    def pssra(
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
        """Pigment Specific Simple."""
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
        self.current_cmd.append(f'PssraOp {" ".join(cmd_params)}')
        return self._call(suffix='PSSRA', output_name=output_name)

    def pvi(
        self,
        red_source_band: Optional[str] = None,
        nir_source_band: Optional[str] = None,
        red_factor: float = 1.0,
        nir_factor: float = 1.0,
        angle_soil_line_nir_axis: float = 45.0,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Perpendicular Vegetation Index."""
        self._reset_command()
        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PredFactor={red_factor}',
            f'-PnirFactor={nir_factor}',
            f'-PangleSoilLineNIRAxis={angle_soil_line_nir_axis}'
        ]
        if red_source_band:
            cmd_params.append(f'-PredSourceBand={red_source_band}')
        if nir_source_band:
            cmd_params.append(f'-PnirSourceBand={nir_source_band}')
        self.current_cmd.append(f'PviOp {" ".join(cmd_params)}')
        return self._call(suffix='PVI', output_name=output_name)

    def reip(
        self,
        red_source_band4: Optional[str] = None,
        red_source_band5: Optional[str] = None,
        red_source_band6: Optional[str] = None,
        nir_source_band: Optional[str] = None,
        red_b4_factor: float = 1.0,
        red_b5_factor: float = 1.0,
        red_b6_factor: float = 1.0,
        nir_factor: float = 1.0,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Red Edge Inflection."""
        self._reset_command()
        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PredB4Factor={red_b4_factor}',
            f'-PredB5Factor={red_b5_factor}',
            f'-PredB6Factor={red_b6_factor}',
            f'-PnirFactor={nir_factor}'
        ]
        if red_source_band4:
            cmd_params.append(f'-PredSourceBand4={red_source_band4}')
        if red_source_band5:
            cmd_params.append(f'-PredSourceBand5={red_source_band5}')
        if red_source_band6:
            cmd_params.append(f'-PredSourceBand6={red_source_band6}')
        if nir_source_band:
            cmd_params.append(f'-PnirSourceBand={nir_source_band}')
        self.current_cmd.append(f'ReipOp {" ".join(cmd_params)}')
        return self._call(suffix='REIP', output_name=output_name)

    def ri(
        self,
        red_source_band: Optional[str] = None,
        green_source_band: Optional[str] = None,
        red_factor: float = 1.0,
        green_factor: float = 1.0,
        resample_type: str = 'None',
        upsampling: str = 'Nearest',
        downsampling: str = 'First',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Redness Index."""
        self._reset_command()
        cmd_params = [
            f'-PresampleType="{resample_type}"',
            f'-Pupsampling={upsampling}',
            f'-Pdownsampling={downsampling}',
            f'-PredFactor={red_factor}',
            f'-PgreenFactor={green_factor}'
        ]
        if red_source_band:
            cmd_params.append(f'-PredSourceBand={red_source_band}')
        if green_source_band:
            cmd_params.append(f'-PgreenSourceBand={green_source_band}')
        self.current_cmd.append(f'RiOp {" ".join(cmd_params)}')
        return self._call(suffix='RI', output_name=output_name)

    def c2rcc_msi(
        self,
        valid_pixel_expression: Optional[str] = 'B8 > 0 && B8 < 0.1',
        salinity: float = 35.0,
        temperature: float = 15.0,
        ozone: float = 330.0,
        press: float = 1000.0,
        elevation: float = 0.0,
        tsm_fac: float = 1.06,
        tsm_exp: float = 0.942,
        chl_exp: float = 1.04,
        chl_fac: float = 21.0,
        threshold_rtosa_oos: float = 0.05,
        threshold_ac_reflec_oos: float = 0.1,
        threshold_cloud_t_down865: float = 0.955,
        atmospheric_aux_data_path: Optional[str] = None,
        alternative_nn_path: Optional[str] = None,
        net_set: str = 'C2RCC-Nets',
        output_as_rrs: bool = False,
        derive_rw_from_path_and_transmittance: bool = False,
        use_ecmwf_aux_data: bool = False,
        dem_name: str = 'Copernicus 90m Global DEM',
        output_rtoa: bool = True,
        output_rtosa_gc: bool = False,
        output_rtosa_gc_aann: bool = False,
        output_rpath: bool = False,
        output_tdown: bool = False,
        output_tup: bool = False,
        output_ac_reflectance: bool = True,
        output_rhown: bool = True,
        output_oos: bool = False,
        output_kd: bool = True,
        output_uncertainties: bool = True,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """C2RCC MSI atmospheric correction."""
        self._reset_command()
        cmd_params = [
            f'-Psalinity={salinity}',
            f'-Ptemperature={temperature}',
            f'-Pozone={ozone}',
            f'-Ppress={press}',
            f'-Pelevation={elevation}',
            f'-PTSMfac={tsm_fac}',
            f'-PTSMexp={tsm_exp}',
            f'-PCHLexp={chl_exp}',
            f'-PCHLfac={chl_fac}',
            f'-PthresholdRtosaOOS={threshold_rtosa_oos}',
            f'-PthresholdAcReflecOos={threshold_ac_reflec_oos}',
            f'-PthresholdCloudTDown865={threshold_cloud_t_down865}',
            f'-PnetSet="{net_set}"',
            f'-PoutputAsRrs={str(output_as_rrs).lower()}',
            f'-PderiveRwFromPathAndTransmittance={str(derive_rw_from_path_and_transmittance).lower()}',
            f'-PuseEcmwfAuxData={str(use_ecmwf_aux_data).lower()}',
            f'-PdemName="{dem_name}"',
            f'-PoutputRtoa={str(output_rtoa).lower()}',
            f'-PoutputRtosaGc={str(output_rtosa_gc).lower()}',
            f'-PoutputRtosaGcAann={str(output_rtosa_gc_aann).lower()}',
            f'-PoutputRpath={str(output_rpath).lower()}',
            f'-PoutputTdown={str(output_tdown).lower()}',
            f'-PoutputTup={str(output_tup).lower()}',
            f'-PoutputAcReflectance={str(output_ac_reflectance).lower()}',
            f'-PoutputRhown={str(output_rhown).lower()}',
            f'-PoutputOos={str(output_oos).lower()}',
            f'-PoutputKd={str(output_kd).lower()}',
            f'-PoutputUncertainties={str(output_uncertainties).lower()}'
        ]
        if valid_pixel_expression:
            cmd_params.append(f'-PvalidPixelExpression="{valid_pixel_expression}"')
        if atmospheric_aux_data_path:
            cmd_params.append(f'-PatmosphericAuxDataPath="{atmospheric_aux_data_path}"')
        if alternative_nn_path:
            cmd_params.append(f'-PalternativeNNPath="{alternative_nn_path}"')
        self.current_cmd.append(f'c2rcc.msi {" ".join(cmd_params)}')
        return self._call(suffix='C2MSI', output_name=output_name)

    def c2rcc_olci(
        self,
        valid_pixel_expression: Optional[str] = '!quality_flags.invalid && (!quality_flags.land || quality_flags.fresh_inland_water)',
        salinity: float = 35.0,
        temperature: float = 15.0,
        ozone: float = 330.0,
        press: float = 1000.0,
        tsm_fac: float = 1.06,
        tsm_exp: float = 0.942,
        chl_exp: float = 1.04,
        chl_fac: float = 21.0,
        threshold_rtosa_oos: float = 0.01,
        threshold_ac_reflec_oos: float = 0.15,
        threshold_cloud_t_down865: float = 0.955,
        atmospheric_aux_data_path: Optional[str] = None,
        alternative_nn_path: Optional[str] = None,
        output_as_rrs: bool = False,
        derive_rw_from_path_and_transmittance: bool = False,
        use_ecmwf_aux_data: bool = True,
        dem_name: Optional[str] = None,
        output_rtoa: bool = True,
        output_rtosa_gc: bool = False,
        output_rtosa_gc_aann: bool = False,
        output_rpath: bool = False,
        output_tdown: bool = False,
        output_tup: bool = False,
        output_ac_reflectance: bool = True,
        output_rhown: bool = True,
        output_oos: bool = False,
        output_kd: bool = True,
        output_uncertainties: bool = True,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """C2RCC OLCI atmospheric correction."""
        self._reset_command()
        cmd_params = [
            f'-Psalinity={salinity}',
            f'-Ptemperature={temperature}',
            f'-Pozone={ozone}',
            f'-Ppress={press}',
            f'-PTSMfac={tsm_fac}',
            f'-PTSMexp={tsm_exp}',
            f'-PCHLexp={chl_exp}',
            f'-PCHLfac={chl_fac}',
            f'-PthresholdRtosaOOS={threshold_rtosa_oos}',
            f'-PthresholdAcReflecOos={threshold_ac_reflec_oos}',
            f'-PthresholdCloudTDown865={threshold_cloud_t_down865}',
            f'-PoutputAsRrs={str(output_as_rrs).lower()}',
            f'-PderiveRwFromPathAndTransmittance={str(derive_rw_from_path_and_transmittance).lower()}',
            f'-PuseEcmwfAuxData={str(use_ecmwf_aux_data).lower()}',
            f'-PoutputRtoa={str(output_rtoa).lower()}',
            f'-PoutputRtosaGc={str(output_rtosa_gc).lower()}',
            f'-PoutputRtosaGcAann={str(output_rtosa_gc_aann).lower()}',
            f'-PoutputRpath={str(output_rpath).lower()}',
            f'-PoutputTdown={str(output_tdown).lower()}',
            f'-PoutputTup={str(output_tup).lower()}',
            f'-PoutputAcReflectance={str(output_ac_reflectance).lower()}',
            f'-PoutputRhown={str(output_rhown).lower()}',
            f'-PoutputOos={str(output_oos).lower()}',
            f'-PoutputKd={str(output_kd).lower()}',
            f'-PoutputUncertainties={str(output_uncertainties).lower()}'
        ]
        if valid_pixel_expression:
            cmd_params.append(f'-PvalidPixelExpression="{valid_pixel_expression}"')
        if atmospheric_aux_data_path:
            cmd_params.append(f'-PatmosphericAuxDataPath="{atmospheric_aux_data_path}"')
        if alternative_nn_path:
            cmd_params.append(f'-PalternativeNNPath="{alternative_nn_path}"')
        if dem_name:
            cmd_params.append(f'-PdemName="{dem_name}"')
        self.current_cmd.append(f'c2rcc.olci {" ".join(cmd_params)}')
        return self._call(suffix='C2OLCI', output_name=output_name)

    def c2rcc_s2_msi(
        self,
        valid_pixel_expression: Optional[str] = 'B8 > 0 && B8 < 0.1',
        salinity: float = 35.0,
        temperature: float = 15.0,
        ozone: float = 330.0,
        press: float = 1000.0,
        elevation: float = 0.0,
        tsm_fac: float = 1.06,
        tsm_exp: float = 0.942,
        chl_exp: float = 1.04,
        chl_fac: float = 21.0,
        threshold_rtosa_oos: float = 0.05,
        threshold_ac_reflec_oos: float = 0.1,
        threshold_cloud_t_down865: float = 0.955,
        atmospheric_aux_data_path: Optional[str] = None,
        alternative_nn_path: Optional[str] = None,
        net_set: str = 'C2RCC-Nets',
        output_as_rrs: bool = False,
        derive_rw_from_path_and_transmittance: bool = False,
        use_ecmwf_aux_data: bool = False,
        dem_name: str = 'Copernicus 90m Global DEM',
        output_rtoa: bool = True,
        output_rtosa_gc: bool = False,
        output_rtosa_gc_aann: bool = False,
        output_rpath: bool = False,
        output_tdown: bool = False,
        output_tup: bool = False,
        output_ac_reflectance: bool = True,
        output_rhown: bool = True,
        output_oos: bool = False,
        output_kd: bool = True,
        output_uncertainties: bool = True,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """C2RCC S2 MSI correction."""
        self._reset_command()
        cmd_params = [
            f'-Psalinity={salinity}',
            f'-Ptemperature={temperature}',
            f'-Pozone={ozone}',
            f'-Ppress={press}',
            f'-Pelevation={elevation}',
            f'-PTSMfac={tsm_fac}',
            f'-PTSMexp={tsm_exp}',
            f'-PCHLexp={chl_exp}',
            f'-PCHLfac={chl_fac}',
            f'-PthresholdRtosaOOS={threshold_rtosa_oos}',
            f'-PthresholdAcReflecOos={threshold_ac_reflec_oos}',
            f'-PthresholdCloudTDown865={threshold_cloud_t_down865}',
            f'-PnetSet="{net_set}"',
            f'-PoutputAsRrs={str(output_as_rrs).lower()}',
            f'-PderiveRwFromPathAndTransmittance={str(derive_rw_from_path_and_transmittance).lower()}',
            f'-PuseEcmwfAuxData={str(use_ecmwf_aux_data).lower()}',
            f'-PdemName="{dem_name}"',
            f'-PoutputRtoa={str(output_rtoa).lower()}',
            f'-PoutputRtosaGc={str(output_rtosa_gc).lower()}',
            f'-PoutputRtosaGcAann={str(output_rtosa_gc_aann).lower()}',
            f'-PoutputRpath={str(output_rpath).lower()}',
            f'-PoutputTdown={str(output_tdown).lower()}',
            f'-PoutputTup={str(output_tup).lower()}',
            f'-PoutputAcReflectance={str(output_ac_reflectance).lower()}',
            f'-PoutputRhown={str(output_rhown).lower()}',
            f'-PoutputOos={str(output_oos).lower()}',
            f'-PoutputKd={str(output_kd).lower()}',
            f'-PoutputUncertainties={str(output_uncertainties).lower()}'
        ]
        if valid_pixel_expression:
            cmd_params.append(f'-PvalidPixelExpression="{valid_pixel_expression}"')
        if atmospheric_aux_data_path:
            cmd_params.append(f'-PatmosphericAuxDataPath="{atmospheric_aux_data_path}"')
        if alternative_nn_path:
            cmd_params.append(f'-PalternativeNNPath="{alternative_nn_path}"')
        self.current_cmd.append(f'c2rcc.msi {" ".join(cmd_params)}')
        return self._call(suffix='C2S2MSI', output_name=output_name)

    def biophysical(
        self,
        sensor: str = 'S2A',
        resolution: str = '60',
        compute_lai: bool = True,
        compute_fapar: bool = True,
        compute_fcover: bool = True,
        compute_cab: bool = True,
        compute_cw: bool = True,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Biophysical processor."""
        self._reset_command()
        cmd_params = [
            f'-Psensor="{sensor}"',
            f'-Presolution="{resolution}"',
            f'-PcomputeLAI={str(compute_lai).lower()}',
            f'-PcomputeFapar={str(compute_fapar).lower()}',
            f'-PcomputeFcover={str(compute_fcover).lower()}',
            f'-PcomputeCab={str(compute_cab).lower()}',
            f'-PcomputeCw={str(compute_cw).lower()}'
        ]
        self.current_cmd.append(f'BiophysicalOp {" ".join(cmd_params)}')
        return self._call(suffix='BIO', output_name=output_name)

    def biophysical_10m(
        self,
        sensor: str = 'S2A_10m',
        compute_lai: bool = True,
        compute_fapar: bool = True,
        compute_fcover: bool = True,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Biophysical processor 10m."""
        self._reset_command()
        cmd_params = [
            f'-Psensor="{sensor}"',
            f'-PcomputeLAI={str(compute_lai).lower()}',
            f'-PcomputeFapar={str(compute_fapar).lower()}',
            f'-PcomputeFcover={str(compute_fcover).lower()}'
        ]
        self.current_cmd.append(f'Biophysical10mOp {" ".join(cmd_params)}')
        return self._call(suffix='BIO10M', output_name=output_name)

    def biophysical_landsat8(
        self,
        compute_lai: bool = True,
        compute_fapar: bool = True,
        compute_fcover: bool = True,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Biophysical Landsat8 processor."""
        self._reset_command()
        cmd_params = [
            f'-PcomputeLAI={str(compute_lai).lower()}',
            f'-PcomputeFapar={str(compute_fapar).lower()}',
            f'-PcomputeFcover={str(compute_fcover).lower()}'
        ]
        self.current_cmd.append(f'BiophysicalLandsat8Op {" ".join(cmd_params)}')
        return self._call(suffix='BIOL8', output_name=output_name)

    def compactpol_radar_vegetation_index(
        self,
        window_size_str: str = '3',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compact-pol radar vegetation index."""
        self._reset_command()
        cmd_params = [f'-PwindowSizeStr="{window_size_str}"']
        self.current_cmd.append(f'Compactpol-Radar-Vegetation-Index {" ".join(cmd_params)}')
        return self._call(suffix='CPRVI', output_name=output_name)

    def generalized_radar_vegetation_indices(
        self,
        window_size: int = 5,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Generalized radar vegetation indices."""
        self._reset_command()
        cmd_params = [f'-PwindowSize={window_size}']
        self.current_cmd.append(f'Generalized-Radar-Vegetation-Index {" ".join(cmd_params)}')
        return self._call(suffix='GRVI', output_name=output_name)

    def radar_vegetation_index(
        self,
        window_size: int = 5,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Radar vegetation index."""
        self._reset_command()
        cmd_params = [f'-PwindowSize={window_size}']
        self.current_cmd.append(f'Radar-Vegetation-Index {" ".join(cmd_params)}')
        return self._call(suffix='RVINDEX', output_name=output_name)

    def change_detection(
        self,
        source_bands: Optional[List[str]] = None,
        mask_upper_threshold: float = 2.0,
        mask_lower_threshold: float = -2.0,
        include_source_bands: bool = False,
        output_difference: bool = False,
        output_ratio: bool = False,
        output_log_ratio: bool = True,
        output_normalized_ratio: bool = False,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Log ratio change detection."""
        self._reset_command()
        cmd_params = [
            f'-PmaskUpperThreshold={mask_upper_threshold}',
            f'-PmaskLowerThreshold={mask_lower_threshold}',
            f'-PincludeSourceBands={str(include_source_bands).lower()}',
            f'-PoutputDifference={str(output_difference).lower()}',
            f'-PoutputRatio={str(output_ratio).lower()}',
            f'-PoutputLogRatio={str(output_log_ratio).lower()}',
            f'-PoutputNormalizedRatio={str(output_normalized_ratio).lower()}'
        ]
        if source_bands:
            cmd_params.append(f'-PsourceBands={",".join(source_bands)}')
        self.current_cmd.append(f'Change-Detection {" ".join(cmd_params)}')
        return self._call(suffix='CHGDET', output_name=output_name)

    def change_vector_analysis(
        self,
        source_band1: Optional[str] = None,
        source_band2: Optional[str] = None,
        magnitude_threshold: str = '0',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Change vector analysis."""
        self._reset_command()
        cmd_params = [f'-PmagnitudeThreshold="{magnitude_threshold}"']
        if source_band1:
            cmd_params.append(f'-PsourceBand1="{source_band1}"')
        if source_band2:
            cmd_params.append(f'-PsourceBand2="{source_band2}"')
        self.current_cmd.append(f'ChangeVectorAnalysisOp {" ".join(cmd_params)}')
        return self._call(suffix='CVA', output_name=output_name)

    def add_land_cover(
        self,
        land_cover_names: Optional[List[str]] = None,
        external_files: Optional[List[str]] = None,
        resampling_method: str = 'NEAREST_NEIGHBOUR',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Add land cover band."""
        self._reset_command()
        cmd_params = [f'-PresamplingMethod={resampling_method}']
        if land_cover_names:
            cmd_params.append(f'-PlandCoverNames={",".join(land_cover_names)}')
        if external_files:
            cmd_params.append(f'-PexternalFiles={",".join(external_files)}')
        self.current_cmd.append(f'AddLandCover {" ".join(cmd_params)}')
        return self._call(suffix='LC', output_name=output_name)

    def cloud_prob(
        self,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Cloud probability detection."""
        self._reset_command()
        self.current_cmd.append('CloudProb')
        return self._call(suffix='CLDPRB', output_name=output_name)

    def double_difference_interferogram(
        self,
        coh_win_size: str = '5',
        output_coherence: bool = False,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compute double difference interferogram."""
        self._reset_command()
        cmd_params = [
            f'-PcohWinSize={coh_win_size}',
            f'-PoutputCoherence={str(output_coherence).lower()}'
        ]
        self.current_cmd.append(f'Double-Difference-Interferogram {" ".join(cmd_params)}')
        return self._call(suffix='DDIFF', output_name=output_name)

    def add_elevation(
        self,
        dem_name: str = 'SRTM 3Sec',
        dem_resampling_method: str = 'BICUBIC_INTERPOLATION',
        elevation_band_name: str = 'elevation',
        external_dem_file: Optional[str] = None,
        external_dem_no_data_value: float = 0.0,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Creates DEM band."""
        self._reset_command()
        cmd_params = [
            f'-PdemName={dem_name}',
            f'-PdemResamplingMethod={dem_resampling_method}',
            f'-PelevationBandName={elevation_band_name}',
            f'-PexternalDEMNoDataValue={external_dem_no_data_value}'
        ]
        if external_dem_file:
            cmd_params.append(f'-PexternalDEMFile={external_dem_file}')
        self.current_cmd.append(f'AddElevation {" ".join(cmd_params)}')
        return self._call(suffix='ELEV', output_name=output_name)

    def fill_dem_hole(
        self,
        no_data_value: float = 0.0,
        source_bands: Optional[List[str]] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Fill DEM holes."""
        self._reset_command()
        cmd_params = [f'-PNoDataValue={no_data_value}']
        if source_bands:
            cmd_params.append(f'-PsourceBands={",".join(source_bands)}')
        self.current_cmd.append(f'Fill-DEM-Hole {" ".join(cmd_params)}')
        return self._call(suffix='FILLDEM', output_name=output_name)

    def compute_slope_aspect(
        self,
        dem_band_name: str = 'elevation',
        dem_name: str = 'SRTM 1Sec HGT',
        dem_resampling_method: str = 'BILINEAR_INTERPOLATION',
        external_dem_apply_egm: bool = False,
        external_dem_file: Optional[str] = None,
        external_dem_no_data_value: float = 0.0,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compute slope and aspect."""
        self._reset_command()
        cmd_params = [
            f'-PdemBandName={dem_band_name}',
            f'-PdemName={dem_name}',
            f'-PdemResamplingMethod={dem_resampling_method}',
            f'-PexternalDEMApplyEGM={str(external_dem_apply_egm).lower()}',
            f'-PexternalDEMNoDataValue={external_dem_no_data_value}'
        ]
        if external_dem_file:
            cmd_params.append(f'-PexternalDEMFile={external_dem_file}')
        self.current_cmd.append(f'Compute-Slope-Aspect {" ".join(cmd_params)}')
        return self._call(suffix='SLOPE', output_name=output_name)

    def cp_decomposition(
        self,
        decomposition: str = 'M-Chi Decomposition',
        window_size_x_str: str = '5',
        window_size_y_str: str = '5',
        compute_alpha_by_t3: bool = True,
        output_rvog: bool = True,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compact polarimetric decomposition."""
        self._reset_command()
        cmd_params = [
            f'-Pdecomposition={decomposition}',
            f'-PwindowSizeXStr={window_size_x_str}',
            f'-PwindowSizeYStr={window_size_y_str}',
            f'-PcomputeAlphaByT3={str(compute_alpha_by_t3).lower()}',
            f'-PoutputRVOG={str(output_rvog).lower()}'
        ]
        self.current_cmd.append(f'CP-Decomposition {" ".join(cmd_params)}')
        return self._call(suffix='CPDECOMP', output_name=output_name)

    def cp_simulation(
        self,
        compact_mode: str = 'Right Circular Hybrid Mode',
        output_format: str = 'Covariance Matrix C2',
        noise_power: float = -25.0,
        simulate_noise_floor: bool = False,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compact pol simulation."""
        self._reset_command()
        cmd_params = [
            f'-PcompactMode={compact_mode}',
            f'-PoutputFormat={output_format}',
            f'-PnoisePower={noise_power}',
            f'-PsimulateNoiseFloor={str(simulate_noise_floor).lower()}'
        ]
        self.current_cmd.append(f'CP-Simulation {" ".join(cmd_params)}')
        return self._call(suffix='CPSIM', output_name=output_name)

    def cp_stokes_parameters(
        self,
        window_size_x_str: str = '5',
        window_size_y_str: str = '5',
        output_stokes_vector: bool = False,
        output_degree_of_polarization: bool = True,
        output_degree_of_depolarization: bool = True,
        output_degree_of_circularity: bool = True,
        output_degree_of_ellipticity: bool = True,
        output_cpr: bool = True,
        output_lpr: bool = True,
        output_relative_phase: bool = True,
        output_alphas: bool = True,
        output_conformity: bool = True,
        output_phase_phi: bool = True,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Compact pol Stokes parameters."""
        self._reset_command()
        cmd_params = [
            f'-PwindowSizeXStr={window_size_x_str}',
            f'-PwindowSizeYStr={window_size_y_str}',
            f'-PoutputStokesVector={str(output_stokes_vector).lower()}',
            f'-PoutputDegreeOfPolarization={str(output_degree_of_polarization).lower()}',
            f'-PoutputDegreeOfDepolarization={str(output_degree_of_depolarization).lower()}',
            f'-PoutputDegreeOfCircularity={str(output_degree_of_circularity).lower()}',
            f'-PoutputDegreeOfEllipticity={str(output_degree_of_ellipticity).lower()}',
            f'-PoutputCPR={str(output_cpr).lower()}',
            f'-PoutputLPR={str(output_lpr).lower()}',
            f'-PoutputRelativePhase={str(output_relative_phase).lower()}',
            f'-PoutputAlphas={str(output_alphas).lower()}',
            f'-PoutputConformity={str(output_conformity).lower()}',
            f'-PoutputPhasePhi={str(output_phase_phi).lower()}'
        ]
        self.current_cmd.append(f'CP-Stokes-Parameters {" ".join(cmd_params)}')
        return self._call(suffix='CPSTOKES', output_name=output_name)

    def coregistration(
        self,
        master_source_band: Optional[str] = None,
        slave_source_band: Optional[str] = None,
        levels: int = 6,
        rank: int = 4,
        iterations: int = 2,
        radius: str = '32, 28, 24, 20, 16, 12, 8',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Coregister two rasters."""
        self._reset_command()
        cmd_params = [
            f'-Plevels={levels}',
            f'-Prank={rank}',
            f'-Piterations={iterations}',
            f'-Pradius={radius}'
        ]
        if master_source_band:
            cmd_params.append(f'-PmasterSourceBand={master_source_band}')
        if slave_source_band:
            cmd_params.append(f'-PslaveSourceBand={slave_source_band}')
        self.current_cmd.append(f'CoregistrationOp {" ".join(cmd_params)}')
        return self._call(suffix='COREG', output_name=output_name)

    def ellipsoid_correction_gg(
        self,
        img_resampling_method: str = 'BILINEAR_INTERPOLATION',
        map_projection: str = 'WGS84(DD)',
        source_bands: Optional[List[str]] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """GG ellipsoid correction."""
        self._reset_command()
        cmd_params = [
            f'-PimgResamplingMethod={img_resampling_method}',
            f'-PmapProjection={map_projection}'
        ]
        if source_bands:
            cmd_params.append(f'-PsourceBands={",".join(source_bands)}')
        self.current_cmd.append(f'Ellipsoid-Correction-GG {" ".join(cmd_params)}')
        return self._call(suffix='ELLCORR', output_name=output_name)

    def cross_resampling(
        self,
        warp_polynomial_order: int = 2,
        interpolation_method: str = 'Cubic convolution (6 points)',
        target_geometry: str = 'ERS',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Cross resampling operation."""
        self._reset_command()
        cmd_params = [
            f'-PwarpPolynomialOrder={warp_polynomial_order}',
            f'-PinterpolationMethod={interpolation_method}',
            f'-PtargetGeometry={target_geometry}'
        ]
        self.current_cmd.append(f'CrossResampling {" ".join(cmd_params)}')
        return self._call(suffix='CROSSRES', output_name=output_name)

    def bands_difference(
        self,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Bands difference operation."""
        self._reset_command()
        self.current_cmd.append('BandsDifferenceOp')
        return self._call(suffix='BANDDIFF', output_name=output_name)

    def bands_extractor(
        self,
        source_band_names: Optional[List[str]] = None,
        source_mask_names: Optional[List[str]] = None,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Extract specific bands."""
        self._reset_command()
        cmd_params = []
        if source_band_names:
            cmd_params.append(f'-PsourceBandNames={",".join(source_band_names)}')
        if source_mask_names:
            cmd_params.append(f'-PsourceMaskNames={",".join(source_mask_names)}')
        self.current_cmd.append(f'BandsExtractorOp {" ".join(cmd_params)}')
        return self._call(suffix='BANDEXT', output_name=output_name)

    def binning(
        self,
        num_rows: int = 2160,
        mask_expr: Optional[str] = None,
        source_product_paths: Optional[List[str]] = None,
        source_product_format: Optional[str] = None,
        region: Optional[str] = None,
        start_date_time: Optional[str] = None,
        period_duration: Optional[float] = None,
        time_filter_method: str = 'NONE',
        min_data_hour: Optional[float] = None,
        super_sampling: int = 1,
        max_distance_on_earth: int = -1,
        output_binned_data: bool = False,
        output_mapped_product: bool = True,
        output_type: str = 'Product',
        output_format: str = 'BEAM-DIMAP',
        output_file: Optional[str] = None,
        metadata_aggregator_name: str = 'NAME',
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Spatial temporal binning."""
        self._reset_command()
        cmd_params = [
            f'-PnumRows={num_rows}',
            f'-PsuperSampling={super_sampling}',
            f'-PmaxDistanceOnEarth={max_distance_on_earth}',
            f'-PoutputBinnedData={str(output_binned_data).lower()}',
            f'-PoutputMappedProduct={str(output_mapped_product).lower()}',
            f'-PoutputType={output_type}',
            f'-PoutputFormat={output_format}',
            f'-PmetadataAggregatorName={metadata_aggregator_name}',
            f'-PtimeFilterMethod={time_filter_method}'
        ]
        if mask_expr:
            cmd_params.append(f'-PmaskExpr={mask_expr}')
        if source_product_paths:
            cmd_params.append(f'-PsourceProductPaths={",".join(source_product_paths)}')
        if source_product_format:
            cmd_params.append(f'-PsourceProductFormat={source_product_format}')
        if region:
            cmd_params.append(f'-Pregion={region}')
        if start_date_time:
            cmd_params.append(f'-PstartDateTime={start_date_time}')
        if period_duration:
            cmd_params.append(f'-PperiodDuration={period_duration}')
        if min_data_hour is not None:
            cmd_params.append(f'-PminDataHour={min_data_hour}')
        if output_file:
            cmd_params.append(f'-PoutputFile={output_file}')
        self.current_cmd.append(f'Binning {" ".join(cmd_params)}')
        return self._call(suffix='BIN', output_name=output_name)

    def fu_classification(
        self,
        copy_all_source_bands: bool = False,
        input_is_irradiance_reflectance: bool = False,
        valid_expression: Optional[str] = None,
        reflectance_name_pattern: Optional[str] = None,
        instrument: str = 'AUTO_DETECT',
        include_dominant_lambda: bool = False,
        include_intermediate_results: bool = True,
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Forel-Ule classification."""
        self._reset_command()
        cmd_params = [
            f'-PcopyAllSourceBands={str(copy_all_source_bands).lower()}',
            f'-PinputIsIrradianceReflectance={str(input_is_irradiance_reflectance).lower()}',
            f'-Pinstrument={instrument}',
            f'-PincludeDominantLambda={str(include_dominant_lambda).lower()}',
            f'-PincludeIntermediateResults={str(include_intermediate_results).lower()}'
        ]
        if valid_expression:
            cmd_params.append(f'-PvalidExpression={valid_expression}')
        if reflectance_name_pattern:
            cmd_params.append(f'-PreflectanceNamePattern={reflectance_name_pattern}')
        self.current_cmd.append(f'FuClassification {" ".join(cmd_params)}')
        return self._call(suffix='FUCLASS', output_name=output_name)

    def flh_mci(
        self,
        preset: str = 'NONE',
        lower_baseline_band_name: Optional[str] = None,
        upper_baseline_band_name: Optional[str] = None,
        signal_band_name: Optional[str] = None,
        line_height_band_name: Optional[str] = None,
        slope: bool = True,
        slope_band_name: Optional[str] = None,
        mask_expression: Optional[str] = None,
        cloud_correction_factor: float = 1.005,
        invalid_flh_mci_value: float = float('nan'),
        output_name: Optional[str] = None
    ) -> Optional[str]:
        """Fluorescence line height."""
        self._reset_command()
        cmd_params = [
            f'-Ppreset={preset}',
            f'-Pslope={str(slope).lower()}',
            f'-PcloudCorrectionFactor={cloud_correction_factor}',
            f'-PinvalidFlhMciValue={invalid_flh_mci_value}'
        ]
        if lower_baseline_band_name:
            cmd_params.append(f'-PlowerBaselineBandName={lower_baseline_band_name}')
        if upper_baseline_band_name:
            cmd_params.append(f'-PupperBaselineBandName={upper_baseline_band_name}')
        if signal_band_name:
            cmd_params.append(f'-PsignalBandName={signal_band_name}')
        if line_height_band_name:
            cmd_params.append(f'-PlineHeightBandName={line_height_band_name}')
        if slope_band_name:
            cmd_params.append(f'-PslopeBandName={slope_band_name}')
        if mask_expression:
            cmd_params.append(f'-PmaskExpression={mask_expression}')
        self.current_cmd.append(f'FlhMci {" ".join(cmd_params)}')
        return self._call(suffix='FLH', output_name=output_name)

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
    SetNoDataValue = set_no_data_value
    SupervisedWishartClassification = supervised_wishart_classification
    StatisticsOp = statistics_op
    SubGraph = subgraph
    StampsExport = stamps_export
    StackSplit = stack_split
    AatsrSST = aatsr_sst
    AatsrUngrid = aatsr_ungrid
    WindFieldEstimation = wind_field_estimation
    Wdvi = wdvi
    SaviOp = savi
    Warp = warp
    SRGR = srgr
    SmacOp = smac_op
    SMDielectricModeling = sm_dielectric_modeling
    SliceAssembly = slice_assembly
    UpdateGeoReference = update_geo_reference
    AddElevation = add_elevation
    ThreePassDInSAR = three_pass_dinsar
    TemporalPercentile = temporal_percentile
    SpectralAngleMapperOp = spectral_angle_mapper
    SpeckleFilter = speckle_filter
    SpeckleDivergence = speckle_divergence
    SnaphuImport = snaphu_import
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
    SarSimTerrainCorrection = sar_sim_terrain_correction
    SarSimulation = sar_simulation
    SarMosaic = sar_mosaic
    S2Resampling = s2_resampling
    S2repOp = s2rep
    Demodulate = demodulate
    Write = write
    TileWriter = tile_writer
    StackAveraging = stack_averaging
    BandMaths = band_maths
    BandSelect = band_select
    BandMerge = band_merge
    Coherence = coherence
    Interferogram = interferogram
    GoldsteinPhaseFiltering = goldstein_phase_filtering
    BackGeocoding = back_geocoding
    CreateStack = create_stack
    Resample = resample
    LinearToFromdB = linear_to_from_db
    RemoveGRDBorderNoise = remove_grd_border_noise
    Collocate = collocate
    PolarimetricDecomposition = polarimetric_decomposition
    PolarimetricParameters = polarimetric_parameters
    OffsetTracking = offset_tracking
    Ndvi = ndvi
    Reproject = reproject
    PolarimetricMatrices = polarimetric_matrices
    PolarimetricSpeckleFilter = polarimetric_speckle_filter
    PolarimetricClassification = polarimetric_classification
    ReadProduct = read_product
    MergeProducts = merge_products
    Mosaic = mosaic
    Flip = flip
    ImageFilter = image_filter
    ConvertDatatype = convert_datatype
    LandSeaMask = land_sea_mask
    EnhancedSpectralDiversity = enhanced_spectral_diversity
    PhaseToDisplacement = phase_to_displacement
    PhaseToElevation = phase_to_elevation
    PhaseToHeight = phase_to_height
    SnaphuExport = snaphu_export
    CrossCorrelation = cross_correlation
    DEMAssistedCoregistration = dem_assisted_coregistration
    EAPPhaseCorrection = eap_phase_correction
    EllipsoidCorrectionRD = ellipsoid_correction_rd
    RangeFilter = range_filter
    AzimuthFilter = azimuth_filter
    BandPassFilter = band_pass_filter
    Oversample = oversample
    GLCM = glcm
    PCA = pca
    KMeansClusterAnalysis = k_means_cluster_analysis
    EMClusterAnalysis = em_cluster_analysis
    RandomForestClassifier = random_forest_classifier
    DecisionTree = decision_tree
    Arvi = arvi
    Dvi = dvi
    Gemi = gemi
    Gndvi = gndvi
    Ipvi = ipvi
    Mcari = mcari
    Mndwi = mndwi
    Msavi2 = msavi2
    Msavi = msavi
    Mtci = mtci
    Ndwi = ndwi
    Rvi = rvi
    Bi2 = bi2
    Bi = bi
    Ci = ci
    Ireci = ireci
    Ndi45 = ndi45
    Ndpi = ndpi
    Ndti = ndti
    Ndwi2 = ndwi2
    Pssra = pssra
    Pvi = pvi
    Reip = reip
    Ri = ri
    C2rccMsi = c2rcc_msi
    C2rccOlci = c2rcc_olci
    C2rccS2Msi = c2rcc_s2_msi
    Biophysical = biophysical
    Biophysical10m = biophysical_10m
    BiophysicalLandsat8 = biophysical_landsat8
    CompactpolRadarVegetationIndex = compactpol_radar_vegetation_index
    GeneralizedRadarVegetationIndices = generalized_radar_vegetation_indices
    RadarVegetationIndex = radar_vegetation_index
    ChangeDetection = change_detection
    ChangeVectorAnalysis = change_vector_analysis
    AddLandCover = add_land_cover
    CloudProb = cloud_prob
    DoubleDifferenceInterferogram = double_difference_interferogram
    AddElevation = add_elevation
    FillDemHole = fill_dem_hole
    ComputeSlopeAspect = compute_slope_aspect
    CpDecomposition = cp_decomposition
    CpSimulation = cp_simulation
    CpStokesParameters = cp_stokes_parameters
    Coregistration = coregistration
    EllipsoidCorrectionGg = ellipsoid_correction_gg
    CrossResampling = cross_resampling
    BandsDifference = bands_difference
    BandsExtractor = bands_extractor
    Binning = binning
    FuClassification = fu_classification
    FlhMci = flh_mci



