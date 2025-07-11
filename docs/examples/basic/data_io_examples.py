#!/usr/bin/env python3
"""
SARPYX Data I/O Examples
========================

This module demonstrates various data input/output operations in SARPYX,
including reading SAR data, handling different formats, and basic data
manipulation.

Author: SARPYX Development Team
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# SARPYX imports
from sarpyx.sla import SLAProcessor
from sarpyx.utils.io import SARDataReader, SARDataWriter
from sarpyx.utils.viz import SARVisualizer
from sarpyx.science.indices import VegetationIndices

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIOExamples:
    """
    Collection of examples demonstrating SARPYX data I/O capabilities.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the examples with optional data path.
        
        Parameters
        ----------
        data_path : str, optional
            Path to SAR data directory
        """
        self.data_path = Path(data_path) if data_path else Path("../../../data")
        self.output_path = Path("./outputs")
        self.output_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.reader = SARDataReader()
        self.writer = SARDataWriter()
        self.visualizer = SARVisualizer()
        
    def example_1_basic_reading(self):
        """
        Example 1: Basic SAR data reading
        
        Demonstrates how to read different SAR data formats supported by SARPYX.
        """
        print("\n" + "="*60)
        print("Example 1: Basic SAR Data Reading")
        print("="*60)
        
        try:
            # Read Sentinel-1 SAFE format
            safe_path = self.data_path / "S1A_S3_SLC__1SSH_20240621T052251_20240621T052319_054417_069F07_8466.SAFE"
            
            if safe_path.exists():
                print(f"Reading Sentinel-1 data from: {safe_path}")
                
                # Read metadata
                metadata = self.reader.read_metadata(safe_path)
                print(f"Product type: {metadata.get('product_type', 'Unknown')}")
                print(f"Acquisition mode: {metadata.get('acquisition_mode', 'Unknown')}")
                print(f"Polarizations: {metadata.get('polarizations', [])}")
                print(f"Image dimensions: {metadata.get('image_size', (0, 0))}")
                
                # Read SLC data
                slc_data = self.reader.read_slc(safe_path, polarization='HH')
                print(f"SLC data shape: {slc_data.shape}")
                print(f"SLC data type: {slc_data.dtype}")
                print(f"SLC amplitude range: {np.abs(slc_data).min():.2f} - {np.abs(slc_data).max():.2f}")
                
                return slc_data, metadata
            else:
                print(f"Warning: Sample data not found at {safe_path}")
                # Generate synthetic data for demonstration
                return self._generate_synthetic_slc(), self._generate_synthetic_metadata()
                
        except Exception as e:
            logger.error(f"Error reading data: {e}")
            return None, None
    
    def example_2_format_conversion(self):
        """
        Example 2: Format conversion between different SAR data formats
        
        Shows how to convert between SAFE, GeoTIFF, and HDF5 formats.
        """
        print("\n" + "="*60)
        print("Example 2: Format Conversion")
        print("="*60)
        
        # Get data from previous example
        slc_data, metadata = self.example_1_basic_reading()
        
        if slc_data is None:
            return
        
        # Convert amplitude to dB
        amplitude_db = 20 * np.log10(np.abs(slc_data) + 1e-10)
        
        try:
            # Save as GeoTIFF
            geotiff_path = self.output_path / "amplitude_db.tif"
            self.writer.write_geotiff(amplitude_db, geotiff_path, metadata)
            print(f"Saved amplitude data as GeoTIFF: {geotiff_path}")
            
            # Save as HDF5
            hdf5_path = self.output_path / "slc_data.h5"
            self.writer.write_hdf5({
                'slc_complex': slc_data,
                'amplitude_db': amplitude_db,
                'metadata': metadata
            }, hdf5_path)
            print(f"Saved complex data as HDF5: {hdf5_path}")
            
            # Save as NumPy array
            npy_path = self.output_path / "slc_array.npy"
            np.save(npy_path, slc_data)
            print(f"Saved as NumPy array: {npy_path}")
            
            # Demonstrate reading back the data
            print("\nReading back converted data:")
            loaded_amplitude = self.reader.read_geotiff(geotiff_path)
            print(f"Loaded GeoTIFF shape: {loaded_amplitude.shape}")
            
            loaded_data = self.reader.read_hdf5(hdf5_path)
            print(f"Loaded HDF5 keys: {list(loaded_data.keys())}")
            
            loaded_array = np.load(npy_path)
            print(f"Loaded NumPy array shape: {loaded_array.shape}")
            
        except Exception as e:
            logger.error(f"Error in format conversion: {e}")
    
    def example_3_data_subsetting(self):
        """
        Example 3: Data subsetting and ROI extraction
        
        Demonstrates how to extract regions of interest and subset large datasets.
        """
        print("\n" + "="*60)
        print("Example 3: Data Subsetting and ROI Extraction")
        print("="*60)
        
        # Get data
        slc_data, metadata = self.example_1_basic_reading()
        
        if slc_data is None:
            return
        
        try:
            # Define ROI coordinates (pixel coordinates)
            roi_start_row, roi_end_row = 100, 500
            roi_start_col, roi_end_col = 200, 600
            
            # Extract ROI
            roi_data = slc_data[roi_start_row:roi_end_row, roi_start_col:roi_end_col]
            print(f"Original data shape: {slc_data.shape}")
            print(f"ROI data shape: {roi_data.shape}")
            
            # Create geographic ROI based on coordinates
            if 'geotransform' in metadata:
                geo_bounds = self._pixel_to_geo_bounds(
                    (roi_start_row, roi_start_col, roi_end_row, roi_end_col),
                    metadata['geotransform']
                )
                print(f"Geographic bounds: {geo_bounds}")
            
            # Multiple ROI extraction
            roi_list = [
                (50, 150, 200, 300),    # ROI 1
                (300, 450, 400, 550),   # ROI 2
                (600, 750, 100, 250)    # ROI 3
            ]
            
            extracted_rois = []
            for i, (r1, r2, c1, c2) in enumerate(roi_list):
                if r2 < slc_data.shape[0] and c2 < slc_data.shape[1]:
                    roi = slc_data[r1:r2, c1:c2]
                    extracted_rois.append(roi)
                    print(f"ROI {i+1}: {roi.shape}")
            
            # Save ROI data
            roi_path = self.output_path / "roi_data.npy"
            np.save(roi_path, roi_data)
            print(f"ROI saved to: {roi_path}")
            
            return roi_data, extracted_rois
            
        except Exception as e:
            logger.error(f"Error in data subsetting: {e}")
            return None, None
    
    def example_4_batch_processing(self):
        """
        Example 4: Batch processing multiple files
        
        Shows how to process multiple SAR files in batch mode.
        """
        print("\n" + "="*60)
        print("Example 4: Batch Processing")
        print("="*60)
        
        # Find all SAR files in data directory
        sar_files = list(self.data_path.glob("**/*.SAFE"))
        
        if not sar_files:
            print("No SAFE files found, creating synthetic dataset for demonstration")
            # Create synthetic files for demonstration
            sar_files = [f"synthetic_file_{i}.SAFE" for i in range(3)]
        
        print(f"Found {len(sar_files)} SAR files for processing")
        
        batch_results = []
        
        for i, file_path in enumerate(sar_files):
            try:
                print(f"\nProcessing file {i+1}/{len(sar_files)}: {Path(file_path).name}")
                
                if isinstance(file_path, str) and "synthetic" in file_path:
                    # Generate synthetic data for demonstration
                    data = self._generate_synthetic_slc((200, 200))
                    metadata = self._generate_synthetic_metadata()
                else:
                    # Read real data
                    metadata = self.reader.read_metadata(file_path)
                    data = self.reader.read_slc(file_path, polarization='HH')
                
                # Basic processing: calculate amplitude statistics
                amplitude = np.abs(data)
                stats = {
                    'file': Path(file_path).name,
                    'mean_amplitude': float(np.mean(amplitude)),
                    'std_amplitude': float(np.std(amplitude)),
                    'max_amplitude': float(np.max(amplitude)),
                    'min_amplitude': float(np.min(amplitude)),
                    'shape': data.shape
                }
                
                batch_results.append(stats)
                print(f"  Mean amplitude: {stats['mean_amplitude']:.2f}")
                print(f"  Shape: {stats['shape']}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        # Save batch results
        import json
        results_path = self.output_path / "batch_results.json"
        with open(results_path, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        print(f"\nBatch processing complete. Results saved to: {results_path}")
        return batch_results
    
    def example_5_memory_efficient_processing(self):
        """
        Example 5: Memory-efficient processing for large datasets
        
        Demonstrates techniques for processing large SAR datasets without
        loading everything into memory.
        """
        print("\n" + "="*60)
        print("Example 5: Memory-Efficient Processing")
        print("="*60)
        
        try:
            # Simulate large dataset
            large_shape = (5000, 5000)
            chunk_size = (512, 512)
            
            print(f"Simulating processing of large dataset: {large_shape}")
            print(f"Using chunk size: {chunk_size}")
            
            # Memory-mapped approach for large files
            temp_file = self.output_path / "large_dataset.dat"
            
            # Create memory-mapped array
            large_array = np.memmap(
                temp_file, dtype=np.complex64, mode='w+', shape=large_shape
            )
            
            # Fill with synthetic data in chunks
            for i in range(0, large_shape[0], chunk_size[0]):
                for j in range(0, large_shape[1], chunk_size[1]):
                    end_i = min(i + chunk_size[0], large_shape[0])
                    end_j = min(j + chunk_size[1], large_shape[1])
                    
                    # Generate chunk
                    chunk_shape = (end_i - i, end_j - j)
                    chunk_data = self._generate_synthetic_slc(chunk_shape)
                    
                    # Write to memory-mapped array
                    large_array[i:end_i, j:end_j] = chunk_data
                    
                    # Process chunk (example: calculate statistics)
                    amplitude = np.abs(chunk_data)
                    mean_amp = np.mean(amplitude)
                    
                    if (i // chunk_size[0]) % 5 == 0 and (j // chunk_size[1]) % 5 == 0:
                        print(f"  Processed chunk ({i}:{end_i}, {j}:{end_j}), "
                              f"mean amplitude: {mean_amp:.2f}")
            
            # Demonstrate chunked statistics calculation
            print("\nCalculating statistics using chunked processing...")
            chunk_stats = []
            
            for i in range(0, large_shape[0], chunk_size[0]):
                for j in range(0, large_shape[1], chunk_size[1]):
                    end_i = min(i + chunk_size[0], large_shape[0])
                    end_j = min(j + chunk_size[1], large_shape[1])
                    
                    chunk = large_array[i:end_i, j:end_j]
                    amplitude = np.abs(chunk)
                    
                    chunk_stats.append({
                        'position': (i, j),
                        'mean': float(np.mean(amplitude)),
                        'std': float(np.std(amplitude))
                    })
            
            # Overall statistics from chunks
            overall_mean = np.mean([s['mean'] for s in chunk_stats])
            overall_std = np.mean([s['std'] for s in chunk_stats])
            
            print(f"Overall mean amplitude: {overall_mean:.2f}")
            print(f"Overall std amplitude: {overall_std:.2f}")
            print(f"Total chunks processed: {len(chunk_stats)}")
            
            # Clean up
            del large_array
            temp_file.unlink()
            
        except Exception as e:
            logger.error(f"Error in memory-efficient processing: {e}")
    
    def example_6_data_validation(self):
        """
        Example 6: Data validation and quality checks
        
        Shows how to validate SAR data and perform quality assessments.
        """
        print("\n" + "="*60)
        print("Example 6: Data Validation and Quality Checks")
        print("="*60)
        
        # Get data
        slc_data, metadata = self.example_1_basic_reading()
        
        if slc_data is None:
            return
        
        try:
            validation_results = {}
            
            # 1. Check for NaN/Inf values
            nan_count = np.sum(np.isnan(slc_data))
            inf_count = np.sum(np.isinf(slc_data))
            validation_results['nan_count'] = int(nan_count)
            validation_results['inf_count'] = int(inf_count)
            
            print(f"NaN values: {nan_count}")
            print(f"Inf values: {inf_count}")
            
            # 2. Dynamic range analysis
            amplitude = np.abs(slc_data)
            amplitude_db = 20 * np.log10(amplitude + 1e-10)
            
            validation_results['amplitude_stats'] = {
                'min_db': float(np.min(amplitude_db)),
                'max_db': float(np.max(amplitude_db)),
                'mean_db': float(np.mean(amplitude_db)),
                'std_db': float(np.std(amplitude_db))
            }
            
            print(f"Amplitude range: {validation_results['amplitude_stats']['min_db']:.1f} to "
                  f"{validation_results['amplitude_stats']['max_db']:.1f} dB")
            
            # 3. Phase analysis
            phase = np.angle(slc_data)
            validation_results['phase_stats'] = {
                'min_rad': float(np.min(phase)),
                'max_rad': float(np.max(phase)),
                'mean_rad': float(np.mean(phase)),
                'std_rad': float(np.std(phase))
            }
            
            # 4. Check for data gaps or missing lines
            line_means = np.mean(amplitude, axis=1)
            col_means = np.mean(amplitude, axis=0)
            
            # Detect lines/columns with unusually low values (potential gaps)
            threshold = np.mean(line_means) * 0.1
            problematic_lines = np.where(line_means < threshold)[0]
            problematic_cols = np.where(col_means < threshold)[0]
            
            validation_results['data_gaps'] = {
                'problematic_lines': problematic_lines.tolist(),
                'problematic_cols': problematic_cols.tolist()
            }
            
            print(f"Problematic lines: {len(problematic_lines)}")
            print(f"Problematic columns: {len(problematic_cols)}")
            
            # 5. SNR estimation
            # Simple SNR estimation using image statistics
            signal_power = np.mean(amplitude**2)
            noise_power = np.var(amplitude)  # Simplified noise estimation
            snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            validation_results['snr_db'] = float(snr_db)
            print(f"Estimated SNR: {snr_db:.1f} dB")
            
            # Save validation results
            import json
            validation_path = self.output_path / "validation_results.json"
            with open(validation_path, 'w') as f:
                json.dump(validation_results, f, indent=2)
            
            print(f"\nValidation results saved to: {validation_path}")
            
            # Generate validation report
            self._generate_validation_report(validation_results, slc_data)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in data validation: {e}")
            return None
    
    def _generate_synthetic_slc(self, shape=(1000, 1000)):
        """Generate synthetic SLC data for demonstration."""
        # Create realistic SLC with speckle and some structure
        real_part = np.random.randn(*shape) * 0.5
        imag_part = np.random.randn(*shape) * 0.5
        
        # Add some spatial structure
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        structure = np.exp(-((x - shape[1]//2)**2 + (y - shape[0]//2)**2) / (shape[0]//4)**2)
        
        slc_data = (real_part + 1j * imag_part) * (1 + structure)
        return slc_data.astype(np.complex64)
    
    def _generate_synthetic_metadata(self):
        """Generate synthetic metadata for demonstration."""
        return {
            'product_type': 'SLC',
            'acquisition_mode': 'S3',
            'polarizations': ['HH'],
            'image_size': (1000, 1000),
            'pixel_spacing': (5.0, 20.0),
            'geotransform': [0, 20, 0, 0, 0, -5],
            'center_frequency': 5.405e9,
            'bandwidth': 42.86e6
        }
    
    def _pixel_to_geo_bounds(self, pixel_bounds, geotransform):
        """Convert pixel bounds to geographic bounds."""
        r1, c1, r2, c2 = pixel_bounds
        gt = geotransform
        
        x1 = gt[0] + c1 * gt[1] + r1 * gt[2]
        y1 = gt[3] + c1 * gt[4] + r1 * gt[5]
        x2 = gt[0] + c2 * gt[1] + r2 * gt[2]
        y2 = gt[3] + c2 * gt[4] + r2 * gt[5]
        
        return {
            'min_x': min(x1, x2),
            'max_x': max(x1, x2),
            'min_y': min(y1, y2),
            'max_y': max(y1, y2)
        }
    
    def _generate_validation_report(self, validation_results, slc_data):
        """Generate a visual validation report."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('SAR Data Validation Report', fontsize=16)
        
        # Amplitude histogram
        amplitude = np.abs(slc_data)
        amplitude_db = 20 * np.log10(amplitude + 1e-10)
        
        axes[0, 0].hist(amplitude_db.flatten(), bins=100, alpha=0.7)
        axes[0, 0].set_title('Amplitude Distribution (dB)')
        axes[0, 0].set_xlabel('Amplitude (dB)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Phase histogram
        phase = np.angle(slc_data)
        axes[0, 1].hist(phase.flatten(), bins=100, alpha=0.7)
        axes[0, 1].set_title('Phase Distribution')
        axes[0, 1].set_xlabel('Phase (radians)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Amplitude image (subset)
        subset = amplitude[::10, ::10]  # Downsample for display
        im1 = axes[1, 0].imshow(subset, cmap='gray', aspect='auto')
        axes[1, 0].set_title('Amplitude Image (Downsampled)')
        plt.colorbar(im1, ax=axes[1, 0])
        
        # Phase image (subset)
        phase_subset = phase[::10, ::10]
        im2 = axes[1, 1].imshow(phase_subset, cmap='hsv', aspect='auto')
        axes[1, 1].set_title('Phase Image (Downsampled)')
        plt.colorbar(im2, ax=axes[1, 1])
        
        plt.tight_layout()
        
        # Save report
        report_path = self.output_path / "validation_report.png"
        plt.savefig(report_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Validation report saved to: {report_path}")


def run_all_examples():
    """Run all data I/O examples."""
    print("SARPYX Data I/O Examples")
    print("=" * 60)
    
    # Initialize examples
    examples = DataIOExamples()
    
    # Run examples
    try:
        examples.example_1_basic_reading()
        examples.example_2_format_conversion()
        examples.example_3_data_subsetting()
        examples.example_4_batch_processing()
        examples.example_5_memory_efficient_processing()
        examples.example_6_data_validation()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print(f"Output files saved to: {examples.output_path}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")


if __name__ == "__main__":
    run_all_examples()
