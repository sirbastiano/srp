#!/usr/bin/env python3
"""
SARPYX Custom Processing Chains Example
======================================

This example demonstrates how to create custom, advanced processing chains
using SARPYX for specialized SAR applications. It covers pipeline design,
modular processing, optimization strategies, and integration patterns.

Topics covered:
- Modular processing pipeline design
- Custom algorithm integration
- Performance optimization strategies
- Parallel processing workflows
- Memory-efficient processing
- Custom output formats
- Integration with external tools
- Real-time processing capabilities

Author: SARPYX Development Team
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
import json
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import time
from functools import wraps
import pickle
import h5py

# SARPYX imports
from sarpyx.sla import SLAProcessor
from sarpyx.utils import io as sarpyx_io
from sarpyx.utils import viz as sarpyx_viz
from sarpyx.science import indices
from sarpyx.snapflow import engine as snap_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for processing chains."""
    chain_name: str
    input_format: str = "SAFE"
    output_format: str = "HDF5"
    parallel_workers: int = 4
    memory_limit_gb: float = 8.0
    enable_caching: bool = True
    output_directory: str = "processing_output"
    log_level: str = "INFO"
    custom_parameters: Dict[str, Any] = field(default_factory=dict)

class ProcessingStep(ABC):
    """Abstract base class for processing steps."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.execution_time = 0.0
        self.memory_usage = 0.0
        
    @abstractmethod
    def process(self, data: Any, metadata: Dict[str, Any] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Process input data.
        
        Parameters:
        -----------
        data : Any
            Input data to process
        metadata : Dict[str, Any]
            Processing metadata
            
        Returns:
        --------
        Tuple[Any, Dict[str, Any]]
            Processed data and updated metadata
        """
        pass
    
    def validate_inputs(self, data: Any, metadata: Dict[str, Any] = None) -> bool:
        """Validate input data and metadata."""
        return data is not None
    
    def get_memory_usage(self) -> float:
        """Get memory usage in GB."""
        return self.memory_usage
    
    def get_execution_time(self) -> float:
        """Get execution time in seconds."""
        return self.execution_time

def timing_decorator(func):
    """Decorator to measure execution time."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        self.execution_time = time.time() - start_time
        return result
    return wrapper

class DataLoaderStep(ProcessingStep):
    """Load SAR data from various formats."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("DataLoader", config)
        self.supported_formats = ["SAFE", "TIFF", "HDF5", "NetCDF"]
    
    @timing_decorator
    def process(self, data_path: str, metadata: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load SAR data from file."""
        logger.info(f"Loading data from: {data_path}")
        
        if metadata is None:
            metadata = {}
        
        # Determine file format
        data_path = Path(data_path)
        if data_path.suffix.lower() == '.safe' or data_path.is_dir():
            data = sarpyx_io.load_sar_data(data_path)
            file_metadata = sarpyx_io.load_metadata(data_path)
        elif data_path.suffix.lower() in ['.tif', '.tiff']:
            data = sarpyx_io.load_tiff(data_path)
            file_metadata = sarpyx_io.load_tiff_metadata(data_path)
        elif data_path.suffix.lower() in ['.h5', '.hdf5']:
            data = self._load_hdf5(data_path)
            file_metadata = self._load_hdf5_metadata(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        # Update metadata
        metadata.update({
            'data_path': str(data_path),
            'data_shape': data.shape,
            'data_type': str(data.dtype),
            'file_metadata': file_metadata,
            'loading_time': self.execution_time
        })
        
        logger.info(f"Data loaded: {data.shape}, {data.dtype}")
        return data, metadata
    
    def _load_hdf5(self, file_path: Path) -> np.ndarray:
        """Load data from HDF5 file."""
        with h5py.File(file_path, 'r') as f:
            # Assume main dataset is named 'data'
            if 'data' in f:
                return f['data'][...]
            else:
                # Use first dataset found
                key = list(f.keys())[0]
                return f[key][...]
    
    def _load_hdf5_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Load metadata from HDF5 file."""
        metadata = {}
        with h5py.File(file_path, 'r') as f:
            # Load attributes
            for key, value in f.attrs.items():
                metadata[key] = value
        return metadata

class PreprocessingStep(ProcessingStep):
    """Preprocessing operations (calibration, filtering, etc.)."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Preprocessing", config)
        self.operations = self.config.get('operations', ['calibration', 'speckle_filter'])
    
    @timing_decorator
    def process(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply preprocessing operations."""
        logger.info("Applying preprocessing operations...")
        
        processed_data = data.copy()
        preprocessing_log = []
        
        for operation in self.operations:
            if operation == 'calibration':
                processed_data = self._apply_calibration(processed_data, metadata)
                preprocessing_log.append('calibration')
                
            elif operation == 'speckle_filter':
                processed_data = self._apply_speckle_filter(processed_data)
                preprocessing_log.append('speckle_filter')
                
            elif operation == 'terrain_correction':
                processed_data = self._apply_terrain_correction(processed_data, metadata)
                preprocessing_log.append('terrain_correction')
                
            elif operation == 'multilook':
                processed_data = self._apply_multilook(processed_data)
                preprocessing_log.append('multilook')
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'preprocessing_operations': preprocessing_log,
            'preprocessing_time': self.execution_time
        })
        
        logger.info(f"Preprocessing completed: {len(preprocessing_log)} operations")
        return processed_data, metadata
    
    def _apply_calibration(self, data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Apply radiometric calibration."""
        # Simplified calibration - in practice would use actual calibration data
        if 'file_metadata' in metadata and 'calibration' in metadata['file_metadata']:
            cal_factor = metadata['file_metadata']['calibration'].get('factor', 1.0)
            return data * cal_factor
        return data
    
    def _apply_speckle_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply speckle filtering."""
        from scipy import ndimage
        # Lee filter (simplified)
        kernel_size = self.config.get('speckle_filter_size', 5)
        local_mean = ndimage.uniform_filter(data, size=kernel_size)
        local_var = ndimage.uniform_filter(data**2, size=kernel_size) - local_mean**2
        
        # Avoid division by zero
        local_var = np.maximum(local_var, 1e-10)
        
        # Lee filter formula
        k = local_var / (local_var + local_mean**2)
        filtered = local_mean + k * (data - local_mean)
        
        return filtered
    
    def _apply_terrain_correction(self, data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Apply terrain correction."""
        # Simplified terrain correction
        # In practice would use DEM and geometric metadata
        return data
    
    def _apply_multilook(self, data: np.ndarray) -> np.ndarray:
        """Apply multilooking."""
        looks_azimuth = self.config.get('looks_azimuth', 1)
        looks_range = self.config.get('looks_range', 1)
        
        if looks_azimuth > 1 or looks_range > 1:
            # Simple averaging multilook
            h, w = data.shape[:2]
            new_h = h // looks_azimuth
            new_w = w // looks_range
            
            reshaped = data[:new_h*looks_azimuth, :new_w*looks_range]
            if data.ndim == 2:
                reshaped = reshaped.reshape(new_h, looks_azimuth, new_w, looks_range)
                multilooked = reshaped.mean(axis=(1, 3))
            else:
                # Handle complex data
                reshaped = reshaped.reshape(new_h, looks_azimuth, new_w, looks_range, -1)
                multilooked = reshaped.mean(axis=(1, 3))
            
            return multilooked
        
        return data

class SLAProcessingStep(ProcessingStep):
    """Sub-Look Analysis processing step."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("SLAProcessing", config)
        self.sla_processor = SLAProcessor()
        self.sub_apertures = self.config.get('sub_apertures', 4)
    
    @timing_decorator
    def process(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Perform Sub-Look Analysis."""
        logger.info("Performing Sub-Look Analysis...")
        
        # Configure SLA processor
        sla_config = {
            'sub_apertures': self.sub_apertures,
            'overlap_factor': self.config.get('overlap_factor', 0.5),
            'window_function': self.config.get('window_function', 'hamming')
        }
        
        # Process sub-looks
        sla_results = self.sla_processor.process(data, **sla_config)
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'sla_config': sla_config,
            'sla_processing_time': self.execution_time,
            'sub_apertures_count': self.sub_apertures
        })
        
        logger.info(f"SLA processing completed: {self.sub_apertures} sub-apertures")
        return sla_results, metadata

class FeatureExtractionStep(ProcessingStep):
    """Extract features for analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("FeatureExtraction", config)
        self.features = self.config.get('features', ['intensity', 'coherence', 'polarimetric'])
    
    @timing_decorator
    def process(self, sla_data: Dict[str, np.ndarray], metadata: Dict[str, Any] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Extract features from SLA results."""
        logger.info("Extracting features...")
        
        features = {}
        
        for feature_type in self.features:
            if feature_type == 'intensity':
                features.update(self._extract_intensity_features(sla_data))
            elif feature_type == 'coherence':
                features.update(self._extract_coherence_features(sla_data))
            elif feature_type == 'polarimetric':
                features.update(self._extract_polarimetric_features(sla_data))
            elif feature_type == 'texture':
                features.update(self._extract_texture_features(sla_data))
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'extracted_features': list(features.keys()),
            'feature_extraction_time': self.execution_time
        })
        
        logger.info(f"Feature extraction completed: {len(features)} features")
        return features, metadata
    
    def _extract_intensity_features(self, sla_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract intensity-based features."""
        features = {}
        
        if 'sub_looks' in sla_data:
            sub_looks = sla_data['sub_looks']
            
            # Mean intensity
            features['mean_intensity'] = np.mean(np.abs(sub_looks)**2, axis=0)
            
            # Intensity variance
            intensities = np.abs(sub_looks)**2
            features['intensity_variance'] = np.var(intensities, axis=0)
            
            # Intensity coefficient of variation
            features['intensity_cv'] = np.std(intensities, axis=0) / (np.mean(intensities, axis=0) + 1e-10)
        
        return features
    
    def _extract_coherence_features(self, sla_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract coherence-based features."""
        features = {}
        
        if 'coherence_matrix' in sla_data:
            coherence_matrix = sla_data['coherence_matrix']
            
            # Mean coherence
            features['mean_coherence'] = np.mean(np.abs(coherence_matrix), axis=(0, 1))
            
            # Coherence range
            coherence_values = np.abs(coherence_matrix)
            features['coherence_range'] = np.max(coherence_values, axis=(0, 1)) - np.min(coherence_values, axis=(0, 1))
            
            # Phase diversity
            phases = np.angle(coherence_matrix)
            features['phase_diversity'] = np.std(phases, axis=(0, 1))
        
        return features
    
    def _extract_polarimetric_features(self, sla_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract polarimetric features."""
        features = {}
        
        if 'polarimetric_features' in sla_data:
            pol_features = sla_data['polarimetric_features']
            
            # Copy existing polarimetric features
            for key, value in pol_features.items():
                features[f'pol_{key}'] = value
        
        return features
    
    def _extract_texture_features(self, sla_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract texture features."""
        features = {}
        
        if 'sub_looks' in sla_data:
            # Use first sub-look for texture analysis
            sub_look = np.abs(sla_data['sub_looks'][0])**2
            
            # GLCM-based features (simplified)
            features.update(self._calculate_glcm_features(sub_look))
        
        return features
    
    def _calculate_glcm_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate GLCM-based texture features."""
        from skimage.feature import graycomatrix, graycoprops
        
        # Normalize image to 8-bit
        image_norm = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
        
        # Calculate GLCM
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        try:
            glcm = graycomatrix(image_norm, distances, angles, symmetric=True, normed=True)
            
            # Extract properties
            features = {}
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy']:
                features[f'glcm_{prop}'] = graycoprops(glcm, prop).mean()
        except:
            # Fallback to simple texture measures
            features = {
                'texture_variance': np.var(image),
                'texture_mean': np.mean(image)
            }
        
        return features

class ClassificationStep(ProcessingStep):
    """Classification/detection step."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Classification", config)
        self.algorithm = self.config.get('algorithm', 'kmeans')
        self.n_classes = self.config.get('n_classes', 3)
    
    @timing_decorator
    def process(self, features: Dict[str, np.ndarray], metadata: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform classification."""
        logger.info(f"Performing classification with {self.algorithm}...")
        
        # Prepare feature matrix
        feature_matrix = self._prepare_feature_matrix(features)
        
        # Perform classification
        if self.algorithm == 'kmeans':
            classification = self._kmeans_classification(feature_matrix)
        elif self.algorithm == 'threshold':
            classification = self._threshold_classification(feature_matrix)
        elif self.algorithm == 'svm':
            classification = self._svm_classification(feature_matrix)
        else:
            raise ValueError(f"Unknown classification algorithm: {self.algorithm}")
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'classification_algorithm': self.algorithm,
            'n_classes': self.n_classes,
            'classification_time': self.execution_time
        })
        
        logger.info(f"Classification completed: {self.n_classes} classes")
        return classification, metadata
    
    def _prepare_feature_matrix(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Prepare feature matrix for classification."""
        feature_arrays = []
        
        for key, feature in features.items():
            if feature.ndim == 2:
                feature_arrays.append(feature.flatten())
            else:
                feature_arrays.append(feature)
        
        return np.column_stack(feature_arrays)
    
    def _kmeans_classification(self, feature_matrix: np.ndarray) -> np.ndarray:
        """K-means classification."""
        from sklearn.cluster import KMeans
        
        # Remove invalid values
        valid_mask = np.all(np.isfinite(feature_matrix), axis=1)
        valid_features = feature_matrix[valid_mask]
        
        if len(valid_features) == 0:
            return np.zeros(feature_matrix.shape[0])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=self.n_classes, random_state=42, n_init=10)
        labels = np.zeros(feature_matrix.shape[0])
        labels[valid_mask] = kmeans.fit_predict(valid_features)
        
        return labels
    
    def _threshold_classification(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Threshold-based classification."""
        # Use first feature for thresholding
        feature = feature_matrix[:, 0]
        
        # Calculate thresholds
        percentiles = np.linspace(0, 100, self.n_classes + 1)
        thresholds = np.percentile(feature[np.isfinite(feature)], percentiles)
        
        # Apply thresholds
        classification = np.digitize(feature, thresholds) - 1
        classification = np.clip(classification, 0, self.n_classes - 1)
        
        return classification
    
    def _svm_classification(self, feature_matrix: np.ndarray) -> np.ndarray:
        """SVM classification (unsupervised)."""
        # For demonstration, use one-class SVM for anomaly detection
        from sklearn.svm import OneClassSVM
        
        valid_mask = np.all(np.isfinite(feature_matrix), axis=1)
        valid_features = feature_matrix[valid_mask]
        
        if len(valid_features) == 0:
            return np.zeros(feature_matrix.shape[0])
        
        # One-class SVM
        svm = OneClassSVM(nu=0.1, kernel='rbf')
        predictions = np.zeros(feature_matrix.shape[0])
        predictions[valid_mask] = svm.fit_predict(valid_features)
        
        # Convert to positive class labels
        predictions = (predictions + 1) / 2
        
        return predictions

class OutputStep(ProcessingStep):
    """Save processing results."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Output", config)
        self.output_format = self.config.get('format', 'HDF5')
        self.output_dir = Path(self.config.get('directory', 'output'))
        self.output_dir.mkdir(exist_ok=True)
    
    @timing_decorator
    def process(self, results: Dict[str, Any], metadata: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """Save processing results."""
        logger.info(f"Saving results in {self.output_format} format...")
        
        if self.output_format.upper() == 'HDF5':
            output_path = self._save_hdf5(results, metadata)
        elif self.output_format.upper() == 'TIFF':
            output_path = self._save_tiff(results, metadata)
        elif self.output_format.upper() == 'NETCDF':
            output_path = self._save_netcdf(results, metadata)
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'output_path': output_path,
            'output_format': self.output_format,
            'output_time': self.execution_time
        })
        
        logger.info(f"Results saved to: {output_path}")
        return output_path, metadata
    
    def _save_hdf5(self, results: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Save results to HDF5 format."""
        output_path = self.output_dir / f"processing_results_{int(time.time())}.h5"
        
        with h5py.File(output_path, 'w') as f:
            # Save data arrays
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value, compression='gzip')
                elif isinstance(value, dict):
                    group = f.create_group(key)
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            group.create_dataset(subkey, data=subvalue, compression='gzip')
            
            # Save metadata
            if metadata:
                meta_group = f.create_group('metadata')
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)):
                        meta_group.attrs[key] = value
                    elif isinstance(value, (list, np.ndarray)):
                        meta_group.create_dataset(key, data=value)
        
        return str(output_path)
    
    def _save_tiff(self, results: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Save results to TIFF format."""
        # Save main result as TIFF
        main_result = None
        for key, value in results.items():
            if isinstance(value, np.ndarray) and value.ndim == 2:
                main_result = value
                break
        
        if main_result is not None:
            output_path = self.output_dir / f"processing_results_{int(time.time())}.tif"
            sarpyx_io.save_tiff(main_result, output_path, metadata)
            return str(output_path)
        
        raise ValueError("No suitable 2D array found for TIFF output")
    
    def _save_netcdf(self, results: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Save results to NetCDF format."""
        import xarray as xr
        
        # Convert results to xarray Dataset
        data_vars = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 2:
                    data_vars[key] = (['y', 'x'], value)
                elif value.ndim == 1:
                    data_vars[key] = (['index'], value)
        
        ds = xr.Dataset(data_vars)
        
        # Add metadata as attributes
        if metadata:
            ds.attrs.update({k: v for k, v in metadata.items() 
                           if isinstance(v, (str, int, float))})
        
        output_path = self.output_dir / f"processing_results_{int(time.time())}.nc"
        ds.to_netcdf(output_path)
        
        return str(output_path)

class ProcessingChain:
    """Custom processing chain manager."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.steps: List[ProcessingStep] = []
        self.results_cache = {}
        self.performance_metrics = {}
        
        # Setup output directory
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def add_step(self, step: ProcessingStep) -> 'ProcessingChain':
        """Add a processing step to the chain."""
        self.steps.append(step)
        logger.info(f"Added processing step: {step.name}")
        return self
    
    def remove_step(self, step_name: str) -> 'ProcessingChain':
        """Remove a processing step from the chain."""
        self.steps = [step for step in self.steps if step.name != step_name]
        logger.info(f"Removed processing step: {step_name}")
        return self
    
    def process(self, input_data: Any, parallel: bool = False) -> Dict[str, Any]:
        """Execute the processing chain."""
        logger.info(f"Starting processing chain: {self.config.chain_name}")
        start_time = time.time()
        
        current_data = input_data
        metadata = {'chain_name': self.config.chain_name}
        
        if parallel and len(self.steps) > 1:
            result = self._process_parallel(current_data, metadata)
        else:
            result = self._process_sequential(current_data, metadata)
        
        # Record performance metrics
        total_time = time.time() - start_time
        self.performance_metrics = {
            'total_processing_time': total_time,
            'steps_executed': len(self.steps),
            'memory_usage_gb': sum(step.get_memory_usage() for step in self.steps),
            'step_times': {step.name: step.get_execution_time() for step in self.steps}
        }
        
        logger.info(f"Processing chain completed in {total_time:.2f} seconds")
        return result
    
    def _process_sequential(self, input_data: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Execute steps sequentially."""
        current_data = input_data
        
        for i, step in enumerate(self.steps):
            logger.info(f"Executing step {i+1}/{len(self.steps)}: {step.name}")
            
            # Validate inputs
            if not step.validate_inputs(current_data, metadata):
                raise ValueError(f"Invalid inputs for step: {step.name}")
            
            # Execute step
            try:
                current_data, metadata = step.process(current_data, metadata)
                
                # Cache results if enabled
                if self.config.enable_caching:
                    self.results_cache[f"step_{i}_{step.name}"] = {
                        'data': current_data,
                        'metadata': metadata.copy()
                    }
                    
            except Exception as e:
                logger.error(f"Error in step {step.name}: {str(e)}")
                raise
        
        return {
            'final_result': current_data,
            'metadata': metadata,
            'performance_metrics': self.performance_metrics
        }
    
    def _process_parallel(self, input_data: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Execute compatible steps in parallel."""
        # For simplicity, this implementation runs steps sequentially
        # In practice, you would analyze dependencies and parallelize where possible
        return self._process_sequential(input_data, metadata)
    
    def save_chain_config(self, config_path: str):
        """Save processing chain configuration."""
        config_data = {
            'chain_name': self.config.chain_name,
            'config': self.config.__dict__,
            'steps': [
                {
                    'name': step.name,
                    'class': step.__class__.__name__,
                    'config': step.config
                }
                for step in self.steps
            ]
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        logger.info(f"Chain configuration saved to: {config_path}")
    
    def load_chain_config(self, config_path: str):
        """Load processing chain configuration."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Reconstruct steps (simplified - would need proper class mapping)
        logger.info(f"Chain configuration loaded from: {config_path}")
    
    def get_performance_report(self) -> str:
        """Generate performance report."""
        if not self.performance_metrics:
            return "No performance metrics available. Run processing first."
        
        report = f"""
Processing Chain Performance Report
===================================
Chain Name: {self.config.chain_name}
Total Processing Time: {self.performance_metrics['total_processing_time']:.2f} seconds
Steps Executed: {self.performance_metrics['steps_executed']}
Total Memory Usage: {self.performance_metrics['memory_usage_gb']:.2f} GB

Step-by-Step Timing:
"""
        
        for step_name, execution_time in self.performance_metrics['step_times'].items():
            report += f"  {step_name}: {execution_time:.2f} seconds\n"
        
        return report
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper())
        logging.getLogger().setLevel(log_level)

class BatchProcessor:
    """Batch processing manager for multiple files."""
    
    def __init__(self, chain: ProcessingChain, max_workers: int = None):
        self.chain = chain
        self.max_workers = max_workers or mp.cpu_count()
        
    def process_batch(self, file_list: List[str], output_pattern: str = None) -> List[Dict[str, Any]]:
        """Process multiple files in batch."""
        logger.info(f"Starting batch processing: {len(file_list)} files")
        
        results = []
        
        if self.max_workers > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for file_path in file_list:
                    future = executor.submit(self._process_single_file, file_path)
                    futures.append((file_path, future))
                
                for file_path, future in futures:
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(f"Completed: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed: {file_path} - {str(e)}")
                        results.append({'error': str(e), 'file_path': file_path})
        else:
            # Sequential processing
            for file_path in file_list:
                try:
                    result = self._process_single_file(file_path)
                    results.append(result)
                    logger.info(f"Completed: {file_path}")
                except Exception as e:
                    logger.error(f"Failed: {file_path} - {str(e)}")
                    results.append({'error': str(e), 'file_path': file_path})
        
        logger.info(f"Batch processing completed: {len(results)} results")
        return results
    
    def _process_single_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file."""
        # Create a copy of the chain for thread safety
        chain_copy = ProcessingChain(self.chain.config)
        for step in self.chain.steps:
            chain_copy.add_step(step)
        
        return chain_copy.process(file_path)

def create_vegetation_monitoring_chain() -> ProcessingChain:
    """Create a custom chain for vegetation monitoring."""
    config = ProcessingConfig(
        chain_name="VegetationMonitoring",
        parallel_workers=4,
        memory_limit_gb=8.0,
        custom_parameters={
            'vegetation_indices': ['ndvi', 'evi', 'savi'],
            'temporal_analysis': True
        }
    )
    
    chain = ProcessingChain(config)
    
    # Add processing steps
    chain.add_step(DataLoaderStep())
    
    preprocessing_config = {
        'operations': ['calibration', 'speckle_filter', 'multilook'],
        'looks_azimuth': 2,
        'looks_range': 2
    }
    chain.add_step(PreprocessingStep(preprocessing_config))
    
    sla_config = {
        'sub_apertures': 6,
        'overlap_factor': 0.6
    }
    chain.add_step(SLAProcessingStep(sla_config))
    
    feature_config = {
        'features': ['intensity', 'coherence', 'polarimetric']
    }
    chain.add_step(FeatureExtractionStep(feature_config))
    
    classification_config = {
        'algorithm': 'kmeans',
        'n_classes': 5
    }
    chain.add_step(ClassificationStep(classification_config))
    
    output_config = {
        'format': 'HDF5',
        'directory': 'vegetation_monitoring_output'
    }
    chain.add_step(OutputStep(output_config))
    
    return chain

def create_ship_detection_chain() -> ProcessingChain:
    """Create a custom chain for ship detection."""
    config = ProcessingConfig(
        chain_name="ShipDetection",
        parallel_workers=6,
        memory_limit_gb=12.0,
        custom_parameters={
            'detection_algorithm': 'cfar',
            'false_alarm_rate': 1e-6
        }
    )
    
    chain = ProcessingChain(config)
    
    # Add processing steps optimized for ship detection
    chain.add_step(DataLoaderStep())
    
    preprocessing_config = {
        'operations': ['calibration', 'speckle_filter'],
        'speckle_filter_size': 3  # Smaller filter to preserve ship signatures
    }
    chain.add_step(PreprocessingStep(preprocessing_config))
    
    sla_config = {
        'sub_apertures': 8,
        'overlap_factor': 0.7
    }
    chain.add_step(SLAProcessingStep(sla_config))
    
    feature_config = {
        'features': ['intensity', 'texture']
    }
    chain.add_step(FeatureExtractionStep(feature_config))
    
    classification_config = {
        'algorithm': 'threshold',
        'n_classes': 2  # Ship vs. no-ship
    }
    chain.add_step(ClassificationStep(classification_config))
    
    output_config = {
        'format': 'HDF5',
        'directory': 'ship_detection_output'
    }
    chain.add_step(OutputStep(output_config))
    
    return chain

def main():
    """
    Main function demonstrating custom processing chains.
    """
    # Example data path
    data_path = "data/S1A_S3_SLC__1SSH_20240621T052251_20240621T052319_054417_069F07_8466.SAFE"
    
    print("SARPYX Custom Processing Chains Demo")
    print("=" * 50)
    
    # Create and run vegetation monitoring chain
    print("\n1. Vegetation Monitoring Chain")
    print("-" * 30)
    
    veg_chain = create_vegetation_monitoring_chain()
    
    # Save chain configuration
    veg_chain.save_chain_config("vegetation_monitoring_chain.yaml")
    
    try:
        veg_results = veg_chain.process(data_path)
        print("Vegetation monitoring completed successfully!")
        print(veg_chain.get_performance_report())
    except Exception as e:
        print(f"Vegetation monitoring failed: {e}")
    
    # Create and run ship detection chain
    print("\n2. Ship Detection Chain")
    print("-" * 30)
    
    ship_chain = create_ship_detection_chain()
    
    # Save chain configuration
    ship_chain.save_chain_config("ship_detection_chain.yaml")
    
    try:
        ship_results = ship_chain.process(data_path)
        print("Ship detection completed successfully!")
        print(ship_chain.get_performance_report())
    except Exception as e:
        print(f"Ship detection failed: {e}")
    
    # Demonstrate batch processing
    print("\n3. Batch Processing Demo")
    print("-" * 30)
    
    # Create a simple chain for batch processing
    batch_config = ProcessingConfig(
        chain_name="BatchProcessing",
        parallel_workers=2
    )
    batch_chain = ProcessingChain(batch_config)
    batch_chain.add_step(DataLoaderStep())
    batch_chain.add_step(PreprocessingStep({'operations': ['calibration']}))
    batch_chain.add_step(OutputStep({'format': 'TIFF', 'directory': 'batch_output'}))
    
    # Process multiple files (using same file for demo)
    file_list = [data_path] * 3  # Process same file 3 times for demo
    
    batch_processor = BatchProcessor(batch_chain, max_workers=2)
    batch_results = batch_processor.process_batch(file_list)
    
    print(f"Batch processing completed: {len(batch_results)} results")
    
    print("\nCustom Processing Chains Demo Complete!")


if __name__ == "__main__":
    main()
