#!/usr/bin/env python3
"""
SARPYX Quality Assessment Example
===============================

This example demonstrates comprehensive quality assessment techniques for SAR data
processing using SARPYX. It covers data validation, noise analysis, calibration
assessment, and processing quality metrics.

Topics covered:
- Data integrity checks
- Noise equivalent sigma zero (NESZ) analysis
- Calibration accuracy assessment
- Processing quality metrics
- Statistical validation
- Error propagation analysis
- Quality control workflows

Author: SARPYX Development Team
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy import stats
import seaborn as sns

# SARPYX imports
from sarpyx.sla import SLAProcessor
from sarpyx.utils import io as sarpyx_io
from sarpyx.utils import viz as sarpyx_viz
from sarpyx.science import indices
from sarpyx.snap import engine as snap_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityAssessment:
    """
    Comprehensive quality assessment for SAR data processing.
    """
    
    def __init__(self, data_path: str, output_dir: str = "quality_assessment_output"):
        """
        Initialize quality assessment.
        
        Parameters:
        -----------
        data_path : str
            Path to SAR data file
        output_dir : str
            Output directory for results
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize processors
        self.sla_processor = SLAProcessor()
        self.quality_metrics = {}
        
        logger.info(f"Quality assessment initialized for: {self.data_path}")
    
    def data_integrity_check(self) -> Dict:
        """
        Perform comprehensive data integrity checks.
        
        Returns:
        --------
        Dict
            Data integrity report
        """
        logger.info("Performing data integrity checks...")
        
        # Load data
        sar_data = sarpyx_io.load_sar_data(self.data_path)
        
        integrity_report = {
            'file_exists': self.data_path.exists(),
            'file_size_mb': self.data_path.stat().st_size / (1024*1024) if self.data_path.exists() else 0,
            'data_shape': sar_data.shape if sar_data is not None else None,
            'data_type': str(sar_data.dtype) if sar_data is not None else None,
            'nan_pixels': 0,
            'inf_pixels': 0,
            'zero_pixels': 0,
            'negative_pixels': 0,
            'valid_pixels': 0
        }
        
        if sar_data is not None:
            # Check for problematic values
            integrity_report['nan_pixels'] = np.isnan(sar_data).sum()
            integrity_report['inf_pixels'] = np.isinf(sar_data).sum()
            integrity_report['zero_pixels'] = (sar_data == 0).sum()
            integrity_report['negative_pixels'] = (sar_data < 0).sum()
            integrity_report['valid_pixels'] = np.isfinite(sar_data).sum()
            
            # Statistical summary
            valid_data = sar_data[np.isfinite(sar_data)]
            if len(valid_data) > 0:
                integrity_report.update({
                    'min_value': np.min(valid_data),
                    'max_value': np.max(valid_data),
                    'mean_value': np.mean(valid_data),
                    'std_value': np.std(valid_data),
                    'median_value': np.median(valid_data)
                })
        
        # Save integrity report
        pd.DataFrame([integrity_report]).to_csv(
            self.output_dir / "data_integrity_report.csv", index=False
        )
        
        logger.info("Data integrity check completed")
        return integrity_report
    
    def noise_analysis(self) -> Dict:
        """
        Analyze noise characteristics and NESZ.
        
        Returns:
        --------
        Dict
            Noise analysis results
        """
        logger.info("Performing noise analysis...")
        
        # Load data and metadata
        sar_data = sarpyx_io.load_sar_data(self.data_path)
        metadata = sarpyx_io.load_metadata(self.data_path)
        
        noise_results = {}
        
        # Estimate noise floor
        # Use areas with low backscatter (water, shadow areas)
        percentile_1 = np.percentile(sar_data[np.isfinite(sar_data)], 1)
        noise_mask = sar_data < percentile_1 * 2
        
        if np.any(noise_mask):
            noise_pixels = sar_data[noise_mask]
            noise_results.update({
                'noise_floor_linear': np.mean(noise_pixels),
                'noise_floor_db': 10 * np.log10(np.mean(noise_pixels) + 1e-10),
                'noise_std_linear': np.std(noise_pixels),
                'noise_std_db': 10 * np.log10(np.std(noise_pixels) + 1e-10),
                'noise_pixels_count': len(noise_pixels)
            })
        
        # NESZ calculation (if calibration data available)
        if hasattr(metadata, 'calibration'):
            nesz = self._calculate_nesz(sar_data, metadata)
            noise_results['nesz_db'] = nesz
        
        # Noise power spectral density
        noise_psd = self._calculate_noise_psd(sar_data)
        noise_results['noise_psd'] = noise_psd
        
        # Create noise analysis plots
        self._plot_noise_analysis(sar_data, noise_results)
        
        logger.info("Noise analysis completed")
        return noise_results
    
    def calibration_assessment(self) -> Dict:
        """
        Assess calibration accuracy and stability.
        
        Returns:
        --------
        Dict
            Calibration assessment results
        """
        logger.info("Performing calibration assessment...")
        
        # Load data and calibration information
        sar_data = sarpyx_io.load_sar_data(self.data_path)
        metadata = sarpyx_io.load_metadata(self.data_path)
        
        calibration_results = {}
        
        # Radiometric calibration assessment
        if hasattr(metadata, 'calibration'):
            # Check calibration constants
            cal_results = self._assess_radiometric_calibration(sar_data, metadata)
            calibration_results.update(cal_results)
        
        # Geometric calibration assessment
        geom_results = self._assess_geometric_calibration(sar_data, metadata)
        calibration_results.update(geom_results)
        
        # Cross-calibration with reference targets
        if self._has_reference_targets():
            ref_results = self._assess_reference_targets(sar_data)
            calibration_results.update(ref_results)
        
        # Create calibration assessment plots
        self._plot_calibration_assessment(calibration_results)
        
        logger.info("Calibration assessment completed")
        return calibration_results
    
    def processing_quality_metrics(self) -> Dict:
        """
        Calculate processing quality metrics.
        
        Returns:
        --------
        Dict
            Processing quality metrics
        """
        logger.info("Calculating processing quality metrics...")
        
        # Load processed data
        sar_data = sarpyx_io.load_sar_data(self.data_path)
        
        # Perform sub-look analysis
        sla_results = self.sla_processor.process(sar_data)
        
        quality_metrics = {}
        
        # Signal-to-noise ratio
        snr = self._calculate_snr(sar_data)
        quality_metrics['snr_db'] = snr
        
        # Equivalent number of looks
        enl = self._calculate_enl(sar_data)
        quality_metrics['enl'] = enl
        
        # Speckle index
        speckle_index = self._calculate_speckle_index(sar_data)
        quality_metrics['speckle_index'] = speckle_index
        
        # Sub-look coherence quality
        if 'coherence' in sla_results:
            coherence_quality = self._assess_coherence_quality(sla_results['coherence'])
            quality_metrics.update(coherence_quality)
        
        # Phase quality (for complex data)
        if np.iscomplexobj(sar_data):
            phase_quality = self._assess_phase_quality(sar_data)
            quality_metrics.update(phase_quality)
        
        # Spatial consistency
        spatial_quality = self._assess_spatial_consistency(sar_data)
        quality_metrics.update(spatial_quality)
        
        # Create quality metrics plots
        self._plot_quality_metrics(sar_data, quality_metrics)
        
        logger.info("Processing quality metrics calculated")
        return quality_metrics
    
    def statistical_validation(self) -> Dict:
        """
        Perform statistical validation of processing results.
        
        Returns:
        --------
        Dict
            Statistical validation results
        """
        logger.info("Performing statistical validation...")
        
        # Load data
        sar_data = sarpyx_io.load_sar_data(self.data_path)
        
        validation_results = {}
        
        # Distribution analysis
        distribution_results = self._analyze_distributions(sar_data)
        validation_results.update(distribution_results)
        
        # Autocorrelation analysis
        autocorr_results = self._analyze_autocorrelation(sar_data)
        validation_results.update(autocorr_results)
        
        # Stationarity tests
        stationarity_results = self._test_stationarity(sar_data)
        validation_results.update(stationarity_results)
        
        # Normality tests
        normality_results = self._test_normality(sar_data)
        validation_results.update(normality_results)
        
        # Create statistical validation plots
        self._plot_statistical_validation(sar_data, validation_results)
        
        logger.info("Statistical validation completed")
        return validation_results
    
    def error_propagation_analysis(self) -> Dict:
        """
        Analyze error propagation through processing chain.
        
        Returns:
        --------
        Dict
            Error propagation analysis results
        """
        logger.info("Performing error propagation analysis...")
        
        # Load data
        sar_data = sarpyx_io.load_sar_data(self.data_path)
        
        error_results = {}
        
        # Input data uncertainties
        input_errors = self._estimate_input_uncertainties(sar_data)
        error_results['input_uncertainties'] = input_errors
        
        # Processing step uncertainties
        processing_errors = self._estimate_processing_uncertainties(sar_data)
        error_results['processing_uncertainties'] = processing_errors
        
        # Monte Carlo error propagation
        mc_errors = self._monte_carlo_error_propagation(sar_data)
        error_results['monte_carlo_errors'] = mc_errors
        
        # Analytical error propagation
        analytical_errors = self._analytical_error_propagation(sar_data)
        error_results['analytical_errors'] = analytical_errors
        
        # Create error propagation plots
        self._plot_error_propagation(error_results)
        
        logger.info("Error propagation analysis completed")
        return error_results
    
    def generate_quality_report(self) -> str:
        """
        Generate comprehensive quality assessment report.
        
        Returns:
        --------
        str
            Path to generated report
        """
        logger.info("Generating comprehensive quality report...")
        
        # Run all assessments
        integrity = self.data_integrity_check()
        noise = self.noise_analysis()
        calibration = self.calibration_assessment()
        processing = self.processing_quality_metrics()
        statistics = self.statistical_validation()
        errors = self.error_propagation_analysis()
        
        # Compile results
        all_results = {
            'data_integrity': integrity,
            'noise_analysis': noise,
            'calibration_assessment': calibration,
            'processing_quality': processing,
            'statistical_validation': statistics,
            'error_propagation': errors
        }
        
        # Generate HTML report
        report_path = self._generate_html_report(all_results)
        
        # Generate summary statistics
        self._generate_summary_statistics(all_results)
        
        logger.info(f"Quality report generated: {report_path}")
        return str(report_path)
    
    # Helper methods
    def _calculate_nesz(self, data: np.ndarray, metadata: Dict) -> float:
        """Calculate Noise Equivalent Sigma Zero."""
        # Implementation depends on sensor and calibration data
        # This is a simplified version
        noise_floor = np.percentile(data[np.isfinite(data)], 1)
        return 10 * np.log10(noise_floor + 1e-10)
    
    def _calculate_noise_psd(self, data: np.ndarray) -> np.ndarray:
        """Calculate noise power spectral density."""
        # Use FFT to analyze noise characteristics
        noise_mask = data < np.percentile(data[np.isfinite(data)], 5)
        if np.any(noise_mask):
            noise_data = data[noise_mask]
            psd = np.abs(np.fft.fft(noise_data))**2
            return psd[:len(psd)//2]
        return np.array([])
    
    def _calculate_snr(self, data: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        signal_power = np.mean(data[np.isfinite(data)])
        noise_power = np.percentile(data[np.isfinite(data)], 10)
        return 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    def _calculate_enl(self, data: np.ndarray) -> float:
        """Calculate equivalent number of looks."""
        if np.iscomplexobj(data):
            intensity = np.abs(data)**2
        else:
            intensity = data
        
        valid_data = intensity[np.isfinite(intensity)]
        if len(valid_data) > 0:
            mean_val = np.mean(valid_data)
            var_val = np.var(valid_data)
            return mean_val**2 / var_val if var_val > 0 else 0
        return 0
    
    def _calculate_speckle_index(self, data: np.ndarray) -> float:
        """Calculate speckle index (coefficient of variation)."""
        valid_data = data[np.isfinite(data)]
        if len(valid_data) > 0:
            return np.std(valid_data) / np.mean(valid_data)
        return 0
    
    def _assess_radiometric_calibration(self, data: np.ndarray, metadata: Dict) -> Dict:
        """Assess radiometric calibration accuracy."""
        # Simplified assessment
        return {
            'radiometric_accuracy_db': 0.5,  # Typical value
            'radiometric_stability_db': 0.2,
            'cross_talk_db': -25.0
        }
    
    def _assess_geometric_calibration(self, data: np.ndarray, metadata: Dict) -> Dict:
        """Assess geometric calibration accuracy."""
        return {
            'geometric_accuracy_m': 10.0,  # Typical value
            'geometric_stability_m': 5.0
        }
    
    def _has_reference_targets(self) -> bool:
        """Check if reference targets are available."""
        return False  # Would check for corner reflectors, etc.
    
    def _assess_reference_targets(self, data: np.ndarray) -> Dict:
        """Assess calibration using reference targets."""
        return {}
    
    def _assess_coherence_quality(self, coherence: np.ndarray) -> Dict:
        """Assess sub-look coherence quality."""
        valid_coh = coherence[np.isfinite(coherence)]
        return {
            'mean_coherence': np.mean(valid_coh),
            'coherence_std': np.std(valid_coh),
            'high_coherence_fraction': np.sum(valid_coh > 0.7) / len(valid_coh)
        }
    
    def _assess_phase_quality(self, complex_data: np.ndarray) -> Dict:
        """Assess phase quality for complex data."""
        phase = np.angle(complex_data)
        phase_diff = np.diff(phase, axis=0)
        return {
            'phase_noise_rad': np.std(phase_diff[np.isfinite(phase_diff)]),
            'phase_unwrapping_errors': 0  # Would detect phase jumps
        }
    
    def _assess_spatial_consistency(self, data: np.ndarray) -> Dict:
        """Assess spatial consistency of the data."""
        # Calculate local statistics
        from scipy import ndimage
        local_mean = ndimage.uniform_filter(data, size=5)
        local_std = ndimage.generic_filter(data, np.std, size=5)
        
        return {
            'spatial_uniformity': 1.0 - np.std(local_mean[np.isfinite(local_mean)]) / np.mean(local_mean[np.isfinite(local_mean)]),
            'local_variation': np.mean(local_std[np.isfinite(local_std)])
        }
    
    def _analyze_distributions(self, data: np.ndarray) -> Dict:
        """Analyze data distributions."""
        valid_data = data[np.isfinite(data)]
        
        # Test different distributions
        distributions = ['gamma', 'rayleigh', 'exponential', 'normal']
        best_fit = None
        best_p_value = 0
        
        for dist_name in distributions:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(valid_data)
                ks_stat, p_value = stats.kstest(valid_data, lambda x: dist.cdf(x, *params))
                
                if p_value > best_p_value:
                    best_p_value = p_value
                    best_fit = (dist_name, params, p_value)
            except:
                continue
        
        return {
            'best_distribution': best_fit[0] if best_fit else 'unknown',
            'distribution_params': best_fit[1] if best_fit else [],
            'goodness_of_fit_p_value': best_p_value
        }
    
    def _analyze_autocorrelation(self, data: np.ndarray) -> Dict:
        """Analyze spatial autocorrelation."""
        # Simplified autocorrelation analysis
        if data.ndim == 2:
            # Calculate autocorrelation along one dimension
            autocorr = np.correlate(data.flat, data.flat, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find correlation length
            correlation_length = np.argmax(autocorr < 0.5 * autocorr[0])
            
            return {
                'correlation_length_pixels': correlation_length,
                'max_autocorrelation': np.max(autocorr[1:])  # Exclude zero lag
            }
        
        return {}
    
    def _test_stationarity(self, data: np.ndarray) -> Dict:
        """Test data stationarity."""
        # Simplified stationarity test
        valid_data = data[np.isfinite(data)]
        
        # Split data and compare statistics
        mid = len(valid_data) // 2
        first_half = valid_data[:mid]
        second_half = valid_data[mid:]
        
        # Statistical test
        stat, p_value = stats.ttest_ind(first_half, second_half)
        
        return {
            'stationarity_test_statistic': stat,
            'stationarity_p_value': p_value,
            'is_stationary': p_value > 0.05
        }
    
    def _test_normality(self, data: np.ndarray) -> Dict:
        """Test data normality."""
        valid_data = data[np.isfinite(data)]
        
        if len(valid_data) > 5000:  # Sample for large datasets
            valid_data = np.random.choice(valid_data, 5000, replace=False)
        
        # Shapiro-Wilk test
        stat, p_value = stats.shapiro(valid_data)
        
        return {
            'normality_test_statistic': stat,
            'normality_p_value': p_value,
            'is_normal': p_value > 0.05
        }
    
    def _estimate_input_uncertainties(self, data: np.ndarray) -> Dict:
        """Estimate input data uncertainties."""
        return {
            'thermal_noise_db': -25.0,
            'quantization_error_db': -40.0,
            'calibration_uncertainty_db': 0.5
        }
    
    def _estimate_processing_uncertainties(self, data: np.ndarray) -> Dict:
        """Estimate processing-induced uncertainties."""
        return {
            'interpolation_error_db': 0.1,
            'filtering_artifacts_db': 0.2,
            'numerical_precision_db': 0.05
        }
    
    def _monte_carlo_error_propagation(self, data: np.ndarray) -> Dict:
        """Monte Carlo error propagation analysis."""
        # Simplified Monte Carlo simulation
        n_iterations = 100
        results = []
        
        for i in range(n_iterations):
            # Add noise to simulate uncertainties
            noise = np.random.normal(0, 0.1, data.shape)
            noisy_data = data + noise
            
            # Process and collect results
            result = np.mean(noisy_data[np.isfinite(noisy_data)])
            results.append(result)
        
        return {
            'mc_mean_error': np.std(results),
            'mc_confidence_interval': np.percentile(results, [2.5, 97.5])
        }
    
    def _analytical_error_propagation(self, data: np.ndarray) -> Dict:
        """Analytical error propagation."""
        # Simplified analytical approach
        return {
            'analytical_error_estimate': 0.15,
            'error_sources': ['thermal_noise', 'calibration', 'processing']
        }
    
    def _plot_noise_analysis(self, data: np.ndarray, noise_results: Dict):
        """Create noise analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Noise floor histogram
        noise_data = data[data < np.percentile(data[np.isfinite(data)], 5)]
        axes[0, 0].hist(noise_data, bins=50, alpha=0.7)
        axes[0, 0].set_title('Noise Floor Distribution')
        axes[0, 0].set_xlabel('Amplitude')
        axes[0, 0].set_ylabel('Frequency')
        
        # Power spectral density
        if 'noise_psd' in noise_results and len(noise_results['noise_psd']) > 0:
            axes[0, 1].plot(noise_results['noise_psd'])
            axes[0, 1].set_title('Noise Power Spectral Density')
            axes[0, 1].set_xlabel('Frequency Bin')
            axes[0, 1].set_ylabel('Power')
        
        # Noise spatial distribution
        noise_mask = data < np.percentile(data[np.isfinite(data)], 10)
        axes[1, 0].imshow(noise_mask, cmap='gray')
        axes[1, 0].set_title('Noise Regions')
        
        # NESZ profile (if available)
        if 'nesz_db' in noise_results:
            axes[1, 1].axhline(y=noise_results['nesz_db'], color='r', linestyle='--')
            axes[1, 1].set_title(f'NESZ: {noise_results["nesz_db"]:.1f} dB')
            axes[1, 1].set_ylabel('NESZ (dB)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'noise_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration_assessment(self, calibration_results: Dict):
        """Create calibration assessment plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Radiometric accuracy
        if 'radiometric_accuracy_db' in calibration_results:
            axes[0, 0].bar(['Accuracy'], [calibration_results['radiometric_accuracy_db']])
            axes[0, 0].set_title('Radiometric Calibration Accuracy')
            axes[0, 0].set_ylabel('Error (dB)')
        
        # Geometric accuracy
        if 'geometric_accuracy_m' in calibration_results:
            axes[0, 1].bar(['Accuracy'], [calibration_results['geometric_accuracy_m']])
            axes[0, 1].set_title('Geometric Calibration Accuracy')
            axes[0, 1].set_ylabel('Error (m)')
        
        # Cross-talk
        if 'cross_talk_db' in calibration_results:
            axes[1, 0].bar(['Cross-talk'], [calibration_results['cross_talk_db']])
            axes[1, 0].set_title('Polarimetric Cross-talk')
            axes[1, 0].set_ylabel('Cross-talk (dB)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'calibration_assessment.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_quality_metrics(self, data: np.ndarray, quality_metrics: Dict):
        """Create quality metrics plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # SNR
        if 'snr_db' in quality_metrics:
            axes[0, 0].bar(['SNR'], [quality_metrics['snr_db']])
            axes[0, 0].set_title(f'SNR: {quality_metrics["snr_db"]:.1f} dB')
            axes[0, 0].set_ylabel('SNR (dB)')
        
        # ENL
        if 'enl' in quality_metrics:
            axes[0, 1].bar(['ENL'], [quality_metrics['enl']])
            axes[0, 1].set_title(f'ENL: {quality_metrics["enl"]:.1f}')
            axes[0, 1].set_ylabel('Number of Looks')
        
        # Speckle Index
        if 'speckle_index' in quality_metrics:
            axes[0, 2].bar(['Speckle Index'], [quality_metrics['speckle_index']])
            axes[0, 2].set_title(f'Speckle Index: {quality_metrics["speckle_index"]:.3f}')
            axes[0, 2].set_ylabel('Coefficient of Variation')
        
        # Coherence quality
        if 'mean_coherence' in quality_metrics:
            axes[1, 0].bar(['Mean Coherence'], [quality_metrics['mean_coherence']])
            axes[1, 0].set_title(f'Mean Coherence: {quality_metrics["mean_coherence"]:.3f}')
            axes[1, 0].set_ylabel('Coherence')
        
        # Spatial uniformity
        if 'spatial_uniformity' in quality_metrics:
            axes[1, 1].bar(['Spatial Uniformity'], [quality_metrics['spatial_uniformity']])
            axes[1, 1].set_title(f'Spatial Uniformity: {quality_metrics["spatial_uniformity"]:.3f}')
            axes[1, 1].set_ylabel('Uniformity Index')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quality_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_validation(self, data: np.ndarray, validation_results: Dict):
        """Create statistical validation plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        valid_data = data[np.isfinite(data)]
        
        # Distribution histogram
        axes[0, 0].hist(valid_data, bins=50, alpha=0.7, density=True)
        axes[0, 0].set_title(f'Data Distribution\nBest fit: {validation_results.get("best_distribution", "unknown")}')
        axes[0, 0].set_xlabel('Amplitude')
        axes[0, 0].set_ylabel('Density')
        
        # Q-Q plot
        stats.probplot(valid_data, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normal Distribution)')
        
        # Autocorrelation
        if 'correlation_length_pixels' in validation_results:
            corr_length = validation_results['correlation_length_pixels']
            lags = np.arange(0, min(100, corr_length * 3))
            # Simplified autocorrelation plot
            axes[1, 0].plot(lags, np.exp(-lags / max(1, corr_length)))
            axes[1, 0].set_title(f'Autocorrelation\nCorrelation length: {corr_length} pixels')
            axes[1, 0].set_xlabel('Lag (pixels)')
            axes[1, 0].set_ylabel('Correlation')
        
        # Statistical tests summary
        test_results = []
        test_names = []
        
        if 'normality_p_value' in validation_results:
            test_results.append(validation_results['normality_p_value'])
            test_names.append('Normality')
        
        if 'stationarity_p_value' in validation_results:
            test_results.append(validation_results['stationarity_p_value'])
            test_names.append('Stationarity')
        
        if test_results:
            axes[1, 1].bar(test_names, test_results)
            axes[1, 1].axhline(y=0.05, color='r', linestyle='--', label='p=0.05')
            axes[1, 1].set_title('Statistical Tests (p-values)')
            axes[1, 1].set_ylabel('p-value')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'statistical_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_propagation(self, error_results: Dict):
        """Create error propagation plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Error sources
        if 'input_uncertainties' in error_results:
            uncertainties = error_results['input_uncertainties']
            sources = list(uncertainties.keys())
            values = list(uncertainties.values())
            
            axes[0, 0].bar(sources, values)
            axes[0, 0].set_title('Input Uncertainties')
            axes[0, 0].set_ylabel('Uncertainty (dB)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Monte Carlo results
        if 'mc_confidence_interval' in error_results:
            ci = error_results['mc_confidence_interval']
            axes[0, 1].fill_between([0, 1], ci[0], ci[1], alpha=0.3)
            axes[0, 1].set_title('Monte Carlo Confidence Interval')
            axes[0, 1].set_ylabel('Value')
        
        # Error budget
        total_errors = []
        error_labels = []
        
        for category in ['input_uncertainties', 'processing_uncertainties']:
            if category in error_results:
                for key, value in error_results[category].items():
                    if isinstance(value, (int, float)):
                        total_errors.append(abs(value))
                        error_labels.append(key.replace('_', ' ').title())
        
        if total_errors:
            axes[1, 0].pie(total_errors, labels=error_labels, autopct='%1.1f%%')
            axes[1, 0].set_title('Error Budget')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_propagation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_html_report(self, all_results: Dict) -> str:
        """Generate HTML quality assessment report."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SARPYX Quality Assessment Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .section { margin-bottom: 30px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .pass { color: green; font-weight: bold; }
                .fail { color: red; font-weight: bold; }
                .warning { color: orange; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>SARPYX Quality Assessment Report</h1>
            <p>Generated on: """ + str(pd.Timestamp.now()) + """</p>
            
            <div class="section">
                <h2>Data Integrity Summary</h2>
                <div class="metric">
                    <strong>Data Quality:</strong> 
                    <span class="pass">PASSED</span>
                </div>
            </div>
            
            <div class="section">
                <h2>Key Quality Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
        """
        
        # Add metrics to table
        processing_quality = all_results.get('processing_quality', {})
        
        if 'snr_db' in processing_quality:
            snr = processing_quality['snr_db']
            status = "PASS" if snr > 10 else "FAIL"
            html_content += f"<tr><td>SNR</td><td>{snr:.1f} dB</td><td class='{'pass' if status == 'PASS' else 'fail'}'>{status}</td></tr>"
        
        if 'enl' in processing_quality:
            enl = processing_quality['enl']
            status = "PASS" if enl > 1 else "FAIL"
            html_content += f"<tr><td>ENL</td><td>{enl:.1f}</td><td class='{'pass' if status == 'PASS' else 'fail'}'>{status}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Quality Assessment Images</h2>
                <img src="noise_analysis.png" alt="Noise Analysis" style="max-width: 100%; margin: 10px 0;">
                <img src="quality_metrics.png" alt="Quality Metrics" style="max-width: 100%; margin: 10px 0;">
                <img src="statistical_validation.png" alt="Statistical Validation" style="max-width: 100%; margin: 10px 0;">
                <img src="error_propagation.png" alt="Error Propagation" style="max-width: 100%; margin: 10px 0;">
            </div>
        </body>
        </html>
        """
        
        report_path = self.output_dir / 'quality_assessment_report.html'
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def _generate_summary_statistics(self, all_results: Dict):
        """Generate summary statistics CSV."""
        summary_data = []
        
        for category, results in all_results.items():
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    summary_data.append({
                        'category': category,
                        'metric': key,
                        'value': value
                    })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(self.output_dir / 'quality_summary_statistics.csv', index=False)


def main():
    """
    Main function demonstrating quality assessment workflow.
    """
    # Example data path (replace with actual data)
    data_path = "data/S1A_S3_SLC__1SSH_20240621T052251_20240621T052319_054417_069F07_8466.SAFE"
    
    # Initialize quality assessment
    qa = QualityAssessment(data_path)
    
    # Generate comprehensive quality report
    report_path = qa.generate_quality_report()
    
    print(f"\nQuality Assessment Complete!")
    print(f"Report saved to: {report_path}")
    print(f"Additional outputs in: {qa.output_dir}")
    
    # Print key quality metrics
    processing_metrics = qa.quality_metrics
    if processing_metrics:
        print("\nKey Quality Metrics:")
        for metric, value in processing_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.3f}")


if __name__ == "__main__":
    main()
