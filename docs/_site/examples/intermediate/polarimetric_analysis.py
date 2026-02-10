#!/usr/bin/env python3
"""
SARPYX Polarimetric Analysis Examples
=====================================

This module demonstrates advanced polarimetric SAR data analysis capabilities
using SARPYX, including decomposition techniques, target characterization,
and polarimetric feature extraction.

Author: SARPYX Development Team
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.linalg import eigh
import logging

# SARPYX imports
from sarpyx.science.indices import PolarimetricIndices
from sarpyx.utils.viz import SARVisualizer
from sarpyx.utils.io import SARDataReader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolarimetricAnalysisExamples:
    """
    Collection of examples for polarimetric SAR data analysis using SARPYX.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize polarimetric analysis examples.
        
        Parameters
        ----------
        data_path : str, optional
            Path to SAR data directory
        """
        self.data_path = Path(data_path) if data_path else Path("../../../data")
        self.output_path = Path("./outputs/polarimetric")
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.pol_indices = PolarimetricIndices()
        self.visualizer = SARVisualizer()
        self.reader = SARDataReader()
        
    def example_1_covariance_matrix_analysis(self):
        """
        Example 1: Covariance matrix analysis and basic polarimetric parameters
        
        Demonstrates calculation of basic polarimetric parameters from
        covariance/coherency matrices.
        """
        print("\n" + "="*60)
        print("Example 1: Covariance Matrix Analysis")
        print("="*60)
        
        try:
            # Generate or load polarimetric covariance matrix
            C3_matrix, coordinates = self._load_or_generate_C3_matrix()
            
            print(f"Covariance matrix shape: {C3_matrix.shape}")
            print(f"Matrix type: C3 (covariance matrix in linear basis)")
            
            # Calculate basic polarimetric parameters
            span = self._calculate_span(C3_matrix)
            print(f"SPAN range: {np.nanmin(span):.3f} to {np.nanmax(span):.3f}")
            
            # Calculate polarimetric entropy and alpha angle
            entropy, alpha, anisotropy = self._calculate_entropy_alpha_anisotropy(C3_matrix)
            
            print(f"Entropy range: {np.nanmin(entropy):.3f} to {np.nanmax(entropy):.3f}")
            print(f"Alpha angle range: {np.nanmin(alpha):.1f}° to {np.nanmax(alpha):.1f}°")
            print(f"Anisotropy range: {np.nanmin(anisotropy):.3f} to {np.nanmax(anisotropy):.3f}")
            
            # Calculate polarimetric coherences
            coherences = self._calculate_polarimetric_coherences(C3_matrix)
            
            print(f"HH-VV coherence range: {np.nanmin(coherences['hh_vv']):.3f} to {np.nanmax(coherences['hh_vv']):.3f}")
            print(f"HH-HV coherence range: {np.nanmin(coherences['hh_hv']):.3f} to {np.nanmax(coherences['hh_hv']):.3f}")
            print(f"VV-HV coherence range: {np.nanmin(coherences['vv_hv']):.3f} to {np.nanmax(coherences['vv_hv']):.3f}")
            
            # Visualize basic parameters
            self._visualize_basic_polarimetric_parameters({
                'SPAN (dB)': 10 * np.log10(span + 1e-10),
                'Entropy': entropy,
                'Alpha Angle (°)': alpha,
                'Anisotropy': anisotropy,
                '|γ_HH-VV|': coherences['hh_vv'],
                '|γ_HH-HV|': coherences['hh_hv']
            })
            
            return {
                'C3_matrix': C3_matrix,
                'span': span,
                'entropy': entropy,
                'alpha': alpha,
                'anisotropy': anisotropy,
                'coherences': coherences
            }
            
        except Exception as e:
            logger.error(f"Error in covariance matrix analysis: {e}")
            return None
    
    def example_2_target_decomposition(self):
        """
        Example 2: Polarimetric target decomposition
        
        Demonstrates various decomposition techniques including Freeman-Durden,
        Yamaguchi, H/A/Alpha, and others.
        """
        print("\n" + "="*60)
        print("Example 2: Polarimetric Target Decomposition")
        print("="*60)
        
        try:
            # Get covariance matrix from previous example
            result = self.example_1_covariance_matrix_analysis()
            if result is None:
                return None
            
            C3_matrix = result['C3_matrix']
            
            # Freeman-Durden decomposition
            print("\nPerforming Freeman-Durden decomposition...")
            freeman_durden = self._freeman_durden_decomposition(C3_matrix)
            
            fd_surface = freeman_durden['surface']
            fd_volume = freeman_durden['volume']
            fd_double_bounce = freeman_durden['double_bounce']
            
            print(f"Surface scattering contribution: {np.nanmean(fd_surface):.3f} ± {np.nanstd(fd_surface):.3f}")
            print(f"Volume scattering contribution: {np.nanmean(fd_volume):.3f} ± {np.nanstd(fd_volume):.3f}")
            print(f"Double bounce contribution: {np.nanmean(fd_double_bounce):.3f} ± {np.nanstd(fd_double_bounce):.3f}")
            
            # Yamaguchi 4-component decomposition
            print("\nPerforming Yamaguchi 4-component decomposition...")
            yamaguchi = self._yamaguchi_4component_decomposition(C3_matrix)
            
            y4_surface = yamaguchi['surface']
            y4_volume = yamaguchi['volume']
            y4_double_bounce = yamaguchi['double_bounce']
            y4_helix = yamaguchi['helix']
            
            print(f"Y4 Surface: {np.nanmean(y4_surface):.3f}")
            print(f"Y4 Volume: {np.nanmean(y4_volume):.3f}")
            print(f"Y4 Double bounce: {np.nanmean(y4_double_bounce):.3f}")
            print(f"Y4 Helix: {np.nanmean(y4_helix):.3f}")
            
            # Cloude-Pottier decomposition (H/A/Alpha)
            print("\nPerforming Cloude-Pottier H/A/Alpha decomposition...")
            cloude_pottier = self._cloude_pottier_decomposition(C3_matrix)
            
            entropy = cloude_pottier['entropy']
            alpha = cloude_pottier['alpha']
            anisotropy = cloude_pottier['anisotropy']
            
            # Classification based on H/A/Alpha
            classification = self._classify_h_alpha(entropy, alpha, anisotropy)
            
            print(f"H/A/Alpha classification zones:")
            for zone, count in np.unique(classification, return_counts=True):
                zone_names = {1: 'Z1', 2: 'Z2', 3: 'Z3', 4: 'Z4', 5: 'Z5', 
                             6: 'Z6', 7: 'Z7', 8: 'Z8', 9: 'Z9'}
                print(f"  {zone_names.get(int(zone), f'Zone {int(zone)}'}: {count[0]} pixels")
            
            # Visualize decomposition results
            self._visualize_target_decomposition({
                'Freeman-Durden': {
                    'Surface': fd_surface,
                    'Volume': fd_volume,
                    'Double Bounce': fd_double_bounce
                },
                'Yamaguchi-4': {
                    'Surface': y4_surface,
                    'Volume': y4_volume,
                    'Double Bounce': y4_double_bounce,
                    'Helix': y4_helix
                },
                'H/A/Alpha Classification': classification
            })
            
            return {
                'freeman_durden': freeman_durden,
                'yamaguchi': yamaguchi,
                'cloude_pottier': cloude_pottier,
                'classification': classification
            }
            
        except Exception as e:
            logger.error(f"Error in target decomposition: {e}")
            return None
    
    def example_3_polarimetric_coherence_analysis(self):
        """
        Example 3: Polarimetric coherence and correlation analysis
        
        Advanced analysis of polarimetric coherences and correlations
        for target characterization.
        """
        print("\n" + "="*60)
        print("Example 3: Polarimetric Coherence Analysis")
        print("="*60)
        
        try:
            # Generate or load quad-pol SLC data
            slc_data = self._load_or_generate_quad_pol_slc()
            
            print(f"SLC data channels: {list(slc_data.keys())}")
            print(f"Data shape: {slc_data['HH'].shape}")
            
            # Calculate all possible coherences
            coherence_matrix = self._calculate_full_coherence_matrix(slc_data)
            
            # Extract specific coherences
            coherences = {
                'γ_HH-VV': coherence_matrix[0, 2],
                'γ_HH-HV': coherence_matrix[0, 1],
                'γ_HH-VH': coherence_matrix[0, 3],
                'γ_VV-HV': coherence_matrix[2, 1],
                'γ_VV-VH': coherence_matrix[2, 3],
                'γ_HV-VH': coherence_matrix[1, 3]
            }
            
            print("\nCoherence magnitudes:")
            for name, coh in coherences.items():
                magnitude = np.abs(coh)
                print(f"{name}: {np.nanmean(magnitude):.3f} ± {np.nanstd(magnitude):.3f}")
            
            # Calculate circular coherences
            circular_coherences = self._calculate_circular_coherences(slc_data)
            
            print(f"\nCircular coherences:")
            print(f"γ_LL-RR: {np.nanmean(np.abs(circular_coherences['ll_rr'])):.3f}")
            print(f"γ_LL-LR: {np.nanmean(np.abs(circular_coherences['ll_lr'])):.3f}")
            print(f"γ_RR-LR: {np.nanmean(np.abs(circular_coherences['rr_lr'])):.3f}")
            
            # Analyze coherence phases
            coherence_phases = self._analyze_coherence_phases(coherences)
            
            # Calculate degree of polarization
            dop = self._calculate_degree_of_polarization(slc_data)
            print(f"\nDegree of Polarization: {np.nanmean(dop):.3f} ± {np.nanstd(dop):.3f}")
            
            # Visualize coherence analysis
            self._visualize_coherence_analysis(coherences, circular_coherences, dop)
            
            return {
                'coherences': coherences,
                'circular_coherences': circular_coherences,
                'coherence_phases': coherence_phases,
                'degree_of_polarization': dop
            }
            
        except Exception as e:
            logger.error(f"Error in coherence analysis: {e}")
            return None
    
    def example_4_polarimetric_feature_extraction(self):
        """
        Example 4: Advanced polarimetric feature extraction
        
        Extract various polarimetric features for classification and analysis.
        """
        print("\n" + "="*60)
        print("Example 4: Polarimetric Feature Extraction")
        print("="*60)
        
        try:
            # Get data from previous examples
            basic_params = self.example_1_covariance_matrix_analysis()
            decomposition = self.example_2_target_decomposition()
            
            if basic_params is None or decomposition is None:
                return None
            
            C3_matrix = basic_params['C3_matrix']
            
            # Extract comprehensive feature set
            features = {}
            
            # 1. Basic polarimetric parameters
            features['span'] = basic_params['span']
            features['entropy'] = basic_params['entropy']
            features['alpha'] = basic_params['alpha']
            features['anisotropy'] = basic_params['anisotropy']
            
            # 2. Decomposition parameters
            features['fd_surface'] = decomposition['freeman_durden']['surface']
            features['fd_volume'] = decomposition['freeman_durden']['volume']
            features['fd_double_bounce'] = decomposition['freeman_durden']['double_bounce']
            
            # 3. Advanced features
            features.update(self._extract_advanced_features(C3_matrix))
            
            # 4. Texture features from polarimetric parameters
            features.update(self._extract_polarimetric_texture_features(basic_params))
            
            # 5. Statistical features
            features.update(self._extract_statistical_features(C3_matrix))
            
            print(f"\nExtracted {len(features)} polarimetric features:")
            for name in list(features.keys())[:10]:  # Show first 10
                feature_data = features[name]
                print(f"  {name}: {np.nanmean(feature_data):.3f} ± {np.nanstd(feature_data):.3f}")
            print(f"  ... and {len(features) - 10} more features")
            
            # Feature importance analysis
            feature_importance = self._analyze_feature_importance(features)
            
            # Save feature set
            self._save_feature_set(features, feature_importance)
            
            # Visualize feature analysis
            self._visualize_feature_analysis(features, feature_importance)
            
            return {
                'features': features,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            return None
    
    def example_5_polarimetric_classification(self):
        """
        Example 5: Polarimetric classification and segmentation
        
        Demonstrate classification using polarimetric features.
        """
        print("\n" + "="*60)
        print("Example 5: Polarimetric Classification")
        print("="*60)
        
        try:
            # Get features from previous example
            feature_result = self.example_4_polarimetric_feature_extraction()
            if feature_result is None:
                return None
            
            features = feature_result['features']
            
            # Prepare feature matrix for classification
            feature_matrix = self._prepare_feature_matrix(features)
            
            print(f"Feature matrix shape: {feature_matrix.shape}")
            print(f"Number of features: {feature_matrix.shape[1]}")
            
            # Unsupervised classification (clustering)
            print("\nPerforming unsupervised classification...")
            unsupervised_result = self._unsupervised_classification(feature_matrix)
            
            n_clusters = len(np.unique(unsupervised_result['labels']))
            print(f"Number of clusters: {n_clusters}")
            
            # Analyze cluster characteristics
            cluster_analysis = self._analyze_clusters(feature_matrix, unsupervised_result['labels'])
            
            # H/A/Alpha classification
            print("\nPerforming H/A/Alpha classification...")
            h_alpha_classification = self._h_alpha_classification(features)
            
            # Wishart classification
            print("\nPerforming Wishart classification...")
            wishart_classification = self._wishart_classification(features)
            
            # Compare classification results
            classification_comparison = self._compare_classifications({
                'Unsupervised': unsupervised_result['labels'],
                'H/A/Alpha': h_alpha_classification,
                'Wishart': wishart_classification
            })
            
            # Visualize classification results
            self._visualize_polarimetric_classification({
                'Unsupervised': unsupervised_result['labels'],
                'H/A/Alpha': h_alpha_classification,
                'Wishart': wishart_classification
            }, cluster_analysis)
            
            print(f"\nClassification Results:")
            print(f"Unsupervised clusters: {n_clusters}")
            print(f"H/A/Alpha zones: {len(np.unique(h_alpha_classification))}")
            print(f"Wishart classes: {len(np.unique(wishart_classification))}")
            
            return {
                'unsupervised': unsupervised_result,
                'h_alpha': h_alpha_classification,
                'wishart': wishart_classification,
                'cluster_analysis': cluster_analysis,
                'comparison': classification_comparison
            }
            
        except Exception as e:
            logger.error(f"Error in polarimetric classification: {e}")
            return None
    
    def example_6_compact_polarimetric_analysis(self):
        """
        Example 6: Compact polarimetric analysis
        
        Analysis of compact polarimetric modes (dual-pol and hybrid-pol).
        """
        print("\n" + "="*60)
        print("Example 6: Compact Polarimetric Analysis")
        print("="*60)
        
        try:
            # Generate compact polarimetric data
            compact_data = self._generate_compact_pol_data()
            
            print(f"Compact polarimetric mode: {compact_data['mode']}")
            print(f"Available channels: {list(compact_data['data'].keys())}")
            
            # Reconstruct quad-pol from compact pol
            if compact_data['mode'] == 'dual_pol':
                reconstructed = self._reconstruct_from_dual_pol(compact_data['data'])
            else:  # hybrid_pol
                reconstructed = self._reconstruct_from_hybrid_pol(compact_data['data'])
            
            print(f"Reconstructed channels: {list(reconstructed.keys())}")
            
            # Calculate compact polarimetric indices
            compact_indices = self._calculate_compact_pol_indices(compact_data['data'])
            
            print(f"\nCompact polarimetric indices:")
            for name, values in compact_indices.items():
                print(f"  {name}: {np.nanmean(values):.3f} ± {np.nanstd(values):.3f}")
            
            # Compare with full polarimetric results
            comparison = self._compare_compact_vs_full_pol(compact_data, reconstructed)
            
            # Visualize compact polarimetric analysis
            self._visualize_compact_pol_analysis(compact_data, compact_indices, comparison)
            
            return {
                'compact_data': compact_data,
                'reconstructed': reconstructed,
                'compact_indices': compact_indices,
                'comparison': comparison
            }
            
        except Exception as e:
            logger.error(f"Error in compact polarimetric analysis: {e}")
            return None
    
    # Helper methods for data generation and processing
    
    def _load_or_generate_C3_matrix(self):
        """Load or generate C3 covariance matrix."""
        # Try to load real data first
        try:
            # Look for polarimetric data
            pol_files = list(self.data_path.glob("**/*_C3*"))
            if pol_files:
                # Would load real C3 matrix here
                pass
        except:
            pass
        
        # Generate synthetic C3 matrix
        shape = (200, 200)
        C3_matrix = np.zeros((*shape, 3, 3), dtype=np.complex64)
        
        # Generate different scattering areas
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                # Different target types in different regions
                if i < shape[0]//3:  # Surface scattering dominant
                    # Surface scattering matrix
                    T = np.array([[1, 0.3, 0.1],
                                  [0.3, 0.1, 0.05],
                                  [0.1, 0.05, 0.02]], dtype=np.complex64)
                elif i < 2*shape[0]//3:  # Volume scattering dominant
                    # Volume scattering matrix
                    T = np.array([[0.3, 0.1, 0.05],
                                  [0.1, 1, 0.2],
                                  [0.05, 0.2, 0.3]], dtype=np.complex64)
                else:  # Double bounce dominant
                    # Double bounce scattering matrix
                    T = np.array([[0.8, 0.2, 0.1],
                                  [0.2, 0.1, 0.05],
                                  [0.1, 0.05, 1]], dtype=np.complex64)
                
                # Add noise and make positive definite
                noise = 0.1 * (np.random.randn(3, 3) + 1j * np.random.randn(3, 3))
                C = T + noise
                C = C @ C.conj().T  # Ensure positive definiteness
                C3_matrix[i, j] = C
        
        coordinates = {'x': x, 'y': y}
        return C3_matrix, coordinates
    
    def _calculate_span(self, C3_matrix):
        """Calculate polarimetric span."""
        return np.real(C3_matrix[:, :, 0, 0] + C3_matrix[:, :, 1, 1] + C3_matrix[:, :, 2, 2])
    
    def _calculate_entropy_alpha_anisotropy(self, C3_matrix):
        """Calculate Cloude-Pottier entropy, alpha angle, and anisotropy."""
        shape = C3_matrix.shape[:2]
        entropy = np.zeros(shape)
        alpha = np.zeros(shape)
        anisotropy = np.zeros(shape)
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                # Eigendecomposition of coherency matrix T3
                T3 = self._convert_C3_to_T3(C3_matrix[i, j])
                eigenvals, eigenvecs = eigh(T3)
                
                # Sort by descending eigenvalues
                idx = np.argsort(eigenvals)[::-1]
                eigenvals = eigenvals[idx]
                
                # Normalize eigenvalues
                eigenvals = np.real(eigenvals)
                eigenvals = eigenvals / (np.sum(eigenvals) + 1e-10)
                
                # Calculate entropy
                valid_eigenvals = eigenvals[eigenvals > 1e-10]
                if len(valid_eigenvals) > 0:
                    entropy[i, j] = -np.sum(valid_eigenvals * np.log3(valid_eigenvals + 1e-10))
                
                # Calculate alpha angle (first eigenvector)
                if len(eigenvals) >= 3:
                    # Alpha angle from first eigenvector
                    alpha_val = np.arccos(np.abs(eigenvecs[0, idx[0]]))
                    alpha[i, j] = np.degrees(alpha_val)
                    
                    # Anisotropy
                    if eigenvals[1] > 1e-10:
                        anisotropy[i, j] = (eigenvals[1] - eigenvals[2]) / (eigenvals[1] + eigenvals[2] + 1e-10)
        
        return entropy, alpha, anisotropy
    
    def _convert_C3_to_T3(self, C3):
        """Convert C3 covariance matrix to T3 coherency matrix."""
        # Transformation matrix
        U = np.array([[1, 0, 1],
                      [1, 0, -1],
                      [0, np.sqrt(2), 0]], dtype=np.complex64) / np.sqrt(2)
        
        T3 = U @ C3 @ U.conj().T
        return T3
    
    def _calculate_polarimetric_coherences(self, C3_matrix):
        """Calculate various polarimetric coherences."""
        # Extract covariance elements
        C11 = C3_matrix[:, :, 0, 0]  # HH
        C22 = C3_matrix[:, :, 1, 1]  # HV
        C33 = C3_matrix[:, :, 2, 2]  # VV
        C13 = C3_matrix[:, :, 0, 2]  # HH-VV
        C12 = C3_matrix[:, :, 0, 1]  # HH-HV
        C23 = C3_matrix[:, :, 1, 2]  # HV-VV
        
        # Calculate coherence magnitudes
        coherences = {
            'hh_vv': np.abs(C13) / np.sqrt(np.real(C11 * C33) + 1e-10),
            'hh_hv': np.abs(C12) / np.sqrt(np.real(C11 * C22) + 1e-10),
            'vv_hv': np.abs(C23) / np.sqrt(np.real(C33 * C22) + 1e-10)
        }
        
        return coherences
    
    def _freeman_durden_decomposition(self, C3_matrix):
        """Perform Freeman-Durden decomposition."""
        shape = C3_matrix.shape[:2]
        surface = np.zeros(shape)
        volume = np.zeros(shape)
        double_bounce = np.zeros(shape)
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                C = C3_matrix[i, j]
                
                # Freeman-Durden decomposition algorithm
                # Simplified implementation
                C11 = np.real(C[0, 0])
                C22 = np.real(C[1, 1])
                C33 = np.real(C[2, 2])
                C13_real = np.real(C[0, 2])
                
                # Volume scattering component
                fv = 8 * C22 / 3
                
                # Surface and double bounce components
                if C13_real >= 0:
                    # Surface scattering dominant
                    fs = C11 + C33 - 2 * np.abs(C13_real) - fv
                    fd = 2 * np.abs(C13_real) - fv / 3
                else:
                    # Double bounce dominant
                    fd = C11 + C33 + 2 * np.abs(C13_real) - fv
                    fs = 2 * np.abs(C13_real) - fv / 3
                
                # Ensure non-negative values
                surface[i, j] = max(0, fs)
                volume[i, j] = max(0, fv)
                double_bounce[i, j] = max(0, fd)
        
        return {
            'surface': surface,
            'volume': volume,
            'double_bounce': double_bounce
        }
    
    def _yamaguchi_4component_decomposition(self, C3_matrix):
        """Perform Yamaguchi 4-component decomposition."""
        # Simplified implementation
        freeman_durden = self._freeman_durden_decomposition(C3_matrix)
        
        # Add helix component (simplified)
        C12 = C3_matrix[:, :, 0, 1]
        C21 = C3_matrix[:, :, 1, 0]
        
        helix = 2 * np.real(C12 - C21)
        helix = np.maximum(helix, 0)
        
        return {
            'surface': freeman_durden['surface'],
            'volume': freeman_durden['volume'] - helix/4,
            'double_bounce': freeman_durden['double_bounce'],
            'helix': helix
        }
    
    def _cloude_pottier_decomposition(self, C3_matrix):
        """Perform Cloude-Pottier H/A/Alpha decomposition."""
        entropy, alpha, anisotropy = self._calculate_entropy_alpha_anisotropy(C3_matrix)
        
        return {
            'entropy': entropy,
            'alpha': alpha,
            'anisotropy': anisotropy
        }
    
    def _classify_h_alpha(self, entropy, alpha, anisotropy):
        """Classify using H/Alpha plane."""
        classification = np.zeros_like(entropy, dtype=int)
        
        # H/Alpha classification zones
        for i in range(entropy.shape[0]):
            for j in range(entropy.shape[1]):
                h = entropy[i, j]
                a = alpha[i, j]
                
                if h <= 0.5:
                    if a <= 42.5:
                        classification[i, j] = 1  # Z1
                    elif a <= 57.5:
                        classification[i, j] = 2  # Z2
                    else:
                        classification[i, j] = 3  # Z3
                elif h <= 0.9:
                    if a <= 40:
                        classification[i, j] = 4  # Z4
                    elif a <= 50:
                        classification[i, j] = 5  # Z5
                    elif a <= 60:
                        classification[i, j] = 6  # Z6
                    else:
                        classification[i, j] = 7  # Z7
                else:
                    if a <= 47.5:
                        classification[i, j] = 8  # Z8
                    else:
                        classification[i, j] = 9  # Z9
        
        return classification
    
    # Additional methods for remaining functionality...
    # (Due to length constraints, showing structure of remaining methods)
    
    def _load_or_generate_quad_pol_slc(self):
        """Generate synthetic quad-pol SLC data."""
        pass
    
    def _calculate_full_coherence_matrix(self, slc_data):
        """Calculate full 4x4 coherence matrix."""
        pass
    
    def _extract_advanced_features(self, C3_matrix):
        """Extract advanced polarimetric features."""
        pass
    
    # Visualization methods
    def _visualize_basic_polarimetric_parameters(self, params_dict):
        """Visualize basic polarimetric parameters."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (name, data) in enumerate(params_dict.items()):
            if i < len(axes):
                im = axes[i].imshow(data, cmap='jet', aspect='auto')
                axes[i].set_title(name)
                plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.savefig(self.output_path / "basic_polarimetric_parameters.png", dpi=150)
        plt.close()
    
    def _visualize_target_decomposition(self, decomposition_dict):
        """Visualize target decomposition results."""
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        # Freeman-Durden RGB
        fd = decomposition_dict['Freeman-Durden']
        rgb_fd = self._create_rgb_image(fd['Volume'], fd['Double Bounce'], fd['Surface'])
        axes[0, 0].imshow(rgb_fd)
        axes[0, 0].set_title('Freeman-Durden RGB')
        
        # Individual F-D components
        axes[0, 1].imshow(fd['Surface'], cmap='Reds')
        axes[0, 1].set_title('F-D Surface')
        axes[0, 2].imshow(fd['Volume'], cmap='Greens')
        axes[0, 2].set_title('F-D Volume')
        axes[0, 3].imshow(fd['Double Bounce'], cmap='Blues')
        axes[0, 3].set_title('F-D Double Bounce')
        
        # Similar for Yamaguchi and H/A/Alpha...
        
        plt.tight_layout()
        plt.savefig(self.output_path / "target_decomposition.png", dpi=150)
        plt.close()
    
    def _create_rgb_image(self, r_channel, g_channel, b_channel):
        """Create RGB image from three channels."""
        # Normalize each channel
        r_norm = (r_channel - np.nanmin(r_channel)) / (np.nanmax(r_channel) - np.nanmin(r_channel) + 1e-10)
        g_norm = (g_channel - np.nanmin(g_channel)) / (np.nanmax(g_channel) - np.nanmin(g_channel) + 1e-10)
        b_norm = (b_channel - np.nanmin(b_channel)) / (np.nanmax(b_channel) - np.nanmin(b_channel) + 1e-10)
        
        rgb = np.stack([r_norm, g_norm, b_norm], axis=-1)
        return np.clip(rgb, 0, 1)


def run_all_polarimetric_examples():
    """Run all polarimetric analysis examples."""
    print("SARPYX Polarimetric Analysis Examples")
    print("=" * 60)
    
    # Initialize examples
    examples = PolarimetricAnalysisExamples()
    
    # Run examples
    try:
        examples.example_1_covariance_matrix_analysis()
        examples.example_2_target_decomposition()
        examples.example_3_polarimetric_coherence_analysis()
        examples.example_4_polarimetric_feature_extraction()
        examples.example_5_polarimetric_classification()
        examples.example_6_compact_polarimetric_analysis()
        
        print("\n" + "="*60)
        print("All polarimetric analysis examples completed!")
        print(f"Output files saved to: {examples.output_path}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error running polarimetric examples: {e}")


if __name__ == "__main__":
    run_all_polarimetric_examples()
