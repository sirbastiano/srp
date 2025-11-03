#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAR Dataset Splitter with Geographic and Temporal Balance

This script creates balanced training, validation, and test splits from SAR products
with the following criteria:
- Temporal split: 2023 -> train, 2024 -> validation, 2025 -> test
- Geographic clustering: Equal representation from different global regions
- Scene type balance: Equal representation of land, sea, and coast scenes
- Non-overlapping locations: Ensures geographic separation between splits

Requirements:
- pandas, numpy, scikit-learn
- geopy (for land/sea classification)
- folium (for visualization)

Input: sar_products_locations.csv (from world map notebook)
Output: Balanced train/validation/test zarr file lists
"""

import pandas as pd
import numpy as np
import json
import ast
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Try to import land/sea classification libraries
try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    HAS_GEOPY = True
    print("✓ Geopy available for land/sea classification")
except ImportError:
    HAS_GEOPY = False
    print("! Geopy not available, using coordinate-based classification")

try:
    import folium
    HAS_FOLIUM = True
    print("✓ Folium available for map visualization")
except ImportError:
    HAS_FOLIUM = False
    print("! Folium not available")


class LandSeaClassifier:
    """Classify coordinates as land, sea, or coast using multiple methods."""
    
    def __init__(self):
        self.geolocator = None
        if HAS_GEOPY:
            self.geolocator = Nominatim(user_agent="sar_dataset_splitter")
            self.geocode = RateLimiter(self.geolocator.reverse, min_delay_seconds=0.1)
    
    def classify_coordinate_simple(self, lat: float, lon: float) -> str:
        """
        Simple coordinate-based classification using known geographic patterns.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            
        Returns:
            str: 'land', 'sea', or 'coast'
        """
        # Major ocean regions (simplified)
        ocean_regions = [
            # Pacific Ocean
            {"lat_range": (-60, 60), "lon_range": (-180, -90)},
            {"lat_range": (-60, 60), "lon_range": (90, 180)},
            # Atlantic Ocean  
            {"lat_range": (-60, 70), "lon_range": (-80, 20)},
            # Indian Ocean
            {"lat_range": (-60, 30), "lon_range": (20, 110)},
            # Arctic Ocean
            {"lat_range": (70, 90), "lon_range": (-180, 180)},
            # Southern Ocean
            {"lat_range": (-90, -60), "lon_range": (-180, 180)}
        ]
        
        # Check if in major ocean regions
        for region in ocean_regions:
            lat_min, lat_max = region["lat_range"]
            lon_min, lon_max = region["lon_range"]
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                # Check if near coastlines (rough approximation)
                if self._is_near_coast_simple(lat, lon):
                    return "coast"
                return "sea"
        
        # Default to land if not in ocean regions
        return "land"
    
    def _is_near_coast_simple(self, lat: float, lon: float) -> bool:
        """Simple coastline detection using continental boundaries."""
        # Major continental coastline proximity (very simplified)
        coastline_regions = [
            # North America West Coast
            {"lat_range": (30, 60), "lon_range": (-130, -115)},
            # North America East Coast
            {"lat_range": (25, 50), "lon_range": (-85, -65)},
            # Europe West Coast
            {"lat_range": (35, 70), "lon_range": (-15, 5)},
            # Mediterranean
            {"lat_range": (30, 45), "lon_range": (-5, 40)},
            # Asia East Coast
            {"lat_range": (20, 50), "lon_range": (110, 140)},
            # Australia Coast
            {"lat_range": (-40, -15), "lon_range": (110, 155)},
        ]
        
        for region in coastline_regions:
            lat_min, lat_max = region["lat_range"]
            lon_min, lon_max = region["lon_range"]
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                return True
        return False
    
    def classify_coordinate_geopy(self, lat: float, lon: float) -> str:
        """
        Use Geopy reverse geocoding for land/sea classification.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            
        Returns:
            str: 'land', 'sea', or 'coast'
        """
        if not self.geolocator:
            return self.classify_coordinate_simple(lat, lon)
        
        try:
            location = self.geocode((lat, lon), timeout=10)
            if location:
                address = location.address.lower()
                # Check for water bodies
                water_indicators = ['ocean', 'sea', 'bay', 'gulf', 'strait', 'channel']
                if any(indicator in address for indicator in water_indicators):
                    return "sea"
                # Check for coastal areas
                coast_indicators = ['coast', 'coastal', 'shore', 'harbor', 'port']
                if any(indicator in address for indicator in coast_indicators):
                    return "coast"
                return "land"
            else:
                # No address found, likely over water
                return "sea"
        except Exception:
            # Fallback to simple classification
            return self.classify_coordinate_simple(lat, lon)
    
    def classify_coordinate(self, lat: float, lon: float, use_geopy: bool = False) -> str:
        """
        Classify a coordinate as land, sea, or coast.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude  
            use_geopy (bool): Whether to use Geopy API (slower but more accurate)
            
        Returns:
            str: 'land', 'sea', or 'coast'
        """
        if use_geopy and HAS_GEOPY:
            return self.classify_coordinate_geopy(lat, lon)
        else:
            return self.classify_coordinate_simple(lat, lon)


def parse_polygon_coordinates(coordinates) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract centroid coordinates from polygon coordinates.
    
    Args:
        coordinates: Can be one of:
            - GeoJSON dict: {'type': 'Polygon', 'coordinates': [[[lon, lat], ...]]}
            - Nested list: [[[lon, lat], [lon, lat], ...]]
            - Simple list: [[lon, lat], [lon, lat], ...]
            - String representation of any of the above
        
    Returns:
        Tuple[float, float]: (latitude, longitude) of centroid
    """
    try:
        # Handle string representation
        if isinstance(coordinates, str):
            coordinates = ast.literal_eval(coordinates)
        
        # Handle None or empty
        if coordinates is None:
            return None, None
        
        # Handle GeoJSON format: {'type': 'Polygon', 'coordinates': [...]}
        if isinstance(coordinates, dict):
            if 'coordinates' in coordinates:
                coordinates = coordinates['coordinates']
            else:
                return None, None
        
        # Now coordinates should be a list
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) == 0:
            return None, None
        
        # Extract lons and lats from the nested structure
        # GeoJSON Polygon format: [[[lon1, lat1], [lon2, lat2], ...]]
        # We need to flatten one level if it's a polygon (list of rings)
        if isinstance(coordinates[0], (list, tuple)) and isinstance(coordinates[0][0], (list, tuple)):
            # This is [[ring1], [ring2], ...] - take first ring (outer boundary)
            coordinates = coordinates[0]
        
        lons = []
        lats = []
        
        for coord in coordinates:
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                # coord is [lon, lat]
                lons.append(float(coord[0]))
                lats.append(float(coord[1]))
        
        if len(lons) > 0 and len(lats) > 0:
            # Calculate centroid
            centroid_lat = sum(lats) / len(lats)
            centroid_lon = sum(lons) / len(lons)
            return centroid_lat, centroid_lon
        
    except Exception as e:
        print(f"Error parsing coordinates: {e}, input was: {coordinates}")
        return None, None
    
    return None, None


def extract_year_from_filename(filename: str) -> Optional[int]:
    """
    Extract year from SAR product filename.
    
    Args:
        filename (str): SAR product filename
        
    Returns:
        Optional[int]: Year or None if not found
    """
    import re
    
    # Pattern for date extraction (YYYYMMDD format)
    date_pattern = r'(\d{4})(\d{2})(\d{2})'
    match = re.search(date_pattern, filename)
    
    if match:
        year = int(match.group(1))
        return year
    
    return None


def calculate_geographic_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate geodesic distance between two points in kilometers.
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
        
    Returns:
        float: Distance in kilometers
    """
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers


def create_geographic_clusters(df: pd.DataFrame, n_clusters: int = 10) -> pd.DataFrame:
    """
    Create geographic clusters from the dataset using K-means.
    
    Args:
        df (pd.DataFrame): Dataset with lat/lon columns
        n_clusters (int): Number of clusters to create
        
    Returns:
        pd.DataFrame: Dataset with cluster assignments
    """
    print(f"Creating {n_clusters} geographic clusters...")
    
    # Prepare coordinates for clustering
    coords = df[['latitude', 'longitude']].values
    
    # Standardize coordinates for better clustering
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['geo_cluster'] = kmeans.fit_predict(coords_scaled)
    
    # Print cluster statistics
    print("Geographic cluster distribution:")
    cluster_counts = df['geo_cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        cluster_data = df[df['geo_cluster'] == cluster_id]
        lat_center = cluster_data['latitude'].mean()
        lon_center = cluster_data['longitude'].mean()
        print(f"  Cluster {cluster_id}: {count} products (center: {lat_center:.2f}°, {lon_center:.2f}°)")
    
    return df


def ensure_geographic_separation(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                                test_df: pd.DataFrame, min_distance_km: float = 100.0) -> bool:
    """
    Check if train/validation/test sets have sufficient geographic separation.
    
    Args:
        train_df, val_df, test_df: Dataset splits
        min_distance_km: Minimum required distance between splits
        
    Returns:
        bool: True if adequately separated
    """
    print(f"Checking geographic separation (min distance: {min_distance_km} km)...")
    
    # Calculate centroids for each split
    train_centroid = (train_df['latitude'].mean(), train_df['longitude'].mean())
    val_centroid = (val_df['latitude'].mean(), val_df['longitude'].mean())
    test_centroid = (test_df['latitude'].mean(), test_df['longitude'].mean())
    
    # Calculate distances between centroids
    train_val_dist = calculate_geographic_distance(*train_centroid, *val_centroid)
    train_test_dist = calculate_geographic_distance(*train_centroid, *test_centroid)
    val_test_dist = calculate_geographic_distance(*val_centroid, *test_centroid)
    
    print(f"  Train-Validation distance: {train_val_dist:.1f} km")
    print(f"  Train-Test distance: {train_test_dist:.1f} km")
    print(f"  Validation-Test distance: {val_test_dist:.1f} km")
    
    min_dist = min(train_val_dist, train_test_dist, val_test_dist)
    is_separated = min_dist >= min_distance_km
    
    if is_separated:
        print(f"✓ Geographic separation adequate (min: {min_dist:.1f} km)")
    else:
        print(f"⚠ Geographic separation insufficient (min: {min_dist:.1f} km)")
    
    return is_separated


def balance_scene_types(df: pd.DataFrame, target_samples_per_type: int) -> pd.DataFrame:
    """
    Balance the dataset to have equal representation of land, sea, and coast scenes.
    
    Args:
        df (pd.DataFrame): Dataset with scene_type column
        target_samples_per_type (int): Target number of samples per scene type
        
    Returns:
        pd.DataFrame: Balanced dataset
    """
    print(f"Balancing scene types (target: {target_samples_per_type} per type)...")
    
    scene_type_counts = df['scene_type'].value_counts()
    print("Original scene type distribution:")
    for scene_type, count in scene_type_counts.items():
        print(f"  {scene_type}: {count} products")
    
    balanced_dfs = []
    
    for scene_type in ['land', 'sea', 'coast']:
        scene_data = df[df['scene_type'] == scene_type]
        
        if len(scene_data) >= target_samples_per_type:
            # Randomly sample if we have enough
            sampled = scene_data.sample(n=target_samples_per_type, random_state=42)
        else:
            # Use all available samples if we don't have enough
            sampled = scene_data
            print(f"  ⚠ Only {len(sampled)} {scene_type} samples available (target: {target_samples_per_type})")
        
        balanced_dfs.append(sampled)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    print("Balanced scene type distribution:")
    balanced_counts = balanced_df['scene_type'].value_counts()
    for scene_type, count in balanced_counts.items():
        print(f"  {scene_type}: {count} products")
    
    return balanced_df


def create_balanced_splits(csv_file: str, output_dir: str,
                          n_clusters: int = 20, sampling_fraction: float = 0.9,
                          use_geopy: bool = False, force_recreate: bool = False,
                          train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
    """
    Create balanced train/validation/test splits from SAR products CSV.
    
    NEW STRATEGY:
    - Geographic-based splitting (train/val/test from different geographic regions)
    - Preserves polarization, year, and scene type distributions from original dataset
    - Maximizes sample retention (uses all available samples)
    - Ensures wide geographic coverage in training set
    
    Args:
        csv_file (str): Path to CSV file with SAR product locations
        output_dir (str): Directory to save split lists
        n_clusters (int): Number of geographic clusters for splitting
        sampling_fraction (float): Reserved for compatibility (not used in new strategy)
        use_geopy (bool): Whether to use Geopy for land/sea classification
        force_recreate (bool): If True, recreate splits even if they already exist
        train_ratio (float): Fraction of data for training (default 0.7 = 70%)
        val_ratio (float): Fraction of data for validation (default 0.15 = 15%, test gets remainder)
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with train/validation/test DataFrames
    """
    print("="*80)
    print("SAR DATASET BALANCED SPLITTING")
    print("="*80)
    
    output_path = Path(output_dir)
    
    # Check if split files already exist
    if not force_recreate:
        train_csv = output_path / "train_products.csv"
        val_csv = output_path / "validation_products.csv"
        test_csv = output_path / "test_products.csv"
        
        if train_csv.exists() and val_csv.exists() and test_csv.exists():
            print(f"\n✓ Split files already exist in {output_dir}")
            print("  Loading existing splits...")
            # Try loading with semicolon delimiter first (new format)
            try:
                splits = {
                    'train': pd.read_csv(train_csv, sep=';'),
                    'validation': pd.read_csv(val_csv, sep=';'),
                    'test': pd.read_csv(test_csv, sep=';')
                }
            except:
                # Fall back to comma delimiter (old format)
                print("  (Loading with comma delimiter - old format)")
                splits = {
                    'train': pd.read_csv(train_csv),
                    'validation': pd.read_csv(val_csv),
                    'test': pd.read_csv(test_csv)
                }
            print(f"  Train: {len(splits['train'])} products")
            print(f"  Validation: {len(splits['validation'])} products")
            print(f"  Test: {len(splits['test'])} products")
            print("\n  (Use force_recreate=True to regenerate splits)")
            return splits
    
    # Load and prepare data
    print(f"\n📁 Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    
    print(f"Total products loaded: {len(df)}")
    
    # Extract coordinates from polygon coordinates column
    print("\n🌍 Extracting centroid coordinates from polygon data...")
    
    # Check which column name is used (coordinates or location)
    coord_column = 'coordinates' if 'coordinates' in df.columns else 'location'
    if coord_column not in df.columns:
        raise ValueError(f"Neither 'coordinates' nor 'location' column found in CSV. Available columns: {df.columns.tolist()}")
    
    coords = df[coord_column].apply(parse_polygon_coordinates)
    df['latitude'] = coords.apply(lambda x: x[0] if x[0] is not None else np.nan)
    df['longitude'] = coords.apply(lambda x: x[1] if x[1] is not None else np.nan)
    
    # Remove products without valid coordinates
    df = df.dropna(subset=['latitude', 'longitude'])
    print(f"Products with valid coordinates: {len(df)}")
    
    # Extract years from filenames
    print("\n📅 Extracting years from filenames...")
    df['year'] = df['filename'].apply(extract_year_from_filename)
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    
    year_counts = df['year'].value_counts().sort_index()
    print("Year distribution:")
    for year, count in year_counts.items():
        print(f"  {year}: {count} products")
    
    # STEP 1: Classify scene types FIRST (before clustering)
    # This ensures all scene types are properly represented
    print(f"\n🏞️ STEP 1: Classifying scene types (using {'Geopy API' if use_geopy else 'coordinate-based'})...")
    classifier = LandSeaClassifier()
    
    df['scene_type'] = df.apply(
        lambda row: classifier.classify_coordinate(row['latitude'], row['longitude'], use_geopy),
        axis=1
    )
    
    scene_counts = df['scene_type'].value_counts()
    print("Scene type distribution BEFORE clustering:")
    for scene_type, count in scene_counts.items():
        print(f"  {scene_type}: {count} products")
    
    # STEP 2: Create geographic clusters WITHIN each scene type
    # This ensures balanced representation across scene types
    print(f"\n🗺️ STEP 2: Creating geographic clusters WITHIN each scene type...")
    print(f"Target: {n_clusters} clusters per scene type")
    
    # Cluster each scene type separately
    clustered_dfs = []
    for scene_type in ['land', 'sea', 'coast']:
        scene_df = df[df['scene_type'] == scene_type].copy()
        if len(scene_df) == 0:
            print(f"  ⚠️  No {scene_type} samples found, skipping...")
            continue
        
        # Adjust number of clusters based on available samples
        # Use fewer clusters if we don't have enough samples
        scene_n_clusters = min(n_clusters, max(1, len(scene_df) // 5))  # At least 5 samples per cluster
        
        print(f"\n  {scene_type.upper()} scene clustering:")
        print(f"    Samples: {len(scene_df)}")
        print(f"    Clusters: {scene_n_clusters}")
        
        if scene_n_clusters > 0:
            # Perform clustering for this scene type
            coords = scene_df[['latitude', 'longitude']].values
            scaler = StandardScaler()
            coords_scaled = scaler.fit_transform(coords)
            
            kmeans = KMeans(n_clusters=scene_n_clusters, random_state=42, n_init=10)
            # Use scene-specific cluster IDs: scene_type + cluster_id
            scene_df['geo_cluster'] = [f"{scene_type}_{i}" for i in kmeans.fit_predict(coords_scaled)]
            
            cluster_counts = scene_df['geo_cluster'].value_counts().sort_index()
            for cluster_id, count in cluster_counts.items():
                cluster_data = scene_df[scene_df['geo_cluster'] == cluster_id]
                lat_center = cluster_data['latitude'].mean()
                lon_center = cluster_data['longitude'].mean()
                print(f"      {cluster_id}: {count} products (center: {lat_center:.2f}°, {lon_center:.2f}°)")
        
        clustered_dfs.append(scene_df)
    
    # Combine all clustered scene types
    if clustered_dfs:
        df = pd.concat(clustered_dfs, ignore_index=True)
    else:
        raise ValueError("No valid scene types found for clustering")
    
    print(f"\n✅ Clustering complete: {len(df)} products with scene-aware geographic clusters")
    
    # NEW STRATEGY: Geographic-based splitting with distribution preservation
    print(f"\n📊 STEP 3: Splitting by GEOGRAPHY while preserving distributions...")
    print("="*80)
    
    # Calculate split sizes
    test_ratio = 1.0 - train_ratio - val_ratio
    total_samples = len(df)
    target_train = int(total_samples * train_ratio)
    target_val = int(total_samples * val_ratio)
    target_test = total_samples - target_train - target_val
    
    print(f"Total samples: {total_samples}")
    print(f"Target split sizes:")
    print(f"  Train: {target_train} ({train_ratio*100:.1f}%)")
    print(f"  Validation: {target_val} ({val_ratio*100:.1f}%)")
    print(f"  Test: {target_test} ({test_ratio*100:.1f}%)")
    
    # Extract polarization from filename
    print(f"\n🔍 Extracting polarization types...")
    df['polarization'] = df['filename'].str.extract(r'-s-([hv]{2})-')[0]
    
    # Show original distributions
    print(f"\n📈 ORIGINAL DATASET DISTRIBUTIONS:")
    print(f"\nPolarization:")
    pol_dist = df['polarization'].value_counts()
    for pol, count in pol_dist.items():
        print(f"  {pol}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\nYear:")
    year_dist = df['year'].value_counts().sort_index()
    for year, count in year_dist.items():
        print(f"  {year}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\nScene Type:")
    scene_dist = df['scene_type'].value_counts()
    for scene, count in scene_dist.items():
        print(f"  {scene}: {count} ({count/len(df)*100:.1f}%)")
    
    # NEW STRATEGY: Assign clusters to splits ensuring EQUAL representation of 
    # scene types AND polarization types across ALL splits
    print(f"\n🗺️ Assigning geographic clusters to train/val/test splits...")
    print(f"Strategy: Balanced stratified assignment for scene types + polarization types")
    
    # Get all unique clusters with their characteristics
    all_clusters = df['geo_cluster'].unique()
    
    # Create detailed description of each cluster with actual composition
    cluster_info = []
    for cluster_id in all_clusters:
        cluster_data = df[df['geo_cluster'] == cluster_id]
        
        # Count samples by scene type
        scene_counts = cluster_data['scene_type'].value_counts().to_dict()
        
        # Count samples by polarization
        pol_counts = cluster_data['polarization'].value_counts().to_dict()
        
        cluster_info.append({
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'scene_type': cluster_data['scene_type'].mode()[0],  # Dominant scene type
            'dominant_pol': cluster_data['polarization'].mode()[0],  # Dominant polarization
            'lat_center': cluster_data['latitude'].mean(),
            'lon_center': cluster_data['longitude'].mean(),
            # Store actual counts for precise balancing
            'land_count': scene_counts.get('land', 0),
            'sea_count': scene_counts.get('sea', 0),
            'coast_count': scene_counts.get('coast', 0),
            'pol_counts': pol_counts  # Dict of polarization counts
        })
    
    cluster_df = pd.DataFrame(cluster_info)
    
    print(f"\nCluster statistics:")
    print(f"  Total clusters: {len(cluster_df)}")
    print(f"  Size range: {cluster_df['size'].min()}-{cluster_df['size'].max()} samples per cluster")
    
    # Initialize split assignments
    train_clusters = []
    val_clusters = []
    test_clusters = []
    
    # Track counts by scene type and polarization for each split
    train_counts = {'total': 0, 'land': 0, 'sea': 0, 'coast': 0, 'polarizations': {}}
    val_counts = {'total': 0, 'land': 0, 'sea': 0, 'coast': 0, 'polarizations': {}}
    test_counts = {'total': 0, 'land': 0, 'sea': 0, 'coast': 0, 'polarizations': {}}
    
    # Get target counts for each scene type and polarization in each split
    total_samples = len(df)
    scene_types = ['land', 'sea', 'coast']
    polarizations = df['polarization'].unique().tolist()
    
    targets = {
        'train': {
            'total': int(total_samples * train_ratio),
            'land': int(df[df['scene_type'] == 'land'].shape[0] * train_ratio),
            'sea': int(df[df['scene_type'] == 'sea'].shape[0] * train_ratio),
            'coast': int(df[df['scene_type'] == 'coast'].shape[0] * train_ratio),
            'polarizations': {pol: int(df[df['polarization'] == pol].shape[0] * train_ratio) for pol in polarizations}
        },
        'val': {
            'total': int(total_samples * val_ratio),
            'land': int(df[df['scene_type'] == 'land'].shape[0] * val_ratio),
            'sea': int(df[df['scene_type'] == 'sea'].shape[0] * val_ratio),
            'coast': int(df[df['scene_type'] == 'coast'].shape[0] * val_ratio),
            'polarizations': {pol: int(df[df['polarization'] == pol].shape[0] * val_ratio) for pol in polarizations}
        },
        'test': {
            'total': int(total_samples * test_ratio),
            'land': int(df[df['scene_type'] == 'land'].shape[0] * test_ratio),
            'sea': int(df[df['scene_type'] == 'sea'].shape[0] * test_ratio),
            'coast': int(df[df['scene_type'] == 'coast'].shape[0] * test_ratio),
            'polarizations': {pol: int(df[df['polarization'] == pol].shape[0] * test_ratio) for pol in polarizations}
        }
    }
    
    print(f"\n📊 Target distributions for balanced splits:")
    for split_name, target in targets.items():
        print(f"\n  {split_name.upper()}:")
        print(f"    Total: {target['total']} samples")
        print(f"    Scene types: land={target['land']}, sea={target['sea']}, coast={target['coast']}")
        print(f"    Polarizations: {', '.join([f'{pol}={cnt}' for pol, cnt in target['polarizations'].items()])}")
    
    # Sort clusters by composite score: balance scene rarity with cluster diversity
    # Coast and land are rare, so they get higher priority
    # But also consider polarization diversity within the cluster
    cluster_df['scene_rarity'] = cluster_df.apply(
        lambda row: (row['coast_count'] * 10 + row['land_count'] * 5), 
        axis=1
    )
    cluster_df['pol_diversity'] = cluster_df.apply(
        lambda row: len([p for p in row['pol_counts'].values() if p > 0]),
        axis=1
    )
    cluster_df['composite_priority'] = (
        cluster_df['scene_rarity'] * 100 +  # Rare scenes get high priority
        cluster_df['pol_diversity'] * 20 +   # Diverse polarizations are valuable
        cluster_df['size'] / 10               # Size as tiebreaker
    )
    cluster_df = cluster_df.sort_values('composite_priority', ascending=False).reset_index(drop=True)
    cluster_df = cluster_df.drop(columns=['scene_rarity', 'pol_diversity', 'composite_priority'])
    
    # Assign each cluster to the split that needs it most
    print(f"\n🎯 Assigning clusters with multi-dimensional balancing...")
    print(f"    (Balancing scene types AND polarizations)")
    
    for idx, row in cluster_df.iterrows():
        cluster_id = row['cluster_id']
        cluster_size = row['size']
        
        # Calculate deficit scores for each split considering:
        # 1. Scene type balance
        # 2. Polarization balance
        # 3. Overall size
        
        def calculate_deficit_score(split_name, counts_dict, target_dict, row):
            """Calculate how much this cluster would help balance the split."""
            score = 0.0
            
            # Scene type deficits (weight = 5.0 - increased to ensure representation)
            for scene in scene_types:
                scene_target = target_dict[scene]
                scene_current = counts_dict[scene]
                if scene_target > 0:
                    scene_deficit = (scene_target - scene_current) / scene_target
                    contribution = row[f'{scene}_count']
                    score += scene_deficit * contribution * 5.0
            
            # Polarization deficits (weight = 3.0)
            for pol, pol_count in row['pol_counts'].items():
                pol_target = target_dict['polarizations'].get(pol, 0)
                pol_current = counts_dict['polarizations'].get(pol, 0)
                if pol_target > 0:
                    pol_deficit = (pol_target - pol_current) / pol_target
                    score += pol_deficit * pol_count * 3.0
            
            # Total size deficit (weight = 1.0)
            total_target = target_dict['total']
            total_current = counts_dict['total']
            if total_target > 0:
                total_deficit = (total_target - total_current) / total_target
                score += total_deficit * cluster_size * 1.0
            
            return score
        
        # Calculate scores for each split
        train_score = calculate_deficit_score('train', train_counts, targets['train'], row)
        val_score = calculate_deficit_score('val', val_counts, targets['val'], row)
        test_score = calculate_deficit_score('test', test_counts, targets['test'], row)
        
        # Assign to split with highest deficit score
        if train_score >= val_score and train_score >= test_score:
            train_clusters.append(cluster_id)
            train_counts['total'] += cluster_size
            train_counts['land'] += row['land_count']
            train_counts['sea'] += row['sea_count']
            train_counts['coast'] += row['coast_count']
            for pol, cnt in row['pol_counts'].items():
                train_counts['polarizations'][pol] = train_counts['polarizations'].get(pol, 0) + cnt
            assigned_to = "train"
        elif val_score >= test_score:
            val_clusters.append(cluster_id)
            val_counts['total'] += cluster_size
            val_counts['land'] += row['land_count']
            val_counts['sea'] += row['sea_count']
            val_counts['coast'] += row['coast_count']
            for pol, cnt in row['pol_counts'].items():
                val_counts['polarizations'][pol] = val_counts['polarizations'].get(pol, 0) + cnt
            assigned_to = "val"
        else:
            test_clusters.append(cluster_id)
            test_counts['total'] += cluster_size
            test_counts['land'] += row['land_count']
            test_counts['sea'] += row['sea_count']
            test_counts['coast'] += row['coast_count']
            for pol, cnt in row['pol_counts'].items():
                test_counts['polarizations'][pol] = test_counts['polarizations'].get(pol, 0) + cnt
            assigned_to = "test"
        
        if idx < 10 or idx % 20 == 0:  # Print first 10 and every 20th for monitoring
            print(f"    Cluster {cluster_id}: {cluster_size} samples (land:{row['land_count']}, sea:{row['sea_count']}, coast:{row['coast_count']}) → {assigned_to}")
    
    print(f"\n✅ Cluster assignment complete:")
    print(f"  Train: {len(train_clusters)} clusters, {train_counts['total']} samples ({train_counts['total']/len(df)*100:.1f}%)")
    print(f"    Scene: land={train_counts['land']}, sea={train_counts['sea']}, coast={train_counts['coast']}")
    print(f"    Polarizations: {', '.join([f'{pol}={cnt}' for pol, cnt in sorted(train_counts['polarizations'].items())])}")
    print(f"  Validation: {len(val_clusters)} clusters, {val_counts['total']} samples ({val_counts['total']/len(df)*100:.1f}%)")
    print(f"    Scene: land={val_counts['land']}, sea={val_counts['sea']}, coast={val_counts['coast']}")
    print(f"    Polarizations: {', '.join([f'{pol}={cnt}' for pol, cnt in sorted(val_counts['polarizations'].items())])}")
    print(f"  Test: {len(test_clusters)} clusters, {test_counts['total']} samples ({test_counts['total']/len(df)*100:.1f}%)")
    print(f"    Scene: land={test_counts['land']}, sea={test_counts['sea']}, coast={test_counts['coast']}")
    print(f"    Polarizations: {', '.join([f'{pol}={cnt}' for pol, cnt in sorted(test_counts['polarizations'].items())])}")
    
    # Create splits based on cluster assignments
    splits = {
        'train': df[df['geo_cluster'].isin(train_clusters)].copy(),
        'validation': df[df['geo_cluster'].isin(val_clusters)].copy(),
        'test': df[df['geo_cluster'].isin(test_clusters)].copy()
    }
    
    # Verify and display distributions for each split
    print(f"\n✅ FINAL SPLIT DISTRIBUTIONS:")
    print("="*80)
    
    for split_name, split_df in splits.items():
        print(f"\n{split_name.upper()} SET ({len(split_df)} samples):")
        
        # Polarization distribution
        print(f"  Polarization:")
        pol_counts = split_df['polarization'].value_counts()
        for pol, count in pol_counts.items():
            orig_pct = pol_dist[pol] / len(df) * 100
            split_pct = count / len(split_df) * 100
            diff = split_pct - orig_pct
            print(f"    {pol}: {count} ({split_pct:.1f}%, original: {orig_pct:.1f}%, Δ{diff:+.1f}%)")
        
        # Year distribution
        print(f"  Years:")
        year_counts = split_df['year'].value_counts().sort_index()
        for year, count in year_counts.items():
            orig_pct = year_dist[year] / len(df) * 100
            split_pct = count / len(split_df) * 100
            diff = split_pct - orig_pct
            print(f"    {year}: {count} ({split_pct:.1f}%, original: {orig_pct:.1f}%, Δ{diff:+.1f}%)")
        
        # Scene type distribution
        print(f"  Scene Types:")
        scene_counts = split_df['scene_type'].value_counts()
        for scene, count in scene_counts.items():
            orig_pct = scene_dist[scene] / len(df) * 100
            split_pct = count / len(split_df) * 100
            diff = split_pct - orig_pct
            print(f"    {scene}: {count} ({split_pct:.1f}%, original: {orig_pct:.1f}%, Δ{diff:+.1f}%)")
        
        # Geographic bounds
        lat_bounds = (split_df['latitude'].min(), split_df['latitude'].max())
        lon_bounds = (split_df['longitude'].min(), split_df['longitude'].max())
        print(f"  Geographic coverage:")
        print(f"    Latitude: {lat_bounds[0]:.2f}° to {lat_bounds[1]:.2f}°")
        print(f"    Longitude: {lon_bounds[0]:.2f}° to {lon_bounds[1]:.2f}°")
        print(f"    Unique clusters: {split_df['geo_cluster'].nunique()}")
    
    # Check that we used all samples
    total_split_samples = sum(len(split_df) for split_df in splits.values())
    print(f"\n📊 SAMPLE RETENTION:")
    print(f"  Original dataset: {len(df)} samples")
    print(f"  Total in splits: {total_split_samples} samples")
    print(f"  Retention rate: {total_split_samples/len(df)*100:.1f}%")
    
    # Check geographic separation
    if len(splits['train']) > 0 and len(splits['validation']) > 0 and len(splits['test']) > 0:
        print(f"\n🌍 GEOGRAPHIC SEPARATION CHECK:")
        ensure_geographic_separation(splits['train'], splits['validation'], splits['test'])
    
    # Save split lists
    print(f"\n💾 Saving split lists to {output_dir}...")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for split_name, split_df in splits.items():
        if len(split_df) > 0:
            # Save full CSV with metadata using semicolon delimiter
            # This prevents issues with commas in coordinate fields
            csv_path = output_path / f"{split_name}_products.csv"
            split_df.to_csv(csv_path, index=False, sep=';')
            print(f"  ✓ Saved {csv_path} (using ';' delimiter)")
            
            # Save simple file list (just zarr filenames)
            file_list_path = output_path / f"{split_name}_files.txt"
            split_df['filename'].to_csv(file_list_path, index=False, header=False)
            print(f"  ✓ Saved {file_list_path}")
    
    # Create summary statistics
    summary_path = output_path / "split_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("SAR Dataset Split Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total products processed: {len(df)}\n")
        f.write(f"Geographic clusters: {n_clusters}\n")
        f.write(f"Sampling fraction: {sampling_fraction:.2f}\n\n")
        
        for split_name, split_df in splits.items():
            f.write(f"{split_name.upper()} SET: {len(split_df)} products\n")
            if len(split_df) > 0:
                year_dist = split_df['year'].value_counts().sort_index()
                scene_dist = split_df['scene_type'].value_counts()
                cluster_dist = split_df['geo_cluster'].value_counts().sort_index()
                
                f.write(f"  Years: {dict(year_dist)}\n")
                f.write(f"  Scene types: {dict(scene_dist)}\n")
                f.write(f"  Clusters: {dict(cluster_dist)}\n")
            f.write("\n")
    
    print(f"  ✓ Saved {summary_path}")
    
    return splits


def visualize_splits(splits: Dict[str, pd.DataFrame], output_dir: str = "./split_lists"):
    """
    Create visualizations of the dataset splits.
    
    Args:
        splits: Dictionary containing train/validation/test DataFrames
        output_dir: Directory to save visualizations
    """
    if not HAS_FOLIUM:
        print("Folium not available, skipping map visualization")
        return
    
    print(f"\n🗺️ Creating split visualizations...")
    
    # Create interactive map
    map_center = [0, 0]  # World center
    split_map = folium.Map(location=map_center, zoom_start=2)
    
    colors = {'train': 'blue', 'validation': 'green', 'test': 'red'}
    
    for split_name, split_df in splits.items():
        if len(split_df) == 0:
            continue
            
        color = colors.get(split_name, 'gray')
        
        for _, row in split_df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                popup=f"{split_name}: {row['filename']}",
                color=color,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(split_map)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 150px; height: 90px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px;
                "><p><b>Dataset Splits</b></p>
    <p><i class="fa fa-circle" style="color:blue"></i> Training</p>
    <p><i class="fa fa-circle" style="color:green"></i> Validation</p>  
    <p><i class="fa fa-circle" style="color:red"></i> Test</p>
    </div>
    '''
    split_map.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    map_path = Path(output_dir) / "dataset_splits_map.html"
    split_map.save(str(map_path))
    print(f"  ✓ Saved interactive map: {map_path}")
    
    # Create statistical plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Dataset Split Analysis', fontsize=16)
    
    # Plot 1: Sample counts by split
    ax1 = axes[0, 0]
    split_counts = {name: len(df) for name, df in splits.items() if len(df) > 0}
    ax1.bar(split_counts.keys(), split_counts.values(), color=['blue', 'green', 'red'])
    ax1.set_title('Samples per Split')
    ax1.set_ylabel('Number of Products')
    
    # Plot 2: Scene type distribution
    ax2 = axes[0, 1]
    scene_data = []
    for split_name, split_df in splits.items():
        if len(split_df) > 0:
            for scene_type in ['land', 'sea', 'coast']:
                count = len(split_df[split_df['scene_type'] == scene_type])
                scene_data.append({'split': split_name, 'scene_type': scene_type, 'count': count})
    
    if scene_data:
        scene_df = pd.DataFrame(scene_data)
        scene_pivot = scene_df.pivot(index='scene_type', columns='split', values='count').fillna(0)
        scene_pivot.plot(kind='bar', ax=ax2, color=['blue', 'green', 'red'])
        ax2.set_title('Scene Type Distribution')
        ax2.set_ylabel('Number of Products')
        ax2.legend(title='Split')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 3: Geographic distribution (latitude)
    ax3 = axes[1, 0]
    for split_name, split_df in splits.items():
        if len(split_df) > 0:
            ax3.hist(split_df['latitude'], alpha=0.6, label=split_name, bins=20)
    ax3.set_title('Latitude Distribution')
    ax3.set_xlabel('Latitude (degrees)')
    ax3.set_ylabel('Number of Products')
    ax3.legend()
    
    # Plot 4: Geographic distribution (longitude)
    ax4 = axes[1, 1] 
    for split_name, split_df in splits.items():
        if len(split_df) > 0:
            ax4.hist(split_df['longitude'], alpha=0.6, label=split_name, bins=20)
    ax4.set_title('Longitude Distribution')
    ax4.set_xlabel('Longitude (degrees)')
    ax4.set_ylabel('Number of Products')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / "dataset_splits_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved analysis plot: {plot_path}")
    plt.close()


def main():
    """Main function to create balanced dataset splits."""
    
    # Configuration
    csv_file = "/Data_large/marine/PythonProjects/SAR/sarpyx/dataloader/sar_products_locations.csv"
    output_dir = "/Data_large/marine/PythonProjects/SAR/sarpyx/dataset_splits"
    
    # Parameters for balancing
    n_clusters = 12  # Number of geographic clusters per scene type
    sampling_fraction = 0.8  # Use 80% of smallest cluster size
    use_geopy = False  # Set to True for more accurate land/sea classification (slower)
    
    print("SAR Dataset Balanced Splitter")
    print(f"Input: {csv_file}")
    print(f"Output: {output_dir}")
    print(f"Geographic clusters: {n_clusters}")
    print(f"Sampling fraction: {sampling_fraction}")
    print(f"Land/sea classification: {'Geopy API' if use_geopy else 'Coordinate-based'}")
    
    # Check if input file exists
    if not Path(csv_file).exists():
        print(f"❌ Input file not found: {csv_file}")
        print("Please run the world map notebook first to generate the CSV file.")
        return
    
    # Create balanced splits
    try:
        splits = create_balanced_splits(
            csv_file=csv_file,
            output_dir=output_dir,
            n_clusters=n_clusters,
            sampling_fraction=sampling_fraction,
            use_geopy=use_geopy,
            force_recreate=False  # Set to True to force recreation of splits
        )
        
        # Create visualizations
        visualize_splits(splits, output_dir)
        
        print(f"\n🎉 Dataset splitting completed successfully!")
        print(f"📁 Output files saved to: {output_dir}")
        print(f"\nFiles created:")
        print(f"  - train_products.csv & train_files.txt")
        print(f"  - validation_products.csv & validation_files.txt")
        print(f"  - test_products.csv & test_files.txt")
        print(f"  - split_summary.txt")
        print(f"  - dataset_splits_map.html")
        print(f"  - dataset_splits_analysis.png")
        
    except Exception as e:
        print(f"❌ Error during splitting: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()