#!/usr/bin/env python3
"""
Test script to verify dataset splitting works correctly.
"""

from dataloader.create_balanced_dataset_splits import create_balanced_splits
from pathlib import Path

# Configuration
csv_file = "dataloader/sar_products_locations.csv"
output_dir = "balanced_samples"

# Run the splitting
print("Running dataset splitting...")
print(f"CSV file: {csv_file}")
print(f"Output directory: {output_dir}")
print("=" * 80)

splits = create_balanced_splits(
    csv_file=csv_file,
    output_dir=output_dir,
    n_clusters=3,  # Fewer clusters for small dataset
    sampling_fraction=0.8,  # Use 80% of smallest cluster
    use_geopy=False,  # Use simple coordinate-based classification
    force_recreate=True  # Force recreation to test
)

print("\n" + "=" * 80)
print("SPLITTING COMPLETE!")
print("=" * 80)

# Print summary
for split_name, split_df in splits.items():
    print(f"\n{split_name.upper()}:")
    print(f"  Total samples: {len(split_df)}")
    if len(split_df) > 0:
        print(f"  Years: {split_df['year'].unique().tolist()}")
        print(f"  Scene types: {split_df['scene_type'].value_counts().to_dict()}")
        print(f"  Clusters: {split_df['geo_cluster'].nunique()}")

print("\n✓ Test completed successfully!")
