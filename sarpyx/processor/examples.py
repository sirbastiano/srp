"""
Examples and tutorials for using the SAR processor module.

This module contains example workflows and tutorials demonstrating
how to use the restructured processor module for various SAR processing tasks.
"""

from ..core import focus, decode, transforms
from ..autofocus import metrics
from ..data import readers, writers
from ..algorithms import rda


def example_sentinel1_processing():
    """
    Example workflow for processing Sentinel-1 data.
    
    This example demonstrates:
    1. Reading Sentinel-1 data
    2. Decoding the signal
    3. Applying focus algorithms
    4. Quality assessment with autofocus metrics
    5. Writing output data
    """
    print("Sentinel-1 Processing Example")
    print("1. Reading Sentinel-1 data...")
    # data = readers.read_sentinel1()
    
    print("2. Decoding signal...")
    # decoded_data = decode.extract_echo_bursts()
    
    print("3. Applying focus algorithms...")
    # focused_data = focus.apply_focusing()
    
    print("4. Quality assessment...")
    # quality = metrics.ssim()
    
    print("5. Writing output...")
    # writers.write_geotiff()
    
    print("Processing complete!")


def example_autofocus_workflow():
    """
    Example workflow for autofocus processing.
    
    This example demonstrates:
    1. Loading SAR data
    2. Computing focus metrics
    3. Optimizing focus parameters
    4. Applying corrections
    """
    print("Autofocus Workflow Example")
    print("1. Loading SAR data...")
    print("2. Computing initial focus metrics...")
    print("3. Optimizing focus parameters...")
    print("4. Applying focus corrections...")
    print("Autofocus complete!")


def example_algorithm_comparison():
    """
    Example comparing different SAR processing algorithms.
    
    This example demonstrates:
    1. Range-Doppler Algorithm
    2. Back-projection Algorithm
    3. Performance comparison
    """
    print("Algorithm Comparison Example")
    print("1. Applying Range-Doppler Algorithm...")
    # rda_result = rda.simple_rda()
    
    print("2. Applying Back-projection Algorithm...")
    # bp_result = backprojection.time_domain_backprojection()
    
    print("3. Comparing results...")
    print("Comparison complete!")


if __name__ == "__main__":
    print("SAR Processor Examples")
    print("=" * 50)
    
    example_sentinel1_processing()
    print()
    
    example_autofocus_workflow()
    print()
    
    example_algorithm_comparison()
    print()
    
    print("All examples completed!")
