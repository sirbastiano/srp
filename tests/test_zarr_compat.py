#!/usr/bin/env python3
"""Test Zarr 3.x compatibility."""
import zarr
import tempfile
import os
import numpy as np

print(f"Zarr version: {zarr.__version__}")

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, 'test.zarr')
    
    # Create a group-based store like dask_slice_saver does
    store = zarr.open_group(path, mode='w')
    store.create_array('az', shape=(10, 10), dtype='complex64', chunks=(5, 5))
    store.create_array('raw', shape=(10, 10), dtype='complex64', chunks=(5, 5))
    
    # Now open like ZarrManager does
    opened = zarr.open(path, mode='r')
    print(f"Opened type: {type(opened)}")
    print(f"Is Group? {isinstance(opened, zarr.Group)}")
    
    if isinstance(opened, zarr.Group):
        print(f"Has 'data'? {'data' in opened}")
        print(f"Has 'az'? {'az' in opened}")
        
        # What happens if we assign the group and try to access shape?
        arr = opened  # This is what happens when 'data' is not found
        print(f"Type of arr: {type(arr)}")
        try:
            print(f"Shape: {arr.shape}")
        except AttributeError as e:
            print(f"Shape access failed: Group has no shape attribute")

print("\nTest passed!")
