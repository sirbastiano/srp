import pytest
import numpy as np
import zarr
import tempfile
import os
from pathlib import Path
from typing import Union, List, Tuple, Optional
import pandas as pd
import datashader as ds
from datashader import transfer_functions as tf
from ..utils.zarr_utils import ProductHandler

import matplotlib.pyplot as plt

# Import the ProductHandler class using relative import


@pytest.fixture
def mock_zarr_store():
    """Create a mock Zarr store with test arrays for visualization testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a Zarr group
        store = zarr.open(temp_dir, mode='w')
        
        # Create test arrays (all 200x150)
        # 'raw' array - complex data
        raw_data = np.random.normal(0, 5, (200, 150)) + 1j * np.random.normal(0, 5, (200, 150))
        store.create_dataset('raw', data=raw_data, chunks=(50, 50))
        
        # 'rc' array - range compressed data
        rc_data = np.random.normal(0, 20, (200, 150)) + 1j * np.random.normal(0, 20, (200, 150))
        store.create_dataset('rc', data=rc_data, chunks=(50, 50))
        
        # 'rcmc' array - range cell migration corrected data
        rcmc_data = np.random.normal(0, 50, (200, 150)) + 1j * np.random.normal(0, 50, (200, 150))
        store.create_dataset('rcmc', data=rcmc_data, chunks=(50, 50))
        
        # 'az' array - azimuth compressed (focused) data
        az_data = np.random.normal(0, 100, (200, 150)) + 1j * np.random.normal(0, 100, (200, 150))
        store.create_dataset('az', data=az_data, chunks=(50, 50))
        
        # Add metadata and ephemeris as attributes
        metadata = [{'id': i, 'value': f'test_{i}'} for i in range(200)]
        ephemeris = [{'time': i, 'position': [i, i*2, i*3]} for i in range(50)]
        
        store.attrs['metadata'] = {'data': metadata}
        store.attrs['ephemeris'] = {'data': ephemeris}
        
        yield temp_dir


def test_visualize_arrays_basic(mock_zarr_store, monkeypatch):
    """Test basic functionality of visualize_arrays method."""
    # Mock plt.show to prevent plots from displaying during tests
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    # Create ProductHandler with mock store
    handler = ProductHandler(mock_zarr_store)
    
    # Test with a single array
    handler.visualize_arrays('raw', rows=(0, 50), cols=(0, 50), show=False)
    
    # Test with multiple arrays
    handler.visualize_arrays(['raw', 'az'], rows=(0, 50), cols=(0, 50), show=False)
    
    # No assertion needed as we're just checking it runs without errors


def test_visualize_arrays_plot_types(mock_zarr_store, monkeypatch):
    """Test different plot types in visualize_arrays."""
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    handler = ProductHandler(mock_zarr_store)
    
    for plot_type in ['magnitude', 'phase', 'real', 'imag']:
        handler.visualize_arrays('az', plot_type=plot_type, show=False)


def test_visualize_arrays_vminmax(mock_zarr_store, monkeypatch):
    """Test different vminmax settings in visualize_arrays."""
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    handler = ProductHandler(mock_zarr_store)
    
    # Test with custom vminmax
    handler.visualize_arrays('rcmc', vminmax=(0, 200), show=False)
    
    # Test with auto vminmax
    handler.visualize_arrays('rcmc', vminmax='auto', show=False)


def test_visualize_arrays_with_datashader():
    """Test implementation of visualize_arrays using datashader."""
    # This is a standalone function to demonstrate datashader implementation
    
    def visualize_arrays_with_datashader(handler, 
                                        array_names: Union[str, List[str]],
                                        rows: Tuple[int, int] = (0, 100),
                                        cols: Tuple[int, int] = (0, 100),
                                        plot_type: str = 'magnitude',
                                        output_dir: Optional[str] = None,
                                        plot_width: int = 400,
                                        plot_height: int = 400,
                                        cmap: str = 'viridis'):
        """Visualize arrays using datashader.
        
        Args:
            handler: ProductHandler instance
            array_names: Name(s) of arrays to visualize
            rows: Row range (start, end)
            cols: Column range (start, end)
            plot_type: Type of plot ('magnitude', 'phase', 'real', 'imag')
            output_dir: Directory to save output images (None for no saving)
            plot_width: Width of output plot in pixels
            plot_height: Height of output plot in pixels
            cmap: Colormap name
            
        Returns:
            Dict mapping array names to datashader images
        """
        if isinstance(array_names, str):
            array_names = [array_names]
        
        # Convert tuple ranges to slices
        row_slice = slice(rows[0], rows[1])
        col_slice = slice(cols[0], cols[1])
        
        # Get data from handler
        result = handler.get_slice(array_names, rows=row_slice, cols=col_slice, include_metadata=False)
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Process each array
        images = {}
        for array_name in array_names:
            data = result['arrays'][array_name]
            
            # Apply appropriate transformation based on plot_type
            if plot_type == 'magnitude' and np.iscomplexobj(data):
                plot_data = np.abs(data)
                suffix = 'magnitude'
            elif plot_type == 'phase' and np.iscomplexobj(data):
                plot_data = np.angle(data)
                suffix = 'phase'
            elif plot_type == 'real':
                plot_data = np.real(data)
                suffix = 'real'
            elif plot_type == 'imag':
                plot_data = np.imag(data)
                suffix = 'imag'
            else:
                plot_data = data
                suffix = 'data'
            
            # Convert 2D array to DataFrame for datashader
            y_coords, x_coords = np.mgrid[0:plot_data.shape[0], 0:plot_data.shape[1]]
            df = pd.DataFrame({
                'y': y_coords.flatten(),
                'x': x_coords.flatten(),
                'z': plot_data.flatten()
            })
            
            # Create datashader canvas and aggregate
            cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height)
            agg = cvs.points(df, 'x', 'y', ds.mean('z'))
            
            # Create image
            img = tf.shade(agg, cmap=cmap)
            images[array_name] = img
            
            # Save image if output directory is specified
            if output_dir:
                output_path = os.path.join(output_dir, f"{array_name}_{suffix}.png")
                img.to_pil().save(output_path)
        
        return images
    
    # Test the function with a mock zarr store
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock zarr store
        mock_store_path = os.path.join(temp_dir, 'mock_zarr')
        store = zarr.open(mock_store_path, mode='w')
        
        # Create test array
        test_data = np.random.normal(0, 10, (100, 80)) + 1j * np.random.normal(0, 10, (100, 80))
        store.create_dataset('az', data=test_data)
        store.create_dataset('raw', data=test_data)
        
        # Add minimal metadata
        store.attrs['metadata'] = {'data': [{'id': i} for i in range(100)]}
        store.attrs['ephemeris'] = {'data': [{'time': i} for i in range(10)]}
        
        # Create handler
        handler = ProductHandler(mock_store_path)
        
        # Create output directory for images
        output_dir = os.path.join(temp_dir, 'output')
        
        # Test with single array
        images = visualize_arrays_with_datashader(
            handler, 'az', rows=(0, 50), cols=(0, 40), 
            output_dir=output_dir
        )
        assert 'az' in images
        assert os.path.exists(os.path.join(output_dir, 'az_magnitude.png'))
        
        # Test with multiple arrays
        images = visualize_arrays_with_datashader(
            handler, ['az', 'raw'], rows=(0, 50), cols=(0, 40),
            output_dir=output_dir
        )
        assert 'az' in images
        assert 'raw' in images
        
        # Test with different plot types
        for plot_type in ['magnitude', 'phase', 'real', 'imag']:
            images = visualize_arrays_with_datashader(
                handler, 'az', plot_type=plot_type, 
                output_dir=output_dir
            )
            assert os.path.exists(os.path.join(output_dir, f'az_{plot_type}.png'))


class TestProductHandlerWithDatashader:
    """Tests for ProductHandler with datashader visualization integration."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Add datashader visualization method to ProductHandler class."""
        # Add datashader visualization method to ProductHandler
        def visualize_with_datashader(self, 
                                     array_names: Union[str, List[str]],
                                     rows: Tuple[int, int] = (0, 100),
                                     cols: Tuple[int, int] = (0, 100),
                                     plot_type: str = 'magnitude',
                                     output_dir: Optional[str] = None,
                                     plot_width: int = 600,
                                     plot_height: int = 600,
                                     cmap: str = 'viridis'):
            """Visualize arrays using datashader for high-performance rendering.
            
            Args:
                array_names: Name(s) of arrays to visualize
                rows: Row range to visualize
                cols: Column range to visualize
                plot_type: Type of plot ('magnitude', 'phase', 'real', 'imag')
                output_dir: Directory to save output images
                plot_width: Width of output images in pixels
                plot_height: Height of output images in pixels
                cmap: Colormap name
                
            Returns:
                Dict mapping array names to datashader images
            """
            if isinstance(array_names, str):
                array_names = [array_names]
            
            # Get data slices
            result = self.get_slice(array_names, 
                                   rows=slice(rows[0], rows[1]), 
                                   cols=slice(cols[0], cols[1]),
                                   include_metadata=False)
            
            # Create output directory if specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            images = {}
            for array_name in array_names:
                data = result['arrays'][array_name]
                
                # Process data based on plot type
                if plot_type == 'magnitude' and np.iscomplexobj(data):
                    plot_data = np.abs(data)
                    title = f"{array_name.upper()} - Magnitude"
                elif plot_type == 'phase' and np.iscomplexobj(data):
                    plot_data = np.angle(data)
                    title = f"{array_name.upper()} - Phase"
                elif plot_type == 'real':
                    plot_data = np.real(data)
                    title = f"{array_name.upper()} - Real"
                elif plot_type == 'imag':
                    plot_data = np.imag(data)
                    title = f"{array_name.upper()} - Imaginary"
                else:
                    plot_data = data
                    title = f"{array_name.upper()}"
                
                # Create DataFrame for datashader
                y, x = np.mgrid[0:plot_data.shape[0], 0:plot_data.shape[1]]
                df = pd.DataFrame({
                    'y': y.flatten(),
                    'x': x.flatten(),
                    'z': plot_data.flatten()
                })
                
                # Create datashader canvas and render image
                cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height)
                agg = cvs.points(df, 'x', 'y', ds.mean('z'))
                img = tf.shade(agg, cmap=cmap)
                
                images[array_name] = img
                
                # Save image if output directory is specified
                if output_dir:
                    filename = f"{array_name}_{plot_type}.png"
                    output_path = os.path.join(output_dir, filename)
                    img.to_pil().save(output_path)
            
            return images
        
        # Add method to ProductHandler class
        ProductHandler.visualize_with_datashader = visualize_with_datashader
    
    def test_datashader_visualization(self, mock_zarr_store):
        """Test the integrated datashader visualization method."""
        handler = ProductHandler(mock_zarr_store)
        
        with tempfile.TemporaryDirectory() as temp_output:
            # Test single array visualization
            images = handler.visualize_with_datashader(
                'az', rows=(0, 100), cols=(0, 100),
                output_dir=temp_output
            )
            assert 'az' in images
            assert os.path.exists(os.path.join(temp_output, 'az_magnitude.png'))
            
            # Test multiple arrays
            images = handler.visualize_with_datashader(
                ['raw', 'rcmc'], output_dir=temp_output
            )
            assert 'raw' in images
            assert 'rcmc' in images
            
            # Test different plot types
            for plot_type in ['magnitude', 'phase', 'real', 'imag']:
                handler.visualize_with_datashader(
                    'rc', plot_type=plot_type, output_dir=temp_output
                )
                assert os.path.exists(os.path.join(temp_output, f'rc_{plot_type}.png'))
    
    def test_datashader_performance(self, mock_zarr_store):
        """Test datashader visualization with larger arrays for performance testing."""
        handler = ProductHandler(mock_zarr_store)
        
        # Only test with a subset of data for performance
        with tempfile.TemporaryDirectory() as temp_output:
            # Use different image sizes
            for size in [(200, 200), (400, 400)]:
                images = handler.visualize_with_datashader(
                    'az', 
                    plot_width=size[0], 
                    plot_height=size[1],
                    output_dir=temp_output
                )
                # Check that images were created
                assert 'az' in images