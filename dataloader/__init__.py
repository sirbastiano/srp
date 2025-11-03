import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to enable absolute imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Also add current directory for direct module access
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def _import_with_fallback():
    """Import modules with multiple fallback strategies."""
    
    # Strategy 1: Try relative imports (when used as package)
    try:
        from .dataloader import (
            SARZarrDataset,
            KPatchSampler,
            get_sar_dataloader, 
            SARDataloader, 
            SampleFilter
        )
        from .location_utils import (
            get_products_spatial_mapping
        )
        from .normalization import (
            SARTransform, 
            NormalizationModule,
            ComplexNormalizationModule,
            IdentityModule,
            BaseTransformModule
        )
        from .api import (
            list_base_files_in_repo,
            fetch_chunk_from_hf_zarr,
            download_metadata_from_product
        )
        from .utils import (
            get_chunk_name_from_coords,
            get_sample_visualization,
            get_zarr_version,
            minmax_normalize,
            minmax_inverse,
            extract_stripmap_mode_from_filename,
            RC_MAX, RC_MIN, GT_MAX, GT_MIN
        )
        return (
            SARZarrDataset, SARTransform, NormalizationModule, ComplexNormalizationModule,
            IdentityModule, BaseTransformModule, KPatchSampler, SARDataloader, SampleFilter, get_sar_dataloader,
            get_products_spatial_mapping, list_base_files_in_repo, fetch_chunk_from_hf_zarr, download_metadata_from_product,
            get_chunk_name_from_coords, get_sample_visualization, get_zarr_version,
            minmax_normalize, minmax_inverse, extract_stripmap_mode_from_filename, RC_MAX, RC_MIN, GT_MAX, GT_MIN
        )
    except ImportError as e1:
        print(f"Relative import failed: {e1}")
        
        # Strategy 2: Try absolute imports from dataloader package
        try:
            from dataloader.dataloader import (
                SARZarrDataset,
                KPatchSampler,
                get_sar_dataloader,
                SARDataloader, 
                SampleFilter
            )
            from dataloader.location_utils import (
                get_products_spatial_mapping
            )
            from dataloader.normalization import (
                SARTransform,
                NormalizationModule,
                ComplexNormalizationModule,
                IdentityModule,
                BaseTransformModule
            )
            from dataloader.api import (
                list_base_files_in_repo,
                fetch_chunk_from_hf_zarr,
                download_metadata_from_product
            )
            from dataloader.utils import (
                get_chunk_name_from_coords,
                get_sample_visualization,
                get_zarr_version,
                minmax_normalize,
                minmax_inverse,
                extract_stripmap_mode_from_filename,
                RC_MAX, RC_MIN, GT_MAX, GT_MIN
            )
            return (
                SARZarrDataset, SARTransform, NormalizationModule, ComplexNormalizationModule,
                IdentityModule, BaseTransformModule, KPatchSampler, SARDataloader, SampleFilter, get_sar_dataloader,
                get_products_spatial_mapping, list_base_files_in_repo, fetch_chunk_from_hf_zarr, download_metadata_from_product,
                get_chunk_name_from_coords, get_sample_visualization, get_zarr_version,
                minmax_normalize, minmax_inverse, extract_stripmap_mode_from_filename, RC_MAX, RC_MIN, GT_MAX, GT_MIN
            )
        except ImportError as e2:
            print(f"Absolute import from dataloader failed: {e2}")
            
            # Strategy 3: Direct imports (when modules are in same directory or on path)
            try:
                # First, modify the dataloader.py to use absolute imports temporarily
                import importlib.util
                
                # Load modules directly by file path
                dataloader_path = current_dir / "dataloader.py"
                normalization_path = current_dir / "normalization.py"
                api_path = current_dir / "api.py"
                utils_path = current_dir / "utils.py"
                
                # Load utils first
                spec_utils = importlib.util.spec_from_file_location("utils", utils_path)
                utils_module = importlib.util.module_from_spec(spec_utils)
                sys.modules["utils"] = utils_module
                spec_utils.loader.exec_module(utils_module)
                
                # Load location_utils
                location_utils_path = current_dir / "location_utils.py"
                spec_location_utils = importlib.util.spec_from_file_location("location_utils", location_utils_path)
                location_utils_module = importlib.util.module_from_spec(spec_location_utils)
                sys.modules["location_utils"] = location_utils_module
                spec_location_utils.loader.exec_module(location_utils_module)
                # Load api
                spec_api = importlib.util.spec_from_file_location("api", api_path)
                api_module = importlib.util.module_from_spec(spec_api)
                sys.modules["api"] = api_module
                spec_api.loader.exec_module(api_module)

                # Load normalization
                spec_normalization = importlib.util.spec_from_file_location("normalization", normalization_path)
                normalization_module = importlib.util.module_from_spec(spec_normalization)
                sys.modules["normalization"] = normalization_module
                spec_normalization.loader.exec_module(normalization_module)

                # Load dataloader
                spec_dataloader = importlib.util.spec_from_file_location("dataloader_main", dataloader_path)
                dataloader_module = importlib.util.module_from_spec(spec_dataloader)
                spec_dataloader.loader.exec_module(dataloader_module)
                
                return (
                    dataloader_module.SARZarrDataset,
                    normalization_module.SARTransform,
                    normalization_module.NormalizationModule,
                    normalization_module.ComplexNormalizationModule,
                    normalization_module.IdentityModule,
                    normalization_module.BaseTransformModule,
                    dataloader_module.KPatchSampler,
                    dataloader_module.SARDataloader,
                    dataloader_module.SampleFilter,
                    dataloader_module.get_sar_dataloader,
                    location_utils_module.get_products_spatial_mapping,
                    api_module.list_base_files_in_repo,
                    api_module.fetch_chunk_from_hf_zarr,
                    api_module.download_metadata_from_product,
                    utils_module.get_chunk_name_from_coords,
                    utils_module.get_sample_visualization,
                    utils_module.get_zarr_version,
                    utils_module.minmax_normalize,
                    utils_module.minmax_inverse,
                    utils_module.extract_stripmap_mode_from_filename,
                    utils_module.RC_MAX, utils_module.RC_MIN, 
                    utils_module.GT_MAX, utils_module.GT_MIN
                )
            except Exception as e3:
                print(f"Direct file import failed: {e3}")
                raise ImportError(f"All import strategies failed. Errors: {e1}, {e2}, {e3}")

# Import all modules
try:
    (SARZarrDataset, SARTransform, NormalizationModule, ComplexNormalizationModule,
     IdentityModule, BaseTransformModule, KPatchSampler, SARDataloader, SampleFilter, get_sar_dataloader,
     get_products_spatial_mapping, list_base_files_in_repo, fetch_chunk_from_hf_zarr, download_metadata_from_product,
     get_chunk_name_from_coords, get_sample_visualization, get_zarr_version,
     minmax_normalize, minmax_inverse, extract_stripmap_mode_from_filename, RC_MAX, RC_MIN, GT_MAX, GT_MIN) = _import_with_fallback()
except ImportError as e:
    print(f"Failed to import dataloader modules: {e}")
    # Create dummy objects to prevent further import errors
    class DummyClass: pass
    def dummy_function(*args, **kwargs): pass
    
    SARZarrDataset = SARTransform = NormalizationModule = ComplexNormalizationModule = DummyClass
    IdentityModule = BaseTransformModule = KPatchSampler = DummyClass
    get_sar_dataloader = list_base_files_in_repo = fetch_chunk_from_hf_zarr = dummy_function
    download_metadata_from_product = get_chunk_name_from_coords = dummy_function
    get_sample_visualization = get_zarr_version = normalize = dummy_function
    extract_stripmap_mode_from_filename = dummy_function
    RC_MAX = RC_MIN = GT_MAX = GT_MIN = 0

# Define what gets imported with "from dataloader import *"
__all__ = [
    # Main classes
    'SARZarrDataset',
    'KPatchSampler',
    'get_sar_dataloader',
    'SARDataloader',
    'SampleFilter',
    
    # Normalization and transformation modules
    'SARTransform', 
    'NormalizationModule',
    'ComplexNormalizationModule', 
    'IdentityModule',
    'BaseTransformModule',
    
    
    
    # API functions
    'list_base_files_in_repo',
    'fetch_chunk_from_hf_zarr', 
    'download_metadata_from_product',
    
    # Utility functions
    'get_products_spatial_mapping',
    'get_chunk_name_from_coords',
    'get_sample_visualization',
    'get_zarr_version',
    'minmax_normalize',
    'minmax_inverse',
    'extract_stripmap_mode_from_filename',
    
    # Constants
    'RC_MAX', 'RC_MIN', 'GT_MAX', 'GT_MIN'
]

# Version info
__version__ = "1.0.0"
__author__ = "Gabriele Daga"