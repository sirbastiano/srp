"""
PyTorch Lightning DataModule for SAR data loading.
This module wraps the existing SAR dataloader functionality.
"""

import functools
import lightning as L
from dataloader import get_sar_dataloader, SARTransform
from utils import minmax_normalize, RC_MIN, RC_MAX, GT_MIN, GT_MAX
from typing import Optional


class SARDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for SAR data.
    
    This DataModule wraps the existing get_sar_dataloader functionality
    to work seamlessly with PyTorch Lightning trainers.
    """
    
    def __init__(
        self,
        train_dir: str = "/Data_large/marine/PythonProjects/SAR/sarpyx/SSM4SAR/data/maya4_data/train",
        val_dir: str = "/Data_large/marine/PythonProjects/SAR/sarpyx/SSM4SAR/data/maya4_data/validation",
        train_batch_size: int = 10,
        val_batch_size: int = 10,
        level_from: str = "rc",
        level_to: str = "az",
        num_workers: int = 16,
        patch_mode: str = "rectangular",
        patch_size: tuple = (10000, 1),
        buffer: tuple = (1000, 1000),
        stride: tuple = (1, 300),
        shuffle_files: bool = False,
        patch_order: str = "col",
        complex_valued: bool = True,
        positional_encoding: bool = True,
        save_samples: bool = False,
        backend: str = "zarr",
        verbose: bool = True,
        samples_per_prod: int = 10000,
        cache_size: int = 100,
        online: bool = False,
        max_products: int = 1,
        transform: Optional[SARTransform] = None
    ):
        """
        Initialize the SARDataModule.
        
        Args:
            train_dir: Directory containing the SAR training data
            val_dir: Directory containing the SAR validation data
            train_batch_size: Batch size for training
            val_batch_size: Batch size for validation
            level_from: Input SAR processing level
            level_to: Target SAR processing level
            num_workers: Number of worker processes for data loading
            patch_mode: Mode for patch extraction
            patch_size: Size of patches to extract
            buffer: Buffer size for patch extraction
            stride: Stride for patch extraction
            shuffle_files: Whether to shuffle files
            patch_order: Order of patch extraction
            complex_valued: Whether to use complex-valued data
            positional_encoding: Whether to add positional encoding
            save_samples: Whether to save samples
            backend: Backend to use for data loading
            verbose: Whether to print verbose output
            samples_per_prod: Number of samples per product
            cache_size: Size of the cache
            online: Whether to use online data loading
            max_products: Maximum number of products to use
            transform: Optional SARTransform to apply
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Store all parameters
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.level_from = level_from
        self.level_to = level_to
        self.num_workers = num_workers
        self.patch_mode = patch_mode
        self.patch_size = patch_size
        self.buffer = buffer
        self.stride = stride
        self.shuffle_files = shuffle_files
        self.patch_order = patch_order
        self.complex_valued = complex_valued
        self.positional_encoding = positional_encoding
        self.save_samples = save_samples
        self.backend = backend
        self.verbose = verbose
        self.samples_per_prod = samples_per_prod
        self.cache_size = cache_size
        self.online = online
        self.max_products = max_products
        
        # Create or use provided transform
        if transform is None:
            self.transform = SARTransform(
                transform_raw=functools.partial(minmax_normalize, array_min=RC_MIN, array_max=RC_MAX),
                transform_rc=functools.partial(minmax_normalize, array_min=RC_MIN, array_max=RC_MAX),
                transform_rcmc=functools.partial(minmax_normalize, array_min=RC_MIN, array_max=RC_MAX),
                transform_az=functools.partial(minmax_normalize, array_min=GT_MIN, array_max=GT_MAX)
            )
        else:
            self.transform = transform
    
    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single process
        in distributed settings.
        """
        pass
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup is called at the beginning of fit (train + validate), validate, test, or predict.
        This is a good hook for setting up datasets.
        
        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        # Setup logic can go here if needed
        # For now, dataloaders are created on-demand
        pass
    
    def train_dataloader(self):
        """
        Create and return the training dataloader.
        
        Returns:
            SARDataloader: Training dataloader
        """
        loader = get_sar_dataloader(
            data_dir=self.train_dir,
            level_from=self.level_from,
            level_to=self.level_to,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            patch_mode=self.patch_mode,
            patch_size=self.patch_size,
            buffer=self.buffer,
            stride=self.stride,
            shuffle_files=self.shuffle_files,
            patch_order=self.patch_order,
            complex_valued=self.complex_valued,
            positional_encoding=self.positional_encoding,
            save_samples=self.save_samples,
            backend=self.backend,
            verbose=self.verbose,
            samples_per_prod=self.samples_per_prod,
            cache_size=self.cache_size,
            online=self.online,
            max_products=self.max_products,
            transform=self.transform
        )
        
        # Workaround for dataloader length issue
        # The dataloader works but reports length 0
        # We'll patch the __len__ method to return the correct length
        dataset_len = len(loader.dataset)
        expected_len = max(1, dataset_len // self.train_batch_size) if dataset_len > 0 else 0
        
        class DataLoaderWrapper:
            def __init__(self, loader, length):
                self.loader = loader
                self.length = length
                
            def __iter__(self):
                return iter(self.loader)
            
            def __len__(self):
                return self.length
            
            def __getattr__(self, name):
                return getattr(self.loader, name)
        
        wrapped_loader = DataLoaderWrapper(loader, expected_len)
        return wrapped_loader
    
    def val_dataloader(self):
        """
        Create and return the validation dataloader.
        
        Returns:
            SARDataloader: Validation dataloader
        """
        loader = get_sar_dataloader(
            data_dir=self.val_dir,
            level_from=self.level_from,
            level_to=self.level_to,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            patch_mode=self.patch_mode,
            patch_size=self.patch_size,
            buffer=self.buffer,
            stride=self.stride,
            shuffle_files=self.shuffle_files,
            patch_order=self.patch_order,
            complex_valued=self.complex_valued,
            positional_encoding=self.positional_encoding,
            save_samples=self.save_samples,
            backend=self.backend,
            verbose=self.verbose,
            samples_per_prod=self.samples_per_prod,
            cache_size=self.cache_size,
            online=self.online,
            max_products=self.max_products,
            transform=self.transform
        )
        
        # Apply same workaround as train_dataloader
        dataset_len = len(loader.dataset)
        expected_len = max(1, dataset_len // self.val_batch_size) if dataset_len > 0 else 0
        
        class DataLoaderWrapper:
            def __init__(self, loader, length):
                self.loader = loader
                self.length = length
                
            def __iter__(self):
                return iter(self.loader)
            
            def __len__(self):
                return self.length
            
            def __getattr__(self, name):
                return getattr(self.loader, name)
        
        wrapped_loader = DataLoaderWrapper(loader, expected_len)
        return wrapped_loader
    
    def test_dataloader(self):
        """
        Create and return the test dataloader.
        Uses the same configuration as validation dataloader.
        
        Returns:
            SARDataloader: Test dataloader
        """
        return self.val_dataloader()
    
    def predict_dataloader(self):
        """
        Create and return the prediction dataloader.
        Uses the same configuration as validation dataloader.
        
        Returns:
            SARDataloader: Prediction dataloader
        """
        return self.val_dataloader()