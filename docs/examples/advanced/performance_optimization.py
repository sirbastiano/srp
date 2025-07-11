#!/usr/bin/env python3
"""
SARPYX Performance Optimization Example
======================================

This example demonstrates advanced performance optimization techniques for SAR data
processing using SARPYX. It covers memory management, computational optimization,
parallel processing strategies, and profiling techniques.

Topics covered:
- Memory-efficient data processing
- Computational optimization strategies
- Parallel and distributed processing
- GPU acceleration techniques
- Algorithm optimization
- Performance profiling and benchmarking
- Cache management and optimization
- Memory mapping for large datasets
- Streaming processing for real-time applications
- Code optimization best practices

Author: SARPYX Development Team
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Iterator
import time
import gc
import psutil
import threading
from functools import wraps, lru_cache
from dataclasses import dataclass
import cProfile
import pstats
import io
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from multiprocessing import shared_memory
import mmap
import h5py
import warnings

# SARPYX imports
from sarpyx.sla import SLAProcessor
from sarpyx.utils import io as sarpyx_io
from sarpyx.utils import viz as sarpyx_viz
from sarpyx.science import indices
from sarpyx.snap import engine as snap_engine

# Optional GPU acceleration imports
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

# Optional advanced optimization imports
try:
    import numba
    from numba import jit, cuda, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    numba = None
    jit = lambda func: func  # Fallback decorator
    prange = range

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_percent: float = 0.0
    throughput_mb_per_sec: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    gpu_memory_mb: float = 0.0

class PerformanceProfiler:
    """Comprehensive performance profiler for SAR processing."""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.start_time = 0.0
        self.start_memory = 0.0
        self.profiler = None
        self.monitoring = False
        self.monitor_thread = None
        self.peak_memory = 0.0
        
    @contextmanager
    def profile(self, enable_line_profiling: bool = False):
        """Context manager for performance profiling."""
        # Start profiling
        self.start_profiling()
        
        try:
            yield self
        finally:
            # Stop profiling
            self.stop_profiling()
    
    def start_profiling(self):
        """Start performance profiling."""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / (1024*1024)
        self.peak_memory = self.start_memory
        
        # Start CPU profiling
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        
        # Start memory monitoring
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Performance profiling started")
    
    def stop_profiling(self):
        """Stop performance profiling."""
        # Stop CPU profiling
        if self.profiler:
            self.profiler.disable()
        
        # Stop memory monitoring
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        # Calculate final metrics
        self.metrics.execution_time = time.time() - self.start_time
        current_memory = psutil.Process().memory_info().rss / (1024*1024)
        self.metrics.memory_usage_mb = current_memory - self.start_memory
        self.metrics.peak_memory_mb = self.peak_memory
        self.metrics.cpu_percent = psutil.cpu_percent()
        
        if GPU_AVAILABLE:
            try:
                self.metrics.gpu_memory_mb = cp.get_default_memory_pool().used_bytes() / (1024*1024)
            except:
                pass
        
        logger.info(f"Performance profiling completed: {self.metrics.execution_time:.2f}s")
    
    def _monitor_resources(self):
        """Monitor system resources during profiling."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                memory_mb = process.memory_info().rss / (1024*1024)
                self.peak_memory = max(self.peak_memory, memory_mb)
                time.sleep(0.1)  # Monitor every 100ms
            except:
                break
    
    def get_cpu_profile_stats(self) -> str:
        """Get CPU profiling statistics."""
        if not self.profiler:
            return "No profiling data available"
        
        stats_stream = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        return stats_stream.getvalue()
    
    def get_memory_profile(self) -> Dict[str, float]:
        """Get memory profiling information."""
        return {
            'current_memory_mb': psutil.Process().memory_info().rss / (1024*1024),
            'peak_memory_mb': self.metrics.peak_memory_mb,
            'memory_increase_mb': self.metrics.memory_usage_mb,
            'available_memory_mb': psutil.virtual_memory().available / (1024*1024)
        }

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

def memory_efficient_decorator(func):
    """Decorator to optimize memory usage."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Force garbage collection before function
        gc.collect()
        
        start_memory = psutil.Process().memory_info().rss / (1024*1024)
        
        try:
            result = func(*args, **kwargs)
        finally:
            # Force garbage collection after function
            gc.collect()
            
            end_memory = psutil.Process().memory_info().rss / (1024*1024)
            memory_used = end_memory - start_memory
            
            if memory_used > 100:  # Log if significant memory usage
                logger.info(f"{func.__name__} used {memory_used:.1f} MB")
        
        return result
    return wrapper

class MemoryMappedArray:
    """Memory-mapped array for efficient large data handling."""
    
    def __init__(self, file_path: str, shape: Tuple[int, ...], dtype: np.dtype, mode: str = 'r'):
        self.file_path = Path(file_path)
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        self._mmap = None
        self._array = None
        
    def __enter__(self):
        """Enter context manager."""
        self._file = open(self.file_path, 'rb' if 'r' in self.mode else 'r+b')
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._array = np.frombuffer(self._mmap, dtype=self.dtype).reshape(self.shape)
        return self._array
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self._mmap:
            self._mmap.close()
        if self._file:
            self._file.close()

class ChunkedProcessor:
    """Process large arrays in memory-efficient chunks."""
    
    def __init__(self, chunk_size_mb: float = 100.0):
        self.chunk_size_mb = chunk_size_mb
        
    def process_array(self, array: np.ndarray, process_func: Callable, 
                     overlap: int = 0, **kwargs) -> np.ndarray:
        """Process array in chunks."""
        # Calculate chunk size based on memory limit
        element_size = array.dtype.itemsize
        elements_per_mb = (1024 * 1024) // element_size
        chunk_elements = int(self.chunk_size_mb * elements_per_mb)
        
        # Determine chunk dimensions
        if array.ndim == 1:
            chunk_size = min(chunk_elements, array.shape[0])
            return self._process_1d_chunks(array, process_func, chunk_size, overlap, **kwargs)
        elif array.ndim == 2:
            # Process row-wise chunks
            rows_per_chunk = max(1, chunk_elements // array.shape[1])
            return self._process_2d_chunks(array, process_func, rows_per_chunk, overlap, **kwargs)
        else:
            raise ValueError("Only 1D and 2D arrays supported")
    
    def _process_1d_chunks(self, array: np.ndarray, process_func: Callable,
                          chunk_size: int, overlap: int, **kwargs) -> np.ndarray:
        """Process 1D array in chunks."""
        results = []
        
        for start in range(0, len(array), chunk_size - overlap):
            end = min(start + chunk_size, len(array))
            chunk = array[start:end]
            
            result = process_func(chunk, **kwargs)
            
            # Handle overlap removal
            if overlap > 0 and start > 0:
                result = result[overlap:]
            
            results.append(result)
        
        return np.concatenate(results)
    
    def _process_2d_chunks(self, array: np.ndarray, process_func: Callable,
                          rows_per_chunk: int, overlap: int, **kwargs) -> np.ndarray:
        """Process 2D array in chunks."""
        results = []
        
        for start_row in range(0, array.shape[0], rows_per_chunk - overlap):
            end_row = min(start_row + rows_per_chunk, array.shape[0])
            chunk = array[start_row:end_row, :]
            
            result = process_func(chunk, **kwargs)
            
            # Handle overlap removal
            if overlap > 0 and start_row > 0:
                result = result[overlap:, :]
            
            results.append(result)
        
        return np.concatenate(results, axis=0)

class StreamingProcessor:
    """Streaming processor for real-time or large-scale processing."""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.buffer = []
        self.results = []
        
    def process_stream(self, data_generator: Iterator[np.ndarray], 
                      process_func: Callable, **kwargs) -> Iterator[np.ndarray]:
        """Process data stream in real-time."""
        for data_chunk in data_generator:
            self.buffer.append(data_chunk)
            
            # Process when buffer is full
            if len(self.buffer) >= self.buffer_size:
                combined_data = np.concatenate(self.buffer)
                result = process_func(combined_data, **kwargs)
                yield result
                
                # Clear buffer
                self.buffer.clear()
        
        # Process remaining data in buffer
        if self.buffer:
            combined_data = np.concatenate(self.buffer)
            result = process_func(combined_data, **kwargs)
            yield result

class OptimizedSLAProcessor:
    """Optimized Sub-Look Analysis processor with performance enhancements."""
    
    def __init__(self, use_gpu: bool = False, use_numba: bool = False):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self.cache = {}
        
        # Initialize GPU if available
        if self.use_gpu:
            logger.info("GPU acceleration enabled")
        
        # Initialize Numba if available
        if self.use_numba:
            logger.info("Numba acceleration enabled")
    
    @timing_decorator
    @memory_efficient_decorator
    def process_optimized(self, data: np.ndarray, sub_apertures: int = 4,
                         overlap_factor: float = 0.5, **kwargs) -> Dict[str, np.ndarray]:
        """Optimized SLA processing with multiple acceleration methods."""
        
        # Choose processing method based on available acceleration
        if self.use_gpu and self._can_use_gpu(data):
            return self._process_gpu(data, sub_apertures, overlap_factor, **kwargs)
        elif self.use_numba:
            return self._process_numba(data, sub_apertures, overlap_factor, **kwargs)
        else:
            return self._process_optimized_cpu(data, sub_apertures, overlap_factor, **kwargs)
    
    def _can_use_gpu(self, data: np.ndarray) -> bool:
        """Check if GPU processing is suitable for the data size."""
        data_size_mb = data.nbytes / (1024*1024)
        gpu_memory_mb = cp.get_default_memory_pool().total_bytes() / (1024*1024)
        
        # Use GPU only if data fits comfortably in GPU memory
        return data_size_mb < gpu_memory_mb * 0.5
    
    def _process_gpu(self, data: np.ndarray, sub_apertures: int,
                    overlap_factor: float, **kwargs) -> Dict[str, np.ndarray]:
        """GPU-accelerated SLA processing using CuPy."""
        logger.info("Processing with GPU acceleration")
        
        # Transfer data to GPU
        gpu_data = cp.asarray(data)
        
        # Perform SLA processing on GPU
        results = {}
        
        # Sub-aperture processing
        sub_looks = self._extract_sub_looks_gpu(gpu_data, sub_apertures, overlap_factor)
        results['sub_looks'] = cp.asnumpy(sub_looks)
        
        # Coherence calculation
        coherence_matrix = self._calculate_coherence_gpu(sub_looks)
        results['coherence_matrix'] = cp.asnumpy(coherence_matrix)
        
        # Additional features
        if np.iscomplexobj(data):
            intensity = cp.abs(gpu_data)**2
            results['intensity'] = cp.asnumpy(intensity)
            
            phase = cp.angle(gpu_data)
            results['phase'] = cp.asnumpy(phase)
        
        logger.info("GPU processing completed")
        return results
    
    def _extract_sub_looks_gpu(self, gpu_data: 'cp.ndarray', sub_apertures: int,
                              overlap_factor: float) -> 'cp.ndarray':
        """Extract sub-looks using GPU acceleration."""
        # Simplified sub-look extraction for demonstration
        # In practice, this would involve FFT operations and windowing
        
        if gpu_data.ndim == 1:
            # 1D case - split into sub-apertures
            aperture_size = len(gpu_data) // sub_apertures
            sub_looks = cp.zeros((sub_apertures, aperture_size), dtype=gpu_data.dtype)
            
            for i in range(sub_apertures):
                start_idx = i * aperture_size
                end_idx = start_idx + aperture_size
                sub_looks[i] = gpu_data[start_idx:end_idx]
        
        else:
            # 2D case - split along azimuth dimension
            aperture_size = gpu_data.shape[0] // sub_apertures
            sub_looks = cp.zeros((sub_apertures, aperture_size, gpu_data.shape[1]), 
                               dtype=gpu_data.dtype)
            
            for i in range(sub_apertures):
                start_idx = i * aperture_size
                end_idx = start_idx + aperture_size
                sub_looks[i] = gpu_data[start_idx:end_idx]
        
        return sub_looks
    
    def _calculate_coherence_gpu(self, sub_looks: 'cp.ndarray') -> 'cp.ndarray':
        """Calculate coherence matrix using GPU acceleration."""
        n_looks = sub_looks.shape[0]
        coherence_matrix = cp.zeros((n_looks, n_looks), dtype=cp.complex64)
        
        for i in range(n_looks):
            for j in range(i, n_looks):
                # Calculate cross-correlation
                look_i = sub_looks[i].flatten()
                look_j = sub_looks[j].flatten()
                
                cross_corr = cp.mean(look_i * cp.conj(look_j))
                auto_corr_i = cp.mean(cp.abs(look_i)**2)
                auto_corr_j = cp.mean(cp.abs(look_j)**2)
                
                coherence = cross_corr / cp.sqrt(auto_corr_i * auto_corr_j + 1e-10)
                coherence_matrix[i, j] = coherence
                coherence_matrix[j, i] = cp.conj(coherence)
        
        return coherence_matrix
    
    def _process_numba(self, data: np.ndarray, sub_apertures: int,
                      overlap_factor: float, **kwargs) -> Dict[str, np.ndarray]:
        """Numba-accelerated SLA processing."""
        logger.info("Processing with Numba acceleration")
        
        results = {}
        
        # Sub-aperture processing with Numba
        sub_looks = self._extract_sub_looks_numba(data, sub_apertures)
        results['sub_looks'] = sub_looks
        
        # Coherence calculation with Numba
        coherence_matrix = self._calculate_coherence_numba(sub_looks)
        results['coherence_matrix'] = coherence_matrix
        
        logger.info("Numba processing completed")
        return results
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _extract_sub_looks_numba(data: np.ndarray, sub_apertures: int) -> np.ndarray:
        """Numba-accelerated sub-look extraction."""
        if data.ndim == 1:
            aperture_size = len(data) // sub_apertures
            sub_looks = np.zeros((sub_apertures, aperture_size), dtype=data.dtype)
            
            for i in prange(sub_apertures):
                start_idx = i * aperture_size
                end_idx = start_idx + aperture_size
                sub_looks[i] = data[start_idx:end_idx]
        else:
            aperture_size = data.shape[0] // sub_apertures
            sub_looks = np.zeros((sub_apertures, aperture_size, data.shape[1]), 
                               dtype=data.dtype)
            
            for i in prange(sub_apertures):
                start_idx = i * aperture_size
                end_idx = start_idx + aperture_size
                sub_looks[i] = data[start_idx:end_idx]
        
        return sub_looks
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _calculate_coherence_numba(sub_looks: np.ndarray) -> np.ndarray:
        """Numba-accelerated coherence calculation."""
        n_looks = sub_looks.shape[0]
        coherence_matrix = np.zeros((n_looks, n_looks), dtype=np.complex64)
        
        for i in prange(n_looks):
            for j in range(i, n_looks):
                # Flatten the sub-looks for processing
                look_i = sub_looks[i].flatten()
                look_j = sub_looks[j].flatten()
                
                # Calculate coherence
                cross_corr = np.mean(look_i * np.conj(look_j))
                auto_corr_i = np.mean(np.abs(look_i)**2)
                auto_corr_j = np.mean(np.abs(look_j)**2)
                
                coherence = cross_corr / np.sqrt(auto_corr_i * auto_corr_j + 1e-10)
                coherence_matrix[i, j] = coherence
                coherence_matrix[j, i] = np.conj(coherence)
        
        return coherence_matrix
    
    def _process_optimized_cpu(self, data: np.ndarray, sub_apertures: int,
                              overlap_factor: float, **kwargs) -> Dict[str, np.ndarray]:
        """CPU-optimized SLA processing."""
        logger.info("Processing with CPU optimization")
        
        results = {}
        
        # Use vectorized operations and memory-efficient processing
        sub_looks = self._extract_sub_looks_vectorized(data, sub_apertures)
        results['sub_looks'] = sub_looks
        
        # Optimized coherence calculation
        coherence_matrix = self._calculate_coherence_vectorized(sub_looks)
        results['coherence_matrix'] = coherence_matrix
        
        logger.info("CPU optimization processing completed")
        return results
    
    def _extract_sub_looks_vectorized(self, data: np.ndarray, sub_apertures: int) -> np.ndarray:
        """Vectorized sub-look extraction for CPU optimization."""
        if data.ndim == 1:
            # 1D vectorized processing
            aperture_size = len(data) // sub_apertures
            data_reshaped = data[:aperture_size * sub_apertures].reshape(sub_apertures, aperture_size)
            return data_reshaped
        else:
            # 2D vectorized processing
            aperture_size = data.shape[0] // sub_apertures
            total_size = aperture_size * sub_apertures
            data_truncated = data[:total_size]
            return data_truncated.reshape(sub_apertures, aperture_size, data.shape[1])
    
    def _calculate_coherence_vectorized(self, sub_looks: np.ndarray) -> np.ndarray:
        """Vectorized coherence calculation for CPU optimization."""
        n_looks = sub_looks.shape[0]
        
        # Flatten sub-looks for vectorized operations
        flattened_looks = sub_looks.reshape(n_looks, -1)
        
        # Calculate all cross-correlations at once
        cross_corr = np.dot(flattened_looks, flattened_looks.conj().T) / flattened_looks.shape[1]
        
        # Calculate auto-correlations
        auto_corr = np.diag(np.real(cross_corr))
        
        # Normalize to get coherence
        coherence_matrix = cross_corr / np.sqrt(auto_corr[:, None] * auto_corr[None, :] + 1e-10)
        
        return coherence_matrix

class CacheManager:
    """Intelligent cache management for SAR processing."""
    
    def __init__(self, max_size_mb: float = 1000.0):
        self.max_size_mb = max_size_mb
        self.cache = {}
        self.access_times = {}
        self.cache_size_mb = 0.0
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any) -> bool:
        """Put item in cache with size management."""
        # Estimate size
        if isinstance(value, np.ndarray):
            size_mb = value.nbytes / (1024*1024)
        else:
            size_mb = 1.0  # Default size estimate
        
        # Check if we need to evict items
        while self.cache_size_mb + size_mb > self.max_size_mb and self.cache:
            self._evict_lru()
        
        # Add new item
        if self.cache_size_mb + size_mb <= self.max_size_mb:
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.cache_size_mb += size_mb
            return True
        
        return False
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        # Find LRU item
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove from cache
        value = self.cache.pop(lru_key)
        del self.access_times[lru_key]
        
        # Update size
        if isinstance(value, np.ndarray):
            self.cache_size_mb -= value.nbytes / (1024*1024)
        else:
            self.cache_size_mb -= 1.0
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_times.clear()
        self.cache_size_mb = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size_mb': self.cache_size_mb,
            'max_size_mb': self.max_size_mb,
            'items_count': len(self.cache),
            'utilization_percent': (self.cache_size_mb / self.max_size_mb) * 100
        }

class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.chunked_processor = ChunkedProcessor()
        self.streaming_processor = StreamingProcessor()
        self.optimized_sla = OptimizedSLAProcessor(use_gpu=GPU_AVAILABLE, use_numba=NUMBA_AVAILABLE)
        self.cache_manager = CacheManager()
        
    def benchmark_processing_methods(self, data: np.ndarray, iterations: int = 3) -> Dict[str, Dict[str, float]]:
        """Benchmark different processing methods."""
        logger.info("Starting performance benchmarking...")
        
        benchmarks = {}
        
        # Standard processing
        benchmarks['standard'] = self._benchmark_method(
            "Standard Processing",
            lambda: self._standard_processing(data),
            iterations
        )
        
        # Optimized CPU processing
        benchmarks['optimized_cpu'] = self._benchmark_method(
            "Optimized CPU Processing",
            lambda: self.optimized_sla._process_optimized_cpu(data, 4, 0.5),
            iterations
        )
        
        # Numba processing (if available)
        if NUMBA_AVAILABLE:
            benchmarks['numba'] = self._benchmark_method(
                "Numba Processing",
                lambda: self.optimized_sla._process_numba(data, 4, 0.5),
                iterations
            )
        
        # GPU processing (if available)
        if GPU_AVAILABLE and self.optimized_sla._can_use_gpu(data):
            benchmarks['gpu'] = self._benchmark_method(
                "GPU Processing",
                lambda: self.optimized_sla._process_gpu(data, 4, 0.5),
                iterations
            )
        
        # Chunked processing
        benchmarks['chunked'] = self._benchmark_method(
            "Chunked Processing",
            lambda: self.chunked_processor.process_array(data, self._simple_process_func),
            iterations
        )
        
        return benchmarks
    
    def _benchmark_method(self, name: str, method: Callable, iterations: int) -> Dict[str, float]:
        """Benchmark a specific processing method."""
        logger.info(f"Benchmarking: {name}")
        
        times = []
        memory_usage = []
        
        for i in range(iterations):
            gc.collect()  # Clean up before measurement
            
            start_memory = psutil.Process().memory_info().rss / (1024*1024)
            start_time = time.time()
            
            try:
                result = method()
                execution_time = time.time() - start_time
                times.append(execution_time)
                
                end_memory = psutil.Process().memory_info().rss / (1024*1024)
                memory_usage.append(end_memory - start_memory)
                
            except Exception as e:
                logger.error(f"Benchmark failed for {name}: {e}")
                times.append(float('inf'))
                memory_usage.append(0)
        
        return {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'avg_memory_mb': np.mean(memory_usage),
            'max_memory_mb': np.max(memory_usage)
        }
    
    def _standard_processing(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Standard processing method for comparison."""
        # Simplified standard processing
        if np.iscomplexobj(data):
            intensity = np.abs(data)**2
            phase = np.angle(data)
            return {'intensity': intensity, 'phase': phase}
        else:
            return {'intensity': data}
    
    def _simple_process_func(self, chunk: np.ndarray) -> np.ndarray:
        """Simple processing function for chunked processing."""
        if np.iscomplexobj(chunk):
            return np.abs(chunk)**2
        else:
            return chunk
    
    def optimize_memory_usage(self, data_path: str, output_path: str, 
                             target_memory_mb: float = 1000.0) -> Dict[str, Any]:
        """Optimize processing for limited memory environments."""
        logger.info(f"Optimizing for memory limit: {target_memory_mb} MB")
        
        # Estimate data size
        with h5py.File(data_path, 'r') as f:
            dataset_names = list(f.keys())
            first_dataset = f[dataset_names[0]]
            data_size_mb = first_dataset.nbytes / (1024*1024)
        
        if data_size_mb <= target_memory_mb * 0.8:
            # Data fits in memory - use standard processing
            logger.info("Data fits in memory - using standard processing")
            
            with h5py.File(data_path, 'r') as f:
                data = f[dataset_names[0]][...]
            
            result = self.optimized_sla.process_optimized(data)
            
            # Save result
            with h5py.File(output_path, 'w') as f:
                for key, value in result.items():
                    f.create_dataset(key, data=value, compression='gzip')
            
            return {'method': 'standard', 'data_size_mb': data_size_mb}
        
        else:
            # Data too large - use chunked processing
            logger.info("Data too large - using chunked processing")
            
            chunk_size_mb = target_memory_mb * 0.5  # Use half of available memory
            self.chunked_processor.chunk_size_mb = chunk_size_mb
            
            with h5py.File(data_path, 'r') as input_file, h5py.File(output_path, 'w') as output_file:
                dataset = input_file[dataset_names[0]]
                
                # Process in chunks
                result = self.chunked_processor.process_array(dataset, self._simple_process_func)
                
                # Save result
                output_file.create_dataset('processed_data', data=result, compression='gzip')
            
            return {'method': 'chunked', 'data_size_mb': data_size_mb, 'chunk_size_mb': chunk_size_mb}
    
    def parallel_batch_processing(self, file_list: List[str], output_dir: str,
                                 max_workers: int = None) -> Dict[str, Any]:
        """Parallel processing of multiple files."""
        max_workers = max_workers or mp.cpu_count()
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"Processing {len(file_list)} files with {max_workers} workers")
        
        def process_single_file(file_path: str) -> Dict[str, Any]:
            """Process a single file."""
            try:
                start_time = time.time()
                
                # Load and process data
                data = sarpyx_io.load_sar_data(file_path)
                result = self.optimized_sla.process_optimized(data)
                
                # Save result
                output_file = output_path / f"{Path(file_path).stem}_processed.h5"
                with h5py.File(output_file, 'w') as f:
                    for key, value in result.items():
                        f.create_dataset(key, data=value, compression='gzip')
                
                processing_time = time.time() - start_time
                
                return {
                    'file_path': file_path,
                    'output_path': str(output_file),
                    'processing_time': processing_time,
                    'status': 'success'
                }
                
            except Exception as e:
                return {
                    'file_path': file_path,
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single_file, file_list))
        
        # Compile statistics
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        return {
            'total_files': len(file_list),
            'successful': len(successful),
            'failed': len(failed),
            'average_processing_time': np.mean([r['processing_time'] for r in successful]) if successful else 0,
            'results': results
        }
    
    def generate_performance_report(self, benchmarks: Dict[str, Dict[str, float]], 
                                  output_path: str = "performance_report.html") -> str:
        """Generate comprehensive performance report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SARPYX Performance Optimization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .best {{ background-color: #d4edda; font-weight: bold; }}
                .worst {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <h1>SARPYX Performance Optimization Report</h1>
            <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Processing Method Comparison</h2>
                <table>
                    <tr>
                        <th>Method</th>
                        <th>Avg Time (s)</th>
                        <th>Min Time (s)</th>
                        <th>Max Time (s)</th>
                        <th>Std Time (s)</th>
                        <th>Avg Memory (MB)</th>
                        <th>Max Memory (MB)</th>
                    </tr>
        """
        
        # Find best and worst methods
        avg_times = {method: data['avg_time'] for method, data in benchmarks.items() 
                    if data['avg_time'] != float('inf')}
        
        if avg_times:
            best_method = min(avg_times.keys(), key=lambda k: avg_times[k])
            worst_method = max(avg_times.keys(), key=lambda k: avg_times[k])
        else:
            best_method = worst_method = None
        
        for method, data in benchmarks.items():
            css_class = ""
            if method == best_method:
                css_class = "best"
            elif method == worst_method:
                css_class = "worst"
            
            html_content += f"""
                <tr class="{css_class}">
                    <td>{method.replace('_', ' ').title()}</td>
                    <td>{data['avg_time']:.3f}</td>
                    <td>{data['min_time']:.3f}</td>
                    <td>{data['max_time']:.3f}</td>
                    <td>{data['std_time']:.3f}</td>
                    <td>{data['avg_memory_mb']:.1f}</td>
                    <td>{data['max_memory_mb']:.1f}</td>
                </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Optimization Recommendations</h2>
                <ul>
        """
        
        # Add recommendations based on results
        if GPU_AVAILABLE and 'gpu' in benchmarks:
            html_content += "<li>GPU acceleration is available and recommended for large datasets</li>"
        
        if NUMBA_AVAILABLE and 'numba' in benchmarks:
            html_content += "<li>Numba acceleration provides significant performance improvements</li>"
        
        html_content += """
                    <li>Use chunked processing for datasets larger than available memory</li>
                    <li>Consider parallel processing for batch operations</li>
                    <li>Implement caching for frequently accessed intermediate results</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>System Information</h2>
                <table>
                    <tr><th>Property</th><th>Value</th></tr>
        """
        
        # Add system information
        system_info = {
            'CPU Cores': mp.cpu_count(),
            'Available Memory (GB)': psutil.virtual_memory().total / (1024**3),
            'GPU Available': GPU_AVAILABLE,
            'Numba Available': NUMBA_AVAILABLE
        }
        
        for prop, value in system_info.items():
            html_content += f"<tr><td>{prop}</td><td>{value}</td></tr>"
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Performance report generated: {output_path}")
        return output_path

def main():
    """
    Main function demonstrating performance optimization techniques.
    """
    print("SARPYX Performance Optimization Demo")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = PerformanceOptimizer()
    
    # Generate test data
    print("\n1. Generating Test Data")
    print("-" * 30)
    
    # Create synthetic SAR data for testing
    data_size = (1000, 1000)
    test_data = np.random.randn(*data_size) + 1j * np.random.randn(*data_size)
    test_data = test_data.astype(np.complex64)
    
    data_size_mb = test_data.nbytes / (1024*1024)
    print(f"Test data size: {data_size}, {data_size_mb:.1f} MB")
    
    # Benchmark processing methods
    print("\n2. Benchmarking Processing Methods")
    print("-" * 30)
    
    try:
        benchmarks = optimizer.benchmark_processing_methods(test_data, iterations=3)
        
        print("\nBenchmark Results:")
        for method, results in benchmarks.items():
            print(f"{method:15}: {results['avg_time']:.3f}s (±{results['std_time']:.3f}s), "
                  f"Memory: {results['avg_memory_mb']:.1f} MB")
    
    except Exception as e:
        print(f"Benchmarking failed: {e}")
        benchmarks = {}
    
    # Memory optimization demo
    print("\n3. Memory Optimization Demo")
    print("-" * 30)
    
    # Save test data to HDF5 for memory optimization demo
    test_file = "test_data.h5"
    with h5py.File(test_file, 'w') as f:
        f.create_dataset('sar_data', data=test_data, compression='gzip')
    
    try:
        memory_result = optimizer.optimize_memory_usage(
            test_file, "optimized_output.h5", target_memory_mb=500
        )
        print(f"Memory optimization result: {memory_result}")
    except Exception as e:
        print(f"Memory optimization failed: {e}")
    
    # Performance profiling demo
    print("\n4. Performance Profiling Demo")
    print("-" * 30)
    
    try:
        with optimizer.profiler.profile() as profiler:
            # Perform some processing
            result = optimizer.optimized_sla.process_optimized(test_data[:500, :500])
        
        print(f"Profiling completed:")
        print(f"  Execution time: {profiler.metrics.execution_time:.3f}s")
        print(f"  Memory usage: {profiler.metrics.memory_usage_mb:.1f} MB")
        print(f"  Peak memory: {profiler.metrics.peak_memory_mb:.1f} MB")
        
        # Print top CPU functions
        print("\nTop CPU-intensive functions:")
        cpu_stats = profiler.get_cpu_profile_stats()
        # Print first few lines of CPU stats
        stats_lines = cpu_stats.split('\n')[:15]
        for line in stats_lines:
            if line.strip():
                print(f"  {line}")
    
    except Exception as e:
        print(f"Performance profiling failed: {e}")
    
    # Cache performance demo
    print("\n5. Cache Performance Demo")
    print("-" * 30)
    
    cache_manager = optimizer.cache_manager
    
    # Test cache performance
    test_key = "test_array"
    test_array = np.random.randn(100, 100)
    
    # Cache the array
    success = cache_manager.put(test_key, test_array)
    print(f"Cached array: {success}")
    
    # Retrieve from cache
    start_time = time.time()
    cached_array = cache_manager.get(test_key)
    cache_time = time.time() - start_time
    
    print(f"Cache retrieval time: {cache_time*1000:.3f} ms")
    print(f"Cache stats: {cache_manager.get_stats()}")
    
    # Generate performance report
    print("\n6. Generating Performance Report")
    print("-" * 30)
    
    if benchmarks:
        try:
            report_path = optimizer.generate_performance_report(benchmarks)
            print(f"Performance report generated: {report_path}")
        except Exception as e:
            print(f"Report generation failed: {e}")
    
    # Cleanup
    print("\n7. Cleanup")
    print("-" * 30)
    
    # Clean up test files
    for file_path in ["test_data.h5", "optimized_output.h5"]:
        if Path(file_path).exists():
            Path(file_path).unlink()
            print(f"Cleaned up: {file_path}")
    
    print("\nPerformance Optimization Demo Complete!")
    
    # Print optimization recommendations
    print("\nOptimization Recommendations:")
    print("- Use GPU acceleration for large datasets (if available)")
    print("- Enable Numba for CPU-intensive computations")
    print("- Implement chunked processing for memory-limited environments")
    print("- Use parallel processing for batch operations")
    print("- Monitor memory usage and implement appropriate caching strategies")


if __name__ == "__main__":
    main()
