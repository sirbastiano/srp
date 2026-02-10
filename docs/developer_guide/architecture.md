# Architecture Guide

This document provides an overview of sarpyx's architecture, design patterns, and implementation details for developers.

## Overview

sarpyx is designed as a modular SAR processing library with the following core principles:

- **Modularity**: Each component has a clear, well-defined responsibility
- **Extensibility**: Easy to add new algorithms and processing methods
- **Performance**: Optimized for large-scale SAR data processing
- **Integration**: Seamless integration with existing SAR tools (SNAP, GDAL)
- **Usability**: Intuitive API for both beginners and experts

## Architecture Overview

```
sarpyx Architecture

┌─────────────────────────────────────────────────────────────┐
│                    Public API Layer                        │
├─────────────────────────────────────────────────────────────┤
│  sla/        │  snap/       │  science/    │  utils/        │
│  (Sub-Look   │  (SNAP       │  (Scientific │  (Utilities)   │
│   Analysis)  │   Integration)│   Algorithms)│                │
├─────────────────────────────────────────────────────────────┤
│                   Core Processing Layer                     │
├─────────────────────────────────────────────────────────────┤
│                    I/O and Data Layer                       │
├─────────────────────────────────────────────────────────────┤
│                 External Dependencies                       │
│  NumPy  │ SciPy │ GDAL │ SNAP │ Matplotlib │ Numba │ ...   │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

### 1. Sub-Look Analysis (`sla/`)

Core module for sub-look decomposition and analysis.

```python
sla/
├── __init__.py           # Public API exports
├── core.py              # Main SubLookAnalyzer class
├── decomposition.py     # Decomposition algorithms
├── coherence.py         # Coherence estimation
├── statistics.py        # Statistical analysis
└── validation.py        # Quality assessment
```

**Key Classes:**

- `SubLookAnalyzer`: Main interface for sub-look analysis
- `DecompositionResult`: Container for decomposition outputs
- `CoherenceEstimator`: Coherence calculation algorithms
- `StatisticalAnalyzer`: Statistical metrics and validation

**Design Patterns:**

```python
# Factory pattern for different decomposition methods
class DecompositionFactory:
    @staticmethod
    def create_decomposer(method: str) -> BaseDecomposer:
        if method == "frequency":
            return FrequencyDecomposer()
        elif method == "spatial":
            return SpatialDecomposer()
        # ...

# Strategy pattern for different coherence estimators
class CoherenceEstimator:
    def __init__(self, strategy: CoherenceStrategy):
        self._strategy = strategy
    
    def estimate(self, data1, data2):
        return self._strategy.calculate(data1, data2)
```

### 2. SNAP Integration (`snap/`)

Interface layer for ESA SNAP toolkit integration.

```python
snap/
├── __init__.py           # Public API exports
├── workflows.py         # Pre-defined processing workflows
├── operators.py         # Individual SNAP operators
├── graph_builder.py     # Processing graph construction
└── batch_processing.py  # Batch processing utilities
```

**Key Classes:**

- `SNAPWorkflow`: High-level workflow orchestration
- `GraphBuilder`: Dynamic processing graph creation
- `OperatorWrapper`: Python wrapper for SNAP operators
- `BatchProcessor`: Large-scale batch processing

**Architecture:**

```python
# Command pattern for SNAP operations
class SNAPOperation:
    def __init__(self, operator: str, parameters: Dict):
        self.operator = operator
        self.parameters = parameters
    
    def execute(self, input_product):
        # Execute SNAP operation
        pass

# Builder pattern for processing graphs
class GraphBuilder:
    def __init__(self):
        self._operations = []
    
    def add_operation(self, operation: SNAPOperation):
        self._operations.append(operation)
        return self
    
    def build(self) -> ProcessingGraph:
        return ProcessingGraph(self._operations)
```

### 3. Science Algorithms (`science/`)

Scientific algorithms and indices for SAR analysis.

```python
science/
├── __init__.py           # Public API exports
├── indices.py           # Vegetation and polarimetric indices
├── polarimetry.py       # Polarimetric decomposition
├── interferometry.py    # InSAR processing
├── target_detection.py  # Target detection algorithms
└── time_series.py       # Time series analysis
```

**Key Classes:**

- `VegetationIndices`: Vegetation monitoring algorithms
- `PolarimetricDecomposer`: H/A/α, Freeman-Durden, etc.
- `InterferogramProcessor`: InSAR processing
- `CFARDetector`: Target detection algorithms
- `TimeSeriesAnalyzer`: Temporal analysis

### 4. Utilities (`utils/`)

Common utilities and helper functions.

```python
utils/
├── __init__.py           # Public API exports
├── io.py                # Data I/O operations
├── visualization.py     # Plotting and visualization
├── validation.py        # Data validation
├── performance.py       # Performance monitoring
└── config.py            # Configuration management
```

## Data Flow Architecture

### 1. Input Processing

```python
# Data ingestion pipeline
Input Data (SAR Product)
    ↓
Format Detection (SAFE, GeoTIFF, etc.)
    ↓
Metadata Extraction
    ↓
Data Validation
    ↓
Memory Mapping (for large files)
    ↓
Ready for Processing
```

### 2. Processing Pipeline

```python
# Configurable processing pipeline
class ProcessingPipeline:
    def __init__(self):
        self._stages = []
    
    def add_stage(self, stage: ProcessingStage):
        self._stages.append(stage)
    
    def execute(self, data: SARData) -> ProcessingResult:
        for stage in self._stages:
            data = stage.process(data)
        return data
```

### 3. Output Handling

```python
# Results management
Processing Result
    ↓
Format Conversion (if needed)
    ↓
Metadata Preservation
    ↓
Quality Assessment
    ↓
Output Writing
    ↓
Final Product
```

## Performance Architecture

### 1. Memory Management

- **Lazy Loading**: Data loaded only when needed
- **Memory Mapping**: Large files processed without full loading
- **Chunked Processing**: Large datasets processed in chunks
- **Garbage Collection**: Explicit memory cleanup

```python
class MemoryEfficientProcessor:
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
    
    def process_large_array(self, data: np.ndarray) -> np.ndarray:
        # Process in chunks to avoid memory issues
        result = np.empty_like(data)
        
        for i in range(0, data.shape[0], self.chunk_size):
            chunk = data[i:i+self.chunk_size]
            result[i:i+self.chunk_size] = self._process_chunk(chunk)
            
            # Explicit cleanup
            del chunk
            gc.collect()
        
        return result
```

### 2. Parallel Processing

- **Thread-level parallelism**: For I/O bound operations
- **Process-level parallelism**: For CPU-intensive tasks
- **Numba acceleration**: JIT compilation for hot paths
- **GPU acceleration**: CUDA support for applicable algorithms

```python
from concurrent.futures import ProcessPoolExecutor
from numba import jit, cuda

# CPU parallelization
def parallel_processing(data_chunks):
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_chunk, data_chunks))
    return np.concatenate(results)

# GPU acceleration
@cuda.jit
def gpu_coherence_kernel(data1, data2, result):
    # CUDA kernel for coherence calculation
    pass
```

### 3. Caching Strategy

- **Result caching**: Cache expensive computations
- **Disk caching**: Temporary results stored on disk
- **Memory caching**: Frequently accessed data in memory

```python
from functools import lru_cache
from diskcache import Cache

# Memory caching
@lru_cache(maxsize=128)
def expensive_computation(params):
    # Expensive calculation
    pass

# Disk caching
cache = Cache('/tmp/sarpyx_cache')

@cache.memoize(expire=3600)  # 1 hour expiry
def very_expensive_computation(data):
    # Very expensive calculation
    pass
```

## Error Handling Architecture

### 1. Exception Hierarchy

```python
class sarpyxError(Exception):
    """Base exception for sarpyx."""
    pass

class DataError(sarpyxError):
    """Data-related errors."""
    pass

class ProcessingError(sarpyxError):
    """Processing-related errors."""
    pass

class ValidationError(sarpyxError):
    """Validation-related errors."""
    pass

class SNAPError(sarpyxError):
    """SNAP integration errors."""
    pass
```

### 2. Error Handling Patterns

```python
# Context manager for resource cleanup
class SARDataProcessor:
    def __enter__(self):
        self._allocate_resources()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_resources()
        if exc_type is not None:
            self._log_error(exc_type, exc_val, exc_tb)

# Retry pattern for unreliable operations
def with_retry(max_attempts=3, delay=1.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except RetryableError as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay * (2 ** attempt))
        return wrapper
    return decorator
```

## Configuration Architecture

### 1. Configuration Management

```python
# Hierarchical configuration
config/
├── default.yaml         # Default settings
├── development.yaml     # Development overrides
├── production.yaml      # Production overrides
└── user.yaml           # User customizations
```

### 2. Configuration Loading

```python
class ConfigManager:
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        # Load default configuration
        config = load_yaml("config/default.yaml")
        
        # Apply environment-specific overrides
        env = os.getenv("sarpyx_ENV", "development")
        env_config = load_yaml(f"config/{env}.yaml")
        config.update(env_config)
        
        # Apply user customizations
        user_config_path = os.path.expanduser("~/.sarpyx/config.yaml")
        if os.path.exists(user_config_path):
            user_config = load_yaml(user_config_path)
            config.update(user_config)
        
        return config
```

## Testing Architecture

### 1. Test Structure

```python
# Test categories
Unit Tests:     Individual function/method testing
Integration Tests: Module interaction testing
End-to-End Tests: Complete workflow testing
Performance Tests: Benchmarking and profiling
Regression Tests: Preventing feature regressions
```

### 2. Test Data Management

```python
# Test data fixtures
@pytest.fixture(scope="session")
def sample_sar_data():
    """Create or load sample SAR data for testing."""
    data_path = Path(__file__).parent / "data" / "sample.safe"
    if not data_path.exists():
        # Generate synthetic data
        data = generate_synthetic_sar_data()
        save_sar_data(data, data_path)
    return load_sar_data(data_path)

# Property-based testing
from hypothesis import given, strategies as st

@given(
    data=st.arrays(np.complex128, shape=(100, 100)),
    n_sublooks=st.integers(min_value=2, max_value=16)
)
def test_sublook_decomposition_properties(data, n_sublooks):
    """Test properties that should always hold."""
    result = sublook_decomposition(data, n_sublooks)
    
    # Property: number of sublooks should match input
    assert result.sublooks.shape[0] == n_sublooks
    
    # Property: coherence should be between 0 and 1
    assert np.all(0 <= result.coherence <= 1)
```

## Extensibility Architecture

### 1. Plugin System

```python
# Plugin interface
class ProcessingPlugin:
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def process(self, data: SARData) -> ProcessingResult:
        pass

# Plugin registry
class PluginRegistry:
    def __init__(self):
        self._plugins = {}
    
    def register(self, plugin: ProcessingPlugin):
        self._plugins[plugin.name()] = plugin
    
    def get_plugin(self, name: str) -> ProcessingPlugin:
        return self._plugins.get(name)
```

### 2. Algorithm Registration

```python
# Algorithm registry for extensible processing
class AlgorithmRegistry:
    _algorithms = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(algorithm_class):
            cls._algorithms[name] = algorithm_class
            return algorithm_class
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs):
        if name not in cls._algorithms:
            raise ValueError(f"Unknown algorithm: {name}")
        return cls._algorithms[name](**kwargs)

# Usage
@AlgorithmRegistry.register("my_decomposition")
class MyDecompositionAlgorithm:
    def __init__(self, param1=None):
        self.param1 = param1
    
    def decompose(self, data):
        # Custom implementation
        pass
```

This architecture provides a solid foundation for maintainable, extensible, and high-performance SAR processing capabilities while maintaining clean separation of concerns and easy testability.
