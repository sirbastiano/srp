# Contributing to sarpyx

Thank you for your interest in contributing to sarpyx! This guide will help you get started with developing and contributing to the library.

## Quick Start for Contributors

### 1. Development Setup

```bash
# Clone the repository
git clone https://github.com/ESA-sarpyx/sarpyx.git
cd sarpyx

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 2. Development Workflow

1. **Fork and Clone**: Fork the repository and clone your fork
2. **Branch**: Create a feature branch from `main`
3. **Develop**: Make your changes with tests
4. **Test**: Run the test suite
5. **Document**: Update documentation if needed
6. **Submit**: Create a pull request

## Code Standards

### Code Style

We follow PEP 8 with some modifications:
- Line length: 88 characters (Black default)
- Use type hints for all public functions
- Use docstrings for all public classes and functions

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### Docstring Format

We use NumPy style docstrings:

```python
def sub_look_decomposition(
    data: np.ndarray,
    n_sublooks: int = 8,
    overlap: float = 0.5,
    window_type: str = "hann"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform sub-look decomposition of SAR data.

    Parameters
    ----------
    data : np.ndarray
        Input SAR data array with shape (height, width).
    n_sublooks : int, optional
        Number of sub-looks to generate, by default 8.
    overlap : float, optional
        Overlap between sub-looks as fraction, by default 0.5.
    window_type : str, optional
        Window function type, by default "hann".

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Sub-look data and coherence maps.

    Raises
    ------
    ValueError
        If n_sublooks is less than 2.
    TypeError
        If data is not a numpy array.

    Examples
    --------
    >>> data = np.random.complex128((1000, 1000))
    >>> sublooks, coherence = sub_look_decomposition(data, n_sublooks=4)
    >>> print(sublooks.shape)
    (4, 1000, 1000)
    """
```

## Testing

### Test Structure

```
tests/
├── conftest.py          # pytest configuration and fixtures
├── test_sla/           # Sub-look analysis tests
│   ├── test_decomposition.py
│   └── test_coherence.py
├── test_snap/          # SNAP integration tests
│   ├── test_workflows.py
│   └── test_operators.py
├── test_science/       # Science modules tests
│   ├── test_indices.py
│   └── test_polarimetry.py
└── test_utils/         # Utilities tests
    ├── test_io.py
    └── test_visualization.py
```

### Writing Tests

Use pytest for all tests:

```python
import pytest
import numpy as np
from sarpyx.sla import SubLookAnalyzer

class TestSubLookAnalyzer:
    @pytest.fixture
    def sample_data(self):
        """Create sample SAR data for testing."""
        return np.random.complex128((100, 100))
    
    def test_decomposition_basic(self, sample_data):
        """Test basic sub-look decomposition."""
        analyzer = SubLookAnalyzer(n_sublooks=4)
        result = analyzer.decompose(sample_data)
        
        assert result.sublooks.shape[0] == 4
        assert result.coherence.shape == sample_data.shape
    
    @pytest.mark.parametrize("n_sublooks", [2, 4, 8, 16])
    def test_decomposition_different_sublooks(self, sample_data, n_sublooks):
        """Test decomposition with different numbers of sub-looks."""
        analyzer = SubLookAnalyzer(n_sublooks=n_sublooks)
        result = analyzer.decompose(sample_data)
        
        assert result.sublooks.shape[0] == n_sublooks
    
    def test_invalid_input(self):
        """Test handling of invalid inputs."""
        analyzer = SubLookAnalyzer(n_sublooks=4)
        
        with pytest.raises(TypeError):
            analyzer.decompose("not_an_array")
        
        with pytest.raises(ValueError):
            SubLookAnalyzer(n_sublooks=1)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_sla/test_decomposition.py

# Run with coverage
pytest --cov=sarpyx --cov-report=html

# Run with verbose output
pytest -v

# Run tests for specific module
pytest tests/test_sla/
```

## Documentation

### Building Documentation

We use Sphinx for documentation generation:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html

# Serve documentation locally
python -m http.server 8000 -d _build/html
```

### Documentation Guidelines

1. **Update docstrings** for any new or modified functions
2. **Add examples** to the examples directory
3. **Update tutorials** if adding new features
4. **Update API documentation** for new modules
5. **Test documentation examples** to ensure they work

### Adding Examples

When adding examples:

1. Place in appropriate directory (`basic/`, `intermediate/`, `advanced/`)
2. Include comprehensive docstrings
3. Add error handling
4. Include visualization when relevant
5. Update the examples README

## Module Development

### Adding New Modules

1. **Create module directory** under `sarpyx/`
2. **Add `__init__.py`** with public API
3. **Create implementation files**
4. **Add comprehensive tests**
5. **Update main `__init__.py`**
6. **Add documentation**

Example structure for new module:

```
sarpyx/
└── new_module/
    ├── __init__.py
    ├── core.py
    ├── algorithms.py
    └── utils.py
```

### API Design Guidelines

1. **Consistent naming**: Use descriptive, consistent function names
2. **Type hints**: Always include type hints
3. **Default parameters**: Provide sensible defaults
4. **Error handling**: Include proper error messages
5. **Documentation**: Comprehensive docstrings with examples

## Performance Guidelines

### Memory Management

```python
# Good: Use memory-efficient operations
def process_large_data(data: np.ndarray) -> np.ndarray:
    # Process in chunks to avoid memory issues
    chunk_size = 1000
    result = np.empty_like(data)
    
    for i in range(0, data.shape[0], chunk_size):
        end_idx = min(i + chunk_size, data.shape[0])
        chunk = data[i:end_idx]
        result[i:end_idx] = process_chunk(chunk)
    
    return result

# Bad: Load everything into memory
def process_large_data_bad(data: np.ndarray) -> np.ndarray:
    # Creates unnecessary copies
    temp1 = data.copy()
    temp2 = temp1 * 2
    temp3 = temp2 + 1
    return temp3
```

### Numba Optimization

Use Numba for performance-critical functions:

```python
from numba import jit, prange
import numpy as np

@jit(nopython=True, parallel=True)
def fast_coherence_calculation(
    data1: np.ndarray, 
    data2: np.ndarray,
    window_size: int
) -> np.ndarray:
    """Fast coherence calculation using Numba."""
    height, width = data1.shape
    coherence = np.zeros((height, width), dtype=np.float64)
    
    for i in prange(window_size//2, height - window_size//2):
        for j in range(window_size//2, width - window_size//2):
            # Calculate coherence for window
            window1 = data1[i-window_size//2:i+window_size//2+1,
                           j-window_size//2:j+window_size//2+1]
            window2 = data2[i-window_size//2:i+window_size//2+1,
                           j-window_size//2:j+window_size//2+1]
            
            coherence[i, j] = calculate_window_coherence(window1, window2)
    
    return coherence
```

## Release Process

### Version Management

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Creating a Release

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md**
3. **Create release branch**
4. **Run full test suite**
5. **Build documentation**
6. **Create GitHub release**
7. **Upload to PyPI**

```bash
# Build package
python -m build

# Upload to PyPI (test first)
python -m twine upload --repository testpypi dist/*
python -m twine upload dist/*
```

## Getting Help

- **Issues**: Report bugs and request features on GitHub Issues
- **Discussions**: Join GitHub Discussions for questions
- **Email**: Contact maintainers at roberto.delprete@esa.int
- **Documentation**: Check the full documentation at [docs](../README.md)

## Code of Conduct

Please read and follow our [Code of Conduct](code_of_conduct.md) to ensure a welcoming environment for all contributors.
