# Installing s1isp for SAR Decoding

The `s1isp` package is required for SAR decoding functionality but requires compilation of C extensions.

## Prerequisites

You need Python development headers and build tools installed:

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install python3.12-dev python3-dev build-essential
```

### CentOS/RHEL/Fedora
```bash
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel python3.12-devel
```

### macOS
```bash
# Install Xcode command line tools
xcode-select --install
```

## Installation

Once you have the prerequisites:

### Option 1: Using uv (recommended)
```bash
# Install the SAR decoding optional dependencies
uv pip install sarpyx[sar-decode]

# Or install s1isp separately
uv add git+https://github.com/avalentino/s1isp.git
```

### Option 2: Using pip
```bash
pip install "sarpyx[sar-decode]"

# Or install s1isp separately
pip install git+https://github.com/avalentino/s1isp.git
```

### Option 3: Manual build (if you don't have sudo access)

If you can't install system packages, you might be able to use a conda environment or ask your system administrator to install the development packages.

## Verification

Test that s1isp is properly installed:

```python
import s1isp
print("s1isp successfully installed!")
```

## Troubleshooting

- **"Python.h not found"**: Install python3-dev/python3.12-dev
- **"gcc not found"**: Install build-essential or Development Tools
- **Permission denied**: Use `sudo` or ask your system administrator
- **No sudo access**: Try using conda/mamba or contact your administrator

## Alternative Solutions

If you still can't install s1isp, SAR decoding functionality will not be available, but all other sarpyx features will work normally.