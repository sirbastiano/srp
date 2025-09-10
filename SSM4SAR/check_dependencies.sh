#!/bin/bash
# Installation and dependency check script for SSM4SAR

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Environment path
VENV_PATH='/Data_large/marine/PythonProjects/SAR/sarpyx/.venv'
PROJECT_DIR='/Data_large/marine/PythonProjects/SAR/sarpyx/SSM4SAR'

echo -e "${BLUE}SSM4SAR Dependency Check and Installation${NC}"
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}Error: Virtual environment not found at $VENV_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Virtual environment found${NC}"

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Check Python packages
echo -e "\n${BLUE}Checking Python packages...${NC}"

packages=(
    "torch"
    "lightning" 
    "wandb"
    "einops"
    "opt_einsum"
    "kornia"
    "zarr"
    "xarray"
    "numpy"
    "scipy"
    "scikit-image"
    "tensorboard"
)

missing_packages=()

for package in "${packages[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo -e "${GREEN}✓ $package${NC}"
    else
        echo -e "${RED}✗ $package${NC}"
        missing_packages+=("$package")
    fi
done

# Install missing packages
if [ ${#missing_packages[@]} -gt 0 ]; then
    echo -e "\n${YELLOW}Installing missing packages...${NC}"
    for package in "${missing_packages[@]}"; do
        echo -e "${BLUE}Installing $package...${NC}"
        pip install "$package"
    done
else
    echo -e "\n${GREEN}All Python packages are installed!${NC}"
fi

# Check for Nextflow
echo -e "\n${BLUE}Checking Nextflow...${NC}"
if command -v nextflow &> /dev/null; then
    echo -e "${GREEN}✓ Nextflow is installed${NC}"
    nextflow -version
else
    echo -e "${RED}✗ Nextflow not found${NC}"
    echo -e "${YELLOW}Installing Nextflow...${NC}"
    
    # Check Java
    if command -v java &> /dev/null; then
        java_version=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}')
        echo -e "${GREEN}✓ Java version: $java_version${NC}"
        
        # Install Nextflow
        curl -s https://get.nextflow.io | bash
        
        # Move to a location in PATH
        if [ -w "/usr/local/bin" ]; then
            sudo mv nextflow /usr/local/bin/
            echo -e "${GREEN}✓ Nextflow installed to /usr/local/bin${NC}"
        else
            mkdir -p ~/bin
            mv nextflow ~/bin/
            echo 'export PATH=$PATH:~/bin' >> ~/.bashrc
            echo 'export PATH=$PATH:~/bin' >> ~/.zshrc
            echo -e "${GREEN}✓ Nextflow installed to ~/bin${NC}"
            echo -e "${YELLOW}Note: You may need to restart your shell or run 'source ~/.bashrc'${NC}"
        fi
    else
        echo -e "${RED}✗ Java is required for Nextflow${NC}"
        echo -e "${YELLOW}Please install Java 11 or later${NC}"
    fi
fi

# Check GPU availability
echo -e "\n${BLUE}Checking GPU availability...${NC}"
if python -c "import torch; print('✓ CUDA available:', torch.cuda.is_available()); print('✓ GPU count:', torch.cuda.device_count())" 2>/dev/null; then
    echo -e "${GREEN}GPU check completed${NC}"
else
    echo -e "${YELLOW}Could not check GPU availability${NC}"
fi

# Check data directories
echo -e "\n${BLUE}Checking data directories...${NC}"
TRAIN_DIR="$PROJECT_DIR/maya4_data/training"
VAL_DIR="$PROJECT_DIR/maya4_data/validation"

if [ -d "$TRAIN_DIR" ]; then
    echo -e "${GREEN}✓ Training data directory exists${NC}"
    echo "  Files: $(find "$TRAIN_DIR" -name "*.zarr" | wc -l) zarr files"
else
    echo -e "${RED}✗ Training data directory not found: $TRAIN_DIR${NC}"
fi

if [ -d "$VAL_DIR" ]; then
    echo -e "${GREEN}✓ Validation data directory exists${NC}"
    echo "  Files: $(find "$VAL_DIR" -name "*.zarr" | wc -l) zarr files"
else
    echo -e "${RED}✗ Validation data directory not found: $VAL_DIR${NC}"
fi

# W&B login check
echo -e "\n${BLUE}Checking Weights & Biases...${NC}"
if python -c "import wandb; print('✓ wandb imported successfully')" 2>/dev/null; then
    if wandb status 2>/dev/null | grep -q "Logged in"; then
        echo -e "${GREEN}✓ W&B logged in${NC}"
    else
        echo -e "${YELLOW}⚠ W&B not logged in${NC}"
        echo -e "${BLUE}To login to W&B, run: wandb login${NC}"
    fi
else
    echo -e "${RED}✗ W&B import failed${NC}"
fi

echo -e "\n${BLUE}=============================================="
echo -e "Dependency check completed!${NC}"
echo -e "\n${BLUE}Quick start commands:${NC}"
echo -e "${GREEN}# Single training run:${NC}"
echo -e "cd $PROJECT_DIR"
echo -e "./run_train.sh --epochs 15 --wandb"
echo -e "\n${GREEN}# Parameter sweep:${NC}"
echo -e "nextflow run sweep.nf --outdir results/sweep_\$(date +%Y%m%d)"
echo -e "\n${GREEN}# W&B login:${NC}"
echo -e "wandb login"
