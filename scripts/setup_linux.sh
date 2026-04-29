#!/bin/bash

# OptiMat Alloys Setup Script for Linux/WSL2
# This script creates the conda environment and installs all dependencies

set -e  # Exit on error

echo "================================================"
echo "OptiMat Alloys Setup for Linux/WSL2"
echo "================================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found!"
    echo ""
    echo "Please install Miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    echo ""
    echo "Then close and reopen your terminal, and run this script again."
    exit 1
fi

echo "Found conda installation: $(which conda)"
echo ""

# Check if environment already exists
if conda env list | grep -q "^optimat-alloys "; then
    echo "WARNING: Environment 'optimat-alloys' already exists!"
    echo ""
    echo "Options:"
    echo "  1. Remove and recreate from scratch (recommended for clean install)"
    echo "  2. Keep existing and update packages"
    echo "  3. Cancel installation"
    echo ""
    read -p "Enter choice (1/2/3): " env_choice

    case $env_choice in
        1)
            echo ""
            echo "Removing existing environment..."
            conda deactivate 2>/dev/null || true
            conda env remove -n optimat-alloys -y
            echo "Environment removed successfully."
            ;;
        2)
            echo ""
            echo "Keeping existing environment..."
            ;;
        3)
            echo ""
            echo "Installation cancelled."
            exit 0
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
fi

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "^optimat-alloys "; then
    echo ""
    echo "Creating conda environment 'optimat-alloys' with Python 3.11..."
    echo "Note: ORB models requires Python >=3.10, <3.13"
    conda create -n optimat-alloys python=3.11 -y

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create conda environment"
        exit 1
    fi
fi

# Activate environment
echo ""
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate optimat-alloys

# Verify Python version
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $python_version"

if [[ $(echo "$python_version < 3.10" | bc) -eq 1 ]]; then
    echo "ERROR: Python version is too old for ORB models"
    echo "ORB models requires Python >=3.10"
    exit 1
fi

# Ask about GPU support
echo ""
echo "Do you have an NVIDIA GPU for CUDA acceleration? (y/n)"
read -p "Enter choice: " gpu_choice

if [[ "$gpu_choice" == "y" || "$gpu_choice" == "Y" ]]; then
    echo ""
    echo "Installing PyTorch with CUDA 12.4 support..."
    echo "Note: This is compatible with CUDA 12.x-13.x"
    pip install torch>=2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

    if [ $? -ne 0 ]; then
        echo ""
        echo "WARNING: Failed to install PyTorch with CUDA"
        echo "Falling back to CPU-only PyTorch..."
        pip install torch>=2.6.0 torchvision torchaudio
    else
        echo "PyTorch with CUDA installed successfully!"

        # Check if CUDA is actually available
        if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            echo "✓ CUDA is available and working!"
        else
            echo "⚠ WARNING: PyTorch CUDA installed but no GPU detected."
            echo "Make sure NVIDIA drivers are installed and working."
        fi
    fi

    # Ask about cuML for large systems
    echo ""
    echo "Will you simulate large systems? (5000+ atoms PBC or 30000+ atoms non-PBC)"
    echo "cuML can provide 2-10x faster graph creation and 2-100x better GPU memory efficiency"
    read -p "Install cuML optimization? (y/n): " cuml_choice

    if [[ "$cuml_choice" == "y" || "$cuml_choice" == "Y" ]]; then
        echo ""
        echo "Select cuML version based on your CUDA:"
        echo "  1. cuML for CUDA 11.x (CUDA 11.4-11.8)"
        echo "  2. cuML for CUDA 12.x/13.x (CUDA 12.0-13.x)"
        echo "  3. Skip cuML installation"
        echo ""
        read -p "Enter choice (1/2/3): " cuml_ver

        case $cuml_ver in
            1)
                echo "Installing cuML for CUDA 11.x..."
                pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu11==25.2.*"
                if [ $? -ne 0 ]; then
                    echo "WARNING: cuML installation failed - continuing without it"
                    echo "You can install it later with:"
                    echo "  pip install --extra-index-url=https://pypi.nvidia.com \"cuml-cu11==25.2.*\""
                else
                    echo "cuML installed successfully!"
                fi
                ;;
            2)
                echo "Installing cuML for CUDA 12.x/13.x..."
                pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu12==25.2.*"
                if [ $? -ne 0 ]; then
                    echo "WARNING: cuML installation failed - continuing without it"
                    echo "You can install it later with:"
                    echo "  pip install --extra-index-url=https://pypi.nvidia.com \"cuml-cu12==25.2.*\""
                else
                    echo "cuML installed successfully!"
                fi
                ;;
            3)
                echo "Skipping cuML installation."
                ;;
            *)
                echo "Invalid choice. Skipping cuML."
                ;;
        esac
    fi
else
    echo ""
    echo "Installing PyTorch (CPU-only)..."
    pip install torch>=2.6.0 torchvision torchaudio
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

# Verify key packages
echo ""
echo "Verifying installation..."
python -c "
import chainlit
import ase
import orb_models
import ovito
import torch
print('✓ All core packages imported successfully!')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Package verification failed"
    exit 1
fi

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Set your OpenAI API key:"
echo "   export OPENAI_API_KEY='your-api-key-here'"
echo "   "
echo "   Or create a .env file:"
echo "   cp .env.example .env"
echo "   # Edit .env and add your API key"
echo ""
echo "2. Run the application:"
echo "   ./run_chat.sh"
echo ""
echo "Note: First run will precompute reference data (takes several hours)"
echo ""
echo "For more information:"
echo "  - README.md - Quick start guide"
echo "  - docs/WSL2_SETUP.md - WSL2-specific setup (Windows users)"
echo "  - docs/SETUP_LINUX.md - Linux-specific tips"
echo ""
