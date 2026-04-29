#!/bin/bash

# Fix CUDA 12.4 Installation - libtinfo5 Issue
# This script fixes the missing libtinfo5 dependency

set -e

echo "================================================"
echo "Fixing CUDA Installation Dependencies"
echo "================================================"
echo ""

# Check Ubuntu version
echo "Checking Ubuntu version..."
. /etc/os-release
echo "Ubuntu version: $VERSION"
echo ""

# Solution 1: Install libtinfo5 from Ubuntu 20.04 repository
echo "[1/3] Installing libtinfo5..."

if [ "$VERSION_ID" = "22.04" ] || [ "$VERSION_ID" = "24.04" ]; then
    echo "  Downloading libtinfo5 from Ubuntu 20.04 repository..."
    wget http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.2-0ubuntu2_amd64.deb
    sudo dpkg -i libtinfo5_6.2-0ubuntu2_amd64.deb
    rm libtinfo5_6.2-0ubuntu2_amd64.deb
    echo "  ✓ libtinfo5 installed"
else
    echo "  Attempting to install libtinfo5 from repositories..."
    sudo apt install -y libtinfo5 || echo "  ⚠ Could not install from repos, trying alternative..."
fi

# Solution 2: Install CUDA toolkit without problematic packages
echo ""
echo "[2/3] Installing CUDA 12.4 toolkit (excluding nsight-systems)..."
sudo apt install -y \
    cuda-toolkit-12-4 \
    --no-install-recommends \
    -o Dpkg::Options::="--force-overwrite" || {

    echo ""
    echo "  Alternative: Installing core CUDA components only..."
    sudo apt install -y \
        cuda-compiler-12-4 \
        cuda-libraries-12-4 \
        cuda-libraries-dev-12-4 \
        cuda-command-line-tools-12-4 \
        cuda-nvcc-12-4 \
        cuda-cudart-12-4 \
        cuda-cudart-dev-12-4
}

# Configure environment
echo ""
echo "[3/3] Configuring environment variables..."

if grep -q "cuda-12.4" ~/.bashrc; then
    echo "  CUDA paths already in ~/.bashrc"
else
    echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo "  ✓ Added CUDA paths to ~/.bashrc"
fi

export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# Verify
echo ""
echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""

if command -v nvcc &> /dev/null; then
    echo "✓ CUDA installed successfully:"
    nvcc --version
    echo ""
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "⚠ nvcc not found. Checking installation..."
    if [ -f /usr/local/cuda-12.4/bin/nvcc ]; then
        echo "  nvcc exists but not in PATH"
        echo "  Run: source ~/.bashrc"
    else
        echo "  Installation may have failed"
    fi
fi

echo ""
echo "Next steps:"
echo "  1. source ~/.bashrc"
echo "  2. nvcc --version"
echo "  3. ./setup_linux.sh"
echo ""
