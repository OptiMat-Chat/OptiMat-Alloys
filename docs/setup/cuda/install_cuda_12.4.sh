#!/bin/bash

# CUDA 12.4 Installation Script for WSL2
# This script installs CUDA 12.4 toolkit to match PyTorch requirements

set -e  # Exit on error

echo "================================================"
echo "CUDA 12.4 Toolkit Installation for WSL2"
echo "================================================"
echo ""
echo "This script will:"
echo "  1. Download CUDA keyring"
echo "  2. Install CUDA repository"
echo "  3. Update package list"
echo "  4. Install CUDA 12.4 toolkit (~4GB download)"
echo "  5. Configure environment variables"
echo ""
echo "You will be prompted for your sudo password."
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Step 1: Clean up old keys
echo ""
echo "[1/6] Cleaning up old CUDA keys..."
sudo apt-key del 7fa2af80 2>/dev/null || echo "  No old key to remove"

# Step 2: Download CUDA keyring
echo ""
echo "[2/6] Downloading CUDA keyring..."
wget -q --show-progress https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb

# Step 3: Install keyring
echo ""
echo "[3/6] Installing CUDA keyring..."
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

# Step 4: Update package list
echo ""
echo "[4/6] Updating package list..."
sudo apt update

# Step 5: Install CUDA 12.4 toolkit
echo ""
echo "[5/6] Installing CUDA 12.4 toolkit..."
echo "  This will download ~4GB and take 5-10 minutes..."
sudo apt install -y cuda-toolkit-12-4

# Step 6: Configure environment variables
echo ""
echo "[6/6] Configuring environment variables..."

# Check if already in bashrc
if grep -q "cuda-12.4" ~/.bashrc; then
    echo "  CUDA paths already in ~/.bashrc"
else
    echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo "  Added CUDA paths to ~/.bashrc"
fi

# Apply immediately
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# Verify installation
echo ""
echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""
echo "Verifying installation..."
echo ""

if command -v nvcc &> /dev/null; then
    echo "✓ CUDA Compiler (nvcc) installed:"
    nvcc --version
    echo ""
    echo "✓ NVIDIA Driver:"
    nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader
else
    echo "✗ WARNING: nvcc not found in PATH"
    echo "  You may need to close and reopen your terminal"
fi

echo ""
echo "================================================"
echo "Next Steps:"
echo "================================================"
echo ""
echo "1. Close and reopen your terminal (to apply PATH changes)"
echo "   OR run: source ~/.bashrc"
echo ""
echo "2. Verify installation:"
echo "   nvcc --version"
echo ""
echo "3. Install OptiMat Alloys dependencies:"
echo "   cd ~/OptiMat-Alloys"
echo "   ./setup_linux.sh"
echo ""
