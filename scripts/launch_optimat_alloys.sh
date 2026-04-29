#!/bin/bash

# OptiMat Alloys Launcher Script for Linux/WSL2
# This script activates the conda environment and runs the application

set -e  # Exit on error

echo "================================================"
echo "OptiMat Alloys Launcher"
echo "================================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found!"
    echo "Please install Miniconda or run setup_linux.sh first."
    exit 1
fi

# Activate conda environment
echo "Activating optimat-alloys environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate optimat-alloys

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate 'optimat-alloys' environment"
    echo "Did you run setup_linux.sh first?"
    exit 1
fi

# Load .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Note: API key is now handled by Chainlit UI prompt (no environment check needed)

# Parse command line arguments
PORT=8000
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            EXTRA_ARGS="$EXTRA_ARGS --host $HOST"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo ""
    echo "WARNING: Port $PORT is already in use!"
    echo ""
    echo "Options:"
    echo "  1. Use a different port (e.g., ./launch_optimat_alloys.sh --port 8001)"
    echo "  2. Kill the process using port $PORT"
    echo "  3. Cancel"
    echo ""
    read -p "Enter choice (1/2/3): " port_choice

    case $port_choice in
        1)
            read -p "Enter new port number: " PORT
            ;;
        2)
            PID=$(lsof -t -i:$PORT)
            echo "Killing process $PID..."
            kill $PID
            sleep 2
            ;;
        3)
            exit 0
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
fi

echo ""
echo "Starting OptiMat Alloys on port $PORT..."
echo "⏳ Waiting for server to be ready..."
echo ""

# Detect WSL2 environment
IS_WSL=false
if grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
    IS_WSL=true
fi

# Start Chainlit in headless mode (no auto-browser)
chainlit run run_chat.py -h --port $PORT $EXTRA_ARGS &
CHAINLIT_PID=$!

# Poll HTTP endpoint until server is ready
MAX_WAIT=30
COUNTER=0
SERVER_READY=false

while [ $COUNTER -lt $MAX_WAIT ]; do
    if curl -s http://localhost:$PORT > /dev/null 2>&1; then
        SERVER_READY=true
        break
    fi
    sleep 1
    COUNTER=$((COUNTER + 1))
done

if [ "$SERVER_READY" = true ]; then
    echo "✅ Server is ready!"
    echo ""

    if [ "$IS_WSL" = true ]; then
        # WSL2: Try to open Windows browser
        echo "Detected WSL2 environment"
        echo "📂 Opening Windows browser..."

        if command -v powershell.exe &> /dev/null; then
            powershell.exe -Command "Start-Process 'http://localhost:$PORT'" 2>/dev/null || true
        fi

        echo ""
        echo "📋 Access the application at:"
        echo "   http://localhost:$PORT"
    else
        # Native Linux: Try xdg-open
        echo "📂 Opening browser..."

        if command -v xdg-open &> /dev/null; then
            xdg-open http://localhost:$PORT 2>/dev/null || true
        fi

        echo ""
        echo "📋 Access the application at:"
        echo "   http://localhost:$PORT"
    fi
else
    echo "⚠️  Server didn't start within ${MAX_WAIT}s"
    echo ""
    echo "Try manually accessing: http://localhost:$PORT"
    echo "Or check logs for errors"
fi

echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Wait for Chainlit process to exit
wait $CHAINLIT_PID
