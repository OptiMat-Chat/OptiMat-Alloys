#!/bin/bash
set -e

# Copy .env.example if .env doesn't exist
if [ ! -f /app/.env ]; then
    cp /app/.env.example /app/.env 2>/dev/null || true
fi

# Ensure persistent directories exist
mkdir -p /app/cache/nequip
mkdir -p /app/structures

# Start Ollama server in background
ollama serve &
for i in $(seq 1 10); do
    if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
        echo "Ollama server ready"
        break
    fi
    sleep 1
done

# Start Xvfb for headless OVITO rendering if no display available
if [ -z "$DISPLAY" ]; then
    Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp &
    export DISPLAY=:99
    sleep 1
fi

echo ""
echo "============================================"
echo "  OptiMat Alloys is starting..."
echo "  Open in your browser: http://localhost:8000"
echo "============================================"
echo ""

exec "$@"
