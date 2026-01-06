#!/bin/bash
set -e

echo "🚀 Starting Distributed Llama Container..."

# 1. Prepare Persistent Volume
# RunPod typically uses /workspace as the persistent volume mount point.
if [ -d "/workspace" ]; then
    echo "💾 Volume detected at /workspace"
    mkdir -p /workspace/models
    
    # If /app/models exists as a directory (from build), replace it with a symlink
    if [ -d "/app/models" ] && [ ! -L "/app/models" ]; then
        echo "🔗 Linking /app/models -> /workspace/models"
        rm -rf /app/models
        ln -s /workspace/models /app/models
    elif [ ! -e "/app/models" ]; then
        echo "🔗 Linking /app/models -> /workspace/models"
        ln -s /workspace/models /app/models
    fi
else
    echo "⚠️  No persistent volume found at /workspace. Using local /app/models (Data will be lost on restart!)"
    mkdir -p /app/models
fi

# 2. Check for Models and Auto-download
# If the models directory is empty, run the download script.
if [ -z "$(ls -A /app/models)" ]; then
    echo "📥 No models found. Downloading default models..."
    if [ -f "./prepare_models.sh" ]; then
        ./prepare_models.sh
    else
        echo "❌ prepare_models.sh not found!"
    fi
else
    echo "✅ Models detected in /app/models"
fi

# 3. Execute Command
# If arguments are provided (e.g., ./hpipe-worker ...), execute them.
# Otherwise, start a bash shell.
if [ "$#" -eq 0 ]; then
    exec /bin/bash
else
    exec "$@"
fi
