#!/bin/bash
set -e

# Ensure we are running from the project root
cd "$(dirname "$0")/.."

# Download Llama 3.2 1B Instruct Q40
echo "⬇️  Downloading Llama 3.2 1B..."
python3 launch.py llama3_2_1b_instruct_q40 -skip-run -y

# Download Llama 3.2 3B Instruct Q40
echo "⬇️  Downloading Llama 3.2 3B..."
python3 launch.py llama3_2_3b_instruct_q40 -skip-run -y

# Download Llama 3.1 8B Instruct Q40
echo "⬇️  Downloading Llama 3.1 8B..."
python3 launch.py llama3_1_8b_instruct_q40 -skip-run -y

echo "✅ All models downloaded successfully."
