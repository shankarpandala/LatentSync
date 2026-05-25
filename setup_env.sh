#!/bin/bash

# Create a new conda environment
conda create -y -n latentsync python=3.10.13
conda activate latentsync

# Install ffmpeg
conda install -y -c conda-forge ffmpeg

# Detect platform and install matching Python dependencies.
#   - Apple Silicon (arm64 macOS) uses MPS-compatible wheels (CPU-only ONNX, no CUDA torch).
#   - Linux / other platforms install the CUDA-enabled wheels listed in requirements.txt.
UNAME_S="$(uname -s)"
UNAME_M="$(uname -m)"
if [ "$UNAME_S" = "Darwin" ] && [ "$UNAME_M" = "arm64" ]; then
    echo "Detected Apple Silicon (arm64 macOS). Installing MPS-compatible requirements."
    pip install -r requirements_mps.txt
else
    pip install -r requirements.txt
    # OpenCV dependencies (Linux only)
    if [ "$UNAME_S" = "Linux" ]; then
        sudo apt -y install libgl1
    fi
fi

# Download the checkpoints required for inference from HuggingFace
huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints
huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints
