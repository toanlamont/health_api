#!/bin/sh
# Install torch matching the machine: CUDA build if an NVIDIA GPU is present,
# otherwise the small CPU-only build. Override with TORCH_VARIANT=cpu|cu126.
set -eu

variant="${TORCH_VARIANT:-auto}"
if [ "$variant" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        variant=cu126
    else
        variant=cpu
    fi
fi
echo "Installing torch (variant: $variant)"

index="https://download.pytorch.org/whl/$variant"
if command -v uv >/dev/null 2>&1; then
    uv pip install torch --index-url "$index"
else
    pip install --no-cache-dir torch --index-url "$index"
fi
