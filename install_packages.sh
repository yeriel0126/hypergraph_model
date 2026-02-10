#!/bin/bash
# Installation script for Hyperbolic Hypergraph Model

echo "Installing required packages for Hyperbolic Hypergraph Model..."
echo ""

# Core packages
echo "1. Installing core packages..."
pip install torch torch-geometric geoopt ogb fasttext numpy tqdm scikit-learn

# Try to install torch-scatter (optional)
echo ""
echo "2. Attempting to install torch-scatter (optional)..."
echo "   If this fails, the model will still work without it."

# Detect PyTorch version and CUDA
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null | cut -d'+' -f1)
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'cpu')" 2>/dev/null)

echo "   Detected PyTorch version: $PYTORCH_VERSION"
echo "   Detected CUDA: $CUDA_VERSION"

# Try installing from PyG wheels
if [ "$CUDA_VERSION" != "None" ] && [ "$CUDA_VERSION" != "cpu" ]; then
    echo "   Attempting CUDA installation..."
    pip install torch-scatter -f https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION}.html || echo "   Failed to install torch-scatter with CUDA, trying CPU version..."
fi

# Fallback to CPU version
pip install torch-scatter -f https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+cpu.html || echo "   Warning: Could not install torch-scatter. The model will work without it."

echo ""
echo "Installation complete!"
echo ""
echo "To verify installation, run:"
echo "  python -c \"import torch; import torch_geometric; import geoopt; print('All packages installed successfully!')\""

