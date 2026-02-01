#!/bin/bash
set -e

echo "=== proma2 (verl) Installation Script ==="
echo "Tested on: Ubuntu 22.04, NVIDIA H100 80GB, CUDA drivers 12.8"
echo ""

# Step 1: Install System Dependencies
echo "=== Step 1: Installing system dependencies ==="
sudo apt-get update
sudo apt-get install -y libc6-dev build-essential

# Step 2: Install Anaconda (if not already installed)
echo "=== Step 2: Installing Anaconda ==="
if [ ! -d "$HOME/anaconda3" ]; then
    cd ~
    curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
    bash Anaconda3-2024.10-1-Linux-x86_64.sh -b -p ~/anaconda3
    rm Anaconda3-2024.10-1-Linux-x86_64.sh
    ~/anaconda3/bin/conda init bash
else
    echo "Anaconda already installed at ~/anaconda3"
fi

source ~/anaconda3/etc/profile.d/conda.sh

# Step 3: Create Conda Environment
echo "=== Step 3: Creating conda environment 'isopo2' ==="
conda create -n isopo2 python==3.12 -y
conda activate isopo2

# Step 4: Install CUDA Toolkit from Conda
echo "=== Step 4: Installing CUDA 12.4 from conda ==="
conda install -c nvidia/label/cuda-12.4.0 cuda -y

# Step 5: Install Core Dependencies
echo "=== Step 5: Installing core dependencies ==="
cd "$(dirname "$0")"
bash scripts/install_vllm_sglang_mcore.sh

# Step 6: Fix numpy Version
echo "=== Step 6: Fixing numpy version ==="
pip install "numpy<2.0.0"

# Step 7: Install the Project
echo "=== Step 7: Installing project ==="
pip install --no-deps -e .

# Step 8: Download Model and Data
echo "=== Step 8: Downloading model and data ==="
pip install huggingface_hub[cli]
mkdir -p models
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./models/Qwen3-0.6B
mkdir -p ~/data/gsm8k
python examples/data_preprocess/gsm8k.py

# Step 9: Wandb login (optional)
echo "=== Step 9: Wandb login (optional) ==="
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
    echo "Logged into wandb"
else
    echo "WANDB_API_KEY not set, skipping wandb login"
    echo "Run 'wandb login <your-api-key>' manually if needed"
fi

echo ""
echo "=== Installation complete! ==="
echo ""
echo "To run training, use:"
echo "  source ~/anaconda3/etc/profile.d/conda.sh"
echo "  conda activate isopo2"
echo "  export CUDA_HOME=/home/ubuntu/anaconda3/envs/isopo2"
echo "  export FLASHINFER_CUDA_HOME=\$CUDA_HOME"
echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
echo "  bash run/run.sh"
