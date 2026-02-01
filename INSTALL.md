# Installation Instructions for proma2 (verl) with Flashinfer

These instructions set up the conda environment to run `run/run.sh` (GRPO) with flashinfer enabled.

**Tested on**: Ubuntu 22.04, NVIDIA H100 80GB, CUDA drivers 12.8

## Prerequisites

- Ubuntu 22.04 (tested)
- NVIDIA GPU with CUDA support (tested on H100 80GB)
- System CUDA drivers installed

## Step 1: Install System Dependencies

Install C development headers required for flashinfer JIT compilation:

```bash
sudo apt-get update
sudo apt-get install -y libc6-dev build-essential
```

## Step 2: Install Anaconda

```bash
cd ~
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh -b -p ~/anaconda3
source ~/anaconda3/bin/activate
~/anaconda3/bin/conda init bash
```

## Step 3: Create Conda Environment

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n isopo2 python==3.12 -y
conda activate isopo2
```

## Step 4: Install CUDA Toolkit from Conda

**IMPORTANT**: The system CUDA (12.8) may cause flashinfer compilation errors. Install CUDA 12.4 from conda:

```bash
conda install -c nvidia/label/cuda-12.4.0 cuda -y
```

## Step 5: Install Core Dependencies

Run the installation script:

```bash
bash scripts/install_vllm_sglang_mcore.sh
```

Note: TransformerEngine may fail to build due to cuDNN issues - this is OK, the core functionality still works.

## Step 6: Fix numpy Version

```bash
pip install "numpy<2.0.0"
```

## Step 7: Install the Project

```bash
pip install --no-deps -e .
```

## Step 8: Download Model and Data

```bash
pip install huggingface_hub[cli]
mkdir -p models
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./models/Qwen3-0.6B
mkdir -p ~/data/gsm8k
python examples/data_preprocess/gsm8k.py
```

## Step 9: Login to Wandb (if logging to wandb)

```bash
wandb login $WANDB_API_KEY
```

## Running the Training

**IMPORTANT**: You must set the CUDA environment variables correctly and use the conda Python explicitly.

### Option 1: Use the wrapper script

Create `run_grpo.sh`:

```bash
#!/bin/bash
set -e

# Set up conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate isopo2

# Set CUDA environment - CRITICAL for flashinfer
export CUDA_HOME=/home/ubuntu/anaconda3/envs/isopo2
export FLASHINFER_CUDA_HOME=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH

# Clear flashinfer cache if there are compilation issues
# rm -rf ~/.cache/flashinfer

# Run GRPO
/home/ubuntu/anaconda3/envs/isopo2/bin/python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=128 \
    ... # (see run/run.sh for full parameters)
```

### Option 2: Manual execution

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate isopo2
export CUDA_HOME=/home/ubuntu/anaconda3/envs/isopo2
export FLASHINFER_CUDA_HOME=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH

# Clear flashinfer cache if needed
rm -rf ~/.cache/flashinfer

# Run using explicit python path
/home/ubuntu/anaconda3/envs/isopo2/bin/python3 -m verl.trainer.main_ppo ...
```

## Troubleshooting

### Error: `math.h: No such file or directory` during flashinfer compilation

**Solution**: Install C development headers:
```bash
sudo apt-get install -y libc6-dev build-essential
rm -rf ~/.cache/flashinfer  # Clear the cache
```

### Error: `ModuleNotFoundError: No module named 'ray'`

**Solution**: This happens when system Python is used instead of conda Python. Either:
1. Use explicit Python path: `/home/ubuntu/anaconda3/envs/isopo2/bin/python3`
2. Ensure CUDA_HOME path is added BEFORE system paths in PATH

### Flashinfer compilation errors with system CUDA

**Solution**: Install CUDA 12.4 from conda and set `CUDA_HOME` to the conda environment:
```bash
conda install -c nvidia/label/cuda-12.4.0 cuda -y
export CUDA_HOME=/home/ubuntu/anaconda3/envs/isopo2
export FLASHINFER_CUDA_HOME=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
```

### Verify flashinfer is working

After a successful run, you should see compiled modules in:
```bash
ls ~/.cache/flashinfer/90/cached_ops/sampling/
# Should show: sampling.so, *.cuda.o files
```

## Quick Start (After Installation)

```bash
# Activate environment and set CUDA
source ~/anaconda3/etc/profile.d/conda.sh
conda activate isopo2
export CUDA_HOME=/home/ubuntu/anaconda3/envs/isopo2
export FLASHINFER_CUDA_HOME=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH

# Run GRPO (using the wrapper script)
bash run_grpo.sh
```

## Verified Package Versions

- Python: 3.12.0
- PyTorch: 2.8.0
- vLLM: 0.11.0
- sglang: 0.5.2
- flashinfer-python: 0.3.1
- CUDA (conda): 12.4
- numpy: 1.26.4

## Environment Summary

The key insight is that **flashinfer requires JIT compilation at runtime**, and the CUDA toolkit used must be:
1. Compatible with the installed flashinfer version
2. Have all necessary headers available
3. Be accessible via `CUDA_HOME` environment variable

Using conda's CUDA 12.4 toolkit instead of the system CUDA 12.8 resolves the compatibility issues.

## Files Created

- `INSTALL.md` - This installation guide
- `run_grpo.sh` - Wrapper script with proper environment setup for running GRPO
