# 1. Use the NVIDIA image - it has the right CUDA, NCCL, and Python for H100
FROM nvcr.io/nvidia/pytorch:24.03-py3

# 2. Set the working directory
WORKDIR /app

# 3. Optimize the build for H100 (Hopper)
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV DEBIAN_FRONTEND=noninteractive

# 4. Install system essentials needed for Ray and FlashAttention
RUN apt-get update && apt-get install -y \
    curl git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 5. Install Python dependencies directly
# We install ninja first so flash-attn builds in minutes, not hours
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir ninja packaging

# 6. Install Flash-Attention with "No Build Isolation"
# This is the FIX for your PackageNotFoundError
RUN pip install flash-attn --no-build-isolation

# 7. Install the verl-specific stack (vLLM and Ray)
# Note: ray[default] is often needed for the dashboard/workers
RUN pip install vllm==0.5.4 ray[default]==2.10.0 transformers>=4.57.0 

# 8. Copy your repo and install it in editable mode
# This uses your local setup.py instead of an environment.yml
COPY . .
RUN pip install --no-deps -e .

# 9. Set the environment variables your script was looking for
ENV CUDA_HOME=/usr/local/cuda
ENV FLASHINFER_CUDA_HOME=/usr/local/cuda
ENV PYTHONPATH=$PYTHONPATH:/app

# Standard Ray ports
EXPOSE 6379 8265 10001
