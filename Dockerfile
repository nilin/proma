# 1. Start with an image that has CUDA 12.4 and Conda pre-installed
FROM mambaorg/micromamba:latest

# Switch to root to install system tools
USER root
RUN apt-get update && apt-get install -y curl git build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Create the environment (Micromamba is a faster 'conda' drop-in)
# This replaces your 'conda create' and 'anaconda.sh' lines
RUN micromamba create -n suppo python=3.12 -c conda-forge -y

# Make sure all subsequent commands run in the 'suppo' environment
ARG MAMBA_DOCKERFILE_LOGLEVEL=debug
ENV PATH /opt/conda/envs/suppo/bin:$PATH

# 3. Handle the "Heavy" installs
# We copy ONLY the script needed for vllm/mcore first to cache the layer
COPY scripts/install_vllm_sglang_mcore.sh ./scripts/
RUN bash scripts/install_vllm_sglang_mcore.sh

# 4. Handle the "Math Problems" (FlashInfer & CUDA)
# The base image already has CUDA, but we pin the environment variables
ENV CUDA_HOME=/opt/conda/envs/suppo
ENV FLASHINFER_CUDA_HOME=$CUDA_HOME

RUN pip install flashinfer-python -i https://pypi.org/simple
RUN pip install huggingface_hub[cli]

# 5. Copy the repo and install it
COPY . .
RUN pip install --no-deps -e .

# 6. Pre-download the model (This saves time during runtime!)
# Note: This makes the image large, but very fast to start
RUN huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./models/Qwen3-0.6B

# Pre-process data so it's ready to go
RUN python examples/data_preprocess/gsm8k.py

CMD ["python", "your_main_script.py"]
