conda create -n seppo2 python==3.12 -y
conda activate seppo2
bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .


# Now there are problems with math

conda install -c nvidia/label/cuda-12.4.0 cuda -y
export CUDA_HOME=/home/ubuntu/anaconda3/envs/seppo2
export FLASHINFER_CUDA_HOME=$CUDA_HOME
pip uninstall -y flashinfer flashinfer-python || true
rm -rf ~/.cache/flashinfer

# 2) (Optional but good) sanity-check nvcc is CUDA 12.x
nvcc --version

# 3) Install FlashInfer from PyPI (correct package name)
pip install flashinfer-python -i https://pypi.org/simple

