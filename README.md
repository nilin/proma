# Self-Proximal Policy Gradients without pi-old
**Nilin Abrahamsen**

This repo is a fork of [VeRL](https://github.com/volcengine/verl) to demonstrate the SEPPO self-proximal policy gradient.


### Setup

```
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh -b
source ~/anaconda3/bin/activate
~/anaconda3/bin/conda init

conda create -n seppo2 python==3.12
conda activate seppo2
bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .


# Now there are problems with math

conda install -c nvidia/label/cuda-12.4.0 cuda
export CUDA_HOME=/home/ubuntu/anaconda3/envs/seppo2
export FLASHINFER_CUDA_HOME=$CUDA_HOME
pip uninstall -y flashinfer flashinfer-python || true
rm -rf ~/.cache/flashinfer

# 2) (Optional but good) sanity-check nvcc is CUDA 12.x
nvcc --version

# 3) Install FlashInfer from PyPI (correct package name)
pip install flashinfer-python -i https://pypi.org/simple


pip install huggingface_hub[cli]
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./models/Qwen3-0.6B
python examples/data_preprocess/gsm8k.py

# substitute with your wandb key
wandb login $WANDB_API_KEY
```


**Old version**
```
conda env create -f environment.yml -n seppo1
conda activate seppo1
pip install flash-attn==2.8.3 --no-build-isolation

pip install huggingface_hub[cli]
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./models/Qwen3-0.6B

python examples/data_preprocess/gsm8k.py

# substitute with your wandb key
wandb login $WANDB_API_KEY
```
