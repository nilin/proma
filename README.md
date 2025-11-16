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


conda env create -f environment.yml -n seppo1
conda activate seppo1
pip install flash-attn==2.8.3 --no-build-isolation

pip install huggingface_hub[cli]
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./models/Qwen3-0.6B

python examples/data_preprocess/gsm8k.py

# substitute with your wandb key
wandb login $WANDB_API_KEY
```
