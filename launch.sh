# Build docker image (run once)
# sudo docker build -f Dockerfile.proma2 -t proma2 .
# sudo docker run --rm -v /home/ubuntu/proma2:/app proma2 huggingface-cli download Qwen/Qwen3-0.6B --local-dir /app/models/Qwen3-0.6B

sudo fuser -k /dev/nvidia* 2>/dev/null || true



### for q in 0 1 2; do
### 
### for LR in 5e-6 2e-6; do
### 
### 
### 
### 
### # Code benchmark: MBPP train -> HumanEval val (OOD)
### sudo docker run --rm --gpus all --shm-size=16g \
###     -v /home/ubuntu/proma2:/app \
###     -v /home/ubuntu/proma2/checkpoints:/app/checkpoints \
###     -e WANDB_API_KEY=$WANDB_API_KEY \
###     proma2 code_reinforce \
###         ++actor_rollout_ref.actor.optim.lr=$LR
### 
### sudo fuser -k /dev/nvidia* 2>/dev/null || true
### 
### source next.sh
### 
### 
### # Code benchmark: MBPP train -> HumanEval val (OOD)
### sudo docker run --rm --gpus all --shm-size=16g \
###     -v /home/ubuntu/proma2:/app \
###     -v /home/ubuntu/proma2/checkpoints:/app/checkpoints \
###     -e WANDB_API_KEY=$WANDB_API_KEY \
###     proma2 code_grpo \
###         ++actor_rollout_ref.actor.optim.lr=$LR
### 
### sudo fuser -k /dev/nvidia* 2>/dev/null || true
### 
### source next.sh
### 
### 
### done
### 
### done


for LR in 5e-6 2e-6; do
  for q in 0 1; do
  
    # Best PROMA intra: d=10, ishr=1
    sudo docker run --rm --gpus all --shm-size=16g \
        -v /home/ubuntu/proma2:/app \
        -v /home/ubuntu/proma2/checkpoints:/app/checkpoints \
        -e WANDB_API_KEY=$WANDB_API_KEY \
        proma2 code_proma_intra \
            ++actor_rollout_ref.actor.optim.lr=$LR \
            ++actor_rollout_ref.actor.proma_intra_dim=10
    
    sudo fuser -k /dev/nvidia* 2>/dev/null || true
    source next.sh
  done
done


### # Best non-intra PROMA: skip=0.5
### sudo docker run --rm --gpus all --shm-size=16g \
###     -v /home/ubuntu/proma2:/app \
###     -v /home/ubuntu/proma2/checkpoints:/app/checkpoints \
###     -e WANDB_API_KEY=$WANDB_API_KEY \
###     proma2 code_proma \
###         ++actor_rollout_ref.actor.optim.lr=5e-6 \
###         +actor_rollout_ref.actor.proma_skip_fraction=0.5
### 
### sudo fuser -k /dev/nvidia* 2>/dev/null || true
### 
### source next.sh



