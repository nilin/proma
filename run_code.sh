# Build docker image (run once)
sudo docker build -f Dockerfile.proma2 -t proma2 .


# Code benchmark: MBPP train -> HumanEval val (OOD)
sudo docker run --rm --gpus all --shm-size=16g \
    -v /home/ubuntu/proma2/checkpoints:/app/checkpoints \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    proma2 code_proma_intra_eig \
    +actor_rollout_ref.actor.proma_shrinkage=1.0 \
    ++actor_rollout_ref.actor.optim.lr=2e-6 \
    ++actor_rollout_ref.actor.proma_intra_dim=100

sudo fuser -k /dev/nvidia* 2>/dev/null || true


sudo docker run --rm --gpus all --shm-size=16g \
    -v /home/ubuntu/proma2/checkpoints:/app/checkpoints \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    proma2 code_grpo \
    ++actor_rollout_ref.actor.optim.lr=2e-6 

sudo fuser -k /dev/nvidia* 2>/dev/null || true
