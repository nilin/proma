# Build docker image (run once)
# sudo docker build -f Dockerfile.proma2 -t proma2 .


for q in 0 1 2; do

for LR in 5e-6 1e-5 2e-6 1e-6; do

# proma

# Code benchmark: MBPP train -> HumanEval val (OOD)
sudo docker run --rm --gpus all --shm-size=16g \
    -v /home/ubuntu/proma2:/app \
    -v /home/ubuntu/proma2/checkpoints:/app/checkpoints \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    proma2 code_proma \
         ++actor_rollout_ref.actor.optim.lr=$LR \
         ++actor_rollout_ref.actor.proma_shrinkage=1.0 \
         ++actor_rollout_ref.actor.proma_skip_fraction=0.001

sudo fuser -k /dev/nvidia* 2>/dev/null || true

source next.sh


# proma intra

# Code benchmark: MBPP train -> HumanEval val (OOD)
sudo docker run --rm --gpus all --shm-size=16g \
    -v /home/ubuntu/proma2:/app \
    -v /home/ubuntu/proma2/checkpoints:/app/checkpoints \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    proma2 code_proma_intra \
        ++actor_rollout_ref.actor.optim.lr=$LR \
        ++actor_rollout_ref.actor.proma_shrinkage=0.0 \
        ++actor_rollout_ref.actor.proma_intra_shrinkage=1.0 \
        ++actor_rollout_ref.actor.proma_intra_to_accumulated=False \
        ++actor_rollout_ref.actor.proma_intra_to_accumulated_after=False \
        ++actor_rollout_ref.actor.proma_skip_fraction=0.0 \
        ++actor_rollout_ref.actor.proma_intra_dim=10 

sudo fuser -k /dev/nvidia* 2>/dev/null || true

source next.sh


# proma both (intra + sequence-wise)

# Code benchmark: MBPP train -> HumanEval val (OOD)
sudo docker run --rm --gpus all --shm-size=16g \
    -v /home/ubuntu/proma2:/app \
    -v /home/ubuntu/proma2/checkpoints:/app/checkpoints \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    proma2 code_proma_both \
        ++actor_rollout_ref.actor.optim.lr=$LR \
        ++actor_rollout_ref.actor.proma_skip_fraction=0.001 \
        ++actor_rollout_ref.actor.proma_intra_dim=10

sudo fuser -k /dev/nvidia* 2>/dev/null || true

source next.sh


# Code benchmark: MBPP train -> HumanEval val (OOD)
sudo docker run --rm --gpus all --shm-size=16g \
    -v /home/ubuntu/proma2:/app \
    -v /home/ubuntu/proma2/checkpoints:/app/checkpoints \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    proma2 code_reinforce \
        ++actor_rollout_ref.actor.optim.lr=$LR

sudo fuser -k /dev/nvidia* 2>/dev/null || true

source next.sh


# Code benchmark: MBPP train -> HumanEval val (OOD)
sudo docker run --rm --gpus all --shm-size=16g \
    -v /home/ubuntu/proma2:/app \
    -v /home/ubuntu/proma2/checkpoints:/app/checkpoints \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    proma2 code_grpo \
        ++actor_rollout_ref.actor.optim.lr=$LR

sudo fuser -k /dev/nvidia* 2>/dev/null || true

source next.sh


done

done



