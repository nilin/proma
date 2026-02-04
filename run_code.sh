# Build docker image (run once)
# sudo docker build -f Dockerfile.proma2 -t proma2 .

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Fast test with smaller batches
sudo docker run --rm --gpus all --shm-size=16g \
    -v /home/ubuntu/proma2/checkpoints:/app/checkpoints \
    -v /home/ubuntu/proma2/verl/workers/actor/dp_actor.py:/app/verl/workers/actor/dp_actor.py \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    proma2 code_proma_intra_eig \
    ++data.train_batch_size=16 \
    ++actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    ++actor_rollout_ref.rollout.n=4 \
    ++trainer.total_training_steps=5 \
    ++trainer.val_before_train=False \
    ++trainer.test_freq=100 \
    ++actor_rollout_ref.actor.proma_shrinkage=1.0 \
    ++actor_rollout_ref.actor.proma_intra_shrinkage=1.0 \
    ++actor_rollout_ref.actor.optim.lr=2e-6 \
    ++actor_rollout_ref.actor.proma_intra_from_accumulated=False \
    ++trainer.experiment_name=test_proma_fast_$TIMESTAMP \
    ++actor_rollout_ref.actor.proma_intra_dim=100 \
    ++trainer.project_name=test-proma-code

# # Code benchmark: MBPP train -> HumanEval val (OOD)
# sudo docker run --rm --gpus all --shm-size=16g \
#     -v /home/ubuntu/proma2/checkpoints:/app/checkpoints \
#     -e WANDB_API_KEY=$WANDB_API_KEY \
#     proma2 code_proma_intra_eig \
#     ++actor_rollout_ref.actor.proma_shrinkage=1.0 \
#     ++actor_rollout_ref.actor.proma_intra_shrinkage=0.9 \
#     ++actor_rollout_ref.actor.optim.lr=2e-6 \
#     ++actor_rollout_ref.actor.proma_intra_from_accumulated=True \
#     ++trainer.experiment_name=mbpp-humaneval-proma_intra_eig_intra-acc30_90_$TIMESTAMP \
#     ++actor_rollout_ref.actor.proma_intra_dim=30
# 
# sudo fuser -k /dev/nvidia* 2>/dev/null || true
# 
# # Code benchmark: MBPP train -> HumanEval val (OOD)
# sudo docker run --rm --gpus all --shm-size=16g \
#     -v /home/ubuntu/proma2/checkpoints:/app/checkpoints \
#     -e WANDB_API_KEY=$WANDB_API_KEY \
#     proma2 code_proma_intra_eig \
#     ++actor_rollout_ref.actor.proma_shrinkage=1.0 \
#     ++actor_rollout_ref.actor.proma_intra_shrinkage=0.95 \
#     ++actor_rollout_ref.actor.optim.lr=2e-6 \
#     ++actor_rollout_ref.actor.proma_intra_from_accumulated=False \
#     ++trainer.experiment_name=mbpp-humaneval-proma_intra_eig_95_$TIMESTAMP \
#     ++actor_rollout_ref.actor.proma_intra_dim=100
# 
# sudo fuser -k /dev/nvidia* 2>/dev/null || true
# 
# source next.sh
# 
# sudo docker run --rm --gpus all --shm-size=16g \
#     -v /home/ubuntu/proma2/checkpoints:/app/checkpoints \
#     -e WANDB_API_KEY=$WANDB_API_KEY \
#     proma2 code_grpo \
#     ++actor_rollout_ref.actor.optim.lr=5e-6 
# 
# sudo fuser -k /dev/nvidia* 2>/dev/null || true
