  
#sudo docker build -f Dockerfile.proma2 -t proma2 .                                                                                                                        

sudo docker run --rm --gpus all --shm-size=16g -e WANDB_API_KEY=$WANDB_API_KEY proma2 proma_intra_eig +actor_rollout_ref.actor.proma_shrinkage=1.0 ++trainer.experiment_name=proma_intra_eig_both-lr7 ++actor_rollout_ref.actor.optim.lr=7e-6 ++trainer.total_training_steps=102 ++actor_rollout_ref.actor.proma_intra_dim=100
sudo fuser -k /dev/nvidia* 2>/dev/null || true 

sudo docker run --rm --gpus all --shm-size=16g -e WANDB_API_KEY=$WANDB_API_KEY proma2 proma_intra_eig +actor_rollout_ref.actor.proma_shrinkage=1.0 ++trainer.experiment_name=proma_intra_eig_both-lr7 ++actor_rollout_ref.actor.optim.lr=7e-6 ++trainer.total_training_steps=102 ++actor_rollout_ref.actor.proma_intra_dim=100
sudo fuser -k /dev/nvidia* 2>/dev/null || true 

source next.sh

