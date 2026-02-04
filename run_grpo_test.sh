#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

sudo docker run --rm --gpus all --shm-size=16g \
    -v /home/ubuntu/proma2/checkpoints:/app/checkpoints \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    proma2 code_grpo \
    ++data.train_batch_size=16 \
    ++actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    ++actor_rollout_ref.rollout.n=4 \
    ++trainer.total_training_steps=5 \
    ++trainer.val_before_train=False \
    ++trainer.test_freq=100 \
    ++trainer.experiment_name=test_grpo_fast_$TIMESTAMP \
    ++trainer.project_name=test-proma-code
