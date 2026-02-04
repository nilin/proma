#!/bin/bash
# Evaluate checkpoints on SVAMP (or other val set)
#
# Usage:
#   ./eval_checkpoints.sh checkpoints/commac/proma_*/
#   ./eval_checkpoints.sh checkpoints/commac/reinforce_20260203_*/global_step_100
#
# Or evaluate a single checkpoint:
#   ./eval_checkpoints.sh checkpoints/commac/proma_20260203_123456/global_step_50

# Validation dataset (SVAMP by default)
VAL_DATA=${VAL_DATA:-$HOME/data/svamp/test.parquet}

eval_checkpoint () {
    local ckpt_path="$1"

    # Extract experiment name from path for logging
    local exp_name=$(basename $(dirname "$ckpt_path"))
    local step_name=$(basename "$ckpt_path")

    echo "=========================================="
    echo "Evaluating: $ckpt_path"
    echo "Val data: $VAL_DATA"
    echo "=========================================="

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$HOME/data/gsm8k/train.parquet \
        data.val_files=$VAL_DATA \
        data.train_batch_size=128 \
        data.max_prompt_length=512 \
        data.max_response_length=2048 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=$ckpt_path/actor \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.entropy_coeff=0.0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
        actor_rollout_ref.actor.strategy=fsdp2 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.temperature=0.0 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        algorithm.kl_ctrl.kl_coef=0.0 \
        critic.strategy=fsdp2 \
        trainer.critic_warmup=0 \
        trainer.logger='["console"]' \
        trainer.project_name='eval' \
        trainer.experiment_name="eval_${exp_name}_${step_name}" \
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=1 \
        trainer.total_epochs=1 \
        trainer.resume_mode=disable \
        trainer.val_before_train=True \
        trainer.val_only=True \
        2>&1 | tee -a eval_results.log

    echo ""
}

# Main: iterate over all provided checkpoint paths
if [ $# -eq 0 ]; then
    echo "Usage: $0 <checkpoint_path> [checkpoint_path2] ..."
    echo ""
    echo "Examples:"
    echo "  $0 checkpoints/commac/proma_*/global_step_*"
    echo "  $0 checkpoints/commac/reinforce_20260203_123456/global_step_100"
    echo ""
    echo "Environment variables:"
    echo "  VAL_DATA - validation parquet file (default: ~/data/svamp/test.parquet)"
    exit 1
fi

echo "Evaluation results will be appended to: eval_results.log"
echo ""

for ckpt in "$@"; do
    # Check if it's a valid checkpoint directory (should have actor/ subfolder)
    if [ -d "$ckpt/actor" ]; then
        eval_checkpoint "$ckpt"
    elif [ -d "$ckpt" ]; then
        # Maybe it's an experiment dir, look for global_step_* subdirs
        for step_dir in "$ckpt"/global_step_*; do
            if [ -d "$step_dir/actor" ]; then
                eval_checkpoint "$step_dir"
            fi
        done
    else
        echo "Warning: $ckpt is not a valid checkpoint directory (no actor/ subfolder)"
    fi
done

echo "=========================================="
echo "Evaluation complete. Results in eval_results.log"
echo "=========================================="
