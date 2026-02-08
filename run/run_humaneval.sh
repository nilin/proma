# Code benchmark run script for PROMA experiments
#
# Supports:
#   - HumanEval only (164 problems, same train/val)
#   - MBPP -> HumanEval cross-benchmark (OOD val)
#
# Data preprocessing:
#   python examples/data_preprocess/humaneval.py --local_save_dir ~/data/humaneval
#   python examples/data_preprocess/mbpp.py --local_save_dir ~/data/mbpp

# Default: MBPP train -> HumanEval val (cross-benchmark OOD)
TRAIN_DATA=${TRAIN_DATA:-$HOME/data/mbpp/train.parquet}
VAL_DATA=${VAL_DATA:-$HOME/data/humaneval/test.parquet}

# Timestamp for unique experiment names (prevents checkpoint overwriting)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# pure means no clipping, i.e. reinforce and not ppo
pure="actor_rollout_ref.actor.clip_ratio=1e9 actor_rollout_ref.actor.clip_ratio_high=1e9 actor_rollout_ref.actor.clip_ratio_low=1e9 actor_rollout_ref.actor.clip_ratio_c=1e9"
ppo="actor_rollout_ref.actor.clip_ratio=0.2 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_c=3"
test="trainer.val_before_train=False trainer.project_name=test-code data.train_batch_size=16 actor_rollout_ref.actor.ppo_mini_batch_size=16"
lr=actor_rollout_ref.actor.optim.lr

run-alg () {
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$TRAIN_DATA \
        data.val_files=$VAL_DATA \
        data.train_batch_size=64 \
        data.max_prompt_length=512 \
        data.max_response_length=1024 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=models/Qwen3-0.6B \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.policy_loss.ppo_kl_coef=0.0 \
        actor_rollout_ref.actor.entropy_coeff=0.0 \
        actor_rollout_ref.actor.policy_loss.loss_mode="vanilla" \
        actor_rollout_ref.actor.loss_agg_mode="token-mean" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
        actor_rollout_ref.actor.strategy=fsdp2 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=16 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        algorithm.kl_ctrl.kl_coef=0.0 \
        critic.strategy=fsdp2 \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name='proma-code' \
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=1 \
        trainer.save_freq=10 \
        trainer.test_freq=2 \
        trainer.total_epochs=200 \
        trainer.total_training_steps=52 \
        trainer.resume_mode=disable \
        trainer.val_before_train=True \
        actor_rollout_ref.actor.optim.lr=2e-6 \
        +actor_rollout_ref.actor.isopo_keep_small_invariant=False \
        +actor_rollout_ref.actor.bypass_isopo_scaling=True \
        +actor_rollout_ref.actor.quick_ntk=False \
        $pure \
        "$@"
}

# MBPP train -> HumanEval val (cross-benchmark OOD) - default
reinforce () { run-alg +actor_rollout_ref.actor.use_proma_isopo=False trainer.experiment_name=mbpp-humaneval-reinforce_$TIMESTAMP "$@"; }
grpo () { run-alg +actor_rollout_ref.actor.use_proma_isopo=False $ppo trainer.experiment_name=mbpp-humaneval-grpo_$TIMESTAMP "$@"; }
proma () { run-alg +actor_rollout_ref.actor.use_proma_isopo=True +actor_rollout_ref.actor.proma_shrinkage=1.0 trainer.experiment_name=mbpp-humaneval-proma_$TIMESTAMP "$@"; }
proma_intra () { run-alg +actor_rollout_ref.actor.use_proma_isopo=True +actor_rollout_ref.actor.proma_intra=True +actor_rollout_ref.actor.proma_intra_dim=250 +actor_rollout_ref.actor.proma_intra_shrinkage=1.0 trainer.experiment_name=mbpp-humaneval-proma_intra_$TIMESTAMP "$@"; }
proma_intra_acc () { run-alg +actor_rollout_ref.actor.use_proma_isopo=True +actor_rollout_ref.actor.proma_shrinkage=0.0 +actor_rollout_ref.actor.proma_intra=True +actor_rollout_ref.actor.proma_intra_dim=25 +actor_rollout_ref.actor.proma_intra_shrinkage=1.0 +actor_rollout_ref.actor.proma_intra_to_accumulated=True trainer.experiment_name=mbpp-humaneval-proma_intra_acc_$TIMESTAMP "$@"; }
proma_both () { run-alg +actor_rollout_ref.actor.use_proma_isopo=True +actor_rollout_ref.actor.proma_shrinkage=1.0 +actor_rollout_ref.actor.proma_intra=True +actor_rollout_ref.actor.proma_intra_dim=25 +actor_rollout_ref.actor.proma_intra_shrinkage=1.0 +actor_rollout_ref.actor.proma_intra_to_accumulated=True trainer.experiment_name=mbpp-humaneval-proma_both_$TIMESTAMP "$@"; }

# HumanEval train -> HumanEval val (in-distribution)
humaneval_reinforce () { TRAIN_DATA=$HOME/data/humaneval/train.parquet VAL_DATA=$HOME/data/humaneval/test.parquet run-alg +actor_rollout_ref.actor.use_proma_isopo=False trainer.experiment_name=humaneval-reinforce_$TIMESTAMP "$@"; }
humaneval_grpo () { TRAIN_DATA=$HOME/data/humaneval/train.parquet VAL_DATA=$HOME/data/humaneval/test.parquet run-alg +actor_rollout_ref.actor.use_proma_isopo=False $ppo trainer.experiment_name=humaneval-grpo_$TIMESTAMP "$@"; }
humaneval_proma () { TRAIN_DATA=$HOME/data/humaneval/train.parquet VAL_DATA=$HOME/data/humaneval/test.parquet run-alg +actor_rollout_ref.actor.use_proma_isopo=True +actor_rollout_ref.actor.proma_shrinkage=1.0 trainer.experiment_name=humaneval-proma_$TIMESTAMP "$@"; }

# Quick test run with smaller batches
test_run () { run-alg $test trainer.experiment_name=code-test_$TIMESTAMP "$@"; }
