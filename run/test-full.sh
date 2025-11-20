# https://github.com/volcengine/verl/commit/b75b1f0bf1dbff60f004795eac845446da32a22d

switch-alg () {

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=models/Qwen3-0.6B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.policy_loss.ppo_kl_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode="vanilla" \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.clip_ratio=1e9 \
    actor_rollout_ref.actor.clip_ratio_high=1e9 \
    actor_rollout_ref.actor.clip_ratio_low=1e9 \
    actor_rollout_ref.actor.clip_ratio_c=1e9 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='nspg' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
        trainer.experiment_name='SWITCH-ALG' \
        actor_rollout_ref.actor.optim.lr=2e-6 \
	$@

}


switch-alg-2updates () {
    switch-alg \
	data.train_batch_size=128 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        trainer.project_name='paper2' \
	actor_rollout_ref.rollout.temperature=1.0 \
	actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
	trainer.total_epochs=20 \
    trainer.resume_mode=disable \
    trainer.project_name='seppo2' \
    trainer.total_training_steps=102 \
    trainer.n_gpus_per_node=2 \
	$@

    bash run/next.sh
}

switch-alg-2updates \
	+actor_rollout_ref.actor.use_seppo=True \
	+actor_rollout_ref.actor.seppo_testing=False \
	actor_rollout_ref.actor.optim.optimizer=SGD \
	actor_rollout_ref.actor.optim.optimizer_impl=torch.optim \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    trainer.val_before_train=False \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
	actor_rollout_ref.actor.strategy=fsdp2 \
	critic.strategy=fsdp2 \
    trainer.experiment_name=TEST-FULL 

#switch-alg-2updates \
#    +actor_rollout_ref.actor.extra.seppo=True \
#    actor_rollout_ref.actor.extra.algorithm="seppo" \
#    +actor_rollout_ref.actor.optim.extra.optimizer=sgd \
#    actor_rollout_ref.actor.optim.lr=2e-5 \
#    trainer.val_before_train=False \
#    +actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
#	actor_rollout_ref.actor.strategy=fsdp2 \
#	critic.strategy=fsdp2 \
#    trainer.experiment_name=TEST-FULL 

    #+trainer.strategy=fsdp2 \
    #actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \

#        switch-alg-2updates \
#            +actor_rollout_ref.actor.extra.nspg_type=full \
#            trainer.val_before_train=False \
#            +actor_rollout_ref.actor.extra.nspg=True \
#            actor_rollout_ref.actor.extra.algorithm="seppo" \
#            +actor_rollout_ref.actor.extra.nspg_tokenwise_clip=True \
#            trainer.experiment_name=TEST-FULL 

