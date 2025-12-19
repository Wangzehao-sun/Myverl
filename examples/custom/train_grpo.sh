set -x

unset ROCR_VISIBLE_DEVICES

echo $HOME
export RAY_DEDUP_LOGS=0
export WANDB_MODE=offline
train_path=$HOME/LLM/Train/data/s1K_se_standard.parquet
test_path=$HOME/LLM/Train/data/valid_with_aime25.parquet
test1_path=$HOME/LLM/Train/data/split_by_source/aime.parquet
test2_path=$HOME/LLM/Train/data/split_by_source/aime25.parquet
train_files="['$train_path']"
val_files="['$test_path']"
#val_files="['$test1_path','$test2_path']"

name="vanilla"

MODEL_DIR=/home/jwangxgroup/jiewang/shared
MODEL_PATH=$MODEL_DIR/${1:-"Qwen2.5-Math-7B-16k-think"} #Model/Qwen2.5-Math-7B-16k-think

#MODEL_PATH=${1:-"$HOME/Model/Qwen2.5-Math-7B-16k-think"} #Model/Qwen2.5-Math-7B-16k-think
suffix="clip"
PROJECT_NAME="train_${name}_${suffix}_$(basename $MODEL_PATH)_$(basename $train_path .parquet)"
#MODEL_PATH=$HOME/Model/Qwen2.5-Math-7B-16k-think
#PROJECT_NAME="l_grpo_${name}_test1_$(basename $MODEL_PATH)"
EXP_NAME="training"
LOG_DIR=$HOME/LLM/Train/verl/logs/${PROJECT_NAME}
mkdir -p ${LOG_DIR}
LOG_PATH=${LOG_DIR}/${PROJECT_NAME}.log

GPU_NUM=4
TENSOR_PARALLEL=2

DATA_DIR=$HOME/LLM/Train/data/

cd $HOME/LLM/Train/verl/
echo "change to dir: $PWD"
if [ -n "$1" ]; then
    shift
fi
# Train over a single node, 8 A100-80GB GPUs.
python -m verl.trainer.main_ppo_new \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.000 \
    algorithm.norm_adv_by_std_in_grpo=False \
    +algorithm.filter_reward=False \
    data.train_files=$train_files \
    data.val_files="$val_files" \
    data.train_batch_size=128 \
    data.val_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=10240 \
    data.return_full_prompt=True \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=4 \
    data.shuffle=False \
    +data.reward_impl_version=4 \
    +data.filter_targets=True \
    reward_model.reward_manager='math' \
    +se_model.enable=False \
    +actor_rollout_se.model.path=$MODEL_PATH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.policy_loss.loss_mode=$name \
    actor_rollout_ref.actor.clip_ratio=0.28 \
    +actor_rollout_ref.actor.policy_loss.off_policy_reshape="vanilla" \
    +actor_rollout_ref.actor.use_sft_prefix_reward=False \
    +actor_rollout_ref.actor.use_off_policy_loss=True \
    +actor_rollout_ref.actor.off_policy_normalize=False \
    +actor_rollout_ref.actor.off_policy_strategy=$name \
    +actor_rollout_ref.actor.off_policy_loss_impl=token \
    +actor_rollout_ref.actor.off_policy_max_clip=1.2 \
    +actor_rollout_ref.actor.off_policy_min_clip=-1 \
    +actor_rollout_ref.actor.all_max_clip=3 \
    +actor_rollout_ref.actor.use_off_policy_probs=False \
    +actor_rollout_ref.actor.loss_remove_token_mean=True \
    +actor_rollout_ref.actor.loss_remove_clip=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    +actor_rollout_ref.rollout.se_top_k=-1 \
    +actor_rollout_ref.rollout.se_top_p=1 \
    +actor_rollout_ref.rollout.n_val=1 \
    +actor_rollout_ref.rollout.max_prefix_len=10240 \
    +actor_rollout_ref.rollout.n_off=0 \
    +actor_rollout_ref.rollout.n_prefix=0 \
    +actor_rollout_ref.rollout.n_se=0 \
    +actor_rollout_ref.rollout.prefix_ratio=1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$GPU_NUM \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=10 \
    trainer.balance_batch=False \
    trainer.rollout_data_dir=$LOG_DIR/rollout_data \
    +trainer.log_prob_dir=$LOG_DIR/log_probs \
    +trainer.save_tensors_dir=$LOG_DIR/save_tensors \
    +trainer.metrics_data_dir=$LOG_DIR \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=3 $@ 2>&1 | tee ${LOG_PATH}
