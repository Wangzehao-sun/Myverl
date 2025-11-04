set -x

unset ROCR_VISIBLE_DEVICES

echo $HOME
export RAY_DEDUP_LOGS=0
export WANDB_MODE=offline
train_path=$HOME/LLM/Train/data/openr1.parquet
test_path=$HOME/LLM/Train/data/valid.parquet
train_files="['$train_path']"
val_files="['$test_path']"

name="luffy_se"
MODEL_PATH=${1:-"$HOME/Model/Qwen2.5-Math-7B-16k-think"} #Model/Qwen2.5-Math-7B-16k-think
echo "Model Path: $MODEL_PATH"
PROJECT_NAME="calculate_probs_entropy_$(basename $MODEL_PATH)_$(basename $train_path .parquet)"
EXP_NAME="calculate_probs_entropy"
LOG_DIR=$HOME/LLM/Train/verl/logs/${PROJECT_NAME}
mkdir -p ${LOG_DIR}
LOG_PATH=${LOG_DIR}/${EXP_NAME}.log

GPU_NUM=2
TENSOR_PARALLEL=1

DATA_DIR=$HOME/LLM/Train/data/

cd $HOME/LLM/Train/verl/
echo "change to dir: $PWD"

if [ -n "$1" ]; then
    shift
fi
#shift
# Train over a single node, 8 A100-80GB GPUs.
python -m verl.trainer.calculate_probs_entropy \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.000 \
    algorithm.norm_adv_by_std_in_grpo=False \
    +algorithm.filter_reward=False \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.train_batch_size=512 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.return_full_prompt=True \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=4 \
    data.shuffle=False \
    +data.reward_impl_version=3 \
    reward_model.reward_manager='math' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    +actor_rollout_ref.rollout.max_prefix_len=8192 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$GPU_NUM \
    trainer.nnodes=1 \
    trainer.balance_batch=False \
    trainer.rollout_data_dir=$LOG_DIR/rollout_data \
    +trainer.save_tensors_dir=$LOG_DIR/save_tensors \
    +trainer.log_file=$LOG_DIR/entropy_probs.jsonl \
    +trainer.max_training_steps=8 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 $@ 2>&1 | tee ${LOG_PATH}
