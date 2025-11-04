set -x


unset ROCR_VISIBLE_DEVICES
data_path=$HOME/LLM/Train/data/omni.parquet
MODEL_PATH=${1:-"$HOME/Model/Qwen2.5-Math-7B-16k-think"} #Model/Qwen2.5-Math-7B-16k-think
PROJECT_NAME="eval_$(basename $MODEL_PATH)_$(basename $data_path .parquet)"
LOG_DIR=$HOME/LLM/Train/verl/logs/${PROJECT_NAME}
mkdir -p ${LOG_DIR}
save_path=${LOG_DIR}/generation_results.parquet

GPU_NUM=2
TENSOR_PARALLEL=1


cd $HOME/LLM/Train/verl/
echo "change to dir: $PWD"



python -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$GPU_NUM \
    data.path=$data_path \
    data.prompt_key=prompt \
    +data.reward_model_key=reward_model \
    +data.data_source_key=data_source \
    data.n_samples=8 \
    data.batch_size=128 \
    data.output_path=$save_path \
    model.path=$MODEL_PATH \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=2048 \
    rollout.response_length=8192 \
    rollout.max_num_batched_tokens=16384 \
    rollout.tensor_model_parallel_size=$TENSOR_PARALLEL \
    rollout.gpu_memory_utilization=0.7 \
    +is_eval=True \
    +reward_model.reward_impl_version=4 \
