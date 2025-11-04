set -x

data_path=$HOME/LLM/Train/data/openr1_standard.parquet
MODEL_PATH=${1:-"$HOME/Model/Qwen2.5-Math-7B-16k-think"} #Model/Qwen2.5-Math-7B-16k-think

save_path=$HOME/LLM/Train/verl/logs/eval__$(basename $MODEL_PATH)_$(basename $data_path)

GPU_NUM=2
TENSOR_PARALLEL=1


python -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$GPU_NUM \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.response_key=responses \
    data.data_source_key=data_source \
    data.n_samples=8 \
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
    is_eval=True \
    reward_model.reward_manager='math' \
