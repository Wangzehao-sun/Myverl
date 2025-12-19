# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""

import os,sys

import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from tqdm import tqdm

@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    chat_lst = dataset[config.data.prompt_key].tolist()

    chat_lst = [chat.tolist() for chat in chat_lst]

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)
    
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    output_lst = [[] for _ in range(config.data.n_samples)]
    total_avg_score = 0.0
    for batch_idx in tqdm(range(num_batch), desc="Generating Batches"):
        if (batch_idx + 1) > config.max_steps:
            print(f"Reached max steps {config.max_steps}, stopping generation.")
            break
        #print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        start_idx = batch_idx * config_batch_size
        end_idx = (batch_idx + 1) * config_batch_size
        
        # 1. 获取当前批次的数据
        batch_dataset = dataset.iloc[start_idx:end_idx].copy()
        batch_chat_lst = chat_lst[start_idx:end_idx]
        #batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        inputs = tokenizer.apply_chat_template(
            batch_chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        data = DataProto.from_dict(batch_dict)
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

        # START TO GENERATE FOR n_samples TIMES
        avg_token_length = 0
        print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")
        batch_output_lst = [[] for _ in range(config.data.n_samples)]
        for n_sample in range(config.data.n_samples):
            output_padded = wg.generate_sequences(data_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)

            output_texts = []
            for i in range(len(output)):
                data_item = output[i]
                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = data_item.batch["responses"][:valid_response_length]
                response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                output_texts.append(response_str)
                avg_token_length += valid_response_length.item()
            #output_lst[n_sample].extend(output_texts)
            batch_output_lst[n_sample].extend(output_texts)
        avg_token_length /= (len(output) * config.data.n_samples)
        print(f"Average token length for batch {batch_idx + 1}: {avg_token_length:.2f}")
    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    # output_lst = np.array(output_lst, dtype=object)
    # output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()
        batch_output_lst = np.array(batch_output_lst, dtype=object)
        batch_output_lst = np.transpose(batch_output_lst, axes=(1, 0)).tolist()
        #batch_dataset
    # add to the data frame
        batch_dataset["responses"] = batch_output_lst
        if config.is_eval == True:
            # evaluate if needed
            print("Start to evaluate generated results.")
            from verl.custom.math_verify_reward import math_select_rm_score_fn

            #compute_score = math_select_rm_score_fn

            score_lst = []
            data_sources = batch_dataset[config.data.data_source_key]
            reward_dataset = batch_dataset[config.data.reward_model_key]
            responses_lst = batch_dataset["responses"]
            for i in range(len(batch_dataset)):
                data_source = data_sources.iloc[i]
                reward_data = reward_dataset.iloc[i]
                responses = responses_lst.iloc[i]
                ground_truth = reward_data["ground_truth"]
                compute_score = math_select_rm_score_fn(data_source, reward_impl_version=config.reward_model.reward_impl_version)
                score_per_response = [compute_score(solution_str=r, ground_truth=ground_truth) for r in responses]
                score_lst.append({
                    "scores_per_response": score_per_response,
                    "mean_score": np.mean(score_per_response),
                })
            batch_dataset["test_score"] = score_lst
            #打印当前批次的平均分数
            batch_mean_scores = [score["mean_score"] for score in score_lst]
            batch_average_score = np.mean(batch_mean_scores)
            total_avg_score += batch_average_score
            print(f"Batch {batch_idx} average score: {batch_average_score}",flush=True)
        # 5. 将当前批次的结果保存到独立文件
        batch_output_filename = f"{batch_idx}.parquet"
        batch_output_path = os.path.join(output_dir, batch_output_filename)
        batch_dataset.to_parquet(batch_output_path)
        sys.stdout.flush()
        tqdm.write(f"Batch {batch_idx} saved to {batch_output_path}")
    
    print("All batches have been processed and saved.")
    total_avg_score /= num_batch
    print(f"Total average score so far: {total_avg_score}")
    # if config.is_eval == True:
    #     # evaluate if needed
    #     print("Start to evaluate generated results.")
    #     from verl.custom.math_verify_reward import math_select_rm_score_fn

    #     #compute_score = math_select_rm_score_fn

    #     score_lst = []
    #     data_sources = dataset[config.data.data_source_key]
    #     reward_dataset = dataset[config.data.reward_model_key]
    #     responses_lst = dataset["responses"]
    #     for i in range(len(dataset)):
    #         data_source = data_sources.iloc[i]
    #         reward_data = reward_dataset.iloc[i]
    #         responses = responses_lst.iloc[i]
    #         ground_truth = reward_data["ground_truth"]
    #         compute_score = math_select_rm_score_fn(data_source, reward_impl_version=config.reward_model.reward_impl_version)
    #         score_per_response = [compute_score(data_source, r, ground_truth) for r in responses]
    #         score_lst.append({
    #             "scores_per_response": score_per_response,
    #             "mean_score": np.mean(score_per_response),
    #         })
    #     dataset["test_score"] = score_lst
    # write to a new parquet
    
    #dataset.to_parquet(config.data.output_path)


if __name__ == "__main__":
    main()
