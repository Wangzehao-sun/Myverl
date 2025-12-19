# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os,sys
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import List, Optional, Type
import time
import numpy as np
import ray
import torch
import math
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from tensordict import TensorDict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer, 
    Role, 
    ResourcePoolManager, 
    WorkerType, 
    # compute_data_metrics, 
    compute_timing_metrics, 
    # compute_advantage, 
    reduce_metrics
)
from verl.custom.new_vllm_rollout import _pre_process_inputs_right_pad, _pre_process_inputs
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, config=None):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            # Get length from the initial response mask
            response_length = grpo_calculation_mask.size(1)
            # This mask is the one intended for GRPO
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


def generate_masks_from_input_ids(input_ids: torch.Tensor, pad_token_id: int,dtype1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    根据 input_ids 生成 attention_mask 和 position_ids。
    这个实现对于左填充和右填充的序列都有效。

    Args:
        input_ids (torch.Tensor): 输入的 token ID 张量。
        pad_token_id (int): padding token 的 ID。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: attention_mask 和 position_ids 张量。
    """
    # 创建 attention_mask，非 padding token 为 1，padding token 为 0
    attention_mask = (input_ids != pad_token_id).to(dtype1)
    
    # 创建 position_ids
    # 对 attention_mask 进行累加，然后减 1，得到从 0 开始的 token 位置
    position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)
    return attention_mask, position_ids
class NewRayPPOTrainer(RayPPOTrainer):
    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            print(role_worker_mapping)
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        # TODO: we have to make sure the batch size is divisible by the dp size
        from torch.utils.data import DataLoader, SequentialSampler
        from verl.utils.dataset.rl_dataset import RLHFDataset
        from verl.utils.dataset.rl_dataset import collate_fn as defaultcollate_fn
        from .rl_dataset_with_target import RLHFDatasetWithTarget
        from torchdata.stateful_dataloader import StatefulDataLoader
        from verl.trainer.main_ppo_new import create_rl_dataset, create_rl_sampler
        self.train_dataset = RLHFDatasetWithTarget(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         config=self.config.data,
                                         max_target_length=self.config.actor_rollout_ref.rollout.max_prefix_len,
                                         filter_targets=self.config.data.get('filter_targets', False),
                                         sample_target_ratio=self.config.data.get('sample_target_ratio', 1.0),
                                         target_key=self.config.data.get('target_key', 'target'),
                                         use_se=self.config.data.get('use_se', True),)

        # use sampler for better ckpt resume
        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           num_workers=self.config.data.get("dataloader_num_workers", 8),
                                           drop_last=True,
                                           collate_fn=defaultcollate_fn if collate_fn is None else collate_fn,
                                           sampler=train_sampler)
        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)
        self.val_dataset = RLHFDataset(data_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       config=self.config.data)
        self.val_dataloader = StatefulDataLoader(dataset=self.val_dataset,
                                         batch_size=val_batch_size,
                                         num_workers=self.config.data.get("dataloader_num_workers", 8),
                                         shuffle=False,
                                         drop_last=True,
                                         collate_fn=defaultcollate_fn if collate_fn is None else collate_fn,)

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")
    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
            # 2. [新增] 次要的 ActorRollout (例如用于 SE 生成)
            # 假设你在 role_worker_mapping 中传入了一个自定义的 key，例如 "ActorRolloutSE"
            # 并且在 resource_pool_manager 中也配置了对应的资源池
            if Role.ActorRolloutSE in self.role_worker_mapping:
                resource_pool_se = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutSE)
                acl_config = deepcopy(self.config.actor_rollout_ref)
                se_config = self.config.actor_rollout_se if hasattr(self.config, 'actor_rollout_se') else acl_config
                # 根据se_config更新acl_config
                acl_config = OmegaConf.merge(acl_config, se_config)
                #打印acl_config
                print("Using SE config for ActorRolloutSE:", acl_config)
                #print("Using SE config for ActorRolloutSE:", acl_config.model.path)
                actor_rollout_se_cls = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.ActorRolloutSE],
                    # 这里可以使用相同的 config，也可以在 config 中定义一个新的 actor_rollout_se 节点
                    config=acl_config, 
                    role="se_rollout_ref",
                )
                self.resource_pool_to_cls[resource_pool_se]["se_rollout_ref"] = actor_rollout_se_cls

        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, "worker_nsight_options must be set when profile_steps is set"
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(OmegaConf.select(self.config.trainer, "worker_nsight_options"))

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # [新增] 初始化第二个 actor_rollout_wg
        if "se_rollout_ref" in all_wg:
            self.actor_rollout_se_wg = all_wg["se_rollout_ref"]
            self.actor_rollout_se_wg.init_model()
            #print("Secondary ActorRollout (SE) initialized.")


        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.workers.rollout.async_server import AsyncLLMServerManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )
    
    def _dump_batch_to_jsonl(self, batch: DataProto, dump_path: str):
        """
        将一个批次的 DataProto 对象反批次化，并逐样本写入 JSONL 文件。
        """
        # 确保目标目录存在
        os.makedirs(dump_path, exist_ok=True)
        # 定义文件名，使用 global_steps 确保唯一性
        filepath = os.path.join(dump_path, f"rollout_data_step_{self.global_steps}.jsonl")

        try:
            # 获取批次大小
            batch_size = batch.meta_info.get("batch_size", len(next(iter(batch.batch.values()))))
            
            all_samples = []
            for i in range(batch_size):
                sample_dict = {}
                
                # 1. 处理张量数据 (batch.batch)
                for key, tensor in batch.batch.items():
                    # 将张量移到 CPU，转换为 NumPy 数组，再转换为列表
                    sample_dict[key] = tensor[i].cpu().numpy().tolist()
                
                # 2. 处理非张量数据 (batch.non_tensor_batch)
                for key, value_array in batch.non_tensor_batch.items():
                    sample_value = value_array[i]
                    # 处理 NumPy 的特殊类型，如 np.str_
                    if hasattr(sample_value, 'item'):
                        sample_value = sample_value.item()
                    sample_dict[key] = sample_value
                
                # 3. (可选) 添加元信息作为上下文
                sample_dict['meta_info'] = batch.meta_info
                all_samples.append(sample_dict)
            # 一次性写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                for sample in all_samples:
                    # 将字典转换为 JSON 字符串并写入，后跟换行符
                    f.write(json.dumps(sample) + '\n')
            
            print(f"Successfully dumped batch data for step {self.global_steps} to {filepath}")

        except Exception as e:
            print(f"Error dumping batch data to JSONL: {e}")
    def _prepare_off_policy_from_tgt(self, tgt_inputs_ids: torch.Tensor, gen_batch_off: DataProto, train_batch_size: int, prefix_ratio: float) -> DataProto:
        """
        根据离线数据(tgt_input_ids)和指定的比例(prefix_ratio)来准备off-policy的输入数据。

        Args:
            batch (DataProto): 原始的批次数据，包含 'tgt_input_ids'。
            gen_batch_off (DataProto): 用于构建 off-policy 数据的 DataProto 对象。
            train_batch_size (int): 训练批次大小。
            prefix_ratio (float): 要使用的 tgt 序列的长度比例 (0.0 到 1.0)。

        Returns:
            DataProto: 更新了 'input_ids', 'attention_mask', 'position_ids' 的 gen_batch_off。
        """
        original_prompts = gen_batch_off.batch.pop('input_ids')
        original_attention_mask = gen_batch_off.batch.pop('attention_mask')
        original_position_ids = gen_batch_off.batch.pop('position_ids')
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id
        #print("pad_token_id:", pad_token_id, "eos_token_id:", eos_token_id)
        # 1. 预处理 tgt，移除 padding
        tgt_list = [
            _pre_process_inputs_right_pad(pad_token_id, tgt_inputs_ids[i]) for i in range(train_batch_size)
        ]
        #print("tgt_list lengths:", [len(t) for t in tgt_list])
        tgt_list = [
            tgt_list[i] + [eos_token_id,] if len(tgt_list[i]) > 0 else tgt_list[i]
            for i in range(train_batch_size)
        ]
        #print("tgt_list lengths:", [len(t) for t in tgt_list])
        # 2. 根据 prefix_ratio 截断 tgt 序列
        prefix_list = [tgt_list[i][:int(len(tgt_list[i]) * prefix_ratio)] for i in range(len(tgt_list))]
        
        # 3. 预处理 prompt 并与截断后的 tgt 拼接
        prompt_list = [_pre_process_inputs(pad_token_id, original_prompts[i]) for i in range(train_batch_size)]

        concatenated_sequences = [prompt_list[i] + prefix_list[i] for i in range(len(prompt_list))]

        # 4. 将拼接后的序列进行右对齐填充
        max_input_len = max([len(seq) for seq in concatenated_sequences]) if concatenated_sequences else 0
        off_policy_input_ids = torch.full(
            (train_batch_size, max_input_len), 
            pad_token_id, 
            dtype=original_prompts.dtype, 
            device=original_prompts.device
        )
        #打印每一个序列的最后一个token_id
        for i, seq in enumerate(concatenated_sequences):
            off_policy_input_ids[i, -len(seq):] = torch.tensor(seq, dtype=original_prompts.dtype)

        # 5. 重新计算 attention_mask 和 position_ids
        off_policy_attention_mask, off_policy_position_ids = generate_masks_from_input_ids(
            off_policy_input_ids, 
            pad_token_id, 
            original_attention_mask.dtype,
        )

        # 6. 更新 gen_batch_off
        gen_batch_off.batch['input_ids'] = off_policy_input_ids
        gen_batch_off.batch['attention_mask'] = off_policy_attention_mask
        gen_batch_off.batch['position_ids'] = off_policy_position_ids
        #print("Last token ids of concatenated sequences:", [seq[-1] if len(seq) > 0 else None for seq in gen_batch_off.batch['input_ids'].tolist()])
        
        return gen_batch_off,tgt_list
    
    def _build_hybrid_off_policy_output(
        self,
        n_divide: int,
        gen_batch: DataProto,
        off_responses: torch.Tensor,
        tgt_list: list,
        train_batch_size: int,
        prefix_ratio: float,
    ) -> DataProto:
        """
        根据离线目标(tgt_list)的前缀和模型生成的响应(off_responses)构建混合的off-policy输出。

        Args:
            gen_batch (DataProto): 未经repeat的原始on-policy输入，用于获取prompts。
            off_responses (torch.Tensor): 模型为off-policy prompts生成的响应张量。
            tgt_list (list): 预处理过的离线目标序列列表。
            train_batch_size (int): 原始训练批次大小。
            prefix_ratio (float): 用于从tgt_list中提取前缀的比例。

        Returns:
            DataProto: 构建完成并经过repeat的off-policy输出。
        """
        n_on = n_divide
        n_off = self.config.actor_rollout_ref.rollout.n_off
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id
        device = gen_batch.batch['input_ids'].device
        print("设备:", device)
        dtype = gen_batch.batch['input_ids'].dtype
        print("数据类型:", dtype)
        if off_responses == None:
            off_responses = torch.full((train_batch_size*n_off,1),pad_token_id, dtype=dtype, device=device)
        off_len = n_off*train_batch_size
        assert off_responses.size(0) == off_len, f"off_responses batch size mismatch: expected {off_len}, got {off_responses.size(0)}"
        # 1. 从未经repeat的gen_batch中获取原始prompts
        original_prompts = gen_batch.batch['input_ids'][::n_on]
        original_prompts = original_prompts.repeat_interleave(n_off, dim=0)
        # 2. 根据prefix_ratio从tgt_list中提取前缀
        tgt_prefixes = [
            t[:int(len(t) * prefix_ratio)]
            for t in tgt_list
        ]
        #打印tgt_prefixes中每个元素的长度
        tgt_prefixes_lens = [len(t) for t in tgt_prefixes]
        #print("tgt_prefixes_lens:", tgt_prefixes_lens)
        off_responses_list = [
            _pre_process_inputs_right_pad(pad_token_id, off_responses[i])
            for i in range(off_len)
        ]
        #打印list中每个元素的长度
        #off_responses_lens = [len(res) for res in off_responses_list]
        #print("off_responses_lens:", off_responses_lens)
        # 3. 将tgt前缀和模型生成的off_responses拼接成新的混合responses
        #    需要先将off_responses中的padding去掉
        final_responses = []
        max_response_len = max(self.config.data.max_response_length,off_responses.size(1))
        prefix_mask = torch.zeros([off_len, max_response_len], dtype=torch.bool, device=device)
        
        print("off_prefix_mask size:", prefix_mask.size())
        
        #max_response_len = max(self.config.data.max_response_length,off_responses.size(1))
        for i in range(off_len):
            # 找到off_responses[i]中第一个pad_token的位置，以确定有效长度
            #if tgt_prefixes[i][-1] == eos_token_id:
                #final_responses.append(torch.tensor(tgt_prefixes[i], dtype=dtype))
            #else:
            final_responses.append(torch.tensor(tgt_prefixes[i]+ off_responses_list[i], dtype=dtype))
            prefix_len = min(len(tgt_prefixes[i]), max_response_len)
            prefix_mask[i, :prefix_len] = 1
        # 4. 对混合responses进行左填充
        padded_responses = torch.full(
            (len(final_responses), max_response_len), pad_token_id, dtype=dtype,device=device
        )
        for i, res in enumerate(final_responses):
            copy_len = min(len(res), max_response_len)
            # 从 res 中切取相应长度的内容进行赋值
            padded_responses[i, :copy_len] = res[:copy_len]

        # 5. 构建完整的input_ids, attention_mask, 和 position_ids
        off_input_ids = torch.cat([original_prompts, padded_responses], dim=-1)
        off_attention_mask, off_position_ids = generate_masks_from_input_ids(
            off_input_ids,
            pad_token_id,
            gen_batch.batch['attention_mask'].dtype,
        )

        # 6. 构建DataProto对象
        gen_batch_off_output = DataProto(
            batch=TensorDict(
                {
                "prompts": original_prompts,
                "responses": padded_responses,
                "input_ids": off_input_ids,
                "attention_mask": off_attention_mask,
                "position_ids": off_position_ids,
                "prefix_mask": prefix_mask,
            },
            batch_size = off_len
            ),
        )

        # 7. 根据n_off进行repeat
        return gen_batch_off_output

    def _build_se_off_policy_output(
        self,
        gen_batch: DataProto,
        off_responses: torch.Tensor,
        n_split: int,
        n_off: int,
        train_batch_size: int,
    ) -> DataProto:
        """
        根据模型为se-prompts生成的响应(off_responses)构建off-policy输出。

        Args:
            gen_batch_output (DataProto): 经过 on-policy 生成的输出，用于获取原始 prompts。
            off_responses (torch.Tensor): 模型为 se-prompts 生成的响应张量。
            train_batch_size (int): 原始训练批次大小。

        Returns:
            DataProto: 构建完成的 off-policy 输出，其中 prefix_mask 全为 True。
        """
        
        # 1. 从 on-policy 输出中推导出未经 repeat 的原始 prompts
        original_prompts = gen_batch.batch['input_ids'][::n_split]

        repeated_prompts = original_prompts.repeat_interleave(n_off, dim=0)
        #打印有效的responses 长度
        # responses_list = [
        #     _pre_process_inputs_right_pad(self.tokenizer.pad_token_id, off_responses[i])
        #     for i in range(off_responses.size(0))
        # ]
        #print("responses_list lengths:", [len(res) for res in responses_list])
        # 3. 构建完整的 input_ids, attention_mask, 和 position_ids
        off_input_ids = torch.cat([repeated_prompts, off_responses], dim=-1)
        
        attention_mask_dtype = gen_batch.batch['attention_mask'].dtype
        #position_ids_dtype = gen_batch.batch['position_ids'].dtype
        
        off_attention_mask, off_position_ids = generate_masks_from_input_ids(
            off_input_ids,
            self.tokenizer.pad_token_id,
            attention_mask_dtype,
        )
        # 打印responses的长度
        #print("off_responses lengths:", [len(res) for res in off_responses.tolist()])
        # 4. 创建一个与 responses 维度相同且全为 True 的 prefix_mask
        prefix_mask = torch.ones_like(off_responses, dtype=torch.bool, device=off_responses.device)

        # 5. 构建 DataProto 对象
        gen_batch_off_output = DataProto(
            batch=TensorDict(
            {
                "prompts": repeated_prompts,
                "responses": off_responses,
                "input_ids": off_input_ids,
                "attention_mask": off_attention_mask,
                "position_ids": off_position_ids,
                "prefix_mask": prefix_mask,
            },
            batch_size=n_off*train_batch_size
            ),
        )
        
        return gen_batch_off_output
    def _merge_on_off_policy_batches(
        self,
        on_policy_batch: DataProto,
        off_policy_batch: DataProto,
        train_batch_size: int,
        n_on: int,
        n_off: int,
    ) -> DataProto:
        """
        合并 on-policy 和 off-policy 的生成结果。

        Args:
            on_policy_batch (DataProto): on-policy 生成的批次数据。
            off_policy_batch (DataProto): off-policy 生成的批次数据。
            train_batch_size (int): 原始训练批次大小。
            n_on (int): 每个 prompt 的 on-policy 样本数。
            n_off (int): 每个 prompt 的 off-policy 样本数。

        Returns:
            DataProto: 合并后的批次数据。
        """
        merged_batch = TensorDict(batch_size=train_batch_size * (n_on + n_off))
        for key in on_policy_batch.batch.keys():
            on_policy_tensor = on_policy_batch.batch[key]
            # off_policy_batch 可能没有 on-policy 的所有 key，例如 'rollout_log_probs'
            if key not in off_policy_batch.batch:
                continue
            off_policy_tensor = off_policy_batch.batch[key]

            # (train_batch_size * n_on, ...) -> (train_batch_size, n_on, ...)
            on_policy_tensor_reshaped = on_policy_tensor.view(train_batch_size, n_on, *on_policy_tensor.shape[1:])
            # (train_batch_size * n_off, ...) -> (train_batch_size, n_off, ...)
            off_policy_tensor_reshaped = off_policy_tensor.view(train_batch_size, n_off, *off_policy_tensor.shape[1:])

            # 沿 n_on/n_off 维度拼接: (train_batch_size, n_on + n_off, ...)
            combined_tensor = torch.cat([on_policy_tensor_reshaped, off_policy_tensor_reshaped], dim=1)

            # 恢复 batch 维度: (train_batch_size * (n_on+n_off), ...)
            merged_batch[key] = combined_tensor.view(train_batch_size * (n_on + n_off), *combined_tensor.shape[2:])

        return DataProto(batch=merged_batch, meta_info=on_policy_batch.meta_info)

    def _replace_failed_on_policy_with_off_policy(
        self,
        on_policy_batch: DataProto,
        off_policy_batch: DataProto,
        #reward_info: dict,
        train_batch_size: int,
        n: int,
        n_off: int,
    ) -> DataProto:
        """
        在 se_filter 模式下，检查每个问题（包含 n 个采样）是否全部回答错误。
        如果是，则用离线数据 off_policy_batch 中对应的条目替换该问题在 on_policy_batch 中的最后一个采样。
        """
        # 计算当前生成的reward
        # 将 reward_info 按插入的方式重复 n 次
        # reward_info 是一个列表，我们需要将每个元素重复 n 次
        #repeated_reward_info = [item for item in reward_info for _ in range(n)]

        #on_policy_batch.non_tensor_batch['reward_model'] = repeated_reward_info
        reward_tensor_on, _ = compute_reward(on_policy_batch, self.reward_fn)
        
        # reward_tensor_on 是 token-level 的，需要求和得到每个样本的得分
        reward_sum = reward_tensor_on.sum(dim=-1)
        
        # reshape reward to (train_batch_size, n)
        rewards_reshaped = reward_sum.view(train_batch_size, n)
        
        success_value = 1
        
        # 检查每个问题是否全部回答错误
        has_success = (rewards_reshaped == success_value).any(dim=1)
        all_wrong_mask = ~has_success
        all_wrong_indices = torch.nonzero(all_wrong_mask).squeeze(-1)
        
        if all_wrong_indices.numel() > 0:
            print(f"Replacing {all_wrong_indices.numel()} problems' last response with off-policy data.")
            
            # 替换每个全错问题的最后一个回复
            target_indices = all_wrong_indices * n + (n - 1)
            
            # gen_batch_off_output 的结构是 (train_batch_size * n_off)
            # 我们取每个问题的第一个 off-policy 回复
            source_indices = all_wrong_indices * n_off
            
            # 确保 on_policy_batch 中有 off_old_log_probs 字段
            if 'off_old_log_probs' not in on_policy_batch.batch:
                on_policy_batch.batch['off_old_log_probs'] = torch.zeros_like(on_policy_batch.batch['responses'], dtype=torch.float32)

            for key in off_policy_batch.batch.keys():
                if key in on_policy_batch.batch:
                    on_policy_batch.batch[key][target_indices] = off_policy_batch.batch[key][source_indices]
                else:
                    print(f"Key {key} not found in on_policy_batch; skipping replacement for this key.")

                # elif key == 'off_old_log_probs':
                #     on_policy_batch.batch[key][target_indices] = off_policy_batch.batch[key][source_indices]
        
        return on_policy_batch
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        #这一块儿的代码好像没啥用
        n_samples = self.config.actor_rollout_ref.rollout.n
        if self.config.data.get('add_tgt_with_acc', False):
            n_samples = n_samples - 1 # if filter tgt with acc, we either use tgt or on policy samples.

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                #break
                do_profile = self.global_steps in self.config.trainer.profile_steps if self.config.trainer.profile_steps is not None else False
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                train_batch_size = batch.batch['input_ids'].size(0)
                #metrics["global_steps"] = self.global_steps
                #解码打印一条batch se_input_ids中的数据
                if self.global_steps == 1:
                    pass
                #     if 'se_input_ids' in batch.batch:
                #         print("se_input_ids[0]:", self.tokenizer.decode(batch.batch['se_input_ids'][0], skip_special_tokens=True))
                #     if 'input_ids' in batch.batch:
                #         print("input_ids[0]:", self.tokenizer.decode(batch.batch['input_ids'][0], skip_special_tokens=True))
                # #break
                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids",]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if self.global_steps == 1:
                    print("Tensor batch keys:", batch.batch.keys(), "NonTensor batch keys:", batch.non_tensor_batch.keys())
                #没有
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                #没有
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                #没有
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                #没有
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    #non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                n_on = self.config.actor_rollout_ref.rollout.n - self.config.actor_rollout_ref.rollout.n_off
                n_off = self.config.actor_rollout_ref.rollout.n_off

                prefix_ratio = self.config.actor_rollout_ref.rollout.prefix_ratio
                if n_off>0:
                    tgt_list = None
                    #提取off-policy的回复内容
                    gen_batch_off = deepcopy(gen_batch)
                    #if 'tgt_input_ids' in batch.batch and self.config.actor_rollout_ref.rollout.n_prefix>0:
                    #se_tgt_input_ids = None
                    se_tgt_list = None
                    if 'tgt_input_ids' in batch.batch:
                        tgt_inputs_ids = deepcopy(batch.batch['tgt_input_ids'])
                        
                        gen_batch_off_standard,tgt_list = self._prepare_off_policy_from_tgt(tgt_inputs_ids, gen_batch_off, train_batch_size, prefix_ratio)
            
                        if  self.config.actor_rollout_ref.rollout.n_prefix>0:
                            gen_batch_off = gen_batch_off_standard
                    if 'se_input_ids' in batch.batch and self.config.actor_rollout_ref.rollout.n_se>0:
                        gen_batch_off = batch.pop(
                            batch_keys=["se_input_ids", "se_attention_mask", "se_position_ids"],
                        )
                        # 重命名键，去掉 "se_" 前缀
                        gen_batch_off.batch["input_ids"] = gen_batch_off.batch.pop("se_input_ids")
                        #打印一条解码后的input_ids
                        #print("gen_batch_off input_ids[0]:", self.tokenizer.decode(gen_batch_off.batch['input_ids'][0], skip_special_tokens=False))
                        gen_batch_off.batch["attention_mask"] = gen_batch_off.batch.pop("se_attention_mask")
                        gen_batch_off.batch["position_ids"] = gen_batch_off.batch.pop("se_position_ids")
                        if 'se_tgt_input_ids' in batch.batch:
                            se_tgt_input_ids = batch.batch.pop('se_tgt_input_ids')
                            pad_token_id = self.tokenizer.pad_token_id
                            eos_token_id = self.tokenizer.eos_token_id

                            se_tgt_list = [
                                _pre_process_inputs_right_pad(pad_token_id, se_tgt_input_ids[i]) for i in range(train_batch_size)
                            ]
                            #print("tgt_list lengths:", [len(t) for t in tgt_list])
                            se_tgt_list = [
                                se_tgt_list[i] + [eos_token_id,] if len(se_tgt_list[i]) > 0 else se_tgt_list[i]
                                for i in range(train_batch_size)
                            ]
                             
                    else:
                        raise ValueError("When using off-policy samples, the training batch must contain 'tgt_input_ids' or 'se_input_ids'.")
                    
                    gen_batch_off = gen_batch_off.repeat(repeat_times=n_off, interleave=True)
                    print("使用off-policy样本,batch_size*n_off:", len(gen_batch_off.batch['input_ids']))
                    #print("Last token ids of concatenated sequences:", [seq[-1] if len(seq) > 0 else None for seq in gen_batch_off.batch['input_ids'].tolist()])
                
                
                if self.config.actor_rollout_ref.actor.policy_loss.loss_mode=='se_filter':
                    gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                else:
                    gen_batch = gen_batch.repeat(repeat_times=n_on, interleave=True)
                gen_batch.meta_info['global_steps'] = self.global_steps
                gen_batch.meta_info['is_se'] = False
                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen_on", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                        #为gen_batch-output添加prefix_mask字段，表示哪些response token是离线的
                        gen_batch_output.batch['prefix_mask'] = torch.zeros((gen_batch_output.batch['responses'].size(0), gen_batch_output.batch['responses'].size(1)),
                                           dtype=torch.bool) # empty dummy tensor
                    #import time
                    #time.sleep(5) # wait for a while to make the logs more readable
                    with marked_timer("gen_off", timing_raw, color="blue"):
                        if self.config.actor_rollout_ref.rollout.n_off>0:
                            #为每个prompt构建一个具有标准答案的离线样本，大小为train_batch_size
                            assert n_off==1, "Currently we only support n_off=1."
                            if self.config.actor_rollout_ref.actor.policy_loss.loss_mode=="se_luffy":
                            
                                gen_batch_off_output_standard= self._build_hybrid_off_policy_output(
                                        n_on,
                                        gen_batch,
                                        None,
                                        tgt_list,
                                        train_batch_size,
                                        prefix_ratio,
                                    )
                                gen_batch_off_output_standard.batch['off_old_log_probs'] = torch.zeros((gen_batch_off_output_standard.batch['responses'].size(0), gen_batch_off_output_standard.batch['responses'].size(1)),dtype=torch.float32)
                                #为什么要将将非张量数据也复制过去，不是空的吗
                                gen_batch_off_output_standard.non_tensor_batch = deepcopy(batch.non_tensor_batch)
                                print("non_tensor_batch keys:", gen_batch_off_output_standard.non_tensor_batch.keys())
            
                            if self.config.actor_rollout_ref.rollout.n_prefix>0:
                                if self.config.actor_rollout_ref.rollout.prefix_ratio<1.0:
                                    #raise ValueError("When using n_prefix>0, prefix_ratio must be in (0.0, 1.0].")
                                    gen_batch_off_output = self.actor_rollout_wg.generate_sequences(gen_batch_off)
                                    timing_raw.update(gen_batch_off_output.meta_info["timing"])
                                    gen_batch_off_output.meta_info.pop("timing", None)
                                # #打印一下解码的前五条inputs_ids
                                # for i in range(min(3, len(gen_batch_off_output.batch['input_ids']))):
                                #     print(f"gen_batch_off_output input_ids[{i}]:", self.tokenizer.decode(gen_batch_off_output.batch['input_ids'][i], skip_special_tokens=False))
                                    off_responses = gen_batch_off_output.batch['responses']
                                else:
                                    off_responses = None
                                
                                gen_batch_off_output= self._build_hybrid_off_policy_output(
                                    n_on,
                                    gen_batch,
                                    off_responses,
                                    tgt_list,
                                    train_batch_size,
                                    prefix_ratio,
                                )
                                
                            elif self.config.actor_rollout_ref.rollout.n_se>0:

                                gen_batch_output.batch['off_old_log_probs'] = torch.zeros((gen_batch_output.batch['responses'].size(0), gen_batch_output.batch['responses'].size(1)),
                                           dtype=torch.float32)
                                

                                gen_batch_off.meta_info['is_se'] = True
                                se_target = self.config.se.get("target", None)
                                if se_target == "standard":
                                    #se_responses =  se_tgt_input_ids
                                    gen_batch_off_output = self._build_hybrid_off_policy_output(
                                        n_off,
                                        gen_batch_off,
                                        None,
                                        se_tgt_list,
                                        train_batch_size,
                                        1,
                                    )
                                elif se_target == "dynamic":
                                    gen_batch_off_output = self.actor_rollout_wg.generate_sequences(gen_batch_off)
                                    timing_raw.update(gen_batch_off_output.meta_info["timing"])
                                    gen_batch_off_output.meta_info.pop("timing", None)
                                    # #打印一下解码的前五条inputs_ids
                                    # for i in range(min(3, len(gen_batch_off_output.batch['input_ids']))):
                                    #     print(f"gen_batch_off_output input_ids[{i}]:", self.tokenizer.decode(gen_batch_off_output.batch['input_ids'][i], skip_special_tokens=False))
                                    #打印prompts的有效长度
                                    #print("gen_batch_off_output prompts lengths:", (gen_batch_off_output.batch['prompts']!=self.tokenizer.pad_token_id).sum(dim=-1).tolist())
                                elif se_target == "synch": #
                                    gen_batch_off_output = self.actor_rollout_se_wg.generate_sequences(gen_batch_off)
                                    timing_raw.update(gen_batch_off_output.meta_info["timing"])
                                    gen_batch_off_output.meta_info.pop("timing", None)
                                    
                                se_responses = gen_batch_off_output.batch['responses']


                                    #将解码后的inputs_ids保存到文件中
                                se_dir = self.config.trainer.get("rollout_data_dir", None)
                                if se_dir is not None and se_dir != "" :
                                    inputs = self.tokenizer.batch_decode(gen_batch_off_output.batch["prompts"], skip_special_tokens=True)
                                    outputs = self.tokenizer.batch_decode(gen_batch_off_output.batch["responses"], skip_special_tokens=True)
                                    #se_dir = self.config.trainer.get("rollout_data_dir", None)
                                    os.makedirs(se_dir, exist_ok=True)
                                    with open(f"{se_dir}/se_generation_step_{self.global_steps}.jsonl", "w", encoding="utf-8") as f:
                                        for inp, out in zip(inputs, outputs):
                                            json_line = json.dumps({"input": inp, "output": out}, ensure_ascii=False)
                                            f.write(json_line + "\n")
                                
                                
                                if se_target in ["standard",'synch']:
                                    #如果是标准答案，则不需要计算旧的log概率
                                    off_old_log_probs = self.actor_rollout_se_wg.compute_ref_log_prob(gen_batch_off_output)
                                    se_old_log_probs = off_old_log_probs.batch.pop('ref_log_prob')
                                else:
                                    off_old_log_probs = self.actor_rollout_wg.compute_log_prob(gen_batch_off_output)
                                    se_old_log_probs = off_old_log_probs.batch.pop('old_log_probs')

                                gen_batch_off_output = self._build_se_off_policy_output(
                                    gen_batch,
                                    se_responses,
                                    self.config.actor_rollout_ref.rollout.n if self.config.actor_rollout_ref.actor.policy_loss.loss_mode=='se_filter' else n_on,
                                    n_off,
                                    train_batch_size,
                                )
                                gen_batch_off_output.batch['off_old_log_probs'] = se_old_log_probs
                    
                            else:
                                raise ValueError("When using off-policy samples, n_prefix or n_se must be greater than 0.")
                            #gen_batch_off_output.batch['off_old_log_probs'] = off_old_log_probs.batch.pop('old_log_probs')
                    #合并on-policy和off-policy的生成结果
                    if self.config.actor_rollout_ref.rollout.n_off>0:
                        if self.config.actor_rollout_ref.actor.policy_loss.loss_mode=='se_filter':
                            # 获取 batch 中所有的 non_tensor_batch
                            non_tensor_batch = batch.non_tensor_batch
                            
                            # 将 non_tensor_batch 中的每个元素重复 n 次
                            repeated_non_tensor_batch = {}
                            n = self.config.actor_rollout_ref.rollout.n
                            for k, v in non_tensor_batch.items():
                                if isinstance(v, list):
                                    repeated_non_tensor_batch[k] = [item for item in v for _ in range(n)]
                                elif isinstance(v, np.ndarray):
                                    repeated_non_tensor_batch[k] = np.repeat(v, n, axis=0)
                                else:
                                    # 其他类型根据需要处理，这里假设直接复制
                                    repeated_non_tensor_batch[k] = v # 可能需要更复杂的处理

                            # 将重复后的 non_tensor_batch 赋值给 gen_batch_output
                            gen_batch_output.non_tensor_batch = repeated_non_tensor_batch

                            #reward_info = batch.non_tensor_batch.get('reward_model', None)
                            gen_batch_output = self._replace_failed_on_policy_with_off_policy(
                                on_policy_batch=gen_batch_output,
                                off_policy_batch=gen_batch_off_output,
                                train_batch_size=train_batch_size,
                                n=self.config.actor_rollout_ref.rollout.n,
                                n_off=n_off,
                            )
                        else:
                            gen_batch_output = self._merge_on_off_policy_batches(
                                on_policy_batch=gen_batch_output,
                                off_policy_batch=gen_batch_off_output,
                                train_batch_size=train_batch_size,
                                n_on=n_on,
                                n_off=n_off,
                            )
                    assert len(gen_batch_output.batch['input_ids']) == train_batch_size * self.config.actor_rollout_ref.rollout.n, \
                        f"Expected {train_batch_size * self.config.actor_rollout_ref.rollout.n} input_ids, but got {len(gen_batch_output.batch['input_ids'])}"
                    #打印合并后的prefix_mask列表：
                    #print("合并后的prefix_mask:", gen_batch_output.batch['prefix_mask'].any(-1))
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    print("聚合后的batch_size:", len(gen_batch_output.batch['input_ids']))
                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if self.global_steps==1 and rollout_data_dir is not None and rollout_data_dir != "":
                        self._dump_batch_to_jsonl(batch, rollout_data_dir)
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    if n_off>0:
                        if self.config.actor_rollout_ref.rollout.n_prefix>0:
                            batch.batch['se_mask'] = torch.zeros((batch.batch['responses'].size(0), batch.batch['responses'].size(1)),
                                           dtype=torch.bool)
                        elif self.config.actor_rollout_ref.rollout.n_se>0:
                            batch.batch['se_mask'] = deepcopy(batch.batch['prefix_mask'])
                    else:
                        batch.batch['se_mask'] = torch.zeros((batch.batch['responses'].size(0), batch.batch['responses'].size(1)),
                                           dtype=torch.bool)
                    metrics['batch/avg_prefix_ratio'] = prefix_ratio
                    
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        print("Reward tensor shape:", reward_tensor.shape)
                        batch.batch["token_level_scores"] = reward_tensor

                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch['uid']
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        if self.config.data.reward_impl_version == 3 or self.config.data.reward_impl_version == 4:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        else:
                            raise ValueError(f'Invalid reward implementation version: {self.config.data.reward_impl_version}')
                        solve_none = 0
                        solve_all = 0
                        solve_none_format = 0
                        solve_one = 0
                        #if self.config.actor_rollout_ref.actor.policy_loss.loss_mode=="se_filter":
                        reward_mask = torch.ones((batch.batch['responses'].size(0), batch.batch['responses'].size(1)), dtype=torch.bool)

                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence
                            
                            # Check if all rewards are 0 or all are 1 for this uid
                            if (uid_rewards == fail_value).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards == success_value).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1
                            elif (uid_rewards == format_value).all():
                                valid_mask[uid_mask] = False
                                solve_none_format += 1
                            # 如果只有一个对的
                            elif (uid_rewards == success_value).sum() == 1:
                                reward_mask[uid_mask,:] = True
                                solve_one += 1
                        if not self.config.algorithm.get("filter_reward",True):
                            # if self.config.trainer.skip_valid_mask:
                            valid_mask[:] = True
                            # Log to metrics
                        reward_sum_per_sample = reward_tensor.sum(-1)

                        # 2. 将该张量转换为 Python 列表并存入 non_tensor_batch
                        #    使用 .cpu() 确保数据在 CPU 上，.tolist() 转换为列表
                        batch.non_tensor_batch['reward_sum'] = reward_sum_per_sample.unsqueeze(1).cpu().numpy()
                        metrics['batch/solve_none'] = solve_none
                        metrics['batch/solve_none_format'] = solve_none_format
                        metrics['batch/solve_all'] = solve_all
                        metrics['batch/solve_one'] = solve_one
                        batch.batch['reward_mask'] = reward_mask
                        # add more metrics
                        metrics['batch/solved'] = (reward_tensor.sum(-1) == success_value).sum().item() / len(uids)
                        metrics['batch/failed'] = (reward_tensor.sum(-1) == fail_value).sum().item() / len(uids)
                        # add on-policy metrics
                        prefix_mask = batch.batch['prefix_mask']
                        off_policy_mask = prefix_mask.any(-1)
                        on_policy_mask = ~off_policy_mask
                        metrics['batch/on_solved'] = (reward_tensor[on_policy_mask].sum(-1) == success_value).sum().item() / (on_policy_mask.sum().item() + 1e-6)
                        metrics['batch/off_solved'] = (reward_tensor[off_policy_mask].sum(-1) == success_value).sum().item() / (off_policy_mask.sum().item() + 1e-6)
                        

                    if self.config.actor_rollout_ref.actor.policy_loss.loss_mode=="se_luffy":
                        #将off_policy中所有错误的样本替换为相应的gen_batch_off_standard中的样本
                        
                        reward_standard,_ = compute_reward(gen_batch_off_output_standard, self.reward_fn)
                        gen_batch_off_output_standard.batch['token_level_scores'] = reward_standard
                        
                        # 在 off-policy 样本中，找到奖励为错误值的样本
                        incorrect_off_policy_mask = (
                            (reward_sum_per_sample == fail_value)
                        ) & off_policy_mask
                        
                        # 2. 找到长度过短的样本,最小长度为当前se的平均长度
                        min_response_len = 1000 #int((batch.batch['response_mask'][off_policy_mask]).sum().item())/ (off_policy_mask.sum().item() + 1e-6)
                        #print(f"Minimum response length for off-policy samples: {min_response_len:.2f}")
                        response_lengths = batch.batch["response_mask"].sum(dim=1)
                        short_off_policy_mask = (response_lengths < min_response_len) & off_policy_mask
                        replace_mask = incorrect_off_policy_mask | short_off_policy_mask
                        #统计一下 正确但太短的样本数量
                        num_short_but_correct = (short_off_policy_mask & (reward_sum_per_sample == success_value)).sum().item()
                        metrics['batch/off_short_but_correct'] = num_short_but_correct

                        incorrect_indices_in_batch = torch.where(replace_mask)[0]
                        print(f"错误样本indices in batch: {incorrect_indices_in_batch.tolist()}")
                        if incorrect_indices_in_batch.numel() > 0:
                            print(f"Replacing {incorrect_indices_in_batch.numel()} incorrect off-policy samples.")
                            for idx in incorrect_indices_in_batch:
                                standard_sample_pos = idx // self.config.actor_rollout_ref.rollout.n 
                                #standard_sample = gen_batch_off_output_standard.get_item(standard_sample_pos)
                                for key in gen_batch_off_output_standard.batch.keys():
                                    if key in batch.batch.keys():
                                        batch.batch[key][idx] = gen_batch_off_output_standard.batch[key][standard_sample_pos]
                                batch.batch['se_mask'][idx] = torch.zeros_like(batch.batch['se_mask'][idx])
                        else:
                            print("No incorrect off-policy samples to replace.")
                    
                    se_mask = batch.batch['se_mask'].any(-1)
                    standard_off_policy_mask = off_policy_mask & (~se_mask)
                    metrics['batch/on_tokens'] = int((batch.batch['response_mask'][on_policy_mask]).sum().item())
                    metrics['batch/on_avg_response_len'] = (batch.batch['response_mask'][on_policy_mask].sum().item())/(on_policy_mask.sum().item()+1e-6)
                    metrics['batch/off_standard_tokens'] = int((batch.batch['response_mask'][standard_off_policy_mask]).sum().item())
                    metrics['batch/off_standard_avg_response_len'] = (batch.batch['response_mask'][standard_off_policy_mask].sum().item())/(standard_off_policy_mask.sum().item()+1e-6)
                    metrics['batch/off_se_tokens'] = int((batch.batch['response_mask'][se_mask]).sum().item())
                    metrics['batch/off_se_avg_response_len'] = (batch.batch['response_mask'][se_mask].sum().item())/(se_mask.sum().item()+1e-6)
                    #打印一下se的最终个数和平均response长度
                    print(f"SE samples in the batch: {se_mask.sum().item()}, average response length: {batch.batch['response_mask'][se_mask].sum().item()/(se_mask.sum().item()+1e-6):.2f}")
                    
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        # 计算standard_off_policy_mask上的old_log_prob的均值
                        #old_log_prob_metrics = {}
                        #old_log_prob
                        old_probs_off = torch.exp(old_log_prob.batch["old_log_probs"])
                        old_prob_standard_off = verl_F.masked_mean(old_probs_off[standard_off_policy_mask], response_masks[standard_off_policy_mask])
                        old_prob_se_off = verl_F.masked_mean(old_probs_off[se_mask], response_masks[se_mask])
                        #print(f"Average old_log_prob on standard off-policy samples before update policy: {old_prob_standard_off:.4f}")
                        #print()
                        #根据prefix_mask将old_log_prob中off_policy的部分替换为old_log_prob_off
                        if "off_old_log_probs" in batch.batch:
                            print(f"{off_policy_mask.sum().item()} off-policy samples in the batch.")
                            old_log_probs_rollout = batch.batch.pop("off_old_log_probs")
                            old_log_prob.batch["old_log_probs"][off_policy_mask] = old_log_probs_rollout[off_policy_mask]
                        
                        batch = batch.union(old_log_prob)

                        def _safe_entropy_avg(sample_mask: torch.Tensor) -> float:
                            if sample_mask.any():
                                e = agg_loss(
                                        loss_mat=entropys[sample_mask],
                                        loss_mask=response_masks[sample_mask],
                                        loss_agg_mode='token-mean',
                                )
                                return float(e.detach().item())
                            return 0.0
                        # se_mask = batch.batch['se_mask'].any(-1)
                        # standard_off_policy_mask = off_policy_mask & (~se_mask)
                        entropy_on = _safe_entropy_avg(on_policy_mask)
                        entropy_off_standard = _safe_entropy_avg(off_policy_mask*(~se_mask))
                        entropy_off_se = _safe_entropy_avg(se_mask)
                        old_probs = torch.exp(batch.batch["old_log_probs"])
                        old_on_prob = verl_F.masked_mean(old_probs[on_policy_mask], response_masks[on_policy_mask])
                        old_off_prob_se = verl_F.masked_mean(old_probs[se_mask], response_masks[se_mask])  
                        #old_off_prob_standard = verl_F.masked_mean(old_probs[standard_off_policy_mask], response_masks[standard_off_policy_mask])  
                        #计算old_on_prob和old_off_prob的平均值
            
                        old_log_prob_metrics.update(
                                {
                                    "batch/entropy_on": entropy_on,
                                    "batch/entropy_off_standard": entropy_off_standard,
                                    "batch/entropy_off_se": entropy_off_se,
                                    "batch/old_prob_on": old_on_prob.detach().item(),
                                    "batch/old_prob_off_standard": old_prob_standard_off.detach().item(),
                                    "batch/old_prob_se": old_off_prob_se.detach().item(),
                                    "batch/old_prob_se_off": old_prob_se_off.detach().item(),
                                }
                            )
                        metrics.update(old_log_prob_metrics)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        #batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available，默认为False
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor
                        # 返回token级别的advantages和returns
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    logger.log(data=metrics, step=self.global_steps)
                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                        log_prob = actor_output.batch.get("log_prob", None)
                        batch.batch["log_probs"] = log_prob
                        #根据prefix_mask和response_mask计算on_ratio和off_ratio的std
                        
                        if actor_output is not None:
                            prob_ratio_metrics = self._compute_ratio_metrics(actor_output, batch)
                            metrics.update(prob_ratio_metrics)
                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                metrics_data_dir = self.config.trainer.get("metrics_data_dir", None)
                if metrics_data_dir is not None and metrics_data_dir != "":
                    metrics_filepath = os.path.join(metrics_data_dir, "training_metrics.jsonl")
                    self._log_metrics_to_file(metrics, metrics_filepath)
                
                save_tensors_dir = self.config.trainer.get("save_tensors_dir", None)
                if save_tensors_dir is not None and save_tensors_dir != "":
                    keys_to_save = ["old_log_probs", "log_probs","prefix_mask","reward_sum","uid",'se_mask','response_mask']
                    save_dtype = self.config.trainer.get("save_tensor_dtype", "fp16")
                    self._save_tensors_from_batch(batch, save_dir=save_tensors_dir, keys_to_save=keys_to_save, save_dtype=save_dtype)
                progress_bar.update(1)
                self.global_steps += 1

                #sleep for a while to let the logging finish
                
                time.sleep(10)
                sys.stdout.flush()
                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    def calculate_probs_entropy(self):
        """
        计算并返回当前策略的平均概率和熵。
        该方法可用于监控训练过程中策略的变化。
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        if self.config.trainer.get("max_training_steps", None) and self.config.trainer.max_training_steps > 0:
            self.total_training_steps = min(self.config.trainer.max_training_steps, self.total_training_steps)
        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        #这一块儿的代码好像没啥用
       
        total_entropy_list = []
        total_probs_list = []
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                #break
                
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                train_batch_size = batch.batch['input_ids'].size(0)
                #metrics["global_steps"] = self.global_steps
                #解码打印一条batch se_input_ids中的数据
                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids",]
                
                if self.global_steps == 1:
                    print("Tensor batch keys:", batch.batch.keys(), "NonTensor batch keys:", batch.non_tensor_batch.keys())
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    #non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                tgt_list = None
                #提取off-policy的回复内容
                gen_batch_off = deepcopy(gen_batch)
                #if 'tgt_input_ids' in batch.batch and self.config.actor_rollout_ref.rollout.n_prefix>0:
                if 'tgt_input_ids' in batch.batch:
                    tgt_inputs_ids = batch.batch.pop('tgt_input_ids')

                    _,tgt_list = self._prepare_off_policy_from_tgt(tgt_inputs_ids, gen_batch_off, train_batch_size, 1.0)

                #gen_batch = gen_batch.repeat(repeat_times=n_on, interleave=True)
                
                gen_batch.meta_info['global_steps'] = self.global_steps
                #gen_batch.meta_info['is_se'] = False
                is_last_step = self.global_steps >= self.total_training_steps

                original_prompts = gen_batch.batch['input_ids']

                pad_token_id = self.tokenizer.pad_token_id
                device = original_prompts.device
                dtype = original_prompts.dtype
                final_responses = []
                max_response_len = self.config.data.max_response_length
                for i in range(train_batch_size):
                    final_responses.append(torch.tensor(tgt_list[i], dtype=dtype))
                padded_responses = torch.full(
                    ( len(final_responses), max_response_len), pad_token_id, dtype=dtype,device=device
                )
                for i, res in enumerate(final_responses):
                    copy_len = min(len(res), max_response_len)
                    # 从 res 中切取相应长度的内容进行赋值
                    padded_responses[i, :copy_len] = res[:copy_len]
                # 5. 构建完整的input_ids, attention_mask, 和 position_ids
                off_input_ids = torch.cat([original_prompts, padded_responses], dim=-1)
                off_attention_mask, off_position_ids = generate_masks_from_input_ids(
                    off_input_ids,
                    pad_token_id,
                    gen_batch.batch['attention_mask'].dtype,
                )

                # 6. 构建DataProto对象
                gen_batch_off_output = DataProto(
                    batch=TensorDict(
                        {
                        "prompts": original_prompts,
                        "responses": padded_responses,
                        "input_ids": off_input_ids,
                        "attention_mask": off_attention_mask,
                        "position_ids": off_position_ids,
                    },
                    batch_size = train_batch_size
                    ),
                    #meta_info=gen_batch.meta_info,
                )
                batch = batch.union(gen_batch_off_output)
                #打印一下meta_info的keys
                if self.global_steps == 1:
                    print("Meta info keys in batch:", batch.meta_info.keys())
                batch.batch["response_mask"] = compute_response_mask(batch)


                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                log_probs = old_log_prob.batch["old_log_probs"]
                response_masks = batch.batch["response_mask"]
                
                #计算每个response的平均entropy，probs
                entropy_list = (entropys * response_masks).sum(dim=-1) / response_masks.sum(dim=-1)
                probs_list = (torch.exp(log_probs) * response_masks).sum(dim=-1) / response_masks.sum(dim=-1)
                #计算整个batch的平均entropy，probs
                total_entropy_list.extend(entropy_list.detach().cpu().tolist())
                total_probs_list.extend(probs_list.detach().cpu().tolist())
                
                batch = batch.union(old_log_prob)
                if self.config.trainer.save_tensors_dir is not None and self.config.trainer.save_tensors_dir != "":
                    #outputs = [self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False) for ids in batch.batch["responses"].tolist()]
                    outputs = [[self.tokenizer.decode(i_ids, skip_special_tokens=False) for i_ids in ids ] for ids in batch.batch["responses"].tolist()]
                    #print(outputs[0])
                    batch.non_tensor_batch['decoded_responses'] = outputs
                    keys = ["old_log_probs","response_mask","entropys","responses","decoded_responses","extra_info"]
                    
                    print(f"Saving tensors at step {self.global_steps} to {self.config.trainer.save_tensors_dir}")
                    self._save_tensors_from_batch(batch, save_dir=self.config.trainer.save_tensors_dir, keys_to_save=keys, save_dtype="fp32",extra_info=True,train_batch_size=train_batch_size)


                progress_bar.update(1)
                self.global_steps += 1 
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    #保存平均entropy和probs到文件
                    self._log_metrics_to_file({
                        "model_path":self.config.actor_rollout_ref.model.path,
                        "total_entropy": total_entropy_list,
                        "total_probs": total_probs_list,

                    }, self.config.trainer.log_file)
                    progress_bar.close()
                    return

    def _save_tensors_from_batch(self, batch: DataProto, save_dir: str = None, keys_to_save: List[str] = None, save_dtype: str = "fp16", extra_info: bool = False, train_batch_size: int = 0):
        """
        如果配置了保存目录和要保存的键，则将 batch 中指定的张量/数据保存到本地文件。
        
        需要配置:
        - trainer.save_tensor_dir: 保存文件的目录。
        - trainer.save_tensor_keys: 一个列表，包含要从 batch 中保存的键名。
        - trainer.save_tensor_dtype (可选): 保存精度，如 "fp16", "bf16", "fp32"。
        """

        # 如果未配置目录或要保存的键，则直接返回
        if not save_dir or not keys_to_save:
            return

        try:
            # 确保目录存在
            os.makedirs(save_dir, exist_ok=True)
            # 从配置获取保存精度，默认为 fp16 以节省空间
            save_dtype = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }.get(save_dtype, torch.float16)

            # 组装需要保存的张量和元数据
            save_dict = {
                "step": int(self.global_steps),
            }
            #batch_size = batch.batch["batch_size"]
            for key in keys_to_save:
                # 优先从张量字典中查找
                if key in batch.batch:
                    tensor_data = batch.batch[key].detach()
                    # 移动到 CPU 并转换类型
                    if key in ["responses"]:
                        save_dict[key] = tensor_data.to(dtype=torch.long, device="cpu")
                    else:
                        save_dict[key] = tensor_data.to(dtype=save_dtype, device="cpu")
                # 如果找不到，再从非张量字典中查找
                elif key in batch.non_tensor_batch:
                    non_tensor_data = batch.non_tensor_batch[key]
                    # 尝试使其可序列化 (例如，将 tuple/numpy array 转为 list)
                    try:
                        save_dict[key] = list(non_tensor_data)
                    except TypeError:
                        save_dict[key] = non_tensor_data
                else:
                    print(f"Key '{key}' not found in batch at step {self.global_steps}, skipping.")
            if extra_info == True:
                for k, v in batch.meta_info.items():
                    print(f"Meta info key: {k}, type: {type(v)}")
                    if k == 'extra_info' and len(v) == train_batch_size:
                        save_dict[k] = v
            # 只有在确实找到了要保存的内容时才执行保存操作
            if len(save_dict) > 1:
                # 定义输出路径并保存
                out_path = os.path.join(save_dir, f"tensors_step_{self.global_steps:07d}.pt")
                torch.save(save_dict, out_path)
        
        except Exception as e:
            print(f"Failed to save tensors at step {self.global_steps}. Error: {e}")
    def _compute_ratio_metrics(self,actor_output: DataProto, batch: DataProto) -> dict:
        """
        根据新的 log_prob 和批次数据计算 on-policy 部分的 ratio 均值和方差。

        Args:
            log_prob (torch.Tensor): 模型更新后计算出的新 log probabilities。
            batch (DataProto): 包含 old_log_probs, response_mask 和 prefix_mask 的批次数据。

        Returns:
            dict: 包含 on-policy ratio 均值和标准差的指标字典。
        """
        log_prob = actor_output.batch.get("log_prob", None)
        reconstruct_prefix_mask = actor_output.batch.get("prefix_mask", None)
        if log_prob is None:
            return {}
        probs = torch.exp(log_prob)
        #probs = log_prob
        response_masks = batch.batch["response_mask"]
        # 计算 off,on,se 的prob的平均值

        prefix_mask = batch.batch["prefix_mask"]
        #判断reconstruct_prefix_mask和prefix_mask是否相等
        # if not torch.equal(reconstruct_prefix_mask, prefix_mask):
        #     print("Warning: reconstruct_prefix_mask and prefix_mask are not equal!")
        #     print(reconstruct_prefix_mask.any(-1).tolist())
        
        off_policy_mask = prefix_mask.any(-1)
        on_policy_mask = ~off_policy_mask
        se_mask = batch.batch['se_mask'].any(-1)
        standard_off_policy_mask = off_policy_mask & (~se_mask)
        on_probs = verl_F.masked_mean(probs[on_policy_mask], response_masks[on_policy_mask])
        on_probs_var = verl_F.masked_var(probs[on_policy_mask], response_masks[on_policy_mask],False)
        off_standard_probs = verl_F.masked_mean(probs[standard_off_policy_mask], response_masks[standard_off_policy_mask])
        #计算所有每条数据的平均prob,得到一个每条数据的平均probs的列表
        off_standard_probs_var = verl_F.masked_var(probs[standard_off_policy_mask], response_masks[standard_off_policy_mask],False)

        probs_list = (probs * response_masks).sum(dim=-1) / response_masks.sum(dim=-1)
        # for i, p in enumerate(probs_list):

        #     print(f"Sample of average probs per sequence {i}:", p.item(),'off_policy:', off_policy_mask[i].item())
        #     #打印非0的response_mask个数
        #     print("Non-zero response_mask count:", response_masks[i].sum().item())
        #     #打印非1的probs个数
        #     print("Non-one probs count:", (probs[i] < 1.0).sum().item())

        #print(probs[standard_off_policy_mask][0].tolist())
        #print(response_masks[standard_off_policy_mask][0].tolist())
        off_se_probs = verl_F.masked_mean(probs[se_mask], response_masks[se_mask])
        off_se_probs_var = verl_F.masked_var(probs[se_mask], response_masks[se_mask],False)
        probs_metrics = {
            "batch/prob_on": on_probs.detach().item(),
            "batch/prob_on_var": on_probs_var.detach().item(),
            "batch/prob_off_standard": off_standard_probs.detach().item(),
            "batch/prob_off_standard_var": off_standard_probs_var.detach().item(),
            "batch/prob_off_se": off_se_probs.detach().item(),
            "batch/prob_off_se_var": off_se_probs_var.detach().item(),
        }
        ratio = torch.exp(log_prob - batch.batch["old_log_probs"])
        clipped_ratio = torch.clamp(ratio, 1e-3, 10)
        # 计算on,off,se的ratio均值和方差
        
        on_ratio_mean = verl_F.masked_mean(clipped_ratio[on_policy_mask], response_masks[on_policy_mask])
        on_ratio_var = verl_F.masked_var(clipped_ratio[on_policy_mask], response_masks[on_policy_mask],False)

        off_standard_ratio_mean = verl_F.masked_mean(clipped_ratio[standard_off_policy_mask], response_masks[standard_off_policy_mask])
        off_standard_ratio_var = verl_F.masked_var(clipped_ratio[standard_off_policy_mask], response_masks[standard_off_policy_mask],False)

        off_se_ratio_mean = verl_F.masked_mean(clipped_ratio[se_mask], response_masks[se_mask])
        off_se_ratio_var = verl_F.masked_var(clipped_ratio[se_mask], response_masks[se_mask],False)

        # 准备要返回的指标
        ratio_metrics = {
            "batch/ratio_on_mean": on_ratio_mean.detach().item(),
            "batch/ratio_on_var": on_ratio_var.detach().item(),
            "batch/ratio_off_standard_mean": off_standard_ratio_mean.detach().item(),
            "batch/ratio_off_standard_var": off_standard_ratio_var.detach().item(),
            "batch/ratio_off_se_mean": off_se_ratio_mean.detach().item(),
            "batch/ratio_off_se_var": off_se_ratio_var.detach().item(),
        }
        return {**probs_metrics, **ratio_metrics}
    
    def _log_metrics_to_file(self, metrics: dict, filepath: str):

        if not filepath:
            return

        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # 清理 metrics 字典，将 tensor 和 numpy 类型转换为基本 Python 类型
            sanitized_metrics = _to_jsonable(metrics)
            
            # 以追加模式打开文件并写入
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(sanitized_metrics) + '\n')

        except Exception as e:
            # 打印错误但不要中断训练
            print(f"Warning: Could not write metrics to {filepath}. Error: {e}")
def _to_jsonable(obj):
        # 基础类型直接返回
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        # PyTorch Tensor
        if isinstance(obj, torch.Tensor):
            t = obj.detach()
            if t.numel() == 1:
                return t.item()
            return t.cpu().tolist()
        # numpy 标量与数组
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # 容器类型递归处理
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(v) for v in obj]
        if isinstance(obj, dict):
            return {str(k): _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, set):
            return [_to_jsonable(v) for v in obj]
        # 其他不可序列化类型，退化为字符串
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)
        

    