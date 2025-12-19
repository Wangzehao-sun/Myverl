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
Single Process Actor
"""

import itertools
import logging
import os
from typing import List, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

from tensordict import TensorDict
__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
from verl.workers.actor.dp_actor import DataParallelPPOActor

class NewDataParallelPPOActor(DataParallelPPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        super().__init__(config, actor_module, actor_optimizer)
        #self.use_ulysses_sp = False  # 关闭Ulysses SP
    
    
    
    def calculate_off_policy_data_and_ess(self, mini_batch, temperature=1.0):
        
        # 从mini_batch中分离off_policy的数据
        off_data = None
        prefix_mask = mini_batch['prefix_mask']
        off_policy_mask = prefix_mask.any(-1)  # 只要prefix_mask中有一个True，就认为是off_policy数据
        if off_policy_mask.any():
            off_num = off_policy_mask.sum().item()
             # 获取当前设备
            device = prefix_mask.device
            print("prefix_mask to ess device:", device)
            # 创建TensorDict并确保所有张量在同一设备上
            off_mini_batch = TensorDict({
                k: v[off_policy_mask] for k, v in mini_batch.items()
            }, batch_size=off_num,device=get_device_id())
            
            # 安全地访问字段
            print(f"--- new_dp_actor.py, off_policy data size: {off_num}")
            #off_mini_batch = TensorDict({k: v[off_policy_mask] for k, v in mini_batch.items()},batch_size=off_num)
            #构建dataproto
            off_old_log_probs = off_mini_batch['old_log_probs']
            attention_mask = off_mini_batch['attention_mask']
            responses = off_mini_batch['responses']
            response_length = responses.size(1)
            response_mask = attention_mask[:, -response_length:]
            with torch.no_grad():
                _,off_log_probs = self._forward_micro_batch(micro_batch=off_mini_batch, temperature=temperature, calculate_entropy=False)
                off_ratio = torch.exp(torch.clamp(off_log_probs - off_old_log_probs, min=-20, max=10))
                off_ratio_mean = verl_F.masked_mean(off_ratio,response_mask)
                off_ratio_var = verl_F.masked_var(off_ratio,response_mask)
                off_ratio_ess = torch.tensor(0.0)
                # 2. 计算掩码内的 token 数量
                num_off_tokens = response_mask.sum()

                if num_off_tokens > 0:
                    # 3. 只对掩码内的 off_ratio 进行计算
                    sum_of_ratios = (off_ratio * response_mask).sum()
                    sum_of_squared_ratios = ((off_ratio**2) * response_mask).sum()
        
                    # 4. 计算 ESS
                    ess = (sum_of_ratios**2) / (sum_of_squared_ratios + 1e-8)
        
                    # 5. 除以 token 数量进行归一化
                    off_ratio_ess = ess / num_off_tokens
                metrics_data = {
                    'actor/off_ratio_mean_unbias': off_ratio_mean.detach().item(),
                    'actor/off_ratio_var_unbias': off_ratio_var.detach().item(),
                    'actor/off_ratio_ess_unbias': off_ratio_ess.detach().item(),
                }
            return off_ratio_ess,metrics_data
        else:
            return None,None
    
    
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages","prefix_mask",'se_mask','reward_mask']
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        
        if self.config.use_off_policy_loss and self.config.off_policy_loss_impl == 'seq':
            select_keys.append('on_logprobs_mean')
            select_keys.append('on_logprobs_std')
        if self.config.use_off_policy_loss and self.config.use_off_policy_probs:
            select_keys.append('target_probs')
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)
        batch_size = data.batch.batch_size[0]
        #print(f"--- new_dp_actor.py, total data size: {data.batch.batch_size} ---")

        metrics = {}
        all_reconstructed_log_probs = None

        for epoch in range(self.config.ppo_epochs):
            epoch_log_probs_list = []
            epoch_prefix_mask_list = []
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                #print(f"--- new_dp_actor.py, mini_batch size: {mini_batch.batch_size} ---")

                #打印mini_batch的key
                #print(f"--- new_dp_actor.py, mini_batch keys: {list(mini_batch.keys())} ---")
                # --- 步骤1: 为当前 mini_batch 初始化 log_prob 收集容器 ---
                log_probs_placeholder = None
                #reorder_indices = None
                log_probs_mini_list = []
                prefix_mask_mini_list = []
                #从mini_batch中分离off_policy的数据
                
                #计算mini_batchdata的log_prob,从而计算off_policy token的ess
                off_ratio_ess = None
                if self.config.use_off_policy_loss and self.config.policy_loss.loss_mode in ["se","se_luffy"] and self.config.policy_loss.off_policy_reshape=="token":
                    off_ratio_ess,off_ratio_metrics = self.calculate_off_policy_data_and_ess(mini_batch, temperature=temperature)
                    if off_ratio_metrics:
                        append_to_dict(metrics, off_ratio_metrics)
                    off_ratio_ess = off_ratio_ess.detach() if off_ratio_ess is not None else 1.0
                if has_multi_modal_inputs:
                    micro_batches = []
                    if self.config.use_dynamic_bsz:
                        all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                        batch_tensordict_for_rearrange = data.batch

                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        rearranged_text_micro_batches_tds, textual_indices = rearrange_micro_batches(batch=batch_tensordict_for_rearrange, max_token_len=max_token_len)

                        for current_original_indices, text_mb_td in zip(textual_indices, rearranged_text_micro_batches_tds):
                            current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]
                            mb_dict = {k: v for k, v in text_mb_td.items()}
                            mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                            micro_batches.append(mb_dict)
                    else:
                        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                        micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches,reorder_indices  = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    #reorder_indices = textual_indices
                else:
                    
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                #print(f"--- new_dp_actor.py, micro_batch number: {len(micro_batches)} ---")
                #print(f"--- new_dp_actor.py, gradient steps: {self.gradient_accumulation} ---")
                self.actor_optimizer.zero_grad()

                for micro_batch_idx, data in enumerate(micro_batches):
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                    elif isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, torch.Tensor):
                                data[k] = v.to(get_device_id())
                            elif k == "multi_modal_inputs" and v is not None:
                                data[k] = [{kk: vv.to(get_device_id()) for kk, vv in item_dict.items()} for item_dict in v]
                            else:
                                data[k] = v
                    else:
                        data = data.to(get_device_id())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)
                    
                    # --- 步骤2: 在 micro_batch 循环中填充容器 ---

                    log_probs_mini_list.append(log_prob.detach())
                    prefix_mask_mini_list.append(data["prefix_mask"])

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if self.config.policy_loss.loss_mode == "vanilla":
                        #pg_loss表示聚合的策略梯度损失，根据loss_agg_mode,例如token_mean
                        #pg_clipfrac表示有多少比例的词元（token）的损失被裁剪了
                        #ppo_kl表示PPO中的KL散度
                        #pg_clipfrac_lower表示有多少比例的词元（token）的损失被裁剪了，且优势函数为负。
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )
                        metrics_data = {
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                        append_to_dict(metrics, metrics_data)
                    elif self.config.policy_loss.loss_mode in ["luffy","rlpluss","se",'se_luffy','se_filter']:
                        from .new_metrics import get_ratio_stats
                        # ratio_stats = get_ratio_stats(old_log_prob=old_log_prob,
                        #     log_prob=log_prob,
                        #     target_probs=data['target_probs'] if 'target_probs' in data else None,
                        #     off_policy_strategy=self.config.policy_loss.loss_mode,
                        #     prefix_mask=data['prefix_mask'],
                        #     response_mask=response_mask,
                        #     off_policy_reshape=None,
                        # )
                        # append_to_dict(metrics, ratio_stats)
                        from .new_core_alg import compute_token_on_off_policy_loss
                        loss_fn = compute_token_on_off_policy_loss
                        prefix_mask = data['prefix_mask'] if 'prefix_mask' in data else None
                        se_mask = data['se_mask'] if 'se_mask' in data else None
                        target_probs = data['target_probs'] if 'target_probs' in data else None
                        reward_mask = data['reward_mask'] if 'reward_mask' in data else None
                        # clip_upper_bound 默认为1.0
                        # off_policy的裁剪默认为False
                        off_policy_reshape = self.config.policy_loss.get("off_policy_reshape", "p_div_p_0.1")
                        ret_dict = loss_fn(old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            prefix_mask=prefix_mask,
                            se_mask=se_mask,
                            off_max_clip=self.config.off_policy_max_clip if self.config.off_policy_max_clip != -1 else None,
                            off_min_clip=self.config.off_policy_min_clip if self.config.off_policy_min_clip != -1 else None,
                            all_max_clip=self.config.all_max_clip if self.config.all_max_clip != -1 else None,
                            off_policy_strategy=self.config.policy_loss.loss_mode,
                            off_policy_reshape=off_policy_reshape,
                            target_probs=target_probs,
                            loss_remove_token_mean=self.config.loss_remove_token_mean,
                            loss_remove_clip=self.config.loss_remove_clip,
                            off_ratio_ess=off_ratio_ess,
                            reward_mask=reward_mask,
                        )
                        pg_loss = ret_dict['pg_loss']
                        off_pg_loss = ret_dict['off_pg_loss']
                        on_pg_loss = ret_dict['on_pg_loss']
                        off_pg_clipfrac = ret_dict['off_pg_clipfrac']
                        pg_clipfrac = ret_dict['on_pg_clipfrac']
                        ppo_kl = ret_dict['ppo_kl']
                        
                        metrics_data = {
                            'actor/off_pg_loss': off_pg_loss.detach().item(),
                            'actor/on_pg_loss': on_pg_loss.detach().item(),
                            'actor/off_pg_clipfrac': off_pg_clipfrac.detach().item(),
                            'actor/on_pg_clipfrac': pg_clipfrac.detach().item(),
                        }
                        if 'off_policy_prob' in ret_dict:
                            metrics_data['actor/off_policy_prob'] = ret_dict['off_policy_prob'].detach().item()
                        if 'on_policy_prob' in ret_dict:
                            metrics_data['actor/on_policy_prob'] = ret_dict['on_policy_prob'].detach().item()
                        if 'off_ratio_mean' in ret_dict:
                            metrics_data['actor/off_ratio_mean'] = ret_dict['off_ratio_mean'].detach().item()
                        if 'on_ratio_mean' in ret_dict:
                            metrics_data['actor/on_ratio_mean'] = ret_dict['on_ratio_mean'].detach().item()
                        if 'off_ratio_ess' in ret_dict:
                            metrics_data['actor/off_ratio_ess'] = ret_dict['off_ratio_ess'].detach().item()
                        if 'off_ratio_max_clip_frac' in ret_dict:
                            metrics_data['actor/off_ratio_max_clip_frac'] = ret_dict['off_ratio_max_clip_frac'].detach().item()
                        if 'off_ratio_min_clip_frac' in ret_dict:
                            metrics_data['actor/off_ratio_min_clip_frac'] = ret_dict['off_ratio_min_clip_frac'].detach().item()
                        append_to_dict(metrics, metrics_data)
                    else:
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, self.config)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef


                    #为什么要使用这个？
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        if self.config.policy_loss.loss_mode in ["luffy","rlpluss","se"]:
                            loss = policy_loss / self.gradient_accumulation
                        else:            
                            loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                        
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    metrics_data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, metrics_data)

                grad_norm = self._optimizer_step()
                metrics_data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, metrics_data)
                
                #reorder_indices = list(itertools.chain.from_iterable(reorder_indices))
                reconstructed_log_prob = torch.concat(log_probs_mini_list, dim=0)
                if self.config.use_dynamic_bsz:
                    reorder_indices = list(itertools.chain.from_iterable(reorder_indices))
                    revert_indices = torch.tensor(get_reverse_idx(reorder_indices), dtype=torch.long)
                    reconstructed_log_prob = reconstructed_log_prob[revert_indices]
                epoch_log_probs_list.append(reconstructed_log_prob)

                # reconstructed_prefix_mask = torch.concat(prefix_mask_mini_list, dim=0)
                # if self.config.use_dynamic_bsz:
                #     reconstructed_prefix_mask = reconstructed_prefix_mask[revert_indices]
                # epoch_prefix_mask_list.append(reconstructed_prefix_mask)
            all_reconstructed_log_probs=epoch_log_probs_list
            #all_reconstructed_prefix_mask=epoch_prefix_mask_list

        self.actor_optimizer.zero_grad()
        full_log_prob_shard = None
        #full_prefix_mask_shard = None
        if all_reconstructed_log_probs:
            full_log_prob_shard = torch.cat(all_reconstructed_log_probs, dim=0)
            # full_prefix_mask_shard = torch.cat(all_reconstructed_prefix_mask, dim=0)
            # # 安全检查：确保拼接后的尺寸与传入的 data.batch 的尺寸一致
            # if full_prefix_mask_shard.shape[0] != batch_size:
            #     logger.warning(
            #         f"Reconstructed prefix_mask shard size ({full_prefix_mask_shard.shape[0]}) "
            #         f"does not match original data shard size ({batch_size}). "
            #         "Returning None for prefix_mask shard."
            #     )
            #     full_prefix_mask_shard = None
            if full_log_prob_shard.shape[0] != batch_size:
                logger.warning(
                    f"Reconstructed log_prob shard size ({full_log_prob_shard.shape[0]}) "
                    f"does not match original data shard size ({batch_size}). "
                    "Returning None for log_prob shard."
                )
                full_log_prob_shard = None
        tensors_dict_to_return = {
            "log_prob": full_log_prob_shard
            #,"prefix_mask": full_prefix_mask_shard
        }
        return metrics, tensors_dict_to_return
