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

from omegaconf import ListConfig
import os
from typing import List, Union

import pandas as pd
import copy 

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs
from omegaconf import DictConfig, ListConfig
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.torch_functional import pad_sequence_to_length


import logging
import os
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'INFO'))


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output

from verl.utils.dataset.rl_dataset import RLHFDataset

class RLHFDatasetWithTarget(RLHFDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 config: DictConfig,
                 target_key='target',
                 max_target_length=8192,
                 filter_targets=False,
                 sample_target_ratio=1.0,
                 target_list_key='target_lst',
                 max_num_targets=5,
                 target_probs_key='target_ds_qwen_7b_probs',
                 se_prompt_key='se_prompt',  # 新增: 保存 SE prompt 的列名
        ):
        super().__init__(parquet_files, tokenizer, config=config)
        
        self.max_target_length = max_target_length
        self.filter_targets = filter_targets
        self.target_key = target_key
        self.se_prompt_key = se_prompt_key  # 新增: 保存 SE prompt 的列名
        self.sample_target_ratio = sample_target_ratio
        self.target_list_key = target_list_key
        self.target_probs_key = target_probs_key
        self.max_num_targets = max_num_targets

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        # 步骤 1: 直接调用父类的 __getitem__ 方法
        # 这是最核心的改动。我们让父类去处理所有关于 prompt 的复杂逻辑，
        # 包括多模态、应用聊天模板、分词、填充和计算 position_ids。
        # 这样，无论父类如何更新，我们都能自动享受到最新的 prompt 处理能力。
        # `super()` 返回的 `row_dict` 已经包含了 'input_ids', 'attention_mask', 'position_ids' 等。
        row_dict = super().__getitem__(item)

        

        # 步骤 2: 从原始数据中获取 target 相关信息
        # 因为父类可能已经从 `row_dict` 中 pop 了一些键，
        # 所以我们从最原始的数据源 `self.dataframe[item]` 中重新获取 target 相关字段。
        original_row: dict = self.dataframe[item]
        if self.se_prompt_key in original_row:
            se_prompt_messages = original_row.pop(self.se_prompt_key)
            #print("se_prompt_messages:", se_prompt_messages)
            if se_prompt_messages:
                # 1. 应用聊天模板，与父类处理标准 prompt 的方式保持一致
                messages = se_prompt_messages
                se_full_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # 2. 分词
                se_input_ids = self.tokenizer(se_full_prompt, add_special_tokens=False, return_tensors='pt')['input_ids']

                # 3. 填充或截断，与标准 prompt 使用相同的最大长度
                se_input_ids = pad_sequence_to_length(
                    se_input_ids,
                    max_seq_len=self.max_target_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True  # prompt 通常进行左填充
                )
                
                # 4. 计算 attention_mask 和 position_ids
                se_attention_mask = (se_input_ids != self.tokenizer.pad_token_id).to(se_input_ids.dtype)
                se_position_ids = compute_position_id_with_mask(se_attention_mask)
                # 5. 添加到 row_dict 中
                row_dict['se_input_ids'] = se_input_ids.squeeze(0)
                row_dict['se_attention_mask'] = se_attention_mask.squeeze(0)
                row_dict['se_position_ids'] = se_position_ids.squeeze(0)
            else:
                # 如果 se_prompt 为空，则创建与标准 prompt 形状相同的填充张量
                row_dict['se_input_ids'] = torch.full_like(row_dict['input_ids'], self.tokenizer.pad_token_id)
                row_dict['se_attention_mask'] = torch.zeros_like(row_dict['attention_mask'])
                row_dict['se_position_ids'] = torch.zeros_like(row_dict['position_ids'])
                # 5. 添加到 row_dict 中
        # 步骤 3: 处理核心的 `target` 序列 (tgt_input_ids)
        # 这部分逻辑与你原来的代码几乎完全相同，因为这是子类的核心功能。
        tgt = original_row.get(self.target_key)
        sample = np.random.rand() < self.sample_target_ratio

        if tgt is not None and sample is True:
            tgt = tgt[0]
            
            # 获取父类处理好的、带模板的 prompt 字符串，用于后续逻辑判断
            prompt_with_chat_template = row_dict.get("full_prompts", "")

            if prompt_with_chat_template.endswith('<think>\n') and tgt['content'].startswith('<think>\n'):
                tgt['content'] = tgt['content'][len('<think>\n'):]
            
            tgt_input_ids = self.tokenizer(tgt['content'], add_special_tokens=False, return_tensors='pt')['input_ids'].reshape(-1)
            tgt_input_ids = tgt_input_ids.reshape(1, -1)
        else:
            # 如果不采样 target，则创建一个空张量
            tgt_input_ids = torch.tensor([], dtype=torch.long).reshape(1, 0)

        # 对 `tgt_input_ids` 进行填充或截断，逻辑保持不变
        sequence_length = tgt_input_ids.shape[-1]
        if sequence_length < self.max_target_length:
            tgt_input_ids = pad_sequence_to_length(tgt_input_ids,
                                                   max_seq_len=self.max_target_length,
                                                   pad_token_id=self.tokenizer.pad_token_id,
                                                   left_pad=False)
        else:
            assert self.truncation in ('right', 'error')
            tgt_input_ids = tgt_input_ids[:, :self.max_target_length]
        
        row_dict['tgt_input_ids'] = tgt_input_ids.squeeze(0)

        # 步骤 4: 处理 `target_list`，逻辑保持不变
        if getattr(self, 'target_list_key', "target_list_key") in original_row:
            target_list = original_row.get(self.target_list_key)
            prompt_with_chat_template = row_dict.get("full_prompts", "")
            if target_list is None:
                tgt_input_ids_lst = [torch.zeros_like(row_dict['tgt_input_ids']).fill_(self.tokenizer.pad_token_id)] * self.max_num_targets
            else:
                tgt_input_ids_lst = [self._process_target(tgt, prompt_with_chat_template, add_eos=True) for tgt in target_list]
                if len(tgt_input_ids_lst) <= self.max_num_targets:
                    tgt_input_ids_lst.extend([torch.zeros_like(tgt_input_ids_lst[0]).fill_(self.tokenizer.pad_token_id)] * (self.max_num_targets - len(tgt_input_ids_lst)))
                else:
                    tgt_input_ids_lst = tgt_input_ids_lst[:self.max_num_targets]
            row_dict['tgt_input_ids_lst'] = torch.stack(tgt_input_ids_lst, dim=0)

        # 步骤 5: 处理 `target_probs`，逻辑保持不变
        if getattr(self, 'target_probs_key', "target_probs_key") in original_row:
            target_probs = original_row.get(self.target_probs_key)
            if target_probs is not None:
                target_probs_pt = torch.tensor(target_probs, dtype=torch.float32)
                target_probs_pt = target_probs_pt.reshape(1, -1)
                
                tgt_len = (row_dict['tgt_input_ids'] != self.tokenizer.pad_token_id).sum()
                # 这里的断言可能需要根据实际数据微调
                # assert target_probs_pt.shape[-1] == tgt_len + 1

                if target_probs_pt.shape[-1] < self.max_target_length:
                    target_probs_pt = pad_sequence_to_length(target_probs_pt,
                                                             max_seq_len=self.max_target_length,
                                                             pad_token_id=-1,
                                                             left_pad=False)
                else:
                    assert self.truncation in ('right', 'error')
                    target_probs_pt = target_probs_pt[:, :self.max_target_length]
                row_dict['target_probs'] = target_probs_pt.squeeze(0)
            else:
                row_dict['target_probs'] = torch.zeros_like(row_dict['tgt_input_ids'], dtype=torch.float32).fill_(-1)

        # 父类已经处理了 'raw_prompt', 'index' 等字段，我们无需重复
        # 直接返回被我们追加了 target 相关字段的 `row_dict`
        #print(row_dict["input_ids"].shape, row_dict["attention_mask"].shape, row_dict["position_ids"].shape, row_dict['tgt_input_ids'].shape)
        row_dict.pop("full_prompts", None)
        #print(row_dict)
        return row_dict

    def _process_target(self, tgt: str, prompt: str, add_eos=False) -> torch.Tensor:
        if prompt.endswith('<think>\n') and tgt.startswith('<think>\n'):
            tgt = tgt[len('<think>\n'):]
        tgt_input_ids = self.tokenizer(tgt, add_special_tokens=False, return_tensors='pt')['input_ids'].reshape(-1) # [1, l]
        if add_eos:
            tgt_input_ids = torch.cat([tgt_input_ids, torch.tensor([self.tokenizer.eos_token_id], device=tgt_input_ids.device, dtype=tgt_input_ids.dtype).reshape(-1)])

        tgt_input_ids = tgt_input_ids.reshape(1, -1)
        # padding or truncate
        sequence_length = tgt_input_ids.shape[-1]
        if sequence_length < self.max_target_length:
            # right pad for tgt_input_ids
            tgt_input_ids = pad_sequence_to_length(tgt_input_ids,
                                            max_seq_len=self.max_target_length,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            left_pad=False)
        else:
            assert self.truncation in ('right', 'error')
            tgt_input_ids = tgt_input_ids[:, :self.max_target_length]
        
        tgt_input_ids = tgt_input_ids.squeeze(0)

        return tgt_input_ids

from verl import DataProto
class BufferedDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.buffer = []
        self.dataloader_iter = None

    def start_new_epoch(self):
        """Reset for new epoch"""
        self.dataloader_iter = iter(self.dataloader)

    def get_next_batch(self):
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            raise StopIteration

    def __len__(self):
        return len(self.dataloader)

    def add_to_buffer(self, samples):
        if len(self.buffer) == 0:
            self.buffer = samples
        else:
            self.buffer = DataProto.concat([self.buffer, samples])

    def get_from_buffer(self, count, dp_size):
        if count > self.buffer_size():
            count = (self.buffer_size() // dp_size) * dp_size
        samples = self.buffer.slice(range(0, count))
        self.buffer = self.buffer.slice(range(count, self.buffer_size()))
        return samples

    def buffer_size(self):
        return len(self.buffer)

import torch

class ResumableRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    #data_source: Sized
    #replacement: bool

    def __init__(self, data_source):
        self.data_source = data_source
        self.generator = torch.Generator()
        self.generator.manual_seed(47)
        
        self.perm_index = 0
        self.perm = torch.randperm(self.num_samples, generator=self.generator)
        
    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        if self.perm_index >= len(self.perm):
            self.perm_index = 0
            self.perm = torch.randperm(self.num_samples, generator=self.generator)
            
        while self.perm_index < len(self.perm):
            self.perm_index += 1
            yield self.perm[self.perm_index-1].item() # the output index should be int

    def __len__(self):
        return self.num_samples
    
    def get_state(self):
        return {"perm": self.perm, "perm_index": self.perm_index, "generator_state": self.generator.get_state()}
    
    def set_state(self, state):
        self.perm = state["perm"]
        self.perm_index = state["perm_index"]
        self.generator.set_state(state["generator_state"])

def _pre_process_inputs_right_pad(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)
    token_ids = prompt_token_ids[:non_pad_index[-1][0]].tolist()
    return token_ids