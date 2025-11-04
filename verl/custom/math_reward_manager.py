from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math
#sys.path.append('~/LLM/Train/verl/verl/custom/')
#from .deepscaler.rewards.math_reward import deepscaler_reward_fn, THOUGHT_DELIMITER_END, THOUGHT_DELIMITER_START
from typing import List, Union
#from verl.custom.reward_with_format import deepscaler_reward_fn_impl1
from verl.custom.math_verify_reward import reward_fn_math_verify, reward_fn_math_verify_no_think
from verl.workers.reward_manager import register
# def deepscaler_reward_fn_nothink(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False):
#     solution_str = f"{THOUGHT_DELIMITER_START}\n{THOUGHT_DELIMITER_END}\n{solution_str}"
#     return deepscaler_reward_fn(solution_str, ground_truth, enable_llm)

def _select_rm_score_fn(data_source, reward_impl_version):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    else:
        if reward_impl_version == 0:
            raise NotImplementedError
            #return deepscaler_reward_fn
        elif reward_impl_version == 1:
            raise NotImplementedError
            #return deepscaler_reward_fn_impl1
        elif reward_impl_version == 2:
            raise NotImplementedError
            #return deepscaler_reward_fn_nothink
        elif reward_impl_version == 3:
            return reward_fn_math_verify
        elif reward_impl_version == 4:
            return reward_fn_math_verify_no_think
        else:
            raise NotImplementedError

@register("math")
class MathRewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, reward_impl_version) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_impl_version = reward_impl_version

    def __call__(self, data: DataProto,return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        from concurrent.futures import ThreadPoolExecutor
        from typing import Dict, Any
        #import threading
        # Thread-safe dict for tracking printed data sources
        # print_lock = threading.Lock()
        
        def process_item(args):
            i, data_item, already_print_data_sources = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            # sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences = valid_response_ids
            sequences_str = self.tokenizer.decode(sequences)
            # if not "no_think" in self.reward_impl_version:
            from .deepscaler.globals import THOUGHT_DELIMITER_START
            # sequences_str = [THOUGHT_DELIMITER_START + seq.strip() for seq in sequences_str]
            if self.reward_impl_version != 4:
                sequences_str = THOUGHT_DELIMITER_START + '\n' + sequences_str
            # else:
            #     breakpoint()

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source, reward_impl_version=self.reward_impl_version)
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
            
            # with print_lock:
            #     if data_source not in already_print_data_sources:
            #         already_print_data_sources[data_source] = 0

            #     if already_print_data_sources[data_source] < self.num_examine:
            #         already_print_data_sources[data_source] += 1
            #         print(sequences_str)      
            return i, score, valid_response_length

        if self.reward_impl_version in {3, 4}:
            args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
            results = list(process_item(args[i]) for i in range(len(args)))
        else:
            # Process items in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=96) as executor:
                args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
                results = list(executor.map(process_item, args))

        # Fill reward tensor with results
        for i, score, valid_response_length in results:
            reward_tensor[i, valid_response_length - 1] = score

        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor

