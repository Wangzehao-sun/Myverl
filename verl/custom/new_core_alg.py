import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import register_policy_loss
def compute_sft_pure_loss(log_prob, eos_mask):
    sft_losses = -log_prob
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask)
    return sft_loss


def compute_grpo_outcome_advantage_split(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   on_policy_mask: torch.Tensor,
                                   epsilon: float = 1e-6,
                                   use_std: bool = True):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            # only include on-policy samples for mean and std calculation
            if on_policy_mask[i].item() is True:
                id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        # process std
        for idx in id2std:
            if id2std[idx].item() == 0:
                id2std[idx] = torch.tensor(1.0)
        for i in range(bsz):
            if use_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = (scores[i] - id2mean[index[i]])
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_ess(off_ratio, mask):
    """Compute effective sample size (ESS) for off-policy ratios.

    Args:
        off_ratio: `(torch.Tensor)`
            shape: (bs, response_length)
        mask: `(torch.Tensor)`
            shape: (bs, response_length)
    Returns:
        ess: `(torch.Tensor)`
            shape: (bs,)
    """
    off_ratio_ess = torch.tensor(0.0)
    # 2. 计算掩码内的 token 数量
    num_off_tokens = mask.sum()

    if num_off_tokens > 0:
        # 3. 只对掩码内的 off_ratio 进行计算
        sum_of_ratios = (off_ratio * mask).sum()
        sum_of_squared_ratios = ((off_ratio**2) * mask).sum()
        
        # 4. 计算 ESS
        ess = (sum_of_ratios**2) / (sum_of_squared_ratios + 1e-8)
        
        # 5. 除以 token 数量进行归一化
        off_ratio_ess = ess / num_off_tokens

    return off_ratio_ess
def compute_sequence_ess(ratio, mask):
    """计算每个序列的有效样本量(ESS)，不参与梯度计算
    
    Args:
        ratio: `(torch.Tensor)`
            形状: (batch_size, sequence_length)，重要性权重比率
        mask: `(torch.Tensor)`
            形状: (batch_size, sequence_length)，标识哪些token是有效的
            
    Returns:
        sequence_ess: `(torch.Tensor)`
            形状: (batch_size,)，每个序列的归一化ESS值，已detach
    """
    # 使用torch.no_grad()确保不会计算梯度
    with torch.no_grad():
        batch_size = ratio.shape[0]
        sequence_ess = torch.zeros(batch_size, device=ratio.device)
        
        for i in range(batch_size):
            # 获取当前序列的比率和掩码
            seq_ratio = torch.clamp(ratio[i], max=100.0)  # 可选：对比率进行裁剪，防止极端值
            seq_mask = mask[i]
            
            # 计算当前序列掩码内的token数量
            num_tokens = seq_mask.sum().item()
            
            if num_tokens > 0:
                # 只对掩码内的token计算ESS
                sum_of_ratios = (seq_ratio * seq_mask).sum()
                sum_of_squared_ratios = ((seq_ratio**2) * seq_mask).sum()
                
                # 计算ESS并归一化
                ess = (sum_of_ratios**2) / (sum_of_squared_ratios + 1e-8)
                sequence_ess[i] = ess / num_tokens
            else:
                # 如果没有有效token，设置ESS为1.0（不调整权重）
                sequence_ess[i] = 1.0
                
    return sequence_ess  # 已经在no_grad内计算，所以不需要额外detach


def compute_token_on_off_policy_loss(
    old_log_prob, 
    log_prob, 
    advantages, 
    response_mask, 
    cliprange, 
    prefix_mask,
    se_mask,
    reward_mask,
    cliprange_low=None,
    cliprange_high=None,
    off_max_clip=None, 
    off_min_clip=None,
    all_max_clip=None, 
    off_policy_strategy="luffy", 
    off_policy_reshape="p_div_p_0.1",
    target_probs=None,
    loss_remove_token_mean=False,
    loss_remove_clip=False,
    off_ratio_ess=None,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        prefix_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    # off-policy loss
    # compute off-policy probability
    
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    ratio = torch.exp(negative_approx_kl) # [bsz, l]

    on_pg_losses = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    upper_bound = max(1.0 + cliprange_high, 1.0 + cliprange)
    if loss_remove_clip is False:
        on_pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange_low, upper_bound)
        on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), response_mask)
        on_pg_losses_clip = torch.max(on_pg_losses, on_pg_losses2)
        on_pg_losses3 = -advantages * 3.0
        on_pg_losses_clip2 = torch.min(on_pg_losses3, on_pg_losses_clip)
        on_pg_losses = torch.where(advantages < 0, on_pg_losses_clip2, on_pg_losses_clip)
        on_pg_loss = verl_F.masked_mean(on_pg_losses, (~prefix_mask) * response_mask)
    else:
        on_pg_loss = verl_F.masked_mean(on_pg_losses, (~prefix_mask) * response_mask)
        on_pg_clipfrac = torch.tensor(0.0)
    on_ratio_mean = verl_F.masked_mean(ratio, (~prefix_mask) * response_mask)
    #计算方差
    #on_ratio_std = verl_F.masked_var(ratio, (~prefix_mask) * response_mask)
    if on_ratio_mean.isnan().any().item():
        on_ratio_mean = torch.tensor(0.0)
    # compute off-policy loss
    if target_probs is None:
        off_ratio = torch.exp(log_prob) # [bsz, l]
        if off_policy_strategy == "no_reshape":
            pass
        elif off_policy_strategy == "rlpluss":
            omiga_prob= 0.5*(torch.exp(old_log_prob) + 1)
            off_ratio = 2* torch.exp(log_prob)/(torch.exp(old_log_prob) + omiga_prob) 
        elif off_policy_strategy == "luffy":
            if off_policy_reshape == "no_reshape":
                pass
            elif off_policy_reshape == "p_div_p_0.1":
                off_ratio = off_ratio / (off_ratio + 0.1)
            else:
                off_ratio = off_ratio / (off_ratio + 0.1)
                #raise ValueError(f"Invalid off_policy_reshape: {off_policy_reshape}")
        elif off_policy_strategy in ["se","se_filter"]:
            if off_policy_reshape == "sequence":
                #计算每个序列的ess
                # 计算每个序列的ess，确保不参与梯度计算
                ess_seq = compute_sequence_ess(ratio, prefix_mask * response_mask)
                 # 使用每个序列的ess对ratio进行加权
                batch_size = ratio.shape[0]
                # ess_seq已经是detached的，不会传递梯度
                off_ratio = ratio * ess_seq.view(batch_size, 1).expand_as(ratio)
            elif off_policy_reshape == "p_div_p_0.1":
                off_ratio = ratio / (ratio + 0.1)
            elif off_policy_reshape == "rlpluss":
                detached_log_prob = log_prob.detach()
                off_ratio = off_ratio * (1 - torch.exp(detached_log_prob))**(0.5)
            elif off_policy_reshape == "vanilla":
                off_ratio = torch.exp(negative_approx_kl)
            elif off_policy_reshape == "filter":
                off_ratio = torch.exp(negative_approx_kl)* reward_mask
            elif off_policy_reshape == "cispo":
                off_ratio = torch.exp(negative_approx_kl).detach() * reward_mask * log_prob
            else:
                if off_ratio_ess is None:
                    off_ratio = ratio
                else:
                    off_ratio = ratio* off_ratio_ess
        elif off_policy_strategy == "se_luffy":
            # 注意非se的部分，ess=1
            if off_policy_reshape == "sequence":
                #计算每个序列的ess
                # 计算每个序列的ess，确保不参与梯度计算
                ess_seq = compute_sequence_ess(ratio, prefix_mask * response_mask)
                 # 使用每个序列的ess对ratio进行加权
                batch_size = ratio.shape[0]
                # ess_seq已经是detached的，不会传递梯度
                off_ratio = ratio * ess_seq.view(batch_size, 1).expand_as(ratio)
                # off_ratio = off_ratio * se_mask + ratio * (1 - se_mask)
                off_ratio = off_ratio * se_mask + (ratio /(ratio+0.1)) * (~se_mask)
        else:
            raise ValueError(f"Invalid off_policy_strategy: {off_policy_strategy}")
    else:
        assert target_probs.shape == log_prob.shape
        off_ratio = torch.exp(log_prob) / (target_probs+1e-6)
        # off_ratio[log_prob == 0] = 0
        off_ratio = off_ratio * prefix_mask
        # assert ((target_probs > 0) == prefix_mask).all()

    # clip off-policy ratio
    if off_max_clip is not None:
        off_ratio = torch.clamp(off_ratio, max=off_max_clip)
        off_ratio_max_clip_frac = verl_F.masked_mean((off_ratio == off_max_clip).float(), prefix_mask * response_mask)
    else:
        off_ratio_max_clip_frac = torch.tensor(0.0)
        
    if off_min_clip is not None:
        #print("clipping off_ratio with min value:", off_min_clip)
        #print("before clipping, off_ratio stats - min:", off_ratio.min().item(), "mean:", off_ratio.mean().item())
        off_ratio = torch.clamp(off_ratio, min=off_min_clip)
        #print("after clipping, off_ratio stats - min:", off_ratio.min().item(), "mean:", off_ratio.mean().item())
        off_ratio_min_clip_frac = verl_F.masked_mean((off_ratio == off_min_clip).float(), prefix_mask * response_mask)
    else:
        off_ratio_min_clip_frac = torch.tensor(0.0)
    #print("off_tokens:", int((prefix_mask * response_mask).sum().item()))
    off_ratio_mean = verl_F.masked_mean(off_ratio, prefix_mask * response_mask)
    # off_ratio_std = verl_F.masked_var(off_ratio, prefix_mask * response_mask)
    if off_ratio_mean.isnan().any().item():
        off_ratio_mean = torch.tensor(0.0)
    #计算off——ratio的有效样本数量ess
    # 1. 定义 off-policy 的掩码
    #off_ratio_ess = compute_ess(off_ratio, prefix_mask * response_mask)
    

    if off_policy_strategy == "rlpluss":
        detached_log_prob = log_prob.detach()
        explosion_based_advantages  = advantages * (1 - torch.exp(detached_log_prob))**(0.5)
        off_pg_losses = -off_ratio * explosion_based_advantages 
    elif off_policy_strategy in ["se","se_filter"]:
        clipped_advantages = torch.clamp(advantages, min=0.0)
        off_pg_losses = -off_ratio * clipped_advantages
        #off_pg_losses = -off_ratio * advantages
    else:
        off_pg_losses = -off_ratio * advantages
    off_pg_loss = verl_F.masked_mean(off_pg_losses, prefix_mask * response_mask)
    if off_pg_loss.isnan().item() is True:
        off_pg_loss = torch.tensor(0.0)
    off_pg_clipfrac = torch.tensor(0.0)
    
    prefix_mask = prefix_mask.float()
    se_mask = se_mask.float()
    pg_losses = off_pg_losses * prefix_mask + on_pg_losses * (1 - prefix_mask)
    
    # log on/off probs
    #这段代码的核心目标是回答以下两个问题：

    #对于离策略（Off-Policy）部分（即给定的前缀），当前模型给出的平均置信度（概率）是多少？
    #对于在策略（On-Policy）部分（即模型自己生成的部分），生成这些数据的旧策略给出的平均置信度是多少？
    #通过监控这两个值，开发者可以获得宝贵的洞察：

    #如果 off_policy_prob 持续很低，说明模型在模仿给定的前缀方面学得很差。
    #如果 off_policy_prob 很快变得非常接近 1.0，可能意味着模型对前缀部分过拟合了。
    #on_policy_prob 的变化可以反映旧策略（rollout 策略）的探索程度。
    off_policy_probs = torch.exp(log_prob)
    off_policy_prob = verl_F.masked_mean(off_policy_probs, prefix_mask * response_mask)
    #print("Average off_policy_prob on standard off-policy samples before update policy:", off_policy_prob.item())
    
    if off_policy_prob.isnan().item() is True:
        off_policy_prob = torch.tensor(0.0)
    on_policy_probs = torch.exp(old_log_prob)
    on_policy_prob = verl_F.masked_mean(on_policy_probs, (1.0-prefix_mask) * response_mask)
    if on_policy_prob.isnan().item() is True:
        on_policy_prob = torch.tensor(0.0)
            
    if all_max_clip is not None:
        p_on = torch.exp(log_prob)
        p_on_mask = (p_on <= all_max_clip).float()
        response_mask = response_mask * p_on_mask
        pg_losses = pg_losses * p_on_mask
        
    if loss_remove_token_mean is True:
        pg_loss = (pg_losses * response_mask).sum() / response_mask.shape[-1]
        #print(f'no token mean: mean normalization {response_mask.shape[-1]}')
    else:
        pg_loss = verl_F.masked_mean(pg_losses, response_mask)

    return {
        "pg_loss": pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_loss": on_pg_loss,
        "off_pg_clipfrac": off_pg_clipfrac,
        "on_pg_clipfrac": on_pg_clipfrac,
        "ppo_kl": ppo_kl,
        "off_policy_prob": off_policy_prob,
        "on_policy_prob": on_policy_prob,
        "off_ratio_mean": off_ratio_mean,
        "on_ratio_mean": on_ratio_mean,
        "off_ratio_max_clip_frac": off_ratio_max_clip_frac,
        "off_ratio_min_clip_frac": off_ratio_min_clip_frac,
    }


def compute_token_on_off_policy_loss_luffy(
    old_log_prob, 
    log_prob, 
    advantages, 
    eos_mask, 
    cliprange, 
    clip_upper_bound,
    prefix_mask, 
    off_cliprange, 
    off_normalize=False, 
    off_abs_cliprange=None, 
    off_max_clip=None, 
    off_min_clip=None,
    all_max_clip=None, 
    off_policy_reshape="no_reshape", 
    off_policy_reshape_weight=1.0, 
    off_policy_reshape_pow_exp=0.5,
    on_policy_reshape="no_reshape", 
    on_policy_reshape_weight=1.0,
    on_policy_reshape_pow_exp=0.5,
    target_probs=None,
    loss_remove_token_mean=False,
    loss_remove_clip=False,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        prefix_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    # off-policy loss
    # compute off-policy probability
    
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    if on_policy_reshape == "no_reshape":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
    elif on_policy_reshape == "logp":
        ratio = log_prob - old_log_prob
    elif on_policy_reshape == "p_logp":
        ratio = torch.exp(negative_approx_kl) + on_policy_reshape_weight * negative_approx_kl
    elif on_policy_reshape == "square_root":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
        ratio = torch.sqrt(ratio)
    elif on_policy_reshape == "pow":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
        ratio = torch.pow(ratio, on_policy_reshape_pow_exp)
    elif on_policy_reshape == "p_div_p_0.1":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        f_prob = prob / (prob + 0.1)
        f_old_prob = old_prob / (old_prob + 0.1)
        ratio = f_prob / f_old_prob
    elif on_policy_reshape == "p_div_p_0.5":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        f_prob = prob / (prob + 0.5)
        f_old_prob = old_prob / (old_prob + 0.5)
        ratio = f_prob / f_old_prob
    else:
        raise ValueError(f"Invalid on_policy_reshape: {on_policy_reshape}")

    on_pg_losses = -advantages * ratio
    upper_bound = max(clip_upper_bound, 1.0 + cliprange)
    if upper_bound == clip_upper_bound:
        print('clip upper bound is used: ', clip_upper_bound)

    if loss_remove_clip is False:
        on_pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, upper_bound)
        on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), eos_mask)
        on_pg_losses = torch.max(on_pg_losses, on_pg_losses2)
        on_pg_loss = verl_F.masked_mean(on_pg_losses, (~prefix_mask) * eos_mask)
    else:
        on_pg_loss = verl_F.masked_mean(on_pg_losses, (~prefix_mask) * eos_mask)
        on_pg_clipfrac = torch.tensor(0.0)
    
    # compute off-policy loss
    if target_probs is None:
        off_ratio = torch.exp(log_prob) # [bsz, l]
        if off_policy_reshape == "no_reshape":
            pass
        elif off_policy_reshape == "logp":
            off_ratio = log_prob * off_policy_reshape_weight
        elif off_policy_reshape == "p_logp":
            off_ratio = log_prob * off_policy_reshape_weight + off_ratio
        elif off_policy_reshape == "square_root":
            off_ratio = torch.sqrt(off_ratio)
        elif off_policy_reshape == "p_div_p_0.1":
            off_ratio = off_ratio / (off_ratio + 0.1)
        elif off_policy_reshape == "p_div_p_0.5":
            off_ratio = off_ratio / (off_ratio + 0.5)
        elif off_policy_reshape == "p_div_p_0.3":
            off_ratio = off_ratio / (off_ratio + 0.3)
        elif off_policy_reshape == "pow":
            off_ratio = torch.pow(off_ratio, off_policy_reshape_pow_exp)
        else:
            raise ValueError(f"Invalid off_policy_reshape: {off_policy_reshape}")
    else:
        assert target_probs.shape == log_prob.shape
        off_ratio = torch.exp(log_prob) / (target_probs+1e-6)
        # off_ratio[log_prob == 0] = 0
        off_ratio = off_ratio * prefix_mask
        # assert ((target_probs > 0) == prefix_mask).all()
        
    # clip off-policy ratio
    if off_max_clip is not None:
        off_ratio = torch.clamp(off_ratio, max=off_max_clip)
        off_ratio_max_clip_frac = verl_F.masked_mean((off_ratio == off_max_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_max_clip_frac = torch.tensor(0.0)
        
    if off_min_clip is not None:
        off_ratio = torch.clamp(off_ratio, min=off_min_clip)
        off_ratio_min_clip_frac = verl_F.masked_mean((off_ratio == off_min_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_min_clip_frac = torch.tensor(0.0)

    off_ratio_mean = verl_F.masked_mean(off_ratio, prefix_mask * eos_mask)
    if off_ratio_mean.isnan().any().item():
        off_ratio_mean = torch.tensor(0.0)

    off_pg_losses = -advantages * off_ratio
    off_pg_loss = verl_F.masked_mean(off_pg_losses, prefix_mask * eos_mask)
    if off_pg_loss.isnan().item() is True:
        off_pg_loss = torch.tensor(0.0)
    off_pg_clipfrac = torch.tensor(0.0)
    
    prefix_mask = prefix_mask.float()
    pg_losses = off_pg_losses * prefix_mask + on_pg_losses * (1 - prefix_mask)
    
    # log on/off probs
    off_policy_probs = torch.exp(log_prob)
    off_policy_prob = verl_F.masked_mean(off_policy_probs, prefix_mask * eos_mask)
    print("Average off_policy_prob on standard off-policy samples before update policy:", off_policy_prob.item())
    if off_policy_prob.isnan().item() is True:
        off_policy_prob = torch.tensor(0.0)
    on_policy_probs = torch.exp(old_log_prob)
    on_policy_prob = verl_F.masked_mean(on_policy_probs, (1.0-prefix_mask) * eos_mask)
    if on_policy_prob.isnan().item() is True:
        on_policy_prob = torch.tensor(0.0)
            
    if all_max_clip is not None:
        p_on = torch.exp(log_prob)
        p_on_mask = (p_on <= all_max_clip).float()
        eos_mask = eos_mask * p_on_mask
        pg_losses = pg_losses * p_on_mask
        
    if loss_remove_token_mean is True:
        pg_loss = (pg_losses * eos_mask).sum() / eos_mask.shape[-1]
        print(f'no token mean: mean normalization {eos_mask.shape[-1]}')
    else:
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask)

    return {
        "pg_loss": pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_loss": on_pg_loss,
        "off_pg_clipfrac": off_pg_clipfrac,
        "on_pg_clipfrac": on_pg_clipfrac,
        "ppo_kl": ppo_kl,
        "off_policy_prob": off_policy_prob,
        "on_policy_prob": on_policy_prob,
        "off_ratio_mean": off_ratio_mean,
        "off_ratio_max_clip_frac": off_ratio_max_clip_frac,
        "off_ratio_min_clip_frac": off_ratio_min_clip_frac,
    }