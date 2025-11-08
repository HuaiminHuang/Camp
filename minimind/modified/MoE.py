from transformers import PretrainedConfig
import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

# ffn 作为 moe 的单个专家
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 这里参考的是 llama 的中间维度
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3) 
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    # x --> act_fun(gate(x) * up(x)) --> down(x) --droput--> x
    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = config.n_routed_experts # 路由expeert数量

        self.scoring_func = config.scoring_func # 归一化分数函数（仅支持softmax）
        self.alpha = config.aux_loss_alpha # aux loss alpha系数
        self.seq_aux = config.seq_aux # 是否使用seq级别的loss计算

        self.norm_topk_prob = config.norm_topk_prob # 归一化topk weight 的开关
        self.gating_dim = config.hidden_size  
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    # 初始化
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape # 获取隐藏state的维度
        hidden_states = hidden_states.view(-1, h) # shape change to [btz * seq_len, hidden_size]
        logits = F.linear(hidden_states, self.weight, None) # 获取 logits = <h, W> shape [bts * seq_len, h] * [h, n_expert] --> [b * s, n_expert]
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1) # 归一化logits shape is [b * s, n_expert]
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # 获取 topk 的 index 和 对应权重 [b*s, topK]
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False) 
        
        # 开启 --> 权重归一化
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            # scores_for_aux shape [b*s, n_expert]
            # topk_idx_for_aux_loss shape is [b, s*topK]
            # aux_top [opK,]
            # loss_aux = alpha * sum(E *f_i * p_i) 
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1) # shape is [b, s, n_expert]
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device) # 创建一个零张量 ce shape [b, n_expert]
                #统计每个 batch 中被选中的专家次数, 除以 seq_len * K / n_experts，使平均分布为 1
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                # loss_aux = alpha * 1/b * sum(E * f_i * p_i)
                # pi = 1/s * sum 
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # [b*s*k, n_expert] 的 one-hot 张量，专家是否被选上
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0) # 每个专家是否选上
                Pi = scores_for_aux.mean(0) # 专家平均 gating 后的概率
                fi = ce * self.n_routed_experts 
                # 标准公式 loss_aux = alpha * sum(E *f_i * p_i) 
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        # [b*s, topK]*2, [1,]
        return topk_idx, topk_weight, aux_loss  


class MOEFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 路由专家
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        # 共享专家
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1]) # [b*s, h]
        flat_topk_idx = topk_idx.view(-1) # 展平 [b*s*topK]
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0) # 复制 n_expert 份 [b*s*topK, h]
            y = torch.empty_like(x, dtype=torch.float16)
            # 按照索引分配给expert
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            # weight shape is [b*s, topK] 
            # y.view() shape is [b*s, topK, h] then y shape [b*s, h]
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1) 
            y = y.view(*orig_shape) # [b, s, h]
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x) # [b*s, h]
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache

