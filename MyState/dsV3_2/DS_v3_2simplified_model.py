# simplified_model.py
# 这是 DeepSeek-V3.2 模型的一个简化版本，旨在清晰地展示其核心架构。
# 移除了分布式并行、FP8量化、自定义CUDA核、KV缓存等工程优化。

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F

# --- 1. 模型配置 (ModelArgs) ---
# 定义了模型的尺寸和关键参数

@dataclass
class ModelArgs:
    max_seq_len: int = 1024
    vocab_size: int = 10000
    dim: int = 512
    inter_dim: int = 2048  # Typically 4 * dim
    moe_inter_dim: int = 256
    n_layers: int = 4
    n_dense_layers: int = 1
    n_heads: int = 8
    n_routed_experts: int = 8
    n_shared_experts: int = 2
    n_activated_experts: int = 2
    # MLA (多头潜在注意力) 相关参数
    q_lora_rank: int = 8
    kv_lora_rank: int = 128
    qk_nope_head_dim: int = 32
    qk_rope_head_dim: int = 32 # qk_head_dim = 64
    v_head_dim: int = 64 # v_head_dim = dim / n_heads
    # Indexer (DSA的核心 - 闪电索引器) 相关参数
    index_n_heads: int = 8
    index_head_dim: int = 32
    index_topk: int = 128
    # YaRN (用于长文本扩展的RoPE方法) 相关参数
    original_seq_len: int = 1024
    rope_theta: float = 10000.0
    rope_factor: float = 4.0 # Adjusted for smaller seq len
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.

# --- 2. 标准模块 (Standard Modules) ---
# 包括 RMSNorm, LayerNorm, RoPE/YaRN 位置编码等
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        # input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))
    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)
    def linear_ramp_factor(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len: # YaRN scaling logic
        low, high = find_correction_range(args.beta_fast, args.beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
    dtype = x.dtype
    shape = x.shape
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(dtype)
    
def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    # 简化版本，实际使用了 fast_hadamard_transform
    # 这里用一个简单的线性变换代替，以保持代码可运行
    # from fast_hadamard_transform import hadamard_transform
    # return hadamard_transform(x, scale=x.size(-1) ** -0.5)
    # 实际项目中，这是一个优化的CUDA核。这里我们仅做示意。
    return x # 简化起见，直接返回

# --- 3. DSA 核心：闪电索引器 (Indexer) ---
# 使用标准PyTorch函数重写，以展示其核心计算逻辑
class Indexer(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, qr: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.size()

        # 计算索引器的Q和K
        q = self.wq_b(qr).view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.k_norm(self.wk(x))

        # 应用位置编码和旋转
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        q = torch.cat([q_pe, q_nope], dim=-1)
        q = rotate_activation(q)
        
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)
        k = rotate_activation(k)
        
        # 简化版的索引分数计算 (替代fp8_index CUDA核)
        # 论文公式: I(t,s) = sum_h( w_h * ReLU(q_h . k_s) )
        
        # 1. 计算 q_h . k_s
        # q: (bsz, seqlen, n_heads, head_dim) -> (b, t, h, d)
        # k: (bsz, seqlen, head_dim) -> (b, s, d)
        scores_per_head = torch.einsum('bthd,bsd->btsh', q, k) # (b, t, s, h)
        scores_per_head = F.relu(scores_per_head)

        # 2. 计算 per-head 权重 w_h
        weights = self.weights_proj(x) # (b, t, h)
        
        # 3. 加权求和
        index_score = torch.einsum('btsh,bth->bts', scores_per_head, weights) * self.softmax_scale
        
        if mask is not None:
            index_score += mask

        # 选出 top-k
        topk_indices = index_score.topk(min(self.index_topk, seqlen), dim=-1)[1]
        
        return topk_indices

# --- 4. MLA (多头潜在注意力) & DSA 集成 ---
class MLA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)
        
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)
        self.softmax_scale = self.qk_head_dim ** -0.5

        self.indexer = Indexer(args)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.size()

        # 1. 低秩投影
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        
        kv_proj = self.wkv_a(x)
        kv, k_pe_in = torch.split(kv_proj, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_norm(kv)

        # 2. 计算 Q, K, V
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        q = torch.cat([q_nope, q_pe], dim=-1)

        kv_out = self.wkv_b(kv).view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_out, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        
        k_pe = apply_rotary_emb(k_pe_in.unsqueeze(2), freqs_cis).expand(-1, -1, self.n_heads, -1)
        k = torch.cat([k_nope, k_pe], dim=-1)

        # 3. DSA 集成
        topk_indices = self.indexer(x, qr, freqs_cis, mask)
        index_mask = torch.full((bsz, seqlen, seqlen), float("-inf"), device=x.device)
        index_mask.scatter_(-1, topk_indices, 0)
        
        if mask is not None:
            index_mask += mask

        # 4. 稀疏注意力计算
        scores = torch.einsum("bshd,bthd->bsht", q, k) * self.softmax_scale
        scores += index_mask.unsqueeze(2) # 应用稀疏掩码
        scores = F.softmax(scores, dim=-1)
        
        output = torch.einsum("bsht,bthd->bshd", scores, v)
        output = self.wo(output.flatten(2))

        return output

# --- 5. FFN: MLP 和 MoE ---

class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU 结构
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x, self.weight) # 计算路由分数
        scores = scores.softmax(dim=-1)
        
        weights, indices = scores.topk(self.topk, dim=-1) # 选择top-k个专家
        weights /= weights.sum(dim=-1, keepdim=True) # 归一化权重
        return weights, indices

class Expert(nn.Module):
    # MoE中的每个专家本质上是一个MLP
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) for _ in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        x = x.view(-1, dim)
        
        weights, indices = self.gate(x)
        
        # 路由和计算
        output = torch.zeros_like(x)
        flat_indices = indices.view(-1)
        
        # 将输入x按照专家的索引进行分组
        # 这是一种高效的实现方式，避免使用循环
        x_flat = x.repeat_interleave(self.n_activated_experts, dim=0)
        
        # 计算每个token被分配到的专家输出
        expert_outputs = torch.empty_like(x_flat)
        for i, expert in enumerate(self.experts):
            expert_mask = (flat_indices == i)
            if expert_mask.any():
                expert_outputs[expert_mask] = expert(x_flat[expert_mask])
        
        # 使用路由权重加权
        expert_outputs = expert_outputs * weights.view(-1, 1)
        
        # 汇总结果
        output.scatter_add_(0, indices.view(-1, 1).expand(-1, dim), expert_outputs.view(-1, dim))
        
        # 加上共享专家的输出
        output += self.shared_experts(x)
        
        return output.view(bsz, seq_len, dim)

# --- 6. Transformer 块和完整模型 ---

class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)
        self.attn = MLA(args)
        
        # 第一层使用MLP，其余层使用MoE
        is_moe_layer = layer_id >= args.n_dense_layers
        self.ffn = MoE(args) if is_moe_layer else MLP(args.dim, args.inter_dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        # Pre-Norm 结构
        h = x + self.attn(self.attn_norm(x), freqs_cis, mask)
        out = h + self.ffn(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([Block(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(args)

    def forward(self, tokens: torch.Tensor):
        bsz, seqlen = tokens.shape
        h = self.embed(tokens)
        
        device = h.device
        freqs_cis = self.freqs_cis[:seqlen].to(device)
        mask = torch.full((seqlen, seqlen), float("-inf"), device=device).triu(1) if seqlen > 1 else None

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
            
        h = self.norm(h)
        logits = self.head(h)
        
        # 返回最后一个token的logits用于生成
        return logits[:, -1, :]

# --- 示例用法 ---
if __name__ == "__main__":
    args = ModelArgs()
    model = Transformer(args).cuda()
    
    print(model)
    
    # 创建一个伪输入
    test_tokens = torch.randint(0, args.vocab_size, (1, 128)).cuda()
    
    # 模型前向传播
    with torch.no_grad():
        logits = model(test_tokens)
    
    print(f"Input shape: {test_tokens.shape}")
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (1, args.vocab_size)
    print("Simplified model runs successfully!")

