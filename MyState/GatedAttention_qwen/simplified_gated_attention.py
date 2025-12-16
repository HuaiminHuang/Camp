# simplified_gated_attention.py

import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class SimpleConfig:
    """一个简化的配置类，用于保存模型参数"""
    hidden_size: int = 512
    num_attention_heads: int = 8
    qkv_bias: bool = False
    
    # 门控配置
    headwise_attn_output_gate: bool = False
    elementwise_attn_output_gate: bool = False

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        self.head_dim = self.hidden_size // self.num_attention_heads
        if self.headwise_attn_output_gate and self.elementwise_attn_output_gate:
            raise ValueError("Only one of headwise_gate or elementwise_gate can be True")

class SimplifiedQwen3Attention(nn.Module):
    """
    直接从 Qwen3Attention 简化而来，保留核心门控逻辑，并详细标注维度变化。
    """
    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        # 根据门控配置调整 q_proj 输出维度
        if config.headwise_attn_output_gate:
            q_proj_output_size = self.num_heads * self.head_dim + self.num_heads
        elif config.elementwise_attn_output_gate:
            q_proj_output_size = self.num_heads * self.head_dim * 2
        else:
            q_proj_output_size = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, q_proj_output_size, bias=config.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.qkv_bias)

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states 初始形状: (bsz, q_len, hidden_size)
        bsz, q_len, _ = hidden_states.size()

        # 1. QKV 投影和门控分数提取
        query_states_with_gate = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        gate_score = None
        
        if self.config.headwise_attn_output_gate:
            # query_states_with_gate 形状: (bsz, q_len, num_heads * head_dim + num_heads)
            q_reshaped = query_states_with_gate.view(bsz, q_len, self.num_heads, self.head_dim + 1)
            # -> q_reshaped 形状: (bsz, q_len, num_heads, head_dim + 1)
            
            query_states, gate_score = torch.split(q_reshaped, [self.head_dim, 1], dim=-1)
            # -> query_states 形状: (bsz, q_len, num_heads, head_dim)
            # -> gate_score 形状:   (bsz, q_len, num_heads, 1)
        
        elif self.config.elementwise_attn_output_gate:
            # query_states_with_gate 形状: (bsz, q_len, num_heads * head_dim * 2)
            q_reshaped = query_states_with_gate.view(bsz, q_len, self.num_heads, self.head_dim * 2)
            # -> q_reshaped 形状: (bsz, q_len, num_heads, head_dim * 2)
            
            query_states, gate_score = torch.split(q_reshaped, [self.head_dim, self.head_dim], dim=-1)
            # -> query_states 形状: (bsz, q_len, num_heads, head_dim)
            # -> gate_score 形状:   (bsz, q_len, num_heads, head_dim)
        
        else: # 无门控
            query_states = query_states_with_gate.view(bsz, q_len, self.num_heads, self.head_dim)
            # -> query_states 形状: (bsz, q_len, num_heads, head_dim)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim)
        # -> key_states, value_states 形状: (bsz, q_len, num_heads, head_dim)

        # 2. 重排维度以适应 SDPA
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        # -> Q, K, V 形状变为: (bsz, num_heads, q_len, head_dim)

        # 3. 计算 Scaled Dot-Product Attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=True
        )
        # -> attn_output 形状: (bsz, num_heads, q_len, head_dim)

        # 4. 应用门控 (G1 位置)
        if gate_score is not None:
            gate_score = gate_score.transpose(1, 2)
            # gate_score 从 (bsz, q_len, num_heads, gate_dim) -> (bsz, num_heads, q_len, gate_dim)
            # gate_dim 为 1 (headwise) 或 head_dim (elementwise)
            attn_output = attn_output * torch.sigmoid(gate_score)
            # -> attn_output 形状保持不变: (bsz, num_heads, q_len, head_dim)

        # 5. 最终投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        # -> attn_output 形状: (bsz, q_len, num_heads, head_dim)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        # -> attn_output 形状: (bsz, q_len, hidden_size)
        final_output = self.o_proj(attn_output)
        # -> final_output 形状: (bsz, q_len, hidden_size)

        return final_output


# --- 演示 ---
if __name__ == '__main__':
    # 准备输入数据
    batch_size = 4
    seq_length = 64
    hidden_dim = 512
    num_heads = 8
    
    # 初始输入形状: (bsz, q_len, hidden_size)
    dummy_input = torch.randn(batch_size, seq_length, hidden_dim)
    print(f"--- 演示开始 ---")
    print(f"初始输入形状: {dummy_input.shape}\n")

    # 1. 演示 Element-wise Gating
    config_elementwise = SimpleConfig(
        hidden_size=hidden_dim, 
        num_attention_heads=num_heads, 
        elementwise_attn_output_gate=True
    )
    attn_elementwise = SimplifiedQwen3Attention(config_elementwise)
    output_elementwise = attn_elementwise(dummy_input)
    print(f"Element-wise Gating 最终输出形状: {output_elementwise.shape}")
    assert output_elementwise.shape == dummy_input.shape

    # 2. 演示 Head-wise Gating
    config_headwise = SimpleConfig(
        hidden_size=hidden_dim, 
        num_attention_heads=num_heads,
        headwise_attn_output_gate=True
    )
    attn_headwise = SimplifiedQwen3Attention(config_headwise)
    output_headwise = attn_headwise(dummy_input)
    print(f"Head-wise Gating 最终输出形状:    {output_headwise.shape}")
    assert output_headwise.shape == dummy_input.shape

    # 3. 演示无门控的基线版本
    config_baseline = SimpleConfig(
        hidden_size=hidden_dim,
        num_attention_heads=num_heads
    )
    attn_baseline = SimplifiedQwen3Attention(config_baseline)
    output_baseline = attn_baseline(dummy_input)
    print(f"无门控的基线版本最终输出形状: {output_baseline.shape}")
    assert output_baseline.shape == dummy_input.shape
    
    print("\n所有版本的输出形状均与输入形状一致，符合预期。")
