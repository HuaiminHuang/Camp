# vision_modules.py
# 包含视觉处理相关的所有模块，以及被共同依赖的基础模块

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple

# ==========================================================================================
# 1. 基础/通用模块 (会被视觉和语言模块共同使用)
# ==========================================================================================

class PreTrainedModel(nn.Module):
    """
    一个基础的预训练模型基类，用于统一配置管理。
    所有模型模块都将继承自此类。
    """
    config: Any
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        self.config = config

    def post_init(self):
        pass # No-op

class Qwen2RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    论文中提到的，与LLM架构对齐所使用的归一化方法。
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): 输入张量
                - Shape: [..., hidden_size]
        
        Returns:
            torch.Tensor: 归一化后的张量
                - Shape: [..., hidden_size]
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class Qwen2_5_VLMLP(nn.Module):
    """
    一个标准的 MLP (多层感知机) 模块，使用 SwiGLU 激活函数。
    论文中提到，ViT 使用 SwiGLU 来与 LLM 架构对齐。
    """
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = nn.SiLU()  # SiLU 是 Swish 的一种，与 SwiGLU 思想一致

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state (torch.Tensor): 输入张量
                - Shape: [..., hidden_size]
        
        Returns:
            torch.Tensor: 处理后的张量
                - Shape: [..., hidden_size]
        """
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


# ==========================================================================================
# 2. 视觉（Vision）模块
# ==========================================================================================

def rotate_half(x):
    """辅助函数：旋转输入张量的一半维度，用于RoPE。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_vision(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """为视觉模块应用旋转位置嵌入。"""
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k, cos, sin = q.float(), k.float(), cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)

class Qwen2_5_VisionPatchEmbed(nn.Module):
    """
    (创新点C) 视觉 Patch 嵌入模块。
    使用 Conv3d 将图像/视频帧切割成块（patch）并投影到嵌入空间。
    """
    def __init__(self, patch_size: int = 14, temporal_patch_size: int = 2, in_channels: int = 3, embed_dim: int = 1152) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        # 3D 卷积核，尺寸为 [时间, 高, 宽]，对应论文中的 3D 图像块分区
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): 经过预处理的像素值。
                - Shape: [num_total_patches, C * T * P * P]
        
        Returns:
            torch.Tensor: 视觉嵌入向量。
                - Shape: [num_total_patches, embed_dim]
        """
        # 将输入重塑为 Conv3d 期望的格式
        hidden_states = hidden_states.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        # 通过 3D 卷积进行投影
        hidden_states = self.proj(hidden_states.to(dtype=self.proj.weight.dtype)).view(-1, self.embed_dim)
        return hidden_states

class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    """为视觉模块生成 2D RoPE 的 cos/sin 查找表。"""
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

class Qwen2_5_VLPatchMerger(nn.Module):
    """
    (创新点G) 视觉-语言合并器。
    将来自 ViT 的视觉 token 序列进行压缩，减少长度，以便高效地输入到 LLM 中。
    """
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        # 将 2x2=4 个相邻的 patch 特征拼接起来
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        # 一个两层的 MLP，用于压缩拼接后的特征
        self.mlp = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.GELU(), nn.Linear(self.hidden_size, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 从 ViT block 输出的视觉特征。
                - Shape: [num_patches, context_dim]
        
        Returns:
            torch.Tensor: 压缩后的视觉特征。
                - Shape: [num_patches / 4, dim]
        """
        # 假设输入 x 已经是按 4 个一组排列好的
        # 通过 view 操作实现拼接，然后通过 MLP 进行投影
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x

class Qwen2_5_VLVisionAttention(nn.Module):
    """
    视觉注意力模块。
    在简化版中，它使用标准全注意力，但在完整版中，大部分层使用窗口注意力（创新点A）。
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None, **kwargs) -> torch.Tensor:
        seq_length, _ = hidden_states.shape
        query_states, key_states, value_states = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        
        # 应用 2D 视觉 RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        
        attn_output = F.scaled_dot_product_attention(query_states.transpose(0,1), key_states.transpose(0,1), value_states.transpose(0,1), is_causal=False)
        attn_output = attn_output.transpose(0,1).reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)

class Qwen2_5_VLVisionBlock(nn.Module):
    """
    视觉 Transformer 的基本构建块。
    (创新点B) 采用 RMSNorm 和 SwiGLU(MLP) 与 LLM 对齐。
    """
    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen2_5_VLVisionAttention(config=config)
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, **kwargs)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

class Qwen2_5_VisionTransformer(PreTrainedModel):
    """完整的视觉编码器模型。"""
    def __init__(self, config):
        super().__init__(config)
        self.patch_embed = Qwen2_5_VisionPatchEmbed(patch_size=config.patch_size, temporal_patch_size=config.temporal_patch_size, in_channels=config.in_channels, embed_dim=config.hidden_size)
        self.blocks = nn.ModuleList([Qwen2_5_VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen2_5_VLPatchMerger(dim=config.out_hidden_size, context_dim=config.hidden_size, spatial_merge_size=config.spatial_merge_size)
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(config.hidden_size // config.num_heads // 2)

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.Tensor): 输入的像素数据。
                - Shape: [num_total_patches, C*T*P*P]
        
        Returns:
            torch.Tensor: 经过编码和压缩后的视觉特征。
                - Shape: [num_final_tokens, out_hidden_size]
        """
        # 1. Patch 嵌入
        x = self.patch_embed(pixel_values)
        
        # 2. 通过 Transformer Blocks (简化版的前向传播)
        # 在真实实现中，这里会有复杂的窗口注意力和 RoPE 计算
        cu_seqlens = torch.tensor([0, x.shape[0]], dtype=torch.int32, device=x.device)
        pos_emb = self.rotary_pos_emb(x.shape[0])
        pos_embed = (pos_emb.cos(), pos_emb.sin())
        for blk in self.blocks:
            x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=pos_embed)
            
        # 3. 特征合并与压缩
        x = self.merger(x)
        return x
