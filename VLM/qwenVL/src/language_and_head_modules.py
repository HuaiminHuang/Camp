# language_and_head_modules.py
# 包含语言处理、多模态融合模型和最终生成头相关的模块

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple

# 从视觉模块文件中导入依赖的通用模块和视觉主模块
from vision_modules import (
    PreTrainedModel, 
    Qwen2RMSNorm, 
    Qwen2_5_VLMLP, 
    Qwen2_5_VisionTransformer
)

# ==========================================================================================
# 1. 配置类 (Configuration Classes)
# ==========================================================================================

class Qwen2_5_VLVisionConfig:
    """视觉模块的配置"""
    def __init__(self, hidden_size=1280, intermediate_size=3456, num_hidden_layers=32, num_attention_heads=16, patch_size=14, image_size=448, initializer_range=0.02, layer_norm_eps=1e-6, hidden_act="silu", depth=32, in_channels=3, spatial_merge_size=2, temporal_patch_size=2, tokens_per_second=4, window_size=112, out_hidden_size=2048, fullatt_block_indexes=[7, 15, 23, 31], _attn_implementation="eager", **kwargs):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.depth = depth
        self.in_channels = in_channels
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.out_hidden_size = out_hidden_size
        self._attn_implementation = _attn_implementation
        self.num_heads = num_attention_heads

class Qwen2_5_VLTextConfig:
    """语言模块的配置"""
    def __init__(self, vocab_size=152064, hidden_size=2048, intermediate_size=4864, num_hidden_layers=36, num_attention_heads=16, num_key_value_heads=2, hidden_act="silu", max_position_embeddings=32768, initializer_range=0.02, rms_norm_eps=1e-05, use_cache=True, tie_word_embeddings=False, use_sliding_window=False, sliding_window=4096, max_window_layers=36, attention_dropout=0.0, rope_theta=1000000.0, pad_token_id=None, layer_types=None, rope_parameters=None, _attn_implementation="eager", **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        self.pad_token_id = pad_token_id
        if layer_types is None:
            self.layer_types = ["full_attention"] * num_hidden_layers
        else:
            self.layer_types = layer_types
        self.rope_parameters = rope_parameters if rope_parameters is not None else {}
        self.rope_parameters.setdefault("mrope_section", 128)
        self._attn_implementation = _attn_implementation

class Qwen2_5_VLConfig:
    """顶层总配置，整合视觉和语言配置"""
    def __init__(self, text_config=None, vision_config=None, image_token_id=151655, video_token_id=151656, vision_start_token_id=151652, vision_end_token_id=151653, **kwargs):
        self.text_config = Qwen2_5_VLTextConfig(**text_config) if isinstance(text_config, dict) else text_config if text_config is not None else Qwen2_5_VLTextConfig()
        self.vision_config = Qwen2_5_VLVisionConfig(**vision_config) if isinstance(vision_config, dict) else vision_config if vision_config is not None else Qwen2_5_VLVisionConfig()
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.tie_word_embeddings = self.text_config.tie_word_embeddings


# ==========================================================================================
# 2. 语言（Language）与融合模块
# ==========================================================================================

def rotate_half(x):
    """辅助函数：旋转输入张量的一半维度，用于RoPE。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    辅助函数：为 GQA 复制 Key 和 Value 的头。
    """
    batch, num_kv_heads, seqlen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seqlen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seqlen, head_dim)

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """
    (创新点D) 应用多模态旋转位置嵌入。
    通过交错应用 T, H, W 三个维度的旋转，为 Q, K 向量注入时空信息。
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen2_5_VLRotaryEmbedding(nn.Module):
    """
    (创新点D) 为语言模型生成 MRoPE 的 cos/sin 查找表。
    """
    def __init__(self, config: Qwen2_5_VLTextConfig, device=None):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, position_ids):
        """
        Args:
            x: 输入张量，仅用于获取 dtype 和 device。
            position_ids (torch.Tensor): 3D 位置ID张量。
                - Shape: [3, batch_size, sequence_length]
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: cos 和 sin 查找表。
                - Shape: [3, batch_size, sequence_length, head_dim]
        """
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)

class Qwen2_5_VLAttention(nn.Module):
    """
    语言模型的注意力模块，实现了分组查询注意力 (GQA)。
    """
    def __init__(self, config: Qwen2_5_VLTextConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.mrope_section = config.rope_parameters["mrope_section"]

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(query_states, key_states, cos, sin, self.mrope_section)

        # GQA: 复制 K, V 头以匹配 Q 的头数
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            attention_mask = (attention_mask == 0)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, is_causal=attention_mask is None)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output), None

class Qwen2_5_VLDecoderLayer(nn.Module):
    """标准的 Transformer 解码器层。"""
    def __init__(self, config: Qwen2_5_VLTextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen2_5_VLAttention(config, layer_idx)
        self.mlp = Qwen2_5_VLMLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states,)

class Qwen2_5_VLTextModel(PreTrainedModel):
    """语言模型主干，由多个解码器层堆叠而成。"""
    def __init__(self, config: Qwen2_5_VLTextConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([Qwen2_5_VLDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, **kwargs):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if inputs_embeds is None:
            if input_ids is None:
                 raise ValueError("You have to specify either input_ids or inputs_embeds")
            inputs_embeds = self.embed_tokens(input_ids)
        
        # 如果没有提供位置ID，则创建默认位置ID
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)

        # 生成 MRoPE 旋转矩阵
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        
        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings)
            hidden_states = layer_outputs[0]
            
        hidden_states = self.norm(hidden_states)
        return (hidden_states,)

class Qwen2_5_VLModel(PreTrainedModel):
    """
    顶层多模态模型，将视觉模块和语言模块连接在一起。
    """
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformer(config.vision_config)
        self.language_model = Qwen2_5_VLTextModel(config.text_config)

    def forward(self, input_ids, pixel_values, attention_mask=None, **kwargs):
        # 简化版 forward，真实实现中，视觉特征会在这里与文本嵌入进行拼接
        vision_features = self.visual(pixel_values)
        return self.language_model(input_ids=input_ids, attention_mask=attention_mask)

class Qwen2_5_VLForConditionalGeneration(PreTrainedModel):
    """
    最终的模型，包含了用于生成文本的 lm_head。
    """
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        self.model = Qwen2_5_VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None, **kwargs):
        """
        Args:
            input_ids, pixel_values, ... : 模型的各种输入。
        
        Returns:
            logits (torch.Tensor): 最终的预测分数。
                - Shape: [batch_size, sequence_length, vocab_size]
        """
        outputs = self.model(input_ids, pixel_values, attention_mask, **kwargs)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        return (logits,)
