
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import inspect
from typing import Any, Optional, Union, Tuple, List, Callable
from dataclasses import dataclass

# Helper functions and basic building blocks from transformers library, included for self-containment.

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# Configuration Classes
class Qwen2_5_VLVisionConfig:
    def __init__(
        self,
        hidden_size=1280,
        intermediate_size=3456,
        num_hidden_layers=32,
        num_attention_heads=16,
        patch_size=14,
        image_size=448,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        hidden_act="silu",
        # Custom params
        depth=32,
        in_channels=3,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=4,
        window_size=112,
        out_hidden_size=2048,
        fullatt_block_indexes=[7, 15, 23, 31],
        _attn_implementation="eager",
        **kwargs,
    ):
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
        self.num_heads = num_attention_heads # for vision attention

class Qwen2_5_VLTextConfig:
    def __init__(
        self,
        vocab_size=152064,
        hidden_size=2048,
        intermediate_size=4864,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        tie_word_embeddings=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=36,
        attention_dropout=0.0,
        rope_theta=1000000.0,
        pad_token_id=None,
        layer_types=None,
        rope_parameters=None,
        _attn_implementation="eager",
        **kwargs,
    ):
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
    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        **kwargs,
    ):
        self.text_config = Qwen2_5_VLTextConfig(**text_config) if isinstance(text_config, dict) else text_config if text_config is not None else Qwen2_5_VLTextConfig()
        self.vision_config = Qwen2_5_VLVisionConfig(**vision_config) if isinstance(vision_config, dict) else vision_config if vision_config is not None else Qwen2_5_VLVisionConfig()
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.tie_word_embeddings = self.text_config.tie_word_embeddings

# Dummy PreTrainedModel
class PreTrainedModel(nn.Module):
    config: Any
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        self.config = config

    def post_init(self):
        pass # No-op

# Model Components
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class Qwen2_5_VLMLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 14, temporal_patch_size: int = 2, in_channels: int = 3, embed_dim: int = 1152) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states

class Qwen2_5_VisionRotaryEmbedding(nn.Module):
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
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.GELU(), nn.Linear(self.hidden_size, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x

def apply_rotary_pos_emb_vision(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k, cos, sin = q.float(), k.float(), cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)

class Qwen2_5_VLVisionAttention(nn.Module):
    """简化设计，qwen2.5-VL在固定层数使用滑动窗口注意力，减少计算复杂度"""
    def __init__(self, config: Qwen2_5_VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None, **kwargs) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        
        attn_output = F.scaled_dot_product_attention(query_states.transpose(0,1), key_states.transpose(0,1), value_states.transpose(0,1), is_causal=False)
        attn_output = attn_output.transpose(0,1).reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)

class Qwen2_5_VLVisionBlock(nn.Module):
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
    def __init__(self, config: Qwen2_5_VLVisionConfig):
        super().__init__(config)
        self.patch_embed = Qwen2_5_VisionPatchEmbed(patch_size=config.patch_size, temporal_patch_size=config.temporal_patch_size, in_channels=config.in_channels, embed_dim=config.hidden_size)
        self.blocks = nn.ModuleList([Qwen2_5_VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen2_5_VLPatchMerger(dim=config.out_hidden_size, context_dim=config.hidden_size, spatial_merge_size=config.spatial_merge_size)
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(config.hidden_size // config.num_heads // 2)
        # Simplified forward pass for architecture printing
    def forward(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        # This is a dummy forward, the actual one is more complex.
        # It's enough to show the architecture.
        x = self.patch_embed(pixel_values)
        # Dummy values for showing architecture
        cu_seqlens = torch.tensor([0, x.shape[0]], dtype=torch.int32)
        pos_emb = self.rotary_pos_emb(x.shape[0])
        pos_embed = (pos_emb.cos(), pos_emb.sin())
        for blk in self.blocks:
            x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=pos_embed)
        x = self.merger(x)
        return x

class Qwen2_5_VLRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2_5_VLTextConfig, device=None):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_kv_heads, seqlen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seqlen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seqlen, head_dim)

class Qwen2_5_VLAttention(nn.Module):
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
        query_states, key_states, value_states = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(query_states, key_states, cos, sin, self.mrope_section)

        # Repeat kv_heads to match query_heads for Grouped Query Attention
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            # PyTorch's scaled_dot_product_attention expects a boolean mask where True means "to ignore".
            # The input mask is HF-style (1=attend, 0=pad). So we convert it.
            attention_mask = (attention_mask == 0)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, is_causal=attention_mask is None)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output), None

class Qwen2_5_VLDecoderLayer(nn.Module):
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
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings)
            hidden_states = layer_outputs[0]
        hidden_states = self.norm(hidden_states)
        return (hidden_states,)

class Qwen2_5_VLModel(PreTrainedModel):
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformer(config.vision_config)
        self.language_model = Qwen2_5_VLTextModel(config.text_config)

    def forward(self, input_ids, pixel_values, attention_mask=None, **kwargs):
        # Simplified forward to show architecture. Does not handle merging of modalities.
        vision_features = self.visual(pixel_values)
        # A real implementation would merge vision_features into the language model's inputs
        return self.language_model(input_ids, attention_mask)

class Qwen2_5_VLForConditionalGeneration(PreTrainedModel):
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        self.model = Qwen2_5_VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(input_ids, pixel_values, attention_mask)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        return (logits,)

if __name__ == "__main__":
    print("--- Simplified Qwen2.5-VL Model Architecture ---")

    # Configuration for the Qwen2.5-VL-3B model
    vision_config_3b = {
        "hidden_size": 1280,
        "depth": 32,
        "num_attention_heads": 16,
        "intermediate_size": 3456,
        "patch_size": 14,
        "window_size": 112,
        "fullatt_block_indexes": [7, 15, 23, 31],
        "out_hidden_size": 2048, # Merger output dim
    }

    text_config_3b = {
        "hidden_size": 2048,
        "num_hidden_layers": 36,
        "num_attention_heads": 16, # 2048 / 128
        "num_key_value_heads": 2,
        "intermediate_size": 4864,
        "vocab_size": 151646,
        "rope_parameters": {"mrope_section": 128}, # head_dim
    }

    # Create the full model configuration
    config = Qwen2_5_VLConfig(
        text_config=text_config_3b,
        vision_config=vision_config_3b
    )

    # Instantiate the model
    model = Qwen2_5_VLForConditionalGeneration(config)

    # Print the model architecture
    print(model)

    # Print total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal anzahl of parameters: {total_params / 1_000_000:.2f}M")
    model.eval() # Set to evaluation mode

    print("--- Running forward pass with mock data ---")

    # 1. Mock Text Input
    # This represents a batch of 1 with a sequence of token IDs.
    # We include a placeholder for the image.
    input_ids = torch.tensor([[1, 2, config.image_token_id, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    # 2. Mock Image Input
    # The vision transformer expects a specific input format that the processor usually prepares.
    # For this test, we will create a dummy tensor that mimics a pre-processed image.
    # The VisionPatchEmbed layer uses a 3D convolution, suggesting it can handle temporal data.
    # For a single image, we can simulate a "video" of temporal_patch_size frames.
    
    # Based on the Conv3d in VisionPatchEmbed, the input needs to be divisible into patches.
    # Let's create a tensor that fits the `view` operation in the patch embedder.
    # The view is: (-1, in_channels, temporal_patch_size, patch_size, patch_size)
    h = 448 # Example image height
    w = 448 # Example image width
    
    patch_size = config.vision_config.patch_size
    temporal_patch_size = config.vision_config.temporal_patch_size
    in_channels = config.vision_config.in_channels
    
    # Calculate number of patches
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    # For a single static image, we can imagine the temporal dimension is 1, but the model processes
    # frames in pairs (temporal_patch_size=2). We create a mock input of 2 identical frames.
    num_temporal_patches = 1 
    num_total_patches = num_patches_h * num_patches_w * num_temporal_patches

    # Create a mock tensor that can be reshaped correctly by the patch embedder.
    # The shape should correspond to the total elements of all patches.
    mock_pixel_values = torch.randn(
        num_total_patches,
        in_channels * temporal_patch_size * patch_size * patch_size
    )
    # The vision forward pass in this simplified model is a placeholder. 
    # To make it work, we will mock the output of the vision tower directly.
    
    # Let's replace the model's vision forward pass with a mock one for this test
    # This avoids dealing with the complex vision input processing.
    
    # The vision features need to be merged into the text embeddings.
    # The final number of vision tokens is (H/patch/merge) * (W/patch/merge)
    spatial_merge_size = config.vision_config.spatial_merge_size
    num_vision_tokens = (h // patch_size // spatial_merge_size) * (w // patch_size // spatial_merge_size)
    
    # Let's create mock vision features that will replace the <image> token embedding
    mock_vision_features = torch.randn(num_vision_tokens, config.text_config.hidden_size)

    # 3. Prepare inputs for the language model
    # Get embeddings for text tokens
    # We clone the input_ids and replace the image token with a valid ID (0)
    # to avoid an IndexError, since the image token ID is outside the vocab range.
    # The embedding at this position will be replaced by vision features anyway.
    safe_input_ids = input_ids.clone()
    safe_input_ids[safe_input_ids == config.image_token_id] = 0
    text_embeds = model.model.language_model.embed_tokens(safe_input_ids)

    # Replace the image token embedding with the mock vision features
    image_token_mask = (input_ids == config.image_token_id).squeeze(0)
    
    # Find the index of the image token
    image_token_index = torch.where(image_token_mask)[0]
    
    if image_token_index.numel() > 0:
      start_index = image_token_index[0]
      
      # Concatenate embeddings: text before image, vision features, text after image
      final_inputs_embeds = torch.cat([
          text_embeds[:, :start_index, :],
          mock_vision_features.unsqueeze(0),
          text_embeds[:, start_index + 1:, :]
      ], dim=1)

      # Adjust attention mask
      final_attention_mask = torch.cat([
          attention_mask[:, :start_index],
          torch.ones(1, num_vision_tokens, dtype=torch.long),
          attention_mask[:, start_index + 1:]
      ], dim=1)
    else:
      final_inputs_embeds = text_embeds
      final_attention_mask = attention_mask


    # 4. Run the forward pass with the combined embeddings
    # We pass inputs_embeds directly to the text model
    print(f"Shape of final input embeddings: {final_inputs_embeds.shape}")
    print(f"Shape of final attention mask: {final_attention_mask.shape}")

    # For simplicity, we call the language model directly
    outputs = model.model.language_model(
        inputs_embeds=final_inputs_embeds,
        attention_mask=final_attention_mask
    )

    hidden_states = outputs[0]
    logits = model.lm_head(hidden_states)

    print(f"Forward pass successful!")
    print(f"Output logits shape: {logits.shape}")
    # Expected logits shape: (batch_size, sequence_length_with_vision, vocab_size)
    expected_seq_len = input_ids.shape[1] - 1 + num_vision_tokens
    print(f"Expected sequence length: {expected_seq_len}")

