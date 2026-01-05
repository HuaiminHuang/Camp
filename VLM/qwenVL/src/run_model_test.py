# run_model_test.py
# 实例化模型并运行一个模拟的前向传播过程，以验证数据流。

import torch
from language_and_head_modules import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig

if __name__ == "__main__":
    print("--- Simplified Qwen2.5-VL Model Test ---")

    # 1. 定义模型配置 (3B 版本)
    vision_config_3b = {
        "hidden_size": 1280, "depth": 32, "num_attention_heads": 16,
        "intermediate_size": 3456, "patch_size": 14, "window_size": 112,
        "fullatt_block_indexes": [7, 15, 23, 31], "out_hidden_size": 2048,
    }
    text_config_3b = {
        "hidden_size": 2048, "num_hidden_layers": 36, "num_attention_heads": 16,
        "num_key_value_heads": 2, "intermediate_size": 4864, "vocab_size": 151646,
        "rope_parameters": {"mrope_section": 128},
    }
    config = Qwen2_5_VLConfig(text_config=text_config_3b, vision_config=vision_config_3b)

    # 2. 实例化完整模型
    model = Qwen2_5_VLForConditionalGeneration(config)
    model.eval()
    print("Model instantiated successfully.")
    print(model)

    # 3. 准备模拟输入数据
    # 3.1 模拟文本输入 (batch_size=1, seq_len=5)
    input_ids = torch.tensor([[1, 2, config.image_token_id, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    print(f"\nOriginal input_ids shape: {input_ids.shape}")

    # 3.2 模拟图像输入
    # 假设输入图像尺寸为 448x448
    h, w = 448, 448
    patch_size = config.vision_config.patch_size
    spatial_merge_size = config.vision_config.spatial_merge_size
    # 计算视觉模块处理后，最终输出的视觉 token 数量
    num_vision_tokens = (h // patch_size // spatial_merge_size) * (w // patch_size // spatial_merge_size)
    print(f"Number of visual tokens after ViT & Merger: {num_vision_tokens}")
    # 创建模拟的视觉特征
    mock_vision_features = torch.randn(num_vision_tokens, config.text_config.hidden_size)
    print(f"Shape of mock vision features: {mock_vision_features.shape}")

    # 4. 组装多模态输入序列
    print("\n--- Assembling Multimodal Input Sequence ---")
    
    # 为了避免 embedding 时的 IndexError，先将 image_token_id 替换为有效id (e.g. 0)
    safe_input_ids = input_ids.clone()
    safe_input_ids[safe_input_ids == config.image_token_id] = 0
    # 获取文本部分的词嵌入
    text_embeds = model.model.language_model.embed_tokens(safe_input_ids)
    print(f"Shape of text embeddings (before merging): {text_embeds.shape}")

    # 找到图像占位符的位置并进行替换
    image_token_mask = (input_ids == config.image_token_id).squeeze(0)
    image_token_index = torch.where(image_token_mask)[0]
    
    final_inputs_embeds = None
    final_attention_mask = None
    if image_token_index.numel() > 0:
        start_index = image_token_index[0]
        
        # 核心拼接逻辑
        final_inputs_embeds = torch.cat([
            text_embeds[:, :start_index, :],        # 图像前的文本部分
            mock_vision_features.unsqueeze(0),      # 图像特征
            text_embeds[:, start_index + 1:, :]     # 图像后的文本部分
        ], dim=1)

        # 对应调整 attention mask
        final_attention_mask = torch.cat([
            attention_mask[:, :start_index],
            torch.ones(1, num_vision_tokens, dtype=torch.long),
            attention_mask[:, start_index + 1:]
        ], dim=1)
    else:
        # 如果没有图像，则直接使用文本输入
        final_inputs_embeds = text_embeds
        final_attention_mask = attention_mask

    print(f"Shape of final combined input embeddings: {final_inputs_embeds.shape}")
    print(f"Shape of final attention mask: {final_attention_mask.shape}")

    # 5. 执行前向传播
    print("\n--- Performing Forward Pass ---")
    # 为了简化，我们直接调用语言模型部分，并传入拼接好的 embeds
    # 在真实场景中，会调用 model.forward()，其内部会完成拼接等操作
    with torch.no_grad():
        # 直接将拼接好的序列送入语言模型
        outputs = model.model.language_model(
            inputs_embeds=final_inputs_embeds,
            attention_mask=final_attention_mask
        )
        hidden_states = outputs[0]
        print(f"Shape of LLM output hidden_states: {hidden_states.shape}")

        # 通过最后的 lm_head 得到 logits
        logits = model.lm_head(hidden_states)
        print(f"Shape of final output logits: {logits.shape}")

    print("\nForward pass successful!")
    
    # 6. 验证最终序列长度
    expected_seq_len = input_ids.shape[1] - 1 + num_vision_tokens
    print(f"Expected final sequence length: {expected_seq_len}")
    assert logits.shape[1] == expected_seq_len
    print("Assertion passed: Final sequence length matches expectation.")
