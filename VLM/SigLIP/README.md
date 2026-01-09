# SigLIP Vision-Language Model 项目

本项目深入研究和可视化 SigLIP (Sigmoid Loss for Language-Image Pre-training) 模型，包括其核心机制、注意力机制分析和可视化工具。

## 📋 项目概述

SigLIP 是一种改进的视觉-语言预训练方法，通过 Sigmoid Loss 替代传统的 Softmax 对比损失，实现了更高效的训练和更好的性能。本项目提供了：

- SigLIP 核心机制的实现和演示
- Vision Transformer 各层注意力的捕获和可视化
- Pooler Attention 和 Self-Attention 的对比分析
- 完整的技术分析和文档

## 📁 项目结构

```
SigLIP/
├── google/                      # HuggingFace 模型权重
│   └── siglip2-base-patch16-naflex/
├── siglip1/                     # SigLIP v1 相关文件
│   ├── hf_code/                 # HuggingFace 实现代码
│   ├── images/                  # 示例图片
│   ├── SigLIP_summary.md        # SigLIP v1 总结
│   ├── siglip.mmd               # 架构图
│   └── siglip.py                # SigLIP v1 实现
├── siglip2/                     # SigLIP v2 相关文件
│   ├── hf_code/                 # HuggingFace 实现代码
│   ├── images/                  # 示例图片
│   └── siglip2.mmd              # 架构图
├── visualization/               # 生成的可视化结果
├── cat_and_dog.jpg              # 示例图片
├── dog.png                      # 示例图片
├── pipeline-cat-chonk.jpeg      # 示例图片
├── siglip_core.py               # Sigmoid Loss 核心实现
├── SigLIP_Technical_Analysis.md # 深度技术分析文档
├── siglip2_test.py              # SigLIP v2 测试代码
├── visualization_weight.py      # Pooler Attention 可视化
└── vit_attn_weight_module.py    # Vision Layer Attention 分析工具
```

## 🚀 快速开始

### 环境要求

```bash
pip install torch torchvision transformers pillow matplotlib numpy
```

### 基础使用

#### 1. 运行 Sigmoid Loss 演示

```bash
python siglip_core.py
```

输出示例：
```
============================================================
SigLIP Sigmoid Loss Demo
============================================================

Batch size: 4
Embed dim: 512
Initial logit_scale (exp): 10.0000
Initial logit_bias: -10.0000

Final loss: 9.9388
```

#### 2. 可视化 Pooler Attention

```bash
python visualization_weight.py
```

功能：
- 可视化 Pooler Attention（probe 对所有 patches 的注意力）
- 展示每个 attention head 的独立模式
- 分析注意力权重分布

#### 3. 分析 Vision Transformer Layer Attention

```bash
python vit_attn_weight_module.py
```

功能：
- 捕获所有 12 层的 attention weights
- 可视化特定层、特定 head 的注意力模式
- 分析 attention 在层间的演化

## 📊 核心功能

### 1. Sigmoid Loss 实现 (`siglip_core.py`)

**核心创新**：将对比学习转化为二元分类任务

```python
def sigmoid_loss(image_embeds, text_embeds, logit_scale, logit_bias):
    # 归一化特征
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    
    # 计算余弦相似度
    logits_per_text = torch.matmul(text_embeds, image_embeds.t())
    logits_per_text = logits_per_text * logit_scale.exp() + logit_bias
    
    # 构建标签：对角线为1（正对），其余为-1（负对）
    eye = torch.eye(batch_size)
    m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
    
    # 计算 Sigmoid 损失
    loglik = F.logsigmoid(m1_diag1 * logits_per_text)
    loss = -torch.sum(loglik).mean()
    
    return loss
```

**关键特性**：
- 逐对独立计算，无需全局归一化
- 支持超大 batch size（实验至 1M）
- 内存效率高，支持 chunked 实现

### 2. Pooler Attention 可视化 (`visualization_weight.py`)

**Pooler 机制**：通过可学习的 probe 使用多头注意力聚合全局特征

**可视化功能**：
- 单个 head 的注意力热力图
- 所有 heads 的注意力分布
- 注意力叠加到原图的效果
- 高权重 patch 的位置标记

**颜色说明**：
- 蓝色 = 低注意力分数 (0.0)
- 绿色 = 中等注意力分数
- 红色/黄色 = 高注意力分数 (1.0)

### 3. Vision Transformer Layer Attention 分析 (`vit_attn_weight_module.py`)

**高级分析工具**：
- 自动挂载所有层的 attention hooks
- 支持多图批处理
- 多种可视化方案

**主要方法**：

```python
analyzer = SigLIPAttentionAnalyzer(model)

# 可视化特定层的所有 heads
analyzer.visualize_layer_heads(image, layer_idx=0)

# 可视化特定 head 在不同层的演化
analyzer.visualize_head_evolution(image, head_idx=0)

# 分析 attention 模式
analyzer.analyze_attention_patterns(batch_idx=0)
```

## 🔬 技术深度分析

详见 `SigLIP_Technical_Analysis.md`，包含：

### 1. SigLIP 1 vs SigLIP 2 架构演进

| 维度 | SigLIP 1 | SigLIP 2 |
|------|----------|----------|
| **核心创新** | Sigmoid Loss | 多任务统一配方 |
| **损失函数** | Sigmoid Loss | Sigmoid + LocCa + 自蒸馏 + 掩码预测 |
| **池化机制** | Attention Pooling | Attention Pooling + Mask 支持 |
| **最佳 batch size** | 32k | 32k |
| **词表大小** | 32k | 256k (Gemma tokenizer) |

### 2. Pooler Attention vs ViT Self-Attention

| 维度 | Pooler Attention | ViT Self-Attention |
|------|------------------|-------------------|
| **注意力类型** | 交叉注意力 (probe↔patches) | 自注意力 (patches↔patches) |
| **Query 来源** | 可学习的 probe | 每个 patch 自己 |
| **计算范围** | 1×256 | 256×256 |
| **目的** | 特征聚合和降维 | 特征交互和上下文建模 |
| **输出** | 1 个全局特征 | 256 个 patch 特征 |
| **位置** | 只在最后 | 在每一层 |

### 3. SigLIP vs CLIP 对比

| 维度 | CLIP | SigLIP |
|------|------|--------|
| **损失函数** | Softmax 对比损失 | Sigmoid 损失 |
| **归一化方式** | 全局归一化 | 逐对独立 |
| **内存效率** | O(B²) | O(b²) with chunking |
| **最佳 batch size** | 64k+ | 32k |
| **池化方式** | CLS token | Attention Pooling |
| **训练效率** | 2560 TPUv4-天 | 64 TPUv4-天 |

## 📈 可视化示例

### Pooler Attention 可视化

```
第 1 张图片的各注意力头分布：
┌─────┬─────┬─────┬─────┐
│Head0│Head1│Head2│Head3│
├─────┼─────┼─────┼─────┤
│Head4│Head5│Head6│Head7│
├─────┼─────┼─────┼─────┤
│Head8│Head9│Head10│Head11│
└─────┴─────┴─────┴─────┘
红色星号 = 最高注意力的 patch 位置
```

### Vision Layer Attention 演化

```
Head 0 Attention Across Layers
┌─────┬─────┬─────┬─────┐
│Layer0│Layer1│Layer2│Layer3│
├─────┼─────┼─────┼─────┤
│Layer4│Layer5│Layer6│Layer7│
├─────┼─────┼─────┼─────┤
│Layer8│Layer9│Layer10│Layer11│
└─────┴─────┴─────┴─────┘
展示 attention 如何从浅层到深层演化
```

## 🎯 关键发现

### 1. Pooler Attention 的特点

- **权重分布相对分散**：最大值约 0.115，平均 0.004
- **不同 head 关注不同区域**：某些 head 关注物体，某些关注背景
- **位置编码的影响**：某些位置的 patch（角落、边缘）可能获得更多注意力
- **不是典型的 attention sink**：而是 probe 的初始化偏差和训练策略的结果

### 2. Vision Layer Attention 的演化

- **浅层（Layer 0-3）**：关注局部特征和边缘
- **中层（Layer 4-7）**：开始关注更大的区域和物体部分
- **深层（Layer 8-11）**：关注全局语义和物体整体

### 3. Sigmoid Loss 的优势

- **小 batch 表现更好**：在 batch size < 16k 时显著优于 Softmax
- **内存效率高**：支持 chunked 实现，内存复杂度 O(b²) 而非 O(B²)
- **训练效率高**：相同性能下，训练资源减少 40 倍

## 📚 参考资料

- [SigLIP 论文](https://arxiv.org/abs/2303.15343)
- [SigLIP 2 论文](https://arxiv.org/abs/2403.08569)
- [HuggingFace SigLIP](https://huggingface.co/docs/transformers/model_doc/siglip)
- [CLIP 论文](https://arxiv.org/abs/2103.00020)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目仅供学习和研究使用。

## 📧 联系方式

如有问题或建议，请通过 Issue 联系。

---

**注意**：本项目使用的是 Google SigLIP2-base-patch16-naflex 模型，请确保已下载模型权重到 `./google/siglip2-base-patch16-naflex/` 目录。