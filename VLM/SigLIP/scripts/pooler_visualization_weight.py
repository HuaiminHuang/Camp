import torch
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image


device = "cuda" if torch.cuda.is_available() else "cpu"
# load the model and processor
ckpt = "../google/siglip2-base-patch16-naflex"
model = AutoModel.from_pretrained(ckpt).to(device)
model.eval()
processor = AutoProcessor.from_pretrained(ckpt)
# print(model)

# labels prompt
labels = ["cat", "dog", "lion", "monkey", "猫和狗"]
t = [f"This is a photo of a {item}" for item in labels]

# load the image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
img1 = Image.open(requests.get(url, stream=True).raw)
img2 = load_image("../data/dog.png")
img3 = load_image("../data/cat_and_dog.jpg")
images = [img1, img2, img3]

inputs = processor(text=t, images=images, return_tensors="pt").to(model.device)

vision_embedding = model.vision_model
text_embedding = model.text_model

# visualize vision attention
attention_map = []
attention_per_head = []  # 保存每个头的权重
num_heads = model.vision_model.config.num_attention_heads  # 获取注意力头数

def hook_fn(module, input, output):
    """
    output:
      attn_output, attn_weights (当 need_weights=True)
    """
    print("HOOK TRIGGERED")
    # MultiheadAttention返回 (attn_output, attn_weights)
    # 当average_attn_weights=False时，attn_weights shape: (batch_size, num_heads, target_len, source_len)
    # target_len=1 (probe), source_len=256 (patches)
    if len(output) > 1 and output[1] is not None:
        # 形状: (batch_size, num_heads, 1, 256)
        attn_weights = output[1]
        attention_per_head.append(attn_weights)  # 保存每个头的权重

        # 手动计算所有头的平均 --> (batch_size, 1, 256)
        # squeeze(2)去掉target_len=1的维度
        attn_weights_avg = attn_weights.mean(dim=1).squeeze(1)  # (batch_size, 256)
        attention_map.append(attn_weights_avg)
    else:
        print("Warning: No attention weights returned")

# 修改MultiheadAttention以返回注意力权重
# 设置average_attn_weights=False，这样可以看到每个注意力头的单独权重
original_forward = model.vision_model.head.attention.forward
def forward_with_weights(query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None, average_attn_weights=True, is_causal=False):
    return original_forward(query, key, value, key_padding_mask=key_padding_mask, need_weights=True, attn_mask=attn_mask, average_attn_weights=False, is_causal=is_causal)

model.vision_model.head.attention.forward = forward_with_weights

handle = model.vision_model.head.attention.register_forward_hook(hook_fn)

print("="*50)
print("\nvision model:", vision_embedding)
print("="*50)
print(f"Pooler attention: {model.vision_model.head.attention}")

with torch.no_grad():
    outputs = model(**inputs)
handle.remove()

print(f"捕获到的注意力图数量: {len(attention_map)}")
print(f"注意力头数: {num_heads}")

if attention_map and attention_per_head:
    print(f"平均注意力图形状: {attention_map[-1].shape}")
    print(f"平均注意力图统计: min={attention_map[-1].min():.6f}, max={attention_map[-1].max():.6f}, mean={attention_map[-1].mean():.6f}")
    print(f"平均注意力图前10个值: {attention_map[-1][0, :10]}")

    print(f"每个头的注意力图形状: {attention_per_head[-1].shape}")

    # 分析每个注意力头的权重分布
    print("\n=== 每个注意力头的权重分析 ===")
    head_weights = attention_per_head[0]  # (batch_size, num_heads, 1, 256)
    for head_idx in range(num_heads):
        head_attn = head_weights[0, head_idx, 0, :]  # 第0张图，第head_idx个头
        print(f"Head {head_idx}: max={head_attn.max():.6f}, mean={head_attn.mean():.6f}, "
              f"top-5 indices={head_attn.topk(5).indices.tolist()}")

    # 找出平均注意力最高的10个patch
    avg_attn = attention_map[0][0, :]  # (256,)
    top_k = 10
    top_values, top_indices = avg_attn.topk(top_k)
    print(f"\n=== 平均注意力最高的{top_k}个patch ===")
    for i, (idx, val) in enumerate(zip(top_indices, top_values)):
        row = idx // 16
        col = idx % 16
        print(f"Patch {idx} (row={row}, col={col}): attention={val:.6f}")
else:
    print("未能捕获到注意力图。")


def visualize_attention_heads(image, attention_per_head, batch_idx=0, num_heads=12):
    """
    可视化每个注意力头的权重分布

    维度说明:
    - attention_per_head: 列表，每个元素是(batch_size, num_heads, 1, 256)
    - attention_per_head[0]: 第一次hook的结果，形状(batch_size, num_heads, 1, 256)
    - attention_per_head[0][batch_idx]: 第batch_idx张图片，形状(num_heads, 1, 256)
    """
    grid_size = 16  # 16x16 = 256 patches

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    # 正确的索引方式：先取第一次hook的结果，再取指定batch的图片
    head_weights = attention_per_head[0][batch_idx]  # (num_heads, 1, 256)

    for head_idx in range(min(num_heads, len(axes))):
        head_attn = head_weights[head_idx, 0, :]  # (256,)
        head_attn_np = head_attn.detach().cpu().numpy()

        # 重塑为16x16网格
        heatmap = head_attn_np.reshape(grid_size, grid_size)

        # 归一化
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # 绘制热力图
        im = axes[head_idx].imshow(heatmap_norm, cmap='jet', vmin=0, vmax=1)
        axes[head_idx].set_title(f'Head {head_idx}\nmax={head_attn_np.max():.4f}', fontsize=10)
        axes[head_idx].axis('off')

        # 标记最高注意力的patch
        max_idx = head_attn_np.argmax()
        max_row, max_col = max_idx // grid_size, max_idx % grid_size
        axes[head_idx].plot(max_col, max_row, 'w*', markersize=5, markeredgecolor='white', markeredgewidth=2)

    plt.suptitle('Attention Weights per Head (Red star = max attention)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualization_att_weight(image, attention_map, patch_size=16):
    """
    attention visualization for single image
    attention_map: shape (num_patches,) - 单张图片的注意力权重

    颜色说明:
    - 蓝色 = 低注意力分数 (0.0)
    - 绿色 = 中等注意力分数
    - 红色/黄色 = 高注意力分数 (1.0)
    """

    W, H = image.size
    # 打印原始注意力统计信息
    print(f"  原始注意力: min={attention_map.min():.6f}, max={attention_map.max():.6f}, mean={attention_map.mean():.6f}")

    # 获取注意力图的长度（patch数量）
    nums_patches = attention_map.numel()  # 获取元素总数
    # 计算网格尺寸 (例如 256 得到 16)
    grid_size = int(np.sqrt(nums_patches))

    if grid_size ** 2 != nums_patches:
        print(f"Warning: 无法将 {nums_patches}个patch完美重塑为方形网格。")
        # 如果不是正方形，你可能需要手动指定长宽比，这里暂取 grid_size
        grid_w, grid_h = grid_size, nums_patches // grid_size
    else:
        grid_w, grid_h = grid_size, grid_size

    # 1. 重塑形状 (注意：通常是 Height, Width)
    # 如果你的注意力张量是按行排列的，用 (grid_h, grid_w)
    mask = attention_map.reshape(grid_h, grid_w)
    # 保存原始值用于显示
    mask_original = mask.detach().numpy().copy()

    # 2. 转换为 Numpy 处理归一化（防止值过小导致显示全黑）
    mask = mask.detach().numpy()
    mask_normalized = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    # 3. 使用双线性插值放大到原始图片尺寸
    mask_img = Image.fromarray((mask_normalized * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)

    # 4. 绘图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(image)     # 原始图片
    ax1.axis('off')
    ax1.set_title('Original Image', fontsize=12)
    ax2.imshow(image)     # 注意力叠加图
    im = ax2.imshow(mask_img, alpha=0.6, cmap='jet')
    ax2.axis('off')
    ax2.set_title('Attention Overlay\n(Blue=Low, Red/Yellow=High)', fontsize=12)
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Attention Score', rotation=270, labelpad=15)
    im3 = ax3.imshow(mask_original, cmap='jet')     # 原始注意力热力图（不叠加图片）
    ax3.axis('off')
    ax3.set_title(f'Raw Attention Heatmap\n(Range: {mask_original.min():.4f} - {mask_original.max():.4f})', fontsize=12)
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

# 循环为每一张图片生成可视化
if attention_map and attention_per_head:
    # attention_maps[0] 现在的形状为 (batch_size, 256)
    pooled_attentions = attention_map[0]

    for i in range(len(images)):
        print(f"\n正在为第 {i+1} 张图片生成可视化...")

        # 提取第i张图片的原始图像
        image = images[i]

        # 提取第i张图片的注意力权重
        # 现在形状是 (256,)，不需要squeeze
        attention_for_image = pooled_attentions[i].cpu()

        # 调用可视化函数
        visualization_att_weight(image, attention_for_image)

        # 可视化每个注意力头的权重
        print(f"\n第 {i+1} 张图片的各注意力头分布：")
        visualize_attention_heads(image, attention_per_head, batch_idx=i, num_heads=num_heads)
else:
    print("未能捕获到注意力图。")

"""
==================================================

vision model: Siglip2VisionTransformer(
  (embeddings): Siglip2VisionEmbeddings(
    (patch_embedding): Linear(in_features=768, out_features=768, bias=True)
    (position_embedding): Embedding(256, 768)
  )
  (encoder): Siglip2Encoder(
    (layers): ModuleList(
      (0-11): 12 x Siglip2EncoderLayer(
        (layer_norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (self_attn): Siglip2Attention(
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (layer_norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (mlp): Siglip2MLP(
          (activation_fn): PytorchGELUTanh()
          (fc1): Linear(in_features=768, out_features=3072, bias=True)
          (fc2): Linear(in_features=3072, out_features=768, bias=True)
        )
      )
    )
  )
  (post_layernorm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
  (head): Siglip2MultiheadAttentionPoolingHead(
    (attention): MultiheadAttention(
      (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
    )
    (layernorm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (mlp): Siglip2MLP(
      (activation_fn): PytorchGELUTanh()
      (fc1): Linear(in_features=768, out_features=3072, bias=True)
      (fc2): Linear(in_features=3072, out_features=768, bias=True)
    )
  )
)

Pooler attention: MultiheadAttention(
  (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
)
HOOK TRIGGERED
捕获到的注意力图数量: 1
注意力头数: 12
平均注意力图形状: torch.Size([3, 256])
平均注意力图统计: min=0.000000, max=0.114877, mean=0.003906
平均注意力图前10个值: tensor([0.0011, 0.1002, 0.0022, 0.0003, 0.0008, 0.0017, 0.0221, 0.0024, 0.0013,
        0.0006], device='cuda:0')
每个头的注意力图形状: torch.Size([3, 12, 1, 256])

=== 每个注意力头的权重分析 ===
Head 0: max=0.132742, mean=0.003906, top-5 indices=[1, 81, 83, 121, 80]
Head 1: max=0.087657, mean=0.003906, top-5 indices=[81, 1, 83, 121, 99]   
Head 2: max=0.373683, mean=0.003906, top-5 indices=[208, 37, 189, 38, 19] 
Head 3: max=0.169980, mean=0.003906, top-5 indices=[1, 224, 215, 39, 119] 
Head 4: max=0.268576, mean=0.003906, top-5 indices=[224, 215, 1, 6, 221]  
Head 5: max=0.084055, mean=0.003906, top-5 indices=[1, 121, 102, 101, 81] 
Head 6: max=0.140482, mean=0.003906, top-5 indices=[56, 38, 75, 208, 189] 
Head 7: max=0.120650, mean=0.003906, top-5 indices=[1, 80, 119, 195, 158] 
Head 8: max=0.080155, mean=0.003906, top-5 indices=[56, 75, 38, 237, 243] 
Head 9: max=0.266040, mean=0.003906, top-5 indices=[224, 215, 1, 6, 221]  
Head 10: max=0.143387, mean=0.003906, top-5 indices=[1, 224, 215, 6, 81]  
Head 11: max=0.235172, mean=0.003906, top-5 indices=[224, 215, 1, 6, 221] 

=== 平均注意力最高的10个patch ===
Patch 1 (row=0, col=1): attention=0.100206
Patch 224 (row=14, col=0): attention=0.079686
Patch 215 (row=13, col=7): attention=0.054767
Patch 208 (row=13, col=0): attention=0.042153
Patch 38 (row=2, col=6): attention=0.027775
Patch 56 (row=3, col=8): attention=0.024601
Patch 75 (row=4, col=11): attention=0.023301
Patch 81 (row=5, col=1): attention=0.022815
Patch 6 (row=0, col=6): attention=0.022061
Patch 189 (row=11, col=13): attention=0.016833

正在为第 1 张图片生成可视化...
  原始注意力: min=0.000000, max=0.100206, mean=0.003906

第 1 张图片的各注意力头分布：

正在为第 2 张图片生成可视化...
  原始注意力: min=0.000083, max=0.097686, mean=0.003906

第 2 张图片的各注意力头分布：

正在为第 3 张图片生成可视化...
  原始注意力: min=0.000000, max=0.114877, mean=0.003906

第 3 张图片的各注意力头分布：
"""