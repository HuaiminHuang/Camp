import os
import clip
import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 1. 配置
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32  # 每一批处理 32 张图
NUM_WORKERS = 4  # 使用 4 个线程加载数据
model, preprocess = clip.load('ViT-L/14', device=device)

# 2. 数据集与加载器 (DataLoader 自动帮你处理了 unsqueeze 和 batch 组合)
dataset = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=preprocess)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# 3. 预准备文本特征 (保持不变，因为类别是固定的)
classes = dataset.classes
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# 4. 开始批量推理
os.makedirs("../outputs/res", exist_ok=True)
print(f"开始批量推理，Batch Size: {BATCH_SIZE}...")

with torch.no_grad():
    # 取第一批数据作为演示
    images, labels = next(iter(loader)) 
    images = images.to(device)
    
    image_features = model.encode_image(images)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    # 计算整个批次的相似度 (Matrix Multiplication: [32, 768] @ [768, 100])
    # 结果维度: [32, 100]
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # 获取每张图的前 5 名
    top_probs, top_indices = similarity.topk(5, dim=-1)

    # 计算batch accuracy 
    correct = (top_indices[:, 0] == labels.to(device)).sum().item()
    accuracy = correct / len(labels) * 100
    print(f"\n当前 Batch 准确率: {accuracy:.2f}%")

# 5. 展示并保存该批次的前 4 张图作为示例
for i in range(4):
    plt.figure(figsize=(10, 4))
    
    # 还原图片用于显示 (从 Tensor 转回图像)
    # 注意：preprocess 后的图像被标准化了，直接显示颜色会怪异，这里仅作示意
    img_display = images[i].cpu().permute(1, 2, 0).numpy()
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min()) # 简易归一化显示
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_display)
    plt.title(f"True: {classes[labels[i]]}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    p = top_probs[i].cpu().numpy() * 100
    lbls = [classes[idx] for idx in top_indices[i].cpu().numpy()]
    plt.barh(lbls, p, color='lightgreen')
    plt.gca().invert_yaxis()
    plt.title("Top 5")
    
    plt.tight_layout()
    plt.savefig(f"../outputs/res/batch_res_{i}.png")
    plt.close() # 释放内存

print("批量结果已保存至 ../outputs/res/")