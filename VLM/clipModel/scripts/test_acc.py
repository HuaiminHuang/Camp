import os
import clip
import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from tqdm import tqdm

# 1. 环境配置
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64  # 根据显存大小调整，L/14 建议 32-64
model, preprocess = clip.load('ViT-L/14', device=device)
model.eval()

# 2. 加载全量测试集 (10,000张)
dataset = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=preprocess)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 3. 预计算文本特征 (只做一次)
print("正在编码类别文本特征...")
classes = dataset.classes
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# 4. 完整数据集评估
top1_correct = 0
top3_correct = 0 
top5_correct = 0
total_samples = 0

print(f"开始全量评估 (共 {len(dataset)} 张图片)...")
with torch.no_grad():
    # 使用 tqdm 包裹 loader，展示实时进度条
    for images, labels in tqdm(loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        
        # 提取图像特征并归一化
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # 计算相似度矩阵 [Batch, 100]
        # 注意：这里直接点积即可，无需 softmax 也能判断 Top-K
        logits = image_features @ text_features.T
        
        # 统计 Top-1 和 Top-5
        # topk 返回 (values, indices)
        _, top5_indices = logits.topk(5, dim=-1)
        
        # 扩展标签维度以方便比较: [Batch] -> [Batch, 1]
        labels_reshaped = labels.view(-1, 1)
        
        # 比较命中情况
        top1_correct += (top5_indices[:, :1] == labels_reshaped).sum().item()
        top3_correct += (top5_indices[:, :3] == labels_reshaped).sum().item()
        top5_correct += (top5_indices == labels_reshaped).any(dim=1).sum().item()
        
        total_samples += labels.size(0)

# 5. 打印最终结果
top1_acc = (top1_correct / total_samples) * 100
top3_acc = (top3_correct / total_samples) * 100
top5_acc = (top5_correct / total_samples) * 100

print("\n" + "="*30)
print(f"实验模型: ViT-L/14")
print(f"数据集: CIFAR-100 (Test Set)")
print(f"总样本数: {total_samples}")
print(f"Top-1 Accuracy: {top1_acc:.2f}%")
print(f"Top-3 Accuracy: {top3_acc:.2f}%")
print(f"Top-5 Accuracy: {top5_acc:.2f}%")
print("="*30)