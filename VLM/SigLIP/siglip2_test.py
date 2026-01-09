import torch
import requests
from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image


device = "cuda" if torch.cuda.is_available() else "cpu"
# load the model and processor
ckpt = "./google/siglip2-base-patch16-naflex"
model = AutoModel.from_pretrained(ckpt, device_map="auto").to(device)
model.eval()
processor = AutoProcessor.from_pretrained(ckpt)
print(model)

# labels prompt
labels = ["cat", "dog", "lion", "monkey", "猫和狗"]
t = [f"This is a photo of a {item}" for item in labels]

# load the image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
img1 = Image.open(requests.get(url, stream=True).raw)
img2 = load_image("./dog.png")
img3 = load_image("./cat_and_dog.jpg")
images = [img1, img2, img3]

inputs = processor(text=t, images=images, return_tensors="pt").to(model.device)
print(inputs)

# run infernece
with torch.no_grad():
    outputs = model(**inputs)
# print(outputs)

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image)
print(probs)

# print(f"{probs[0][0]:.1%} that image 0 is '{labels[0]}'")
for img_idx in range(probs.shape[0]):
    print(f"\n--- 结果分析 (图片 {img_idx}) ---")
    # 获取该图片对应的所有标签概率，并排序
    current_probs = probs[img_idx]
    for label_idx, prob in enumerate(current_probs):
        print(f"标签 '{labels[label_idx]}': {prob:.1}")

"""
--- 结果分析 (图片 0) ---
标签 'cat': 0.002
标签 'dog': 2e-05
标签 'lion': 6e-05
标签 'monkey': 4e-05
标签 '猫和狗': 0.002

--- 结果分析 (图片 1) ---
标签 'cat': 7e-06
标签 'dog': 0.005
标签 'lion': 1e-05
标签 'monkey': 2e-06
标签 '猫和狗': 6e-05

--- 结果分析 (图片 2) ---
标签 'cat': 0.0003
标签 'dog': 0.0004
标签 'lion': 6e-06
标签 'monkey': 2e-07
标签 '猫和狗': 0.0003
"""