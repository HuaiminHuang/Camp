import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Optional, Union

class SigLIPAttentionAnalyzer:
    """
    SigLIP 注意力高级分析套件
    支持自动钩子挂载、多图批处理处理和多种可视化方案
    """
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.attentions = {}
        self.hooks = []
        self._attach_hooks()

    def _attach_hooks(self):
        """使用 PyTorch Hook 机制非侵入式捕获注意力"""
        def hook_fn(layer_idx):
            def _hook(module, input, output):
                # The attention module output is (attn_output, attn_weights)
                # We check if attn_weights (output[1]) is available
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    self.attentions[layer_idx] = output[1].detach().cpu()
            return _hook

        layers = self.model.vision_model.encoder.layers
        for i, layer in enumerate(layers):
            # Use register_forward_hook on the self_attn module
            handle = layer.self_attn.register_forward_hook(hook_fn(i))
            self.hooks.append(handle)
        print(f"成功挂载 {len(self.hooks)} 个 Vision Layer 钩子")

    def _release_hooks(self):
        """清理钩子，防止内存泄漏"""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def __del__(self):
        """清理钩子，防止内存泄漏"""
        self._release_hooks()

    def _process_attn_map(self, layer_idx: int, head_idx: Optional[int], batch_idx: int):
        """统一的注意力预处理逻辑"""
        try:
            # Shape of self.attentions[layer_idx]: (batch_size, num_heads, seq_len, seq_len)
            attn_for_batch = self.attentions[layer_idx][batch_idx]  # Shape: (num_heads, seq_len, seq_len)
        except KeyError:
            print(f"--- 错误: 无法在 'self.attentions' 中找到 Layer {layer_idx} 的键 ---")
            print(f"当前已捕获的层: {list(self.attentions.keys())}")
            raise

        if head_idx is not None:
            # Extract specific Head and average over Source Patches
            grid_attn = attn_for_batch[head_idx].mean(dim=0) 
        else:
            # Average over all Heads, then average over Source Patches
            grid_attn = attn_for_batch.mean(dim=0).mean(dim=0)

        seq_len = grid_attn.size(0)
        side = int(math.sqrt(seq_len))
        if side * side != seq_len:
            raise ValueError(f"seq_len {seq_len} is not a perfect square.")
        return grid_attn.view(side, side).numpy()

    @torch.no_grad()
    def infer(self, inputs):
        """执行推理并清空上一轮缓存"""
        self.attentions.clear()
        return self.model(**inputs, output_attentions=True)

    # ==================== 可视化方法 ====================

    def plot_single_view(self, image: Image, layer_idx: int, head_idx: int, batch_idx: int = 0):
        """方案1：单视角对比（原图、热力图、叠加图）"""
        heatmap = self._process_attn_map(layer_idx, head_idx, batch_idx)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        self._core_plot(axes[0], image, "Original Image", fontsize=12)
        self._core_plot(axes[1], heatmap, f"Layer {layer_idx}, Head {head_idx} Attention Heatmap", is_heatmap=True, fontsize=10)
        self._core_plot(axes[2], image, "Attention Overlay", fontsize=12)
        self._apply_overlay(axes[2], image, heatmap)
        
        plt.suptitle(f'ViT Attention Layer {layer_idx}, Head {head_idx}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_layer_evolution(self, image: Image, head_idx: int, batch_idx: int = 0):
        """方案2：展示某一 Head 在所有层中的演化过程（纯热力图风格）"""
        num_layers = len(self.attentions)
        if num_layers == 0:
            print("注意: 未捕获到任何注意力权重，无法生成演化图。")
            return
            
        cols = 4
        rows = math.ceil(num_layers / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten()

        for i in range(num_layers):
            if i < len(axes): # Avoid index out of bounds if num_layers is 0
                heatmap = self._process_attn_map(i, head_idx, batch_idx)
                # Plot only the heatmap, not the overlay
                self._core_plot(axes[i], heatmap, f"Layer {i}", is_heatmap=True, fontsize=10)
        
        # Hide unused subplots
        for j in range(num_layers, len(axes)):
            axes[j].axis('off')
        
        # Get seq_len for the title
        side = int(math.sqrt(self.attentions[0].shape[-1]))
        plt.suptitle(f'Head {head_idx} Attention Across Layers\n({side}x{side} patches)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_all_heads_in_layer(self, layer_idx: int, batch_idx: int = 0):
        """方案3：展示某一 Layer 中所有 Head 的注意力热力图"""
        if layer_idx not in self.attentions:
            print(f"错误: Layer {layer_idx} 的注意力权重未被捕获。")
            return

        # Get shape from the stored attention tensor
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        num_heads = self.attentions[layer_idx].shape[1]
        seq_len = self.attentions[layer_idx].shape[2]
        grid_size = int(math.sqrt(seq_len))

        # Calculate subplot layout
        cols = 4
        rows = math.ceil(num_heads / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten()

        fig.suptitle(f'All Attention Heads in Layer {layer_idx}\n({grid_size}x{grid_size} patches)', fontsize=14, fontweight='bold')

        for i in range(num_heads):
            if i < len(axes):
                heatmap = self._process_attn_map(layer_idx, head_idx=i, batch_idx=batch_idx)
                self._core_plot(axes[i], heatmap, f"Head {i}", is_heatmap=True, fontsize=10)
        
        # Hide unused subplots
        for j in range(num_heads, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
        plt.show()

    # ==================== 绘图辅助工具 ====================

    def _core_plot(self, ax, data, title, is_heatmap=False, fontsize=12):
        if is_heatmap:
            im = ax.imshow(data, cmap='viridis')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.imshow(data)
        ax.set_title(title, fontsize=fontsize)
        ax.axis('off')

    def _apply_overlay(self, ax, image, heatmap, alpha=0.6):
        # 归一化并缩放热力图
        h_min, h_max = heatmap.min(), heatmap.max()
        norm_map = (heatmap - h_min) / (h_max - h_min + 1e-8)
        mask = Image.fromarray((norm_map * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
        ax.imshow(mask, alpha=alpha, cmap='viridis')

if __name__ == "__main__":
    import torch
    import requests
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    from transformers.image_utils import load_image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load the model and processor
    ckpt = "../google/siglip2-base-patch16-naflex"
    model = AutoModel.from_pretrained(
        ckpt, 
        output_attentions=True,
        output_hidden_states=True
        # REMOVED: attn_implementation="eager" was causing issues.
    ).to(device)
    model.eval()
    print(model)
    
    processor = AutoProcessor.from_pretrained(ckpt)
    
    # labels prompt
    labels = ["cat", "dog", "lion", "monkey", "猫和狗"]
    t = [f"This is a photo of a {item}" for item in labels]

    # load the image
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    img1 = Image.open(requests.get(url, stream=True).raw)
    img2 = load_image("../data/dog.png")
    img3 = load_image("../data/cat_and_dog.jpg")
    images = [img1, img2, img3]

    # 1. 初始化
    analyzer = SigLIPAttentionAnalyzer(model)

    # 2. 执行推理
    inputs = processor(text=t, images=images, return_tensors="pt").to(device)
    analyzer.infer(inputs)

    # 3. 各种维度分析
    # 看第一张图（猫），第3层，第3个Head
    analyzer.plot_single_view(images[0], layer_idx=3, head_idx=3, batch_idx=0)

    # 看第二张图（狗），第5个Head在全流程中的变化
    analyzer.plot_layer_evolution(images[1], head_idx=5, batch_idx=1)

    # 看第三张图（猫和狗），第6层的所有Head
    analyzer.plot_all_heads_in_layer(layer_idx=6, batch_idx=2)

    # 手动清理，确保钩子被移除
    del analyzer