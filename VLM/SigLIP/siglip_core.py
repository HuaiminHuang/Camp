
"""
SigLIP核心实现：Sigmoid对比损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
    return_loss: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SigLIP的Sigmoid对比损失核心实现
    
    参数:
        image_embeds: 图像嵌入 [batch_size, embed_dim]
        text_embeds: 文本嵌入 [batch_size, embed_dim]
        logit_scale: 可学习的温度参数 [1]
        logit_bias: 可学习的偏置参数 [1]
        return_loss: 是否返回损失
    
    返回:
        logits_per_image: 图像到文本的logits [batch_size, batch_size]
        logits_per_text: 文本到图像的logits [batch_size, batch_size]
        loss: Sigmoid对比损失（如果return_loss=True）
    """
    # 1. 归一化特征
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    
    # 2. 计算余弦相似度作为logits
    # logits_per_text[i, j] = 文本i与图像j的相似度
    logits_per_text = torch.matmul(text_embeds, image_embeds.t())
    
    # 3. 应用可学习的温度和偏置
    # logit_scale初始化为log(10)，通过exp()确保为正
    # logit_bias初始化为-10
    logits_per_text = logits_per_text * logit_scale.exp() + logit_bias
    
    # 4. 转置得到图像到文本的logits
    logits_per_image = logits_per_text.t()
    
    loss = None
    if return_loss:
        # 5. 构建标签矩阵
        # 对角线元素为1（正对），非对角线为-1（负对）
        batch_size = logits_per_text.size(0)
        eye = torch.eye(batch_size, device=logits_per_text.device)
        # m1_diag1 = -1 + 2*eye = 对角线为1，其余为-1
        m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
        
        # 6. 计算sigmoid损失
        # log_sigmoid(m1_diag1 * logits) = 
        #   - 对于正对 (i,i): log_sigmoid(logits[i,i])
        #   - 对于负对 (i,j≠i): log_sigmoid(-logits[i,j])
        loglik = F.logsigmoid(m1_diag1 * logits_per_text)
        
        # 7. 计算负对数似然
        # 对每行求和：每个文本与所有图像的loss
        nll = -torch.sum(loglik, dim=-1)
        
        # 8. 平均得到最终损失
        loss = nll.mean()
    
    return logits_per_image, logits_per_text, loss


class SiglipLoss(nn.Module):
    """
    SigLIP损失模块封装
    """
    def __init__(self, logit_scale_init: float = 10.0, logit_bias_init: float = -10.0):
        super().__init__()
        # 可学习的温度参数，初始化为log(10)
        self.logit_scale = nn.Parameter(torch.tensor([float(logit_scale_init)]).log())
        # 可学习的偏置参数，初始化为-10
        self.logit_bias = nn.Parameter(torch.tensor([float(logit_bias_init)]))
    
    def forward(
        self, 
        image_embeds: torch.Tensor, 
        text_embeds: torch.Tensor,
        return_loss: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return sigmoid_loss(
            image_embeds, text_embeds, 
            self.logit_scale, self.logit_bias, 
            return_loss
        )


# ==================== 示例与测试 ====================

def demo_sigmoid_loss():
    """演示Sigmoid损失的计算过程"""
    print("=" * 60)
    print("SigLIP Sigmoid Loss Demo")
    print("=" * 60)
    
    # 设置参数
    batch_size = 4
    hidden_dim = 512
    
    # 模拟嵌入
    torch.manual_seed(42)
    image_embeds = torch.randn(batch_size, hidden_dim)
    text_embeds = torch.randn(batch_size, hidden_dim)
    
    # 创建损失模块
    loss_module = SiglipLoss()
    
    print(f"\nBatch size: {batch_size}")
    print(f"Embed dim: {hidden_dim}")
    print(f"Initial logit_scale (exp): {loss_module.logit_scale.exp().item():.4f}")
    print(f"Initial logit_bias: {loss_module.logit_bias.item():.4f}")
    
    # 前向计算
    logits_per_image, logits_per_text, loss = loss_module(image_embeds, text_embeds)
    
    print(f"\nLogits per text shape: {logits_per_text.shape}")
    print(f"Logits per image shape: {logits_per_image.shape}")
    print(f"\nLogits per text:\n{logits_per_text.detach().numpy()}")
    
    print(f"\nFinal loss: {loss.item():.4f}")
    
    # 验证梯度
    loss.backward()
    print(f"\nGradient of logit_scale: {loss_module.logit_scale.grad.item():.4f}")
    print(f"Gradient of logit_bias: {loss_module.logit_bias.grad.item():.4f}")
    
    return loss


def compare_with_softmax():
    """对比Sigmoid与Softmax损失的行为差异"""
    print("\n" + "=" * 60)
    print("Comparing Sigmoid vs Softmax Behavior")
    print("=" * 60)
    
    batch_size = 4
    embed_dim = 512
    
    torch.manual_seed(42)
    image_embeds = torch.randn(batch_size, embed_dim)
    text_embeds = torch.randn(batch_size, embed_dim)
    
    # 归一化
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    
    # 计算相似度
    logits = torch.matmul(text_embeds, image_embeds.t()) * 10.0  # 温度=10
    
    print(f"\nSimilarity logits:\n{logits.detach().numpy()}")
    
    # Softmax损失
    labels = torch.arange(batch_size)
    loss_softmax = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    loss_softmax = loss_softmax / 2
    
    print(f"\nSoftmax loss: {loss_softmax.item():.4f}")
    
    # Sigmoid损失
    eye = torch.eye(batch_size)
    m1_diag1 = -torch.ones_like(logits) + 2 * eye
    loglik = F.logsigmoid(m1_diag1 * logits)
    loss_sigmoid = -torch.sum(loglik) / batch_size
    
    print(f"Sigmoid loss: {loss_sigmoid.item():.4f}")
    
    # 分析hard negative的影响
    print("\n--- Hard Negative Analysis ---")
    # 模拟一个hard negative（让第0个文本与第1个图像非常相似）
    logits_hard = logits.clone()
    logits_hard[0, 1] = logits_hard[0, 0] + 0.5  # 使负样本接近正样本
    
    loss_softmax_hard = F.cross_entropy(logits_hard, labels) + F.cross_entropy(logits_hard.t(), labels)
    loss_softmax_hard = loss_softmax_hard / 2
    
    loglik_hard = F.logsigmoid(m1_diag1 * logits_hard)
    loss_sigmoid_hard = -torch.sum(loglik_hard) / batch_size
    
    print(f"With hard negative:")
    print(f"  Softmax loss change: {loss_softmax_hard.item() - loss_softmax.item():.4f}")
    print(f"  Sigmoid loss change: {loss_sigmoid_hard.item() - loss_sigmoid.item():.4f}")


if __name__ == "__main__":
    # 运行演示
    demo_sigmoid_loss()
    compare_with_softmax()


"""
============================================================
SigLIP Sigmoid Loss Demo
============================================================

Batch size: 4
Embed dim: 512
Initial logit_scale (exp): 10.0000
Initial logit_bias: -10.0000

Logits per text shape: torch.Size([4, 4])
Logits per image shape: torch.Size([4, 4])

Logits per text:
[[ -9.805354  -10.195884  -10.439035  -10.243232 ]
 [ -9.841523   -9.426561  -10.228082  -10.14378  ]
 [-10.252529   -9.656919  -10.379839  -10.339    ]
 [-10.596301  -10.193632  -10.3856535 -10.142628 ]]

Final loss: 9.9388

Gradient of logit_scale: -0.0614
Gradient of logit_bias: -0.9998

============================================================
Comparing Sigmoid vs Softmax Behavior
============================================================

Similarity logits:
[[ 0.19464566 -0.19588387 -0.4390359  -0.24323168]
 [ 0.15847637  0.57343847 -0.2280818  -0.14377958]
 [-0.25252935  0.3430815  -0.37983936 -0.33899963]
 [-0.5963007  -0.19363198 -0.3856535  -0.14262724]]

Softmax loss: 1.2148
Sigmoid loss: 2.4819

--- Hard Negative Analysis ---
With hard negative:
  Softmax loss change: 0.0640
  Sigmoid loss change: 0.1249

"""