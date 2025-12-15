import torch
import torch.nn.functional as F

# ====================
# Cross Entropy
# ====================
def CrossEnropy(logits, labels, ignore_index=100):
    """
    args:
        logits: [btz, seq_len, vocab_size]
        labels: [btz, seq_len]
    """
    b, s, v = logits.size()
    logits = logits.view(-1, v) # 按最后一维展平 [btz*seq, v]
    labels = labels.view(b * s)
    
    # 稳定数值计算减去最大值
    logits = logits - logits.max(dim=-1, keepdim=True)[0] # 取出数值(values, indices)
    log_sum_exp = torch.log(torch.exp(logits).sum(dim=-1)) 
    logits_y = logits[torch.arange(b * s), labels] # 计算

    loss = - (logits_y - log_sum_exp)
    mask = labels != ignore_index
    loss = loss[mask].mean()
    return loss



# 假设您的 CrossEnropy 函数已经定义在此处

def test_cross_enropy():
    # --- 1. 定义测试参数 ---
    btz = 2         # 批量大小
    seq_len = 16     # 序列长度
    vocab_size = 128 # 词汇量大小
    ignore_idx = 0  # 将词汇索引 0 设为忽略索引 (e.g., PAD token)

    # --- 2. 创建随机 Logits 和 Labels ---
    logits = torch.randn(btz, seq_len, vocab_size, dtype=torch.float32)
    labels = torch.randint(low=0, high=vocab_size, size=(btz, seq_len))

    # --- 3. 注入忽略索引 (Masking Test) ---
    labels[0, 12:] = ignore_idx
    print(labels)

    # --- 4. 计算自定义损失 (Your Function) ---
    custom_loss = CrossEnropy(logits, labels, ignore_index=ignore_idx)

    # --- 5. 计算 PyTorch 内置损失 (Ground Truth) ---
    logits_flat = logits.view(btz * seq_len, vocab_size)
    labels_flat = labels.view(btz * seq_len)

    # PyTorch 内置损失函数 (F.cross_entropy 或 nn.CrossEntropyLoss)
    # 关键参数：'ignore_index'
    standard_loss = F.cross_entropy(
        input=logits_flat, 
        target=labels_flat, 
        ignore_index=ignore_idx,
        reduction='mean' # 默认为 'mean'
    )

    # --- 6. 比较结果并断言 ---
    print(f"Custom Loss: {custom_loss.item():.6f}")
    print(f"Standard Loss: {standard_loss.item():.6f}")
    print("-" * 30)
    
# 执行测试
if __name__ == "__main__":
    for i in range(3):
        print(f"--- Test Run {i+1} ---")
        test_cross_enropy()
        print("\n")