# MHA simple concat qkv
import torch, math
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, d_model, n_head) -> None:
        super(MHA, self).__init__()
        assert d_model % n_head == 0, "not valid nums of heads"
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv_projs = nn.Linear(d_model, d_model *3) # 为了方便简化写法
        self.w_o = nn.Linear(d_model,d_model)

    def forward(self, x, mask=None):
        """
        args:
            x shape is [btz, seq_len, d_model]
        return
            attention weight - shape is [btz, seq_len, seq_len]
            y                - shape is [btz, seq_len, d_model]
        """
        btz, seq_len, _ = x.shape
        qkv = self.qkv_projs(x) # [b, s, 3*d_model]
        qkv = qkv.reshape(btz, seq_len, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # shape is [3, b, n_head, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        # print(qkv.shape)

        # scaled_dot_prod_attention
        # shape is [b, n_head, seq_len, head_dim]
        score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        # 掩码操作
        if mask is not None:
            score = score.masked_fill(mask==0, float("-inf"))
        attention_weight = torch.softmax(score, dim=-1) # [b, n_head, seq_len, seq_len]

        outputs = torch.matmul(attention_weight, v) # [b, n_head, seq_len, head_dim]
        # 重组维度为 [btz, seq_len, seq_len]
        outputs = outputs.transpose(1,2).contiguous().view(btz, seq_len, -1)
        # print(outputs.shape)

        attention_outputs = self.w_o(outputs)
        return attention_outputs, attention_weight

def main1():
    print("="*30)
    print("test assert")
    d_model = 10
    n_head = 3
    model_1 = MHA(d_model, n_head)

def main2():
    print("="*30)
    print("test shape")
    d_model = 64
    n_head = 2
    x = torch.randn(4, 32, 64)
    model_2 = MHA(d_model, n_head)
    out, weight = model_2(x)
    print("attention_outputs shape:", out.shape, "\nattention_weight shape:", weight.shape)


if __name__ == "__main__":
    # main1()

    main2()