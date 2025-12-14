import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MHA(nn.Module):
    # attention scorse
    # Attn = softmaxx(QK^T / sqrt(d_k)) V
    def scaled_dot_prod_attention(self, Q, K, V, mask=None):
        """
        args:
            Q/KV: [b, h, s_(q/kv), d]
            mask: [b, h, s_q, s_kv]
        return
            out: [b, h, s_q, d]
        """
        embd_size = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(embd_size)

        if mask is not None:
            scores = scores.masked_fill(mask==0, float("-inf"))

        attn_weight = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weight, V)

        return out, attn_weight


    def __init__(self, d_model, head):
        super(MHA, self).__init__()
        # 断言d_model 需要被 head 整除
        assert d_model % head == 0
        self.d_model = d_model
        self.h = head

        # QKV Linear
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        args:
            QKV shape is [b, s_(q/kv), d_model]
            mask shape is [b, 1, s_q, s_kv]
        output:
            out shape is [b, s_q, d_model]
        """
        btz, seq_len, _ = q.size()
        _, seq_len_kv, _ = k.size()

        # reshape矩阵 shape is [b, s_(q/kv), h, head_dim] --> [b, h, s_(q/kv), head_dim]
        Q = self.w_q(q).view(btz, seq_len, self.h, -1).transpose(1,2)
        K = self.w_k(k).view(btz, seq_len_kv, self.h, -1).transpose(1,2)
        V = self.w_v(v).view(btz, seq_len_kv, self.h, -1).transpose(1,2)

        # shape is [b. h. s_q. h_dim]
        scaled_dot_attention, attn_weight = self.scaled_dot_prod_attention(Q,K,V)
        # shape to [b, s_q, h, h_dim] --> [b, s_q, d_model]
        conacted_output = (scaled_dot_attention
                           .transpose(1, 2)
                           .contiguous().view(btz, -1, self.d_model))
        out = self.fc_out(conacted_output)
        return out, attn_weight
    
def test_MHA():
    d_model = 512
    head = 8
    model = MHA(d_model, head)
    model.eval()

    btz = 2
    s_q = 16
    s_kv = 32

    query = torch.randn(btz, s_q, d_model)
    key = torch.randn(btz, s_kv, d_model)
    value = torch.randn(btz, s_kv, d_model)

    # padding / causal mask 示例
    mask = torch.ones(btz, 1, s_q, s_kv)
    mask[:, :, :, s_q:] = 0

    with torch.no_grad():
        out, atten_weight = model(query, key, value, mask)

    print("model is:", model)
    print("="*20)
    print("outputs shape is ", out.shape)
    print("outputs weights shape is ", atten_weight.shape)
    # print(atten_weight)

if __name__ == "__main__":
    test_MHA()
