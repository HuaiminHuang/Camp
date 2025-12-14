import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# FFN前馈层
# =============================================
# 标准结构 FFN(x) = RELU(xW_1 + b_1)W_2 + b_2
# =============================================
class FeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardNet, self).__init__()
        self.proj_up = nn.Linear(d_model, d_ff)
        self.proj_down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 这里的relu可以换成其他的激活函数，例如gelu
        return self.proj_down(self.dropout(F.relu(self.proj_up(x))))
    
# =============================================
# SwishGLU(x) = W_down(W_up x * swish(W_gate x))
# =============================================
class SwishGLU(nn.Module):
    def __init__(self, d_model):
        super(SwishGLU,self).__init__()
        mid_dim = d_model * 8 // 3

        self.up = nn.Linear(d_model, mid_dim)
        self.down = nn.Linear(mid_dim, d_model)
        self.gate = nn.Linear(d_model, mid_dim)

    def forward(self, x):
        return self.down(F.silu(
            self.gate(x) * self.up(x)
        ))


def test_ffn():
    batch_size = 4
    seq_len = 32
    d_model = 64
    d_ff = 256 # 一般选择1：4的比例升维

    ffn = FeedForwardNet(d_model, d_ff, dropout=0.1)
    ffn.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    y = ffn(x)

    print("model:", ffn)
    print("Inputs shape", x.shape)
    print("Output shape", y.shape)

def test_swishGLU():
    batch_size = 4
    seq_len = 32
    d_model = 64

    swishGLU = SwishGLU(d_model)
    swishGLU.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    y = swishGLU(x)
    
    print("model:", swishGLU)
    print("Inputs shape", x.shape)
    print("Output shape", y.shape)

if __name__ == "__main__":
    test_ffn()
    test_swishGLU()