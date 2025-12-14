import torch
import torch.nn as nn

# ================================================================
# LayerNorm 标准化
# LayerNorm =[ (x - mu) /sqrt(eps + sigma**2) ] * gama + beta
# ================================================================

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x shape is [B, ..., d_model]
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x-mean) / torch.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta
    

# ==============================================
# RMSNorm 标准化(不减去均值，只是用RMS标准化)
# RMSNorm =[ x /sqrt(eps + sigma**2) ] * gama
# ==============================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))   

    def forward(self, x):
        # x shape is [B, ..., d_model]
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return x_norm * self.gamma

def test_Norm():
    btz = 2
    seq_len = 32
    d_model = 512
    x = torch.randn(btz, seq_len, d_model)
    LN = LayerNorm(d_model)
    RMSN = RMSNorm(d_model)
    norm1 = LN(x)
    norm2 = RMSN(x)
    print("gama,beta", LN.gamma.shape, LN.beta.shape)
    print("LN", norm1.shape)
    print("=" * 20)
    print("gama", RMSN.gamma.shape)
    print("RMSN", norm2.shape)

if __name__ == "__main__":
    test_Norm()