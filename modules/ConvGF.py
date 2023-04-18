import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import torch.fft


class ConvGF(nn.Module):
    def __init__(self, dim, T=200, ratios=0.25, K=4, temperature=34):
        super().__init__()
        self.dw = nn.Conv1d(dim//2, dim//2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.K = K
    
        self.fn1 = nn.Conv1d(dim//2, dim//2, kernel_size=7, stride=2, groups=dim//2)
        self.fn2 = nn.Conv1d(dim//2, dim//2, kernel_size=7, stride=2, groups=dim//2)
        
        self.pre_norm = nn.BatchNorm1d(dim)
        self.post_norm = nn.BatchNorm1d(dim)
        
    def forward(self, x):
        x = self.pre_norm(x)
        B, C, T = x.shape

        rw = self.fn1(F.pad(x, (3,4), mode='reflect'))
        iw = self.fn2(F.pad(x, (3,4), mode='reflect'))
        
        weight = torch.cat([rw.unsqueeze(-1), iw.unsqueeze(-1)], dim=-1)
        weight = torch.view_as_complex(weight)
        
        x = torch.fft.rfft(x, dim=-1, norm='ortho')
        x = x * weight
        x = torch.fft.irfft(x, n=T, dim=-1, norm='ortho') # s确定输出信号形状
        
        x = self.post_norm(x)
        return x