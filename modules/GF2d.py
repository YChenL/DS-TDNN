import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import torch.fft



class GF2d(nn.Module):
    def __init__(self, dim, T=200, dropout=0, K=1):
        super().__init__()
        self.complex_weight = (nn.Parameter(torch.randn(dim, (T//2)+1, 2, dtype=torch.float32) * 0.02))
        trunc_normal_(self.complex_weight, std=.02)
         
    def forward(self, x):
        B, C, T = x.shape
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = self.complex_weight # [C, T//2, 2]
        if not weight.shape[1:2] == x.shape[1:2]:
            weight = F.interpolate(weight.permute(2,0,1), size=x.shape[1:2], mode='linear', align_corners=True).permute(1,2,0).contiguous()
        
        weight = torch.view_as_complex(weight)
        x = x * weight
        x = torch.fft.irfft2(x, n=(C, T), dim=(1, 2), norm='ortho')
        return x   