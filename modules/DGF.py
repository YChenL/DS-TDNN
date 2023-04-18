import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import torch.fft
from .utils import attn_fn



class DGF(nn.Module):
    def __init__(self, dim, T=200, ratios=0.25, K=8, temperature=34):
        super().__init__()
        self.K = K
        self.complex_weight = (nn.Parameter(torch.randn(K, dim, (T//2)+1, 2, dtype=torch.float32) * 0.02))
        trunc_normal_(self.complex_weight, std=.02)
        self.fn = attn_fn(dim, ratios, K, temperature)
        
    def forward(self, x):
        B, C, T = x.shape
        weight = self.complex_weight # [K, C, T//2, 2]
        attn = self.fn(x) #[B, K]
        x = torch.fft.rfft(x, dim=-1, norm='ortho')
        if not weight.shape[2] == x.shape[-1]:
            # [2, K, C, T//2]
            weight = F.interpolate(weight.permute(3,0,1,2), size=x.shape[1:3], mode='bilinear', align_corners=True).permute(1,2,3,0).contiguous()
       
        agg_weight = torch.mm(attn, weight.reshape(self.K, -1)).reshape(B, C, -1, 2) 
        agg_weight = torch.view_as_complex(agg_weight) 
        x = x * agg_weight
        x = torch.fft.irfft(x, n=T, dim=-1, norm='ortho')
        return x  
    
    
