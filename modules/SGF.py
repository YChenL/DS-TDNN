import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import torch.fft
from .utils import Sparse


    
class SparseGF(nn.Module):
    def __init__(self, dim, T=200, dropout=0.2, K=1):
        super().__init__()
        self.K = K
        self.complex_weight = nn.Parameter(torch.randn(dim, (T//2)+1, 2, dtype=torch.float32)*0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.sparse = Sparse(sparse=dropout)
        
    def forward_(self, x):
        B, C, T = x.shape
        x = self.pre_norm(x)
        x = torch.fft.rfft(x, dim=-1, norm='ortho')
        weight = self.complex_weight # [K, C, T//2+1, 2]
        if not weight.shape[2] == x.shape[-1]:
            weight = F.interpolate(weight.permute(2,0,1), size=x.shape[-1], mode='linear', align_corners=True).permute(1,2,0).contiguous()
            
        weight = torch.view_as_complex(weight) #[K, C, T//2+1] 
        sparse_w, mask = self.sparse(weight)
        x = (x*mask)+(x*sparse_w)
        x = torch.fft.irfft(x, n=T, dim=-1, norm='ortho') 
        return x   
    
    def forward(self, x):
        B, C, T = x.shape
        x = torch.fft.rfft(x, dim=-1, norm='ortho')
        weight = self.complex_weight # [K, C, T//2, 2]
        if not weight.shape[2] == x.shape[-1]:
            weight = F.interpolate(weight.permute(2,0,1), size=x.shape[-1], mode='linear', align_corners=True).permute(1,2,0).contiguous()
        
        weight = torch.view_as_complex(weight) 
        x = x * weight
        x = torch.fft.irfft(x, n=T, dim=-1, norm='ortho') 
        x = self.post_norm(x)
        return x    
    