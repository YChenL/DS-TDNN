import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_dwconv1d


class gnconv1d(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, T=200, s=1.0/2.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)] # 0 1 2 dims = [dim, dim/2, dim/4]
        self.dims.reverse()
        self.proj_in = nn.Conv1d(dim, 2*dim, 1)
        if gflayer is None:
            self.dwconv = get_dwconv1d(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), T)
        
        self.proj_out = nn.Conv1d(dim, dim, 1)
        self.pws = nn.ModuleList(
            [nn.Conv1d(self.dims[i], self.dims[i+1], 1) for i in range(order-1)])
        
        self.scale = s
        print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f'%self.scale)

    def forward(self, x, mask=None, dummy=False):
        B, C, W = x.shape
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        for i in range(self.order -1):
            x = self.pws[i](x) * dw_list[i+1]

        x = self.proj_out(x)
        return x