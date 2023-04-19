import torch, math
import torch.nn as nn
import torch.nn.functional as F



class res2conv1d(nn.Module):
    r""" Res2Conv1d
    """
    def __init__(self, dim, kernel_size, dilation, scale=8):
        super().__init__()
        width      = int(math.floor(dim / scale))
        self.nums  = scale -1
        convs      = []
        bns        = []
        num_pad    = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns   = nn.ModuleList(bns)
        self.act   = nn.ReLU()
        self.width = width
        
    def forward(self, x):
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.act(sp)
          sp = self.bns[i](sp)
          if i==0:
            x = sp
          else:
            x = torch.cat((x, sp), 1)
        x = torch.cat((x, spx[self.nums]),1)  
        return x
