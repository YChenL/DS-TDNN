from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiAvgPooling,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
    MinkowskiReLU,
    MinkowskiGELU,
    MinkowskiTanh,
    MinkowskiSigmoid,
    MinkowskiSoftmax,
    MinkowskiBatchNorm,
)
from MinkowskiOps import (
    to_sparse,
)


import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F


class Sparse_SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(Sparse_SEModule, self).__init__()
        self.se = nn.Sequential(
            MinkowskiAvgPooling(1, dimension=1),
            MinkowskiConvolution(channels, bottleneck, kernel_size=1, bias=True, dimension=1),
            MinkowskiReLU(),
            MinkowskiConvolution(bottleneck, channels, kernel_size=1, bias=True, dimension=1),
            MinkowskiSigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x
    
    
class Sparse_Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Sparse_Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = MinkowskiConvolution(inplanes, width*scale, kernel_size=1, bias=True, dimension=1)
        self.bn1    = MinkowskiBatchNorm(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []     
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(MinkowskiConvolution(width, width, kernel_size=kernel_size, dilation=dilation, bias=True, dimension=1))
            bns.append(MinkowskiBatchNorm(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = MinkowskiConvolution(width*scale, planes, kernel_size=1, bias=True, dimension=1)
        self.bn3    = MinkowskiBatchNorm(planes)
        self.relu   = MinkowskiReLU()
        self.width  = width
        self.se     = Sparse_SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 
    
    
class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x
    
    
class Sparse_ECAPA_TDNN(nn.Module):

    def __init__(self, C):

        super(Sparse_ECAPA_TDNN, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )

        self.specaug = FbankAug() # Spec augmentation

        self.conv1  = MinkowskiConvolution(80, C, kernel_size=5, stride=1, bias=True, dimension=1)
        self.relu   = MinkowskiReLU()
        self.bn1    = MinkowskiBatchNorm(C)
        self.layer1 = Sparse_Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Sparse_Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Sparse_Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = MinkowskiConvolution(3*C, 1536, kernel_size=1, bias=True, dimension=1)
        self.attention = nn.Sequential(
            MinkowskiConvolution(4608, 256, kernel_size=1, bias=True, dimension=1),
            MinkowskiReLU(),
            MinkowskiBatchNorm(256),
            MinkowskiTanh(), # I add this layer
            MinkowskiConvolution(256, 1536, kernel_size=1, bias=True, dimension=1),
            MinkowskiSoftmax(dim=2),
            )
        self.bn5 = MinkowskiBatchNorm(3072)
        self.fc6 = MinkowskiLinear(3072, 192)
        self.bn6 = MinkowskiBatchNorm(192)


    def forward(self, x, aug):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x