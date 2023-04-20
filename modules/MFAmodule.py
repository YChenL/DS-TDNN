import torch, math
import torch.nn as nn
import torch.nn.functional as F



class att_TDNN(nn.Module):
    def __init__(self, C, F, CE):
        super().__init__() 
        dim = int(C*F)
        self.mlp  = nn.Linear(dim, dim) 
        self.TDNN = nn.Conv1d(dim, CE, kernel_size=1)
        
    def FCA(self, x, B, C, F):
        # time dimension GAP, x=[B, C, Freq, Time]
        skip = x
        x = torch.mean(x, dim=-1, keepdim=False).view(B, -1) # [B, CF]
        x = self.mlp(x).view(B, C, F, 1)
        return skip*x 
        
    def forward(self, x):
        B, C, F, T = x.shape
        x = self.FCA(x, B, C, F).view(B, -1, T) # [BC, F, T]
        x = self.TDNN(x).view(B, -1, 1, T) 
        return x
    

    
class MFAmodule(nn.Module):
    """ 
     MFAmodule in "MFA: TDNN WITH MULTI-SCALE FREQUENCY-CHANNEL ATTENTION FOR 
                   TEXT-INDEPENDENT SPEAKER VERIFICATION WITH SHORT UTTERANCES" ICASSP 2022
    """
    def __init__(self, dim=32, CE=32, kernel_size=3, dilation=1, scale=4, D=80):
        super().__init__()
        self.conv1 = nn.Conv2d(1, dim, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, dilation=1, padding=1)
        self.last  = nn.Conv1d(dim, CE, kernel_size=1, dilation=1)
        
        width      = int(math.floor(dim / scale))
        self.scale = scale
        convs      = []
        att_TDNNs  = []
        num_pad    = math.floor(kernel_size/2)*dilation
        for i in range(self.scale-1):
            convs.append(nn.Conv2d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
        self.convs = nn.ModuleList(convs)

        for i in range(self.scale):
            att_TDNNs.append(att_TDNN(width, D, CE))
        self.att_TDNNs = nn.ModuleList(att_TDNNs)
        self.width     = width
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
         
        spx = torch.split(x, self.width, 1)
        conv_outs = []
        conv_outs.append(self.att_TDNNs[0](spx[0]))  
        x = conv_outs[0]
        for i in range(1, self.scale):
            if i==1:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            
            sp = self.att_TDNNs[i](self.convs[i-1](sp)+conv_outs[i-1])
            conv_outs.append(sp)  
            x = torch.cat((x, conv_outs[i]),1)
     
        x = x.squeeze(-2)
        x = self.last(x) + x
        return x
