from functools import partial
import torch, torchaudio, math
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import torch.fft

from modules.utils import *
from modules.GF import GF
from modules.DGF import DGF
from modules.SGF import SparseGF
from modules.SDGF import SparseDGF
from modules.Res2Conv import res2conv1d
from modules.MSA import MSA



class LocalBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1, scale=4, drop_path=0.):
        super().__init__()
        self.res2conv = res2conv1d(dim, kernel_size, dilation, scale)     
        # self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)   
           
        self.norm1 = nn.BatchNorm1d(dim)   
        self.norm2 = nn.BatchNorm1d(dim)   
        self.norm3 = nn.BatchNorm1d(dim)   
        self.proj1 = nn.Conv1d(dim, dim, kernel_size=1)  
        self.proj2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.act   = nn.ReLU()
        # self.act = nn.GELU()
        self.se    = SEModule(dim)

    def forward(self, x):
        skip = x
        
        x = self.proj1(x)
        x = self.act(x)
        x = self.norm1(x)
        
        x = self.res2conv(x)
        # x = self.dwconv(x)
        # x = self.act(x)
        # x = self.norm2(x)
  
        x = self.proj2(x)
        x = self.act(x)
        x = self.norm3(x)
        
        x = skip + self.se(x)
        
        return x    
    
    

class GlobalBlock(nn.Module):
    """ 
     Global block: if global modules = MSA or LSTM, need to permute the dimension of input tokens
    """
    def __init__(self, dim, T=200, dropout=0.2, K=4):
        super().__init__()
        self.gf = SparseDGF(dim, T, dropout=dropout, K=K) # Dynamic global-aware filters with sparse regularization
#         self.gf = SparseGF(dim, T, dropout=dropout) # Global-aware filters with sparse regularization
#         self.gf = DGF(dim, T, K=K) # Dynamic global-aware filters
#         self.gf = GF(dim, T) # Global-aware filters
#         self.gf = MSA(num_attention_heads=K, input_size=dim, input_size=dim) # Multi-head self-attention
#         self.gf = LSTM(input_size=dim, hidden_size=dim, bias=False, bidirectional=False) # LSTM
        
        self.norm1 = nn.BatchNorm1d(dim)  
        self.norm2 = nn.BatchNorm1d(dim)  
        self.norm3 = nn.BatchNorm1d(dim)  
        self.proj1 = nn.Conv1d(dim, dim, kernel_size=1)  
        self.proj2 = nn.Conv1d(dim, dim, kernel_size=1)  
        self.act   = nn.ReLU()

    def forward_(self, x):
        skip = x
        
        x = self.proj1(x)
        x = self.act(x)
        x = self.norm1(x)
        
        x = self.gf.forward_(x) 
        x = self.act(x)    
        x = self.norm2(x)
        
        x = self.proj2(x)
        x = self.act(x) 
        x = self.norm3(x) 
        
        x = skip + x
        
        return x    
    
    def forward(self, x):
        skip = x
        
        x = self.proj1(x)
        x = self.act(x)
        x = self.norm1(x) 
        
        x = self.gf(x) 
        x = self.act(x)      
        x = self.norm2(x)
        
        x = self.proj2(x)
        x = self.act(x) 
        x = self.norm3(x) 
        
        x = skip + x
        
        return x    
  
     
    
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
                 
                    
                    
class DS_TDNN(nn.Module):
    def __init__(self, C, uniform_init=True, if_sparse=True):
        super(DS_TDNN, self).__init__()
        self.sparse=True
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )

        self.specaug = FbankAug() # Spec augmentation

        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.gelu   = nn.GELU()
        self.bn1    = nn.BatchNorm1d(C)
        
        # local branch
        self.llayer1 = LocalBlock(C//2, kernel_size=3, scale=4, dilation=1) #默认8 8 8 C=1024; 尝试4 6 8 C=960
        self.llayer2 = LocalBlock(C//2, kernel_size=3, scale=4, dilation=1)
        self.llayer3 = LocalBlock(C//2, kernel_size=3, scale=8, dilation=1)
         
        #global branch
        self.glayer1 = GlobalBlock(C//2, T=200, dropout=0.3,  K=4)
        self.glayer2 = GlobalBlock(C//2, T=200, dropout=0.1,  K=8)
        self.glayer3 = GlobalBlock(C//2, T=200, dropout=0.1,  K=8)
        
        
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        # ASP
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)
        
        self.uniform_init = uniform_init

    
    def forward_(self, x, aug=None):    
        '''
         Sparse forward
        '''
        assert self.sparse==True       
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        
        lx, gx = torch.chunk(x, 2, dim=1)
        
        #Dual branch:
        lx1 = self.llayer1(lx)
        gx1 = self.glayer1.forward_(gx)
                
        lx2 = self.llayer2(0.8*lx1+0.2*gx1)
        gx2 = self.glayer2.forward_(0.8*gx1+0.2*lx1)
        
        lx3 = self.llayer3(0.8*lx2+0.2*gx2)
        gx3 = self.glayer3.forward_(0.8*gx2+0.2*lx2)    
        
        x = self.layer4(torch.cat((lx1,gx1, lx2,gx2, lx3,gx3),dim=1))
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
    
    
    
    def forward(self, x, aug=None):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        
        lx, gx = torch.chunk(x, 2, dim=1)
        
        #Dual branch:
        lx1 = self.llayer1(lx)
        gx1 = self.glayer1(gx)
                
        lx2 = self.llayer2(0.8*lx1+0.2*gx1)
        gx2 = self.glayer2(0.8*gx1+0.2*lx1)
        
        lx3 = self.llayer3(0.8*lx2+0.2*gx2)
        gx3 = self.glayer3(0.8*gx2+0.2*lx2)   

        x = self.layer4(torch.cat((lx1,gx1, lx2,gx2, lx3,gx3),dim=1))
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

      
    def hook(self, x, aug=None):
        '''
         hook for different-scale feature maps
        '''
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        
        stem_o = x
        
        lx, gx = torch.chunk(x, 2, dim=1)
        
        #Dual branch:
        lx1 = self.llayer1(lx)
        gx1 = self.glayer1(gx)
                
        lx2 = self.llayer2(0.8*lx1+0.2*gx1)
        gx2 = self.glayer2(0.8*gx1+0.2*lx1)
        
        lx3 = self.llayer3(0.8*lx2+0.2*gx2)
        gx3 = self.glayer3(0.8*gx2+0.2*lx2)        

        x = self.layer4(torch.cat((lx1,gx1, lx2,gx2, lx3,gx3),dim=1))
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

        return x, stem_o, [lx, lx1, lx2, lx3], [gx, gx1, gx2, gx3], [lx1+gx1, lx2+gx2]
    
    
    def _init_weights(self, m):
        if not self.uniform_init:
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)     
                    
                    
                    
                