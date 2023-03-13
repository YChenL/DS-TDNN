from functools import partial
import torch, torchaudio, math
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import torch.fft


def get_dwconv1d(dim, kernel, bias):
    return nn.Conv1d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)


class Sparse(nn.Module):
    def __init__(self, sparse=0.1):
        super().__init__()
        assert 0 <= sparse <= 1
        self.sparse = sparse
        
    def forward(self, x): # x=[B, C, T//2+1]
        if self.sparse == 1:
            return torch.zeros_like(x).cuda()
    
        elif self.sparse == 0 :
            return x
        
        else:
            mask   = (torch.rand(x.shape[:-1]) > self.sparse).float().cuda()
            mask_f = mask.unsqueeze(-1).expand(x.shape)
            return mask_f*x, (1.0-mask_f)*(x.abs().mean())
            # mask_f*x/(1-self.sparse), (1.0-mask_f)*self.sparse

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

    
    
class attention1d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention1d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv1d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv1d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)
    
    
    
        
class AdapativeGlobalLocalFilter1d(nn.Module):
    def __init__(self, dim, T=200, ratios=0.25, K=8, temperature=34):
        super().__init__()
        self.K = K

        self.complex_weight = (nn.Parameter(torch.randn(K, dim, (T//2)+1, 2, dtype=torch.float32) * 0.02))
        trunc_normal_(self.complex_weight, std=.02)
        
        self.fn = attention1d(dim, ratios, K, temperature)
        
        self.pre_norm = nn.BatchNorm1d(dim)
        self.post_norm = nn.BatchNorm1d(dim)
        
    def forward(self, x):
        B, C, T = x.shape
        x = self.pre_norm(x)

        weight = self.complex_weight # [K, C, T//2, 2]
        attn = self.fn(x) #[B, K]
        
        x = torch.fft.rfft(x, dim=-1, norm='ortho')
        if not weight.shape[2] == x.shape[-1]:
            # [2, K, C, T//2]
            weight = F.interpolate(weight.permute(3,0,1,2), size=x.shape[1:3], mode='bilinear', align_corners=True).permute(1,2,3,0).contiguous()
        
        #动态权重的生成   
        aggregate_weight = torch.mm(attn, weight.reshape(self.K, -1)).reshape(B, C, -1, 2) 
        aggregate_weight = torch.view_as_complex(aggregate_weight) # 把weight转成复数
        
        x = x * aggregate_weight
        x = torch.fft.irfft(x, n=T, dim=-1, norm='ortho') # s确定输出信号形状
        x = self.post_norm(x)
        return x    
         
        

        
class SparseDGF(nn.Module):
    def __init__(self, dim, T=200, ratios=0.25, K=4, temperature=34, dropout=0.2):
        super().__init__()
        self.K = K

        self.complex_weight = (nn.Parameter(torch.randn(K, dim, (T//2)+1, 2, dtype=torch.float32) * 0.02))
        trunc_normal_(self.complex_weight, std=.02)
        
        self.fn = attention1d(dim, ratios, K, temperature)

        self.sparse = Sparse(sparse=dropout)
        
    def forward_(self, x):
        B, C, T = x.shape

        weight = self.complex_weight # [K, C, T//2, 2]
        attn = self.fn(x) #[B, K]
        
        x = torch.fft.rfft(x, dim=-1, norm='ortho')
        if not weight.shape[2] == x.shape[-1]:
            # [2, K, C, T//2]
            weight = F.interpolate(weight.permute(3,0,1,2), size=x.shape[1:3], mode='bilinear', align_corners=True).permute(1,2,3,0).contiguous()
        
        #动态权重的生成   
        agg_weight = torch.mm(attn, weight.reshape(self.K, -1)).reshape(B, C, -1, 2) 
        agg_weight = torch.view_as_complex(agg_weight) # 把weight转成复数
        
        sparse_w, mask = self.sparse(agg_weight)
        x = (x*mask)+(x*sparse_w)
        x = torch.fft.irfft(x, n=T, dim=-1, norm='ortho') # s确定输出信号形状
 
        return x   
    
    def forward(self, x):
        B, C, T = x.shape

        weight = self.complex_weight # [K, C, T//2, 2]
        attn = self.fn(x) #[B, K]
        
        x = torch.fft.rfft(x, dim=-1, norm='ortho')
        if not weight.shape[2] == x.shape[-1]:
            # [2, K, C, T//2]
            weight = F.interpolate(weight.permute(3,0,1,2), size=x.shape[1:3], mode='bilinear', align_corners=True).permute(1,2,3,0).contiguous()
        
        #动态权重的生成   
        agg_weight = torch.mm(attn, weight.reshape(self.K, -1)).reshape(B, C, -1, 2) 
        agg_weight = torch.view_as_complex(agg_weight) # 把weight转成复数
        
        x = x * agg_weight
        x = torch.fft.irfft(x, n=T, dim=-1, norm='ortho') # s确定输出信号形状

        return x  
    
    

class SparseGF(nn.Module):
    def __init__(self, dim, T=200, dropout=0.2, K=1):
        super().__init__()
        self.K = K

        self.complex_weight = (nn.Parameter(torch.randn(dim, (T//2)+1, 2, dtype=torch.float32) * 0.02))
        trunc_normal_(self.complex_weight, std=.02)

        self.sparse = Sparse(sparse=dropout)
        
    def forward_(self, x):
        B, C, T = x.shape

        x = torch.fft.rfft(x, dim=-1, norm='ortho')
        weight = self.complex_weight # [K, C, T//2, 2]
        if not weight.shape[2] == x.shape[-1]:
            weight = F.interpolate(weight.permute(2,0,1), size=x.shape[-1], mode='linear', align_corners=True).permute(1,2,0).contiguous()
            
        weight = torch.view_as_complex(weight) # 把weight转成复数
        sparse_w, mask = self.sparse(weight)
        x = (x*mask)+(x*sparse_w)
        x = torch.fft.irfft(x, n=T, dim=-1, norm='ortho') # s确定输出信号形状

        return x   
    
    def forward(self, x):
        B, C, T = x.shape
    
        x = torch.fft.rfft(x, dim=-1, norm='ortho')
        weight = self.complex_weight # [K, C, T//2, 2]
        if not weight.shape[2] == x.shape[-1]:
            weight = F.interpolate(weight.permute(2,0,1), size=x.shape[-1], mode='linear', align_corners=True).permute(1,2,0).contiguous()
        
        weight = torch.view_as_complex(weight) # 把weight转成复数
        x = x * weight
        x = torch.fft.irfft(x, n=T, dim=-1, norm='ortho') # s确定输出信号形状

        return x    
    
    
    
class DWGlobalLocalFilter1d(nn.Module):
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
    
  

class GlobalFilter1d(nn.Module):
    def __init__(self, dim, T=200, dropout=0, K=1):
        super().__init__()
        self.complex_weight = (nn.Parameter(torch.randn(dim, (T//2)+1, 2, dtype=torch.float32) * 0.02))
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = nn.BatchNorm1d(dim)
        self.post_norm = nn.BatchNorm1d(dim)
        
    def forward(self, x):
        B, C, T = x.shape
        x = self.pre_norm(x)
        x = torch.fft.rfft(x, dim=-1, norm='ortho')
        weight = self.complex_weight # [K, C, T//2, 2]
        if not weight.shape[2] == x.shape[-1]:
            weight = F.interpolate(weight.permute(2,0,1), size=x.shape[-1], mode='linear', align_corners=True).permute(1,2,0).contiguous()
        
        weight = torch.view_as_complex(weight) # 把weight转成复数
        x = x * weight
        x = torch.fft.irfft(x, n=T, dim=-1, norm='ortho') # s确定输出信号形状
        x = self.post_norm(x)
        return x    
    

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
            [nn.Conv1d(self.dims[i], self.dims[i+1], 1) for i in range(order-1)]
        )

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
    
    
class LayerNorm1d(nn.Module):
    r""" LayerNorm1d that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x
        
        
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



class LocalBlock(nn.Module):
    r""" HorNet block
    """
    def __init__(self, dim, kernel_size=3, dilation=1, scale=4, drop_path=0.):
        super().__init__()
      
        self.res2conv  = res2conv1d(dim, kernel_size, dilation, scale)     
           
        self.norm1   = nn.BatchNorm1d(dim)   
        self.pwconv1 = nn.Conv1d(dim, dim, kernel_size=1)  
        self.pwconv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.act = nn.ReLU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.se  = SEModule(dim)

    def forward(self, x):
        skip = x
        
        x = self.res2conv(self.norm1(x))
        x = self.pwconv1(x)
        x = self.act(x) 
        x = self.pwconv2(x)
        
        x = skip + self.se(x)
        
        return x    
    
    


class GlobalBlock(nn.Module):
    r""" HorNet block
    """
    def __init__(self, dim, T=200, dropout=0.2, K=4):
        super().__init__()
      
        self.gf  = SparseDGF(dim, T, dropout=dropout, K=K) # depthwise conv 
           
        self.norm1   = LayerNorm1d(dim, eps=1e-6, data_format='channels_first')
        self.norm2   = LayerNorm1d(dim, eps=1e-6)
             
        self.pwconv1 = nn.Linear(dim, dim)
        self.pwconv2 = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward_(self, x):
        skip = x
        
        x = self.gf.forward_(self.norm1(x)) 
        x = x.permute(0,2,1)
        x = self.norm2(x) 
        
        x = self.pwconv1(x)
        x = self.act(x) 
        x = self.pwconv2(x)
        x = x.permute(0,2,1)
        
        x = skip + x
        
        return x    
    
    def forward(self, x):
        skip = x
        
        x = self.gf(self.norm1(x))
        x = x.permute(0,2,1)
        x = self.norm2(x) 
        
        x = self.pwconv1(x) 
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0,2,1)
        
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
        self.llayer2 = LocalBlock(C//2, kernel_size=3, scale=4, dilation=2)
        self.llayer3 = LocalBlock(C//2, kernel_size=3, scale=4, dilation=3)
         
        #global branch
        self.glayer1 = GlobalBlock(C//2, T=200, dropout=0.4,  K=4)
        self.glayer2 = GlobalBlock(C//2, T=200, dropout=0.2,  K=4)
        self.glayer3 = GlobalBlock(C//2, T=200, dropout=0.1,  K=4)
        
        
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
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
        
        #loacl branch:
        lx1 = self.llayer1(lx)
        gx1 = self.glayer1.forward_(gx)
                
        lx2 = self.llayer2(lx1+gx1)
        gx2 = self.glayer2.forward_(gx1+lx1)
        
        lx3 = self.llayer3(lx2+gx2)
        gx3 = self.glayer3.forward_(gx2+lx2)        
        
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
        
        #loacl branch:
        lx1 = self.llayer1(lx)
        gx1 = self.glayer1(gx)
                
        lx2 = self.llayer2(lx1+gx1)
        gx2 = self.glayer2(gx1+lx1)
        
        lx3 = self.llayer3(lx2+gx2)
        gx3 = self.glayer3(gx2+lx2)        

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
                    
                    
                    
