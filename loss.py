'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
'''

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from tools import *


class AAMsoftmax(nn.Module):
    def __init__(self, n_class, m, s):
        
        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)
        self.ce = nn.CrossEntropyLoss()
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        
        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss, prec1
    
    
class SubCenterArcFace(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample, defualt = 192
            n_class: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, n_class, m, s, k):
        super(SubCenterArcFace, self).__init__()
        
        self.n_class = n_class
        self.m = m
        self.s = s
        self.k = k
        
        self.weight = nn.Parameter(torch.FloatTensor(n_class*k, 512), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_uniform_(self.weight, gain=1)
        
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        
    def forward(self, x, label=None):
        
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        cosine = torch.reshape(cosine, (-1, self.n_class, self.k))
        cosine = torch.max(cosine, 2)[0]
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = (cosine * self.cos_m - sine * self.sin_m)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        output *= self.s
        
        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss, prec1
    
    


