#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from modules import *



class SEResNet(nn.Module):
    '''
     block: 'basic' or 'bottleneck'
     Example:
      num_filters = [64, 128, 256, 512]
      model = SEResNet('bottleneck', [3, 4, 6, 3], num_filters, 512) # SEResNet50_c64
    '''
    def __init__(self, block, layers, num_filters, nOut, encoder_type='ASP', n_mels=80, log_input=True, **kwargs):
        super(SEResNet, self).__init__()

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))
        
        self.inplanes     = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels       = n_mels
        self.log_input    = log_input

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=7, stride=2, padding=3)
        self.relu  = nn.ReLU(inplace=True)
        self.bn1   = nn.BatchNorm2d(num_filters[0])
        
        if block=='basic':
            self.block  = SEBasicBlock
            attn_cl     = 5*num_filters[3]
        else:    
            self.block  = SEBottleneck  
            attn_cl     = 20*num_filters[3]
            
        self.layer1 = self._make_layer(self.block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(self.block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(self.block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(self.block, num_filters[3], layers[3], stride=(2, 2))

        self.instancenorm = nn.InstanceNorm1d(n_mels)
        self.torchfbank   = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )

        
        self.attention = nn.Sequential(
            nn.Conv1d(attn_cl, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, attn_cl, kernel_size=1),
            nn.Softmax(dim=2),
            )

        if self.encoder_type == "SAP":
            out_dim =  int(attn_cl/2)
        elif self.encoder_type == "ASP":
            out_dim =  attn_cl
        else:
            raise ValueError('Undefined encoder')

  
        self.fc = nn.Linear(2*out_dim, nOut)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x, aug=None):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)
                
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # [B, C, 80/32, W/32]

        print(x.size())
        x = x.reshape(x.size()[0],-1,x.size()[-1]) # [B, -1, W/32]

        w = self.attention(x)

        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-5) )
            x = torch.cat((mu,sg),1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x



