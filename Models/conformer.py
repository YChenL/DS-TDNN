import torch, torchaudio, math
import torch.nn as nn
import torch.nn.functional as F
# from wenet.transformer.encoder import ConformerEncoder
# from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
# from speechbrain.lobes.models.ECAPA_TDNN import BatchNorm1d
from modules import *


    
class Conformer(torch.nn.Module):
    def __init__(self, n_mels=80, num_blocks=6, output_size=256, embedding_dim=192, input_layer="conv2d2", 
            pos_enc_layer_type="rel_pos"):
        super(Conformer, self).__init__()
        self.conformer = ConformerEncoder(input_size=n_mels, num_blocks=num_blocks, 
                output_size=output_size, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type)
        
        # ASP
        self.attention = nn.Sequential(
            nn.Conv1d(output_size*3, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, output_size, kernel_size=1),
            nn.Softmax(dim=2),
            )
        
        self.bn = nn.BatchNorm1d(output_size*2)
        self.fc = torch.nn.Linear(output_size*2, embedding_dim)
        
        self.torchfbank = torch.nn.Sequential(PreEmphasis(),            
                                              torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, \
                                                                                   hop_length=160, f_min = 20, f_max = 7600, \
                                                                                   window_fn=torch.hamming_window, n_mels=80))
        self.specaug = FbankAug() # Spec augmentation
        
        
    def forward(self, feat, aug=None):  
        with torch.no_grad():
            feat = self.torchfbank(feat)+1e-6
            feat = feat.log()   
            feat = feat - torch.mean(feat, dim=-1, keepdim=True)
            
        feat = feat.squeeze(1).permute(0, 2, 1)
        lens = torch.ones(feat.shape[0]).to(feat.device)
        lens = torch.round(lens*feat.shape[1]).int()
        x, masks = self.conformer(feat, lens)
        
        x = x.permute(0, 2, 1)      
        # ASP
        t = x.size()[-1]

        global_x = torch.cat((x, torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), \
                              torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )
        x = torch.cat((mu,sg),1)
        # x = x.permute(0, 2, 1) 
        ####
        
        x = self.bn(x)
        x = self.fc(x)
        x = x.squeeze(1)
        return x

    
    
def conformer(n_mels=80, num_blocks=6, output_size=280, 
              embedding_dim=192, input_layer="conv2d", pos_enc_layer_type="rel_pos"):
    model = Conformer(n_mels=n_mels, num_blocks=num_blocks, output_size=output_size, 
            embedding_dim=embedding_dim, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type)
    return model



