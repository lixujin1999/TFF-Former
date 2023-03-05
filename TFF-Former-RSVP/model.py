from __future__ import division
from __future__ import print_function
from config import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import os
import glob
import time
import math
import random
import argparse
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.autograd import Function
import collections
import random
from tqdm import tqdm
from torch import nn
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from metric import GELU


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, max_len, d_model=config.d_model, dropout=config.p):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],  requires_grad=False)
        return self.dropout(x)

class PatchEmbedding(nn.Module):
    def __init__(self, device, in_channels: int = 1, patch_sizeh: int = config.patchsize, patch_sizew: int = config.patchsize, emb_size: int = config.d_model, img_size1: int = config.C, img_size2:int = config.T):
        super(PatchEmbedding,self).__init__()
        self.patch_sizeh = patch_sizeh
        self.patch_sizew = patch_sizew
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=(self.patch_sizeh,self.patch_sizew), stride=(self.patch_sizeh,self.patch_sizew),padding=(0,0)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
       
        self.positions = nn.Parameter(torch.randn(((img_size1*img_size2) // (self.patch_sizew*self.patch_sizeh)), emb_size))
        self.nonpara = PositionalEncoding(((img_size1*img_size2) // (self.patch_sizew*self.patch_sizeh))).to(device)

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        #x += self.positions
        x = self.nonpara(x)
        return x


class Mutihead_Attention(nn.Module):
    def __init__(self,device,d_model,dim_k,dim_v,n_heads):
        super(Mutihead_Attention, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.n_heads = n_heads
        self.device = device

        self.q = nn.Linear(d_model,dim_k)
        self.k = nn.Linear(d_model,dim_k)
        self.v = nn.Linear(d_model,dim_v)
        #self.v = self.k

        self.o = nn.Linear(dim_v,d_model)
        self.norm_fact = 1 / math.sqrt(d_model)

    def generate_mask(self,dim,score):
        thre = torch.mean(score,dim=-1).to(self.device) 
        thre = torch.unsqueeze(thre, 3)
        vec = torch.ones((1,dim)).to(self.device)
        thre = torch.matmul(thre,vec)  
        cha = score - thre
        one_vec = torch.ones_like(cha).to(self.device)
        zero_vec = torch.zeros_like(cha).to(self.device)
    
        mask = torch.where(cha > 0, zero_vec, one_vec).to(self.device)
        return mask==1

    def forward(self,x,y,requires_mask=True):
        assert self.dim_k % self.n_heads == 0 and self.dim_v % self.n_heads == 0
        # size of x : [batch_size * seq_len * batch_size]
  
        Q = self.q(x).reshape(-1,x.shape[0],x.shape[1],self.dim_k // self.n_heads) # n_heads * batch_size * seq_len * dim_k
        K = self.k(y).reshape(-1,y.shape[0],y.shape[1],self.dim_k // self.n_heads) # n_heads * batch_size * seq_len * dim_k
        V = self.v(y).reshape(-1,y.shape[0],y.shape[1],self.dim_v // self.n_heads) # n_heads * batch_size * seq_len * dim_v
        #print("Attention V shape : {}".format(V.shape))
        attention_score = torch.matmul(Q,K.permute(0,1,3,2)) * self.norm_fact
        if requires_mask:
            mask = self.generate_mask(attention_score.size()[3],attention_score)
            
            attention_score.masked_fill(mask,value=float("-inf")) 
        attention_score = F.softmax(attention_score,dim=-1)
        output = torch.matmul(attention_score,V).reshape(x.shape[0],x.shape[1],-1)
        # print("Attention output shape : {}".format(output.shape))

        output = self.o(output)
        return output

class Feed_Forward1(nn.Module):
    def __init__(self,device,input_dim,hidden_dim):
        super(Feed_Forward1, self).__init__()
        self.L1 = nn.Linear(input_dim,hidden_dim).to(device)
        self.L2 = nn.Linear(hidden_dim,input_dim).to(device)
        self.gelu = GELU().to(device)

    def forward(self,x):
        output = self.gelu((self.L1(x)))
        output = self.L2(output)
        return output

class Feed_Forward(nn.Module):  #output
    def __init__(self,device,input_dim=config.d_model,hidden_dim=config.hidden):
        super(Feed_Forward, self).__init__()
        F1 = 16
        
        self.conv1 = nn.Conv2d(1, F1, (15, 16), bias = False, stride = (15,16))  #Conv2d #F1*4*8
        self.dropout = nn.Dropout(config.p)
        self.gelu = GELU().to(device)

    def forward(self,x):  
        output = self.gelu(self.conv1(x.unsqueeze(1)))
        output = self.dropout(output)
        output = output.contiguous().view(-1, self.num_flat_features(output))
        return output
        
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features = num_features*s
        return num_features


class Add_Norm(nn.Module):
    def __init__(self,device):
        super(Add_Norm, self).__init__()
        self.dropout = nn.Dropout(config.p).to(device)
        self.device = device

    def forward(self,x,sub_layer,**kwargs):
        sub_output = sub_layer(x,**kwargs)
        x = self.dropout(x + sub_output)
        
        layer_norm = nn.LayerNorm(x.size()[1:]).to(self.device)
        out = layer_norm(x)
        return out


class Encoder(nn.Module):
    def __init__(self,device,dim_seq,dim_fea,n_heads,hidden):
        super(Encoder, self).__init__()
        self.dim_seq = dim_seq
        self.dim_fea = dim_fea
        self.n_heads = n_heads
        self.dim_k = self.dim_fea // self.n_heads
        self.dim_v = self.dim_k
        self.hidden = hidden

        self.muti_atten = Mutihead_Attention(device,self.dim_fea,self.dim_k,self.dim_v,self.n_heads).to(device)
        self.feed_forward = Feed_Forward1(device,self.dim_fea,self.hidden).to(device)
        self.add_norm = Add_Norm(device).to(device)

    def forward(self,x): 
        output = self.add_norm(x,self.muti_atten,y=x)
        output = self.add_norm(output,self.feed_forward)
        return output
        
class Encoder_last(nn.Module):
    def __init__(self,device,dim_seq,dim_fea,n_heads,hidden):
        super(Encoder_last, self).__init__()
        self.dim_seq = dim_seq
        self.dim_fea = dim_fea
        self.n_heads = n_heads
        self.dim_k = self.dim_fea // self.n_heads
        self.dim_v = self.dim_k
        self.hidden = hidden

        self.muti_atten = Mutihead_Attention(device,self.dim_fea,self.dim_k,self.dim_v,self.n_heads).to(device)
        self.feed_forward = Feed_Forward(device,self.dim_fea,self.hidden).to(device)
        self.add_norm = Add_Norm(device).to(device)

    def forward(self,x):
        output = self.add_norm(x,self.muti_atten,y=x)
        output = self.feed_forward(output)
        return output
    
class Decoder(nn.Module):
    def __init__(self,device,dim_seq,dim_fea,n_heads,hidden):
        super(Decoder, self).__init__()
        self.dim_seq = dim_seq 
        self.dim_fea = dim_fea
        self.n_heads = n_heads
        self.dim_k = self.dim_fea // self.n_heads
        self.dim_v = self.dim_k
        self.hidden = hidden
        
        self.muti_atten = Mutihead_Attention(device,self.dim_fea,self.dim_k,self.dim_v,self.n_heads).to(device)
        self.feed_forward = Feed_Forward1(device,self.dim_fea,self.hidden).to(device)
        self.add_norm = Add_Norm(device).to(device)

    def forward(self,q,v):
        
        
        output = self.add_norm(v,self.muti_atten,y=q,requires_mask=True)
        output = self.add_norm(output,self.feed_forward)
        output = output + v
        return output

class Cross_modal(nn.Module):
    def __init__(self,device):
        super(Cross_modal, self).__init__()
        self.cross1 = Decoder(device,config.H*config.W,config.d_model,config.n_heads,config.hidden).to(device)
        self.cross2 = Decoder(device,config.H*config.W,config.d_model,config.n_heads,config.hidden).to(device)
        self.fc1 = nn.Linear(2*config.d_model,config.d_model).to(device)

    def forward(self,target,f1):
        re = self.cross1(f1,target)
        return re

class Cross_modalto(nn.Module):
    def __init__(self,device,dim_seq=4*config.H*config.W,dim_fea=config.d_model, n_heads=4, hidden=config.hidden):
        super(Cross_modalto, self).__init__()
        self.dim_seq = dim_seq
        self.long = config.H*config.W
        self.dim_fea = dim_fea
        self.n_heads = n_heads
        self.dim_k = self.dim_fea // self.n_heads
        self.dim_v = self.dim_k
        self.hidden = hidden

        self.muti_atten = Mutihead_Attention(device,self.dim_fea,self.dim_k,self.dim_v,self.n_heads).to(device)
        self.feed_forward = Feed_Forward(device,self.dim_fea,self.hidden).to(device)
        self.add_norm = Add_Norm(device=device).to(device)

    def forward(self,q,v):
        
        output = self.add_norm(v,self.muti_atten,y=q,requires_mask=True)
        output = output + v
        output = self.feed_forward(output)
        return output

# In[78]:

class Transformer_layer(nn.Module):
    def __init__(self, device, dmodel=config.d_model, num_heads=config.n_heads, num_tokens=config.H*config.W):
        super(Transformer_layer, self).__init__()
        self.encoder = Encoder(device,num_tokens,dmodel,num_heads,config.hidden).to(device) 

    def forward(self,x):  
        encoder_output = self.encoder(x) + x
        return encoder_output
        
class Transformer_layer_last(nn.Module):
    def __init__(self,device):
        super(Transformer_layer_last, self).__init__()
        self.encoder = Encoder_last(device,config.H*config.W,config.d_model,config.n_heads,config.hidden).to(device) 

    def forward(self,x):  
        encoder_output = self.encoder(x)
        return encoder_output


# In[79]:

#share
class Transformer(nn.Module):
    def __init__(self,device,N,output_dim):
        super(Transformer, self).__init__()

        self.output_dim = output_dim
        #Patch Embedding
        self.embedding_t = PatchEmbedding(device=device,patch_sizeh=config.patchsizeth, patch_sizew=config.patchsizetw).to(device)
        self.embedding_f = PatchEmbedding(device=device,patch_sizeh=config.patchsizefh, patch_sizew=config.patchsizefw).to(device)
        self.norm = nn.LayerNorm(config.T).to(device)

        #Encoder
        self.model = nn.Sequential(*[Transformer_layer(device) for _ in range(N)]).to(device)

        self.model_last = Cross_modalto(device).to(device)
        
        #cross-modal
        self.t = Cross_modal(device).to(device)
        self.f = Cross_modal(device).to(device)
        self.t1 = Cross_modal(device).to(device)
        self.f1 = Cross_modal(device).to(device)
        self.fc1 = nn.Linear(config.d_model*2,config.d_model)
        self.fc = nn.Linear(16*4*8,output_dim) 
        
    def forward(self,raw,fre):

        x_t1 = self.embedding_t(raw)
        x_f1 = self.embedding_f(self.norm(fre))
        
        #temporal data encoder
        x_t = self.model(x_t1)
        #frequency data encoder
        x_f = self.model(x_f1)
        #cross-modal
        x_t2 = self.t(x_t,x_f)
        x_f2 = self.f(x_f,x_t)

        x_t1 = self.t(x_t2,x_f2)
        x_f1 = self.f(x_f2,x_t2)

        x_t2 = x_t + x_t1
        x_f2 = x_f + x_f1

        x = torch.cat((x_f2,x_t2),axis=1)
        output = self.model_last(x,self.fc1(torch.cat((x_f2,x_t2),axis=-1)))
        output = self.fc(output)
        return output, x_t2, x_f2