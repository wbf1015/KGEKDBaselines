import sys
import os
import logging
import torch
import math
import torch.nn as nn
import numpy as np

CODEPATH = os.path.abspath(os.path.dirname(__file__))
CODEPATH = CODEPATH.rsplit('/', 1)[0]
sys.path.append(CODEPATH)

from Transformers.PoswiseFeedForwardNet import *
from Transformers.ScaleDotAttention import *
from Transformers.SelfAttention import *

class BasicStageAttention(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, d_ff=1, LN=False):
        super(BasicStageAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.d_ff = d_ff                      
        self.LN = LN        
                
        self.local_extraction= SelfAttention2(input_dim//2, output_dim, input_dim, n_heads, residual='V')
        self.forgetgate = nn.Linear(output_dim+input_dim, output_dim)
        if LN is False:
            self.fc = PoswiseFeedForwardNet2(output_dim, d_ff=d_ff)
        else:
            self.fc = PoswiseFeedForwardNet(output_dim, d_ff=d_ff)
        self.sigmoid = nn.Sigmoid()
        
        logging.info(f'Init BasicStageAttention Pruner with input_dim={self.input_dim}, output_dim={self.output_dim}, n_heads={self.n_heads}, d_ff={self.d_ff}, local_extraction={self.local_extraction.__class__.__name__}, forgetgate=nn.Linear, fc={self.fc.__class__.__name__}, sigmoid=nn.Sigmoid')
    
        
    def forward(self, enc_inputs, forget):
        local_inputs1 = enc_inputs[:, :, :self.input_dim//2]
        local_inputs2 = enc_inputs[:, :, self.input_dim//2:]
        outputs = self.local_extraction(local_inputs1, local_inputs2, forget)
        outputs = self.fc(outputs)
        
        forget_ = self.forgetgate(torch.cat((outputs, forget), dim=-1)) # 可以探索一下是使用过fc的outputs还是不过fc的outputs
        forget_ = outputs * self.sigmoid(forget_) # 加还是*都可以试试 StageAttention1/2
        # # forget_ = outputs + forget_
        # forget_ = outputs + outputs * self.sigmoid(forget_)
        
        return outputs, forget_


class NLPStageAttention(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, d_ff=1, LN=False):
        super(NLPStageAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.d_ff = d_ff                      
        self.LN = LN        
                
        self.local_extraction= SelfAttention3(input_dim, output_dim, input_dim, n_heads, residual='V')
        self.forgetgate = nn.Linear(output_dim+input_dim, output_dim)
        if LN is False:
            self.fc = PoswiseFeedForwardNet2(output_dim, d_ff=d_ff)
        else:
            self.fc = PoswiseFeedForwardNet(output_dim, d_ff=d_ff)
        self.sigmoid = nn.Sigmoid()
        
        logging.info(f'Init NLPAttention Pruner with input_dim={self.input_dim}, output_dim={self.output_dim}, n_heads={self.n_heads}, d_ff={self.d_ff}, local_extraction={self.local_extraction.__class__.__name__}, forgetgate=nn.Linear, fc={self.fc.__class__.__name__}, sigmoid=nn.Sigmoid')
    
        
    def forward(self, enc_inputs, forget):
        outputs = self.local_extraction(enc_inputs, enc_inputs, forget)
        outputs = self.fc(outputs)
        
        forget_ = self.forgetgate(torch.cat((outputs, forget), dim=-1)) # 可以探索一下是使用过fc的outputs还是不过fc的outputs
        forget_ = outputs * self.sigmoid(forget_) # 加还是*都可以试试 StageAttention1/2
        # # forget_ = outputs + forget_
        # forget_ = outputs + outputs * self.sigmoid(forget_)
        
        return outputs, forget_


class BasicSemAttention(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, d_ff=1, LN=False):
        super(BasicSemAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.d_ff = d_ff       
        self.LN = LN                
        
        self.sem_transform= SelfAttention2(input_dim, output_dim, input_dim, n_heads, residual='V')
        self.forgetgate = nn.Linear(output_dim+input_dim, output_dim)
        if LN is False:
            self.fc = PoswiseFeedForwardNet2(output_dim, d_ff=d_ff)
        else:
            self.fc = PoswiseFeedForwardNet(output_dim, d_ff=d_ff)
        self.sigmoid = nn.Sigmoid()
        
        logging.info(f'Init BasicSemAttention Pruner with input_dim={self.input_dim}, output_dim={self.output_dim}, n_heads={self.n_heads}, d_ff={self.d_ff}, sem_transform={self.sem_transform.__class__.__name__}, forgetgate=nn.Linear, fc={self.fc.__class__.__name__}, sigmoid=nn.Sigmoid')
    
    def forward(self, enc_inputs, forget):
        outputs = self.sem_transform(enc_inputs, enc_inputs, forget)
        outputs = self.fc(outputs)
        
        forget_ = self.forgetgate(torch.cat((outputs, forget), dim=-1)) # 可以探索一下是使用过fc的outputs还是不过fc的outputs
        forget_ = outputs * self.sigmoid(forget_)
        
        return outputs, forget_

'''
前一半后一半用来从几何特征提取语义特征
然后都使用self-attention
'''
class SemAttention(nn.Module):
    def __init__(self, args):
        super(SemAttention, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention(1024, 256, args.head1, d_ff=args.t_dff)
        self.layer2 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        
        return outputs


'''
在中间的语义变换期间加上LayerNorm
'''
class SemAttentionLN(nn.Module):
    def __init__(self, args):
        super(SemAttentionLN, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention(1024, 256, args.head1, d_ff=args.t_dff, LN=True)
        self.layer2 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        
        return outputs


'''
语义信息的维度递减
'''
class SemAttention2(nn.Module):
    def __init__(self, args):
        super(SemAttention2, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention(1024, 256, args.head1, d_ff=args.t_dff)
        self.layer2 = BasicSemAttention(256, 128, args.head2, d_ff=args.t_dff)
        self.layer3 = BasicSemAttention(128, 64, args.head3, d_ff=args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        outputs, forget = self.layer3(outputs, forget)
        
        return outputs


class SemAttention2LN(nn.Module):
    def __init__(self, args):
        super(SemAttention2LN, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention(1024, 256, args.head1, d_ff=args.t_dff, LN=True)
        self.layer2 = BasicSemAttention(256, 128, args.head2, d_ff=args.t_dff, LN=True)
        self.layer3 = BasicSemAttention(128, 64, args.head3, d_ff=args.t_dff, LN=False)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        outputs, forget = self.layer3(outputs, forget)
        
        return outputs

'''
语义信息的维度不变进行变换
'''
class SemAttention3(nn.Module):
    def __init__(self, args):
        super(SemAttention3, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention(1024, 256, args.head1, d_ff=args.t_dff)
        self.layer2 = BasicSemAttention(256, 256, args.head1, d_ff=args.t_dff)
        self.layer3 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        outputs, forget = self.layer3(outputs, forget)
        
        return outputs

class SemAttention3LN(nn.Module):
    def __init__(self, args):
        super(SemAttention3LN, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention(1024, 256, args.head1, d_ff=args.t_dff, LN=True)
        self.layer2 = BasicSemAttention(256, 256, args.head1, d_ff=args.t_dff, LN=True)
        self.layer3 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff, LN=False)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        outputs, forget = self.layer3(outputs, forget)
        
        return outputs


class SemAttention3TransELN(nn.Module):
    def __init__(self, args):
        super(SemAttention3TransELN, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention(512, 128, args.head1, d_ff=args.t_dff, LN=True)
        self.layer2 = BasicSemAttention(128, 128, args.head1, d_ff=args.t_dff, LN=True)
        self.layer3 = BasicSemAttention(128, 32, args.head2, d_ff=args.t_dff, LN=False)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        outputs, forget = self.layer3(outputs, forget)
        
        return outputs


'''
把语义信息提取模块变为NLPStageAttention
'''
class SemAttention4(nn.Module):
    def __init__(self, args):
        super(SemAttention4, self).__init__()
        self.args = args
        self.layer1 = NLPStageAttention(1024, 256, args.head1, d_ff=args.t_dff)
        self.layer2 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        
        return outputs


class SemAttention4LN(nn.Module):
    def __init__(self, args):
        super(SemAttention4LN, self).__init__()
        self.args = args
        self.layer1 = NLPStageAttention(1024, 256, args.head1, d_ff=args.t_dff, LN=True)
        self.layer2 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff, LN=False)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        
        return outputs


'''
加强版SemAttention4，对标SemAttention3，提高计算量，让区别仅存在于语义信息提取
'''
class SemAttention5(nn.Module):
    def __init__(self, args):
        super(SemAttention5, self).__init__()
        self.args = args
        self.layer1 = NLPStageAttention(1024, 256, args.head1, d_ff=args.t_dff)
        self.layer2 = BasicSemAttention(256, 256, args.head1, d_ff=args.t_dff)
        self.layer3 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        outputs, forget = self.layer3(outputs, forget)
        
        return outputs


class SemAttention5LN(nn.Module):
    def __init__(self, args):
        super(SemAttention5LN, self).__init__()
        self.args = args
        self.layer1 = NLPStageAttention(1024, 256, args.head1, d_ff=args.t_dff, LN=True)
        self.layer2 = BasicSemAttention(256, 256, args.head1, d_ff=args.t_dff, LN=True)
        self.layer3 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff, LN=False)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        outputs, forget = self.layer3(outputs, forget)
        
        return outputs