import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from cmath import inf, nan


class MulDELoss(nn.Module):
    def __init__(self,adv_temperature = None, margin = None, l=0.0):
        super(MulDELoss, self).__init__()
        self.l = l # 软标签的权重
        self.adv_temperature = adv_temperature
        self.margin = margin
    

    def get_postive_score(self, score):
        return score[:, 0]

    def get_negative_score(self, score):
        return score[:, 1:]
    

    def sigmoid_loss_origin(self, p_score, n_score, subsampling_weight=None, sub_margin=False, prefix=''):
        if sub_margin:
            p_score, n_score = self.margin-p_score, self.margin-n_score
        if self.adv_temperature is not None:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(n_score * self.adv_temperature, dim = 1).detach() * ((F.logsigmoid(-n_score)))).sum(dim = 1)
        else:
            negative_score = (F.logsigmoid(-n_score)).mean(dim = 1)
        
        positive_score = F.logsigmoid(p_score)

        if subsampling_weight!=None:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()
                
        # 到这里就是1*1的了
        loss = (positive_sample_loss + negative_sample_loss)/2
        
        loss_record = {
            prefix+'_hard_positive_loss': positive_sample_loss.item(),
            prefix+'_hard_negative_loss': negative_sample_loss.item(),
            prefix+'_hard_loss': loss.item(),
        }
        return loss, loss_record
    
    
    def get_senior_loss(self, Sti, sample, indicies, mode):
        teacher_num = Sti.shape[0]
        k = indicies.shape[-1]
        Sti = Sti.transpose(0,1)
        # Sti:[batch, teacher_num, k], positive_index:[batch], indicies:[batch, k]
        if mode == 'head-batch':
            positive, negative = sample
            positive_index = positive[:, 0]        
        elif mode == 'tail-batch':
            positive, negative = sample
            positive_index =  positive[:, 2]
        matches = indicies==((positive_index.unsqueeze(-1)).expand(-1, k))
        matches_ = matches.unsqueeze(1).expand(-1, teacher_num, -1)
        result = torch.where(
            matches_,
            F.logsigmoid(Sti),
            torch.log(1 - torch.sigmoid(Sti))
        )
        # 计算最终结果的平均值
        LS = result.mean()
        return -LS
        
        
    def forward(self, t_score, s_score, W_rel, epoch, sample, indices, mode, hardloss, hardloss_record, subsampling_weight=None, sub_margin=False):# s_score:[batch, k]; t_score:[teacher_num, batch, k]; W_rel:[batch, teacher_num]
        if sub_margin:
            t_score, s_score = self.margin-t_score, self.margin-s_score
        t_score = t_score * torch.sigmoid((W_rel.transpose(0,1)).unsqueeze(-1))
        s_score_ = (s_score.unsqueeze(0)).repeat(4,1,1)
        
        pi = F.kl_div(F.log_softmax(s_score_, dim=-1), F.softmax(t_score, dim=-1), reduction='none').mean(-1)
        Ltop = (t_score * (torch.sigmoid(-pi/torch.exp(torch.tensor(epoch/5)))).unsqueeze(-1)*W_rel.shape[-1]).sum(dim=0)
        
        Lsoft = F.kl_div(F.log_softmax(s_score, dim=-1), F.softmax(Ltop, dim=-1), reduction='none').mean(-1)
        if subsampling_weight is not None:
            Lsoft = Lsoft.mean()
        else:
            Lsoft = (subsampling_weight*Lsoft).sum()/subsampling_weight.sum()
        
        LS = self.get_senior_loss(t_score, sample, indices, mode)
        
        LJ = (self.l*Lsoft) + (1-self.l)*hardloss
        loss = LJ + LS
        hardloss_record.update({
            'Student soft loss': Lsoft.item(),
            'Junior loss': LJ.item(),
            'total_loss' : loss.item(),
        })
        return loss, hardloss_record
        
            

    def predict(self, t_score, s_score, W_rel, epoch, sample, indices, mode, hardloss, hardloss_record, subsampling_weight=None, sub_margin=False):
        score = self.forward(t_score, s_score, W_rel, epoch, sample, indices, mode, hardloss, hardloss_record, subsampling_weight=subsampling_weight, sub_margin=sub_margin)
        return score.cpu().data.numpy()


if __name__ == '__main__':
    # 假设有一个一维向量
    batch = torch.tensor([1, 2, 3])

    # 设定扩展的大小 k
    k = 7

    # 使用 unsqueeze 增加一个新的维度，并使用 expand 扩展这个向量
    expanded_vector = batch.unsqueeze(1).expand(-1, k)

    # 打印结果
    print("Original vector:", batch)
    print("Expanded vector:\n", expanded_vector)