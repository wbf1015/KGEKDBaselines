import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from cmath import inf, nan

class LearnableSigmoid(nn.Module):
    def __init__(self, ):
        super(LearnableSigmoid, self).__init__()
        self.alpha_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.alpha_weight.data.fill_(1.0)
        self.beta_weight.data.fill_(0)
        
    def forward(self, inputs):
        return 1/(1 +  torch.exp(-self.alpha_weight*(inputs + self.beta_weight)))


class DualLoss(nn.Module):
    def __init__(self,adv_temperature = None, margin = None, l=0):
        super(DualLoss, self).__init__()
        self.l = l # 是否计算教师的Loss
        self.adv_temperature = adv_temperature
        self.margin = margin
        self.s_p_softweight = LearnableSigmoid()
        self.s_n_softweight = LearnableSigmoid()
        self.t_p_softweight = LearnableSigmoid()
        self.t_n_softweight = LearnableSigmoid()
        self.huberloss = nn.SmoothL1Loss(reduction='none')
    
    def setl(self, l):
        self.l=l
    
    def get_postive_score(self, score):
        return score[:, 0]

    def get_negative_score(self, score):
        return score[:, 1:]
    
    def huber_loss(self, t_score, s_score):
        huber_loss = self.huberloss(s_score, t_score)
        huber_loss = torch.where(torch.isnan(huber_loss) | torch.isinf(huber_loss), torch.zeros_like(huber_loss), huber_loss)
        
        return huber_loss

    def cal_angle(self, head, tail): # 输入是[batch, 1, dim] 和 [batch, neg_sampling, dim],谁先谁后都可以
        # 计算模长，并保持维度，方便后续的广播操作
        head_norm = head.norm(p=2, dim=-1, keepdim=True)
        tail_norm = tail.norm(p=2, dim=-1, keepdim=True)
        
        # 规范化向量
        normalized_head = head / head_norm
        normalized_tail = tail / tail_norm

        # 计算点乘
        # 维度变换为 [1024, 1, 512] @ [1024, 512, 256] -> [1024, 1, 256]
        angles = torch.matmul(normalized_head, normalized_tail.transpose(-1, -2))

        # 去掉多余的维度，得到 [1024, 256]
        angles = angles.squeeze()
        
        return angles
    
    def cal_lenrate(self, head, tail): # 输入是[batch, 1, dim] 和 [batch, neg_sampling, dim],谁先谁后都可以
        # 计算模长，结果保留原始维度
        head_norm = head.norm(p=2, dim=-1, keepdim=True)
        tail_norm = tail.norm(p=2, dim=-1, keepdim=True)

        # 去掉模长的最后一维
        head_norm = head_norm.squeeze(-1)
        tail_norm = tail_norm.squeeze(-1)

        # 执行广播除法得到最终结果
        lenrate = head_norm / tail_norm
        
        return lenrate

    def structure_loss(self, t_head, t_tail, s_head, s_tail):
        t_angle = self.cal_angle(t_head, t_tail)
        s_angle = self.cal_angle(s_head, s_tail)
        t_lenrate = self.cal_lenrate(t_head, t_tail)
        s_lenrate = self.cal_lenrate(s_head, s_tail)
        
        structure_loss_dict = {
            't_angle' : t_angle,
            's_angle' : s_angle,
            't_lenrate' : t_lenrate, 
            's_lenrate' : s_lenrate
        }
        
        return structure_loss_dict
        
    
    def sigmoid_loss_origin(self, p_score, n_score, p_soft_weight, n_soft_weight, subsampling_weight=None, sub_margin=False, prefix=''):
        if sub_margin:
            p_score, n_score = self.margin-p_score, self.margin-n_score
        if self.adv_temperature is not None:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(n_score * self.adv_temperature, dim = 1).detach() * ((F.logsigmoid(-n_score))*(n_soft_weight))).sum(dim = 1)
            # negative_score = (F.softmax(n_score * self.adv_temperature, dim = 1).detach() * ((F.logsigmoid(-n_score)))).sum(dim = 1)
        else:
            negative_score = ((1 - n_soft_weight) * F.logsigmoid(-n_score)).mean(dim = 1)
        
        positive_score = (p_soft_weight) * F.logsigmoid(p_score)
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
    
    
    def student_loss(self, t_score, s_score, structure_loss_dict, subsampling_weight=None):
        p_d_score = self.huber_loss(torch.sigmoid(self.get_postive_score(t_score)), torch.sigmoid(self.get_postive_score(s_score)))
        p_d_structure = self.huber_loss(self.get_postive_score(structure_loss_dict['t_angle']), self.get_postive_score(structure_loss_dict['s_angle'])) + \
                        self.huber_loss(self.get_postive_score(structure_loss_dict['t_lenrate']), self.get_postive_score(structure_loss_dict['s_lenrate']))
        p_d_soft = p_d_score + p_d_structure
        
        n_d_score = self.huber_loss(torch.sigmoid(self.get_negative_score(t_score)), torch.sigmoid(self.get_negative_score(s_score)))
        n_d_structure = self.huber_loss(self.get_negative_score(structure_loss_dict['t_angle']), self.get_negative_score(structure_loss_dict['s_angle'])) + \
                        self.huber_loss(self.get_negative_score(structure_loss_dict['t_lenrate']), self.get_negative_score(structure_loss_dict['s_lenrate']))
        n_d_soft = n_d_score + n_d_structure
        
        t_p_score = self.get_postive_score(t_score)
        t_n_score = self.get_negative_score(t_score)
        if subsampling_weight is None:
            soft_student_loss = (self.s_p_softweight(t_p_score)*p_d_soft).mean() + (self.s_n_softweight(t_n_score)*n_d_soft).mean()
        else:
            # print(subsampling_weight.shape, self.s_p_softweight(t_p_score).shape, p_d_soft.shape)
            # print(subsampling_weight.shape, self.s_p_softweight(t_n_score).shape, n_d_soft.shape)
            soft_student_loss = (subsampling_weight*self.s_p_softweight(t_p_score)*p_d_soft).sum()/subsampling_weight.sum() + (subsampling_weight.unsqueeze(-1)*self.s_n_softweight(t_n_score)*n_d_soft).sum()/subsampling_weight.sum()
        s_p_score = self.get_postive_score(s_score)
        s_n_score = self.get_negative_score(s_score)
        if self.margin is not None:
            hard_student_loss, loss_record = self.sigmoid_loss_origin(s_p_score, s_n_score, (1-self.s_p_softweight(t_p_score)), (1-self.s_n_softweight(t_n_score)), subsampling_weight=subsampling_weight, sub_margin=True, prefix='student')
        else:
            hard_student_loss, loss_record = self.sigmoid_loss_origin(s_p_score, s_n_score, (1-self.s_p_softweight(t_p_score)), (1-self.s_n_softweight(t_n_score)), subsampling_weight=subsampling_weight, sub_margin=False, prefix='student')
        
        student_loss = soft_student_loss + hard_student_loss
        
        loss_record.update({'student_soft_loss' : soft_student_loss.item()})
        loss_record.update({'student_total_loss' : student_loss.item()})
        
        return student_loss, loss_record
    
    def teacher_loss(self, t_score, s_score, structure_loss_dict, subsampling_weight=None):
        p_d_score = self.huber_loss(torch.sigmoid(self.get_postive_score(t_score)), torch.sigmoid(self.get_postive_score(s_score)))
        p_d_structure = self.huber_loss(self.get_postive_score(structure_loss_dict['t_angle']), self.get_postive_score(structure_loss_dict['s_angle'])) + \
                        self.huber_loss(self.get_postive_score(structure_loss_dict['t_lenrate']), self.get_postive_score(structure_loss_dict['s_lenrate']))
        p_d_soft = p_d_score + p_d_structure
        
        n_d_score = self.huber_loss(torch.sigmoid(self.get_negative_score(t_score)), torch.sigmoid(self.get_negative_score(s_score)))
        n_d_structure = self.huber_loss(self.get_negative_score(structure_loss_dict['t_angle']), self.get_negative_score(structure_loss_dict['s_angle'])) + \
                        self.huber_loss(self.get_negative_score(structure_loss_dict['t_lenrate']), self.get_negative_score(structure_loss_dict['s_lenrate']))
        n_d_soft = n_d_score + n_d_structure
        
        s_p_score = self.get_postive_score(s_score)
        s_n_score = self.get_negative_score(s_score)
        if subsampling_weight is None:
            soft_teacher_loss = (self.t_p_softweight(s_p_score)*p_d_soft).mean() + (self.t_n_softweight(s_n_score)*n_d_soft).mean()
        else:
            soft_teacher_loss = (subsampling_weight*self.t_p_softweight(s_p_score)*p_d_soft).sum()/subsampling_weight.sum() + (subsampling_weight.unsqueeze(-1)*self.t_n_softweight(s_n_score)*n_d_soft).sum()/subsampling_weight.sum()
        
        t_p_score = self.get_postive_score(t_score)
        t_n_score = self.get_negative_score(t_score)
        if self.margin is not None:
            hard_teacher_loss, loss_record = self.sigmoid_loss_origin(t_p_score, t_n_score, (1-self.t_p_softweight(s_p_score)), (1-self.t_n_softweight(s_n_score)), subsampling_weight=subsampling_weight, sub_margin=True, prefix='teacher')
        else:
            hard_teacher_loss, loss_record = self.sigmoid_loss_origin(t_p_score, t_n_score, (1-self.t_p_softweight(s_p_score)), (1-self.t_n_softweight(s_n_score)), subsampling_weight=subsampling_weight, sub_margin=False, prefix='teacher')
        
        teacher_loss = soft_teacher_loss + hard_teacher_loss
        loss_record.update({'teacher_soft_loss' : soft_teacher_loss.item()})
        loss_record.update({'teacher_total_loss' : teacher_loss.item()})
        
        return teacher_loss, loss_record
    
    
    def forward(self, t_score, s_score, structure_loss_dict, subsampling_weight=None):
        student_loss, student_loss_record =  self.student_loss(t_score, s_score, structure_loss_dict, subsampling_weight=subsampling_weight)
        if self.l != 0:
            teacher_loss, teacher_loss_record =  self.teacher_loss(t_score, s_score, structure_loss_dict, subsampling_weight=subsampling_weight)
            loss = student_loss + teacher_loss
            student_loss_record.update(teacher_loss_record)
            loss_record = student_loss_record
        else:
            loss, loss_record = student_loss, student_loss_record

        return loss, loss_record
            

    def predict(self, t_score, s_score, structure_loss_dict, subsampling_weight=None):
        score = self.forward(t_score, s_score, structure_loss_dict, subsampling_weight=subsampling)
        return score.cpu().data.numpy()


if __name__ == '__main__':
    # 模拟数据
    inputs = torch.rand((1024, 1))
    weighted = 1/(1 +  torch.exp(1*(inputs + 0.5)))
    
    print(weighted.shape)