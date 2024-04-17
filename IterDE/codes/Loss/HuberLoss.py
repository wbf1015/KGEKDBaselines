import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from cmath import inf, nan

class HuberLoss(nn.Module):
    def __init__(self, l):
        super(HuberLoss, self).__init__()
        self.l = l # 软标签占比
        
    def norm(self, x):
        if x < 1.0:
            while x < 1.0:
                x *= 10.0
        elif x >= 10.0:
            while x >= 10.0:
                x /= 10.0
        return x

    def forward(self, t_score, s_score, loss_res, half_epoch):
        t_loss_res = F.smooth_l1_loss(s_score, t_score)
        if t_loss_res == nan or t_loss_res == inf:
            t_loss_res = 0
        if half_epoch == False:
            loss_cmp = t_loss_res * loss_res
            loss_cmp = self.norm(loss_cmp)
        else:
            loss_cmp = 1
        soft_loss = (self.l / loss_cmp) * t_loss_res
        loss = soft_loss + loss_res

        loss_record = {
            'soft_loss': soft_loss.item(),
            'total_loss': loss.item()
        }
        
        return loss, loss_record

    def predict(self, t_score, s_score, loss_res):
        score = self.forward(t_score, s_score, loss_res, True)
        return score.cpu().data.numpy()