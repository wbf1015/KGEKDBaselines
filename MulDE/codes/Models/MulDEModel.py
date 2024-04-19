import sys
import os

import torch.nn.functional as F
import torch
import torch.nn as nn

class MulDEModel(nn.Module):
    def __init__(self, KGE=None, embedding_manager=None, muldeloss=None, args=None):
        super(MulDEModel, self).__init__()
        self.KGE = KGE
        self.EmbeddingManager = embedding_manager
        self.MuldeLoss = muldeloss
        self.args = args
        self.K = self.args.k
        self.Wrel = nn.Parameter(torch.empty(self.args.nrelation, self.args.teacher_num), requires_grad=True)
        nn.init.xavier_uniform_(self.Wrel)

    def get_postive_score(self, score):
        return score[:, 0]


    def get_negative_score(self, score):
        return score[:, 1:]

    def get_wrel(self, sample):
        positive, _ = sample
        
        wrel = torch.index_select(
            self.Wrel, 
            dim=0, 
            index=positive[:, 1]
        ).view(positive.shape[0], self.args.teacher_num)
        
        return wrel

    def forward(self, data, subsampling_weight, mode, epoch):
        # 计算hard loss
        head, relation, tail = self.EmbeddingManager.get_student_embedding(data, mode)
        score = self.KGE(head, relation, tail, mode)
        hard_loss, hard_loss_record = self.MuldeLoss.sigmoid_loss_origin(self.get_postive_score(score), self.get_negative_score(score), subsampling_weight=subsampling_weight, sub_margin=False, prefix='student')
        
        # 挑选用于计算soft loss的样本
        head, relation, tail = self.EmbeddingManager.get_ER_Query(data, mode)
        score = self.KGE(head, relation, tail, mode)
        s_score, indices = score.topk(self.K, dim=1)

        t_score = self.EmbeddingManager.get_teacher_score(data, indices, mode, self.KGE)
        
        loss, loss_record = self.MuldeLoss(t_score, s_score, self.get_wrel(data), epoch, data, indices, mode, hard_loss, hard_loss_record, subsampling_weight=subsampling_weight, sub_margin=True)
        
        return loss, loss_record
    
    def predict(self, data, mode):
        head, relation, tail = self.EmbeddingManager.get_student_embedding(data, mode)            
        score = self.KGE(head, relation, tail, mode)
        return score
        
        