import sys
import os

import torch.nn.functional as F
import torch
import torch.nn as nn

class IterDEModel(nn.Module):
    def __init__(self, KGE=None, embedding_manager=None, hard_loss=None, soft_loss=None, args=None):
        super(IterDEModel, self).__init__()
        self.KGE = KGE
        self.EmbeddingManager = embedding_manager
        self.hard_loss = hard_loss
        self.soft_loss = soft_loss
        self.args = args


    def get_postive_score(self, score):
        return score[:, 0]


    def get_negative_score(self, score):
        return score[:, 1:]

    def get_regularization(self):
        #Use L3 regularization for ComplEx and DistMult
        regularization = self.args.regularization * (
            self.EmbeddingManager.entity_embedding.norm(p = 3)**3 + 
            self.EmbeddingManager.relation_embedding.norm(p = 3).norm(p = 3)**3
        )
        regularization_log = {'regularization': regularization.item()}
        
        return regularization, regularization_log


    def forward(self, data, subsampling_weight, mode, half_epoch):
        t_head, t_relation, t_tail, head, relation, tail = self.EmbeddingManager(data, mode)
        if self.args.cuda:
            t_head, t_relation, t_tail, head, relation, tail = t_head.cuda(), t_relation.cuda(), t_tail.cuda(), head.cuda(), relation.cuda(), tail.cuda()
        
        
        if type(self.KGE).__name__ == 'TransE':
            t_score = self.KGE(t_head, t_relation, t_tail, mode)
            score = self.KGE(head, relation, tail, mode)
            
        if type(self.KGE).__name__ == 'RotatE':
            t_score = self.KGE(t_head, t_relation, t_tail, mode, {'embedding_range' : 6.0+2.0, 'embedding_dim':512})
            score = self.KGE(head, relation, tail, mode)
        
        if type(self.KGE).__name__ == 'SimplE':
            t_score = self.KGE(t_head, t_relation, t_tail, mode)
            score = self.KGE(head, relation, tail, mode)
        
        if type(self.KGE).__name__ == 'ComplEx':
            t_score = self.KGE(t_head, t_relation, t_tail, mode)
            score = self.KGE(head, relation, tail, mode)
        
        # 计算hard_label
        p_score, n_score = self.get_postive_score(score), self.get_negative_score(score)
        
        if type(self.KGE).__name__ == 'ComplEx':
            hard_loss, hard_loss_record = self.hard_loss(p_score, n_score, subsampling_weight, add_margin=True)
        else:
            hard_loss, hard_loss_record = self.hard_loss(p_score, n_score, subsampling_weight, sub_margin=True)
        
        # 计算soft_label
        loss, soft_loss_record = self.soft_loss(t_score, score, hard_loss, half_epoch)

        hard_loss_record.update(soft_loss_record)
        if self.args.regularization != 0.0:
            regularization, regularization_log = self.get_regularization()
            loss += regularization
            hard_loss_record.update(regularization_log)
        
        return loss, hard_loss_record
    
    def predict(self, data, mode):
        _, _, _, head, relation, tail = self.EmbeddingManager(data, mode)            
        score = self.KGE(head, relation, tail, mode)
        return score
        
        