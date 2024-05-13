import sys
import os

import torch.nn.functional as F
import torch
import torch.nn as nn

class DualDEModel(nn.Module):
    def __init__(self, KGE=None, embedding_manager=None, dualloss=None, args=None):
        super(DualDEModel, self).__init__()
        self.KGE = KGE
        self.EmbeddingManager = embedding_manager
        self.DualLoss = dualloss
        self.args = args


    def get_postive_score(self, score):
        return score[:, 0]


    def get_negative_score(self, score):
        return score[:, 1:]


    def forward(self, data, subsampling_weight, mode):
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
        
        structure_loss_dict = self.DualLoss.structure_loss(t_head, t_tail, head, tail)
        
        loss, loss_record = self.DualLoss(t_score, score, structure_loss_dict, subsampling_weight)

        return loss, loss_record
    
    def predict(self, data, mode):
        _, _, _, head, relation, tail = self.EmbeddingManager(data, mode)            
        score = self.KGE(head, relation, tail, mode)
        return score
        
        