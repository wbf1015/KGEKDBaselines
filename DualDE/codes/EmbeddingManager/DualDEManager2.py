import sys
import os

import torch.nn.functional as F
import torch
import torch.nn as nn
'''
按照徐博建议给出的Manager方式
entity的embedding需要对参数进行更新。
relation则不需要
'''
class DualDEManager2(nn.Module):
    def __init__(self, args):
        super(DualDEManager2, self).__init__()
        self.args = args
        # pretrain_model = torch.load(os.path.join(self.args.pretrain_path, 'checkpoint'))
        pretrain_model = torch.load(self.args.pretrain_path)
        self.origin_entity_embedding = nn.Parameter(pretrain_model['model_state_dict']['origin_entity_embedding'].cpu(), requires_grad=True)
        self.origin_relation_embedding = nn.Parameter(pretrain_model['model_state_dict']['origin_relation_embedding'].cpu(), requires_grad=True)
        self.entity_embedding = nn.Parameter(pretrain_model['model_state_dict']['entity_embedding'].cpu(), requires_grad=True)
        self.relation_embedding = nn.Parameter(pretrain_model['model_state_dict']['relation_embedding'].cpu(), requires_grad=True)
        nn.init.xavier_uniform_(self.entity_embedding)
        nn.init.xavier_uniform_(self.relation_embedding)
    
    def forward(self, sample, mode):
        origin_head, origin_tail = self.EntityEmbeddingExtract(self.origin_entity_embedding, sample, mode)
        origin_relation = self.RelationEmbeddingExtract(self.origin_relation_embedding, sample)
        head, tail = self.EntityEmbeddingExtract(self.entity_embedding, sample, mode)
        relation = self.RelationEmbeddingExtract(self.relation_embedding, sample)
        return origin_head, origin_relation, origin_tail, head, relation, tail
    
    def get_embedding(self):
        return self.entity_embedding, self.relation_embedding
    
    def EntityEmbeddingExtract(self, entity_embedding, sample, mode):
        if mode == 'head-batch':
            positive, negative = sample
            batch_size, negative_sample_size = negative.size(0), negative.size(1)
            
            neg_head = torch.index_select(
                entity_embedding, 
                dim=0, 
                index=negative.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            pos_head = torch.index_select(
                entity_embedding, 
                dim=0, 
                index=positive[:, 0]
            ).unsqueeze(1)
            
            head = torch.cat((pos_head, neg_head), dim=1)
            
            tail = torch.index_select(
                entity_embedding, 
                dim=0, 
                index=positive[:, 2]
            ).unsqueeze(1)
        
        elif mode == 'tail-batch':
            positive, negative = sample
            batch_size, negative_sample_size = negative.size(0), negative.size(1)
            
            neg_tail = torch.index_select(
                entity_embedding, 
                dim=0, 
                index=negative.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            pos_tail = torch.index_select(
                entity_embedding, 
                dim=0, 
                index=positive[:, 2]
            ).unsqueeze(1)
            
            tail = torch.cat((pos_tail, neg_tail), dim=1)
            
            head = torch.index_select(
                entity_embedding, 
                dim=0, 
                index=positive[:, 0]
            ).unsqueeze(1)
            
        return head, tail

    def RelationEmbeddingExtract(self, relation_embedding, sample):
        positive, negative = sample
        
        relation = torch.index_select(
                relation_embedding, 
                dim=0, 
                index=positive[:, 1]
            ).unsqueeze(1)
        
        return relation