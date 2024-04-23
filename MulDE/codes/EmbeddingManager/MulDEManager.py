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
class MulDEManager(nn.Module):
    def __init__(self, args):
        super(MulDEManager, self).__init__()
        self.args = args
        self.teacher_num = self.args.teacher_num
        self.nentity = args.nentity
        self.nrelation = args.nrelation

        self.teacher_entity_embeddings = nn.ParameterList()
        self.teacher_relation_embeddings = nn.ParameterList()
        
        for i in range(self.teacher_num):
            # 构造每个模型的预训练路径
            pretrain_path = getattr(self.args, f'pretrain_path{i + 1}')
            pretrain_model = torch.load(pretrain_path)
            
            # 提取实体和关系嵌入，并添加到列表中
            entity_embedding = nn.Parameter(pretrain_model['model_state_dict']['entity_embedding'].cpu(), requires_grad=False)
            relation_embedding = nn.Parameter(pretrain_model['model_state_dict']['relation_embedding'].cpu(), requires_grad=False)
            self.teacher_entity_embeddings.append(entity_embedding)
            self.teacher_relation_embeddings.append(relation_embedding)
        
        self.entity_embedding = nn.Parameter(torch.empty(self.args.nentity, self.args.target_dim*self.args.entity_mul), requires_grad=True)
        self.relation_embedding = nn.Parameter(torch.empty(self.args.nrelation, self.args.target_dim*self.args.relation_mul), requires_grad=True)
        nn.init.xavier_uniform_(self.entity_embedding)
        nn.init.xavier_uniform_(self.relation_embedding)
    
    def forward(self, sample, mode, index):    
        pass
    
    def get_teacher_score(self, sample, negative, mode, KGE):
        positive, _ = sample
        negative = negative
        t_scores = torch.empty(self.teacher_num, positive.shape[0], self.args.k)
        
        for i in range(self.teacher_num):
            head, relation, tail = self.get_teacher_embedding((positive, negative), mode, i)
            t_score = KGE(head, relation, tail, mode, {'embedding_range':11.0, 'embedding_dim':128})
            t_scores[i] = t_score[:, 1:]

        t_scores = t_scores.to(positive.device)

        return t_scores
            
    
    def get_teacher_embedding(self, sample, mode, index):
        origin_head, origin_tail = self.EntityEmbeddingExtract(self.teacher_entity_embeddings[index], sample, mode)
        origin_relation = self.RelationEmbeddingExtract(self.teacher_relation_embeddings[index], sample)
        return origin_head, origin_relation, origin_tail
    
    def get_student_embedding(self, sample, mode):
        head, tail = self.EntityEmbeddingExtract(self.entity_embedding, sample, mode)
        relation = self.RelationEmbeddingExtract(self.relation_embedding, sample)
        return head, relation, tail
    
    def get_ER_Query(self, sample, mode):
        if mode == 'head-batch':
            positive, _ = sample
            negative = torch.arange(0, self.nentity)
            batch_size, negative_sample_size = positive.size(0), negative.size(0)
            negative = negative.repeat(batch_size, 1)
            negative = negative.to(positive.device)

            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=negative.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=positive[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=positive[:, 2]
            ).unsqueeze(1)
            
            
        elif mode == 'tail-batch':
            positive, _ = sample
            negative = torch.arange(0, self.nentity)
            batch_size, negative_sample_size = positive.size(0), negative.size(0)
            negative = negative.repeat(batch_size, 1)
            negative = negative.to(positive.device)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=positive[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=positive[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=negative.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
        return head, relation, tail
        
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