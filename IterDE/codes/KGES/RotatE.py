import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class RotatE(nn.Module):
    def __init__(self, embedding_range=None, embedding_dim=None, margin=None):
        super(RotatE, self).__init__()
        self.embedding_range = embedding_range
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        logging.info(f'Init RotatE with embedding_range={self.embedding_range}, embedding_dim={self.embedding_dim}, margin={self.margin}')

    def forward(self, head, relation, tail, mode, real_dim=None):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]
        if real_dim==None:
            phase_relation = relation/(((self.embedding_range)/self.embedding_dim)/pi)
        else:
            phase_relation = relation/(((self.embedding_range)/real_dim)/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = score.sum(dim = 2)
        
        if self.margin is not None:
            score = self.margin - score
        else:
            score = score
        
        return score 