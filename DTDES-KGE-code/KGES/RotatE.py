import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class RotatE(nn.Module):
    def __init__(self, margin=None, embedding_range=11.0, embedding_dim=512):
        super(RotatE, self).__init__()
        self.margin = margin
        self.origin_margin = embedding_range - 2.0
        self.embedding_range = embedding_range
        self.embedding_dim = embedding_dim
        
        logging.info(f'Init RotatE with embedding_range={self.embedding_range}, embedding_dim={self.embedding_dim}, margin={self.margin}')
    
    def forward(self, head, relation, tail, mode, args):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]
        
        if head.shape[-1]>500:
            phase_relation = relation/(((self.embedding_range)/self.embedding_dim)/pi)
        else:
            embedding_range, embedding_dim = 2.0+args.gamma, args.target_dim
            phase_relation = relation/(((embedding_range)/embedding_dim)/pi)
        
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
        
        if head.shape[-1]>500:
            score = self.origin_margin - score
        else:
            if self.margin is not None:
                score = self.margin - score
            else:
                score = score
        
        return score 
    
    
    def get_distance(self, head, relation, tail, mode, args):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=0)
        re_tail, im_tail = torch.chunk(tail, 2, dim=0)

        #Make phases of relations uniformly distributed in [-pi, pi]
        
        if head.shape[-1]>500:
            phase_relation = relation/(((self.embedding_range)/self.embedding_dim)/pi)
        else:
            embedding_range, embedding_dim = 2.0+args.gamma, args.target_dim
            phase_relation = relation/(((embedding_range)/embedding_dim)/pi)
        
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
        
        distance = torch.cat((re_score, im_score), dim=0)
        return distance