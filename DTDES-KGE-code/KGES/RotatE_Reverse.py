import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class RotatE_Reverse(nn.Module):
    def __init__(self, teacher_margin=9.0, teacher_embedding_dim=512):
        super(RotatE_Reverse, self).__init__()
        # RotatE中的成员变量记录的是教师模型的信息，真正学生的参数在args中保存
        self.teacher_margin = teacher_margin
        self.teacher_embedding_dim = teacher_embedding_dim
        self.teacher_embedding_range = self.teacher_margin + 2.0
        
        # logging.info(f'Init RotatE_Reverse with embedding_range={self.embedding_range}, embedding_dim={self.embedding_dim}, margin={self.margin}')
    
    def forward(self, head, relation, tail):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation/(((self.teacher_embedding_range)/self.teacher_embedding_dim)/pi)
        
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = score.sum(dim = 2)
        score = self.teacher_margin - score
        
        return score

