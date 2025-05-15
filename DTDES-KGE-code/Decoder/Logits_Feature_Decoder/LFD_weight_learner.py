import sys
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

original_directory = os.getcwd()
new_directory = original_directory + '/code/Decoder/Sim_decoder/'
if new_directory not in sys.path:
    sys.path.append(new_directory)
    
from LFD_norm import *
from LFD_hard_loss import *
from LFD_soft_loss import *
from LFD_similarity import *
from LFD_Encoder import *
from transformer import TransformerEncoderLayer

"""
粗粒度的权重学习，通过head和relation指定权重
"""
class weight_learnerv1(nn.Module):
    def __init__(self, args, entity_dim, relation_dim):
        super(weight_learnerv1, self).__init__()
        self.args=args
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.layer_mul = 2
        self.hidden_dim = 32
        self.entity_transfer = nn.Sequential(
            nn.Linear((self.entity_dim), (self.entity_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.entity_dim) * self.layer_mul, (self.hidden_dim))
        )
        self.relation_transfer = nn.Sequential(
            nn.Linear((self.relation_dim), (self.relation_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.relation_dim) * self.layer_mul, (self.hidden_dim))
        )
        self.MLP = nn.Sequential(
            nn.Linear((self.hidden_dim)*2, (self.hidden_dim)*2 * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.hidden_dim)*2 * self.layer_mul, 2)
        )

    def forward(self, head, relation, tail=None, stu_score=None, PT1_score=None, PT2_score=None, data=None):
        head_transfer = self.entity_transfer(head)
        relation_transfer = self.relation_transfer(relation)
        combined = torch.cat((head_transfer, relation_transfer), dim=2)
        x = self.MLP(combined)
        weights = F.softmax(x, dim=2)
                
        return weights


class weight_learnerv1_Res(nn.Module):
    def __init__(self, args, entity_dim, relation_dim):
        super(weight_learnerv1_Res, self).__init__()
        self.args=args
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.layer_mul = 2
        self.hidden_dim = 32
        self.entity_transfer = nn.Sequential(
            nn.Linear((self.entity_dim), (self.entity_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.entity_dim) * self.layer_mul, (self.hidden_dim))
        )
        self.relation_transfer = nn.Sequential(
            nn.Linear((self.relation_dim), (self.relation_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.relation_dim) * self.layer_mul, (self.hidden_dim))
        )
        self.MLP = nn.Sequential(
            nn.Linear((self.hidden_dim)*2, (self.hidden_dim)*2 * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.hidden_dim)*2 * self.layer_mul, 2)
        )

    def forward(self, head, relation, tail=None, stu_score=None, PT1_score=None, PT2_score=None, data=None):
        head_transfer = self.entity_transfer(head)
        relation_transfer = self.relation_transfer(relation)
        
        if self.entity_dim == self.hidden_dim:
            head_transfer = head + head_transfer
            
        if self.relation_dim == self.hidden_dim:
            relation_transfer = relation + relation_transfer
        
        combined = torch.cat((head_transfer, relation_transfer), dim=2)
        x = self.MLP(combined)
        weights = F.softmax(x, dim=2)
                
        return weights


class weight_learnerv1_PreLN(nn.Module):
    def __init__(self, args, entity_dim, relation_dim):
        super(weight_learnerv1_PreLN, self).__init__()
        self.args=args
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.layer_mul = 2
        self.hidden_dim = 32
        self.entity_transfer = nn.Sequential(
            nn.LayerNorm(self.entity_dim),
            nn.Linear((self.entity_dim), (self.entity_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.entity_dim) * self.layer_mul, (self.hidden_dim))
        )
        self.relation_transfer = nn.Sequential(
            nn.LayerNorm(self.relation_dim),
            nn.Linear((self.relation_dim), (self.relation_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.relation_dim) * self.layer_mul, (self.hidden_dim))
        )
        self.MLP = nn.Sequential(
            nn.LayerNorm(self.hidden_dim*2),
            nn.Linear((self.hidden_dim)*2, (self.hidden_dim)*2 * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.hidden_dim)*2 * self.layer_mul, 2)
        )

    def forward(self, head, relation, tail=None, stu_score=None,PT1_score=None, PT2_score=None, data=None):
        head_transfer = self.entity_transfer(head)
        relation_transfer = self.relation_transfer(relation)
        combined = torch.cat((head_transfer, relation_transfer), dim=2)
        x = self.MLP(combined)
        weights = F.softmax(x, dim=2)
                
        return weights


class weight_learnerv1_PostLN(nn.Module):
    def __init__(self, args, entity_dim, relation_dim):
        super(weight_learnerv1_PostLN, self).__init__()
        self.args=args
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.layer_mul = 2
        self.hidden_dim = 32
        self.entity_transfer = nn.Sequential(
            nn.Linear((self.entity_dim), (self.entity_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.entity_dim) * self.layer_mul, (self.hidden_dim)),
            nn.LayerNorm(self.hidden_dim)
        )
        self.relation_transfer = nn.Sequential(
            nn.Linear((self.relation_dim), (self.relation_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.relation_dim) * self.layer_mul, (self.hidden_dim)),
            nn.LayerNorm(self.hidden_dim)
        )
        self.MLP = nn.Sequential(
            nn.LayerNorm(self.hidden_dim*2),
            nn.Linear((self.hidden_dim)*2, (self.hidden_dim)*2 * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.hidden_dim)*2 * self.layer_mul, 2)
        )

    def forward(self, head, relation, tail=None, stu_score=None,PT1_score=None, PT2_score=None, data=None):
        head_transfer = self.entity_transfer(head)
        relation_transfer = self.relation_transfer(relation)
        combined = torch.cat((head_transfer, relation_transfer), dim=2)
        x = self.MLP(combined)
        weights = F.softmax(x, dim=2)
                
        return weights


class weight_learnerv1_PostLNRes(nn.Module):
    def __init__(self, args, entity_dim, relation_dim):
        super(weight_learnerv1_PostLNRes, self).__init__()
        self.args=args
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.layer_mul = 2
        self.hidden_dim = 32
        self.entity_transfer = nn.Sequential(
            nn.Linear((self.entity_dim), (self.entity_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.entity_dim) * self.layer_mul, (self.hidden_dim)),
            nn.LayerNorm(self.hidden_dim)
        )
        self.relation_transfer = nn.Sequential(
            nn.Linear((self.relation_dim), (self.relation_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.relation_dim) * self.layer_mul, (self.hidden_dim)),
            nn.LayerNorm(self.hidden_dim)
        )
        self.MLP = nn.Sequential(
            nn.Linear((self.hidden_dim)*2, (self.hidden_dim)*2 * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.hidden_dim)*2 * self.layer_mul, 2)
        )

    def forward(self, head, relation, tail=None, stu_score=None, PT1_score=None, PT2_score=None, data=None):
        head_transfer = self.entity_transfer(head)
        relation_transfer = self.relation_transfer(relation)
        if self.entity_dim == self.hidden_dim:
            head_transfer = head + head_transfer
        if self.relation_dim == self.hidden_dim:
            relation_transfer = relation + relation_transfer
        combined = torch.cat((head_transfer, relation_transfer), dim=2)
        x = self.MLP(combined)
        weights = F.softmax(x, dim=2)
                
        return weights



class weight_learnerv2(nn.Module):
    def __init__(self, args, entity_dim, relation_dim):
        super(weight_learnerv2, self).__init__()
        self.args=args
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.layer_mul = 2
        self.hidden_dim = 32
        self.entity_transfer = nn.Sequential(
            nn.Linear((self.entity_dim), (self.entity_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.entity_dim) * self.layer_mul, (self.hidden_dim)),
            nn.LayerNorm(self.hidden_dim)
        )
        self.relation_transfer = nn.Sequential(
            nn.Linear((self.relation_dim), (self.relation_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.relation_dim) * self.layer_mul, (self.hidden_dim)),
            nn.LayerNorm(self.hidden_dim)
        )
        self.teacher_message_transfer = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        self.MLP = nn.Sequential(
            nn.Linear((self.hidden_dim)*3, (self.hidden_dim)*2 * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.hidden_dim)*2 * self.layer_mul, 2)
        )
        

    def _cal_tea_message(self, PT1_score, PT2_score):
        PT1_prob, PT2_prob = F.softmax(PT1_score, dim=-1), F.softmax(PT2_score, dim=-1)
        PT1_pos, PT2_pos = PT1_prob[:, 0:1], PT2_prob[:, 0:1]
        
        log_PT1_prob = torch.log(PT1_prob + 1e-9) # 加一个小的epsilon避免log(0)
        log_PT2_prob = torch.log(PT2_prob + 1e-9) # 加一个小的epsilon避免log(0)
        
        # 计算 D(PT1 || PT2)
        kl_div_pt1_pt2_terms = F.kl_div(log_PT2_prob, PT1_prob, reduction='none')
        kl_div_pt2_pt1_terms = F.kl_div(log_PT1_prob, PT2_prob, reduction='none')
        kl_div_pt1_pt2_per_row = kl_div_pt1_pt2_terms.mean(dim=-1).unsqueeze(-1)
        kl_div_pt2_pt1_per_row = kl_div_pt2_pt1_terms.mean(dim=-1).unsqueeze(-1)
        
        teacher_message = torch.cat((PT1_pos, PT2_pos, kl_div_pt1_pt2_per_row, kl_div_pt2_pt1_per_row), dim=-1).unsqueeze(1)
        
        return teacher_message
    
    
    def forward(self, head, relation, tail=None, stu_score=None, PT1_score=None, PT2_score=None, data=None):
        head_transfer = self.entity_transfer(head)
        relation_transfer = self.relation_transfer(relation)
        
        teacher_message = self._cal_tea_message(PT1_score, PT2_score)
        teacher_message_transfer = self.teacher_message_transfer(teacher_message)
        
        combined = torch.cat((head_transfer, relation_transfer, teacher_message_transfer), dim=2)
        x = self.MLP(combined)
        weights = F.softmax(x, dim=2)
                
        return weights


class weight_learnerv2_1(nn.Module):
    def __init__(self, args, entity_dim, relation_dim):
        super(weight_learnerv2_1, self).__init__()
        self.args=args
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.layer_mul = 2
        self.hidden_dim = 32
        self.entity_transfer = nn.Sequential(
            nn.Linear((self.entity_dim), (self.entity_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.entity_dim) * self.layer_mul, (self.hidden_dim)),
            nn.LayerNorm(self.hidden_dim)
        )
        self.relation_transfer = nn.Sequential(
            nn.Linear((self.relation_dim), (self.relation_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.relation_dim) * self.layer_mul, (self.hidden_dim)),
            nn.LayerNorm(self.hidden_dim)
        )
        self.teacher_message_transfer = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 4),
            nn.LayerNorm(4)
        )
        
        self.MLP = nn.Sequential(
            nn.Linear((self.hidden_dim)*2+4, (self.hidden_dim)*2 * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.hidden_dim)*2 * self.layer_mul, 2)
        )
        

    def _cal_tea_message(self, PT1_score, PT2_score):
        PT1_prob, PT2_prob = F.softmax(PT1_score, dim=-1), F.softmax(PT2_score, dim=-1)
        PT1_pos, PT2_pos = PT1_prob[:, 0:1], PT2_prob[:, 0:1]
        
        log_PT1_prob = torch.log(PT1_prob + 1e-9) # 加一个小的epsilon避免log(0)
        log_PT2_prob = torch.log(PT2_prob + 1e-9) # 加一个小的epsilon避免log(0)
        
        # 计算 D(PT1 || PT2)
        kl_div_pt1_pt2_terms = F.kl_div(log_PT2_prob, PT1_prob, reduction='none')
        kl_div_pt2_pt1_terms = F.kl_div(log_PT1_prob, PT2_prob, reduction='none')
        kl_div_pt1_pt2_per_row = kl_div_pt1_pt2_terms.mean(dim=-1).unsqueeze(-1)
        kl_div_pt2_pt1_per_row = kl_div_pt2_pt1_terms.mean(dim=-1).unsqueeze(-1)
        
        teacher_message = torch.cat((PT1_pos, PT2_pos, kl_div_pt1_pt2_per_row, kl_div_pt2_pt1_per_row), dim=-1).unsqueeze(1)
        
        return teacher_message
    
    
    def forward(self, head, relation, tail=None, stu_score=None, PT1_score=None, PT2_score=None, data=None):
        head_transfer = self.entity_transfer(head)
        relation_transfer = self.relation_transfer(relation)
        
        teacher_message = self._cal_tea_message(PT1_score, PT2_score)
        teacher_message_transfer = self.teacher_message_transfer(teacher_message)
        
        combined = torch.cat((head_transfer, relation_transfer, teacher_message_transfer), dim=2)
        x = self.MLP(combined)
        weights = F.softmax(x, dim=2)
                
        return weights



class weight_learnerv3(nn.Module):
    def __init__(self, args, entity_dim, relation_dim):
        super(weight_learnerv3, self).__init__()
        self.args=args
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.layer_mul = 2
        self.hidden_dim = 32
        self.entity_transfer = nn.Sequential(
            nn.Linear((self.entity_dim), (self.entity_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.entity_dim) * self.layer_mul, (self.hidden_dim)),
            nn.LayerNorm(self.hidden_dim)
        )
        self.relation_transfer = nn.Sequential(
            nn.Linear((self.relation_dim), (self.relation_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.relation_dim) * self.layer_mul, (self.hidden_dim)),
            nn.LayerNorm(self.hidden_dim)
        )
        self.teacher_message_transfer = nn.Sequential(
            nn.Linear(4, 16),
            nn.Linear(16, self.hidden_dim),
        )
        
        self.student_message_transfer = nn.Sequential(
            nn.Linear(3, 16),
            nn.Linear(16, self.hidden_dim),
        )
        
        self.MLP = nn.Sequential(
            nn.Linear((self.hidden_dim)*4, (self.hidden_dim)*2 * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.hidden_dim)*2 * self.layer_mul, 2)
        )
        

    def _cal_tea_message(self, PT1_score, PT2_score):
        PT1_prob, PT2_prob = F.softmax(PT1_score, dim=-1), F.softmax(PT2_score, dim=-1)
        PT1_pos, PT2_pos = PT1_prob[:, 0:1], PT2_prob[:, 0:1]
        
        log_PT1_prob = torch.log(PT1_prob + 1e-9) # 加一个小的epsilon避免log(0)
        log_PT2_prob = torch.log(PT2_prob + 1e-9) # 加一个小的epsilon避免log(0)
        
        # 计算 D(PT1 || PT2)
        kl_div_pt1_pt2_terms = F.kl_div(log_PT2_prob, PT1_prob, reduction='none')
        kl_div_pt2_pt1_terms = F.kl_div(log_PT1_prob, PT2_prob, reduction='none')
        kl_div_pt1_pt2_per_row = kl_div_pt1_pt2_terms.mean(dim=-1).unsqueeze(-1)
        kl_div_pt2_pt1_per_row = kl_div_pt2_pt1_terms.mean(dim=-1).unsqueeze(-1)
        
        teacher_message = torch.cat((PT1_pos, PT2_pos, kl_div_pt1_pt2_per_row, kl_div_pt2_pt1_per_row), dim=-1).unsqueeze(1)
        
        return teacher_message
    
    
    def _cal_stu_message(self, stu_score, PT1_score, PT2_score):
        stu_prob, PT1_prob, PT2_prob = F.softmax(stu_score, dim=-1), F.softmax(PT1_score, dim=-1), F.softmax(PT2_score, dim=-1)
        stu_pos, PT1_pos, PT2_pos = stu_prob[:, 0:1], PT1_prob[:, 0:1], PT2_prob[:, 0:1]
        log_stu_prob, log_PT1_prob, log_PT2_prob = torch.log(stu_pos + 1e-9), torch.log(PT1_prob + 1e-9), torch.log(PT2_prob + 1e-9)
        
        kl_div_stu_PT1 = F.kl_div(log_stu_prob, PT1_prob, reduction='none')
        kl_div_stu_PT2 = F.kl_div(log_stu_prob, PT2_prob, reduction='none')
        kl_div_stu_PT1_per_row = kl_div_stu_PT1.mean(dim=-1).unsqueeze(-1)
        kl_div_stu_PT2_per_row = kl_div_stu_PT2.mean(dim=-1).unsqueeze(-1)
        
        # kl_div_stu_PT1_per_row = kl_div_stu_PT1_per_row.detach()
        # kl_div_stu_PT2_per_row = kl_div_stu_PT2_per_row.detach()
        
        stu_message = torch.cat((stu_pos, kl_div_stu_PT1_per_row, kl_div_stu_PT2_per_row), dim=-1).unsqueeze(1)
        
        return stu_message
    
    
    def forward(self, head, relation, tail=None, stu_score=None, PT1_score=None, PT2_score=None, data=None):
        head_transfer = self.entity_transfer(head)
        relation_transfer = self.relation_transfer(relation)
        
        teacher_message = self._cal_tea_message(PT1_score, PT2_score)
        teacher_message_transfer = self.teacher_message_transfer(teacher_message)
        
        stu_message = self._cal_stu_message(stu_score, PT1_score, PT2_score)
        stu_message_transfer = self.student_message_transfer(stu_message)
        
        combined = torch.cat((head_transfer, relation_transfer, teacher_message_transfer, stu_message_transfer), dim=2)
        x = self.MLP(combined)
        weights = F.softmax(x, dim=2)
        
        print(weights.shape)
        
        return weights



class weight_learnerv3_prior(nn.Module):
    def __init__(self, args, entity_dim, relation_dim):
        super(weight_learnerv3_prior, self).__init__()
        self.args=args
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.layer_mul = 2
        self.hidden_dim = 32
                
        self.entity_transfer = nn.Sequential(
            nn.Linear((self.entity_dim), (self.entity_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.entity_dim) * self.layer_mul, (self.hidden_dim)),
            nn.LayerNorm(self.hidden_dim)
        )
        self.relation_transfer = nn.Sequential(
            nn.Linear((self.relation_dim), (self.relation_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.relation_dim) * self.layer_mul, (self.hidden_dim)),
            nn.LayerNorm(self.hidden_dim)
        )
        self.teacher_message_transfer = nn.Sequential(
            nn.Linear(4, 16),
            nn.Linear(16, self.hidden_dim),
        )
        
        self.student_message_transfer = nn.Sequential(
            nn.Linear(3, 16),
            nn.Linear(16, self.hidden_dim),
        )
        
        self.MLP = nn.Sequential(
            nn.Linear((self.hidden_dim)*4, (self.hidden_dim)*2 * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.hidden_dim)*2 * self.layer_mul, 2)
        )
        
        if 'FB15k-237' in self.args.data_path:
            cur_path = self.args.data_path + '/FB15k-237_relation_cur.txt'
            kra_path = self.args.data_path + '/FB15k-237_relation_kra.txt'
        elif 'wn18rr' in self.args.data_path:
            cur_path = self.args.data_path + '/wn18rr_relation_cur.txt'
            kra_path = self.args.data_path + '/wn18rr_relation_kra.txt'
            
        self.cur_dict = self.read_file_to_dict(cur_path)
        self.kra_dict = self.read_file_to_dict(kra_path)
        
        num_rel = self.args.nrelation             # 或者已知关系总数
        device  = torch.device('cuda', 0)         # 按需改成 cpu / 其他 gpu

        self.cur = torch.zeros(num_rel, device=device)
        self.kra = torch.zeros(num_rel, device=device)
        for k, v in self.cur_dict.items():
            self.cur[k] = v
            self.cur[k + num_rel//2] = v
        for k, v in self.kra_dict.items():
            self.kra[k] = v
            self.kra[k + num_rel//2] = v
        self.cur.requires_grad_(False)
        self.kra.requires_grad_(False)
    
    def read_file_to_dict(self, path: str):
        data = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue                # 跳过空行
                first, second, *_ = line.split()  # 支持任意空白分隔
                data[int(first)] = round(float(second), 1)
        return data
    

    def _cal_tea_message(self, PT1_score, PT2_score):
        PT1_prob, PT2_prob = F.softmax(PT1_score, dim=-1), F.softmax(PT2_score, dim=-1)
        PT1_pos, PT2_pos = PT1_prob[:, 0:1], PT2_prob[:, 0:1]
        
        log_PT1_prob = torch.log(PT1_prob + 1e-9) # 加一个小的epsilon避免log(0)
        log_PT2_prob = torch.log(PT2_prob + 1e-9) # 加一个小的epsilon避免log(0)
        
        # 计算 D(PT1 || PT2)
        kl_div_pt1_pt2_terms = F.kl_div(log_PT2_prob, PT1_prob, reduction='none')
        kl_div_pt2_pt1_terms = F.kl_div(log_PT1_prob, PT2_prob, reduction='none')
        kl_div_pt1_pt2_per_row = kl_div_pt1_pt2_terms.mean(dim=-1).unsqueeze(-1)
        kl_div_pt2_pt1_per_row = kl_div_pt2_pt1_terms.mean(dim=-1).unsqueeze(-1)
        
        teacher_message = torch.cat((PT1_pos, PT2_pos, kl_div_pt1_pt2_per_row, kl_div_pt2_pt1_per_row), dim=-1).unsqueeze(1)
        
        return teacher_message
    
    
    def _cal_stu_message(self, stu_score, PT1_score, PT2_score):
        stu_prob, PT1_prob, PT2_prob = F.softmax(stu_score, dim=-1), F.softmax(PT1_score, dim=-1), F.softmax(PT2_score, dim=-1)
        stu_pos, PT1_pos, PT2_pos = stu_prob[:, 0:1], PT1_prob[:, 0:1], PT2_prob[:, 0:1]
        log_stu_prob, log_PT1_prob, log_PT2_prob = torch.log(stu_pos + 1e-9), torch.log(PT1_prob + 1e-9), torch.log(PT2_prob + 1e-9)
        
        kl_div_stu_PT1 = F.kl_div(log_stu_prob, PT1_prob, reduction='none')
        kl_div_stu_PT2 = F.kl_div(log_stu_prob, PT2_prob, reduction='none')
        kl_div_stu_PT1_per_row = kl_div_stu_PT1.mean(dim=-1).unsqueeze(-1)
        kl_div_stu_PT2_per_row = kl_div_stu_PT2.mean(dim=-1).unsqueeze(-1)
        
        # kl_div_stu_PT1_per_row = kl_div_stu_PT1_per_row.detach()
        # kl_div_stu_PT2_per_row = kl_div_stu_PT2_per_row.detach()
        
        stu_message = torch.cat((stu_pos, kl_div_stu_PT1_per_row, kl_div_stu_PT2_per_row), dim=-1).unsqueeze(1)
        
        return stu_message
    
    def cal_prior(self, data):
        cur_thres = -0.4
        cur_opt   = 0.8
        kra_thres = 0.8
        kra_opt   = 0.8

        positive_sample, negative_sample = data
        r_idx = positive_sample[:, 1]
        
        cur = self.cur[r_idx]          # [batch]
        kra = self.kra[r_idx]          # [batch]
        batch = cur.size(0)

        # 不可训练权重张量 —— requires_grad=False
        prior_weight = torch.empty(
            batch, 2,
            dtype=cur.dtype,
            device='cuda:0',
            requires_grad=False
        )

        # ① 先根据 cur 赋值
        mask_cur_low  = cur < cur_thres          # cur < -0.4
        mask_cur_high = ~mask_cur_low            # cur ≥ -0.4

        prior_weight[mask_cur_low, 1]  = cur_opt      # 第二列
        prior_weight[mask_cur_low, 0]  = 1 - cur_opt  # 第一列

        prior_weight[mask_cur_high, 0] = cur_opt
        prior_weight[mask_cur_high, 1] = 1 - cur_opt

        # ② kra > kra_thres
        mask_kra_high = kra > kra_thres           # kra > 0.8
        prior_weight[mask_kra_high, 1] += kra_opt
        prior_weight[mask_kra_high, 0] += 1 - kra_opt

        prior_weight = prior_weight/2
        
        # 调整为 [batch, 1, 2] 形状后返回
        return prior_weight.unsqueeze(1)
        
        
    def forward(self, head, relation, tail=None, stu_score=None, PT1_score=None, PT2_score=None, data=None):
        head_transfer = self.entity_transfer(head)
        relation_transfer = self.relation_transfer(relation)
        
        teacher_message = self._cal_tea_message(PT1_score, PT2_score)
        teacher_message_transfer = self.teacher_message_transfer(teacher_message)
        
        stu_message = self._cal_stu_message(stu_score, PT1_score, PT2_score)
        stu_message_transfer = self.student_message_transfer(stu_message)
        
        combined = torch.cat((head_transfer, relation_transfer, teacher_message_transfer, stu_message_transfer), dim=2)
        x = self.MLP(combined)
        weights = F.softmax(x, dim=2)
        
        prior_weight = self.cal_prior(data)
        pw = 0.1
        weights = (pw * prior_weight + weights) / (1 + pw)
        
        return weights


class weight_learnerv4(nn.Module):
    def __init__(self, args, entity_dim, relation_dim):
        super(weight_learnerv4, self).__init__()
        self.args=args
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.layer_mul = 2
        self.hidden_dim = 32
        self.entity_transfer = nn.Sequential(
            nn.Linear((self.entity_dim), (self.entity_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.entity_dim) * self.layer_mul, (self.hidden_dim)),
            nn.LayerNorm(self.hidden_dim)
        )
        self.relation_transfer = nn.Sequential(
            nn.Linear((self.relation_dim), (self.relation_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.relation_dim) * self.layer_mul, (self.hidden_dim)),
            nn.LayerNorm(self.hidden_dim)
        )
        self.teacher_message_transfer = nn.Sequential(
            nn.Linear(4, 16),
            nn.Linear(16, self.hidden_dim),
        )
        
        self.student_message_transfer = nn.Sequential(
            nn.Linear(3, 16),
            nn.Linear(16, self.hidden_dim),
        )
        
        self.MHA = TransformerEncoderLayer(embedding_dim=self.hidden_dim, num_heads=self.args.heads, dropout=self.args.dropout)
        
        self.MLP = nn.Sequential(
            nn.Linear((self.hidden_dim), (self.hidden_dim)* self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.hidden_dim)* self.layer_mul, 2)
        )
        

    def _cal_tea_message(self, PT1_score, PT2_score):
        PT1_prob, PT2_prob = F.softmax(PT1_score, dim=-1), F.softmax(PT2_score, dim=-1)
        PT1_pos, PT2_pos = PT1_prob[:, 0:1], PT2_prob[:, 0:1]
        
        log_PT1_prob = torch.log(PT1_prob + 1e-9) # 加一个小的epsilon避免log(0)
        log_PT2_prob = torch.log(PT2_prob + 1e-9) # 加一个小的epsilon避免log(0)
        
        # 计算 D(PT1 || PT2)
        kl_div_pt1_pt2_terms = F.kl_div(log_PT2_prob, PT1_prob, reduction='none')
        kl_div_pt2_pt1_terms = F.kl_div(log_PT1_prob, PT2_prob, reduction='none')
        kl_div_pt1_pt2_per_row = kl_div_pt1_pt2_terms.mean(dim=-1).unsqueeze(-1)
        kl_div_pt2_pt1_per_row = kl_div_pt2_pt1_terms.mean(dim=-1).unsqueeze(-1)
        
        teacher_message = torch.cat((PT1_pos, PT2_pos, kl_div_pt1_pt2_per_row, kl_div_pt2_pt1_per_row), dim=-1).unsqueeze(1)
        
        return teacher_message
    
    
    def _cal_stu_message(self, stu_score, PT1_score, PT2_score):
        stu_prob, PT1_prob, PT2_prob = F.softmax(stu_score, dim=-1), F.softmax(PT1_score, dim=-1), F.softmax(PT2_score, dim=-1)
        stu_pos, PT1_pos, PT2_pos = stu_prob[:, 0:1], PT1_prob[:, 0:1], PT2_prob[:, 0:1]
        log_stu_prob, log_PT1_prob, log_PT2_prob = torch.log(stu_pos + 1e-9), torch.log(PT1_prob + 1e-9), torch.log(PT2_prob + 1e-9)
        
        kl_div_stu_PT1 = F.kl_div(log_stu_prob, PT1_prob, reduction='none')
        kl_div_stu_PT2 = F.kl_div(log_stu_prob, PT2_prob, reduction='none')
        kl_div_stu_PT1_per_row = kl_div_stu_PT1.mean(dim=-1).unsqueeze(-1)
        kl_div_stu_PT2_per_row = kl_div_stu_PT2.mean(dim=-1).unsqueeze(-1)
        
        stu_message = torch.cat((stu_pos, kl_div_stu_PT1_per_row, kl_div_stu_PT2_per_row), dim=-1).unsqueeze(1)
        
        return stu_message
    
    
    def forward(self, head, relation, tail=None, stu_score=None, PT1_score=None, PT2_score=None, data=None):
        head_transfer = self.entity_transfer(head)
        relation_transfer = self.relation_transfer(relation)
        
        teacher_message = self._cal_tea_message(PT1_score, PT2_score)
        teacher_message_transfer = self.teacher_message_transfer(teacher_message)
        
        stu_message = self._cal_stu_message(stu_score, PT1_score, PT2_score)
        stu_message_transfer = self.student_message_transfer(stu_message)
        
        query = stu_message_transfer
        key = value = torch.cat((head_transfer, relation_transfer, teacher_message_transfer), dim=1)

        # print(query.shape, key.shape, value.shape)

        outputs = self.MHA(query, key, value)
        outputs = self.MLP(outputs)
        weights = F.softmax(outputs, dim=2)
                
        return weights