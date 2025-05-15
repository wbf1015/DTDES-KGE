import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os


class Local_Combine_hr(nn.Module):
    def __init__(self, entity_dim=512, relation_dim=512, hidden_dim=32, layer_mul=2):
        super(Local_Combine_hr, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.layer_mul = layer_mul
        
        # Define the MLP with BatchNorm
        self.MLP = nn.Sequential(
            nn.Linear((entity_dim + relation_dim), (entity_dim + relation_dim) * self.layer_mul),
            nn.LeakyReLU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((entity_dim + relation_dim) * self.layer_mul, hidden_dim),
            nn.BatchNorm1d(hidden_dim)  # Batch normalization on hidden_dim dimension
        )
    
    def forward(self, eh, er):
        if eh.shape[1]==er.shape[1]:
            combined = torch.cat((eh, er), dim=2)  # Shape: [batch, 1, entity_dim + relation_dim]
        else:                                      # 为predict_augment所保留的代码
            er = er.expand(-1, eh.shape[1], -1)
            combined = torch.cat((eh, er), dim=2)
        batch_size, seq_len, _ = combined.size()
        combined = combined.view(batch_size * seq_len, -1)
        output = self.MLP(combined)
        output = output.view(batch_size, seq_len, self.hidden_dim)
        return output


class Local_BN(nn.Module):
    def __init__(self, input_dim=32):
        super(Local_BN, self).__init__()
        self.input_dim=input_dim
        self.BatchNorm = nn.BatchNorm1d(input_dim)
    
    def forward(self, t):
        batch_size, seq_len, _ = t.size()
        t = t.view(batch_size * seq_len, -1)
        t = self.BatchNorm(t)
        t = t.view(batch_size, seq_len, -1)
        return t


class Local_Constant(nn.Module):
    def __init__(self, ):
        super(Local_Constant, self).__init__()
    
    def forward(self, t):
        return t


class SCCF_Reverse(nn.Module):
    def __init__(self, args, teacher_embedding_dim=512, teacher_temp=0.5, pretrain_path=None):
        super(SCCF_Reverse, self).__init__()
        self.args = args
        self.data_type = torch.double if self.args.data_type == 'double' else torch.float
        self.teacher_embedding_dim = teacher_embedding_dim
        self.entity_embedding_dim = teacher_embedding_dim 
        self.relation_embedding_dim = teacher_embedding_dim
        
        self.combine_hr = Local_Combine_hr(entity_dim=self.entity_embedding_dim, relation_dim=self.relation_embedding_dim, hidden_dim=self.entity_embedding_dim, layer_mul=2)
        self.tail_transform = Local_BN(input_dim=self.entity_embedding_dim)
        self.temperature = teacher_temp # 这个应该是教师得temprature
        
        self.init_parameter(pretrain_path)
    
    def init_parameter(self, pretrain_path):
        # 加载预训练模型的 checkpoint
        pretrain_model = torch.load(os.path.join(pretrain_path, 'checkpoint'), map_location='cpu')

        # 初始化 Local_Combine_hr 中的参数
        self.combine_hr.MLP[0].weight = nn.Parameter(pretrain_model['model_state_dict']['decoder.combine_hr.MLP.0.weight'].cpu().to(self.data_type), requires_grad=False)
        self.combine_hr.MLP[0].bias = nn.Parameter(pretrain_model['model_state_dict']['decoder.combine_hr.MLP.0.bias'].cpu().to(self.data_type), requires_grad=False)
        
        self.combine_hr.MLP[2].weight = nn.Parameter(pretrain_model['model_state_dict']['decoder.combine_hr.MLP.2.weight'].cpu().to(self.data_type), requires_grad=False)
        self.combine_hr.MLP[2].bias = nn.Parameter(pretrain_model['model_state_dict']['decoder.combine_hr.MLP.2.bias'].cpu().to(self.data_type), requires_grad=False)
        
        self.combine_hr.MLP[3].weight = nn.Parameter(pretrain_model['model_state_dict']['decoder.combine_hr.MLP.3.weight'].cpu().to(self.data_type), requires_grad=False)
        self.combine_hr.MLP[3].bias = nn.Parameter(pretrain_model['model_state_dict']['decoder.combine_hr.MLP.3.bias'].cpu().to(self.data_type), requires_grad=False)
        
        self.combine_hr.MLP[3].running_mean = nn.Parameter(pretrain_model['model_state_dict']['decoder.combine_hr.MLP.3.running_mean'].cpu().to(self.data_type), requires_grad=False)
        self.combine_hr.MLP[3].running_var = nn.Parameter(pretrain_model['model_state_dict']['decoder.combine_hr.MLP.3.running_var'].cpu().to(self.data_type), requires_grad=False)
        # self.combine_hr.MLP[3].num_batches_tracked = nn.Parameter(pretrain_model['model_state_dict']['decoder.combine_hr.MLP.3.num_batches_tracked'].cpu().to(self.data_type), requires_grad=False)

        # 初始化 Local_BN 中的参数
        self.tail_transform.BatchNorm.weight = nn.Parameter(pretrain_model['model_state_dict']['decoder.tail_transform.BatchNorm.weight'].cpu().to(self.data_type), requires_grad=False)
        self.tail_transform.BatchNorm.bias = nn.Parameter(pretrain_model['model_state_dict']['decoder.tail_transform.BatchNorm.bias'].cpu().to(self.data_type), requires_grad=False)
        
        self.tail_transform.BatchNorm.running_mean = nn.Parameter(pretrain_model['model_state_dict']['decoder.tail_transform.BatchNorm.running_mean'].cpu().to(self.data_type), requires_grad=False)
        self.tail_transform.BatchNorm.running_var = nn.Parameter(pretrain_model['model_state_dict']['decoder.tail_transform.BatchNorm.running_var'].cpu().to(self.data_type), requires_grad=False)
        # self.tail_transform.BatchNorm.num_batches_tracked = nn.Parameter(pretrain_model['model_state_dict']['decoder.tail_transform.BatchNorm.num_batches_tracked'].cpu().to(self.data_type), requires_grad=False)

    
    def forward(self, head, relation, tail):
        ehr = self.combine_hr(head, relation)
        et = self.tail_transform(tail)
        
        dot_product = torch.sum(ehr * et, dim=-1)
        norm_product = torch.norm(ehr, p=2, dim=-1) * torch.norm(et, p=2, dim=-1)
        sim = torch.exp(dot_product / (self.temperature * norm_product)) + torch.exp((dot_product / norm_product) ** 3 / self.temperature)
        return sim