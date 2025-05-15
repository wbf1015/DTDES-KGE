import sys
import os

import torch.nn.functional as F
import torch
import torch.nn as nn
'''
允许读入两个预训练的模型，因为学习到的embedding是在另一个空间的，所以没关系，entity和relation都是从头学习的
'''
class KDManager_Reverse_2KGE(nn.Module):
    def __init__(self, args):
        super(KDManager_Reverse_2KGE, self).__init__()
        self.args = args
        self.data_type = torch.double if self.args.data_type == 'double' else torch.float
        pretrain_model = torch.load(os.path.join(self.args.pretrain_path, 'checkpoint'))

        # 一般我们默认第一个pretrain_path内容的是欧式空间的embedding
        if 'RotatE' in self.args.pretrain_path or 'SCCF' in self.args.pretrain_path:
            if 'entity_embedding' in pretrain_model['model_state_dict']:
                self.PT_entity_embedding = nn.Parameter(pretrain_model['model_state_dict']['entity_embedding'].cpu().to(self.data_type), requires_grad=False)
                self.PT_relation_embedding = nn.Parameter(pretrain_model['model_state_dict']['relation_embedding'].cpu().to(self.data_type), requires_grad=False)
            else:
                self.PT_entity_embedding = nn.Parameter(pretrain_model['model_state_dict']['EmbeddingManager.entity_embedding'].cpu().to(self.data_type), requires_grad=False)
                self.PT_relation_embedding = nn.Parameter(pretrain_model['model_state_dict']['EmbeddingManager.relation_embedding'].cpu().to(self.data_type), requires_grad=False)
        elif 'AttH' in self.args.pretrain_path:
            self.PT_entity_embedding = nn.Parameter(pretrain_model['model_state_dict']['EmbeddingManager.entity_embedding'].cpu().to(self.data_type), requires_grad=False)
            self.PT_relation_embedding = nn.Parameter(pretrain_model['model_state_dict']['EmbeddingManager.relation_embedding'].cpu().to(self.data_type), requires_grad=False)
        elif 'LorentzKG' in self.args.pretrain_path:
            self.PT_entity_embedding = None
            self.PT_relation_embedding = None
        elif 'MRME' in self.args.pretrain_path:
            self.PT_entity_embedding = None
            self.PT_relation_embedding = None
        elif 'HAKE' in self.args.pretrain_path:
            self.PT_entity_embedding = None
            self.PT_relation_embedding = None
        
        
        if 'RotatE' in self.args.pretrain_path2 or 'SCCF' in self.args.pretrain_path2:
            pretrain_model = torch.load(os.path.join(self.args.pretrain_path2, 'checkpoint'))
            if 'entity_embedding' in pretrain_model['model_state_dict']:
                self.PT_entity_embedding2 = nn.Parameter(pretrain_model['model_state_dict']['entity_embedding'].cpu().to(self.data_type), requires_grad=False)
                self.PT_relation_embedding2 = nn.Parameter(pretrain_model['model_state_dict']['relation_embedding'].cpu().to(self.data_type), requires_grad=False)
            else:
                self.PT_entity_embedding2 = nn.Parameter(pretrain_model['model_state_dict']['EmbeddingManager.entity_embedding'].cpu().to(self.data_type), requires_grad=False)
                self.PT_relation_embedding2 = nn.Parameter(pretrain_model['model_state_dict']['EmbeddingManager.relation_embedding'].cpu().to(self.data_type), requires_grad=False)
        elif 'LorentzKG' in self.args.pretrain_path2:
            self.PT_entity_embedding2 = None
            self.PT_relation_embedding2 = None
        elif 'MRME' in self.args.pretrain_path2:
            self.PT_entity_embedding2 = None
            self.PT_relation_embedding2 = None
        elif 'AttH' in self.args.pretrain_path2: 
            pretrain_model = torch.load(os.path.join(self.args.pretrain_path2, 'checkpoint'))
            self.PT_entity_embedding2 = nn.Parameter(pretrain_model['model_state_dict']['EmbeddingManager.entity_embedding'].cpu().to(self.data_type), requires_grad=False)
            self.PT_relation_embedding2 = nn.Parameter(pretrain_model['model_state_dict']['EmbeddingManager.relation_embedding'].cpu().to(self.data_type), requires_grad=False)
        elif 'HAKE' in self.args.pretrain_path2:
            self.PT_entity_embedding2 = None
            self.PT_relation_embedding2 = None

        self.entity_embedding = nn.Parameter(torch.empty(self.args.nentity, self.args.hidden_dim * self.args.entity_mul, dtype=self.data_type), requires_grad=True)
        self.relation_embedding = nn.Parameter(torch.empty(self.args.nrelation, self.args.hidden_dim * self.args.relation_mul, dtype=self.data_type), requires_grad=True)
        # self.entity_embedding = nn.Embedding(self.args.nentity, self.args.hidden_dim * self.args.entity_mul)
        # self.relation_embedding = nn.Embedding(self.args.nrelation, self.args.hidden_dim * self.args.relation_mul)
        
        # self.entity_embedding.weight.data *= 1e-3
        # self.relation_embedding.weight.data *= 1e-3
        
        nn.init.xavier_uniform_(self.entity_embedding)
        nn.init.xavier_uniform_(self.relation_embedding)

    
    def forward(self, sample):
        PT_head1, PT_tail1 = self.EntityEmbeddingExtract(self.PT_entity_embedding, sample)
        PT_head2, PT_tail2 = self.EntityEmbeddingExtract(self.PT_entity_embedding2, sample)
        head, tail = self.EntityEmbeddingExtract(self.entity_embedding, sample)
        
        PT_relation1 = self.RelationEmbeddingExtract(self.PT_relation_embedding, sample)
        PT_relation2 = self.RelationEmbeddingExtract(self.PT_relation_embedding2, sample)
        relation = self.RelationEmbeddingExtract(self.relation_embedding, sample)
        
        return head, relation, tail, PT_head1, PT_relation1, PT_tail1, PT_head2, PT_relation2, PT_tail2
    
    
    def EntityEmbeddingExtract2(self, entity_embedding, sample):
        if entity_embedding is None:
            return None, None
        
        positive, negative = sample
        batch_size, negative_sample_size = negative.size(0), negative.size(1)
        
        # 直接通过索引获取嵌入
        neg_tail = entity_embedding(negative.view(-1)).view(batch_size, negative_sample_size, -1)
        pos_tail = entity_embedding(positive[:, 2]).unsqueeze(1)
        
        tail = torch.cat((pos_tail, neg_tail), dim=1)
        
        head = entity_embedding(positive[:, 0]).unsqueeze(1)
            
        return head, tail

    def RelationEmbeddingExtract2(self, relation_embedding, sample):
        if relation_embedding is None:
            return None
        
        positive, negative = sample
        
        # 直接通过索引获取嵌入
        relation = relation_embedding(positive[:, 1]).unsqueeze(1)
        
        return relation

    
    
    '''
    
    如果你使用nn.Parameter，你就会用到下面这个函数
    
    '''
    def EntityEmbeddingExtract(self, entity_embedding, sample):
        
        if entity_embedding is None:
            return None, None
        
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
        
        if relation_embedding is None:
            return None
        
        positive, negative = sample
        
        relation = torch.index_select(
                relation_embedding, 
                dim=0, 
                index=positive[:, 1]
            ).unsqueeze(1)
        
        return relation