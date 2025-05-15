import sys
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from torch.backends.cuda import sdp_kernel, SDPBackend

"""

========================================================相似度计算函数============================================

"""


class cal_similarity(nn.Module):
    def __init__(self, args, temperature):
        super(cal_similarity, self).__init__()
        self.temperature = temperature
        self.args = args
        
        self.target_dim = args.target_dim
        self.entity_dim = args.target_dim * args.entity_mul
        self.relation_dim = args.target_dim * args.relation_mul
        self.layer_mul = 2
        
        self.query_distance = nn.Sequential(
            nn.Linear((self.entity_dim + self.relation_dim), (self.entity_dim + self.relation_dim) * self.layer_mul),
            nn.LeakyReLU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.entity_dim + self.relation_dim) * self.layer_mul, 1),
        )
        
        self.tail_distance = nn.Sequential(
            nn.Linear((self.entity_dim + self.entity_dim), (self.entity_dim + self.entity_dim) * self.layer_mul),
            nn.LeakyReLU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.entity_dim + self.entity_dim) * self.layer_mul, 1),
        )
        
    def cosine_similarity(self, ehr, et):
        if ehr.shape[1] < et.shape[1]: 
            ehr = ehr.expand(-1, et.shape[1], -1)
        else:
            et = et.expand(-1, ehr.shape[1], -1)
        sim = F.cosine_similarity(ehr, et, dim=-1)
        return sim
    
    def SCCF_similarity1(self, ehr, et):
        dot_product = torch.sum(ehr * et, dim=-1)
        norm_product = torch.norm(ehr, p=2, dim=-1) * torch.norm(et, p=2, dim=-1)
        sim = torch.exp(dot_product / (self.temperature * norm_product)) + torch.exp((dot_product / norm_product) ** 3 / self.temperature) - torch.exp((dot_product / norm_product) ** 3 / self.temperature)
        return sim
    
    def SCCF_similarity3(self, ehr, et):
        dot_product = torch.sum(ehr * et, dim=-1)
        norm_product = torch.norm(ehr, p=2, dim=-1) * torch.norm(et, p=2, dim=-1)
        sim = torch.exp(dot_product / (self.temperature * norm_product)) + torch.exp((dot_product / norm_product) ** 3 / self.temperature)
        return sim
    
    def similarity1(self, ehr, et):
        return self.cosine_similarity(ehr, et)
    
    def similarity3(self, ehr, et):
        return self.cosine_similarity(ehr, et) + (self.cosine_similarity(ehr, et))**3
    
    def norm_distance(self, ehr, et, norm=2):
        # ehr.shape=[batch,1,dim] et.shape=[batch,nneg+1,dim]
        ehr_norm = torch.norm(ehr, p=norm, dim=-1)  
        et_norm = torch.norm(et, p=norm, dim=-1)
        distance = torch.abs(ehr_norm - et_norm)
        distance = -1 * distance
        return distance
    
    def norm_distance2(self, eh, er, ehr, et):
        query_distance = self.query_distance(torch.cat((eh,er),dim=-1))
        ehr = ehr.expand(-1, et.size(1), -1)
        tail_distance = self.tail_distance(torch.cat((ehr, et), dim=-1))
        distance = query_distance - tail_distance
        distance = -1 * distance
        distance = distance.squeeze(dim=-1)
        return distance
    
    def RotatE(self, eh, er, et):
        pi = 3.14159265358979323846
        margin = self.args.stu_rotate_gamma
        embedding_dim = eh.shape[-1]//2
        embedding_range = margin + 2.0
        
        re_head, im_head = torch.chunk(eh, 2, dim=2)
        re_tail, im_tail = torch.chunk(et, 2, dim=2)
        
        #Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = er/(((embedding_range)/embedding_dim)/pi)
        
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail
        
        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)
        score = score.sum(dim = 2)
        score = margin - score
        
        return score

    def MRME_similarity(self, ehr, et):
        et = et.transpose(1, 2)
        sim = torch.bmm(ehr, et).squeeze(1)

        return sim
    
    
"""

用神经网络拟合

"""


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super(TransformerEncoderLayer, self).__init__()

        # --- Sub-layer 1: Multi-Head Self-Attention ---
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True, # Assume input format is (batch, seq_len, embed_dim)
            dropout=dropout   # Dropout within the attention mechanism itself
        )
        # Dropout layer after attention output, before residual connection
        self.dropout1 = nn.Dropout(dropout)
        # Layer Normalization after the first sub-layer (Attention + Add)
        self.norm1 = nn.LayerNorm(embedding_dim)

        # --- Sub-layer 2: Position-wise Feed-Forward Network ---
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4), # Expansion layer
            nn.GELU(),                                   # Activation (GELU is common, ReLU also used)
            nn.Dropout(dropout),                         # Dropout within FFN
            nn.Linear(embedding_dim * 4, embedding_dim)  # Projection back to embedding_dim
        )
        # Dropout layer after feedforward output, before residual connection
        self.dropout2 = nn.Dropout(dropout)
        # Layer Normalization after the second sub-layer (FFN + Add)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, query, key=None, value=None, src_mask=None, src_key_padding_mask=None):
        # Default to self-attention if key and value not provided
        if key is None and value is None:
            key = value = query
    
        # 1. Multi-Head Attention sub-layer
        # attn_output, _ = self.self_attn(
        #     query, key, value,
        # )
        
        with sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=True):
            try:
                attn_output, _ = self.self_attn(
                    query, key, value,
                    attn_mask=src_mask,
                    key_padding_mask=src_key_padding_mask,
                    need_weights=False
                )
            except RuntimeError as e:
                print(f"Error even with SDP kernels disabled: {e}")
                # 在这里可以加入更多调试信息，比如打印详细形状和 dtype
                print(f"Shapes: Q={query.shape}, K={key.shape}, V={value.shape}")
                print(f"Dtypes: Q={query.dtype}, K={key.dtype}, V={value.dtype}")
                if src_mask is not None: print(f"Attn Mask Shape: {src_mask.shape}, Dtype: {src_mask.dtype}")
                if src_key_padding_mask is not None: print(f"Padding Mask Shape: {src_key_padding_mask.shape}, Dtype: {src_key_padding_mask.dtype}")
                raise e

        # 2. Add & Norm (after Attention)
        x = self.norm1(query + self.dropout1(attn_output))

        # 3. Feed Forward sub-layer
        ff_output = self.feedforward(x)

        # 4. Add & Norm (after Feed Forward)
        out = self.norm2(x + self.dropout2(ff_output))

        return out


class NN_fitting(nn.Module):
    def __init__(self, args):
        super(NN_fitting, self).__init__()
        self.args = args

        assert self.args.entity_mul == self.args.relation_mul
        
        self.cls_token = nn.Parameter(torch.randn(self.args.target_dim * self.args.entity_mul))
        self.encoder = TransformerEncoderLayer(self.args.target_dim * self.args.entity_mul, self.args.heads, self.args.dropout)
        self.decoder = nn.Sequential(
            nn.Linear(self.args.target_dim * self.args.entity_mul, self.args.target_dim * self.args.entity_mul * 4), # Expansion layer
            nn.GELU(),                                   # Activation (GELU is common, ReLU also used)
            nn.Dropout(self.args.dropout),                         # Dropout within FFN
            nn.Linear(self.args.target_dim * self.args.entity_mul * 4, 1)  # Projection back to embedding_dim
        )
        
    def forward(self, eh, er, et):
        batch, nneg_plus, dim = et.shape 
        
        eh = eh.expand(batch, nneg_plus, dim)
        er = er.expand(batch, nneg_plus, dim)
        
        eh = eh.reshape(batch * nneg_plus, 1, dim)
        er = er.reshape(batch * nneg_plus, 1, dim)
        et = et.reshape(batch * nneg_plus, 1, dim)
        cls_token = self.cls_token.unsqueeze(0).unsqueeze(0).expand(batch * nneg_plus, 1, dim)
        
        inputs = torch.cat([cls_token, eh, er, et], dim=1)
        outputs = self.encoder(inputs)
        outputs = outputs[:, 0, :]
        
        logits = self.decoder(outputs)
        logits = logits.reshape(batch, nneg_plus)
        
        return logits
        