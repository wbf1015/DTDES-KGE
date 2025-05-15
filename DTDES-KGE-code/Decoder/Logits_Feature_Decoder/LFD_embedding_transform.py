import sys
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class TransformerEncoderLayer(nn.Module):
    """
    Implements a standard Transformer Encoder Layer as described in
    "Attention Is All You Need".

    Consists of two sub-layers:
    1. Multi-Head Self-Attention followed by Add & Norm.
    2. Position-wise Feed-Forward Network followed by Add & Norm.
    """
    def __init__(self, embedding_dim, num_heads, dropout):
        super(TransformerEncoderLayer, self).__init__()
                
        # --- Sub-layer 1: Multi-Head Self-Attention ---
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
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
        """
        Forward pass for the Transformer Encoder Layer.

        Args:
            query (torch.Tensor): Input tensor (or query for cross-attention).
                                  Shape: (batch_size, sequence_length, embedding_dim)
            key (torch.Tensor, optional): Key tensor for attention. Defaults to query.
                                          Shape: (batch_size, key_sequence_length, embedding_dim)
            value (torch.Tensor, optional): Value tensor for attention. Defaults to query.
                                            Shape: (batch_size, value_sequence_length, embedding_dim)
            src_mask (torch.Tensor, optional): Mask to prevent attention to certain positions
                                               (e.g., future tokens in decoder). Shape: (sequence_length, sequence_length) or others broadcastable.
            src_key_padding_mask (torch.Tensor, optional): Mask to indicate padding tokens in the key sequence.
                                                           Shape: (batch_size, key_sequence_length)

        Returns:
            torch.Tensor: Output tensor of the layer.
                          Shape: (batch_size, sequence_length, embedding_dim)
        """
        
        # Default to self-attention if key and value not provided
        if key is None and value is None:
            key = value = query        
        
        # 1. Multi-Head Attention sub-layer
        # Pass query, key, value, and optional masks to MultiheadAttention
        attn_output ,_ = self.attn(
            query, key, value,
            attn_mask=src_mask,            # Mask for preventing attention to certain positions
            key_padding_mask=src_key_padding_mask, # Mask for ignoring padding tokens
            need_weights=False             # Usually we don't need weights outside the layer
        )

        # 2. Add & Norm (after Attention)
        # Apply dropout to the attention output, add the residual connection (input query),
        # and then apply Layer Normalization.
        # x becomes the input for the next sub-layer
        x = self.norm1(query + self.dropout1(attn_output))

        # 3. Feed Forward sub-layer
        ff_output = self.feedforward(x)

        # 4. Add & Norm (after Feed Forward)
        # Apply dropout to the feedforward output, add the residual connection (output from previous Add & Norm),
        # and then apply Layer Normalization.
        out = self.norm2(x + self.dropout2(ff_output))

        return out


"""

=========================================================嵌入融合变换模块============================================

"""     
    
class Combine_hr(nn.Module):
    def __init__(self, entity_dim=512, relation_dim=512, hidden_dim=32, layer_mul=2):
        super(Combine_hr, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.layer_mul = layer_mul
        
        # Define the MLP with BatchNorm
        self.MLP = nn.Sequential(
            nn.Linear((entity_dim + relation_dim), (entity_dim + relation_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
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


class Combine_hr2(nn.Module):
    def __init__(self, entity_dim=512, relation_dim=512, hidden_dim=32, layer_mul=2):
        super(Combine_hr2, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.layer_mul = layer_mul
        
        # Define the MLP with BatchNorm
        self.MLP = nn.Sequential(
            nn.Linear((entity_dim + relation_dim), (entity_dim + relation_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((entity_dim + relation_dim) * self.layer_mul, hidden_dim),
            nn.LayerNorm(hidden_dim)  # Batch normalization on hidden_dim dimension
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


class Combine_hr3(nn.Module):
    def __init__(self, entity_dim=512, relation_dim=512, hidden_dim=32, layer_mul=2):
        super(Combine_hr3, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.layer_mul = layer_mul
        
        # Define the MLP with BatchNorm
        self.MLP = nn.Sequential(
            nn.Linear((entity_dim + relation_dim), (entity_dim + relation_dim) * self.layer_mul),
            nn.GELU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((entity_dim + relation_dim) * self.layer_mul, hidden_dim),
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


class generate_query(nn.Module):
    def __init__(self, entity_dim=512, relation_dim=512, hidden_dim=32, layer_mul=2):
        super(generate_query, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.layer_mul = layer_mul
        
        assert self.entity_dim == self.relation_dim and self.relation_dim == self.hidden_dim
        self.self_attn = TransformerEncoderLayer(entity_dim, 4, 0.4)
    
    def forward(self, eh, er):
        inputs = torch.cat((eh,er), dim=1)
        outputs = self.self_attn(inputs)
        outputs = outputs.mean(dim=1).unsqueeze(dim=1)
        
        return outputs


class BN(nn.Module):
    def __init__(self, input_dim=32):
        super(BN, self).__init__()
        self.input_dim=input_dim
        self.BatchNorm = nn.BatchNorm1d(input_dim)
    
    def forward(self, t):
        batch_size, seq_len, _ = t.size()
        t = t.view(batch_size * seq_len, -1)
        t = self.BatchNorm(t)
        t = t.view(batch_size, seq_len, -1)
        return t


class LN(nn.Module):
    """
    Layer Norm over the last (feature) dimension of a tensor shaped
    (batch_size, seq_len, input_dim), mirroring the style of the BN example.
    """
    def __init__(self, input_dim: int = 32, eps: float = 1e-5, affine: bool = True):
        super(LN, self).__init__()
        self.input_dim = input_dim
        # LayerNorm operates along the last dimension by default
        self.layer_norm = nn.LayerNorm(normalized_shape=input_dim,
                                       eps=eps,
                                       elementwise_affine=affine)

    def forward(self, x):
        return self.layer_norm(x)



class Constant(nn.Module):
    def __init__(self, ):
        super(Constant, self).__init__()
    
    def forward(self, t):
        return t


class generate_tail(nn.Module):
    def __init__(self, input_dim=32):
        super(generate_tail, self).__init__()
        self.input_dim=input_dim
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4), # Expansion layer
            nn.GELU(),                                   # Activation (GELU is common, ReLU also used)
            nn.Dropout(0.2),                         # Dropout within FFN
            nn.Linear(input_dim * 4, input_dim)  # Projection back to embedding_dim
        )
        self.norm = LN(input_dim)
    
    def forward(self, t):
        t = self.MLP(t)
        t = self.norm(t)
        return t