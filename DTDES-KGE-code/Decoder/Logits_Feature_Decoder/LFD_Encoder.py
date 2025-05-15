import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import  Parameter

class Easy_MLP(nn.Module):
    def __init__(self, args, input_dim=256, output_dim=64, layer_mul=2):
        super(Easy_MLP, self).__init__()
        self.args = args
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_mul = layer_mul
        
        self.MLP = nn.Sequential(
            nn.Linear((input_dim), int((input_dim) * self.layer_mul)),
            nn.LeakyReLU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear(int((input_dim) * self.layer_mul), output_dim),
            nn.BatchNorm1d(output_dim)  # Batch normalization on hidden_dim dimension
        )

    def forward(self, obj_embedding):
        batch, neg_sample, _ = obj_embedding.shape
        obj_embedding = obj_embedding.view(batch*neg_sample, -1)
        output = self.MLP(obj_embedding)
        output = output.view(batch, neg_sample, -1)
        return output
