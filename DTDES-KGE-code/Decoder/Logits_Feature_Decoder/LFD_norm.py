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
sys.path.append(new_directory)

"""

================================================归一化函数===================================================

"""
def global_standardize(scores, eps=1e-6):
    mean = scores.mean()
    std = scores.std()
    standardized_tensor = (scores - mean) / (std + eps)
    return standardized_tensor, mean, std

def inverse_global_standardize(standardized_tensor, mean, std, eps=1e-6):
    original_tensor = standardized_tensor * (std + eps) + mean
    return original_tensor

def local_standardize(scores, eps=1e-6):
    scores_mean = scores.mean(dim=-1, keepdim=True)
    scores_sqrtvar = torch.sqrt(scores.var(dim=-1, keepdim=True) + eps)
    scores_norm = (scores - scores_mean) / scores_sqrtvar
    return scores_norm, scores_mean, scores_sqrtvar

def inverse_local_standardize(standardized_tensor, scores_mean, scores_sqrtvar):
    original_tensor = standardized_tensor * scores_sqrtvar + scores_mean
    return original_tensor

def global_minmax(scores, eps=1e-6):
    scores_max = scores.max()
    scores_min = scores.min()
    scores_norm = (scores - scores_min) / (scores_max - scores_min + eps)
    return scores_norm, scores_max, scores_min

def reverse_global_minmax(scores_norm, scores_max, scores_min, eps=1e-6):
    scores = scores_norm * (scores_max - scores_min + eps) + scores_min
    return scores

def local_minmax(scores, eps = 1e-6):
    scores_max, _ = scores.max(dim=-1, keepdim=True)  # [batch, 1]
    scores_min, _ = scores.min(dim=-1, keepdim=True)  # [batch, 1]
    scores_norm = (scores - scores_min) / (scores_max - scores_min + eps)  # Min-Max归一化
    return scores_norm, scores_max, scores_min

def reverse_local_minmax(scores_norm, scores_max, scores_min, eps=1e-6):
    scores = scores_norm * (scores_max - scores_min + eps) + scores_min
    return scores

# 调整方差，但是保持均值不变
def adjust_var(tensor, k):
    """
    调整张量的方差，同时保持均值不变
    
    参数:
        tensor: 需要调整的张量
        k: 方差调整因子，k>1增加方差，k<1减小方差
    
    返回:
        adjusted_tensor: 调整后的张量
        k: 调整因子
    """
    mean = tensor.mean(dim=1, keepdim=True)
    adjusted_tensor = mean + (tensor - mean) * torch.sqrt(k)
    
    return adjusted_tensor, k

def reverse_adjust_var(adjusted_tensor, k):
    """
    将adjust_var处理过的张量映射回原始状态
    
    参数:
        adjusted_tensor: adjust_var处理后的张量
        k: 原始adjust_var使用的调整因子
    
    返回:
        original_tensor: 映射回原始状态的张量
    """
    mean = adjusted_tensor.mean(dim=1, keepdim=True)
    original_tensor = mean + (adjusted_tensor - mean) / torch.sqrt(k)
    
    return original_tensor

def constant(scores):
    return scores, None, None



"""CVPR 2024"""
def logits_normalize(logits):
    """
    对logits进行归一化处理
    
    参数:
        logits: 需要归一化的logits张量
    
    返回:
        normalized_logits: 归一化后的logits
        mean: 均值
        std: 标准差
    """
    mean = logits.mean(dim=-1, keepdim=True)
    std = logits.std(dim=-1, keepdim=True)
    normalized_logits = (logits - mean) / (std + 1e-6)
    return normalized_logits, mean, std

def reverse_logits_normalize(normalized_logits, mean, std):
    """
    将logits_normalize处理过的张量映射回原始状态
    
    参数:
        normalized_logits: logits_normalize处理后的张量
        mean: 原始logits的均值
        std: 原始logits的标准差
    
    返回:
        original_logits: 映射回原始状态的logits
    """
    original_logits = normalized_logits * (std + 1e-6) + mean
    return original_logits