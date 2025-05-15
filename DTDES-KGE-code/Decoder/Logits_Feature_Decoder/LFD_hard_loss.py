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

from LFD_similarity import *

sys.path.remove(new_directory)
'''

正负样本各占一半

'''
class SigmoidLoss(nn.Module):
    def __init__(self, args):
        super(SigmoidLoss, self).__init__()
        self.args = args
        self.pos_margin = nn.Parameter(torch.Tensor([self.args.pos_gamma]))
        self.neg_margin = nn.Parameter(torch.Tensor([self.args.neg_gamma]))
        self.pos_margin.requires_grad = False
        self.neg_margin.requires_grad = False
        if self.args.negative_adversarial_sampling:
            self.adv_temperature = nn.Parameter(torch.Tensor([self.args.adversarial_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False
    
    def forward(self, similarity, subsampling_weight=None, small_better=False, big_better=False, sub=False, add=False):
        p_score, n_score = similarity[:, 0], similarity[:, 1:]
        if small_better:
            p_score, n_score = self.pos_margin-p_score, self.neg_margin-n_score
        if big_better:
            p_score, n_score = p_score-self.pos_margin, n_score-self.neg_margin
        if sub:
            p_score, n_score = self.pos_margin-p_score, self.neg_margin-n_score
        if add: # big better也可以走这个
            p_score, n_score = self.pos_margin+p_score, self.neg_margin+n_score
        if self.adv_flag:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(n_score * self.adv_temperature, dim = 1).detach()
                            * F.logsigmoid(-n_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-n_score).mean(dim = 1)
            
        positive_score = F.logsigmoid(p_score)
        
        if self.args.subsampling == False:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()
        
        loss = (positive_sample_loss + negative_sample_loss)/2
        loss_record = {
            'Sigmoid_hard_positive_sample_loss': positive_sample_loss.item(),
            'Sigmoid_hard_negative_sample_loss': negative_sample_loss.item(),
            'Sigmoid_hard_loss': loss.item(),
        }
        return loss, loss_record


'''

按个数比例调整权重

'''
class SigmoidLoss2(nn.Module):
    def __init__(self, args, pos_margin=None, neg_margin=None):
        super(SigmoidLoss2, self).__init__()
        self.args = args
        if (pos_margin is None) and (neg_margin is None):
            self.pos_margin = nn.Parameter(torch.Tensor([self.args.pos_gamma]))
            self.neg_margin = nn.Parameter(torch.Tensor([self.args.neg_gamma]))
            self.pos_margin.requires_grad = False
            self.neg_margin.requires_grad = False
            self.prefix = ''
        else:
            self.pos_margin = nn.Parameter(torch.Tensor([pos_margin]))
            self.neg_margin = nn.Parameter(torch.Tensor([neg_margin]))
            self.pos_margin.requires_grad = False
            self.neg_margin.requires_grad = False
            self.prefix = 'FT_teacher_'

        if self.args.negative_adversarial_sampling:
            self.adv_temperature = nn.Parameter(torch.Tensor([self.args.adversarial_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False
    
    def forward(self, similarity, subsampling_weight=None, small_better=False, big_better=False, sub=False, add=False):
        p_score, n_score = similarity[:, 0], similarity[:, 1:]
        n_pos, n_neg = 1, similarity.shape[-1]-1
        if small_better:
            p_score, n_score = self.pos_margin-p_score, self.neg_margin-n_score
        if big_better:
            p_score, n_score = p_score-self.pos_margin, n_score-self.neg_margin
        if sub:
            p_score, n_score = self.pos_margin-p_score, self.neg_margin-n_score
        if add: # big better也可以走这个
            p_score, n_score = self.pos_margin+p_score, self.neg_margin+n_score
        if self.adv_flag:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(n_score * self.adv_temperature, dim = 1).detach()
                            * F.logsigmoid(-n_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-n_score).mean(dim = 1)
            
        positive_score = F.logsigmoid(p_score)
        
        if self.args.subsampling == False:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()
        
        if self.prefix=='':
            loss = (n_pos/(n_pos+n_neg)) * positive_sample_loss + (n_neg/(n_pos+n_neg)) * negative_sample_loss
        else:
            loss = (positive_sample_loss+negative_sample_loss)/2
        
        loss_record = {
            self.prefix + 'Sigmoid2_hard_positive_sample_loss': positive_sample_loss.item(),
            self.prefix + 'Sigmoid2_hard_negative_sample_loss': negative_sample_loss.item(),
            self.prefix + 'Sigmoid2_hard_loss': loss.item(),
        }
        return loss, loss_record


'''

自学习权重，平衡正负样本的loss

'''

class SigmoidLoss3(nn.Module):
    def __init__(self, args):
        super(SigmoidLoss3, self).__init__()
        self.args = args
        self.pos_margin = nn.Parameter(torch.Tensor([self.args.pos_gamma]))
        self.neg_margin = nn.Parameter(torch.Tensor([self.args.neg_gamma]))
        self.pos_margin.requires_grad = False
        self.neg_margin.requires_grad = False
        if self.args.negative_adversarial_sampling:
            self.adv_temperature = nn.Parameter(torch.Tensor([self.args.adversarial_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False
        

        self.loss_weight_param = nn.Parameter(torch.Tensor([0.0]))
        self.loss_weight_param.requires_grad = True

    
    def forward(self, similarity, subsampling_weight=None, small_better=False, big_better=False, sub=False, add=False):
        p_score, n_score = similarity[:, 0], similarity[:, 1:]
        if small_better:
            p_score, n_score = self.pos_margin-p_score, self.neg_margin-n_score
        if big_better:
            p_score, n_score = p_score-self.pos_margin, n_score-self.neg_margin
        if sub:
            p_score, n_score = self.pos_margin-p_score, self.neg_margin-n_score
        if add: # big better也可以走这个
            p_score, n_score = self.pos_margin+p_score, self.neg_margin+n_score
        if self.adv_flag:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(n_score * self.adv_temperature, dim = 1).detach()
                            * F.logsigmoid(-n_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-n_score).mean(dim = 1)
            
        positive_score = F.logsigmoid(p_score)
        
        if self.args.subsampling == False:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()
        

        pos_loss_weight = torch.sigmoid(self.loss_weight_param)
        neg_loss_weight = 1.0 - pos_loss_weight
        loss = pos_loss_weight * positive_sample_loss + neg_loss_weight * negative_sample_loss

        loss_record = {
            'Sigmoid_hard_positive_sample_loss': positive_sample_loss.item(),
            'Sigmoid_hard_negative_sample_loss': negative_sample_loss.item(),
            'Sigmoid_hard_loss': loss.item(),
        }
        return loss, loss_record




class BCELoss(nn.Module):
    def __init__(self, args):
        super(BCELoss, self).__init__()
        self.args = args
        self.bceloss = torch.nn.BCELoss()
        self.sigmoid = torch.nn.Sigmoid()
        self.label_smooth = 0.1
    
    def forward(self, similarity, subsampling_weight):
        target = torch.zeros_like(similarity)
        target[:, 1:] = self.label_smooth/(similarity.shape[-1]-1)
        target[:, 0] = 1-self.label_smooth
        similarity = self.sigmoid(similarity)
        loss = self.bceloss(similarity, target)
        loss_record = {
            'hard_loss': loss.item(),
        }
        return loss, loss_record



class BCELossWithSmoothing(nn.Module):
    def __init__(self, args, label_smooth=0.1, reduction='mean'):
        super().__init__()
        self.args = args
        if not 0.0 <= label_smooth < 1.0:
            raise ValueError("label_smooth value must be between 0.0 and 1.0 (exclusive of 1.0)")
        self.label_smooth = label_smooth
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.final_reduction = reduction

    def forward(self, logits, sample_weights=None):
        batch_size, num_classes = logits.shape
        num_negatives = num_classes - 1

        if num_negatives <= 0 and self.label_smooth > 0:
             print(f"Warning: Label smoothing ({self.label_smooth}) is applied, but there are no negative samples (num_classes={num_classes}). Smoothing will only affect the positive label.")
             # Handle edge case gracefully, maybe disable smoothing for negatives
             neg_smooth_val = 0.0
        elif num_negatives > 0 :
            neg_smooth_val = self.label_smooth / num_negatives
        else: # num_negatives == 0 and self.label_smooth == 0
            neg_smooth_val = 0.0 # No negatives and no smoothing needed

        target = torch.full_like(logits, neg_smooth_val)
        target[:, 0] = 1.0 - self.label_smooth
        element_wise_loss = self.bce_with_logits(logits, target) # Shape: [batch_size, num_classes]


        if sample_weights is not None:
            if sample_weights.shape[0] != batch_size:
                raise ValueError(f"sample_weights shape ({sample_weights.shape}) must match batch_size ({batch_size})")
            weights = sample_weights.unsqueeze(1).to(element_wise_loss.device)
            element_wise_loss = element_wise_loss * weights

        if self.final_reduction == 'mean':
            loss = element_wise_loss.mean()
        elif self.final_reduction == 'sum':
            loss = element_wise_loss.sum()
        elif self.final_reduction == 'none':
            loss = element_wise_loss # Return element-wise or sample-wise loss
        else:
             raise ValueError(f"Unsupported reduction type: {self.final_reduction}")

        loss_record = {
            # Use a more descriptive name if needed
            'hard_loss': loss.item(),
        }

        return loss, loss_record



class CrossEntropyloss(nn.Module):
    def __init__(self, args):
        super(CrossEntropyloss, self).__init__()
        self.args = args
        self.crossentropyloss = torch.nn.CrossEntropyLoss(reduction='mean')
        
    def forward(self, similarity, subsampling_weight):
        # The target should be a tensor of zeros (i.e., the index of the positive sample in each row)
        # Since the first element is the positive sample, the target is always 0 for each row.
        target = torch.zeros(similarity.size(0), dtype=torch.long, device=similarity.device) 
        
        # Compute CrossEntropyLoss
        loss = self.crossentropyloss(similarity, target)
        
        loss_record = {
            'hard_loss': loss.item(),
        }
        
        return loss, loss_record
