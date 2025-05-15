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

from LFD_weight_learner import *
from LFD_fusion import *
from LFD_norm import *
from LFD_soft_loss import *
from LFD_similarity import *


'''
最简单的模仿单个教师模型的蒸馏算法
'''
class Imitation_SingleTeacher(nn.Module):
    def __init__(self, args, stu_score_preprocess='constant', tea_score_preprocess='constant', distill_func='KL_divergency'):
        super(Imitation_SingleTeacher, self).__init__()
        self.args = args
        self.stu_score_preprocess = globals()[stu_score_preprocess]
        self.tea_score_preprocess = globals()[tea_score_preprocess]
        
        self.distill_func = globals()[distill_func](self.args)
    
    def forward(self, stu_dis, tea_dis, prefix=''):
        stu_dis, _, _ = self.stu_score_preprocess(stu_dis)
        tea_dis, _, _ = self.tea_score_preprocess(tea_dis)
        
        if type(self.distill_func).__name__ in ['Margin_HuberLoss', 'HuberLoss', 'KL_divergency', 'KL_divergencyv2', 'DistillKL_Logit_Standard', 'JS_divergence', 'Pearson_loss']:
            loss = self.distill_func(stu_dis, tea_dis)
        else:
            raise ValueError(f"Unsupported distill_func: {self.distill_func.__name__}")
        
        loss_record = {
            'soft_loss'+prefix: loss.item(),
        }
        
        return loss, loss_record


'''
把学生模型的分数映射到教师模型的分数分布上去
'''
class Imitation_SingleTeacher1_5(nn.Module):
    def __init__(self, args, stu_score_preprocess='constant', tea_score_preprocess='constant', distill_func='KL_divergency'):
        super(Imitation_SingleTeacher1_5, self).__init__()
        self.args = args
        self.stu_score_preprocess = globals()[stu_score_preprocess]
        self.tea_score_preprocess = globals()[tea_score_preprocess]
        
        self.distill_func = globals()[distill_func](self.args)
        
        self.eps = 1e-6
    
    
    def match_via_standardization(self, stu_score, tea_score):
        """
        通过行标准化和逆标准化将stu_score映射到tea_score的分布。
        这个过程是可微分的。

        参数：
        stu_score: Tensor, shape = [batch, nneg+1]
        tea_score: Tensor, shape = [batch, nneg+1]

        返回：
        Tensor，形状与输入相同，stu_score经过映射后的结果。
        """
        assert stu_score.shape == tea_score.shape, "两个输入的shape必须一致"
        assert stu_score.ndim == 2, "输入应为2维 [batch, num_scores]"

        # 1. 标准化 stu_score (行操作)
        #    计算每行的均值和标准差
        stu_mean = torch.mean(stu_score, dim=1, keepdim=True)
        stu_std = torch.std(stu_score, dim=1, keepdim=True)
        # 标准化：z = (x - mu) / (std + eps)，保证可微分
        stu_standardized = (stu_score - stu_mean) / (stu_std + self.eps)

        # 2. 使用 tea_score 的统计量进行逆标准化
        #    计算每行的均值和标准差
        tea_mean = torch.mean(tea_score, dim=1, keepdim=True)
        tea_std = torch.std(tea_score, dim=1, keepdim=True)
        # 逆标准化：y = z * tea_std + tea_mean
        # 使用 detach()，因为我们通常不希望通过教师的统计量反向传播梯度到教师模型
        # (如果教师模型是固定的或单独训练的)
        # 如果教师模型也需要同时训练，则不应 detach
        # 假设教师是固定的:
        matched_score = stu_standardized * tea_std.detach() + tea_mean.detach()
        # 如果教师也需要训练:
        # matched_score = stu_standardized * tea_std + tea_mean

        return matched_score
        
        
    def forward(self, stu_dis, tea_dis, prefix=''):
        
        stu_dis = self.match_via_standardization(stu_dis, tea_dis)
        
        stu_dis, _, _ = self.stu_score_preprocess(stu_dis)
        tea_dis, _, _ = self.tea_score_preprocess(tea_dis)
        
        if type(self.distill_func).__name__ in ['Margin_HuberLoss', 'HuberLoss', 'KL_divergency', 'KL_divergencyv2', 'DistillKL_Logit_Standard', 'JS_divergence', 'Pearson_loss']:
            loss = self.distill_func(stu_dis, tea_dis)
        else:
            raise ValueError(f"Unsupported distill_func: {self.distill_func.__name__}")
        
        loss_record = {
            'soft_loss'+prefix: loss.item(),
        }
        
        return loss, loss_record



'''

同时蒸馏all_soft_loss 和 negative_soft_loss

'''
class Imitation_SingleTeacher2(nn.Module):
    def __init__(self, args, stu_score_preprocess='constant', tea_score_preprocess='constant', distill_func='KL_divergency'):
        super(Imitation_SingleTeacher2, self).__init__()
        self.args = args
        self.stu_score_preprocess = globals()[stu_score_preprocess]
        self.tea_score_preprocess = globals()[tea_score_preprocess]
        
        self.distill_func = globals()[distill_func](self.args)
        
    def forward(self, stu_dis, tea_dis, prefix=''):
        
        # step1 : 全分数蒸馏
        stu_dis, _, _ = self.stu_score_preprocess(stu_dis)
        tea_dis, _, _ = self.tea_score_preprocess(tea_dis)
        
        if type(self.distill_func).__name__ in ['Margin_HuberLoss', 'HuberLoss', 'KL_divergency', 'KL_divergencyv2', 'DistillKL_Logit_Standard', 'JS_divergence', 'Pearson_loss']:
            all_loss = self.distill_func(stu_dis, tea_dis)
        else:
            raise ValueError(f"Unsupported distill_func: {self.distill_func.__name__}")
        
        loss_record = {
            'all_soft_loss'+prefix: all_loss.item(),
        }
        
        # step2: 只蒸馏negative
        stu_dis, tea_dis = stu_dis[:, 1:], tea_dis[:, 1:]
        
        stu_dis, _, _ = self.stu_score_preprocess(stu_dis)
        tea_dis, _, _ = self.tea_score_preprocess(tea_dis)
        
        if type(self.distill_func).__name__ in ['Margin_HuberLoss', 'HuberLoss', 'KL_divergency', 'KL_divergencyv2', 'DistillKL_Logit_Standard', 'JS_divergence']:
            negative_loss = self.distill_func(stu_dis, tea_dis)
        else:
            raise ValueError(f"Unsupported distill_func: {self.distill_func.__name__}")
        
        loss_record.update({
            'negative_soft_loss'+prefix: negative_loss.item(),
        })
        
        loss = all_loss + negative_loss
        loss_record.update({
            'soft_loss'+prefix: loss.item(),
        })
        
        return loss, loss_record



'''

蒸馏用的负样本个数可以自己选

'''
class Imitation_SingleTeacher3(nn.Module):
    def __init__(self, args, stu_score_preprocess='constant', tea_score_preprocess='constant', distill_func='KL_divergency'):
        super(Imitation_SingleTeacher3, self).__init__()
        self.args = args
        self.random_sample_count = args.distill_sample_size
        self.stu_score_preprocess = globals()[stu_score_preprocess]
        self.tea_score_preprocess = globals()[tea_score_preprocess]
        
        self.distill_func = globals()[distill_func](self.args)
        
    def forward(self, stu_dis, tea_dis, prefix=''):
            # 1. Preprocess scores ONCE at the beginning
            stu_scores_processed, _, _ = self.stu_score_preprocess(stu_dis)
            tea_scores_processed, _, _ = self.tea_score_preprocess(tea_dis)

            batch_size, num_candidates = stu_scores_processed.shape
            device = stu_scores_processed.device

            # Ensure num_candidates is at least 1
            if num_candidates == 0:
                raise ValueError("num_candidates cannot be zero.")
            elif num_candidates == 1:
                positive_indices = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
                final_indices = positive_indices
                
                stu_dis_selected = torch.gather(stu_scores_processed, 1, final_indices)
                tea_dis_selected = torch.gather(tea_scores_processed, 1, final_indices)

            else:
                # Determine k and m for negative sampling
                k = min(self.args.pre_sample_size, num_candidates - 1)
                num_available_random = max(0, num_candidates - 1 - k)
                m = min(self.random_sample_count, num_available_random)

                # --- Positive Index ---
                positive_indices = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

                # --- Top-k Negative Indices ---
                top_k_indices = torch.empty((batch_size, 0), dtype=torch.long, device=device)
                top_k_relative_indices = torch.empty((batch_size, 0), dtype=torch.long, device=device) # For mask creation
                if k > 0:
                    _, top_k_relative_indices = torch.topk(tea_scores_processed[:, 1:], k, dim=1)
                    top_k_indices = top_k_relative_indices + 1 # Convert to absolute indices

                # --- Random Negative Indices (Vectorized) ---
                random_indices = torch.empty((batch_size, 0), dtype=torch.long, device=device)
                if m > 0:
                    neg_candidate_mask = torch.ones(batch_size, num_candidates - 1, dtype=torch.bool, device=device)

                    # Mark top-k relative indices as invalid for random sampling
                    if k > 0:
                        # scatter_(dim, index, src) -> marks positions specified by index with src (False)
                        neg_candidate_mask.scatter_(1, top_k_relative_indices, False)

                    random_scores = torch.rand(batch_size, num_candidates - 1, device=device)
                    random_scores[~neg_candidate_mask] = -1.0 # Mask out non-candidates

                    _, random_relative_indices = torch.topk(random_scores, m, dim=1)
                    random_indices = random_relative_indices + 1 # Convert to absolute indices


                # --- Combine Indices ---
                final_indices = torch.cat([positive_indices, top_k_indices, random_indices], dim=1)

                # 2. Gather the selected scores using the PREPROCESSED tensors
                stu_dis_selected = torch.gather(stu_scores_processed, 1, final_indices)
                tea_dis_selected = torch.gather(tea_scores_processed, 1, final_indices)


            # 3. Calculate Loss (using selected and preprocessed scores)
            # No need for the second preprocessing call here
            if type(self.distill_func).__name__ in ['Margin_HuberLoss', 'HuberLoss', 'KL_divergency', 'KL_divergencyv2', 'DistillKL_Logit_Standard', 'JS_divergence', 'Pearson_loss']:
                loss = self.distill_func(stu_dis_selected, tea_dis_selected) # Pass selected tensors
            else:
                raise ValueError(f"Unsupported distill_func: {type(self.distill_func).__name__}") # Use type() for class name

            loss_record = {
                'soft_loss'+prefix: loss.item(),
            }

            return loss, loss_record


class Imitation_DualTeacherv1(nn.Module):
    def __init__(self, args, stu_score_preprocess='constant', tea_score_preprocess='constant', distill_func='KL_divergency',  fusion_scores_loss='SigmoidLoss'):
        super(Imitation_DualTeacherv1, self).__init__()
        self.args = args
        self.fusion_scores_loss_name = fusion_scores_loss
        
        self.stu_score_preprocess = globals()[stu_score_preprocess] # 学生分数预处理
        self.tea_score_preprocess = globals()[tea_score_preprocess] # 教师分数预处理
        self.distill_func = globals()[distill_func](self.args) # 学生-教师的软损失函数
        
        if fusion_scores_loss in ['SigmoidLoss', 'SigmoidLoss2']:
            self.fusion_scores_loss_name = fusion_scores_loss
            self.fusion_scores_loss = globals()[fusion_scores_loss](self.args) # 融合分数损失函数
         
        self.teacher_score_fusion = teacher_score_fusion(args=args) # 教师分数融合
    
    
    def _cal_teacher_loss(self, fusion_score, tea_dis1, tea_dis2):
        if type(self.distill_func).__name__ in ['Margin_HuberLoss', 'HuberLoss', 'KL_divergency', 'KL_divergencyv2','DistillKL_Logit_Standard', 'JS_divergence', 'Pearson_loss']:
            fusion_score, _, _ = self.stu_score_preprocess(fusion_score)
            tea_dis1, _, _ = self.tea_score_preprocess(tea_dis1)
            tea_dis2, _, _ = self.tea_score_preprocess(tea_dis2)
            
            soft_loss1 = self.distill_func(fusion_score, tea_dis1)
            soft_loss2 = self.distill_func(fusion_score, tea_dis2)
            
            return soft_loss1 + soft_loss2
    
    
    def forward(self, eh, er, et, ehr_, et_, stu_dis, tea_dis1, tea_dis2, prefix='', weight=None, data=None):
        tea_dis, weight = self.teacher_score_fusion(eh, er, et, ehr_, et_, stu_dis, tea_dis1, tea_dis2, weight=weight, data=data)
        
        if self.fusion_scores_loss_name in ['SigmoidLoss', 'SigmoidLoss2']:
            fusion_score_loss, fusion_score_loss_record = self.fusion_scores_loss(tea_dis, None, big_better=True)
        else:
            fusion_score_loss = self._cal_teacher_loss(stu_dis, tea_dis1, tea_dis2)
        
        stu_dis, _, _ = self.stu_score_preprocess(stu_dis)
        tea_dis, _, _ = self.tea_score_preprocess(tea_dis)


        if type(self.distill_func).__name__ in ['Margin_HuberLoss', 'HuberLoss', 'KL_divergency', 'KL_divergencyv2','DistillKL_Logit_Standard', 'JS_divergence', 'Pearson_loss']:
            soft_loss = self.distill_func(stu_dis, tea_dis)
        else:
            raise ValueError(f"Unsupported distill_func: {self.distill_func.__name__}")
        
        loss = soft_loss + self.args.fusion_score_loss_weight * fusion_score_loss
        
        loss_record = {
            'soft_loss_'+prefix: soft_loss.item(),
            'fusion_score_loss_record'+prefix: fusion_score_loss.item(),
            'loss_record_'+prefix: loss.item()
        }
        
        return loss, loss_record, weight


