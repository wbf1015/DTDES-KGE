import sys
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

"""

========================================知识蒸馏损失函数（软损失函数）===========================================

"""

class Margin_HuberLoss(nn.Module):
    def __init__(self, args, delta=1.0):
        super(Margin_HuberLoss, self).__init__()
        self.delta = delta
        self.args = args
        self.HuberLoss_gamma = 9.0
    
    def forward(self, s_score, t_score):
        residual = torch.abs(t_score - s_score)
        condition = (residual < self.delta).float()
        loss = condition * 0.5 * residual**2 + (1 - condition) * (self.delta * residual - 0.5 * self.delta**2)
        
        loss = self.HuberLoss_gamma - loss
               
        p_loss, n_loss = loss[:, 0], loss[:, 1:]
        
        n_loss = F.logsigmoid(n_loss).mean(dim = 1)
        p_loss = F.logsigmoid(p_loss)
        
        p_loss = - p_loss.mean()
        n_loss = - n_loss.mean()

        # loss = (p_loss + n_loss)/2
        loss = p_loss*(1/(self.args.negative_sample_size+1)) + n_loss*(self.args.negative_sample_size/(self.args.negative_sample_size+1))
        
        return loss


class HuberLoss(nn.Module):
    def __init__(self, args, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.args = args
    
    def forward(self, y_pred, y_true, reduction='mean'):
        residual = torch.abs(y_true - y_pred)
        mask = residual < self.delta
        loss = torch.where(mask, 0.5 * residual ** 2, self.delta * residual - 0.5 * self.delta ** 2)
        
        if reduction=='batchmean':
            loss = loss.sum()/y_pred.shape[0]
        elif reduction=='sum':
            loss = loss.sum()
        elif reduction=='mean':
            loss = loss.mean()
        else:
            loss = loss
        
        return loss


class KL_divergency(nn.Module):
    def __init__(self, args, temprature=None):
        super(KL_divergency, self).__init__()
        self.args = args
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        if temprature is None:
            self.temprature_TS = self.args.temprature_ts
        else:
            self.temprature_TS = temprature
    
    def save_teacher_distributions_for_case_study(self, teacher_dis, teacher_p, file_path, batch_idx=None):
        """
        为 Case Study 保存教师模型的分布（排序后）。

        Args:
            teacher_dis (torch.Tensor): 教师模型的原始 logits。
            teacher_p (torch.Tensor): 经过温度缩放和 Softmax 后的教师模型概率。
            file_path (str): 保存数据的文件路径。
            batch_idx (int, optional): 当前批次的索引（用于在文件中区分）。默认为 None。
        """
        # 从计算图中分离张量，并移至 CPU 以便保存
        teacher_dis_detached = teacher_dis.detach().cpu()
        teacher_p_detached = teacher_p.detach().cpu()

        # 检查输入是否是批处理数据
        is_batched = teacher_dis_detached.dim() > 1
        num_items = teacher_dis_detached.shape[0] if is_batched else 1

        try:
            # 使用 'a' 模式打开文件，以便追加内容
            # 指定 encoding='utf-8' 避免在不同系统下的编码问题
            with open(file_path, 'w', encoding='utf-8') as f:
                # 文件首次写入时，可以添加一些头部信息（可选，这里不在每次调用时添加）
                # if os.path.getsize(file_path) == 0:
                #    f.write(f"--- Case Study: Teacher Distributions (T={self.temprature_TS:.2f}) ---\n")
                #    f.write("-" * 40 + "\n")

                for i in range(num_items):
                    # 获取当前项的张量
                    current_teacher_dis = teacher_dis_detached[i] if is_batched else teacher_dis_detached
                    current_teacher_p = teacher_p_detached[i] if is_batched else teacher_p_detached

                    # 按降序对数值进行排序
                    # torch.sort 返回 (values, indices)
                    sorted_dis_values, _ = torch.sort(current_teacher_dis, descending=True)
                    sorted_p_values, _ = torch.sort(current_teacher_p, descending=True)

                    # 写入排序后的值到文件
                    if is_batched:
                        item_header = f"--- Batch Item {i}"
                        if batch_idx is not None:
                           item_header += f" (Overall Batch {batch_idx})"
                        f.write(item_header + " ---\n")
                    else:
                        f.write(f"--- Single Item (Overall Batch {batch_idx if batch_idx is not None else 'N/A'}) ---\n")

                    f.write("Sorted Teacher Logits (teacher_dis):\n")
                    # 转换为列表以便清晰打印
                    f.write(str(sorted_dis_values.tolist()) + "\n")
                    f.write(f"Sorted Teacher Probabilities (teacher_p, T={self.temprature_TS:.2f}):\n")
                    f.write(str(sorted_p_values.tolist()) + "\n")
                    f.write("-" * 20 + "\n")

        except IOError as e:
            print(f"错误：无法写入 Case Study 文件 '{file_path}': {e}")
        except Exception as e:
             print(f"错误：在保存 Case Study 数据时发生意外错误: {e}")
    
    def forward(self, student_dis, teacher_dis, reduction='batchmean'):
                
        # 直接使用log_softmax避免数值不稳定
        student_log_p = F.log_softmax(student_dis/self.temprature_TS, dim=-1)
        teacher_p = F.softmax(teacher_dis/self.temprature_TS, dim=-1)
        
        # self.save_teacher_distributions_for_case_study(
        #             teacher_dis, # 传入原始 logits
        #             teacher_p,   # 传入计算好的概率
        #             'case_study_LorentzKG_100.txt',
        #             batch_idx=1 # 传递批次索引
        #         )
        # sys.exit(0)
        
        # 使用KL散度的直接形式：KL(p||q) = sum(p * log(p/q))
        loss = F.kl_div(student_log_p, teacher_p, reduction=reduction) * (self.temprature_TS ** 2)
        return loss


"""
CVPR2024 Logit_Standard KD
变化请直接选择Constant
"""
class DistillKL_Logit_Standard(nn.Module):
    """Logit Standardization in Knowledge Distillation"""
    def __init__(self, args):
        super(DistillKL_Logit_Standard, self).__init__()
        self.args = args
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.temperature = self.args.temprature_ts  # 确保有这个属性

    def normalize(self, logits):
        # 标准化logits
        mean = logits.mean(dim=-1, keepdim=True)
        std = logits.std(dim=-1, keepdim=True)
        return (logits - mean) / (std + 1e-6)

    def forward(self, student_logits, teacher_logits, reduction='batchmean'):
        T = self.temperature
        student_p = self.logsoftmax(self.normalize(student_logits) / T)
        teacher_p = self.softmax(self.normalize(teacher_logits) / T)
        
        # 计算知识蒸馏损失
        loss = F.kl_div(student_p, teacher_p, reduction=reduction) * (T * T)
        return loss

"""
用来处理输入直接是概率的情况
"""
class KL_divergencyv2(nn.Module):
    def __init__(self, args, temprature=None):
        super(KL_divergencyv2, self).__init__()
        self.args = args
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        if temprature is None:
            self.temprature_TS = self.args.temprature_ts
        else:
            self.temprature_TS = temprature

    def forward_(self, student_p, teacher_p, reduction='batchmean'):
        loss = F.kl_div(torch.log(student_p), teacher_p, reduction=reduction) * self.temprature_TS * self.temprature_TS
        return loss
    
    def forward(self, student_logits, teacher_prob, reduction='batchmean'):
        T = self.temprature_TS
        student_prob = self.softmax(student_logits / T)
        loss = self.forward_(student_prob, teacher_prob, reduction)
        
        return loss
    


class JS_divergence(nn.Module):
    """
    Computes the Jensen-Shannon divergence between two distributions,
    often used as a loss function in knowledge distillation.

    Mimics the structure of the provided KL_divergence class, including
    temperature scaling suitable for distillation tasks.

    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q) and KL is the Kullback-Leibler divergence.

    Note: Assumes inputs (student_dis, teacher_dis) are logits.
    """
    def __init__(self, args=None, temprature=None):
        """
        Initializes the JS_divergence module.

        Args:
            args (Namespace, optional): Arguments object, expected to have
                                        `args.temprature_ts` if `temprature` is None.
                                        Defaults to None.
            temprature (float, optional): The temperature scaling factor.
                                          Overrides `args.temprature_ts` if provided.
                                          Defaults to None.
        """
        super(JS_divergence, self).__init__()
        self.args = args
        # We don't strictly need softmax/logsoftmax layers here if using F.softmax/F.log_softmax
        # but keeping them for structural similarity if needed elsewhere.
        # self.softmax = nn.Softmax(dim=-1)
        # self.logsoftmax = nn.LogSoftmax(dim=-1)

        # Determine temperature
        # Use provided temperature if available, otherwise fallback to args
        if temprature is not None:
            self.temprature_TS = temprature
        elif args is not None and hasattr(args, 'temprature_ts'):
            self.temprature_TS = args.temprature_ts
        else:
            # Default temperature if neither is provided
            self.temprature_TS = 1.0
            print("Warning: JS_divergence initialized with default temperature T=1.0")

        # Epsilon for numerical stability when calculating log(M)
        self.epsilon = 1e-8

    def forward(self, student_dis, teacher_dis, reduction='batchmean'):
        """
        Calculates the JS divergence loss.

        Args:
            student_dis (torch.Tensor): Logits from the student model.
                                        Shape: (batch_size, num_classes)
            teacher_dis (torch.Tensor): Logits from the teacher model.
                                        Shape: (batch_size, num_classes)
            reduction (str, optional): Specifies the reduction to apply to the output:
                                       'none' | 'batchmean' | 'sum' | 'mean'.
                                       'batchmean': the sum of the output will be divided by batch size.
                                       'mean': the output will be averaged.
                                       'sum': the output will be summed.
                                       'none': no reduction will be applied.
                                       Defaults to 'batchmean'.

        Returns:
            torch.Tensor: The calculated JS divergence loss, potentially scaled by T^2.
                          If reduction is 'none', shape is (batch_size,).
                          Otherwise, it's a scalar tensor.
        """

        # Apply temperature scaling to logits
        student_dis_T = student_dis / self.temprature_TS
        teacher_dis_T = teacher_dis / self.temprature_TS

        # Calculate probabilities P and Q using softmax
        # These are the temperature-scaled probabilities P_T, Q_T
        student_p = F.softmax(student_dis_T, dim=-1)
        teacher_p = F.softmax(teacher_dis_T, dim=-1)

        # Calculate the average distribution M_T = 0.5 * (P_T + Q_T)
        m = 0.5 * (student_p + teacher_p)

        # Calculate log probabilities required for KL divergence calculation
        # It's numerically more stable to use log_softmax directly on scaled logits
        student_log_p = F.log_softmax(student_dis_T, dim=-1)
        teacher_log_p = F.log_softmax(teacher_dis_T, dim=-1)

        # Calculate KL(P_T || M_T)
        # We use the formula: sum(P_T * (log P_T - log M_T))
        # Need log M_T. Add epsilon for numerical stability before taking log.
        log_m = (m + self.epsilon).log()

        # Calculate element-wise KL divergence: P_T * (log P_T - log M_T)
        kl_student_m_elementwise = student_p * (student_log_p - log_m)
        kl_teacher_m_elementwise = teacher_p * (teacher_log_p - log_m)

        # Sum over the class dimension to get KL divergence per batch item
        kl_student_m_persample = torch.sum(kl_student_m_elementwise, dim=-1)
        kl_teacher_m_persample = torch.sum(kl_teacher_m_elementwise, dim=-1)

        # Combine for JS divergence per batch item
        js_div_persample = 0.5 * (kl_student_m_persample + kl_teacher_m_persample)

        # Apply reduction across the batch
        if reduction == 'batchmean':
            # PyTorch's 'batchmean' for KLDiv sums over dims and divides by batch size (dim 0).
            # Here we already summed over the class dim, so we just need to mean over batch dim.
            loss = torch.mean(js_div_persample)
        elif reduction == 'mean':
            # This interpretation usually means averaging over all elements.
            # Since we summed over classes, averaging over batch is the same as batchmean.
            # If it meant averaging over batch *and* classes, the implementation would differ.
            # Assuming 'mean' operates like 'batchmean' after summing over classes.
             loss = torch.mean(js_div_persample)
        elif reduction == 'sum':
            loss = torch.sum(js_div_persample)
        elif reduction == 'none':
            loss = js_div_persample
        else:
            raise ValueError(f"Invalid reduction type: {reduction}")

        # Apply temperature scaling factor (T^2), common practice in distillation
        # to compensate for the gradient softening effect of temperature.
        scaled_loss = loss * (self.temprature_TS ** 2)

        return scaled_loss
    
    

class Pearson_loss(nn.Module):
    def __init__(self, args):
        super(Pearson_loss, self).__init__()
        self.args = args
        self.eps = 1e-8

    def cosine_similarity(self,x, y, eps=1e-8):
        return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + self.eps)

    def forward(self, student_logits, teacher_logits, reduction='batchmean'):
        pearson_loss = self.cosine_similarity(student_logits - student_logits.mean(1).unsqueeze(1), teacher_logits - teacher_logits.mean(1).unsqueeze(1), self.eps)
        pearson_loss = 1 - pearson_loss.mean()
        
        return pearson_loss