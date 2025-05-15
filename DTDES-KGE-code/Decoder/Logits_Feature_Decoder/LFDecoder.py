import sys
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import ast

original_directory = os.getcwd()
new_directory = original_directory + '/code/Decoder/Logits_Feature_Decoder/'
if new_directory not in sys.path:
    sys.path.append(new_directory)

from LFD_fusion import *
from LFD_norm import *
from LFD_soft_loss import *
from LFD_hard_loss import *
from LFD_embedding_transform import *
from LFD_soft_loss_calculation import *
from LFD_similarity import *
from LFD_Encoder import *
from LFD_regularizers import *
from LFDContrastive import *

sys.path.remove(new_directory)

class Decoder_2KGE(nn.Module):
    def __init__(self, args):
        super(Decoder_2KGE, self).__init__()
        self.args = args
        self.target_dim = args.target_dim
        self.entity_dim = args.target_dim * args.entity_mul
        self.relation_dim = args.target_dim * args.relation_mul

        self.stu_pre_process = self.args.stu_pre_process
        self.tea_pre_process = self.args.tea_pre_process
        self.distill_func = self.args.distill_func
        self.weight_mask = self.args.weight_mask
        self.weight_mask = ast.literal_eval(self.weight_mask)
        assert len(self.weight_mask) == 5, "weight_mask的长度必须为5"
        
        # 硬损失函数
        self.hard_loss1 = SigmoidLoss(args=args)
        self.hard_loss2 = SigmoidLoss2(args=args)
        # self.hard_loss2 = BCELossWithSmoothing(args=args)
        # self.hard_loss2 = CrossEntropyloss(args=args)
        
        # 软损失函数
        self.soft_loss = Imitation_SingleTeacher(args=args, stu_score_preprocess=self.stu_pre_process, tea_score_preprocess=self.tea_pre_process, distill_func=self.distill_func)
        self.soft_loss2 = Imitation_DualTeacherv1(args=args, stu_score_preprocess=self.stu_pre_process, tea_score_preprocess=self.tea_pre_process, distill_func=self.distill_func,  fusion_scores_loss='SigmoidLoss')

        # 辅助模块
        # self.stage0weight_learner = weight_learnerv1_PostLNRes(args, entity_dim=self.entity_dim, relation_dim=self.relation_dim)
        # self.stage0weight_learner = weight_learnerv2(args, entity_dim=self.entity_dim, relation_dim=self.relation_dim)
        # self.stage0weight_learner = weight_learnerv2_1(args, entity_dim=self.entity_dim, relation_dim=self.relation_dim)
        # self.stage0weight_learner = weight_learnerv3(args, entity_dim=self.entity_dim, relation_dim=self.relation_dim)
        self.stage0weight_learner = weight_learnerv3_prior(args, entity_dim=self.entity_dim, relation_dim=self.relation_dim)
        # self.stage0weight_learner = weight_learnerv4(args, entity_dim=self.entity_dim, relation_dim=self.relation_dim)
        self.Contrastive = ContrastiveLossv2(self.args)

        # 一些必要的组件
        self.reg = N3(self.args.reg)
        self.cal_sim = cal_similarity(args=self.args, temperature=self.args.temprature)
        self.stage0Combine_hr = Combine_hr(entity_dim=self.entity_dim, relation_dim=self.relation_dim, hidden_dim=self.entity_dim, layer_mul=2)
        self.stage0tail_transform = BN(input_dim=self.entity_dim)
    
    def get_student_score(self, eh_, er_, et_, ehr, et, need_sub=True):        
        stu_score = self.cal_sim.SCCF_similarity3(ehr, et)
        
        if need_sub:
            stu_score[:, 0] = stu_score[:, 0] -  self.args.pos_gamma  # 第一列减去pos_gamma
            stu_score[:, 1:] = stu_score[:, 1:] - self.args.neg_gamma  # 其余列减去neg_gamma
        
        return stu_score  
    
    def loss_ensemble(self, eh_, er_, et_, ehr, et, PT1_score, PT2_score, weight_mask, data=None, Teacher_embeddings=None, subsampling_weight=None, prefix='', epoch=None):
        hard_stu_score= self.get_student_score(eh_, er_, et_, ehr, et)
        soft_stu_score = self.get_student_score(eh_, er_, et_, ehr, et, need_sub=False)
                
        if 'FB15k-237' in self.args.data_path:
            prefix1 = 'RotatE'
            prefix2 = 'LorentzKG'
        elif 'wn18rr' in self.args.data_path:
            prefix1 = 'RotatE'
            prefix2 = 'MRME-KGC'
        elif 'YAGO3-10' in self.args.data_path:
            prefix1 = 'HAKE'
            prefix2 = 'MRME-KGC'
        
        hard_loss, loss_record = self.hard_loss1(hard_stu_score, subsampling_weight) # 硬损失计算方式1 正负样本各占1/2
        hard_loss2, loss_record2 = self.hard_loss2(hard_stu_score, subsampling_weight) # 硬损失计算方式2 正负样本按比例分配权重
        
        soft_loss1, soft_loss_record1 = self.soft_loss(soft_stu_score, PT1_score, prefix=prefix1) # 和第一个教师模型学习的软损失
        soft_loss2, soft_loss_record2 = self.soft_loss(soft_stu_score, PT2_score, prefix=prefix2) # 和第二个教师模型学习的软损失
        soft_lossfusion, soft_loss_recordfusion, weight = self.soft_loss2(eh_, er_, et_, ehr, et, soft_stu_score, PT1_score, PT2_score, prefix='Fusion', weight=self.stage0weight_learner, data=data) # 和两个教师模型学习的软 损失
        
        contrastive_loss, contrastive_loss_record = self.Contrastive(eh_, er_, et, Teacher_embeddings, weight)
        
        if self.args.dynamic_weight:
            weight_mask = self.dynamic_weight(weight_mask, epoch)
        
        loss = weight_mask[0] * hard_loss
        loss += weight_mask[1] * hard_loss2
        loss += weight_mask[2] * self.args.kdloss_weight * soft_loss1 
        loss += weight_mask[3] * self.args.kdloss_weight * soft_loss2
        loss += weight_mask[4] * self.args.kdloss_weight * soft_lossfusion
        loss += self.args.contrastive_weight * contrastive_loss
        
        loss_record.update(loss_record)
        loss_record.update(loss_record2)
        loss_record.update(soft_loss_record1)
        loss_record.update(soft_loss_record2)
        loss_record.update(soft_loss_recordfusion)
        loss_record.update(contrastive_loss_record)
        loss_record.update({'Loss':loss.item()})
        
        loss_record = {prefix + key: value for key, value in loss_record.items()}
        return loss, loss_record
    
    def stage0_Encoder(self, eh, er, et, PT1_score, PT2_score, data=None, Teacher_embeddings=None, subsampling_weight=None, epoch=None):
        ehr_ = self.stage0Combine_hr(eh, er)
        et_ = self.stage0tail_transform(et)
        
        if 'FB15k-237' in self.args.data_path:
            loss, loss_record = self.loss_ensemble(eh, er, et, ehr_, et_, PT1_score, PT2_score, self.weight_mask, data=data, Teacher_embeddings=Teacher_embeddings, subsampling_weight=subsampling_weight, prefix='Stage0_', epoch=epoch)
        elif 'wn18rr' in self.args.data_path:
            loss, loss_record = self.loss_ensemble(eh, er, et, ehr_, et_, PT1_score, PT2_score, self.weight_mask, data=data, Teacher_embeddings=Teacher_embeddings, subsampling_weight=subsampling_weight, prefix='Stage0_', epoch=epoch)
        elif 'YAGO3-10' in self.args.data_path:
            loss, loss_record = self.loss_ensemble(eh, er, et, ehr_, et_, PT1_score, PT2_score, self.weight_mask, data=data, Teacher_embeddings=Teacher_embeddings, subsampling_weight=subsampling_weight, prefix='Stage0_', epoch=epoch)
        
        return loss, loss_record

    def query_sample_batch(self, eh, er, et, PT1_score, PT2_score, data=None, Teacher_embeddings=None, subsampling_weight=None, epoch=None):
        
        reg, reg_record = self.reg(
            [(torch.sqrt(eh.squeeze(1) ** 2), 
             torch.sqrt(er.squeeze(1) ** 2), 
             torch.sqrt(et[:, 0, :].squeeze(1) ** 2))]
            )
        
        PT_head1, PT_relation1, PT_tail1, PT_head2, PT_relation2, PT_tail2 = Teacher_embeddings
        
        loss0, loss_record0 = self.stage0_Encoder(eh, er, et, PT1_score, PT2_score, data=data, Teacher_embeddings=Teacher_embeddings, subsampling_weight=subsampling_weight, epoch=epoch)

        loss = 0
        loss += 1 * loss0
        loss += reg
        
        loss_record = {}
        loss_record.update(loss_record0)
        loss_record.update(reg_record)
        loss_record.update({'loss':loss.item()})
        
        return loss, loss_record

    
    def random_sample_batch(self, eh, er, et, PT1_score, PT2_score, data=None, Teacher_embeddings=None, subsampling_weight=None):
        PT_head1, PT_relation1, PT_tail1, PT_head2, PT_relation2, PT_tail2 = Teacher_embeddings
        loss0, loss_record0 = self.stage0_Encoder(eh, er, et, PT1_score, PT2_score, data=data, Teacher_embeddings=Teacher_embeddings, subsampling_weight=subsampling_weight)

        loss = 0
        loss += 1 * loss0
        
        loss_record = {}
        loss_record.update(loss_record0)
        loss_record.update({'loss':loss.item()})
        
        return loss, loss_record
        
    def forward(self, eh, er, et, PT1_score, PT2_score, data=None, Teacher_embeddings=None, subsampling_weight=None, mode='QueryAwareSample', epoch=None):
        
        # self.case_study(PT1_score, PT2_score)
        
        if mode=='RandomSample':
            loss, loss_record = self.random_sample_batch(eh, er, et, PT1_score, PT2_score, data=data, Teacher_embeddings=Teacher_embeddings, subsampling_weight=subsampling_weight)
        
        if mode=='QueryAwareSample':
            loss, loss_record = self.query_sample_batch(eh, er, et, PT1_score, PT2_score, data=data, Teacher_embeddings=Teacher_embeddings, subsampling_weight=subsampling_weight, epoch=epoch)
    
        return loss, loss_record
    
    def predict(self, eh, er, et, PT1_score=None, PT2_score=None):
        ehr_ = self.stage0Combine_hr(eh, er)
        et_ = self.stage0tail_transform(et)
        stu_score = self.get_student_score(eh, er, et, ehr_, et_, need_sub=False)  
        return stu_score
    
    def dynamic_weight(self, weight_mask, epoch):
        # 定义 soft loss 权重的起始值 (epoch=1)
        # start_soft1 = 0.5
        # start_soft2 = 0.5
        # start_soft3 = 0.0

        # # 定义 soft loss 权重的结束值 (epoch=self.args.epoch + 1)
        # end_soft1 = 0.05
        # end_soft2 = 0.05
        # end_soft3 = 0.90
        
        # stop_epoch = 50

        # 定义 soft loss 权重的起始值 (epoch=1)
        start_soft1 = 0.5
        start_soft2 = 0.5
        start_soft3 = 0.0

        # 定义 soft loss 权重的结束值 (epoch=self.args.epoch + 1)
        end_soft1 = 0.2
        end_soft2 = 0.2
        end_soft3 = 0.60
        
        stop_epoch = 50
        
        
        total_epochs_for_transition = min(stop_epoch, self.args.epoch)

        # 计算原始的 alpha 值
        raw_alpha = (epoch - 1) / total_epochs_for_transition
        # 将 alpha 限制在 [0, 1] 之间，确保在范围外的 epoch 使用起始或结束权重
        alpha = max(0.0, min(1.0, raw_alpha))

        # 根据 alpha 计算当前 soft loss 权重
        current_soft1 = start_soft1 + (end_soft1 - start_soft1) * alpha
        current_soft2 = start_soft2 + (end_soft2 - start_soft2) * alpha # soft_loss1 和 soft_loss2 保持相等
        current_soft3 = start_soft3 + (end_soft3 - start_soft3) * alpha

        weight_mask[2] = current_soft1 # soft_loss1_weight
        weight_mask[3] = current_soft2 # soft_loss2_weight
        weight_mask[4] = current_soft3 # soft_loss3_weight (假设这个索引用于 soft_loss3 的动态权重)

        if epoch > stop_epoch:
            weight_mask[2] = end_soft1
            weight_mask[3] = end_soft2
            weight_mask[4] = end_soft3

        
        # 返回更新后的权重 mask
        return weight_mask
    
    
    def case_study(self, PT1_score, PT2_score):
        """
        将两个形状为 [batch, nneg+1] 的分数张量逐行写入本地文件。
        PT1  → case_study/case_study2_LorentzKG_512.txt
        PT2  → case_study/case_study2_MRME_512.txt
        """
        # 1. 目录准备
        out_dir = "case_study"

        def _save(tensor, filename):
            """把 tensor 逐行保存到 filename（一行 = 一个样本的全部 nneg+1 分数）"""
            # 若在 GPU 上、或需要与计算图解绑，先处理一下
            if torch.is_tensor(tensor):
                tensor = tensor.detach().cpu().float().numpy()
            elif not isinstance(tensor, np.ndarray):
                tensor = np.asarray(tensor, dtype=np.float32)

            path = os.path.join(out_dir, filename)
            # 使用 numpy.savetxt 可以一次性逐行写入；fmt 控制小数位数
            np.savetxt(path, tensor, fmt="%.6f")   # 每行自动写完换行
            # 如果你想看确认信息可以取消下一行注释
            # print(f"Saved {tensor.shape[0]} lines to {path}")

        _save(PT1_score, "case_study2_LorentzKG_512.txt")
        _save(PT2_score, "case_study2_MRME_512.txt")
        
        sys.exit(0)