import sys
import os

import torch.nn.functional as F
import torch
import torch.nn as nn

class EncoderModel_Reverse_2KGE(nn.Module):
    def __init__(self, KGE1=None, KGE2=None, embedding_manager=None, entity_pruner=None, relation_pruner=None, decoder=None, args=None):
        super(EncoderModel_Reverse_2KGE, self).__init__()
        self.KGE1 = KGE1
        self.KGE2 = KGE2
        self.EmbeddingManager = embedding_manager
        self.EntityPruner = entity_pruner
        self.RelationPruner = relation_pruner
        self.decoder = decoder
        self.args = args

    def get_LorentzE_input(self, data):
        positive_sample, negative_sample = data
        e1_idx = positive_sample[:, 0]
        r_idx = positive_sample[:, 1]
        e2_idx = torch.cat([positive_sample[:, 2].unsqueeze(1), negative_sample], dim=1)
        
        return e1_idx, r_idx, e2_idx
    
    def get_MRME_input(self, data):
        positive_sample, negative_sample = data
        e1_idx = positive_sample[:, 0]
        r_idx = positive_sample[:, 1]
        e2_idx = positive_sample[:, 2]
        x = torch.cat([e1_idx.unsqueeze(1), r_idx.unsqueeze(1), e2_idx.unsqueeze(1)], dim=1)
        nneg_plus_idx = torch.cat([e2_idx.unsqueeze(1), negative_sample], dim=1)
        
        return x, nneg_plus_idx
    
    def deal_with_cuda(self, values):
        if self.args.cuda:
            values = [v.cuda() if v is not None else None for v in values]
        head, relation, tail, PT_head1, PT_relation1, PT_tail1, PT_head2, PT_relation2, PT_tail2 = values
        return head, relation, tail, PT_head1, PT_relation1, PT_tail1, PT_head2, PT_relation2, PT_tail2
    
            
    def forward(self, data, subsampling_weight, mode, epoch):
        values = self.EmbeddingManager(data)
        head, relation, tail, PT_head1, PT_relation1, PT_tail1, PT_head2, PT_relation2, PT_tail2 = self.deal_with_cuda(values)
        
        PT1_score, PT2_score = self.get_KGEScore(data)
        Teacher_embeddings = (PT_head1, PT_relation1, PT_tail1, PT_head2, PT_relation2, PT_tail2)
        
        loss, loss_record = self.decoder(head, relation, tail, PT1_score, PT2_score, data, Teacher_embeddings, subsampling_weight, mode, epoch)
        
        return loss, loss_record
    
    def get_KGEScore(self,data):
        with torch.no_grad():
            self.KGE1.eval()
            self.KGE2.eval()
            values = self.EmbeddingManager(data)
            head, relation, tail, PT_head1, PT_relation1, PT_tail1, PT_head2, PT_relation2, PT_tail2 = self.deal_with_cuda(values)
            
            if self.KGE1.__class__.__name__ == 'RotatE_Reverse' or self.KGE1.__class__.__name__ == 'SCCF_Reverse':
                PT1_score = self.KGE1(PT_head1, PT_relation1, PT_tail1)
            if self.KGE1.__class__.__name__ in ['AttH_Reverse']:
                PT1_score = self.KGE1(PT_head1, PT_relation1, PT_tail1, data)
            if self.KGE1.__class__.__name__ == 'HyperNet':
                e1_idx, r_idx, e2_idx = self.get_LorentzE_input(data)
                PT1_score = self.KGE1(e1_idx, r_idx, e2_idx)
            if self.KGE1.__class__.__name__ == 'MRME_KGC':
                x, nneg_plus_idx = self.get_MRME_input(data)
                PT1_score = self.KGE1(x, nneg_plus_idx)
            if self.KGE1.__class__.__name__ == 'HAKE_Reverse':
                PT1_score = self.KGE1(data)
            
            
            if self.KGE2.__class__.__name__ == 'RotatE_Reverse' or self.KGE2.__class__.__name__ == 'SCCF_Reverse':
                PT2_score = self.KGE2(PT_head2, PT_relation2, PT_tail2)
            if self.KGE2.__class__.__name__ in ['AttH_Reverse']:
                PT2_score = self.KGE2(PT_head2, PT_relation2, PT_tail2, data)
            if self.KGE2.__class__.__name__ == 'HyperNet':
                e1_idx, r_idx, e2_idx = self.get_LorentzE_input(data)
                PT2_score = self.KGE2(e1_idx, r_idx, e2_idx)
            if self.KGE2.__class__.__name__ == 'MRME_KGC':
                x, nneg_plus_idx = self.get_MRME_input(data)
                PT2_score = self.KGE2(x, nneg_plus_idx)
            if self.KGE2.__class__.__name__ == 'HAKE_Reverse':
                PT2_score = self.KGE2(data)
        
            return PT1_score, PT2_score
    
    
    def predict(self, data, predict_number=1):
        values = self.EmbeddingManager(data)
        head, relation, tail, PT_head1, PT_relation1, PT_tail1, PT_head2, PT_relation2, PT_tail2 = self.deal_with_cuda(values)
        
        # PT1_score, PT2_score = self.get_KGEScore(data)
        # PT1_score, PT2_score = self.get_KGEScore(data)
        # return PT2_score
  
        score = self.decoder.predict(head, relation, tail)

        return score
    
    def predict_augment(self, data):
        values = self.EmbeddingManager(data)
        head, relation, tail, PT_head1, PT_relation1, PT_tail1, PT_head2, PT_relation2, PT_tail2 = self.deal_with_cuda(values)
        score = self.decoder.predict(head, relation, tail)
        
        positive_sample, negative_sample = data
        positive_sample[:, 1] = torch.where(positive_sample[:, 1] >= self.args.nrelation // 2, positive_sample[:, 1] - self.args.nrelation // 2, positive_sample[:, 1] + self.args.nrelation // 2)
        data = (positive_sample, negative_sample)
        values = self.EmbeddingManager(data)
        head, relation, tail, PT_head1, PT_relation1, PT_tail1, PT_head2, PT_relation2, PT_tail2 = self.deal_with_cuda(values)
        
        score2 = self.decoder.predict(tail, relation, head)
        
        score = score+score2
        return score
    
    
        
