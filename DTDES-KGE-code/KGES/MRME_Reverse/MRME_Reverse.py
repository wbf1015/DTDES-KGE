from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch.nn.functional as F
import torch
import numpy as np
from torch import nn
import os
import sys

from tqdm import tqdm

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import math
from geoopt import ManifoldParameter


original_directory = os.getcwd()
new_directory = original_directory + '/code/KGES/MRME_Reverse/'  
sys.path.append(new_directory)

from hyperbolic import expmap0, project,logmap0,expmap1,logmap1
from euclidean import givens_rotations, givens_reflection
from manifolds import Lorentz

sys.path.remove(new_directory)



EPS = 1e-5
temperature=0.2
max_norm=0.5
max_scale=2


# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        ranks = torch.ones(len(queries))
        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    scores, _,_ = self.forward(these_queries)
                    
                    targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())] 
                        filter_out += [queries[b_begin + i, 2].item()]
                        scores[i, torch.LongTensor(filter_out)] = -1e6
                    # # Calculate the ranking of each query sample, that is, the number of scores greater than or equal to the target score, and add the results to ranks
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                    bar.update(batch_size)
        return ranks

def lorentz_linear(x, weight, scale, bias=None):
    x = x @ weight.transpose(-2, -1)
    # time = x.narrow(-1, 0, 1).sigmoid() * scale + 1.1
    time = x.narrow(-1, 0, 1).sigmoid() * scale + 1.1
    if bias is not None:
        x = x + bias
    x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
    x_narrow = x_narrow / ((x_narrow * x_narrow).sum(dim=-1, keepdim=True) / (time * time - 1)).sqrt()
    x = torch.cat([time, x_narrow], dim=-1)
    return x

"""

MRME_KGC_NoEuclidean

"""
class MRME_KGC(KBCModel):
    def __init__(
        self, sizes: Tuple[int, int, int], rank: int,
        init_size: float = 1e-3
    ):
        super(MRME_KGC, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.init_size = 0.001
        self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()
        self.act = nn.Softmax(dim=1)
        self.num_heads = 2
        self.multihead_attn = nn.MultiheadAttention(embed_dim=rank, num_heads=2)
        self.temperature = 0.2
        self.manifold = Lorentz(max_norm=max_norm)
        self.scale = nn.Parameter(torch.ones(()) * max_scale, requires_grad=False)
        self.manifold = Lorentz(max_norm=max_norm)
        self.emb_entity = ManifoldParameter(self.manifold.random_normal((sizes[0], rank), std=1. / math.sqrt(rank)),
                                            manifold=self.manifold)

        self.relation_transform = nn.Parameter(torch.empty(sizes[0], rank, rank))
        nn.init.kaiming_uniform_(self.relation_transform)

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings1 = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings1[0].weight.data *= init_size
        self.embeddings1[1].weight.data *= init_size
        self.multi_c = 1
        self.data_type = torch.float32

        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
            c_init1 = torch.ones((self.sizes[1], 1), dtype=self.data_type)
            c_init2 = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        else:
            c_init = torch.ones((1, 1), dtype=self.data_type)
            c_init1 = torch.ones((1, 1), dtype=self.data_type)
            c_init2 = torch.ones((1, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)
        self.c1 = nn.Parameter(c_init1, requires_grad=True)
        self.c2 = nn.Parameter(c_init2, requires_grad=True)

    def forward(self, x, nneg_plus_idx):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lo_lhs = self.emb_entity[x]
        lo_rel = self.relation_transform[x]
        
        rel1 = self.embeddings1[0](x[:, 1])
        rel2 = self.embeddings1[1](x[:, 1])
        entities = self.embeddings[0].weight
        batch_size, nneg_plus = nneg_plus_idx.shape
        flat_indices = nneg_plus_idx.view(-1)
        selected_embeddings = entities.index_select(0, flat_indices)
        selected_embeddings = selected_embeddings.view(batch_size, nneg_plus, -1)
        entity1 = selected_embeddings[:, :, :self.rank]
        entity2 = selected_embeddings[:, :, self.rank:]
        lhs_t = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        rel1 = rel1[:, :self.rank]
        rel2 = rel2[:, :self.rank]
        lhs = lhs_t[0]
        
        # 曲率c大于0的球形空间
        c1 = F.softplus(self.c1[x[:, 1]])
        head1 = expmap0(lhs, c1)
        rel11 = expmap0(rel1, c1)
        lhs = head1
        res_c1 = logmap0(givens_reflection(rel2, lhs), c1)
        translation1 = lhs_t[1] * rel[1]
        
        # 曲率c小于0的庞加莱球空间
        c2 = F.softplus(self.c2[x[:, 1]])
        head2 = expmap1(lhs, c2)
        rel12 = expmap1(rel1, c2)
        lhss = head2
        res_c2 = logmap1(givens_rotations(rel2, lhss), c2)
        translation2 = lhs_t[1] * rel[0]
        
        # 洛伦兹空间
        c = F.softplus(self.c[x[:, 1]])
        head = lhs
        
        # 去除欧氏空间部分 (rot_q)
        
        # 洛伦兹空间变换
        lo_h = lorentz_linear(lo_lhs.unsqueeze(1), lo_rel, self.scale).squeeze(1)
        lo_h = lo_h.mean(dim=(1, 2))
        
        # 只使用球形空间、庞加莱球空间和洛伦兹空间
        cands = torch.cat([res_c1.view(-1, 1, self.rank), res_c2.view(-1, 1, self.rank), lo_h.view(-1, 1, self.rank)], dim=1)
        context_vec = self.context_vec(x[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)

        # print((att_q * rel[0] - translation1).shape)
        # print((att_q * rel[1] + translation2).shape)
        # print(entity1.shape)
        # print(entity2.shape)
        # sys.exit(0)
        
        query1 = (att_q * rel[0] - translation1).unsqueeze(1)
        score1 = torch.bmm(query1, entity1.transpose(1, 2)).squeeze(1)
        query2 = (att_q * rel[1] + translation2).unsqueeze(1)
        score2 = torch.bmm(query2, entity2.transpose(1, 2)).squeeze(1)
        score = score1 + score2
        
        return score
        # return (
        #         (att_q * rel[0] - translation1) @ entity1.t() + (att_q * rel[1] + translation2) @ entity2.t()
        #     )