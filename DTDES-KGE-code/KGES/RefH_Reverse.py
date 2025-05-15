import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

"""
RotH 中的entity和relation都不需要double
"""
class RefH_Reverse(nn.Module):
    def __init__(self, args):
        super(RefH_Reverse, self).__init__()
        self.args = args
        self.target_dim = args.target_dim
        self.init_size = args.init_size
        self.data_type = torch.double if self.args.data_type=='double' else torch.float
        self.bias = 'learn'
        
        self.bh = nn.Embedding(args.nentity, 1) # 头实体的偏置项
        self.bh.weight.data = torch.zeros((args.nentity, 1), dtype=self.data_type)
        self.bt = nn.Embedding(args.nentity, 1) # 尾实体的偏置项
        self.bt.weight.data = torch.zeros((args.nentity, 1), dtype=self.data_type)
        
        self.rel_diag = nn.Embedding(args.nrelation, self.target_dim)
        self.rel_diag.weight.data = 2 * torch.rand((args.nrelation, self.target_dim), dtype=self.data_type) - 1.0
        
        c_init = torch.ones((args.nrelation, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)
        
    
    def get_queries(self, head, relation, tail, sample):
        positive_sample, negative_sample = sample
        c = F.softplus(self.c[positive_sample[:, 1]])
        rel, _ = torch.chunk(self.rel(positive_sample[:, 1]), 2, dim=1)
        rel = expmap0(rel, c)
        lhs = givens_reflection(self.rel_diag(positive_sample[:, 1]), head.squeeze(1))
        lhs = expmap0(lhs, c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), self.bh(positive_sample[:, 0])

    
    def get_rhs(self, head, relation, tail, sample):
        """Get embeddings and biases of target entities."""
        positive_sample, negative_sample = sample
        negative_sample = negative_sample.reshape(-1)
        
        positive_bt = self.bt(positive_sample[:, 2])
        negative_bt = self.bt(negative_sample)
        
        positive_bt = positive_bt.reshape(-1, 1)
        negative_bt = negative_bt.reshape(positive_bt.shape[0], -1)
        
        bt = torch.cat((positive_bt, negative_bt), dim=1)
        
        return tail, bt 
        
    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        batch, dim, neg_sampling = lhs_e.shape[0], lhs_e.shape[-1], rhs_e.shape[0] // lhs_e.shape[0]
        lhs_e = lhs_e.unsqueeze(1).expand(-1, neg_sampling, -1).reshape(-1, dim)  # [batch * neg_sampling, dim]
        c = c.unsqueeze(1).expand(-1, neg_sampling, -1).reshape(-1, 1)  # [batch * neg_sampling, 1]
        
        return - hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode=False) ** 2
    
    def score(self, lhs, rhs):
        lhs_e, lhs_biases = lhs
        rhs_e, rhs_biases = rhs
        score = self.similarity_score(lhs_e, rhs_e)
        if self.bias == 'constant':
            return self.args.pos_gamma + score
        elif self.bias == 'learn':
            batch_size = lhs_biases.size(0)
            neg_sampling = rhs_biases.size(1)
            
            # 扩展 lhs_biases
            expanded_lhs_biases = lhs_biases.repeat(1, neg_sampling)  # shape: [batch_size, neg_sampling + 1, 1]
            expanded_lhs_biases = expanded_lhs_biases.view(-1, 1)

            # 将 lhs_biases 的值插入到 rhs_biases 中
            merged_rhs_biases = rhs_biases.view(-1, 1)  # 展平回 [batch_size * (neg_sampling + 1), 1]

            # print(expanded_lhs_biases.shape, merged_rhs_biases.shape, score.shape)
            
            return expanded_lhs_biases + merged_rhs_biases + score
        else:
            return score
    
    
    def forward(self, head, relation, tail, sample):
        batch_size, negative_sampling = head.shape[0], tail.shape[1]
        head_dim, relation_dim, tail_dim = head.shape[-1], relation.shape[-1], tail.shape[-1]
        head, relation, tail = head.reshape(batch_size, head_dim), relation.reshape(batch_size, relation_dim), tail.reshape(batch_size*negative_sampling, tail_dim)
        lhs_e, lhs_biases = self.get_queries(head, relation, tail, sample)
        rhs_e, rhs_biases = self.get_rhs(head, relation, tail, sample)
        predictions = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases))
        predictions = predictions.reshape(batch_size, negative_sampling)

        return predictions
        


def givens_rotations(r, x):
    """Givens rotations.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate

    Returns:
        torch.Tensor os shape (N x d) representing rotation of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    x = x.view((r.shape[0], -1, 2))
    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_rot.view((r.shape[0], -1))


def givens_reflection(r, x):
    """Givens reflections.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to reflect

    Returns:
        torch.Tensor os shape (N x d) representing reflection of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    x = x.view((r.shape[0], -1, 2))
    x_ref = givens[:, :, 0:1] * torch.cat((x[:, :, 0:1], -x[:, :, 1:]), dim=-1) + givens[:, :, 1:] * torch.cat(
        (x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_ref.view((r.shape[0], -1))


MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}


# ################# MATH FUNCTIONS ########################

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


def tanh(x):
    return x.clamp(-15, 15).tanh()


# ################# HYP OPS ########################

def expmap0(u, c):
    """Exponential map taken at the origin of the Poincare ball with curvature c.

    Args:
        u: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with tangent points.
    """
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)


def logmap0(y, c):
    """Logarithmic map taken at the origin of the Poincare ball with curvature c.

    Args:
        y: torch.Tensor of size B x d with tangent points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with hyperbolic points.
    """
    sqrt_c = c ** 0.5
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)


def project(x, c):
    """Project points to Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with projected hyperbolic points.
    """
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def mobius_add(x, y, c):
    """Mobius addition of points in the Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic points
        y: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        Tensor of shape B x d representing the element-wise Mobius addition of x and y.
    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)


# ################# HYP DISTANCES ########################

def hyp_distance(x, y, c, eval_mode=False):
    """Hyperbolic distance on the Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic queries
        y: torch.Tensor with hyperbolic queries, shape n_entities x d if eval_mode is true else (B x d)
        c: torch.Tensor of size 1 with absolute hyperbolic curvature

    Returns: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    sqrt_c = c ** 0.5
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    if eval_mode:
        y2 = torch.sum(y * y, dim=-1, keepdim=True).transpose(0, 1)
        xy = x @ y.transpose(0, 1)
    else:
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
    c1 = 1 - 2 * c * xy + c * y2
    c2 = 1 - c * x2
    num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * y2 - (2 * c1 * c2) * xy)
    denom = 1 - 2 * c * xy + c ** 2 * x2 * y2
    pairwise_norm = num / denom.clamp_min(MIN_NORM)
    dist = artanh(sqrt_c * pairwise_norm)
    return 2 * dist / sqrt_c


def hyp_distance_multi_c(x, v, c, eval_mode=False):
    """Hyperbolic distance on Poincare balls with varying curvatures c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic queries
        y: torch.Tensor with hyperbolic queries, shape n_entities x d if eval_mode is true else (B x d)
        c: torch.Tensor of size B x d with absolute hyperbolic curvatures

    Return: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    sqrt_c = c ** 0.5
    if eval_mode:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True).transpose(0, 1)
        xv = x @ v.transpose(0, 1) / vnorm
    else:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True)
        xv = torch.sum(x * v / vnorm, dim=-1, keepdim=True)
    gamma = tanh(sqrt_c * vnorm) / sqrt_c
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    c1 = 1 - 2 * c * gamma * xv + c * gamma ** 2
    c2 = 1 - c * x2
    num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * (gamma ** 2) - (2 * c1 * c2) * gamma * xv)
    denom = 1 - 2 * c * gamma * xv + (c ** 2) * (gamma ** 2) * x2
    pairwise_norm = num / denom.clamp_min(MIN_NORM)
    dist = artanh(sqrt_c * pairwise_norm)
    return 2 * dist / sqrt_c