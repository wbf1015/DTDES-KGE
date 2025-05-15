import torch
import torch.nn as nn
import torch.nn.functional as F

class DistributionSelector(nn.Module):
    def __init__(self, hidden_dim1=16, hidden_dim2=8, output_dim=2):
        """
        Args:
            input_dim (int): 输入维度 (默认是 4: mean1, mean2, std1, std2)
            hidden_dim1 (int): 第一个隐藏层的维度
            hidden_dim2 (int): 第二个隐藏层的维度 (如果为 0 或 None 则不使用)
            output_dim (int): 输出维度 (默认是 2: mean*, std*_raw)
        """
        super(DistributionSelector, self).__init__()
        
        input_dim = 4
        
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Linear(hidden_dim2, output_dim)
        )
        
        self.eps = 1e-8

    def forward(self, PT1_score, PT2_score):

        # 0. 构造x
        mean1 = torch.mean(PT1_score, dim=1, keepdim=True)
        std1 = torch.std(PT1_score, dim=1, keepdim=True)

        mean2 = torch.mean(PT2_score, dim=1, keepdim=True)
        std2 = torch.std(PT2_score, dim=1, keepdim=True)

        std1 = std1 + self.eps
        std2 = std2 + self.eps

        x = torch.cat((mean1, mean2, std1, std2), dim=1)
        raw_output = self.MLP(x)

        # 分离 mean* 和 std*_raw
        mean_star = raw_output[:, 0:1] # shape [batch_size, 1]
        std_star_raw = raw_output[:, 1:2] # shape [batch_size, 1]
        std_star = F.softplus(std_star_raw) + self.eps

        return mean_star, std_star


class DistributionSelectorv1(nn.Module):
    def __init__(self, hidden_dim1=16, hidden_dim2=8, output_dim=2):
        """
        Args:
            input_dim (int): 输入维度 (默认是 4: mean1, mean2, std1, std2)
            hidden_dim1 (int): 第一个隐藏层的维度
            hidden_dim2 (int): 第二个隐藏层的维度 (如果为 0 或 None 则不使用)
            output_dim (int): 输出维度 (默认是 2: mean*, std*_raw)
        """
        super(DistributionSelectorv1, self).__init__()
        
        input_dim = 4
        
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.GELU(),
            nn.LayerNorm(hidden_dim1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim2),
            nn.Linear(hidden_dim2, output_dim)
        )
        
        self.eps = 1e-8

    def forward(self, PT1_score, PT2_score):

        # 0. 构造x
        mean1 = torch.mean(PT1_score, dim=1, keepdim=True)
        std1 = torch.std(PT1_score, dim=1, keepdim=True)

        mean2 = torch.mean(PT2_score, dim=1, keepdim=True)
        std2 = torch.std(PT2_score, dim=1, keepdim=True)

        std1 = std1 + self.eps
        std2 = std2 + self.eps

        x = torch.cat((mean1, mean2, std1, std2), dim=1)
        raw_output = self.MLP(x)

        # 分离 mean* 和 std*_raw
        mean_star = raw_output[:, 0:1] # shape [batch_size, 1]
        log_std_star = raw_output[:, 1:2] # shape [batch_size, 1]

        log_std_star = torch.clamp(log_std_star, min=-20.0, max=2.0) # 示例：限制 log_std 在 [-20, 2] 之间，对应 std 范围 [~e-9, ~7.4]
        std_star = torch.exp(log_std_star)

        return mean_star, std_star



class DistributionSelector2(nn.Module):
    def __init__(self, hidden_dim1=30, hidden_dim2=6, output_dim=2):
        """
        Args:
            hidden_dim1 (int): 第一个隐藏层的维度
            hidden_dim2 (int): 第二个隐藏层的维度 (如果为 0 或 None 则不使用)
            output_dim (int): 输出维度 (默认是 2: mean*, std*_raw)
        """
        super(DistributionSelector2, self).__init__()
        # 内部输入维度现在是 6: mean1, mean2, std1, std2, mean_diff, std_ratio
        input_dim = 6
        self.use_hidden_layer2 = hidden_dim2 is not None and hidden_dim2 > 0
        self.eps = 1e-6 # 用于计算比率时的数值稳定性

        self.MLP = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim)
        )

    def forward(self, PT1_score, PT2_score):
        # 0. 构造x
        mean1 = torch.mean(PT1_score, dim=1, keepdim=True)
        std1 = torch.std(PT1_score, dim=1, keepdim=True)

        mean2 = torch.mean(PT2_score, dim=1, keepdim=True)
        std2 = torch.std(PT2_score, dim=1, keepdim=True)

        std1 = std1 + self.eps
        std2 = std2 + self.eps

        x = torch.cat((mean1, mean2, std1, std2), dim=1)
        
        # 1. 从输入 x (shape [batch_size, 4]) 中解包原始统计量
        # 使用切片保持维度 [batch_size, 1]
        mean1 = x[:, 0:1]
        mean2 = x[:, 1:2]
        std1 = x[:, 2:3]
        std2 = x[:, 3:4]

        # 2. 计算额外的相对特征
        mean_diff = mean1 - mean2
        # 计算标准差比率，增加 epsilon 防止除零和获得更稳定的梯度
        std_ratio = std1 / (std2 + self.eps)
        # 或者使用对数比率可能更稳定: log_std_ratio = torch.log(std1 + self.eps) - torch.log(std2 + self.eps)

        # 3. 拼接所有特征作为 MLP 的输入 (shape [batch_size, 6])
        inputs = torch.cat((mean1, mean2, std1, std2, mean_diff, std_ratio), dim=1)

        # 4. 通过网络传递增强后的输入
        raw_output = self.MLP(inputs)

        # 5. 分离并处理输出
        mean_star = raw_output[:, 0:1]
        std_star_raw = raw_output[:, 1:2]
        std_star = F.softplus(std_star_raw)

        return mean_star, std_star


class DistributionSelector3(nn.Module):
    def __init__(self, hidden_dim1=30, hidden_dim2=6, output_dim=2):
        """
        Args:
            hidden_dim1 (int): 第一个隐藏层的维度
            hidden_dim2 (int): 第二个隐藏层的维度 (如果为 0 或 None 则不使用)
            output_dim (int): 输出维度 (默认是 2: mean*, std*_raw)
        """
        super(DistributionSelector3, self).__init__()
        # 内部输入维度现在是 6: mean1, mean2, std1, std2, mean_diff, std_ratio
        input_dim = 6
        self.use_hidden_layer2 = hidden_dim2 is not None and hidden_dim2 > 0
        self.eps = 1e-6 # 用于计算比率时的数值稳定性

        self.MLP = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim)
        )

    def forward(self, PT1_score, PT2_score, stu_score):
        # 0. 构造x
        mean1 = torch.mean(PT1_score, dim=1, keepdim=True)
        std1 = torch.std(PT1_score, dim=1, keepdim=True)

        mean2 = torch.mean(PT2_score, dim=1, keepdim=True)
        std2 = torch.std(PT2_score, dim=1, keepdim=True)

        mean_stu = torch.mean(stu_score, dim=1, keepdim=True)
        std_stu = torch.std(stu_score, dim=1, keepdim=True)
        
        std1 = std1 + self.eps
        std2 = std2 + self.eps
        std_stu = std_stu + self.eps

        x = torch.cat((mean1, mean2, mean_stu, std1, std2, std_stu), dim=1)
        
        # 1. 从输入 x (shape [batch_size, 4]) 中解包原始统计量
        # 使用切片保持维度 [batch_size, 1]
        mean1 = x[:, 0:1]
        mean2 = x[:, 1:2]
        mean_stu = x[:, 2:3]
        std1 = x[:, 3:4]
        std2 = x[:, 4:5]
        std_stu = x[:, 5:6]
        
        mean_stu = mean_stu.detach()
        std_stu = std_stu.detach()

        # 3. 拼接所有特征作为 MLP 的输入 (shape [batch_size, 6])
        inputs = torch.cat((mean1, mean2, mean_stu, std1, std2, std_stu), dim=1)

        # 4. 通过网络传递增强后的输入
        raw_output = self.MLP(inputs)

        # 5. 分离并处理输出
        mean_star = raw_output[:, 0:1]
        std_star_raw = raw_output[:, 1:2]
        std_star = F.softplus(std_star_raw)

        return mean_star, std_star

class BasicDistributionSelector(nn.Module):
    def __init__(self, mode='PT1', eps=1e-6):
        """
        Initializes the BasicDistributionSelector.

        Args:
            mode (str): The mapping mode. Options are:
                'PT1': Map both score sets to the distribution of PT1_score (row-wise).
                'PT2': Map both score sets to the distribution of PT2_score (row-wise).
                'SND': Map both score sets to the Standard Normal Distribution (Z-score, row-wise).
                'MM': Apply Min-Max normalization to each score set independently (row-wise) to range [0, 1].
            eps (float): A small epsilon value to prevent division by zero in std dev and min-max scaling.
        """
        super(BasicDistributionSelector, self).__init__()

        if mode not in ['PT1', 'PT2', 'SND', 'MM']:
            raise ValueError(f"Unsupported mode: {mode}. Choose from 'PT1', 'PT2', 'SND', 'MM'.")
        self.mode = mode
        self.eps = eps

    def forward(self, PT1_score, PT2_score):
        """
        Applies the selected distribution mapping/normalization.

        Args:
            PT1_score (torch.Tensor): First set of scores, shape [batch_size, nneg + 1].
            PT2_score (torch.Tensor): Second set of scores, shape [batch_size, nneg + 1].

        Returns:
            tuple(torch.Tensor, torch.Tensor): (PT1_score_mapped, PT2_score_mapped)
                                              The scores after applying the transformation according to self.mode.
                                              Shape: [batch_size, nneg + 1] each.
        """

        # Calculate row-wise statistics for original distributions
        mean1 = torch.mean(PT1_score, dim=1, keepdim=True)
        std1 = torch.std(PT1_score, dim=1, keepdim=True) + self.eps # Add eps for stability

        mean2 = torch.mean(PT2_score, dim=1, keepdim=True)
        std2 = torch.std(PT2_score, dim=1, keepdim=True) + self.eps # Add eps for stability

        if self.mode == 'PT1':
            # Target distribution is PT1's distribution
            target_mean = mean1
            target_std = std1

            # Map PT1 to PT1 (identity mapping, but calculated for clarity)
            # z_score1 = (PT1_score - mean1) / std1
            # PT1_score_mapped = z_score1 * target_std + target_mean
            PT1_score_mapped = PT1_score # Simplified

            # Map PT2 to PT1's distribution
            z_score2 = (PT2_score - mean2) / std2
            PT2_score_mapped = z_score2 * target_std + target_mean

        elif self.mode == 'PT2':
            # Target distribution is PT2's distribution
            target_mean = mean2
            target_std = std2

            # Map PT1 to PT2's distribution
            z_score1 = (PT1_score - mean1) / std1
            PT1_score_mapped = z_score1 * target_std + target_mean

            # Map PT2 to PT2 (identity mapping, but calculated for clarity)
            # z_score2 = (PT2_score - mean2) / std2
            # PT2_score_mapped = z_score2 * target_std + target_mean
            PT2_score_mapped = PT2_score # Simplified

        elif self.mode == 'SND': # Standard Normal Distribution (Z-score)
            # Target mean = 0, Target std = 1
            PT1_score_mapped = (PT1_score - mean1) / std1
            PT2_score_mapped = (PT2_score - mean2) / std2

        elif self.mode == 'MM': # Min-Max Normalization per row
            # Normalize PT1
            min1 = torch.min(PT1_score, dim=1, keepdim=True)[0]
            max1 = torch.max(PT1_score, dim=1, keepdim=True)[0]
            range1 = max1 - min1 + self.eps # Add eps for stability if max == min
            PT1_score_mapped = (PT1_score - min1) / range1

            # Normalize PT2
            min2 = torch.min(PT2_score, dim=1, keepdim=True)[0]
            max2 = torch.max(PT2_score, dim=1, keepdim=True)[0]
            range2 = max2 - min2 + self.eps # Add eps for stability if max == min
            PT2_score_mapped = (PT2_score - min2) / range2

        else:
            # This case should not be reached due to check in __init__
            raise ValueError(f"Unsupported mode: {self.mode}")

        return PT1_score_mapped, PT2_score_mapped


        
class teacher_score_adapter(nn.Module):
    def __init__(self, args): # args 可以是配置对象，这里暂时未使用
        super(teacher_score_adapter, self).__init__()
        self.args = args # 如果需要存储配置
        # self.distribution_selector = DistributionSelector(hidden_dim1=16, hidden_dim2=8, output_dim=2)
        # self.distribution_selector = DistributionSelectorv1(hidden_dim1=16, hidden_dim2=8, output_dim=2)
        self.distribution_selector = DistributionSelector2(hidden_dim1=16, hidden_dim2=8, output_dim=2)
        # self.distribution_selector = DistributionSelector3(hidden_dim1=16, hidden_dim2=8, output_dim=2)
        # self.distribution_selector = BasicDistributionSelector(mode='PT1')
        self.eps = 1e-6 # 用于防止除以零的小常数

    def forward(self, PT1_score, PT2_score, stu_score):
        """
        Args:
            PT1_score (torch.Tensor): 第一个教师模型的分数, shape [batch_size, nneg + 1]
            PT2_score (torch.Tensor): 第二个教师模型的分数, shape [batch_size, nneg + 1]

        Returns:
            tuple(torch.Tensor, torch.Tensor): (PT1_score_mapped, PT2_score_mapped)
                                              映射到目标分布后的分数
                                              shape: [batch_size, nneg + 1] each
        """
        # 1. 分别提取 PT1_score 和 PT2_score 在每一行的 mean 和 std
        #    keepdim=True 保持维度为 [batch_size, 1]，方便后续广播和拼接
        mean1 = torch.mean(PT1_score, dim=1, keepdim=True)
        std1 = torch.std(PT1_score, dim=1, keepdim=True)

        mean2 = torch.mean(PT2_score, dim=1, keepdim=True)
        std2 = torch.std(PT2_score, dim=1, keepdim=True)

        std1 = std1 + self.eps
        std2 = std2 + self.eps
        
        if self.distribution_selector.__class__.__name__ == 'BasicDistributionSelector':
            PT1_score_mapped, PT2_score_mapped = self.distribution_selector(PT1_score, PT2_score)
            return PT1_score_mapped, PT2_score_mapped
        elif self.distribution_selector.__class__.__name__ == 'DistributionSelector3':
            target_mean, target_std = self.distribution_selector(PT1_score, PT2_score, stu_score)
        else:
            target_mean, target_std = self.distribution_selector(PT1_score, PT2_score)

        
        # 再次确保 target_std 不会太接近 0 导致后续计算不稳定
        target_std = target_std + self.eps

        # 3. 把 PT1_score 和 PT2_score 映射到目标分布
        #    映射公式: mapped = ((original - mean_orig) / std_orig) * std_target + mean_target
        #    利用广播机制 (broadcasting)

        # 映射 PT1_score
        z_score1 = (PT1_score - mean1) / std1 # 标准化 (Z-score)
        PT1_score_mapped = z_score1 * target_std + target_mean # 缩放到目标分布

        # 映射 PT2_score
        z_score2 = (PT2_score - mean2) / std2 # 标准化 (Z-score)
        PT2_score_mapped = z_score2 * target_std + target_mean # 缩放到目标分布

        # 4. 返回映射后的分数
        return PT1_score_mapped, PT2_score_mapped



class teacher_score_fusion(nn.Module):
    def __init__(self, args):
        super(teacher_score_fusion, self).__init__()
        self.args = args
        self.teacher_score_adapter = teacher_score_adapter(args)
        self.weight_threshold = 0.99999

    def forward(self, eh, er, et, ehr_, et_, stu_score, PT1_score, PT2_score, weight=None, data=None):
        PT1_score_mapped, PT2_score_mapped = self.teacher_score_adapter(PT1_score, PT2_score, stu_score)
        
        if weight is None:
            fusion_score = (PT1_score_mapped + PT2_score_mapped) / 2
        else:
            weights = weight(eh, er, et, stu_score, PT1_score=PT1_score_mapped, PT2_score=PT2_score_mapped, data=data)
            
            weight_PT1_orig = weights[..., 0]
            weight_PT2_orig = weights[..., 1]
            
            # condition = (weight_PT1_orig > self.weight_threshold) | (weight_PT2_orig > self.weight_threshold)
            # half_weights = torch.full_like(weight_PT1_orig, 0.5)
            # weight_PT1 = torch.where(condition, half_weights, weight_PT1_orig)
            # weight_PT2 = torch.where(condition, half_weights, weight_PT2_orig)
            
            fusion_score = weight_PT1_orig * PT1_score + weight_PT2_orig * PT2_score_mapped  # 计算加权平均分数
    
        return fusion_score, weights