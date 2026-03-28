import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # 1. 归一化特征和权重
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # 2. 【关键】防止数值溢出
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # 3. 如果 m=0 (简单模式)，直接返回 s * cosine
        if self.m == 0.0:
            return cosine * self.s

        # 4. ArcFace 核心逻辑
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # 5. 处理 easy_margin (防止初期训练不收敛)
        # 当 cos(theta) > 0 时使用 phi，否则使用 cosine - mm (减去一个惩罚项)
        # 这里简化处理，直接用 hard check
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # 6. 生成 One-hot 并混合
        # 把 label 变成 (B, 1)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        output *= self.s
        
        return output