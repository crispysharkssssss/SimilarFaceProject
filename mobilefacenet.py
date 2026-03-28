import torch
import torch.nn as nn
import math
# 复用你之前写的 CBAM 模块
from .cbam import CBAM 

# MobileFaceNet 的核心组件：Bottleneck
class Bottleneck(nn.Module):
    def __init__(self, in_planes, exp_planes, out_planes, stride, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_planes == out_planes
        self.use_cbam = use_cbam

        # 1. Pointwise Conv (1x1) - 升维
        self.conv1 = nn.Conv2d(in_planes, exp_planes, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(exp_planes)
        self.prelu1 = nn.PReLU(exp_planes)

        # 2. Depthwise Conv (3x3) - 提取特征
        self.conv2 = nn.Conv2d(exp_planes, exp_planes, 3, stride, 1, groups=exp_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(exp_planes)
        self.prelu2 = nn.PReLU(exp_planes)

        # 3. Pointwise Conv (1x1) - 降维
        self.conv3 = nn.Conv2d(exp_planes, out_planes, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        
        # --- 插入 CBAM ---
        if self.use_cbam:
            self.cbam = CBAM(out_planes, ratio=4) # MobileNet通道少，Ratio改小点避免压太狠

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        # 应用 CBAM
        if self.use_cbam:
            out = self.cbam(out)

        if self.use_res_connect:
            return x + out
        else:
            return out

class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=512, use_cbam=False):
        super(MobileFaceNet, self).__init__()
        
        # MobileFaceNet 标准配置
        # [Input Channels, Output Channels, Stride]
        self.bottleneck_setting = [
            # t: expansion factor, c: output channels, n: number of blocks, s: stride
            [2, 64, 5, 2],
            [4, 128, 1, 2],
            [2, 128, 6, 1],
            [4, 128, 1, 2],
            [2, 128, 2, 1]
        ]

        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu1 = nn.PReLU(64)
        
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.prelu2 = nn.PReLU(64)

        self.layers = self._make_layer(Bottleneck, self.bottleneck_setting, 64, use_cbam)

        # MobileFaceNet 特有的 GDConv (Global Depthwise Conv) 代替 GAP
        self.conv3 = nn.Conv2d(128, 512, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.prelu3 = nn.PReLU(512)
        
        # 这里的 7x7 是指输入图片经过多次下采样后变成 7x7
        # 如果你输入 112x112，这里刚好是 7x7，如果是其他尺寸会报错
        self.conv4 = nn.Conv2d(512, 512, 7, 1, 0, groups=512, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn5 = nn.BatchNorm1d(embedding_size)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting, in_planes, use_cbam):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                stride = s if i == 0 else 1
                exp_planes = in_planes * t
                layers.append(block(in_planes, exp_planes, c, stride, use_cbam))
                in_planes = c
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)
        
        out = self.layers(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.prelu3(out)
        
        out = self.conv4(out)
        out = self.bn4(out)
        
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.bn5(out)
        return out