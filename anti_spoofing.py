import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# ==========================================
# 1. MiniFASNet V1 (无 SE 模块版)
#    专门适配 1.7MB 权重文件
# ==========================================

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# 🔥 修改点：移除了 SE 模块，变回普通的深度可分离卷积 🔥
class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseConv, self).__init__()
        self.conv_dw = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        # 这里没有 SE 模块了
        self.conv_pw = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)
        self.bn_pw = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        # 直接前向传播，不经过 SE
        x = self.conv_dw(x)
        x = self.bn_dw(x)
        x = self.relu(x)
        x = self.conv_pw(x)
        x = self.bn_pw(x)
        return x

class MiniFASNetV1(nn.Module):
    def __init__(self, embedding_size=128, conv6_kernel=(5, 5)):
        super(MiniFASNetV1, self).__init__()
        
        self.conv1 = ConvBnReLU(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBnReLU(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBnReLU(64, 64, kernel_size=3, stride=2, padding=1)
        
        # 使用不带 SE 的 DepthwiseConv
        self.conv4 = DepthwiseConv(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = DepthwiseConv(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = DepthwiseConv(128, 128, kernel_size=3, stride=2, padding=1)
        
        self.conv_final = nn.Conv2d(128, embedding_size, kernel_size=conv6_kernel, bias=False)
        self.bn_final = nn.BatchNorm1d(embedding_size)
        
        self.linear = nn.Linear(embedding_size, 3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv_final(out)
        
        out = out.view(out.size(0), -1)
        out = self.bn_final(out)
        logits = self.linear(out)
        return logits

# ==========================================
# 2. 封装推理类
# ==========================================
class LivenessDetector:
    def __init__(self, weights_path, device):
        self.device = device
        # 使用 V1 (No-SE) 模型
        self.model = MiniFASNetV1(embedding_size=128).to(device)
        self.model_loaded = False
        
        print(f"🧐 正在加载 1.7MB 版本权重: {weights_path}")

        try:
            state_dict = torch.load(weights_path, map_location=device)
            # 移除 module. 前缀
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # 加载权重
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            self.model_loaded = True
            print(f"✅ 活体检测模型加载成功 (MiniFASNetV1 No-SE)")
            
        except Exception as e:
            print(f"❌ 活体检测模型加载失败: {e}")
            print("如果报错 Missing keys: ...se... 说明代码里有多余的 SE 层")
            print("如果报错 Unexpected keys: ...se... 说明权重里有多余的 SE 层")

    def check(self, img_bgr, face_box):
        if not self.model_loaded: return True, 1.0
        
        # 1. 裁剪
        x1, y1, x2, y2 = face_box
        w, h = x2 - x1, y2 - y1
        scale = 2.7
        cx, cy = x1 + w//2, y1 + h//2
        rw, rh = int(w * scale), int(h * scale)
        x1_src, y1_src = max(0, cx - rw//2), max(0, cy - rh//2)
        x2_src, y2_src = min(img_bgr.shape[1], cx + rw//2), min(img_bgr.shape[0], cy + rh//2)
        crop_img = np.zeros((rh, rw, 3), dtype=np.uint8)
        x1_dst, y1_dst = x1_src - (cx - rw//2), y1_src - (cy - rh//2)
        x2_dst, y2_dst = x1_dst + (x2_src - x1_src), y1_dst + (y2_src - y1_src)
        crop_img[y1_dst:y2_dst, x1_dst:x2_dst] = img_bgr[y1_src:y2_src, x1_src:x2_src]
        
        # 2. 预处理
        try:
            img = cv2.resize(crop_img, (80, 80))
            img = torch.from_numpy(img.transpose((2, 0, 1))).float()
            img = img.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(img)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Label 1 = 真人
            real_score = probs[1]
            # 阈值
            is_real = real_score > 0.90
            
            return is_real, real_score
            
        except Exception as e:
            print(f"推理错误: {e}")
            return True, 0.5