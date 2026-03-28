import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm

# 导入你的模型结构
from models.iresnet import iresnet50 

# --- 配置 ---
# 默认测试 LFW
DEFAULT_TEST_DIR = "data/val_data"
DEFAULT_ANN_PATH = "data/val_data/lfw_ann.txt"

class VerificationDataset(Dataset):
    def __init__(self, root_dir, ann_file):
        self.root_dir = root_dir
        self.image_list = []
        with open(ann_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(' ') # 注意分隔符，之前说是空格
                if len(parts) < 3: continue
                label = int(parts[0])
                path_a = parts[1]
                path_b = parts[2]
                self.image_list.append((path_a, path_b, label))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        path_a, path_b, label = self.image_list[idx]
        full_path_a = os.path.join(self.root_dir, path_a)
        full_path_b = os.path.join(self.root_dir, path_b)
        
        img_a = self.read_img(full_path_a)
        img_b = self.read_img(full_path_b)
        
        return img_a, img_b, torch.tensor(label, dtype=torch.float)

    def read_img(self, path):
        # 1. OpenCV 读取 (默认 BGR)
        img = cv2.imread(path)
        if img is None: return torch.zeros(3, 112, 112)
        
        # 2. BGR -> RGB (这一步不做，分数会很低！)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. 归一化 (0-255 -> -1.0-1.0)
        # 对应训练时的 transforms.Normalize([0.5...], [0.5...])
        img = (img.astype(np.float32) - 127.5) / 128.0
        
        # 4. HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        
        return torch.from_numpy(img).float()
    
def get_accuracy(scores, flags, threshold):
    p = scores > threshold
    t = flags == 1
    # 计算预测正确的数量
    accuracy = (p == t).sum() / len(scores)
    return accuracy

def find_best_threshold(scores, flags):
    # 简单的阈值搜索
    best_acc = 0
    best_thresh = 0
    for t in np.arange(0, 1.0, 0.01):
        acc = get_accuracy(scores, flags, t)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    return best_acc, best_thresh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='weights\mobilefacenet_cbam_epoch_25.pth')
    parser.add_argument('--test_dir', type=str, default=DEFAULT_TEST_DIR)
    parser.add_argument('--ann_file', type=str, default=DEFAULT_ANN_PATH)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Evaluating model: {args.model_path}")
    print(f"📂 Test Data: {args.test_dir}")

    # 1. 初始化模型 (一定要确认是 MobileFaceNet!)
    # ---------------------------------------------------------
    from models.mobilefacenet import MobileFaceNet
    # 务必确保 use_cbam 和 embedding_size 与训练时一致
    model = MobileFaceNet(embedding_size=512, use_cbam=True).to(device)
    
    print(f"📥 正在加载权重: {args.model_path}")

    if not os.path.exists(args.model_path):
        print(f"❌ 文件不存在: {args.model_path}")
        return

    # 加载文件
    checkpoint = torch.load(args.model_path, map_location=device)

    # 🔥🔥🔥 核心修复：自动剥离外壳 🔥🔥🔥
    if isinstance(checkpoint, dict) and 'backbone' in checkpoint:
        print("📦 识别为训练存档 (字典格式)，正在提取 backbone 权重...")
        state_dict = checkpoint['backbone']
    else:
        print("📂 识别为纯权重文件...")
        state_dict = checkpoint

    # 处理 DataParallel 前缀 (如果有 module. 开头)
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v

    # 加载进模型
    try:
        # strict=True 能帮你检查有没有加载错！
        model.load_state_dict(clean_state_dict, strict=True)
        print("✅ 权重加载成功！")
    except Exception as e:
        print(f"❌ 权重加载失败！这通常是因为模型结构不匹配(比如没换成MobileFaceNet)。\n错误信息: {e}")
        return
    
    model.eval()
    # ---------------------------------------------------------

    # 2. 数据加载
    dataset = VerificationDataset(args.test_dir, args.ann_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    sims = []
    labels = []

    print("⚡ Computing embeddings with Flip-Test...")
    with torch.no_grad():
        for img_a, img_b, label in tqdm(loader):
            img_a = img_a.to(device)
            img_b = img_b.to(device)

            # --- 修改开始：Flip Test ---
            # 1. 提取原图特征
            feat_a = model(img_a)
            feat_b = model(img_b)
            
            # 2. 提取翻转图特征 (torch.flip 在维度3上翻转)
            feat_a_flip = model(torch.flip(img_a, dims=[3]))
            feat_b_flip = model(torch.flip(img_b, dims=[3]))
            
            # 3. 融合特征 (简单的相加即可)
            feat_a = feat_a + feat_a_flip
            feat_b = feat_b + feat_b_flip
            # --- 修改结束 ---

            # 4. 归一化 (融合后必须重新归一化)
            feat_a = F.normalize(feat_a)
            feat_b = F.normalize(feat_b)

            # 计算相似度
            cos_sim = (feat_a * feat_b).sum(dim=1)
            
            sims.append(cos_sim.cpu().numpy())
            labels.append(label.numpy())

    # 3. 计算准确率
    sims = np.concatenate(sims)
    labels = np.concatenate(labels)

    acc, thresh = find_best_threshold(sims, labels)
    
    print("-" * 30)
    print(f"✅ Evaluation Result:")
    print(f"🏆 Best Accuracy: {acc*100:.2f}%")
    print(f"🎯 Best Threshold: {thresh:.2f}")
    print("-" * 30)

if __name__ == "__main__":
    main()