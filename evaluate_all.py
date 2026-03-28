import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
from prettytable import PrettyTable

# 引入你的模型
from models.mobilefacenet import MobileFaceNet 

# ================= 配置区域 =================
VAL_ROOT = "data/val_data"

BENCHMARKS = {
    "LFW (基础)":     ("lfw_112x112",       "lfw_ann.txt"),
    "CPLFW (姿态)":   ("cplfw_112x112",     "cplfw_ann.txt"),
    "CALFW (年龄)":   ("calfw_112x112",     "calfw_ann.txt"),
    "AgeDB-30 (年龄)":("agedb_30_112x112",  "agedb_30_ann.txt"),
}
# ===========================================

class VerificationDataset(Dataset):
    def __init__(self, root_dir, ann_file):
        self.root_dir = root_dir
        self.image_list = []
        if not os.path.exists(ann_file):
            print(f"⚠️ Warning: Annotation file not found: {ann_file}")
            return
        with open(ann_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
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
        
        # 返回文件路径，方便追踪 Bad Case
        return img_a, img_b, torch.tensor(label, dtype=torch.float), path_a, path_b

    def read_img(self, path):
        img = cv2.imread(path)
        if img is None: return torch.zeros(3, 112, 112)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).float()

def calculate_tar_at_far(scores, flags, far_target):
    pos_scores = scores[flags == 1]
    neg_scores = scores[flags == 0]
    neg_scores = np.sort(neg_scores)[::-1]
    num_neg = len(neg_scores)
    hard_negative_index = int(num_neg * far_target)
    if hard_negative_index >= num_neg: hard_negative_index = num_neg - 1
    threshold = neg_scores[hard_negative_index]
    tar = (pos_scores > threshold).sum() / len(pos_scores)
    return tar, threshold

def find_best_threshold(scores, flags):
    best_acc = 0
    best_thresh = 0
    for t in np.arange(0.1, 0.6, 0.005):
        p = scores > t
        t_flag = flags == 1
        acc = (p == t_flag).sum() / len(scores)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    return best_acc, best_thresh

def run_evaluation(model, dataset_name, img_folder, ann_file, device):
    full_img_path = VAL_ROOT
    full_ann_path = os.path.join(VAL_ROOT, ann_file)
    dataset = VerificationDataset(full_img_path, full_ann_path)
    if len(dataset) == 0: return None
        
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    
    sims = []
    labels = []
    
    # 存储所有的正样本结果 (分数, 图A路径, 图B路径)
    positive_pairs = []

    with torch.no_grad():
        # 注意：这里解包变成了 5 个变量
        for img_a, img_b, label, path_a_tuple, path_b_tuple in tqdm(loader, desc=f"Testing {dataset_name}", leave=False):
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            
            # Flip Test
            feat_a = model(img_a) + model(torch.flip(img_a, dims=[3]))
            feat_b = model(img_b) + model(torch.flip(img_b, dims=[3]))
            
            feat_a = F.normalize(feat_a)
            feat_b = F.normalize(feat_b)
            
            cos_sim = (feat_a * feat_b).sum(dim=1)
            
            # 收集结果
            sims_np = cos_sim.cpu().numpy()
            label_np = label.numpy()
            
            sims.append(sims_np)
            labels.append(label_np)

            # 收集正样本 (同一个人)
            for k in range(len(label_np)):
                if label_np[k] == 1: # 是同一个人
                    score = sims_np[k]
                    p_a = path_a_tuple[k]
                    p_b = path_b_tuple[k]
                    positive_pairs.append((score, p_a, p_b))

    sims = np.concatenate(sims)
    labels = np.concatenate(labels)
    
    # 1. 打印 Bad Cases (分数最低的 Top 5 正样本)
    # 按分数从小到大排序 (分数越低越糟糕)
    positive_pairs.sort(key=lambda x: x[0])
    
    print(f"\n🚨 {dataset_name} - 最难识别的 5 对样本 (Bad Cases):")
    print("-" * 60)
    for i in range(min(5, len(positive_pairs))):
        score, p_a, p_b = positive_pairs[i]
        print(f"No.{i+1} | 相似度: {score:.4f} (极低) | 图片: {p_a} <--> {p_b}")
    print("-" * 60)
    
    # 2. 计算指标
    acc, best_thresh = find_best_threshold(sims, labels)
    tar_1e3, thresh_1e3 = calculate_tar_at_far(sims, labels, 1e-3)
    
    return {"acc": acc, "best_thresh": best_thresh, "tar_1e3": tar_1e3, "thresh_1e3": thresh_1e3}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    
    # 如果命令行里加了 --no_cbam，则 args.no_cbam 为 True
    parser.add_argument('--no_cbam', action='store_true', help="Set this if the model was trained WITHOUT CBAM")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"📥 Loading Model: {args.model_path}")
    use_cbam_flag = not args.no_cbam
    
    print(f"⚙️ Model Configuration: use_cbam = {use_cbam_flag}")
    from models.mobilefacenet import MobileFaceNet
    model = MobileFaceNet(embedding_size=512, use_cbam=use_cbam_flag).to(device)
    
    ckpt = torch.load(args.model_path, map_location=device)
    state = ckpt['backbone'] if isinstance(ckpt, dict) and 'backbone' in ckpt else ckpt
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    
    results = []
    print("\n🚀 开始批量评估...")
    
    table = PrettyTable(["Dataset", "Acc", "Best Thresh", "TAR@FAR=0.1%"])

    for name, (img_dir, ann_file) in BENCHMARKS.items():
        res = run_evaluation(model, name, img_dir, ann_file, device)
        if res:
            acc_str = f"{res['acc']*100:.2f}%"
            thresh_str = f"{res['best_thresh']:.3f}"
            tar_str = f"{res['tar_1e3']*100:.2f}%"
            
            results.append([name, acc_str, thresh_str, tar_str])
            table.add_row([name, acc_str, thresh_str, tar_str])

    print("\n" + "="*50)
    print("🏆 FINAL REPORT")
    print("="*50)
    print(table)

if __name__ == "__main__":
    main()