import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from models.mobilefacenet import MobileFaceNet

# ================= 配置区域 =================
# 1. 你的自定义数据目录
DATA_DIR = "data/star_alike"

# 2. 模型权重路径
MODEL_PATH = "weights/checkpoint_latest.pth" 

# 3. 安全阈值 
THRESHOLD = 0.70 

# 4. 图片总对数
NUM_PAIRS = 7
# ===========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    print(f" 加载模型: {MODEL_PATH}")
    model = MobileFaceNet(embedding_size=512, use_cbam=True).to(device)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("找不到权重文件")
        
    ckpt = torch.load(MODEL_PATH, map_location=device)
    # 智能解包
    if isinstance(ckpt, dict) and 'backbone' in ckpt:
        state = ckpt['backbone']
    else:
        state = ckpt
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def preprocess(img_path):
    """
    读取并预处理图片
    注意：为了最佳效果，你的图片最好已经裁剪出了人脸区域。
    如果没有裁剪，这里会直接Resize整个图，可能影响精度。
    """
    if not os.path.exists(img_path):
        return None, None
        
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: return None, None
    
    # 转RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize 到 112x112 (模型输入标准)
    img_resized = cv2.resize(img_rgb, (112, 112))
    
    # 归一化 [-1, 1]
    img_float = (img_resized.astype(np.float32) - 127.5) / 128.0
    
    # 转 Tensor (CHW)
    img_tensor = np.transpose(img_float, (2, 0, 1))
    tensor = torch.from_numpy(img_tensor).unsqueeze(0).float().to(device)
    
    return tensor, img_rgb

def main():
    if not os.path.exists(DATA_DIR):
        print(f" 错误：请先创建文件夹 {DATA_DIR} 并放入图片")
        return

    model = load_model()
    
    results = []
    scores = []
    
    print(f"\n 开始验证 {NUM_PAIRS} 组相似人脸...")
    print(f" 安全阈值: {THRESHOLD} (低于此值视为'不同人'，即拦截成功)\n")

    # 创建表格
    table = PrettyTable(["Pair ID", "File A", "File B", "Similarity", "Result", "Status"])

    success_count = 0

    for i in range(1, NUM_PAIRS + 1):
        # 尝试匹配常见后缀
        def find_file(base_name):
            for ext in ['.jpg', '.png', '.jpeg']:
                p = os.path.join(DATA_DIR, base_name + ext)
                if os.path.exists(p): return p
            return None

        path_a = find_file(f"{i}_1")
        path_b = find_file(f"{i}_2")
        
        if not path_a or not path_b:
            print(f" 跳过第 {i} 组：文件缺失")
            continue

        # 预处理
        t_a, img_a = preprocess(path_a)
        t_b, img_b = preprocess(path_b)
        
        # 推理
        with torch.no_grad():
            feat_a = F.normalize(model(t_a)).cpu().numpy()[0]
            feat_b = F.normalize(model(t_b)).cpu().numpy()[0]
            
        # 计算相似度
        score = np.dot(feat_a, feat_b)
        scores.append(score)
        
        # 判定
        # 因为我们知道这些都是"不同人"，所以：
        # 分数 < 阈值 -> 判定为不同人 -> 拦截成功 (Success)
        # 分数 > 阈值 -> 判定为同一个人 -> 误识 (Fail)
        if score < THRESHOLD:
            res_str = "Different"
            status = " Success"
            success_count += 1
        else:
            res_str = "Same"
            status = " Fail"
            
        results.append((img_a, img_b, score, status, i))
        
        table.add_row([f"Pair {i}", os.path.basename(path_a), os.path.basename(path_b), 
                       f"{score:.4f}", res_str, status])

    print(table)
    print(f"\n 统计结果: 成功拦截 {success_count}/{len(results)} 对相似人脸")
    
    # --- 可视化绘图 ---
    print(" 正在生成结果图...")
    fig, axes = plt.subplots(len(results), 3, figsize=(12, 3 * len(results)))
    # 如果只有一组数据，axes需要特殊处理
    if len(results) == 1: axes = np.expand_dims(axes, 0)
    
    fig.suptitle(f"Verification on Custom Look-alike Dataset\n(Threshold < {THRESHOLD} means Safe)", fontsize=16, y=0.99)

    for idx, (im_a, im_b, score, status, pair_id) in enumerate(results):
        # 左图
        axes[idx, 0].imshow(im_a)
        axes[idx, 0].set_title(f"Pair {pair_id}_1")
        axes[idx, 0].axis('off')
        
        # 中间：分数条
        axes[idx, 1].axis('off')
        axes[idx, 1].set_xlim(0, 1)
        axes[idx, 1].set_ylim(0, 1)
        
        # 绘制颜色：成功绿，失败红
        color = 'green' if score < THRESHOLD else 'red'
        
        axes[idx, 1].text(0.5, 0.6, f"Similarity:\n{score:.4f}", 
                          fontsize=16, ha='center', va='center', fontweight='bold', color='black')
        
        axes[idx, 1].text(0.5, 0.3, status, 
                          fontsize=14, ha='center', va='center', color=color, fontweight='bold')
        
        # 右图
        axes[idx, 2].imshow(im_b)
        axes[idx, 2].set_title(f"Pair {pair_id}_2")
        axes[idx, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("my_data_result.png", dpi=300)
    print(" 结果图已保存为: my_data_result.png")
    plt.show()

if __name__ == "__main__":
    main()