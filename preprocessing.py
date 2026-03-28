import os
import cv2
import glob
from tqdm import tqdm
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# --- 配置 ---
INPUT_DIR = "data/casia_raw"       # 原始数据路径
OUTPUT_DIR = "data/casia_aligned"  # 输出路径
IMG_SIZE = 112

# 初始化检测器
app = FaceAnalysis(allowed_modules=['detection'])
app.prepare(ctx_id=0, det_size=(640, 640))

def process_dataset(input_root, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    # 获取所有子文件夹 (每个人一个文件夹)
    subfolders = [f.path for f in os.scandir(input_root) if f.is_dir()]
    
    for subfolder in tqdm(subfolders, desc="Processing Identities"):
        identity_name = os.path.basename(subfolder)
        save_dir = os.path.join(output_root, identity_name)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        img_paths = glob.glob(os.path.join(subfolder, "*.jpg")) + glob.glob(os.path.join(subfolder, "*.png"))
        
        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img is None: continue
            
            # 1. 检测人脸
            faces = app.get(img)
            
            if len(faces) > 0:
                # 2. 找最大的人脸
                face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
                
                # 3. 关键点对齐 (Affine Transformation)
                norm_crop = face_align.norm_crop(img, landmark=face.kps, image_size=IMG_SIZE)
                
                # 4. 保存
                save_name = os.path.basename(img_path)
                cv2.imwrite(os.path.join(save_dir, save_name), norm_crop)

if __name__ == "__main__":
    print("开始处理 CASIA-WebFace...")
    process_dataset(INPUT_DIR, OUTPUT_DIR)