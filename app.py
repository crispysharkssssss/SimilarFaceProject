import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import os
import time

# 导入必要的库
from insightface.app import FaceAnalysis
from models.mobilefacenet import MobileFaceNet
# 导入 MediaPipe 动作活体检测模块 (确保 action_liveness.py 在同级目录)
from action_liveness import ActionLivenessDetector 

# ==========================================
# 1. 系统配置
# ==========================================
# 识别模型路径
REC_MODEL_PATH = "weights/checkpoint_latest.pth" 

PAYMENT_THRESHOLD = 0.36 

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 资源加载
# ==========================================
@st.cache_resource
def load_system():
    print("🚀 正在初始化系统资源...")
    
    # --- A. 加载本地 InsightFace 检测器 ---
    # 获取当前脚本所在的目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接出资源目录路径: ./insightface_assets
    insightface_root = os.path.join(base_dir, 'insightface_assets')
    
    # 初始化 (会自动去 insightface_assets/models/buffalo_l 找模型)
    app = FaceAnalysis(name='buffalo_l', root=insightface_root)
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("✅ 检测模型加载成功 (本地 buffalo_l)")
    
    # --- B. 加载你的 MobileFaceNet 识别模型 ---
    rec_model = MobileFaceNet(embedding_size=512, use_cbam=True).to(DEVICE)
    
    if os.path.exists(REC_MODEL_PATH):
        try:
            ckpt = torch.load(REC_MODEL_PATH, map_location=DEVICE)
            # 智能解包逻辑
            if isinstance(ckpt, dict) and 'backbone' in ckpt:
                state_dict = ckpt['backbone']
            else:
                state_dict = ckpt
            
            # 去除 module. 前缀
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            rec_model.load_state_dict(state_dict, strict=True)
            rec_model.eval()
            print("✅ 识别模型加载成功")
        except Exception as e:
            st.error(f"❌ 识别模型加载失败: {e}")
    else:
        st.warning(f"⚠️ 未找到模型文件 {REC_MODEL_PATH}，将无法进行比对")
    
    return app, rec_model

# ==========================================
# 3. 核心功能函数
# ==========================================

def extract_feature_safe(img_np, app, rec_model):
    """
    安全提取人脸特征
    返回: (特征向量/None, 错误信息)
    """
    # 1. 检测
    faces = app.get(img_np)
    if len(faces) == 0:
        return None, "未检测到人脸"
    
    # 取最大人脸
    face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
    
    # 2. 质量/置信度检查
    # buffalo_l 的 det_score 很准，低于 0.60 说明人脸很糊或不是人脸
    if face.det_score < 0.60:
        return None, f"人脸质量过低 (置信度 {face.det_score:.2f})，请正对屏幕"

    # 3. 对齐
    from insightface.utils import face_align
    aligned_face = face_align.norm_crop(img_np, landmark=face.kps, image_size=112)
    
    # 4. 预处理
    img_tensor = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).float().to(DEVICE)
    img_tensor.div_(255).sub_(0.5).div_(0.5) 
    
    # 5. 提取特征
    with torch.no_grad():
        feature_raw = rec_model(img_tensor)
        feature = F.normalize(feature_raw).cpu().numpy()[0]
        
    return feature, "Success"

def run_camera_liveness_flow():
    """
    启动摄像头 -> 眨眼检测 -> 自动抓拍
    返回: (是否通过, 抓拍的高清帧)
    """
    # 初始化 MediaPipe 检测器
    liveness_detector = ActionLivenessDetector()
    
    # UI 占位符
    frame_window = st.image([])
    status_box = st.empty()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ 无法打开摄像头")
        return False, None

    success = False
    captured_frame = None
    start_time = time.time()
    
    status_box.info("👁️ 请正对摄像头，并做一个【眨眼】的动作...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 镜像，方便用户看
        frame = cv2.flip(frame, 1)
        
        # 1. 运行 MediaPipe 逻辑
        is_live, msg, ear = liveness_detector.process_frame(frame)
        
        # 2. 绘制 UI (在视频上画字)
        display_frame = frame.copy()
        color = (0, 255, 0) if is_live else (0, 165, 255) # 绿/橙
        
        cv2.putText(display_frame, f"EAR: {ear:.2f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(display_frame, msg, (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 更新网页
        frame_window.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        
        # 3. 判定成功
        if is_live:
            success = True
            captured_frame = frame # 保存原始帧用于识别
            status_box.success("✅ 活体检测通过！正在验证身份...")
            time.sleep(0.5) # 停顿一下让用户看到
            break
            
        # 4. 超时 (15秒)
        if time.time() - start_time > 15:
            status_box.error("⏳ 检测超时，请重试")
            break
    
    cap.release()
    return success, captured_frame

# ==========================================
# 4. Streamlit 页面
# ==========================================
st.set_page_config(page_title="安全刷脸支付", layout="wide", page_icon="💳")

# 加载资源 (只加载 检测器 和 识别模型)
app_insight, my_net = load_system()

# 初始化底库
if 'user_db' not in st.session_state:
    st.session_state['user_db'] = {}

st.title("💳 安全刷脸支付演示系统")
st.caption(f"安全策略：配合式活体 (眨眼) | 支付阈值 ({PAYMENT_THRESHOLD})")

col1, col2 = st.columns([1, 1.5])

# --- 左侧：注册 ---
with col1:
    st.header("👤 用户注册")
    reg_name = st.text_input("用户名", "User_001")
    
    # [新增] 选择注册方式
    reg_method = st.radio("注册方式", ["📁 上传图片", "📸 摄像头抓拍"], horizontal=True)
    
    img_np = None # 用于存储待注册的图片数据

    # ================= 替换开始 =================
    
    if reg_method == "📁 上传图片":
        reg_file = st.file_uploader("上传文件", type=['jpg', 'png'])
        if reg_file and st.button("提交注册"):
            image = Image.open(reg_file)
            img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            with st.spinner("正在录入..."):
                feat, msg = extract_feature_safe(img_np, app_insight, my_net)
                if feat is not None:
                    st.session_state['user_db'][reg_name] = feat
                    st.success(f"✅ 用户 {reg_name} 注册成功")
                else:
                    st.error(f"注册失败: {msg}")

    elif reg_method == "📸 摄像头抓拍":
        if st.button("启动抓拍并注册"):
            # 复用已有的活体检测流程，确保底库照片质量高
            success, frame = run_camera_liveness_flow()
            
            if success and frame is not None:
                with st.spinner("正在录入..."):
                    feat, msg = extract_feature_safe(frame, app_insight, my_net)
                    if feat is not None:
                        st.session_state['user_db'][reg_name] = feat
                        st.success(f"✅ 用户 {reg_name} 注册成功")
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=200, caption="注册底图")
                    else:
                        st.error(f"注册失败: {msg}")
    
    # ================= 替换结束 =================

    # 显示底库
    if st.session_state['user_db']:
        st.divider()
        st.write(f"📚 底库人数: {len(st.session_state['user_db'])}")

# --- 右侧：支付 ---
with col2:
    st.header("💸 刷脸支付")
    pay_amount = st.number_input("金额 (￥)", value=99.0)
    
    if st.button("📸 启动支付 (眨眼检测)", type="primary"):
        if not st.session_state['user_db']:
            st.error("⚠️ 请先在左侧注册用户！")
        else:
            # 1. 活体检测
            is_real, face_img = run_camera_liveness_flow()
            
            # 2. 身份验证 (只有活体通过才执行)
            if is_real and face_img is not None:
                # 提取特征
                query_feat, msg = extract_feature_safe(face_img, app_insight, my_net)
                
                if query_feat is None:
                    st.error(f"❌ 识别失败: {msg}")
                else:
                    # 1:N 比对
                    max_score = -1
                    matched_user = "Unknown"
                    
                    for name, db_feat in st.session_state['user_db'].items():
                        score = np.dot(query_feat, db_feat)
                        if score > max_score:
                            max_score = score
                            matched_user = name
                    
                    st.divider()
                    c1, c2 = st.columns([2, 1])
                    c2.metric("相似度得分", f"{max_score:.4f}")
                    
                    if max_score > PAYMENT_THRESHOLD:
                        c1.balloons()
                        c1.success(f"🎉 **支付成功！**")
                        c1.markdown(f"### 欢迎, **{matched_user}**")
                        c1.info(f"扣款: ￥{pay_amount}")
                    else:
                        c1.error("❌ **支付失败**")
                        c1.warning("身份验证未通过 (得分低于阈值)")