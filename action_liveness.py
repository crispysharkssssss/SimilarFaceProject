import cv2
import numpy as np
import mediapipe as mp
import time

class ActionLivenessDetector:
    def __init__(self):
        # 初始化 MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 状态机：0-等待, 1-闭眼检测中, 2-通过
        self.state = 0
        self.blink_start_time = 0
        self.last_action_time = 0
        
        # 眼睛关键点索引 (左眼和右眼)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    def _calc_ear(self, landmarks, eye_indices, w, h):
        """计算眼睛纵横比 (Eye Aspect Ratio)"""
        # 获取关键点坐标
        coords = []
        for idx in eye_indices:
            lm = landmarks[idx]
            coords.append(np.array([lm.x * w, lm.y * h]))
        
        # 垂直距离 (上眼睑到下眼睑)
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        
        # 水平距离 (眼角到眼角)
        h_dist = np.linalg.norm(coords[0] - coords[3])
        
        # EAR公式
        ear = (v1 + v2) / (2.0 * h_dist)
        return ear

    def process_frame(self, img_bgr):
        """
        处理每一帧，返回状态
        Returns: 
            is_live (bool): 是否通过活体
            message (str): 提示信息
            debug_info (dict): 用于绘图的额外信息
        """
        h, w, _ = img_bgr.shape
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        
        if not results.multi_face_landmarks:
            self.state = 0
            return False, "未检测到人脸", None
            
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 计算左右眼 EAR
        left_ear = self._calc_ear(landmarks, self.LEFT_EYE, w, h)
        right_ear = self._calc_ear(landmarks, self.RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # 眨眼阈值 (小于 0.22 认为闭眼，大于 0.28 认为睁眼)
        BLINK_THRESH = 0.22
        OPEN_THRESH = 0.28
        
        status_msg = "请对着摄像头眨眨眼"
        is_live = False
        
        # 状态机逻辑
        current_time = time.time()
        
        if self.state == 0: # 等待睁眼
            if avg_ear > OPEN_THRESH:
                self.state = 1 # 眼睛睁开了，准备检测眨眼
                status_msg = "检测中：请眨眼..."
                
        elif self.state == 1: # 等待闭眼
            if avg_ear < BLINK_THRESH:
                self.state = 2 # 捕捉到闭眼
                self.blink_start_time = current_time
                status_msg = "检测到闭眼动作..."
            elif current_time - self.last_action_time > 5.0:
                # 超时重置
                self.state = 0
                
        elif self.state == 2: # 等待再次睁眼 (完成眨眼动作)
            if avg_ear > OPEN_THRESH:
                # 动作完成！
                self.state = 3 
                is_live = True
                status_msg = "✅ 活体检测通过！"
            elif current_time - self.blink_start_time > 1.0:
                # 闭眼太久（可能是睡觉或假照片），重置
                self.state = 0
                status_msg = "动作超时，请重试"
        
        elif self.state == 3:
             is_live = True
             status_msg = "✅ 活体检测通过！"

        # 记录心跳
        self.last_action_time = current_time
        
        return is_live, status_msg, avg_ear