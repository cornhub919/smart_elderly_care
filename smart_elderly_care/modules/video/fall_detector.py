"""
视频跌倒检测模块
使用 MediaPipe Pose 进行人体姿态估计，结合规则判断检测跌倒
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time


@dataclass
class FallEvent:
    """跌倒事件数据类"""
    timestamp: float
    confidence: float
    center_height: float
    body_angle: float
    is_falling: bool
    stillness_duration: float = 0


class FallDetector:
    """跌倒检测器"""
    
    def __init__(self, config: dict = None):
        self.config = config or {
            "fall_threshold_height": 0.3,
            "fall_threshold_time": 1.0,
            "stillness_threshold": 60,
            "frame_skip": 5,
        }
        
        # 初始化 MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 状态跟踪
        self.prev_center = None
        self.prev_time = None
        self.fall_start_time = None
        self.stillness_start_time = None
        self.fall_detected = False
        
        # 历史记录
        self.center_history = []
        self.angle_history = []
        
    def calculate_body_center(self, landmarks) -> Tuple[float, float]:
        """计算人体重心位置（基于关键点）"""
        # 使用肩膀、臀部、膝盖的中点计算重心
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # 计算躯干中心
        center_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4
        center_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
        
        return center_x, center_y
    
    def calculate_body_angle(self, landmarks) -> float:
        """计算躯干与垂直方向的夹角（度）"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # 计算肩膀中点和臀部中点
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_mid_y = (left_hip.y + right_hip.y) / 2
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        hip_mid_x = (left_hip.x + right_hip.x) / 2
        
        # 计算躯干向量
        dx = shoulder_mid_x - hip_mid_x
        dy = shoulder_mid_y - hip_mid_y
        
        # 计算与垂直方向的夹角
        angle = np.abs(np.arctan2(dx, dy)) * 180 / np.pi
        return angle
    
    def calculate_velocity(self, current_center: Tuple[float, float], 
                          current_time: float) -> float:
        """计算重心下降速度"""
        if self.prev_center is None or self.prev_time is None:
            return 0
        
        dt = current_time - self.prev_time
        if dt <= 0:
            return 0
        
        # 计算垂直方向速度（y轴向下为正）
        dy = current_center[1] - self.prev_center[1]
        velocity = dy / dt
        
        return velocity
    
    def detect_fall(self, frame) -> Tuple[bool, Optional[FallEvent], dict]:
        """
        检测是否发生跌倒
        
        Args:
            frame: 输入视频帧
            
        Returns:
            Tuple[bool, Optional[FallEvent], dict]: 
                - 是否检测到跌倒
                - 跌倒事件详情
                - 分析详情字典
        """
        # 转换颜色空间
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 姿态估计
        results = self.pose.process(rgb_frame)
        
        analysis = {
            "center_height": None,
            "body_angle": None,
            "velocity": None,
            "stillness_duration": 0,
            "landmarks_detected": False,
        }
        
        if not results.pose_landmarks:
            return False, None, analysis
        
        landmarks = results.pose_landmarks.landmark
        analysis["landmarks_detected"] = True
        
        current_time = time.time()
        
        # 计算关键指标
        center_x, center_y = self.calculate_body_center(landmarks)
        body_angle = self.calculate_body_angle(landmarks)
        velocity = self.calculate_velocity((center_x, center_y), current_time)
        
        analysis["center_height"] = center_y
        analysis["body_angle"] = body_angle
        analysis["velocity"] = velocity
        
        # 更新历史记录
        self.center_history.append((center_x, center_y, current_time))
        self.angle_history.append((body_angle, current_time))
        
        # 保持历史记录长度
        max_history = 100
        if len(self.center_history) > max_history:
            self.center_history = self.center_history[-max_history:]
            self.angle_history = self.angle_history[-max_history:]
        
        # 跌倒判断逻辑
        fall_indicators = []
        
        # 1. 重心高度快速下降
        if velocity > 0.5:  # 快速下降
            fall_indicators.append(("fast_descent", 0.8))
        
        # 2. 躯干角度接近水平
        if body_angle > 60:  # 角度大于60度
            fall_indicators.append(("horizontal_body", 0.7))
        
        # 3. 重心位置较低
        if center_y > 0.6:  # 重心在画面下方
            fall_indicators.append(("low_center", 0.6))
        
        # 4. 检测静止状态
        if len(self.center_history) >= 10:
            recent_centers = self.center_history[-10:]
            movement = sum([np.sqrt((c[0]-recent_centers[0][0])**2 + 
                                   (c[1]-recent_centers[0][1])**2) 
                          for c in recent_centers])
            if movement < 0.05:  # 几乎没有移动
                if self.stillness_start_time is None:
                    self.stillness_start_time = current_time
                else:
                    stillness_duration = current_time - self.stillness_start_time
                    analysis["stillness_duration"] = stillness_duration
                    if stillness_duration > 5:  # 静止超过5秒
                        fall_indicators.append(("prolonged_stillness", 0.5))
            else:
                self.stillness_start_time = None
        
        # 综合判断
        if len(fall_indicators) >= 2:
            # 计算置信度
            confidence = sum([ind[1] for ind in fall_indicators]) / len(fall_indicators)
            
            if self.fall_start_time is None:
                self.fall_start_time = current_time
            else:
                fall_duration = current_time - self.fall_start_time
                if fall_duration > self.config["fall_threshold_time"]:
                    self.fall_detected = True
                    event = FallEvent(
                        timestamp=current_time,
                        confidence=min(confidence, 1.0),
                        center_height=center_y,
                        body_angle=body_angle,
                        is_falling=True,
                        stillness_duration=analysis["stillness_duration"]
                    )
                    return True, event, analysis
        else:
            self.fall_start_time = None
            self.fall_detected = False
        
        # 更新前一帧状态
        self.prev_center = (center_x, center_y)
        self.prev_time = current_time
        
        return False, None, analysis
    
    def draw_landmarks(self, frame, results):
        """在帧上绘制姿态关键点"""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=3
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=2
                )
            )
        return frame
    
    def reset(self):
        """重置检测器状态"""
        self.prev_center = None
        self.prev_time = None
        self.fall_start_time = None
        self.stillness_start_time = None
        self.fall_detected = False
        self.center_history = []
        self.angle_history = []
    
    def close(self):
        """释放资源"""
        self.pose.close()


def process_video(video_path: str, output_path: str = None, 
                  config: dict = None) -> List[FallEvent]:
    """
    处理视频文件，检测跌倒事件
    
    Args:
        video_path: 视频文件路径
        output_path: 输出视频路径（可选）
        config: 配置字典
        
    Returns:
        List[FallEvent]: 检测到的跌倒事件列表
    """
    detector = FallDetector(config)
    cap = cv2.VideoCapture(video_path)
    
    fall_events = []
    frame_count = 0
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 输出视频写入器
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 检测跌倒
        is_fall, event, analysis = detector.detect_fall(frame)
        
        if is_fall and event:
            fall_events.append(event)
            # 在帧上标记跌倒
            cv2.putText(frame, "FALL DETECTED!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 绘制姿态关键点
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.pose.process(rgb_frame)
        frame = detector.draw_landmarks(frame, results)
        
        # 显示分析信息
        if analysis["landmarks_detected"]:
            info_text = f"Center: {analysis['center_height']:.2f}, Angle: {analysis['body_angle']:.1f}"
            cv2.putText(frame, info_text, (50, height - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if writer:
            writer.write(frame)
    
    cap.release()
    if writer:
        writer.release()
    detector.close()
    
    return fall_events


if __name__ == "__main__":
    # 测试代码
    print("跌倒检测模块测试")
    detector = FallDetector()
    print("模块初始化成功")
