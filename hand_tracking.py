import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

# 1. 验证并初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 2. 配置 RealSense 数据流管道
pipeline = rs.pipeline()
config = rs.config()

# 推荐分辨率与帧率：640x480 @ 30FPS，兼容性最好且延迟低
WIDTH, HEIGHT, FPS = 640, 480, 30
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

print("正在启动 RealSense D435i 相机...")
pipeline.start(config)

# 跳过前 10 帧，等待相机自动曝光与自动白平衡稳定
for _ in range(10):
    pipeline.wait_for_frames()

print("相机启动就绪！已开启手部关键点实时检测。")
print("提示：在弹出的图像窗口中按 'q' 键退出程序。")

window_name = "D435i Real-time Hand Tracking"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

try:
    with mp_hands.Hands(
        static_image_mode=False,        # 视频流模式（启用时序跟踪）
        max_num_hands=2,                # 最多同时检测 2 只手
        min_detection_confidence=0.6,   # 检测置信度阈值
        min_tracking_confidence=0.5     # 跟踪置信度阈值
    ) as hands:

        while True:
            # 获取帧数据
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # 转为 numpy 数组 (BGR 格式)
            image = np.asanyarray(color_frame.get_data())

            # MediaPipe 处理要求 RGB 格式，标记不可写以提速
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)

            # 标记回可写状态以便绘制
            image_rgb.flags.writeable = True

            # 若检测到关键点，绘制骨架连接
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            # 实时显示
            cv2.imshow(window_name, image)

            # 检测退出键 'q' (ASCII 113)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    # 释放硬件资源与显示窗口
    print("正在关闭相机与窗口...")
    pipeline.stop()
    cv2.destroyAllWindows()