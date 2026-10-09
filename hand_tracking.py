import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

# 1. 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 2. 配置 RealSense 数据流管道
pipeline = rs.pipeline()
config = rs.config()

WIDTH, HEIGHT, FPS = 640, 480, 30
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

# 3. 启动相机
pipeline.start(config)

try:
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        print("相机流已启动，在显示窗口按下 'q' 键即可退出。")

        while True:
            # 获取帧流
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # 格式转换
            image = np.asanyarray(color_frame.get_data())
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            # MediaPipe 推理
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True

            # 绘制手部骨骼节点
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            # 实时渲染画面
            cv2.imshow('D435i Real-time Hand Tracking', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()