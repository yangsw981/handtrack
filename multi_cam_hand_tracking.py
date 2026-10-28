import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

# 1. 初始化 MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 2. 查询设备
ctx = rs.context()
devices = ctx.query_devices()
num_devices = len(devices)

if num_devices == 0:
    print("未检测到 RealSense 设备！")
    exit(1)

# ================= 配置参数 =================
# 建议先用 640x480 测试三路 RGBD + 对齐的流畅度；若主板带宽充裕可改为 1280, 720
WIDTH, HEIGHT, FPS = 848, 480, 30
# ============================================

print(f"检测到 {num_devices} 台相机，正在开启 RGB + Depth ({WIDTH}x{HEIGHT} @ {FPS}FPS)...")

pipelines = []
align_objects = []
hands_trackers = []
serials = []
intrinsics_list = []

for i, dev in enumerate(devices):
    serial = dev.get_info(rs.camera_info.serial_number)
    name = dev.get_info(rs.camera_info.name)
    print(f"  [{i + 1}] {name} | SN: {serial}")

    pipeline = rs.pipeline(ctx)
    config = rs.config()
    config.enable_device(serial)

    # 同时开启彩色流与深度流
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    try:
        profile = pipeline.start(config)
        pipelines.append(pipeline)
        serials.append(serial)

        # 为每台相机创建独立的深度对齐器 (将 Depth 对齐到 Color 视角)
        align_objects.append(rs.align(rs.stream.color))

        # 获取对齐基准流（彩色相机）的内参，用于后续计算真实 3D 坐标 (X, Y, Z)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics_list.append(color_profile.get_intrinsics())

        # 为每台相机创建独立的 MediaPipe Hands 实例
        hands_trackers.append(
            mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,  # 轻量模型，降低三路并发 CPU 压力
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5
            )
        )
    except Exception as e:
        print(f"相机 {serial} 启动失败: {e}")

if not pipelines:
    print("没有成功启动任何相机，退出。")
    exit(1)

# 帧缓存：上半部分放彩色图，下半部分放深度伪彩图 (高度为 HEIGHT * 2)
latest_views = [
    np.zeros((HEIGHT * 2, WIDTH, 3), dtype=np.uint8) for _ in range(len(pipelines))
]

window_name = "3x D435i RGB-D 3D Hand Tracking"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("全部启动成功！按 'q' 键退出。")

try:
    while True:
        for idx, (pipe, align, hands, intr, serial) in enumerate(
                zip(pipelines, align_objects, hands_trackers, intrinsics_list, serials)
        ):
            frames = pipe.poll_for_frames()
            if not frames:
                continue

            # 1. 将深度帧对齐到彩色帧
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # 2. 转为 numpy 数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 生成深度伪彩图（仅用于可视化展示，截断在 0~2米 范围内颜色更明显）
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET
            )

            # 3. MediaPipe 手部检测
            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True

            # 4. 绘制关键点并提取 3D 坐标
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 在彩色图上画骨架
                    mp_drawing.draw_landmarks(
                        color_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # 以【食指指尖 (Landmark 8)】为例，获取其 3D 坐标 (X, Y, Z)
                    # MediaPipe 输出的是归一化坐标 [0.0, 1.0]，需乘以宽高转为像素坐标
                    lm_8 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    px = int(lm_8.x * WIDTH)
                    py = int(lm_8.y * HEIGHT)

                    # 确保像素坐标在画面边界内
                    if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                        # 获取该像素点的深度距离（单位：米）
                        dist = depth_frame.get_distance(px, py)

                        if dist > 0:
                            # 将 2D 像素 + 深度 转换为相机坐标系下的 3D 物理坐标 [X, Y, Z] (单位：米)
                            point_3d = rs.rs2_deproject_pixel_to_point(intr, [px, py], dist)
                            X, Y, Z = point_3d

                            # 在指尖旁边显示 3D 坐标 (单位：米)
                            coord_text = f"({X:.2f}, {Y:.2f}, {Z:.2f})m"
                            cv2.circle(color_image, (px, py), 6, (0, 255, 255), -1)
                            cv2.putText(color_image, coord_text, (px + 10, py),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                            # 同时在深度伪彩图上标出对应位置
                            cv2.circle(depth_colormap, (px, py), 6, (255, 255, 255), -1)

            # 标注相机序号
            cv2.putText(color_image, f"Cam {idx + 1} | SN: {serial}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 将当前相机的【彩色图】与【深度图】上下拼接 (vstack)
            cam_view = np.vstack((color_image, depth_colormap))
            latest_views[idx] = cam_view

        # 将三台相机的上下视图横向拼接 (hstack)
        if latest_views:
            combined_all = np.hstack(latest_views)
            # 若分辨率较高，可根据屏幕大小缩放显示
            if WIDTH > 640:
                combined_all = cv2.resize(combined_all, (0, 0), fx=0.7, fy=0.7)
            cv2.imshow(window_name, combined_all)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("正在安全关闭所有相机...")
    for hands in hands_trackers:
        hands.close()
    for pipe in pipelines:
        pipe.stop()
    cv2.destroyAllWindows()