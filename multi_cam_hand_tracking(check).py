import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

# 1. 初始化 MediaPipe
mp_hands = mp.solutions.hands

# 2. 查询设备
ctx = rs.context()
devices = ctx.query_devices()
num_devices = len(devices)

if num_devices == 0:
    print("未检测到 RealSense 设备！")
    exit(1)

# ================= 配置参数 =================
WIDTH, HEIGHT, FPS = 848, 480, 30

# 遮挡过滤开关与阈值
HIDE_OCCLUDED_POINTS = True   # True: 直接隐藏被遮挡的预测点；False: 用红色小点标出被拦截的遮挡点方便调试
MAX_HAND_RADIUS_M = 0.22      # 手部最大物理跨度 (22cm)：任何关键点偏离手掌基准中心超过此距离即判定为穿透/遮挡
OCCLUSION_MARGIN_M = 0.035    # 遮挡容差 (3.5cm)：若相机实测深度比 MediaPipe 预估深度还靠前超过 3.5cm，说明前面有遮挡物(罐子或手背)
MAX_PATCH_STD_M = 0.015       # 局部深度平整度阈值 (1.5cm)：若关键点周围 3x3 像素深度剧烈跳变，说明在物体边缘，丢弃
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

    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    try:
        profile = pipeline.start(config)
        pipelines.append(pipeline)
        serials.append(serial)

        align_objects.append(rs.align(rs.stream.color))

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics_list.append(color_profile.get_intrinsics())

        hands_trackers.append(
            mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.65,  # 略微提高置信度，减少极端残影
                min_tracking_confidence=0.65
            )
        )
    except Exception as e:
        print(f"相机 {serial} 启动失败: {e}")

if not pipelines:
    print("没有成功启动任何相机，退出。")
    exit(1)


def get_clean_depth(depth_image_mm, px, py, half_win=1):
    """获取关键点周围 (3x3) 区域的深度均值与标准差（单位：米）。

    若存在无效黑洞或处在剧烈跳变的边缘，直接返回 (0.0, 999.0)。
    """
    h, w = depth_image_mm.shape
    x1, x2 = max(0, px - half_win), min(w, px + half_win + 1)
    y1, y2 = max(0, py - half_win), min(h, py + half_win + 1)

    patch = depth_image_mm[y1:y2, x1:x2].astype(np.float32) * 0.001  # 毫米转米
    valid = patch[(patch > 0.15) & (patch < 1.20)]  # 只保留 15cm ~ 1.2m 工作区内的深度

    if len(valid) < 3:  # 有效像素太少（处于红外阴影或盲区）
        return 0.0, 999.0

    return float(np.median(valid)), float(np.std(valid))


latest_views = [
    np.zeros((HEIGHT * 2, WIDTH, 3), dtype=np.uint8) for _ in range(len(pipelines))
]

window_name = "3x D435i Anti-Occlusion 3D Hand Tracking"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("全部启动成功！被遮挡的脑补关键点将被自动剔除。按 'q' 键退出。")

try:
    while True:
        for idx, (pipe, align, hands, intr, serial) in enumerate(
                zip(pipelines, align_objects, hands_trackers, intrinsics_list, serials)
        ):
            frames = pipe.poll_for_frames()
            if not frames:
                continue

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())  # uint16, 单位 mm

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET
            )

            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 1. 先提取所有 21 个点的像素坐标与局部深度
                    pts_2d = []
                    raw_depths = []
                    patch_stds = []

                    for lm in hand_landmarks.landmark:
                        px, py = int(lm.x * WIDTH), int(lm.y * HEIGHT)
                        pts_2d.append((px, py))
                        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                            d_med, d_std = get_clean_depth(depth_image, px, py, half_win=1)
                            raw_depths.append(d_med)
                            patch_stds.append(d_std)
                        else:
                            raw_depths.append(0.0)
                            patch_stds.append(999.0)

                    # 2. 计算手掌核心基准深度 (取手腕0、食指根5、中指根9、小指根17等有效深度的中位数)
                    palm_indices = [0, 1, 5, 9, 13, 17]
                    palm_valid_depths = [raw_depths[i] for i in palm_indices if raw_depths[i] > 0]
                    if not palm_valid_depths:
                        continue  # 连手掌基准都完全看不见，整只手丢弃
                    base_palm_z = float(np.median(palm_valid_depths))

                    # 估算手部 2D 像素尺度，用于将 MediaPipe 的相对 z 转换为近似物理米 (经验比例)
                    # 取手腕(0)到中指根(9)的像素距离对应真实约 0.085 米
                    p0, p9 = pts_2d[0], pts_2d[9]
                    pix_palm_len = np.hypot(p0[0] - p9[0], p0[1] - p9[1])
                    mp_z_scale = (0.085 / max(pix_palm_len, 20.0)) * WIDTH

                    # 3. 逐个关键点进行【物理遮挡与脑补剔除检验】
                    valid_mask = [False] * 21
                    points_3d = [None] * 21

                    for i, lm in enumerate(hand_landmarks.landmark):
                        px, py = pts_2d[i]
                        z_real = raw_depths[i]
                        z_std = patch_stds[i]

                        if z_real <= 0:
                            continue

                        # 检验 A：边缘深度跳变检验 (防止点落在手指与背景交界处)
                        if z_std > MAX_PATCH_STD_M:
                            continue

                        # 检验 B：手部物理尺寸球检验 (防止点穿透到桌面或远处背景)
                        if abs(z_real - base_palm_z) > MAX_HAND_RADIUS_M:
                            continue

                        # 检验 C：前后遮挡冲突检验 (核心！识别“被手背或白色罐子挡住的脑补指尖”)
                        # MediaPipe 预估该点相对于手腕的深度偏移 (lm.z > 0 表示在手腕后方远离相机)
                        expected_z = base_palm_z + (lm.z * mp_z_scale)
                        # 如果某点(如弯曲的指尖)本该在后面(expected_z 较大)，但实际深度传感器却测到前面有东西挡着(z_real 明显偏小)
                        if (expected_z - z_real) > OCCLUSION_MARGIN_M:
                            continue

                        # 检验 D：2D 自遮挡碰撞检验 (如果指尖跟手掌根部/手背像素几乎重合，且指尖预期在后方，必是被手背挡住)
                        is_self_occluded = False
                        if i in [4, 8, 12, 16, 20]:  # 针对 5 个指尖重点排查手背遮挡
                            for palm_idx in [0, 1, 2, 5, 9, 13, 17]:
                                dist_2d = np.hypot(px - pts_2d[palm_idx][0], py - pts_2d[palm_idx][1])
                                # 若指尖与手掌节点在画面上距离小于 12 像素，且 MediaPipe 认为指尖在手掌后方 (lm.z 更大)
                                if dist_2d < 12 and (lm.z > hand_landmarks.landmark[palm_idx].z + 0.02):
                                    is_self_occluded = True
                                    break
                        if is_self_occluded:
                            continue

                        # 通过全部检验！确认为【真实可见、未被遮挡】的关键点，计算 3D 坐标
                        valid_mask[i] = True
                        points_3d[i] = rs.rs2_deproject_pixel_to_point(intr, [px, py], z_real)

                    # 4. 只绘制【两端都真实可见】的骨骼连线
                    for connection in mp_hands.HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        if valid_mask[start_idx] and valid_mask[end_idx]:
                            cv2.line(color_image, pts_2d[start_idx], pts_2d[end_idx], (0, 255, 0), 2)

                    # 5. 绘制关键点与 3D 坐标
                    for i in range(21):
                        px, py = pts_2d[i]
                        if valid_mask[i]:
                            # 真实可见点：画亮绿色实心圆
                            cv2.circle(color_image, (px, py), 5, (0, 255, 0), -1)
                            cv2.circle(depth_colormap, (px, py), 4, (255, 255, 255), -1)

                            # 若食指指尖 (8号点) 未被遮挡，显示其 3D 坐标
                            if i == 8 and points_3d[8] is not None:
                                X, Y, Z = points_3d[8]
                                coord_text = f"({X:.2f}, {Y:.2f}, {Z:.2f})m"
                                cv2.circle(color_image, (px, py), 7, (0, 255, 255), -1)
                                cv2.putText(color_image, coord_text, (px + 10, py),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                        else:
                            # 被遮挡的脑补点：若 HIDE_OCCLUDED_POINTS=False，画灰色小叉表示已被成功拦截
                            if not HIDE_OCCLUDED_POINTS and (0 <= px < WIDTH and 0 <= py < HEIGHT):
                                cv2.drawMarker(color_image, (px, py), (128, 128, 128),
                                               markerType=cv2.MARKER_TILTED_CROSS, markerSize=8, thickness=1)

            cv2.putText(color_image, f"Cam {idx + 1} | SN: {serial}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cam_view = np.vstack((color_image, depth_colormap))
            latest_views[idx] = cam_view

        if latest_views:
            combined_all = np.hstack(latest_views)
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