import os
import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

# ================= 1. 加载外参标定文件 =================
CALIB_FILE = "extrinsics_3cams.npz"
if not os.path.exists(CALIB_FILE):
    print(f"未找到标定文件 {CALIB_FILE}，请先运行标定程序！")
    exit(1)

calib_data = np.load(CALIB_FILE)
print(f"成功加载标定文件: {CALIB_FILE}，包含相机序列号: {list(calib_data.files)}")

# ================= 2. 参数配置 =================
WIDTH, HEIGHT, FPS = 848, 480, 30

MAX_HAND_RADIUS_M = 0.22  # 手部最大物理跨度 (22cm)
OCCLUSION_MARGIN_M = 0.035  # 前后遮挡容差 (3.5cm)
MAX_PATCH_STD_M = 0.015  # 局部深度平整度阈值 (1.5cm)
# ===============================================

mp_hands = mp.solutions.hands

ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    print("未检测到 RealSense 设备！")
    exit(1)

pipelines = []
align_objects = []
hands_trackers = []
serials = []
intrinsics_list = []
T_world_from_cam_list = []
T_cam_from_world_list = []

# 为每台相机单独存储滤波器
spatial_filters = []
temporal_filters = []

for i, dev in enumerate(devices):
    serial = dev.get_info(rs.camera_info.serial_number)
    if serial not in calib_data:
        print(f"[警告] 相机 {serial} 未在标定文件中，跳过该相机！")
        continue

    pipeline = rs.pipeline(ctx)
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    try:
        profile = pipeline.start(config)

        # 初始化滤波器
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 3)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)
        temporal = rs.temporal_filter()

        spatial_filters.append(spatial)
        temporal_filters.append(temporal)

        pipelines.append(pipeline)
        serials.append(serial)
        align_objects.append(rs.align(rs.stream.color))

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics_list.append(color_profile.get_intrinsics())

        # 读取该相机的外参矩阵
        T_wc = calib_data[serial]
        T_world_from_cam_list.append(T_wc)
        T_cam_from_world_list.append(np.linalg.inv(T_wc))

        hands_trackers.append(
            mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.75,
                min_tracking_confidence=0.75
            )
        )
        print(f"已启动并绑定外参: Cam {i + 1} | SN: {serial}")
    except Exception as e:
        print(f"相机 {serial} 启动失败: {e}")

num_cams = len(pipelines)
if num_cams == 0:
    exit(1)


def get_clean_depth(depth_image_mm, px, py, half_win=2):
    """获取关键点周围 5x5 区域的鲁棒深度中位数，剔除离群点（单位：米）"""
    h, w = depth_image_mm.shape
    x1, x2 = max(0, px - half_win), min(w, px + half_win + 1)
    y1, y2 = max(0, py - half_win), min(h, py + half_win + 1)

    patch = depth_image_mm[y1:y2, x1:x2].astype(np.float32) * 0.001
    # 缩小深度有效范围，过滤背景
    valid = patch[(patch > 0.15) & (patch < 0.80)]
    if len(valid) < 4:
        return 0.0, 999.0

    # 计算中位数并剔除严重偏离中位数的点（比如手指边缘碰到了木块）
    med = np.median(valid)
    valid = valid[np.abs(valid - med) < 0.02]  # 剔除偏差大于2cm的点
    if len(valid) == 0:
        return med, 999.0

    return float(np.median(valid)), float(np.std(valid))


latest_color_imgs = [np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8) for _ in range(num_cams)]
latest_depth_Maps = [np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8) for _ in range(num_cams)]
cam_world_pts = [[None] * 21 for _ in range(num_cams)]

window_name = "Multi-Cam 3D Hand Fusion (World Coordinates)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("\n================== 运行说明 ==================")
print("1. 绿色圆点：当前相机自己看到的【无遮挡有效关键点】")
print("2. 黄色文字：各相机独立测算的【食指指尖(8号点) 世界坐标 (Xw, Yw, Zw)】")
print("3. 洋红色骨架：三台相机互补融合后的【全局 3D 骨架重投影】")
print("按 'q' 键退出程序。")
print("==============================================\n")

try:
    while True:
        # ---------------- 第一阶段：各相机独立采集、去遮挡并转换到世界坐标系 ----------------
        for idx in range(num_cams):
            frames = pipelines[idx].poll_for_frames()
            if not frames:
                continue

            # 1. 先对齐
            aligned_frames = align_objects[idx].process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # 2. 对对齐后的深度图进行滤波 (空间+时间)
            depth_frame = spatial_filters[idx].process(depth_frame)
            depth_frame = temporal_filters[idx].process(depth_frame)

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            intr = intrinsics_list[idx]
            T_wc = T_world_from_cam_list[idx]

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET
            )

            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands_trackers[idx].process(image_rgb)
            image_rgb.flags.writeable = True

            cam_world_pts[idx] = [None] * 21

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                pts_2d = []
                raw_depths = []
                patch_stds = []

                for lm in hand_landmarks.landmark:
                    px, py = int(lm.x * WIDTH), int(lm.y * HEIGHT)
                    pts_2d.append((px, py))
                    if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                        d_med, d_std = get_clean_depth(depth_image, px, py, half_win=2)
                        raw_depths.append(d_med)
                        patch_stds.append(d_std)
                    else:
                        raw_depths.append(0.0)
                        patch_stds.append(999.0)

                palm_indices = [0, 1, 5, 9, 13, 17]
                palm_valid_depths = [raw_depths[i] for i in palm_indices if raw_depths[i] > 0]

                if palm_valid_depths:
                    base_palm_z = float(np.median(palm_valid_depths))
                    p0, p9 = pts_2d[0], pts_2d[9]
                    pix_palm_len = np.hypot(p0[0] - p9[0], p0[1] - p9[1])
                    mp_z_scale = (0.085 / max(pix_palm_len, 20.0)) * WIDTH

                    for i, lm in enumerate(hand_landmarks.landmark):
                        px, py = pts_2d[i]
                        z_real = raw_depths[i]
                        z_std = patch_stds[i]

                        if z_real <= 0 or z_std > MAX_PATCH_STD_M:
                            continue
                        if abs(z_real - base_palm_z) > MAX_HAND_RADIUS_M:
                            continue

                        expected_z = base_palm_z + (lm.z * mp_z_scale)
                        if (expected_z - z_real) > OCCLUSION_MARGIN_M:
                            continue

                        is_self_occluded = False
                        if i in [4, 8, 12, 16, 20]:
                            for palm_idx in [0, 1, 2, 5, 9, 13, 17]:
                                dist_2d = np.hypot(px - pts_2d[palm_idx][0], py - pts_2d[palm_idx][1])
                                if dist_2d < 12 and (lm.z > hand_landmarks.landmark[palm_idx].z + 0.02):
                                    is_self_occluded = True
                                    break
                        if is_self_occluded:
                            continue

                        # 1. 像素反投影到当前相机局部坐标系
                        pt_cam = np.array(rs.rs2_deproject_pixel_to_point(intr, [px, py], z_real))

                        # 2. 转换到统一的全局世界坐标系
                        pt_world = (T_wc[:3, :3] @ pt_cam) + T_wc[:3, 3]
                        cam_world_pts[idx][i] = pt_world

                        # 绘制单机可见点
                        cv2.circle(color_image, (px, py), 4, (0, 255, 0), -1)
                        cv2.circle(depth_colormap, (px, py), 4, (255, 255, 255), -1)

                        if i == 8:
                            Xw, Yw, Zw = pt_world
                            coord_text = f"W:({Xw:.2f},{Yw:.2f},{Zw:.2f})m"
                            cv2.circle(color_image, (px, py), 7, (0, 255, 255), 2)
                            cv2.putText(color_image, coord_text, (px + 10, py),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            latest_color_imgs[idx] = color_image
            latest_depth_Maps[idx] = depth_colormap

        # ---------------- 第二阶段：三台相机 3D 坐标互补融合 ----------------
        fused_world_hand = [None] * 21
        valid_joint_count = 0

        for joint_idx in range(21):
            obs_list = [cam_world_pts[c_idx][joint_idx] for c_idx in range(num_cams)
                        if cam_world_pts[c_idx][joint_idx] is not None]

            if len(obs_list) == 0:
                continue

            if len(obs_list) == 1:
                fused_world_hand[joint_idx] = obs_list[0]
                valid_joint_count += 1
                continue

            # 离群点剔除校验
            TOLERANCE = 0.02
            best_point = None

            if len(obs_list) == 2:
                if np.linalg.norm(obs_list[0] - obs_list[1]) < TOLERANCE:
                    best_point = (obs_list[0] + obs_list[1]) / 2.0
                else:
                    continue
            elif len(obs_list) >= 3:
                dist_ab = np.linalg.norm(obs_list[0] - obs_list[1])
                dist_ac = np.linalg.norm(obs_list[0] - obs_list[2])
                dist_bc = np.linalg.norm(obs_list[1] - obs_list[2])

                if dist_bc < TOLERANCE:
                    best_point = (obs_list[1] + obs_list[2]) / 2.0
                elif dist_ab < TOLERANCE:
                    best_point = (obs_list[0] + obs_list[1]) / 2.0
                elif dist_ac < TOLERANCE:
                    best_point = (obs_list[0] + obs_list[2]) / 2.0
                else:
                    continue

            if best_point is not None:
                fused_world_hand[joint_idx] = best_point
                valid_joint_count += 1

        # ---------------- 第三阶段：将融合后的完整 3D 骨骼重投影回各相机画面展示 ----------------
        views = []
        for idx in range(num_cams):
            vis_color = latest_color_imgs[idx].copy()
            T_cw = T_cam_from_world_list[idx]
            intr = intrinsics_list[idx]

            reproj_2d = [None] * 21
            for joint_idx in range(21):
                pt_w = fused_world_hand[joint_idx]
                if pt_w is not None:
                    pt_c = (T_cw[:3, :3] @ pt_w) + T_cw[:3, 3]
                    if pt_c[2] > 0.1:
                        uv = rs.rs2_project_point_to_pixel(intr, pt_c.tolist())
                        u, v = int(uv[0]), int(uv[1])
                        if 0 <= u < WIDTH and 0 <= v < HEIGHT:
                            reproj_2d[joint_idx] = (u, v)

            for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
                if reproj_2d[start_idx] is not None and reproj_2d[end_idx] is not None:
                    cv2.line(vis_color, reproj_2d[start_idx], reproj_2d[end_idx], (255, 0, 255), 2)

            for u_v in reproj_2d:
                if u_v is not None:
                    cv2.circle(vis_color, u_v, 3, (255, 0, 255), -1)

            cv2.putText(vis_color, f"Cam {idx + 1} ({serials[idx]}) | Fused Joints: {valid_joint_count}/21",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            views.append(np.vstack((vis_color, latest_depth_Maps[idx])))

        if views:
            combined_all = np.hstack(views)
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
        try:
            pipe.stop()
        except:
            pass
    cv2.destroyAllWindows()