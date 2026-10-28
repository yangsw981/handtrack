import os
import cv2
import numpy as np
import pyrealsense2 as rs

# ================= 1. ChArUco 标定板参数配置 =================
SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH = 0.0344
MARKER_LENGTH = 0.02522

WIDTH, HEIGHT, FPS = 848, 480, 30
SAVE_FILE = "extrinsics_3cams.npz"
# =============================================================

# 初始化 ArUco 字典与 ChArUco 板
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
try:
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
    detector_params = cv2.aruco.DetectorParameters()
except AttributeError:
    board = cv2.aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
    detector_params = cv2.aruco.DetectorParameters_create()

# 全局创建 CharucoDetector 对象
charuco_detector = cv2.aruco.CharucoDetector(board)


def detect_charuco_pose(image, K, dist_coeffs):
    """在图像中检测 ChArUco 板，并求解标定板相对于相机的 4x4 变换矩阵 T_cam_from_board。
    返回: (success, T_cam_from_board, num_corners, vis_image)
    """
    vis_img = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. 检测 ChArUco 角点
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    if charuco_ids is None or len(charuco_ids) < 6:
        num_corners = 0 if charuco_ids is None else len(charuco_ids)
        return False, None, num_corners, vis_img

    num_corners = len(charuco_ids)

    # 强制转换为 OpenCV 5.0 绘图和 solvePnP 要求的形状与类型
    charuco_corners = charuco_corners.reshape(-1, 1, 2).astype(np.float32)
    charuco_ids = charuco_ids.reshape(-1, 1).astype(np.int32)

    # 可视化：绘制 ArUco 标记和 ChArUco 角点
    if marker_corners is not None and marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(vis_img, marker_corners, marker_ids)
    cv2.aruco.drawDetectedCornersCharuco(vis_img, charuco_corners, charuco_ids, (0, 255, 255))

    # 2. 【核心修改】用 matchImagePoints + solvePnP 替代已移除的 estimatePoseCharucoBoard
    try:
        # 获取标定板坐标系下的 3D 对象点，以及图像上对应的 2D 像素点
        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    except cv2.error:
        return False, None, num_corners, vis_img

    if obj_points is None or len(obj_points) < 4:
        return False, None, num_corners, vis_img

    # 使用 solvePnP 求解位姿
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist_coeffs)

    if not success:
        return False, None, num_corners, vis_img

    # 绘制坐标轴
    cv2.drawFrameAxes(vis_img, K, dist_coeffs, rvec, tvec, 0.10, 3)

    # 3. 组装 4x4 齐次变换矩阵
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()

    return True, T, num_corners, vis_img


# ================= 2. 启动所有 RealSense 相机 =================
ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    print("未检测到 RealSense 相机！")
    exit(1)

pipelines = []
serials = []
K_list = []
dist_list = []

for i, dev in enumerate(devices):
    serial = dev.get_info(rs.camera_info.serial_number)
    pipeline = rs.pipeline(ctx)
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    profile = pipeline.start(config)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()

    K = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ], dtype=np.float64)
    D = np.array(intr.coeffs, dtype=np.float64)

    pipelines.append(pipeline)
    serials.append(serial)
    K_list.append(K)
    dist_list.append(D)
    print(f"已启动 Cam {i+1} | SN: {serial}")

print("\n================== 标定操作说明 ==================")
print("【模式 A：一步到位标定】")
print("  将标定板放入视野，当三台相机顶部都显示 [READY] 时，按键盘 '1' 键直接保存！")
print("\n【模式 B：两步传递标定（推荐：解决侧相机看不清平铺桌面问题）】")
print("  第 1 步：将标定板【斜靠在罐子上(约40°)】，三台相机均显示 [READY] 时，按 '2' 键锁定三机相对关系；")
print("  第 2 步：将标定板【纯平铺在绿色桌面上】，只要顶视相机显示 [READY]，按 '3' 键即可将坐标系对齐到桌面并保存！")
print("  按 'q' 键退出程序。")
print("==================================================\n")

window_name = "Multi-D435i ChArUco Calibration"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

T_cam0_from_cam = {}
master_idx = 0

latest_vis = [np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8) for _ in range(len(pipelines))]
current_poses = [None] * len(pipelines)

try:
    while True:
        for idx, (pipe, K, D, serial) in enumerate(zip(pipelines, K_list, dist_list, serials)):
            frames = pipe.poll_for_frames()
            if not frames:
                continue
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            ok, T_cam_board, n_corners, vis_img = detect_charuco_pose(img, K, D)

            if ok:
                current_poses[idx] = T_cam_board
                status_text = f"Cam {idx+1} ({serial}) | [READY] Corners: {n_corners}"
                color = (0, 255, 0)
            else:
                current_poses[idx] = None
                status_text = f"Cam {idx+1} ({serial}) | [WAIT] Corners: {n_corners}/6"
                color = (0, 0, 255)

            cv2.putText(vis_img, status_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            latest_vis[idx] = vis_img

        combined = np.hstack(latest_vis)
        if WIDTH > 640:
            combined = cv2.resize(combined, (0, 0), fx=0.7, fy=0.7)
        cv2.imshow(window_name, combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('1'):
            if all(p is not None for p in current_poses):
                extrinsics = {}
                for idx, serial in enumerate(serials):
                    extrinsics[serial] = np.linalg.inv(current_poses[idx])
                np.savez(SAVE_FILE, **extrinsics)
                print(f"\n[成功] 一步标定完成！已将 {len(serials)} 台相机的外参矩阵保存至 {SAVE_FILE}")
            else:
                print("\n[失败] 请确保三台相机画面左上角同时显示绿色的 [READY] 再按 '1'！")

        elif key == ord('2'):
            if all(p is not None for p in current_poses):
                T_cam0_board = current_poses[0]
                T_cam0_from_cam.clear()
                for idx, serial in enumerate(serials):
                    T_cam0_from_cam[serial] = T_cam0_board @ np.linalg.inv(current_poses[idx])
                print("\n[第 1 步成功] 已锁定三台相机之间的相对空间关系！")
                print(">>> 现在请把标定板【纯平铺在绿色桌面上】，只要有任意一台顶视相机显示绿色 [READY]，按 '3' 键即可！")
            else:
                print("\n[第 1 步失败] 请调整斜放角度，确保三台相机同时显示绿色的 [READY] 再按 '2'！")

        elif key == ord('3'):
            if not T_cam0_from_cam:
                print("\n[提示] 请先完成第 1 步（斜放按 '2' 键）！")
                continue

            valid_top_idx = None
            for idx, p in enumerate(current_poses):
                if p is not None:
                    valid_top_idx = idx
                    break

            if valid_top_idx is not None:
                top_serial = serials[valid_top_idx]
                T_world_from_top = np.linalg.inv(current_poses[valid_top_idx])
                T_world_from_cam0 = T_world_from_top @ np.linalg.inv(T_cam0_from_cam[top_serial])

                extrinsics = {}
                for serial in serials:
                    extrinsics[serial] = T_world_from_cam0 @ T_cam0_from_cam[serial]

                np.savez(SAVE_FILE, **extrinsics)
                print(f"\n[全部成功] 已通过顶视相机 ({top_serial}) 将三台相机统一对齐至水平桌面坐标系！")
                print(f">>> 标定文件已保存至: {SAVE_FILE}")
            else:
                print("\n[第 2 步失败] 当前没有任何一台相机看清桌面上的标定板，请确保至少顶视相机显示绿色 [READY]！")

        elif key == ord('q'):
            break

finally:
    for pipe in pipelines:
        pipe.stop()
    cv2.destroyAllWindows()