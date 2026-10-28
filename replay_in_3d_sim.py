import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import savgol_filter

# ================= 1. 加载录制好的三维轨迹数据 =================
TRAJ_FILE = "grasp_trajectory.npz"
data = np.load(TRAJ_FILE)
timestamps = data["timestamps"]        # shape: (N,)
raw_trajectories = data["trajectories"]  # shape: (N, 21, 3)

num_frames = len(raw_trajectories)
print(f"加载轨迹成功！共 {num_frames} 帧，正在执行【运动学防交叉与解剖学骨长重构】...")

# ================= 2. 核心：人手解剖学运动学约束与防交叉解算器 =================
# 标准成年人手各段骨骼的物理长度 (单位：米)
STD_BONE_LENGTHS = {
    # 大拇指链
    (0, 1): 0.038, (1, 2): 0.032, (2, 3): 0.026, (3, 4): 0.022,
    # 食指链
    (0, 5): 0.088, (5, 6): 0.038, (6, 7): 0.024, (7, 8): 0.018,
    # 中指链
    (0, 9): 0.086, (9, 10): 0.042, (10, 11): 0.026, (11, 12): 0.019,
    # 无名指链
    (0, 13): 0.082, (13, 14): 0.038, (14, 15): 0.024, (15, 16): 0.018,
    # 小指链
    (0, 17): 0.076, (17, 18): 0.030, (18, 19): 0.018, (19, 20): 0.016
}


def enforce_hand_kinematics(traj_N_21_3):
    """对原始 3D 轨迹逐帧施加：

    1. 刚性掌骨扇形平面重整（徹底消除掌骨拧麻花现象）；
    2. 四指横向防交叉排斥约束（保证食指、中指、无名指、小指严格按横向顺序排列，不穿透）；
    3. 正向运动学（FK）恒定骨长约束（消除僵尸点拉伸变形）。
    """
    fixed_traj = np.zeros_like(traj_N_21_3)

    for f in range(len(traj_N_21_3)):
        pts = traj_N_21_3[f].copy()
        wrist = pts[0].copy()
        fixed_pts = np.zeros((21, 3), dtype=np.float32)
        fixed_pts[0] = wrist

        # --- 步骤 A: 建立稳定的手掌局部正交基底 (Right, Forward, Normal) ---
        # 用整只手所有手指点的重心确定手掌主前方 (Forward)
        finger_center = np.mean(pts[5:21], axis=0)
        v_forward = finger_center - wrist
        norm_f = np.linalg.norm(v_forward)
        v_forward = v_forward / (norm_f if norm_f > 1e-5 else 1.0)

        # 用食指侧(1,2,5,6)与小指侧(13,14,17,18)的均值差确定手掌横向展开轴 (Spread Axis)
        radial_side = np.mean(pts[[1, 2, 5, 6]], axis=0)
        ulnar_side = np.mean(pts[[13, 14, 17, 18]], axis=0)
        v_spread = ulnar_side - radial_side
        # 正交化：去除 v_spread 在 v_forward 方向上的分量
        v_spread = v_spread - np.dot(v_spread, v_forward) * v_forward
        norm_s = np.linalg.norm(v_spread)
        if norm_s < 1e-4:
            v_spread = np.cross(v_forward, np.array([0.0, 0.0, 1.0]))
            norm_s = np.linalg.norm(v_spread)
        v_spread = v_spread / (norm_s if norm_s > 1e-5 else 1.0)

        # 手掌法向量 (掌心朝向)
        v_normal = np.cross(v_forward, v_spread)
        v_normal = v_normal / (np.linalg.norm(v_normal) + 1e-8)

        # --- 步骤 B: 重整掌指关节 (MCP: 5, 9, 13, 17) 的扇形分布，根治掌骨重叠 ---
        # 解剖学上，5(食)、9(中)、13(无名)、17(小) 在手掌横向轴上的标准偏移角 (弧度)
        mcp_angles = {5: -0.22, 9: -0.05, 13: 0.12, 17: 0.28}
        mcp_arch_drop = {5: 0.0, 9: 0.004, 13: 0.002, 17: -0.004}  # 掌横弓自然弧度

        for mcp_idx in [5, 9, 13, 17]:
            theta = mcp_angles[mcp_idx]
            bone_len = STD_BONE_LENGTHS[(0, mcp_idx)]
            # 结合原始观测方向与标准掌骨扇面进行混合约束 (70% 解剖学刚性扇面 + 30% 实测微调)
            ideal_dir = np.cos(theta) * v_forward + np.sin(theta) * v_spread + mcp_arch_drop[mcp_idx] * v_normal
            ideal_dir /= np.linalg.norm(ideal_dir)

            raw_dir = pts[mcp_idx] - wrist
            raw_norm = np.linalg.norm(raw_dir)
            if raw_norm > 1e-4:
                raw_dir /= raw_norm
                blended_dir = 0.75 * ideal_dir + 0.25 * raw_dir
                blended_dir /= np.linalg.norm(blended_dir)
            else:
                blended_dir = ideal_dir

            fixed_pts[mcp_idx] = wrist + blended_dir * bone_len

        # --- 步骤 C: 大拇指链 (0->1->2->3->4) 恒定骨长重构 ---
        thumb_chain = [0, 1, 2, 3, 4]
        for k in range(len(thumb_chain) - 1):
            p_idx, c_idx = thumb_chain[k], thumb_chain[k + 1]
            vec = pts[c_idx] - pts[p_idx]
            # 防止大拇指贴进食指内部：对大拇指根部施加向外侧 (-v_spread) 的最小张角保护
            if p_idx == 0:
                vec = 0.6 * vec + 0.4 * (0.5 * v_forward - 0.7 * v_spread + 0.3 * v_normal) * 0.04
            n = np.linalg.norm(vec)
            direction = vec / n if n > 1e-4 else (v_forward - v_spread) / np.sqrt(2)
            fixed_pts[c_idx] = fixed_pts[p_idx] + direction * STD_BONE_LENGTHS[(p_idx, c_idx)]

        # --- 步骤 D: 四指指骨链 (MCP -> PIP -> DIP -> TIP) 骨长重构 + 防交叉排斥 ---
        finger_roots = [5, 9, 13, 17]
        for level in [1, 2, 3]:  # level 1: PIP(6,10,14,18), level 2: DIP(7,11,15,19), level 3: TIP(8,12,16,20)
            curr_joints = [r + level for r in finger_roots]
            prev_joints = [r + level - 1 for r in finger_roots]

            # 1. 先按原始骨骼向量方向 + 标准骨长推算各指节位置
            for p_idx, c_idx in zip(prev_joints, curr_joints):
                vec = pts[c_idx] - pts[p_idx]
                n = np.linalg.norm(vec)
                direction = vec / n if n > 1e-4 else v_forward
                fixed_pts[c_idx] = fixed_pts[p_idx] + direction * STD_BONE_LENGTHS[(p_idx, c_idx)]

            # 2. 【核心防交叉机制】检查四根手指在横向展开轴 (v_spread) 上的投影顺序！
            # 物理铁律：食指、中指、无名指、小指在 v_spread 上的坐标必须单调递增，且最小间距 >= 1.3cm (手指直径)
            min_finger_gap = 0.013  # 13mm 最小手指物理间距
            proj_vals = [np.dot(fixed_pts[c_idx] - wrist, v_spread) for c_idx in curr_joints]

            # 以四指横向中心为锚点，若发现相邻手指间距小于 min_finger_gap 甚至倒序交叉，立即解耦推开
            for iter_repulse in range(3):
                for idx_f in range(3):
                    gap = proj_vals[idx_f + 1] - proj_vals[idx_f]
                    if gap < min_finger_gap:
                        push = 0.5 * (min_finger_gap - gap)
                        proj_vals[idx_f] -= push
                        proj_vals[idx_f + 1] += push

            # 将修正后的横向坐标写回，并重新归一化保持严格骨长
            for idx_f, (p_idx, c_idx) in enumerate(zip(prev_joints, curr_joints)):
                old_proj = np.dot(fixed_pts[c_idx] - wrist, v_spread)
                delta_spread = (proj_vals[idx_f] - old_proj) * v_spread
                corrected_pos = fixed_pts[c_idx] + delta_spread
                # 重新锁定到父节点的标准骨长
                re_vec = corrected_pos - fixed_pts[p_idx]
                re_norm = np.linalg.norm(re_vec)
                if re_norm > 1e-5:
                    fixed_pts[c_idx] = fixed_pts[p_idx] + (re_vec / re_norm) * STD_BONE_LENGTHS[(p_idx, c_idx)]

        fixed_traj[f] = fixed_pts

    return fixed_traj


# 执行运动学防交叉重构 + 时间轴零相位平滑
trajectories = enforce_hand_kinematics(raw_trajectories)
if num_frames > 7:
    trajectories = savgol_filter(trajectories, window_length=7, polyorder=2, axis=0)

# ================= 3. 定义手部拓扑结构与五指配色 =================
FINGER_CHAINS = [
    ([0, 1, 2, 3, 4],    "#FF3B30", "Thumb"),   # 大拇指 - 红
    ([0, 5, 6, 7, 8],    "#FF9500", "Index"),   # 食指   - 橙黄
    ([0, 9, 10, 11, 12], "#34C759", "Middle"),  # 中指   - 绿
    ([0, 13, 14, 15, 16], "#00C7BE", "Ring"),   # 无名指 - 青
    ([0, 17, 18, 19, 20], "#007AFF", "Pinky"),  # 小指   - 蓝
    ([5, 9, 13, 17],     "#8E8E93", "Palm")     # 掌横弓 - 灰
]

# ================= 4. 搭建三维仿真空间窗口 =================
fig = plt.figure(figsize=(10, 8), facecolor="#1E1E1E")
ax = fig.add_subplot(111, projection="3d", facecolor="#1E1E1E")

all_pts = trajectories.reshape(-1, 3)
center = np.median(all_pts, axis=0)
span = 0.16  # 聚焦 32cm 工作空间，看得更清楚

ax.set_xlim(center[0] - span, center[0] + span)
ax.set_ylim(center[1] - span, center[1] + span)
ax.set_zlim(center[2] - span, center[2] + span)
ax.set_box_aspect([1, 1, 1])

ax.set_xlabel("X (m)", color="white", labelpad=8)
ax.set_ylabel("Y (m)", color="white", labelpad=8)
ax.set_zlabel("Z - Height (m)", color="white", labelpad=8)
ax.tick_params(colors="lightgray")

table_z = np.min(all_pts[:, 2]) - 0.02
gx, gy = np.meshgrid(
    np.linspace(center[0] - span, center[0] + span, 10),
    np.linspace(center[1] - span, center[1] + span, 10)
)
gz = np.ones_like(gx) * table_z
ax.plot_wireframe(gx, gy, gz, color="#2E7D32", alpha=0.35, linewidth=1)

lines = [ax.plot([], [], [], color=color, linewidth=3.2, label=name)[0]
         for _, color, name in FINGER_CHAINS]
joints_scatter = ax.scatter([], [], [], color="white", s=38, depthshade=False)
index_trail, = ax.plot([], [], [], color="#FFD60A", linewidth=1.2, linestyle="--", alpha=0.7, label="Index Tip Trail")

title_text = ax.set_title("", color="white", fontsize=13, pad=15)
ax.legend(loc="upper right", facecolor="#2C2C2E", edgecolor="none", labelcolor="white")
ax.view_init(elev=25, azim=-45)


def update(frame_idx):
    pts = trajectories[frame_idx]
    t_sec = timestamps[frame_idx]

    for line, (indices, _, _) in zip(lines, FINGER_CHAINS):
        chain_pts = pts[indices]
        line.set_data(chain_pts[:, 0], chain_pts[:, 1])
        line.set_3d_properties(chain_pts[:, 2])

    joints_scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

    trail_pts = trajectories[:frame_idx + 1, 8, :]
    index_trail.set_data(trail_pts[:, 0], trail_pts[:, 1])
    index_trail.set_3d_properties(trail_pts[:, 2])

    title_text.set_text(f"Kinematic-Constrained 3D Hand | Frame: {frame_idx+1}/{num_frames} | Time: {t_sec:.2f}s")
    return lines + [joints_scatter, index_trail, title_text]


ani = FuncAnimation(fig, update, frames=num_frames, interval=33, blit=False, repeat=True)
plt.tight_layout()
plt.show()