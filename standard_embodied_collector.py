# -*- coding: utf-8 -*-
"""
多视角人手轨迹采集 (v4.5)
=====================================================================
v3 修的是什么
---------------------------------------------------------------------
v2 用"2D 关键点 + 实测深度"反投影，静止张开手时很准，但只要开始抓握，
品红骨架就会整体/局部偏离绿点。原因不是噪声，而是"深度测到的不是手"：

  1) 物体遮挡：手指绕到物体背面或贴着物体时，落在该像素上的深度是
     【物体表面】的深度。物体在手前面时，这一路观测被沿相机视线拉近
     几厘米（一个直径 8cm 的瓶子最多拉近 8cm），而且它自己看起来完全
     "合理"（深度有效、点数充足、时间上稳定）。
  2) 视角数少：3 台相机里只要有 2 台被物体挡住，就变成"多数派错误"，
     v2 的 IRLS（tol=1.5cm）无法剔除——它只能剔少数派。
  3) 先验自锁：v2 的深度先验直接用上一帧【融合 3D】投到本相机的深度，
     一旦某关节偏了，先验就跟着偏，±6cm 窗口会一直"合法"地接受错误
     表面，偏移被锁死，整个抓握过程都不回来。
  4) 2D 本身也会错：手指被物体挡住时 MediaPipe 会把关键点外推到物体
     轮廓上，各视角错得还不一样，三路一平均，融合点落在三条视线的
     中间——于是【每个视角】看到的骨架都离开自己的绿点。

v3 的做法（三条独立的证据互相校验）
---------------------------------------------------------------------
  A. 组织掩膜：用肤色（YCrCb+HSV）判断"这个像素到底是不是手"。
     像素不是手 -> 该关节在该视角的深度点标记为低置信；窗口里同时有
     手指和物体时，优先取肤色占比高的那一簇（不再无脑取最近簇）。
  B. 跨视角几何校验：同一关节的 3 条视线做三角化得到一个"几何点"，
     每个深度点都要和它比。被物体拉近的深度点必然对不上（它是物体
     表面，不是手），直接判为外点——这一步是"多数派错误"的克星，
     因为它完全不依赖深度，也不依赖票数。
  C. 骨长模型：在线学出 20 段骨长（滑动中位数）。某个关节只剩 1 路
     观测时，用"视线 + 骨长"解出它沿视线的唯一位置，而不是把物体
     表面当手指。全部观测都被挡死时，保持上一帧位置并标成"推断"。
  D. 先验只来自【本视角自己】上一次的实测深度，且 5 帧未更新就失效，
     不再用融合 3D 做先验，彻底断掉自锁。

每个关节最终是"候选点里得分最高的那个"，候选包括：三角化点、各视角
深度点、深度点均值、单视线+骨长解、上一帧位置；打分同时看 2D 残差、
深度残差、骨长残差。输出时给每个关节状态：0 缺失 / 1 实测 / 2 推断，
写进 joints_valid_mask 与 joints_inferred_mask，下游可自行取舍。

数据集规范输出结构保持不变，只追加质量字段。
=====================================================================
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

import warnings

warnings.filterwarnings("ignore")

import json
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

# ================= 1. 规范基础配置 (对应 SJ/T 具身智能数据采集规范) =================
DATASET_ROOT = "Embodied_Hand_Grasp_Dataset"
CALIB_FILE = "extrinsics_3cams.npz"

DATASET_DECLARATION = {
    "dataset_name": "MultiView_DexHand_Grasp",
    "author_name": "Researcher",
    "author_organization": "Robotics_Lab",
    "author_email": "researcher@university.edu.cn",
    "dataset_license": "CC BY 4.0",
    "dataset_version": "1.3.5"
}

ENVIRONMENT_INFO = {
    "scene_id": "Lab_Tabletop_Manipulation_01",
    "scene_info": "室内铝型材龙门架操作台，绿色桌面，三相机多视角抓握实验",
    "scene_type": "真实环境",
    "scene_configuration": "900x600x800mm Aluminum Frame with 3x Intel RealSense D435i",
    "light": "室内LED日光灯",
    "temperature": "24.0",
    "humidity": "45.0",
    "noise_level": 42.0,
    "objects": [
        {
            "object_id": "cylinder_jar_01",
            "object_class": "圆柱形罐体",
            "position": [0.0, 0.0, 0.0],
            "size_m": [0.08, 0.08, 0.11]
        }
    ],
    "surface_material": "绿色PVC防静电桌垫",
    "surface_slope": 0.0
}

TASK_DESCRIPTION = {
    "task_id": "TASK_GRASP_CYLINDER_001",
    "task_name": "Cylinder_Power_Grasp",
    "task_description": "人类演示者伸出右手，接近桌面中央的白色圆柱体并完成五指包络抓握与提起动作",
    "task_instruction": "1.手部移至圆柱体上方; 2.五指张开下降; 3.指尖贴合圆柱外壁闭合抓紧; 4.向上提起圆柱体",
    "skill_sequence": "Reach -> Pre-grasp -> Power_Grasp -> Lift",
    "operation_mode": "动捕方式(光学多视角三维关键点追踪)"
}

WIDTH, HEIGHT = 848, 480
STREAM_FPS = 15              # 相机流帧率：RealSense 只支持 6/15/30/60，15 是最接近的低档
FPS = 13                     # 采集节拍：整循环实测 ~72ms/帧(≈13.9fps)，按 13fps 走留余量
                              # 采集节拍 < 相机流帧率 => 每次 poll 都是新帧，不会出现 stale
SYNC_TOLERANCE_MS = 25.0
MAX_LOSS_RATE = 0.001
SAVE_RAW_IMAGES = True

# ---------------- 手部检测 ----------------
MP_MODEL_COMPLEXITY = 0      # 0=lite；CPU 富余可改 1

# ---------------- 深度采样 ----------------
MIN_DEPTH_M = 0.15           # 深度有效范围(米)
MAX_DEPTH_M = 1.00
HAND_CORE_WIN_NEAR_M = 0.20  # 允许比"掌心深度"近多少(手指向相机弯曲)
HAND_CORE_WIN_FAR_M = 0.15   # 允许比"掌心深度"远多少
DEPTH_HALF_WIN = 3           # 采样窗口 7x7
DEPTH_MIN_PIX = 3
DEPTH_CLUSTER_GAP_M = 0.012  # 1.2cm 以上断开即视为另一个表面
DEPTH_PRIOR_WIN_M = 0.045    # 有时间先验时的接受窗口
DEPTH_PRIOR_MAX_AGE = 5      # 先验超过这么多帧未更新就作废

# ---------------- 组织(肤色)掩膜 ----------------
SKIN_CR = (133, 180)
SKIN_CB = (77, 135)
SKIN_H_MAX = 25
SKIN_S_MIN = 30
SKIN_V_MIN = 50
TISSUE_MIN_FRAC = 0.20       # 低于此值视为"这个像素不是手"(被物体挡住)

# ---------------- 跨视角解算 ----------------
PIX_SIGMA = 5.0              # 2D 残差的高斯尺度(px)
DEP_SIGMA = 0.015            # 深度残差的高斯尺度(m)
BONE_SIGMA_REL = 0.18        # 骨长相对残差尺度
BONE_W = 0.35                # 骨长项权重
MIN_SCORE = 1.25             # 置信分下限，低于此值该关节转"推断"
MAX_STEP_M = 0.12            # 单帧最大跳变(防止离谱候选)
BONE_HIST_LEN = 15           # 骨长滑动中位数窗口
# ---------------- P1: 整手刚性联合解算 ----------------
BONE_EXTRA = [(5, 9), (9, 13), (13, 17)]   # 掌骨横撑，让手掌整体刚性
GN_ITERS = 5                # 高斯-牛顿迭代次数
SIGMA_PX = 3.0              # 2D 重投影标准差(px)
HUBER_PX = 12.0             # 2D 残差超过这么多 px 就降权
HUBER_DEP = 0.035           # 深度残差超过 3.5cm 就降权
SIG_BONE_REL = 0.06         # 骨长标准差 = 6% 骨长
HUBER_BONE_REL = 0.25       # 骨长残差超过 25% 就降权
W_PREV = 0.2 / 0.05 ** 2    # 上一帧先验权重(σ=5cm，再乘 0.2)

# ---------------- 时间滤波 ----------------
EMA_ALPHA = 0.65             # 静止/慢速时的平滑系数
EMA_SPEED_FULL_M = 0.020     # 单帧位移达到这个值就不再平滑(消除快动作滞后)
JUMP_TOL_M = 0.030           # 超过此跳变不滤波，直接采用

MIN_VALID_JOINTS = 12        # 少于此有效关节数的一帧不落盘

HAND_JOINT_NAMES = [
    "Wrist",
    "Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_Tip",
    "Index_MCP", "Index_PIP", "Index_DIP", "Index_Tip",
    "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_Tip",
    "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_Tip",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_Tip"
]

# MediaPipe 21 点的运动学树：PARENT[j] 是 j 的父关节
PARENT = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]


# ================= 2. 纯几何工具 =================
def rigid_inverse(T):
    """刚体变换求逆：R^T, -R^T t（比 np.linalg.inv 数值更稳，且保证正交）"""
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def make_cam(K, T_wc):
    """打包一台相机的几何信息"""
    T_cw = rigid_inverse(T_wc)
    return {
        "K": K,
        "T_wc": T_wc,               # 相机 -> 世界
        "T_cw": T_cw,               # 世界 -> 相机
        "P": K @ T_cw[:3, :],       # 3x4 投影矩阵
        "o": T_wc[:3, 3].copy(),    # 光心(世界系)
        "fx": float(K[0, 0]), "fy": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
    }


def project(cam, X):
    """世界点 -> 像素(浮点)；在相机背后返回 None"""
    p = cam["P"] @ np.append(X, 1.0)
    if p[2] <= 1e-6:
        return None
    return np.array([p[0] / p[2], p[1] / p[2]])


def backproject(cam, u, v, z):
    """像素 + 相机系 z 深度(米) -> 世界点。与 project 严格互逆。"""
    x = (u - cam["cx"]) / cam["fx"] * z
    y = (v - cam["cy"]) / cam["fy"] * z
    pc = np.array([x, y, z], dtype=np.float64)
    return cam["T_wc"][:3, :3] @ pc + cam["T_wc"][:3, 3]


def camera_ray(cam, u, v):
    """像素 -> 世界系射线 (原点, 单位方向)"""
    d = np.array([(u - cam["cx"]) / cam["fx"],
                  (v - cam["cy"]) / cam["fy"], 1.0])
    d = cam["T_cw"][:3, :3].T @ d
    return cam["o"].copy(), d / np.linalg.norm(d)


def triangulate(cams, uvs):
    """多视角 DLT 三角化，返回 (X, 各视角重投影误差px)"""
    A = []
    for cam, (u, v) in zip(cams, uvs):
        Pm = cam["P"]
        A.append(u * Pm[2, :] - Pm[0, :])
        A.append(v * Pm[2, :] - Pm[1, :])
    A = np.asarray(A, dtype=np.float64)
    if A.shape[0] < 4:
        return None, None
    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None, None
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return None, None
    X = X[:3] / X[3]
    errs = []
    for cam, (u, v) in zip(cams, uvs):
        p = project(cam, X)
        if p is None:
            return None, None
        errs.append(float(np.hypot(p[0] - u, p[1] - v)))
    return X, np.asarray(errs)


class Obs(object):
    """一路"某相机看到某关节"的观测"""
    __slots__ = ("cam", "uv", "p3d", "sigma", "tissue")

    def __init__(self, cam, uv=None, p3d=None, sigma=0.005, tissue=1.0):
        self.cam = cam        # 相机几何
        self.uv = uv          # 2D 像素 (u,v) 或 None
        self.p3d = p3d        # 深度反投影得到的世界点 或 None
        self.sigma = sigma    # 深度标准差(m)
        self.tissue = tissue  # 该像素邻域里"是手"的比例 0~1


# ================= 2b. P1: 整手刚性联合解算 =================
def _collect_obs(obs_by_cam):
    """把观测整理成便于向量化的列表，并统计每个关节有几路观测"""
    uv_obs, dep_obs = [], []
    n_obs = np.zeros(21, dtype=np.int32)
    for c, per in enumerate(obs_by_cam):
        for j, o in enumerate(per):
            if o is None:
                continue
            if o.uv is not None:
                uv_obs.append((c, j, float(o.uv[0]), float(o.uv[1]),
                               0.25 + 0.75 * o.tissue))
                n_obs[j] += 1
            if o.p3d is not None and np.isfinite(o.p3d).all():
                dep_obs.append((j, o.p3d, max(o.sigma, 0.002),
                                0.15 + 0.85 * o.tissue))
                n_obs[j] += 1
    return uv_obs, dep_obs, n_obs


def _init_pose(uv_obs, dep_obs, prev, cams):
    """初值：深度点均值 > 上一帧 > 沿本视角视线放到手中心深度 > 手中心"""
    X = np.full((21, 3), np.nan)
    for (j, p, s, w) in dep_obs:
        X[j] = p.copy() if not np.isfinite(X[j]).all() else 0.5 * (X[j] + p)
    if prev is not None:
        m = (~np.isfinite(X).all(1)) & np.isfinite(prev).all(1)
        X[m] = prev[m]

    fin = np.isfinite(X).all(1)
    center = X[fin].mean(0) if fin.any() else np.zeros(3)
    for (c, j, u, v, w) in uv_obs:
        if np.isfinite(X[j]).all():
            continue
        o, d = camera_ray(cams[c], u, v)
        X[j] = o + d * float(np.linalg.norm(center - o))
    X[~np.isfinite(X).all(1)] = center
    return X


def _huber(e, delta):
    return np.where(e <= delta, 1.0, delta / np.maximum(e, 1e-9))


def solve_pose(obs_by_cam, cams, prev=None, boneL=None, boneX=None,
               iters=None):
    """
    P1：21 个关节一起做加权最小二乘（整手刚性）。

    代价 = Σ_视角 Σ_关节 Huber(2D 重投影残差)
          + Σ_视角 Σ_关节 Huber(深度残差)
          + Σ_骨长 (长度 - 模型骨长)²      <-- 让手保持刚性
          + 上一帧弱先验

    好处：不再"逐关节各自挑一个视角"，所以骨架不会被撕开；
          某关节只剩 1 路观测时，靠骨长和邻居把它摆到正确位置。
    返回 pts(21x3), scores(每个关节的观测路数), state(1实测/2推断/0缺失)
    """
    if iters is None:
        iters = GN_ITERS

    uv_obs, dep_obs, n_obs = _collect_obs(obs_by_cam)
    state = np.where(n_obs > 0, 1, 2).astype(np.int32)
    if not uv_obs and not dep_obs:
        return np.full((21, 3), np.nan), n_obs.astype(np.float64), \
               np.zeros(21, dtype=np.int32)

    X = _init_pose(uv_obs, dep_obs, prev, cams)

    # ---- 组装骨长约束 ----
    bones = []
    if boneL:
        for j in range(1, 21):
            if j in boneL and boneL[j] > 1e-4:
                bones.append((PARENT[j], j, float(boneL[j])))
    if boneX:
        for (a, b), L in boneX.items():
            if L > 1e-4:
                bones.append((int(a), int(b), float(L)))

    # ---- 观测按视角分组，避免每帧重复建表 ----
    by_cam = {}
    for item in uv_obs:
        by_cam.setdefault(item[0], []).append(item)

    ar3 = np.arange(3)
    I3 = np.eye(3)

    for _ in range(iters):
        H = np.zeros((63, 63))
        g = np.zeros(63)

        # ---------- 2D 重投影 ----------
        for c, items in by_cam.items():
            P = cams[c]["P"]
            idx = np.array([it[1] for it in items], dtype=np.int64)
            uv = np.array([[it[2], it[3]] for it in items])
            w0 = np.array([it[4] for it in items])

            Xs = X[idx]
            hom = np.hstack([Xs, np.ones((len(idx), 1))]) @ P.T     # k,3
            z = hom[:, 2]
            good = z > 0.05
            if not np.any(good):
                continue
            zz = np.where(good, z, 1.0)
            uvp = hom[:, :2] / zz[:, None]
            r = uv - uvp                                            # k,2
            e = np.linalg.norm(r, axis=1)
            w = w0 * _huber(e, HUBER_PX) / SIGMA_PX ** 2 * good

            J = (P[None, :2, :3] - uvp[:, :, None] * P[None, 2:3, :3]) / zz[:, None, None]
            Hb = np.einsum("k,kia,kib->kab", w, J, J)               # k,3,3
            gb = np.einsum("k,kia,ki->ka", w, J, r)                 # k,3

            i3 = idx[:, None] * 3 + ar3[None, :]
            for aa in range(3):
                for bb in range(3):
                    np.add.at(H, (i3[:, aa], i3[:, bb]), Hb[:, aa, bb])
            np.add.at(g, i3, gb)

        # ---------- 深度 ----------
        if dep_obs:
            jj = np.array([d[0] for d in dep_obs], dtype=np.int64)
            pp = np.array([d[1] for d in dep_obs])
            ss = np.array([d[2] for d in dep_obs])
            w0 = np.array([d[3] for d in dep_obs])
            r = pp - X[jj]                                          # k,3
            e = np.linalg.norm(r, axis=1)
            w = w0 * _huber(e, HUBER_DEP) / np.maximum(ss, 0.002) ** 2
            for k in range(3):
                np.add.at(H, (jj * 3 + k, jj * 3 + k), w)
                np.add.at(g, jj * 3 + k, w * r[:, k])

        # ---------- 骨长 ----------
        for (a, b, L) in bones:
            d = X[b] - X[a]
            l = float(np.linalg.norm(d))
            if l < 1e-6:
                continue
            u = d / l
            err = l - L
            ww = float(_huber(np.array([abs(err)]), HUBER_BONE_REL * L)[0]) \
                / (SIG_BONE_REL * L) ** 2
            ia = a * 3 + ar3
            ib = b * 3 + ar3
            uu = ww * np.outer(u, u)
            H[np.ix_(ia, ia)] += uu
            H[np.ix_(ib, ib)] += uu
            H[np.ix_(ia, ib)] -= uu
            H[np.ix_(ib, ia)] -= uu
            g[ia] += ww * u * err
            g[ib] -= ww * u * err

        # ---------- 上一帧弱先验 ----------
        if prev is not None:
            pm = np.isfinite(prev).all(1)
            for j in np.nonzero(pm)[0]:
                s = j * 3 + ar3
                H[np.ix_(s, s)] += W_PREV * I3
                g[s] += W_PREV * (prev[j] - X[j])

        try:
            dx = np.linalg.solve(H + 1e-6 * np.eye(63), g)
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(dx).all():
            break
        X = X + dx.reshape(21, 3)
        if float(np.linalg.norm(dx)) < 1e-7:
            break

    return X, n_obs.astype(np.float64), state


class BoneModel(object):
    """在线学习骨长（含 3 段掌骨横撑），只喂"两端都有 ≥2 路观测"的帧"""

    def __init__(self):
        self.hist = {}
        self.L = {}
        self.histx = {}
        self.Lx = {}

    def reset(self):
        self.hist.clear(); self.L.clear()
        self.histx.clear(); self.Lx.clear()

    def update(self, pts, state, scores):
        edges = [(PARENT[j], j, True) for j in range(1, 21)] + \
                [(a, b, False) for (a, b) in BONE_EXTRA]
        for (a, b, is_tree) in edges:
            if state[a] != 1 or state[b] != 1:
                continue
            if scores[a] < 2 or scores[b] < 2:
                continue
            if not (np.isfinite(pts[a]).all() and np.isfinite(pts[b]).all()):
                continue
            Lv = float(np.linalg.norm(pts[b] - pts[a]))
            if not (0.005 < Lv < 0.15):
                continue
            h = (self.hist if is_tree else self.histx).setdefault(b if is_tree else (a, b), [])
            if h:
                med = float(np.median(h))
                if abs(Lv - med) > 0.35 * med:      # 单帧离谱值不进窗口
                    continue
            h.append(Lv)
            if len(h) > BONE_HIST_LEN:
                h.pop(0)
            if len(h) < 3:                          # 样本太少先不用
                continue
            if is_tree:
                self.L[b] = float(np.median(h))
            else:
                self.Lx[(a, b)] = float(np.median(h))


def mat_to_quat(R):
    """旋转矩阵 -> 单位四元数 [qx, qy, qz, qw]，四分支，无退化"""
    m = np.asarray(R, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


def compute_hand_global_pose(pts_21x3):
    """由 21 关节给出腕部位置 + 姿态四元数；关键点无效时返回 None"""
    need = [0, 5, 9, 17]
    if not np.isfinite(pts_21x3[need]).all():
        return None

    p_wrist = pts_21x3[0]
    v_x = pts_21x3[5] - pts_21x3[17]
    v_x /= (np.linalg.norm(v_x) + 1e-8)
    v_forward = pts_21x3[9] - pts_21x3[0]
    v_forward /= (np.linalg.norm(v_forward) + 1e-8)
    v_z = np.cross(v_x, v_forward)
    v_z /= (np.linalg.norm(v_z) + 1e-8)
    v_y = np.cross(v_z, v_x)

    R = np.column_stack((v_x, v_y, v_z))
    U, _, Vt = np.linalg.svd(R)          # 数值正交化
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1.0
        R = U @ Vt

    q = mat_to_quat(R)
    return [float(p_wrist[0]), float(p_wrist[1]), float(p_wrist[2]),
            float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def to_sim_frame(pts):
    """世界系 -> 仿真系(与仿真器一致：y、z 取反)。注意这是绕 X 轴 180°，det=+1"""
    out = np.array(pts, dtype=np.float64, copy=True)
    out[:, 1] = -out[:, 1]
    out[:, 2] = -out[:, 2]
    return out


def nan_to_list(a):
    a = np.asarray(a, dtype=np.float64).ravel()
    return [0.0 if not np.isfinite(x) else float(x) for x in a]


def opt_float(x, ndigits=6):
    return None if (x is None or not np.isfinite(x)) else round(float(x), ndigits)


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ================= 3. 深度采样 + 组织(肤色)掩膜 =================
def tissue_mask(bgr, bbox, pad=18):
    """
    只在手部 bbox 周围算肤色掩膜，返回整幅 bool 图（ROI 外为 False）。
    手是肤色、瓶子是白的、桌面是绿的，YCrCb 的 Cr/Cb 很容易分开。
    """
    h, w = bgr.shape[:2]
    full = np.zeros((h, w), dtype=bool)
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad); y1 = min(h, y1 + pad)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return full

    roi = bgr[y0:y1, x0:x1]
    ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m = (ycc[:, :, 1] >= SKIN_CR[0]) & (ycc[:, :, 1] <= SKIN_CR[1]) \
        & (ycc[:, :, 2] >= SKIN_CB[0]) & (ycc[:, :, 2] <= SKIN_CB[1]) \
        & (hsv[:, :, 0] <= SKIN_H_MAX) & (hsv[:, :, 1] >= SKIN_S_MIN) \
        & (hsv[:, :, 2] >= SKIN_V_MIN)
    m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
                         np.ones((3, 3), np.uint8)) > 0
    full[y0:y1, x0:x1] = m
    return full


def estimate_hand_depth(depth_mm, bbox):
    """bbox 内缩 25% 的中值深度 = 掌心深度(米)。失败返回 None。"""
    h, w = depth_mm.shape
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    xa, xb = int(x0 + 0.25 * bw), int(x1 - 0.25 * bw)
    ya, yb = int(y0 + 0.25 * bh), int(y1 - 0.25 * bh)
    if xb - xa < 6 or yb - ya < 6:
        return None
    roi = depth_mm[ya:yb, xa:xb].astype(np.float32) * 0.001
    v = roi[(roi > MIN_DEPTH_M) & (roi < MAX_DEPTH_M)]
    if v.size < 50:
        return None
    return float(np.median(v))


def sample_joint_depth(depth_mm, px, py, tissue, prior, hand_zc,
                       half_win=DEPTH_HALF_WIN):
    """
    在 (px,py) 邻域取一个"属于手"的深度值。
    返回 (z, sigma, tissue_frac, npix) 或 None。

    关键点：
      * 先在掌心深度 ± 窗口内筛像素，排除桌面/背景；
      * 按 1.2cm 断点聚类 —— 手指和物体是两个不同深度的表面，会分成两簇；
      * 打分 = 肤色占比(0.55) + 点数(0.25) + 时间先验(0.20)，不再"取最近簇"；
      * 整簇肤色占比低 = 这里根本没有手，上层据此把它当成"不可信"。
    """
    h, w = depth_mm.shape
    x0 = max(0, px - half_win); x1 = min(w, px + half_win + 1)
    y0 = max(0, py - half_win); y1 = min(h, py + half_win + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None

    patch = depth_mm[y0:y1, x0:x1].astype(np.float32) * 0.001
    tm = tissue[y0:y1, x0:x1]

    valid = (patch > MIN_DEPTH_M) & (patch < MAX_DEPTH_M)
    if hand_zc is not None:
        valid &= (patch > hand_zc - HAND_CORE_WIN_NEAR_M) & \
                 (patch < hand_zc + HAND_CORE_WIN_FAR_M)
    if valid.sum() < DEPTH_MIN_PIX:
        return None

    zs = patch[valid]
    ts = tm[valid].astype(np.float32)
    order = np.argsort(zs)
    zs = zs[order]; ts = ts[order]
    cuts = np.where(np.diff(zs) > DEPTH_CLUSTER_GAP_M)[0]
    groups = np.split(np.arange(zs.size), cuts + 1)

    best = None
    best_score = -1.0
    for g in groups:
        if g.size < DEPTH_MIN_PIX:
            continue
        z = zs[g]
        zc = float(np.median(z))
        tf = float(np.mean(ts[g]))
        sd = float(np.std(z))
        if float(z.max() - z.min()) > 0.05:   # 跨了两个表面，不要
            continue
        s = 0.55 * (0.15 + 0.85 * tf) + 0.25 * min(1.0, g.size / 10.0)
        if prior is not None and np.isfinite(prior):
            s += 0.20 * float(np.exp(-0.5 * ((zc - prior) / DEPTH_PRIOR_WIN_M) ** 2))
        else:
            s += 0.20
        if s > best_score:
            best_score = s
            best = (zc, max(0.003, min(sd, 0.02)), tf, int(g.size))
    return best


# ================= 4. 设备与标定初始化 =================
if not os.path.exists(CALIB_FILE):
    print(f"[错误] 未找到标定文件 {CALIB_FILE}，请先执行标定程序！")
    raise SystemExit(1)

calib_data = np.load(CALIB_FILE)
mp_hands = mp.solutions.hands
ctx = rs.context()
devices = ctx.query_devices()

pipelines, align_objects, hands_trackers, serials = [], [], [], []
extrinsics_list, cams = [], []
device_configs = []

for i, dev in enumerate(devices):
    serial = dev.get_info(rs.camera_info.serial_number)
    name = dev.get_info(rs.camera_info.name)
    if serial not in calib_data:
        continue

    for sensor in dev.query_sensors():
        if sensor.supports(rs.option.frames_queue_size):
            try:
                sensor.set_option(rs.option.frames_queue_size, 1)
            except Exception:
                pass
        if sensor.supports(rs.option.global_time_enabled):
            try:
                sensor.set_option(rs.option.global_time_enabled, 1)
            except Exception:
                pass

    pipeline = rs.pipeline(ctx)
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, STREAM_FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, STREAM_FPS)

    profile = pipeline.start(config)
    pipelines.append(pipeline)
    serials.append(serial)
    align_objects.append(rs.align(rs.stream.color))

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    T_wc = np.asarray(calib_data[serial], dtype=np.float64)
    if T_wc.shape != (4, 4):
        print(f"[错误] 标定文件里 {serial} 的外参不是 4x4，实际 {T_wc.shape}")
        raise SystemExit(1)
    extrinsics_list.append(T_wc)

    K = np.array([[intr.fx, 0.0, intr.ppx],
                  [0.0, intr.fy, intr.ppy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    cams.append(make_cam(K, T_wc))

    hands_trackers.append(
        mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=MP_MODEL_COMPLEXITY,
            min_detection_confidence=0.50,
            min_tracking_confidence=0.50
        )
    )

    device_configs.append({
        "device_name": f"Cam_{i + 1}",
        "device_model": name,
        "serial_number": serial,
        "operation_system": "Windows 10/11 64-bit",
        "control_algorithm": "MediaPipe_Hands_v0.10 + MultiView_Triangulation_v3.0",
        "cam_view": "第三人称_顶视" if i == 0 else f"第三人称_侧视_{i}",
        "cam_collect_feq": FPS,
        "cam_resolution": [WIDTH, HEIGHT],
        "cam_intrinsic_parameters": [intr.fx, intr.fy, intr.ppx, intr.ppy] + list(intr.coeffs),
        "cam_extrinsic_parameters": T_wc.flatten().tolist()
    })

num_cams = len(pipelines)
if num_cams == 0:
    print("未检测到已标定的相机！")
    raise SystemExit(1)

# ---- 外参自检：旋转必须正交、行列式为 +1，否则后面的重投影会整体错位 ----
_bad = False
for c in range(num_cams):
    R = extrinsics_list[c][:3, :3]
    orth = float(np.abs(R @ R.T - np.eye(3)).max())
    det = float(np.linalg.det(R))
    if orth > 1e-3 or abs(det - 1.0) > 1e-3:
        _bad = True
        print(f"[警告] Cam{c + 1} 外参旋转异常: 正交误差={orth:.2e}, det={det:.4f} -> 请重新标定")
if not _bad:
    print("[外参自检] 旋转矩阵全部正交且 det=+1，通过。")
for a in range(num_cams):
    for b in range(a + 1, num_cams):
        base = np.linalg.norm(extrinsics_list[a][:3, 3] - extrinsics_list[b][:3, 3])
        print(f"    Cam{a + 1} <-> Cam{b + 1} 基线: {base * 1000:.1f} mm")

np.seterr(all="ignore")

# 三路 MediaPipe 并行（每路一个独立 tracker，互不共享状态）
mp_pool = ThreadPoolExecutor(max_workers=num_cams, thread_name_prefix="mp")


# ================= 5. 规范数据集落盘 =================
def save_standardized_episode(ep_idx, frames_buffer, total_expected,
                              sync_violations, stale_frames, elapsed_s):
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    ep_name = f"episode_{ep_idx:04d}_{ts_str}"
    ep_dir = os.path.join(DATASET_ROOT, ep_name)
    os.makedirs(ep_dir, exist_ok=True)

    print(f"\n>>> 正在按照《具身智能数据采集规范》写入数据集: {ep_dir} ...")

    for c_idx, sn in enumerate(serials):
        c_dir = os.path.join(ep_dir, f"cam_{c_idx + 1}_{sn}")
        if SAVE_RAW_IMAGES:
            os.makedirs(os.path.join(c_dir, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(c_dir, "depth"), exist_ok=True)

    config_json = {
        "data_declaration": DATASET_DECLARATION,
        "environment_data": ENVIRONMENT_INFO,
        "collection_devices": device_configs,
        "timestamp_setting": {
            "timestamp_format": "UNIX",
            "timestamp_type": "绝对时间(UTC) + 相对时间(s) + 相机硬件时间戳(ms)",
            "sync_note": "三台相机未做硬件触发同步(inter_cam_sync_mode 实测无效)，帧间相位固定约 20.6ms(0.3 帧)；同步门限放宽至 25ms"
        },
        "cam_relations": "3台D435i通过ChArUco标定板统一至桌面世界坐标系(+Z垂直桌面向上)",
        "reconstruction_method": "单视角: 2D关键点 + 对齐深度反投影; 跨视角: 三角化几何校验 + 深度/2D/骨长联合投票; "
                                 "被物体遮挡的关节用 视线+骨长 求解或按上一帧推断(见 joints_inferred_mask)",
        "coordinate_note": "pose_trajectory.joints_pose 与 global_pose 已转换为仿真系(y、z取反)，"
                           "collection_devices.cam_extrinsic_parameters 仍为世界系"
    }
    config_path = os.path.join(ep_dir, "config_metadata.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2, ensure_ascii=False)

    pose_records, visual_records = [], []
    sim_timestamps, sim_trajectories, sim_global_poses = [], [], []
    sim_valid_masks, sim_inferred_masks = [], []
    reproj_pool = []
    joints_pool = []
    measured_pool = []
    views_pool = []

    first_pts = frames_buffer[0]["joints_3d"]
    ref = np.where(np.isfinite(first_pts[0]), first_pts[0], 0.0)
    canonical_pose = nan_to_list(first_pts - ref)

    for seq_idx, item in enumerate(frames_buffer):
        seq_id = seq_idx + 1
        utc_iso = datetime.fromtimestamp(item["unix_ts"], tz=timezone.utc).isoformat()
        pts_3d = item["joints_3d"]
        g_pose = compute_hand_global_pose(pts_3d)
        mask = np.asarray(item["valid_mask"], dtype=bool)
        inf_mask = np.asarray(item["inferred_mask"], dtype=bool)

        sim_timestamps.append(item["rel_ts"])
        sim_trajectories.append(pts_3d)
        sim_global_poses.append(g_pose if g_pose is not None else [0.0] * 7)
        sim_valid_masks.append(mask)
        sim_inferred_masks.append(inf_mask)
        joints_pool.append(int(mask.sum()))
        measured_pool.append(int((mask & ~inf_mask).sum()))
        views_pool.append(int(item["n_views"]))
        if item["reproj_px"] is not None:
            reproj_pool.append(float(item["reproj_px"]))

        cam_frame_meta = []
        for c_idx in range(num_cams):
            rgb_rel_path = f"cam_{c_idx + 1}_{serials[c_idx]}/rgb/{seq_id:06d}.jpg"
            depth_rel_path = f"cam_{c_idx + 1}_{serials[c_idx]}/depth/{seq_id:06d}.png"
            if SAVE_RAW_IMAGES:
                cv2.imwrite(os.path.join(ep_dir, rgb_rel_path), item["rgb_list"][c_idx])
                cv2.imwrite(os.path.join(ep_dir, depth_rel_path), item["depth_list"][c_idx])

            cam_frame_meta.append({
                "cam_data_seq": seq_id,
                "cam_timestamp": utc_iso,
                "cam_frame_timestamp_ms": opt_float(item["cam_ts_ms"][c_idx], 3),
                "cam_frame_stale": bool(item["stale"][c_idx]),
                "serial_number": serials[c_idx],
                "cam_image_frame": rgb_rel_path,
                "cam_depth_map": depth_rel_path
            })
        visual_records.append(cam_frame_meta)

        pose_records.append({
            "pose_data_seq": seq_id,
            "pose_timestamp": utc_iso,
            "unix_timestamp": item["unix_ts"],
            "rel_timestamp": round(item["rel_ts"], 6),
            "cam_frame_timestamps_ms": [opt_float(x, 3) for x in item["cam_ts_ms"]],
            "sync_error_ms": opt_float(item["sync_err_ms"], 2),
            "joints_name": HAND_JOINT_NAMES,
            "joints_pose": nan_to_list(pts_3d),
            "joints_valid_mask": mask.astype(int).tolist(),
            "joints_inferred_mask": inf_mask.astype(int).tolist(),
            "joints_state": np.asarray(item["state"], dtype=int).tolist(),
            "num_views": int(item["n_views"]),
            "reproj_rms_px": opt_float(item["reproj_px"], 3),
            "global_pose": g_pose
        })

    trajectory_json = {
        "task_description": {**TASK_DESCRIPTION, "episode": ep_name},
        "canonical_pose": canonical_pose,
        "canonical_note": "canonical_pose = 首帧 joints_pose - 首帧腕点；无效关节以 0 填充，"
                          "请配合 joints_valid_mask / joints_inferred_mask 使用",
        "joint_state_note": "joints_state: 0=缺失 1=实测(至少一路可信观测) 2=推断(被遮挡时由骨长/上一帧维持)，"
                            "训练时建议只用 joints_valid_mask=1 且 joints_inferred_mask=0 的点",
        "pose_trajectory": pose_records,
        "visual_data_index": visual_records
    }
    pose_json_path = os.path.join(ep_dir, "pose_trajectory.json")
    with open(pose_json_path, "w", encoding="utf-8") as f:
        json.dump(trajectory_json, f, indent=2, ensure_ascii=False)

    npz_path = os.path.join(ep_dir, "trajectory_sim_ready.npz")
    np.savez(
        npz_path,
        timestamps=np.array(sim_timestamps, dtype=np.float32),
        trajectories=np.array(sim_trajectories, dtype=np.float32),    # 无效关节为 NaN
        valid_masks=np.array(sim_valid_masks, dtype=bool),
        inferred_masks=np.array(sim_inferred_masks, dtype=bool),
        global_poses=np.array(sim_global_poses, dtype=np.float32)
    )
    np.savez(
        "grasp_trajectory.npz",
        timestamps=np.array(sim_timestamps, dtype=np.float32),
        trajectories=np.array(sim_trajectories, dtype=np.float32),
        valid_masks=np.array(sim_valid_masks, dtype=bool),
        inferred_masks=np.array(sim_inferred_masks, dtype=bool)
    )

    valid_frames = len(frames_buffer)
    lost_frames = max(0, total_expected - valid_frames)
    loss_rate = lost_frames / max(1, total_expected)
    is_loss_qualified = (loss_rate <= MAX_LOSS_RATE)

    manifest = {
        "episode": ep_name,
        "quality_inspection": {
            "collect_duration_s": round(float(elapsed_s), 3),
            "realized_fps": round(valid_frames / max(1e-6, elapsed_s), 2),
            "total_frames_expected": int(total_expected),
            "total_frames_collected": valid_frames,
            "missing_frames_count": int(lost_frames),
            "data_loss_rate": round(loss_rate, 6),
            "pass_loss_rate_0.1_percent": bool(is_loss_qualified),
            "sync_tolerance_ms_limit": SYNC_TOLERANCE_MS,
            "sync_violation_frames": int(sync_violations),
            "stale_camera_frames": int(stale_frames),
            "mean_valid_joints_per_frame": round(float(np.mean(joints_pool)), 2) if joints_pool else 0.0,
            "mean_measured_joints_per_frame": round(float(np.mean(measured_pool)), 2) if measured_pool else 0.0,
            "mean_views_per_frame": round(float(np.mean(views_pool)), 2) if views_pool else 0.0,
            "reprojection_rms_px_mean": round(float(np.mean(reproj_pool)), 3) if reproj_pool else None,
            "reprojection_rms_px_p95": round(float(np.percentile(reproj_pool, 95)), 3) if reproj_pool else None
        },
        "sha256_checksums": {
            "config_metadata.json": sha256_file(config_path),
            "pose_trajectory.json": sha256_file(pose_json_path),
            "trajectory_sim_ready.npz": sha256_file(npz_path)
        }
    }
    manifest_path = os.path.join(ep_dir, "manifest_checksum.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    q = manifest["quality_inspection"]
    print(f">>> [保存完毕] {valid_frames} 帧 / 期望 {total_expected} 帧 | "
          f"丢失率 {loss_rate * 100:.3f}% (合规 <=0.1%: {'通过' if is_loss_qualified else '未通过'})")
    print(f">>> [质量] 实际帧率 {q['realized_fps']} fps | 平均有效关节 {q['mean_valid_joints_per_frame']}/21 "
          f"(实测 {q['mean_measured_joints_per_frame']}) | 平均视角数 {q['mean_views_per_frame']} | "
          f"重投影 {q['reprojection_rms_px_mean']} px")
    if q["reprojection_rms_px_mean"] is not None and q["reprojection_rms_px_mean"] > 8.0:
        print(">>> [警告] 重投影误差 > 8px，重点检查外参、时间同步、肤色掩膜是否失效！")
    print(f">>> [仿真就绪] 已同步更新 grasp_trajectory.npz，可直接运行 replay_in_3d_sim.py 回放！\n")


# ================= 6. 主采集循环 =================
latest_color_imgs = [np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8) for _ in range(num_cams)]
latest_raw_rgb = [np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8) for _ in range(num_cams)]
latest_raw_depth = [np.zeros((HEIGHT, WIDTH), dtype=np.uint16) for _ in range(num_cams)]

cam_uv = [[None] * 21 for _ in range(num_cams)]        # 本帧 2D
cam_obs = [[None] * 21 for _ in range(num_cams)]       # 本帧 Obs
cam_stamp = [-1] * num_cams
cam_has_hand = [False] * num_cams
last_z = [[np.nan] * 21 for _ in range(num_cams)]      # 本视角上次可信深度
last_z_frame = [[-999] * 21 for _ in range(num_cams)]

bone_model = BoneModel()
smoothed_hand_3d = None

is_recording = False
debug_tissue = False
episode_counter = 1
frames_buffer = []
sync_violation_count = 0
stale_frame_count = 0
record_start_perf = 0.0
frame_id = 0

window_name = "SJ/T Standard Embodied Data Collector v4 (SPACE/R: record, Q: quit, D: debug)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("\n================== 具身智能标准数据采集系统 v3 ==================")
print("1. 【重要】请先用鼠标左键点击弹出的相机画面窗口，使其获得键盘焦点！")
print("2. 按 【空格键 (Space)】 或 【r】 键：开始录制当前回合(Episode)，左上角出现红点 ● REC。")
print("3. 抓取完成后，再按一次 【空格键】 或 【r】 键：停止录制并自动生成全套规范数据集。")
print("4. 按 【d】 切换调试显示(红圈 = 该关键点处检测不到手，多半被物体挡住)。")
print("5. 按 【q】 或 【ESC】 键退出系统。")
print("画面上 reproj 为当前融合骨架与本视角绿点的像素残差，正常应 < 4px；")
print("骨架颜色：品红=实测，橙色=推断(被遮挡，靠骨长/上一帧维持)。")
print("=================================================================\n")

try:
    frame_dt = 1.0 / max(FPS, 1)
    while True:
        raw_framesets = [None] * num_cams
        grab_ts = {}
        domains = set()

        # 帧驱动：等三路都出新帧(最多等 1.5 个帧周期)，保证三路是同一"代"，
        # 同步误差才不会因为"睡过头"而变成一整帧。
        _wait_until = time.perf_counter() + 1.5 * frame_dt
        while True:
            for idx in range(num_cams):
                fs = pipelines[idx].poll_for_frames()
                if fs:
                    raw_framesets[idx] = fs
                    grab_ts[idx] = float(fs.get_timestamp())
                    domains.add(fs.get_frame_timestamp_domain())
            if len(grab_ts) >= num_cams or time.perf_counter() >= _wait_until:
                break
            time.sleep(0.001)
        if not grab_ts:
            time.sleep(0.002)
            continue

        # ---- 同步误差：如实统计，不再取模伪造 ----
        sync_known = (len(grab_ts) >= 2 and len(domains) == 1)
        sync_err_ms = (max(grab_ts.values()) - min(grab_ts.values())) if sync_known else float("inf")

        # ---- 阶段 A：三路取帧 + 预处理，并把 MediaPipe 提交到线程池 ----
        pending = []
        for idx in range(num_cams):
            frames = raw_framesets[idx]
            if not frames:
                continue

            aligned_frames = align_objects[idx].process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data()).copy()
            depth_image = np.asanyarray(depth_frame.get_data())
            latest_raw_rgb[idx] = color_image.copy()
            latest_raw_depth[idx] = depth_image.copy()

            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            pending.append((idx, color_image, depth_image,
                            mp_pool.submit(hands_trackers[idx].process, image_rgb)))

        # 三路都没有新帧：说明循环跑得比相机快，这一轮不占帧号
        if not pending:
            time.sleep(0.002)
            continue

        frame_id += 1

        # ---- 阶段 B：收结果 -> 2D + 深度 -> 观测 ----
        for idx, color_image, depth_image, fut in pending:
            try:
                results = fut.result()
            except Exception as exc:
                print(f"[警告] Cam{idx+1} MediaPipe 异常: {exc}")
                continue

            cam = cams[idx]
            cam_uv[idx] = [None] * 21
            cam_obs[idx] = [None] * 21
            cam_stamp[idx] = frame_id
            cam_has_hand[idx] = False

            if not results.multi_hand_landmarks:
                for j in range(21):
                    last_z[idx][j] = np.nan
                    last_z_frame[idx][j] = -999
                latest_color_imgs[idx] = color_image
                continue

            lm = results.multi_hand_landmarks[0].landmark
            uv = np.array([[lm[j].x * WIDTH, lm[j].y * HEIGHT] for j in range(21)],
                          dtype=np.float64)
            uvi = np.clip(np.round(uv).astype(np.int32), 0, [WIDTH - 1, HEIGHT - 1])

            # 手部 bbox -> 肤色掩膜 + 掌心深度
            x0 = int(max(0, uvi[:, 0].min())); x1 = int(min(WIDTH - 1, uvi[:, 0].max()))
            y0 = int(max(0, uvi[:, 1].min())); y1 = int(min(HEIGHT - 1, uvi[:, 1].max()))
            bbox = (x0, y0, x1 + 1, y1 + 1)
            tissue = tissue_mask(color_image, bbox)
            hand_zc = estimate_hand_depth(depth_image, bbox)

            # 掩膜整体失效(极端光照/深肤色)时，不要用它惩罚所有关节。
            # 判据只用掌心/腕部这几个点：它们几乎不可能被物体完全挡住。
            core = [0, 5, 9, 13, 17]
            tf_core = [bool(tissue[int(uvi[j, 1]), int(uvi[j, 0])]) for j in core]
            tissue_ok = int(np.sum(tf_core)) >= 4

            for j in range(21):
                u, v = float(uv[j, 0]), float(uv[j, 1])
                if not (0 <= u < WIDTH and 0 <= v < HEIGHT):
                    continue
                cam_uv[idx][j] = (u, v)

                prior = None
                if (frame_id - last_z_frame[idx][j]) <= DEPTH_PRIOR_MAX_AGE:
                    prior = last_z[idx][j]

                res = sample_joint_depth(depth_image, int(uvi[j, 0]), int(uvi[j, 1]),
                                         tissue, prior, hand_zc)
                if res is None:
                    continue
                z, sd, tf, npix = res
                p3 = backproject(cam, u, v, z)
                t_use = tf if tissue_ok else 1.0
                cam_obs[idx][j] = Obs(cam, (u, v), p3, sigma=sd, tissue=t_use)

                if t_use >= TISSUE_MIN_FRAC:      # 只有"像手"的采样才更新先验
                    last_z[idx][j] = z
                    last_z_frame[idx][j] = frame_id

                if debug_tissue and t_use < TISSUE_MIN_FRAC:
                    cv2.circle(color_image, (int(uvi[j, 0]), int(uvi[j, 1])), 5, (0, 0, 255), 2)
                else:
                    cv2.circle(color_image, (int(uvi[j, 0]), int(uvi[j, 1])), 4, (0, 255, 0), -1)

            cam_has_hand[idx] = True
            latest_color_imgs[idx] = color_image

        # ---- 跨视角联合解算 ----
        fresh = [c for c in range(num_cams) if cam_stamp[c] == frame_id]
        has_stale = len(fresh) < num_cams
        # 只用本帧真正更新过的相机，避免把上一帧的观测混进本帧几何
        obs_by_cam = [cam_obs[c] if c in fresh else [None] * 21 for c in range(num_cams)]

        pts, scores, state = solve_pose(
            obs_by_cam, cams,
            prev=smoothed_hand_3d,
            boneL=bone_model.L if bone_model.L else None,
            boneX=bone_model.Lx if bone_model.Lx else None
        )

        # ---- 骨长模型在线更新(只用高置信实测点) ----
        bone_model.update(pts, state, scores)

        # ---- 时间滤波：稳态 EMA，突变直通 ----
        cur_valid = state > 0
        if smoothed_hand_3d is None:
            smoothed_hand_3d = np.full((21, 3), np.nan)
        both = cur_valid & np.isfinite(smoothed_hand_3d).all(axis=1)
        if both.any():
            jump = np.linalg.norm(pts[both] - smoothed_hand_3d[both], axis=1)
            # 速度自适应：静止时强平滑压抖动，快动作时放开以免拖影
            a = EMA_ALPHA + (1.0 - EMA_ALPHA) * np.clip(jump / EMA_SPEED_FULL_M, 0.0, 1.0)
            a = np.where(jump > JUMP_TOL_M, 1.0, a)[:, None]
            smoothed_hand_3d[both] = a * pts[both] + (1.0 - a) * smoothed_hand_3d[both]
        newj = cur_valid & ~np.isfinite(smoothed_hand_3d).all(axis=1)
        if newj.any():
            smoothed_hand_3d[newj] = pts[newj]
        # 本帧彻底缺测的关节：清空，避免把僵点一直带着
        smoothed_hand_3d[~cur_valid] = np.nan

        out_pts = None
        if np.isfinite(smoothed_hand_3d).any():
            out_pts = smoothed_hand_3d.copy()

        # ---- 重投影质量检查 ----
        reproj_uv = [None] * num_cams
        reproj_err_px = [None] * num_cams
        if out_pts is not None:
            for idx in range(num_cams):
                cam = cams[idx]
                uvj = [None] * 21
                errs = []
                for j in range(21):
                    if not np.isfinite(out_pts[j]).all():
                        continue
                    p = project(cam, out_pts[j])
                    if p is None:
                        continue
                    uvj[j] = (int(round(p[0])), int(round(p[1])))
                    if cam_stamp[idx] == frame_id and cam_uv[idx][j] is not None:
                        errs.append(float(np.hypot(p[0] - cam_uv[idx][j][0],
                                                   p[1] - cam_uv[idx][j][1])))
                reproj_uv[idx] = uvj
                reproj_err_px[idx] = float(np.mean(errs)) if errs else None

        # ---- 录制落盘缓存 ----
        if is_recording:
            valid_mask = np.zeros(21, dtype=bool)
            inferred_mask = np.zeros(21, dtype=bool)
            if out_pts is not None:
                valid_mask = np.isfinite(out_pts).all(axis=1)
                inferred_mask = valid_mask & (state == 2)

            if sync_err_ms > SYNC_TOLERANCE_MS:
                sync_violation_count += 1
            if has_stale:
                stale_frame_count += 1

            if valid_mask.sum() >= MIN_VALID_JOINTS and valid_mask[0]:
                sim_pts = to_sim_frame(out_pts)
                sim_pts[~valid_mask] = np.nan

                errs_all = [e for e in reproj_err_px if e is not None]
                frames_buffer.append({
                    "unix_ts": time.time(),
                    "rel_ts": time.perf_counter() - record_start_perf,
                    "cam_ts_ms": [grab_ts.get(c, float("nan")) for c in range(num_cams)],
                    "sync_err_ms": sync_err_ms,
                    "n_views": int(sum(1 for c in fresh if cam_has_hand[c])),
                    "valid_mask": valid_mask.copy(),
                    "inferred_mask": inferred_mask.copy(),
                    "state": state.copy(),
                    "joints_3d": sim_pts,
                    "reproj_px": float(np.mean(errs_all)) if errs_all else None,
                    "stale": [cam_stamp[c] != frame_id for c in range(num_cams)],
                    "rgb_list": [img.copy() for img in latest_raw_rgb] if SAVE_RAW_IMAGES else [],
                    "depth_list": [d.copy() for d in latest_raw_depth] if SAVE_RAW_IMAGES else []
                })
                if SAVE_RAW_IMAGES and len(frames_buffer) % 150 == 0:
                    est_mb = len(frames_buffer) * num_cams * (WIDTH * HEIGHT * 3 + WIDTH * HEIGHT * 2) / 1e6
                    print(f"[提示] 已缓存 {len(frames_buffer)} 帧，占用内存约 {est_mb:.0f} MB")

        # ---- 可视化渲染 ----
        views = []
        for idx in range(num_cams):
            vis = latest_color_imgs[idx].copy()
            uvj = reproj_uv[idx]
            is_fresh = (cam_stamp[idx] == frame_id)

            if uvj is not None:
                for s_idx, e_idx in mp_hands.HAND_CONNECTIONS:
                    if uvj[s_idx] and uvj[e_idx]:
                        inf = (state[s_idx] == 2) or (state[e_idx] == 2)
                        cv2.line(vis, uvj[s_idx], uvj[e_idx],
                                 (0, 140, 255) if inf else (255, 0, 255), 2)
                for j in range(21):
                    if uvj[j]:
                        cv2.circle(vis, uvj[j], 2,
                                   (0, 165, 255) if state[j] == 2 else (255, 0, 255), -1)

            sync_color = (0, 255, 0) if sync_err_ms <= SYNC_TOLERANCE_MS else (0, 0, 255)
            sync_txt = f"{sync_err_ms:.1f}ms" if np.isfinite(sync_err_ms) else "n/a"
            cv2.putText(vis, f"Cam {idx + 1} | SyncErr: {sync_txt}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sync_color, 2)

            e = reproj_err_px[idx]
            q_txt = f"reproj: {e:.1f}px" if e is not None else "reproj: --"
            if uvj is not None:
                n_j = int(sum(1 for x in uvj if x))
                q_txt += f" | joints: {n_j}/21"
            cv2.putText(vis, q_txt, (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255) if (e is None or e < 8.0) else (0, 0, 255), 2)

            n_inf = int(np.sum(state == 2))
            if n_inf:
                cv2.putText(vis, f"inferred: {n_inf}", (15, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

            if not is_fresh:
                cv2.putText(vis, "STALE", (WIDTH - 110, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if is_recording:
                cv2.circle(vis, (25, 120), 10, (0, 0, 255), -1)
                cv2.putText(vis,
                            f"REC Ep:{episode_counter:02d} | Frames:{len(frames_buffer)} (SPACE to Save)",
                            (45, 127), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(vis, "Press [SPACE] or [R] to Start Recording",
                            (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            views.append(vis)

        combined = np.hstack(views)
        if combined.shape[1] > 1600:
            s = 1600.0 / combined.shape[1]
            combined = cv2.resize(combined, (0, 0), fx=s, fy=s)
        cv2.imshow(window_name, combined)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('r'), ord('R'), 32]:
            if not is_recording:
                is_recording = True
                frames_buffer.clear()
                sync_violation_count = 0
                stale_frame_count = 0
                record_start_perf = time.perf_counter()
                print(f"\n>>> [开始采集 Episode {episode_counter:04d}] 请做抓取动作，完成后再按一次【空格键】保存...")
            else:
                is_recording = False
                elapsed = max(1e-6, time.perf_counter() - record_start_perf)
                total_expected = max(int(round(elapsed * FPS)), len(frames_buffer))
                if len(frames_buffer) > 0:
                    save_standardized_episode(
                        episode_counter, frames_buffer, total_expected,
                        sync_violation_count, stale_frame_count, elapsed
                    )
                    episode_counter += 1
                else:
                    print(">>> [采集失败] 未捕获到有效手部轨迹，请先把右手放入画面后再按空格键！")
        elif key in [ord('d'), ord('D')]:
            debug_tissue = not debug_tissue
            print(f">>> [调试显示] 组织掩膜标注 {'开' if debug_tissue else '关'}")
        elif key in [ord('q'), ord('Q'), 27]:
            break


finally:
    try:
        mp_pool.shutdown(wait=False)
    except Exception:
        pass
    for hands in hands_trackers:
        hands.close()
    for pipe in pipelines:
        pipe.stop()
    cv2.destroyAllWindows()