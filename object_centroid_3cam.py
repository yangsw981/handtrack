# -*- coding: utf-8 -*-
"""
三相机联合物体点云 → 物体中心位置
=====================================================================
输入：3 × Intel RealSense D435i（彩色+深度），外参 extrinsics_3cams.npz
输出：物体在世界系/仿真系的中心位置、尺寸、姿态，以及融合后的物体点云(.ply)

工作原理：
  1) 每台相机抓一帧，把对齐后的深度图整幅反投影成相机系点云；
  2) 用各自外参转到同一个"桌面世界系"（+Z 垂直桌面向上，桌面 z≈0）；
  3) 三份点云合并 → 体素降采样 → RANSAC 剔桌面平面 → DBSCAN 聚类；
  4) 取桌面之上、ROI 之内、点数最多的那个团块当作物体；
  5) 对团块同时给出三种中心估计：
       · 质心（均值，视角越密集的一侧会略微偏）
       · 最小包围盒 OBB 中心（对任意物体通用）
       · 圆柱拟合中心（对圆柱/罐体更准，并回半径，可用来验真值）
     默认选"圆柱拟合"（本数据集物体是圆柱罐体），拟合失败自动退到 OBB。

为什么用 open3d：体素降采样 + DBSCAN + OBB 一条龙，免写 C++；scipy 兜底。
（open3d 0.19.0 / scipy 1.13.1 已确认在本环境安装。）

用法：
  python object_centroid_3cam.py                  # 抓一帧，打印中心并出图/出点云
  python object_centroid_3cam.py --loop           # 连续显示（按 q 退出）
  python object_centroid_3cam.py --out out.json   # 结果写 JSON
  python object_centroid_3cam.py --selftest       # 不碰相机，用合成圆柱自检几何链路
  python object_centroid_3cam.py --roi -0.3 -0.3 0.3 0.3   # 只搜这个 XY 范围（米）

注意事项（本项目血泪史）：
  · 深度流组合用 848x480@15，绝不用 640x360（D435i 不支持，会假死不吐帧）；
  · 启动用 wait_for_frames 点流而不是 poll（poll 在流刚起时一直返回空）；
  · 三机无硬件触发，抓"最新一帧"即可——物体静止时 30ms 相位差无关紧要；
=====================================================================
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone

import numpy as np
import cv2
import pyrealsense2 as rs

CALIB_FILE = "extrinsics_3cams.npz"

W, H, FPS = 848, 480, 15
MIN_DEPTH_M = 0.15
MAX_DEPTH_M = 1.30
VOXEL_M = 0.006            # 体素降采样边长(米)
DBSCAN_EPS_M = 0.012       # 聚类邻域半径(米)
DBSCAN_MIN_PTS = 10        # 体素化后一个团块的最少邻居数（空心壳表面很薄，别设太大）
MIN_CLUSTER_PTS = 300      # 原始点数下限（体素化后 × ~1/8）
PLANE_RANSAC_M = 0.006     # 桌面平面拟合的容差(米)
TABLE_ABOVE_M = 0.008      # 高出桌面多少算"物体"（剔除桌面上残留点）
OBJECT_MAX_H_M = 0.40      # 物体高度上限(米)，过滤人/手臂
OBJECT_MAX_R_M = 0.35      # 物体 XY 半径上限(米)
OBJECT_MAX_DIAM_M = 0.25   # 物体 XY 尺寸上限(米)：实测背景/机架常连成 0.5m 大团块，必须挡掉
MAX_OBJ_ASPECT = 3.0       # 高/底面短边 上限：实测机架立柱(0.30m 高、0.05m 粗)会被当成物体
PRIOR_GATE_M = 0.06        # 上一帧中心 6cm 内算"同一个物体"（手遮挡时靠它接力）
PRIOR_MIN_PTS = 100        # 有先验时，候选点下限从 MIN_CLUSTER_PTS 放宽到这个
SURF_FIT_MAX_PTS = 20000   # 桌面曲面拟合最多用多少点（实时：全量 IRLS 要 300ms+）
SURF_CACHE_S = 2.0         # 桌面曲面缓存多久（秒）重算一次
FAST_MAX_PTS = 4000        # 锁定后直接拟合用的点数上限
TRACK_ALPHA = 0.45         # 实时追踪：中心位置 EMA 系数（越小越稳、越滞后）
LOST_RESET_FRAMES = 15     # 连续这么多帧找不到就取消锁定，回到全局搜索
WHITE_SAT_MAX = 0.20       # 低饱和��"白/灰"点）阈值：用它把物体从手上分出来
DEPROJ_STRIDE = 2          # 反投影隔点取：0.7m 处像素间距 ~1.2mm，隔一点 → 2.4mm，仍细于 5mm 体素
SKIN_REMOVE = True         # 是否剔除肤色（避免手被当成物体）
TABLE_Z_HINT = 0.0         # 世界系里桌面的高度（标定就是按桌面定义的，z≈0）
TABLE_BAND = 0.06          # 先验上下各 6cm 以内算"桌面这一层"
MIN_OBJ_HEIGHT_M = 0.020   # 团块高度低于 2cm 视为桌面碎片，不要
GREEN_REJECT = 0.60        # 绿桌面像素占比超过这个值视为桌面不是物体
Z_UP_SIGN = -1.0           # 世界 +Z 指向桌面【下方】（实测标定：cam1 在 z=-0.729、沿 +Z 看向桌面），
                           # 所以"高出桌面"= z 比桌面更小 → height = Z_UP_SIGN*(z - table_z)
BAND_TOP_M = 0.40          # 桌面之上最多关心 40cm（再往上就是机架/背景）

# 桌面世界系 → 仿真系（与 standard_embodied_collector 一致：y、z 取反）
def to_sim(p):
    o = np.array(p, dtype=np.float64, copy=True)
    o[..., 1] = -o[..., 1]
    o[..., 2] = -o[..., 2]
    return o


def rigid_inverse(T):
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


# ---------------- 几何工具 ----------------
def deproject_cloud(depth_m, intr):
    """整幅深度图反投影成相机系点云 (N,3) 米。仅保留有效深度。"""
    fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
    u = (np.arange(W, dtype=np.float64) - ppx) / fx
    v = (np.arange(H, dtype=np.float64) - ppy) / fy
    valid = (depth_m > MIN_DEPTH_M) & (depth_m < MAX_DEPTH_M)
    ys, xs = np.nonzero(valid)
    z = depth_m[valid]
    X = u[xs] * z
    Y = v[ys] * z
    return np.stack([X, Y, z], axis=1)


def ransac_plane(pts, thresh=PLANE_RANSAC_M, iters=400):
    """拟合主平面（桌面）。返回 (normal, offset, inlier_mask)。normal 已归一，d=dot(n,p)。"""
    pts = np.asarray(pts, dtype=np.float64)
    best = None
    for _ in range(iters):
        i = np.random.choice(pts.shape[0], 3, replace=False)
        p = pts[i]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        d = np.dot(pts, n)
        inl = np.abs(d - np.dot(p[0], n)) < thresh
        cnt = inl.sum()
        if best is None or cnt > best[0]:
            best = (cnt, n, np.dot(p[0], n), inl)
    return best[1], best[2], best[3]


def circle_fit_xy(pts):
    """Kasa 代数圆拟合（在 XY 平面）。返回 (cx, cy, r)。失败返回 None。"""
    pts = np.asarray(pts, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx = sol[0] / 2.0
    cy = sol[1] / 2.0
    r = np.sqrt(sol[2] + cx * cx + cy * cy)
    if not np.isfinite(r) or r <= 0.01 or r > OBJECT_MAX_R_M:
        return None
    # 拟合残差作为"是否真的圆"的判据
    resid = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r
    return float(cx), float(cy), float(r), float(np.sqrt(np.mean(resid ** 2)))


def fit_obb(pts):
    """PCA 最小包围盒。返回 center(3), size(3), R(3x3 列为主轴)。"""
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(axis=0)
    _, s, Vt = np.linalg.svd(pts - c, full_matrices=False)
    R = Vt.T
    q = (pts - c) @ R
    lo, hi = q.min(axis=0), q.max(axis=0)
    center = c + R @ ((lo + hi) / 2.0)
    size = hi - lo
    return center, size, R


def mat_to_quat(R):
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    else:
        k = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        i, j, l = (k + 1) % 3, (k + 2) % 3, k
        s = np.sqrt(1 + R[l, l] - R[i, i] - R[j, j]) * 2
        q = np.zeros(4)
        q[0 + k] = 0.25 * s
        q[3] = (R[j, i] - R[i, j]) / s
        q[i + 1] = (R[i, l] + R[l, i]) / s
        q[j + 1] = (R[j, l] + R[l, j]) / s
    q /= np.linalg.norm(q)
    return q  # [x,y,z,w]


def skin_mask(bgr):
    """肤色掩膜（YCrCb），把可能的手去掉，避免被当成物体。返回 bool (H,W)。"""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cb = ycrcb[:, :, 1].astype(np.float32)
    cr = ycrcb[:, :, 2].astype(np.float32)
    return (cb > 77) & (cb < 135) & (cr > 133) & (cr < 180)


# ---------------- 设备 ----------------
def roi_window(T_wc, intr, roi, table_z=0.0):
    """把 ROI 立方体投影成像素包围框（只在这块区域反投影，点数少 3~5 倍）。"""
    x0, y0, x1, y1 = roi
    z_lo = table_z - 0.02
    z_hi = table_z + BAND_TOP_M * Z_UP_SIGN
    Rc = np.asarray(T_wc, dtype=np.float64)[:3, :3].T
    t = np.asarray(T_wc, dtype=np.float64)[:3, 3]
    us, vs = [], []
    for x in (x0, x1):
        for y in (y0, y1):
            for z in (z_lo, z_hi):
                pc = Rc @ (np.array([x, y, z], dtype=np.float64) - t)
                if pc[2] <= 0.05:
                    continue
                us.append(intr.fx * pc[0] / pc[2] + intr.ppx)
                vs.append(intr.fy * pc[1] / pc[2] + intr.ppy)
    if not us:
        return None
    u0 = max(0, int(np.floor(min(us))) - 6)
    u1 = min(W, int(np.ceil(max(us))) + 6)
    v0 = max(0, int(np.floor(min(vs))) - 6)
    v1 = min(H, int(np.ceil(max(vs))) + 6)
    if u1 - u0 < 16 or v1 - v0 < 16:
        return None
    return (u0, u1, v0, v1)


class Cam:
    def __init__(self, serial, T_wc):
        self.win = None          # ROI 在像素上的包围框 (u0,u1,v0,v1)：只在框内反投影，快 3 倍
        self.serial = serial
        self.T_wc = np.asarray(T_wc, dtype=np.float64)
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
        prof = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        # 点流：等第一帧，超时报错（而不是静默假死）
        fs = self.pipe.wait_for_frames(8000)
        print("  [OK] %s 已出帧" % serial[-6:])

    def grab(self):
        """抓最新一帧，返回 (depth_m, bgr, ts_ms)。"""
        fs = self.pipe.wait_for_frames(5000)
        a = self.align.process(fs)
        d = np.asanyarray(a.get_depth_frame().get_data(), dtype=np.uint16).astype(np.float32) * 0.001
        c = np.asanyarray(a.get_color_frame().get_data())
        return d, c, fs.get_timestamp()

    def cloud_world(self, depth_m, bgr=None):
        """反投影 + 转到世界系。返回 (pts Nx3, colors Nx3, skin N)。

        ROI 窗口 + 隔点取（DEPROJ_STRIDE）：只算真正可能放物体的那块像素、每隔一点算一个，
        点数从 50 万降到十万级，单帧建云 50ms → 15ms；5mm 体素下采样后精度没有区别。
        """
        u0, u1, v0, v1 = self.win if self.win is not None else (0, W, 0, H)
        st = max(1, int(DEPROJ_STRIDE))
        sub = depth_m[v0:v1:st, u0:u1:st]
        valid = (sub > MIN_DEPTH_M) & (sub < MAX_DEPTH_M)
        ys, xs = np.nonzero(valid)
        z = sub[valid].astype(np.float64)
        uu = (u0 + xs * st - self.intr.ppx) / self.intr.fx
        vv = (v0 + ys * st - self.intr.ppy) / self.intr.fy
        pc = np.stack([uu * z, vv * z, z], axis=1)
        pts = (self.T_wc[:3, :3] @ pc.T).T + self.T_wc[:3, 3]
        colors = None
        skin = None
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            colors = rgb[v0 + ys * st, u0 + xs * st].astype(np.float32) / 255.0
            skin = skin_mask(bgr)[v0 + ys * st, u0 + xs * st]
        return pts, colors, skin

    def stop(self):
        try:
            self.pipe.stop()
        except Exception:
            pass


# ---------------- 核心：融合 + 找物体 ----------------
def is_green(rgb):
    """桌面是绿PVC。rgb ∈ [0,1]，返回是否偏绿。"""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return (g > 0.18) & (g > 1.25 * r) & (g > 1.25 * b)


def estimate_table_z(pts, hint=TABLE_Z_HINT, search_m=0.25, bin_m=0.005):
    """在工作区点云里自动找桌面高度（世界系 z）。

    为什么用 z 直方图而不是 RANSAC：工作区里地面、机架、背景墙都是大平面，
    RANSAC 很容易挑中地面那一片，把 table_z 拉低半米，整张桌子就都算成"物体"。
    但桌面是水平的、面积最大、点最密，所以在 z 轴上它是最高的一根尖峰。
    只要把搜索范围限在 hint±25cm，就不会被地面抢走；超过 ±15cm 的峰判为异常，
    退回 hint（这说明相机根本没拍到桌面）。
    """
    z = np.asarray(pts[:, 2], dtype=np.float64)
    lo, hi = hint - search_m, hint + search_m
    zz = z[(z > lo) & (z < hi)]
    if zz.size < 2000:
        return float(hint), "fallback(样本 %d 太少)" % zz.size
    edges = np.arange(lo, hi + bin_m, bin_m)
    hist, edges = np.histogram(zz, bins=edges)
    k = int(np.argmax(hist))
    c = float(0.5 * (edges[k] + edges[k + 1]))
    band = np.abs(z - c) < 0.02
    val = float(np.median(z[band])) if band.sum() > 200 else c
    if abs(val - hint) > 0.15:
        return float(hint), "fallback(峰 %.3f 偏离先验太远)" % val
    return val, "hist_peak(峰 %.3f, 样本 %d)" % (val, int(hist[k]))


def refine_table_z(pts, hint, band_m=0.04, min_n=500):
    """手工给了桌面高度时，只在 hint 附近取中位数精修（避免整体偏一点）。"""
    near = np.abs(pts[:, 2] - hint) < band_m
    if near.sum() < min_n:
        return float(hint), "manual(附近样本不足)"
    return float(np.median(pts[near, 2])), "manual_refined"


def fit_table_plane(band_pts, max_tilt_deg=12.0, max_rms_m=0.010):
    """用桌面峰带内的点拟合平面 n·p = d（n 朝上），返回 (n, d, tilt_deg, rms_m) 或 None。

    为什么不能只用"常数桌面高度 z"：桌面本身可能有点斜，深度在整幅图上也可能有
    系统性斜坡（温度/标定残差）。只要整张桌面比"常数 z"高出 4cm，桌面就会被判成
    "4cm 高的物体"，于是框落到空桌面上 —— 这正是真机截图里发生的事。
    用平面代替常数 z，桌面倾斜/斜坡都被吸收掉，桌面上的点高度恒为 0。
    """
    P = np.asarray(band_pts, dtype=np.float64)
    if P.shape[0] < 500:
        return None
    c = P.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    n = Vt[-1]
    if n[2] < 0:
        n = -n
    n = n / (np.linalg.norm(n) + 1e-12)
    tilt = float(np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0))))
    d = float(n @ c)
    rms = float(np.sqrt(np.mean((P @ n - d) ** 2)))
    if tilt > max_tilt_deg or rms > max_rms_m:
        return None
    return n, d, tilt, rms


def _surf_design(xy):
    x, y = xy[:, 0], xy[:, 1]
    return np.column_stack([np.ones_like(x), x, y, x * y, x * x, y * y])


def surface_z(coef, xy):
    """二次曲面 z = c0 + c1x + c2y + c3xy + c4x^2 + c5y^2 在 xy 处的取值。"""
    return _surf_design(np.asarray(xy, dtype=np.float64)) @ np.asarray(coef, dtype=np.float64)


_SURF = {"surf": None, "table_z": 0.0, "t": 0.0}


def fit_table_surface(band_pts, max_rms_m=0.012):
    """拟合"桌面"曲面（二次），返回 (coef, rms_m) 或 None。

    为什么不是平面：RealSense 的深度误差随距离增长，标定残差又让世界系里的桌面
    呈"中间凹/两边翘"的碗形；用平面时桌面两端会翘起 3~5cm，于是整张桌子被判成
    "几厘米高的物体"，DBSCAN 再把桌子和物体连成一片 → 反而什么都找不到。
    二次曲面能吃掉这种弯曲与倾斜，桌面上任意一点的高度都≈0。
    """
    P = np.asarray(band_pts, dtype=np.float64)
    if P.shape[0] < 2000:
        return None
    if P.shape[0] > SURF_FIT_MAX_PTS:          # 实时：拟合只用 2 万点，精度不变、耗时降一个量级
        P = P[::int(np.ceil(P.shape[0] / float(SURF_FIT_MAX_PTS)))]
    A = _surf_design(P[:, :2])
    z = P[:, 2]
    try:
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    except np.linalg.LinAlgError:
        return None
    # 迭代重加权（Cauchy）：桌点是绝大多数，物体点是少数会被自动压权。
    # 注意不能用"只保留残差在某区间内的点"那种硬剔除——一旦初值偏高，
    # 桌子上的点会被整片剔掉，拟合就顺着物体往上爬，最后桌上没有任何点"高于桌面"。
    for _ in range(6):
        r = z - A @ coef
        med = float(np.median(r))
        sig = 1.4826 * float(np.median(np.abs(r - med))) + 1e-6
        w = 1.0 / (1.0 + ((r - med) / (3.0 * sig)) ** 2)
        W = np.sqrt(w)
        try:
            coef, *_ = np.linalg.lstsq(A * W[:, None], z * W, rcond=None)
        except np.linalg.LinAlgError:
            break
    # 收尾：IRLS 之后曲面已经贴住桌面，此时再做一次"只留桌面核心带"的精拟合。
    # （物体一定在曲面之上 >8mm，所以上界 +4mm 能把它切掉；下界 -8mm 保留桌面噪声。）
    r = z - A @ coef
    core = (r > -0.008) & (r < 0.004)
    if int(core.sum()) > 1500:
        try:
            coef2, *_ = np.linalg.lstsq(A[core], z[core], rcond=None)
            r2 = z - A @ coef2
            if abs(float(np.median(r2))) < 0.005:   # 没有整体上移才采用
                coef = coef2
        except np.linalg.LinAlgError:
            pass
    r = z - A @ coef
    inl = np.abs(r - np.median(r)) < 3.0 * (1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6)
    rms = float(np.sqrt(np.mean(r[inl] ** 2))) if int(inl.sum()) > 0 else 9.9
    if rms > max_rms_m:
        return None
    return coef, rms


def localize_object(world_pts, world_colors, world_skin, roi,
                    table_z_hint=None, debug=False, diag=None,
                    above_m=TABLE_ABOVE_M, prior_xy=None, fast=False):
    """输入多相机世界系点云(合并)，输出物体中心。找不到时返回 None，并把原因写进 diag。

    table_z_hint=None 表示自动估桌面高度；给了数就按那个值精修（--table-z）。
    **不要用全局 RANSAC 找桌面**：地面/机架/背景墙都是大平面，RANSAC 一旦挑中地面，
    table_z 会被拉到 -0.7m，整张桌子都成了"桌面之上的物体"，最后聚出来的是
    桌面上的一块 20cm 见方的碎片（真机第一次跑就是这样）。
    """
    import open3d as o3d
    if diag is None:
        diag = {}

    # 1) 先只按 XY ROI 裁剪，不假设 z（否则桌面高度算错就全盘皆错）
    m_xy = ((world_pts[:, 0] > roi[0]) & (world_pts[:, 0] < roi[2]) &
            (world_pts[:, 1] > roi[1]) & (world_pts[:, 1] < roi[3]))
    n_roi = int(m_xy.sum())
    diag["n_roi"] = n_roi
    if n_roi < 500:
        diag["reason"] = "ROI 内只有 %d 点：相机没对着桌面，或者 --roi 范围太小" % n_roi
        return None
    pts_roi = world_pts[m_xy]
    col_roi = world_colors[m_xy] if world_colors is not None else None
    skn_roi = world_skin[m_xy] if world_skin is not None else None

    # 2) 桌面高度：自动找峰 / 按 hint 精修
    if table_z_hint is None:
        table_z, how = estimate_table_z(pts_roi)
    else:
        table_z, how = refine_table_z(pts_roi, float(table_z_hint))
    diag["table_z"] = table_z
    diag["table_z_how"] = how

    # 2b) 用峰带内的点拟合"桌面平面"，高度一律按平面算（桌面倾斜/深度斜坡都能吸收）
    band0 = np.abs(pts_roi[:, 2] - table_z) < 0.025
    surf = None
    now = time.time()
    if (not debug and _SURF["surf"] is not None
            and abs(_SURF["table_z"] - table_z) < 0.01
            and now - _SURF["t"] < SURF_CACHE_S):
        surf = _SURF["surf"]                    # 2 秒内复用同一张桌面曲面
    else:
        surf = fit_table_surface(pts_roi[band0]) if int(band0.sum()) > 3000 else None
        if surf is not None and not debug:
            _SURF.update(surf=surf, table_z=float(table_z), t=now)
    diag["table_surface"] = None if surf is None else {
        "rms_mm": round(surf[1] * 1000.0, 2), "coef": [float(x) for x in surf[0]]}
    hz_roi = None
    if surf is not None:
        hz_try = Z_UP_SIGN * (pts_roi[:, 2] - surface_z(surf[0], pts_roi[:, :2]))
        # 自检：正常场景里总会有少量点高出桌面（物体本身）。若几乎一个都没有，
        # 说明曲面拟合爬到物体顶部去了（真实数据里出现过：桌面之上=0 点）。
        if float(np.mean(hz_try > max(above_m, 0.008))) < 5e-4:
            diag["table_surface_unusable"] = True
            surf = None
        else:
            hz_roi = hz_try
    if hz_roi is None:
        hz_roi = Z_UP_SIGN * (pts_roi[:, 2] - table_z)

    # 高度分布快照：失败时一眼看出"到底有没有东西高出桌面"
    hv = hz_roi[np.isfinite(hz_roi)]
    if hv.size:
        diag["hz_stats"] = {
            "max_mm": round(float(hv.max()) * 1000.0, 1),
            "p999_mm": round(float(np.percentile(hv, 99.9)) * 1000.0, 1),
            "n_gt_5mm": int((hv > 0.005).sum()),
            "n_gt_10mm": int((hv > 0.010).sum()),
            "n_gt_30mm": int((hv > 0.030).sum()),
            "n_gt_80mm": int((hv > 0.080).sum()),
        }

    # 3) 桌面之上 z 带（剥掉地面、墙、背景）
    m = (hz_roi > -TABLE_BAND) & (hz_roi < BAND_TOP_M)
    diag["n_band"] = int(m.sum())
    if m.sum() < 500:
        diag["reason"] = "桌面上方 z 带内只有 %d 点：桌面高度 %.3f 可能不对" % (m.sum(), table_z)
        return None
    pts = pts_roi[m]
    hgt = hz_roi[m]
    col = col_roi[m] if col_roi is not None else None
    skn = skn_roi[m] if skn_roi is not None else None

    # 4) 桌面之上 + 高度上限 + 去肤色
    sel = (hgt > above_m) & (hgt < OBJECT_MAX_H_M)
    if skn is not None and SKIN_REMOVE:
        sel &= ~skn
    if prior_xy is not None:
        prior_xy = np.asarray(prior_xy, dtype=np.float64).ravel()[:2]
    cand = pts[sel]
    prior_mask = None
    if prior_xy is not None and cand.shape[0] >= PRIOR_MIN_PTS:
        dxy = np.linalg.norm(cand[:, :2] - np.asarray(prior_xy, dtype=np.float64), axis=1)
        near = cand[dxy <= PRIOR_GATE_M]
        if near.shape[0] >= PRIOR_MIN_PTS:
            diag["prior_restrict"] = "%d -> %d 点（收到上一帧中心 %.0f mm 内）" % (
                cand.shape[0], near.shape[0], PRIOR_GATE_M * 1000)
            prior_mask = dxy <= PRIOR_GATE_M
            cand = near
    diag["n_above"] = int(cand.shape[0])
    need_pts = PRIOR_MIN_PTS if prior_xy is not None else MIN_CLUSTER_PTS
    if cand.shape[0] < need_pts:
        diag["reason"] = "桌面之上只有 %d 点（< %d）：物体可能被手挡住或不在 ROI 里" % (
            cand.shape[0], need_pts)
        return None
    cand_col = col[sel] if col is not None else None
    if cand_col is not None and prior_mask is not None:
        cand_col = cand_col[prior_mask]

    # 4) 体素降采样 + DBSCAN（锁定时跳过：先验附近已经是"只有一个物体"的小点集）
    if fast and prior_mask is not None and cand.shape[0] >= PRIOR_MIN_PTS:
        step = max(1, cand.shape[0] // FAST_MAX_PTS)
        downs = cand[::step]
        labels = np.zeros(downs.shape[0], dtype=np.int32)
        diag["clustering"] = "skipped(locked)"
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cand)
        pcd = pcd.voxel_down_sample(VOXEL_M)
        labels = np.asarray(pcd.cluster_dbscan(eps=DBSCAN_EPS_M, min_points=DBSCAN_MIN_PTS))
        if labels.size == 0:
            return None
        downs = np.asarray(pcd.points)
        diag["clustering"] = "dbscan"

    # 5) 逐个团块量尺寸，滤掉"桌面碎片"（扁）、"背景大片"（大）、"绿桌面残留"
    stats = []
    for lb in sorted(set(labels.tolist())):
        if lb < 0:
            continue
        cp = downs[labels == lb]
        ext = np.ptp(cp[:, :2], axis=0)
        if surf is not None:
            hz = Z_UP_SIGN * (cp[:, 2] - surface_z(surf[0], cp[:, :2]))
        else:
            hz = Z_UP_SIGN * (cp[:, 2] - table_z)
        hh = float(np.percentile(hz, 98))
        frac_above = float(np.mean(hz > TABLE_ABOVE_M))
        d_prior = (float(np.linalg.norm(cp[:, :2].mean(axis=0) - np.asarray(prior_xy, float)))
                   if prior_xy is not None else None)
        rec = {"lb": lb, "n": int((labels == lb).sum()), "height": hh,
               "extent": ext.tolist(), "frac_above": frac_above,
               "dist_prior": d_prior,
               "center": cp.mean(axis=0).tolist(), "green": None, "reject": None}

        # ---- 逐条判据，每条都把拒绝原因记下来（失败时要靠它定位） ----
        if hh < MIN_OBJ_HEIGHT_M:
            rec["reject"] = "太扁 h=%.3f<%.3f" % (hh, MIN_OBJ_HEIGHT_M)
        elif hh < 0.35 * float(min(ext)):
            rec["reject"] = "扁平 h=%.3f<0.35*底面%.3f" % (hh, 0.35 * float(min(ext)))
        elif frac_above < 0.25:
            rec["reject"] = "贴桌点太多 %.0f%%" % (frac_above * 100.0)
        elif hh > OBJECT_MAX_H_M:
            rec["reject"] = "太高 h=%.3f>%.3f" % (hh, OBJECT_MAX_H_M)
        elif float(max(ext)) < 0.03:
            rec["reject"] = "太小 底面%.3fm" % float(max(ext))
        elif float(ext.max()) > OBJECT_MAX_DIAM_M:
            rec["reject"] = "底面太大 %.3fm>%.2fm" % (float(ext.max()), OBJECT_MAX_DIAM_M)
        elif hh / max(float(min(ext)), 0.01) > MAX_OBJ_ASPECT:
            rec["reject"] = "细高(高/底面短边=%.1f>%.1f) 像机架立柱" % (
                hh / max(float(min(ext)), 0.01), MAX_OBJ_ASPECT)

        # 手搭在物体顶上时，肤色掩膜常常漏判（逆光/手背），高度会被手带高一倍。
        # 物体是白色/浅色（饱和度低）、手是肉色（饱和度高），所以在团块包围框内
        # 只用"低饱和点"重算一次高度；点不够就保持原值。
        if cand_col is not None:
            _lo = cp[:, :2].min(axis=0) - 0.02
            _hi = cp[:, :2].max(axis=0) + 0.02
            _bm = ((cand[:, 0] > _lo[0]) & (cand[:, 0] < _hi[0]) &
                   (cand[:, 1] > _lo[1]) & (cand[:, 1] < _hi[1]))
            if int(_bm.sum()) > 200:
                _col = cand_col[_bm]
                _sat = _col.max(axis=1) - _col.min(axis=1)
                _white = _sat < WHITE_SAT_MAX
                if float(_white.mean()) > 0.30:
                    _p = cand[_bm][_white]
                    if surf is not None:
                        _hz = Z_UP_SIGN * (_p[:, 2] - surface_z(surf[0], _p[:, :2]))
                    else:
                        _hz = Z_UP_SIGN * (_p[:, 2] - table_z)
                    hh_w = float(np.percentile(_hz, 98))
                    if 0.4 * hh <= hh_w <= hh:      # 只允许"往下"修正
                        hh = hh_w
        if cand_col is not None:
            lo = cp[:, :2].min(axis=0) - 0.02
            hi = cp[:, :2].max(axis=0) + 0.02
            bm = ((cand[:, 0] > lo[0]) & (cand[:, 0] < hi[0]) &
                  (cand[:, 1] > lo[1]) & (cand[:, 1] < hi[1]))
            if int(bm.sum()) > 50:
                rec["green"] = float(np.mean(is_green(cand_col[bm])))
                if rec["reject"] is None and rec["green"] >= GREEN_REJECT:
                    rec["reject"] = "绿色占比%.2f(桌垫)" % rec["green"]
        stats.append(rec)

    diag["n_clusters"] = len(stats)
    diag["clusters"] = sorted(stats, key=lambda t: -t["n"])[:8]
    diag["cand"] = cand
    if debug:
        for t in diag["clusters"]:
            print("     簇%d 点%d 高%.3fm 底面%.3fx%.3fm 贴桌%.0f%% 绿%s -> %s" % (
                t["lb"], t["n"], t["height"], t["extent"][0], t["extent"][1],
                t["frac_above"] * 100.0,
                "-" if t["green"] is None else "%.2f" % t["green"],
                t["reject"] or "OK"))

    ok = [t for t in stats if t["reject"] is None]
    pick = None
    if prior_xy is not None:
        # 1) 合格且离上一帧中心很近 → 认它就是同一个物体（手遮挡/单帧噪声时靠这条接力）
        near_ok = [t for t in ok if t["dist_prior"] is not None
                   and t["dist_prior"] <= PRIOR_GATE_M]
        if near_ok:
            pick = max(near_ok, key=lambda t: t["n"])
        elif ok:
            # 2) 有合格的但都不在先验附近 → 物体可能被挪了，按几何走
            pick = max(ok, key=lambda t: t["n"])
        else:
            # 3) 全被否掉，但先验附近有个像样的团块 → 放行（标记 prior_guided）
            near_any = [t for t in stats if t["dist_prior"] is not None
                        and t["dist_prior"] <= PRIOR_GATE_M and t["n"] >= PRIOR_MIN_PTS]
            if near_any:
                pick = max(near_any, key=lambda t: t["n"])
                pick["prior_guided"] = True
    if pick is None and ok:
        pick = max(ok, key=lambda t: t["n"])
    if pick is None:
        diag["reason"] = "找到 %d 个团块，但全被尺寸/颜色判据否掉了（见候选列表）" % len(stats)
        return None
    cluster_pts = downs[labels == pick["lb"]]
    z0 = float(table_z)
    if surf is not None:
        height = float(np.percentile(
            Z_UP_SIGN * (cluster_pts[:, 2] - surface_z(surf[0], cluster_pts[:, :2])), 99))
    else:
        height = float(np.percentile(Z_UP_SIGN * (cluster_pts[:, 2] - z0), 99))

    centroid = cluster_pts.mean(axis=0)
    obb_c, obb_s, obb_R = fit_obb(cluster_pts)

    result = {
        "n_points": pick["n"],
        "table_z_m": z0,
        "z_base_m": z0,
        "height_m": height,
        "centroid_world": centroid.tolist(),
        "obb_center_world": obb_c.tolist(),
        "obb_size_m": obb_s.tolist(),
        "obb_quat_wxyz": mat_to_quat(obb_R).tolist(),
        "obb_R": np.asarray(obb_R).tolist(),
        "green_ratio": pick["green"],
        "frac_above": pick.get("frac_above"),
        "table_z_how": how,
        "table_surface": diag.get("table_surface"),
        "method": "obb",
        "prior_used": prior_xy is not None,
        "prior_guided": bool(pick.get("prior_guided", False)),
        "dist_prior_m": pick.get("dist_prior"),
        "center_world": [float(obb_c[0]), float(obb_c[1]), float(z0 + Z_UP_SIGN * height / 2.0)],
    }

    # 圆柱拟合（优先，对罐体/瓶子更准，还能回半径验真值）
    cir = circle_fit_xy(cluster_pts)
    if cir is not None:
        cx, cy, r, resid = cir
        if r > 0.010 and resid < 0.02:
            result["method"] = "cylinder_fit"
            result["radius_m"] = r
            result["circle_fit_resid_m"] = resid
            result["center_world"] = [cx, cy, z0 + Z_UP_SIGN * height / 2.0]

    result["center_sim"] = to_sim(result["center_world"]).tolist()
    return result, cluster_pts


# ---------------- 可视化 ----------------
def project(intr, T_cw, p):
    p = np.asarray(p, dtype=np.float64)
    pc = T_cw[:3, :3] @ p + T_cw[:3, 3]
    if pc[2] <= 0.05:
        return None
    u = intr.fx * pc[0] / pc[2] + intr.ppx
    v = intr.fy * pc[1] / pc[2] + intr.ppy
    return int(round(u)), int(round(v))


def draw_result(bgr, intr, T_cw, center, size, R, serial):
    uv = project(intr, T_cw, center)
    if uv is None:
        return bgr
    cv2.drawMarker(bgr, uv, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
    # 8 个角点连线
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                off = R @ (np.array([sx, sy, sz]) * size / 2.0)
                corners.append(center + off)
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7),
             (4, 5), (4, 6), (5, 7), (6, 7)]
    pu = [project(intr, T_cw, c) for c in corners]
    for a, b in edges:
        if pu[a] and pu[b]:
            cv2.line(bgr, pu[a], pu[b], (255, 0, 255), 1)
    return bgr


# ---------------- 自检（不碰相机） ----------------
def _synth_scene(with_floor, with_wall):
    """合成：圆柱(r40 h110) + 桌面；可选加地面(z=-0.7)与背景墙，复现"RANSAC 挑错平面"的场景。"""
    calib = np.load(CALIB_FILE)
    truth = np.array([0.02, -0.01, 0.055])
    r, h = 0.04, 0.11
    z0 = truth[2] - h / 2.0
    all_pts = []
    for serial in calib.files:
        cam_pos = calib[serial][:3, 3]
        ang = np.linspace(0, 2 * np.pi, 240, endpoint=False)
        zz = np.linspace(0, h, 60)
        A, Z = np.meshgrid(ang, zz)
        side = np.stack([truth[0] + r * np.cos(A.ravel()),
                         truth[1] + r * np.sin(A.ravel()),
                         z0 + Z.ravel()], axis=1)
        nrm = np.stack([np.cos(A.ravel()), np.sin(A.ravel()), np.zeros(A.size)], axis=1)
        vis = (nrm * (cam_pos[None, :] - side)).sum(axis=1) > 0
        aa = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        rr = np.linspace(0.001, r, 20)
        RR, AA = np.meshgrid(rr, aa)
        top = np.stack([truth[0] + RR.ravel() * np.cos(AA.ravel()),
                        truth[1] + RR.ravel() * np.sin(AA.ravel()),
                        np.full(RR.size, z0 + h)], axis=1)
        all_pts.append(np.vstack([side[vis], top]) if cam_pos[2] > z0 + h else side[vis])
    world = np.vstack(all_pts)

    gx, gy = np.meshgrid(np.linspace(-0.28, 0.28, 90), np.linspace(-0.28, 0.28, 90))
    table = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    parts = [world, table]
    cols = [np.tile([0.9, 0.9, 0.9], (world.shape[0], 1)),
            np.tile([0.2, 0.6, 0.2], (table.shape[0], 1))]
    if with_floor:                      # 地面：整片大平面，比桌面还大
        fx, fy = np.meshgrid(np.linspace(-2.0, 2.0, 160), np.linspace(-2.0, 2.0, 160))
        floor = np.stack([fx.ravel(), fy.ravel(), np.full(fx.size, -0.70)], axis=1)
        parts.append(floor); cols.append(np.tile([0.5, 0.5, 0.5], (floor.shape[0], 1)))
    if with_wall:                       # 背景墙
        wx, wz = np.meshgrid(np.linspace(-2.0, 2.0, 120), np.linspace(-0.7, 1.7, 80))
        wall = np.stack([np.full(wx.size, 1.30), wx.ravel(), wz.ravel()], axis=1)
        parts.append(wall); cols.append(np.tile([0.8, 0.8, 0.8], (wall.shape[0], 1)))
    world = np.vstack(parts)
    world[:, 2] = -world[:, 2]      # 翻到真实世界的 +Z 朝下约定
    truth = truth.copy()
    truth[2] = -truth[2]
    colors = np.vstack(cols)
    return world, colors, np.zeros(world.shape[0], dtype=bool), truth


def selftest():
    """两种场景各跑一次：只有桌面 vs 桌面+地面+背景墙（后者是用户实际遇到的情况）。"""
    ok_all = True
    for name, wf, ww in [("只有桌面", False, False),
                         ("桌面+地面+背景墙", True, True)]:
        world, colors, skin, truth = _synth_scene(wf, ww)
        res = localize_object(world, colors, skin, roi=(-0.3, -0.3, 0.3, 0.3))
        if res is None:
            print("[自检] %-16s 失败：没找到物体" % name); ok_all = False; continue
        r, _ = res
        c = np.array(r["center_world"])
        err = float(np.linalg.norm(c - truth))
        print("[自检] %-16s 恢复 %s 误差 %.1fmm 方法 %s 半径 %.1fmm 高 %.0fmm 点数 %d" % (
            name, np.round(c, 3), err * 1000, r["method"],
            r.get("radius_m", 0) * 1000, r["height_m"] * 1000, r["n_points"]))
        ok_all &= err < 0.01
    print("[自检]", "PASS" if ok_all else "FAIL")
    return ok_all


# ---------------- 点云窗口（Open3D） ----------------
class CloudViz:
    """世界系点云的实时 Open3D 窗口：灰=全部点，绿=落在候选框里的点，品红=检测框，红球=中心。"""

    def __init__(self, title="world point cloud", w=1000, h=760):
        import open3d as o3d
        self.o3d = o3d
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=title, width=w, height=h)
        ro = self.vis.get_render_option()
        ro.background_color = np.array([0.06, 0.06, 0.08])
        ro.point_size = 2.0
        self._first = True
        self._need_view = True
        self.geoms = []
        self.alive = True

    def _clear(self):
        for g in self.geoms:
            try:
                self.vis.remove_geometry(g, reset_bounding_box=False)
            except Exception:
                pass
        self.geoms = []

    def show(self, pts, cols=None, center=None, R=None, size=None, inliers=None):
        o3d = self.o3d
        if not self.alive:
            return
        self._clear()
        geoms = []
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
        if cols is not None and len(cols) == len(pts):
            c = np.asarray(cols, dtype=np.float64)
            if c.max() > 1.5:
                c = c / 255.0
            pc.colors = o3d.utility.Vector3dVector(np.clip(c, 0, 1))
        else:
            pc.paint_uniform_color([0.65, 0.65, 0.68])
        geoms.append(pc)
        if inliers is not None and len(inliers):
            pc2 = o3d.geometry.PointCloud()
            pc2.points = o3d.utility.Vector3dVector(np.asarray(inliers, dtype=np.float64))
            pc2.paint_uniform_color([0.0, 1.0, 0.0])
            geoms.append(pc2)
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(0.08))
        if center is not None:
            c0 = np.asarray(center, dtype=np.float64)
            sph = o3d.geometry.TriangleMesh.create_sphere(0.008)
            sph.translate(c0)
            sph.paint_uniform_color([1.0, 0.0, 0.0])
            geoms.append(sph)
            if R is not None and size is not None:
                box = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
                M = np.eye(4)
                M[:3, :3] = np.asarray(R, dtype=np.float64) * np.asarray(size, dtype=np.float64)[None, :]
                M[:3, 3] = c0 - np.asarray(R, dtype=np.float64) @ (np.asarray(size, dtype=np.float64) * 0.5)
                box.transform(M)
                box.paint_uniform_color([1.0, 0.0, 1.0])
                geoms.append(box)
        for g in geoms:
            self.vis.add_geometry(g, reset_bounding_box=self._first)
        self._first = False
        self.geoms = geoms
        self.alive = bool(self.vis.poll_events())
        if self.alive:
            self.vis.update_renderer()
            if self._need_view:
                # 必须等窗口真正有了宽高再设视角，否则 Open3D 会报
                # "SetViewPoint() failed because window height and width are not set"
                self.set_top_view()
                self._need_view = False

    def set_top_view(self):
        """把世界系 +Z 摆成屏幕上方、从斜上方看桌面（默认视角是沿 -Z 看，桌面会立起来）。"""
        try:
            vc = self.vis.get_view_control()
            vc.set_up([0.0, 0.0, 1.0])
            vc.set_front([-0.25, -0.55, -0.80])
            vc.set_zoom(0.7)
            self.vis.update_renderer()
        except Exception:
            pass

    def pump(self):
        if self.alive and self.vis is not None:
            self.alive = bool(self.vis.poll_events())
            if self.alive:
                self.vis.update_renderer()

    def close(self):
        try:
            if self.vis is not None:
                self.vis.destroy_window()
        except Exception:
            pass
        self.vis = None
        self.alive = False


# ---------------- 主流程 ----------------
def grab_all(cams):
    """所有相机各抓一帧，返回 (耗时ms, 世界点云, 颜色, 肤色, 各视角bgr, 内参, T_cw)。"""
    t0 = time.time()
    pts_all, col_all, skin_all, bgr_list, intr_list, T_cw_list = [], [], [], [], [], []
    for c in cams:
        d, b, ts = c.grab()
        pts, col, skin = c.cloud_world(d, b)
        pts_all.append(pts); col_all.append(col); skin_all.append(skin)
        bgr_list.append(b); intr_list.append(c.intr)
        T_cw_list.append(rigid_inverse(c.T_wc))
    return ((time.time() - t0) * 1000.0, np.vstack(pts_all), np.vstack(col_all),
            np.concatenate(skin_all), bgr_list, intr_list, T_cw_list)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true", help="连续显示")
    ap.add_argument("--out", default=None, help="结果写 JSON 的路径")
    ap.add_argument("--ply", default=None, help="保存融合物体点云的路径")
    ap.add_argument("--roi", nargs=4, type=float, default=[-0.3, -0.3, 0.3, 0.3],
                    help="搜索范围 x0 y0 x1 y1（米）")
    ap.add_argument("--no-skin-remove", action="store_true", help="关闭肤色剔除")
    ap.add_argument("--table-z", type=float, default=None,
                    help="桌面在世界系里的高度（米）。默认 None = 自动估（推荐）；"
                         "标定桌面明显不在 z=0 时才手动给")
    ap.add_argument("--once", action="store_true",
                    help="只测一帧就退出（默认是连续实时追踪）")
    ap.add_argument("--frames", type=int, default=1,
                    help="已弃用（保留兼容）：实时模式每帧都解")
    ap.add_argument("--debug", action="store_true", help="打印每个候选团块的尺寸/颜色占比")
    ap.add_argument("--selftest", action="store_true", help="合成圆柱自检")
    ap.add_argument("--viz", dest="viz", action="store_true", default=True,
                    help="同时弹一个 Open3D 世界点云窗口（默认开）")
    ap.add_argument("--no-viz", dest="viz", action="store_false", help="只显示三视角叠加")
    ap.add_argument("--max-width", type=int, default=1800,
                    help="三视角拼接后的最大显示宽度（默认 1800；屏幕放不下就调小）")
    args = ap.parse_args()
    args.loop = not args.once          # 默认连续

    if args.selftest:
        ok = selftest()
        sys.exit(0 if ok else 1)

    global SKIN_REMOVE
    SKIN_REMOVE = not args.no_skin_remove

    if not os.path.exists(CALIB_FILE):
        print("[错误] 找不到 %s" % CALIB_FILE)
        sys.exit(1)
    calib = np.load(CALIB_FILE)

    ctx = rs.context()
    cams = []
    for dev in ctx.query_devices():
        sn = dev.get_info(rs.camera_info.serial_number)
        if sn in calib:
            cams.append(Cam(sn, calib[sn]))
    if not cams:
        print("[错误] 没有检测到已标定相机")
        sys.exit(1)

    # 每台相机只在自己那块 ROI 像素窗口内反投影（点数少 3~5 倍），窗口只算一次
    for c in cams:
        c.win = roi_window(c.T_wc, c.intr, tuple(args.roi), args.table_z or 0.0)
        print("  [ROI] %s 像素窗口 %s" % (c.serial[-6:], c.win))

    viz = CloudViz() if args.viz else None
    # WINDOW_NORMAL = 可拖动缩放；OpenCV 会把图像按窗口大小等比拉伸，画面自动跟随窗口
    cv2.namedWindow("object center", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("object center", min(args.max_width, 1700), 380)
    print("[提示] 点云窗口里可以拖动旋转；关闭点云窗口不会退出程序（按 q 或 ESC 退出）")
    print("[实时] 默认连续追踪：第一帧全局搜索，之后每帧用上一帧中心做 6cm 增量搜索；"
          "连续 %d 帧丢失则回到全局搜索" % LOST_RESET_FRAMES)

    # ---- 实时追踪状态 ----
    last_center = None      # 上一帧中心 xy（锁定后作为先验）
    smooth = None           # 输出用的平滑中心
    lost = 0                # 连续丢失帧数
    last_h = None           # 上一次的高度：手压在物体顶上时高度会突然翻倍，用它兜住
    frame_no = 0
    fps_hist = []           # 最近若干帧的帧率

    try:
        while True:
            t_frame = time.perf_counter()
            dt_i, world, colors, skin, bgr_list, intr_list, T_cw_list = grab_all(cams)
            locked = last_center is not None
            res = None
            diag = {}
            # 锁定后只试一个门限（快）；未锁定时按 8/30/50mm 阶梯重试
            for ab in ((TABLE_ABOVE_M,) if locked else (TABLE_ABOVE_M, 0.030, 0.050)):
                diag = {}
                res = localize_object(world, colors, skin, tuple(args.roi),
                                      table_z_hint=args.table_z, debug=False,
                                      diag=diag, above_m=ab, prior_xy=last_center,
                                      fast=locked)
                if res is not None:
                    break

            if res is not None:
                r, cluster = res
                c_raw = np.asarray(r["center_world"], dtype=np.float64)
                if smooth is None or not locked:
                    smooth = c_raw.copy()
                else:
                    smooth = np.array([TRACK_ALPHA * c_raw[0] + (1.0 - TRACK_ALPHA) * smooth[0],
                                       TRACK_ALPHA * c_raw[1] + (1.0 - TRACK_ALPHA) * smooth[1],
                                       c_raw[2]])
                # 先验用"本帧原始测量"而不是平滑值：平滑值回灌会形成正反馈，
                # 物体静止时中心会缓慢漂移（实测 30 帧漂 5mm）；平滑只用于显示/输出
                # 手压在物体顶上时（肤色掩膜漏判），高度会突然翻倍、中心随之下沉。
                # 物体是静止的，所以高度只允许"慢慢变"，突变就沿用上一帧的值。
                h_now = float(r["height_m"])
                if locked and last_h is not None and 1.35 * last_h < h_now < 0.30:
                    h_now = last_h
                    r["height_m"] = h_now
                    r["center_world"][2] = float(r["table_z_m"]) + Z_UP_SIGN * h_now * 0.5
                    c_raw = np.asarray(r["center_world"], dtype=np.float64)
                last_h = h_now
                last_center = c_raw[:2].copy()
                lost = 0
                r["center_world"] = smooth.tolist()
                r["center_sim"] = to_sim(smooth).tolist()
                r["track_locked"] = bool(locked)
                c = smooth
                if args.out:
                    json.dump({"timestamp": datetime.now(timezone.utc).isoformat(), "object": r},
                              open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
                if args.ply:
                    import open3d as o3d
                    pc = o3d.geometry.PointCloud()
                    pc.points = o3d.utility.Vector3dVector(cluster)
                    o3d.io.write_point_cloud(args.ply, pc)
                R = np.array(r.get("obb_R", np.eye(3)), dtype=np.float64)
                size = np.array(r["obb_size_m"])
                if r["method"] == "cylinder_fit":
                    size = np.array([r["radius_m"] * 2, r["radius_m"] * 2, r["height_m"]])
                    R = np.eye(3)
                views = []
                for i, (b, intr, Tcw) in enumerate(zip(bgr_list, intr_list, T_cw_list)):
                    v = draw_result(b, intr, Tcw, c, size, R, "cam%d" % (i + 1))
                    cv2.putText(v, "%s  h=%.3f  r=%.1fmm  %s" % (
                        "TRACK" if locked else "DETECT", r["height_m"],
                        r.get("radius_m", 0.0) * 1000.0, r["method"]),
                        (12, v.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    views.append(v)
                comb = np.hstack(views)
                if comb.shape[1] > args.max_width:
                    k = args.max_width / float(comb.shape[1])
                    comb = cv2.resize(comb, (0, 0), fx=k, fy=k, interpolation=cv2.INTER_AREA)
                mfps = sum(fps_hist[-20:]) / max(1, len(fps_hist[-20:]))
                cv2.putText(comb, "%.1f fps" % mfps, (comb.shape[1] - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("object center", comb)
                if viz is not None:
                    viz.show(world, colors, center=c, R=R, size=size, inliers=cluster)
                if frame_no % 10 == 0 or not locked:
                    print("[中心] world=(%.3f, %.3f, %.3f) m | 高%.3f 半径%.1fmm | 点数%d | %s | %.0f ms%s"
                          % (c[0], c[1], c[2], r["height_m"], r.get("radius_m", 0.0) * 1000.0,
                             r["n_points"], "锁定追踪" if locked else "全局检出",
                             (time.perf_counter() - t_frame) * 1000.0,
                             "  (靠上一帧位置接力)" if r.get("prior_guided") else ""))
            else:
                lost += 1
                if lost >= LOST_RESET_FRAMES and last_center is not None:
                    print("[提示] 连续 %d 帧没找到，取消锁定、回到全局搜索" % lost)
                    last_center = None
                    smooth = None
                    last_h = None
                comb = np.hstack(bgr_list)
                if comb.shape[1] > args.max_width:
                    k = args.max_width / float(comb.shape[1])
                    comb = cv2.resize(comb, (0, 0), fx=k, fy=k, interpolation=cv2.INTER_AREA)
                cv2.putText(comb, "LOST (%d)  按 q 退出" % lost, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow("object center", comb)
                if viz is not None:
                    viz.show(world, colors, inliers=diag.get("cand"))
                if lost == 1:
                    print("[无物体] 没找到物体（%.0f ms/帧）原因: %s"
                          % ((time.perf_counter() - t_frame) * 1000.0, diag.get("reason", "?")))
            frame_no += 1
            fps = 1.0 / max(1e-3, time.perf_counter() - t_frame)
            fps_hist.append(fps)

            if not args.loop:
                # 等按键的同时继续给 Open3D 窗口派事件，否则点云窗口会假死、没法转动查看
                while viz is None or viz.alive:
                    k = cv2.waitKey(30) & 0xFF
                    if k != 255:
                        break
                    if viz is not None:
                        viz.pump()
                    if cv2.getWindowProperty("object center", cv2.WND_PROP_VISIBLE) < 1:
                        break
                break
            if viz is not None:
                viz.pump()
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        if viz is not None:
            viz.close()
        for c in cams:
            c.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
