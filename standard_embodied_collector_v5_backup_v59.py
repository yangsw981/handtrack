# -*- coding: utf-8 -*-
"""
多视角人手轨迹采集 (v5.0)
=====================================================================
v5 修的是什么（被离线评测逼出来的）
---------------------------------------------------------------------
用 663 帧真实 episode 做留一验证(LOO)：纯 2D 三角化的三维误差中位 7.2mm；
而"2D + 深度反投影"这一路的跨视角三维分歧中位 28.5mm、P95 119mm，抓握段
(34~42s)更是飙到 90~100mm。结论很清楚：
  * 2D 这一路在 1cm 量级，深度这一路在 3cm 量级，而且抓握时会发散；
  * v4 把两者按相近权重塞进同一个最小二乘，等于让噪声大 4 倍的深度去
    拉扯本来挺准的横向位置——骨架整体偏移、骨长 CV 13.5%(最大 40%)
    都是这么来的；
  * 最差关节是腕和四个 MCP(20.8/15.2/14.9/14.0/13.3px)，不是指尖：
    手掌是一大片平坦区域，MediaPipe 的 2D 在掌部本来就不太一致。

v5 的三条改动
---------------------------------------------------------------------
  A. 三角化升为主观测：每帧先对每个关节做多视角三角化(重投影 RMS
     <= 6px 才算自洽)，作为 1cm 量级的高斯先验进入 GN，并用它做初值。
  B. 深度降为"沿视线标量约束"：深度只保留 d·(X - p_depth) 这一维
     ("这个点沿这条视线有多远")，横向完全交给 2D。v4 那种全向 3D 残差
     会把错深度的点横向拽走，这是抓握段偏移的直接原因。
  C. 掌部降权：腕/拇指根/四个 MCP 的深度项再乘 0.15，它们测到的多半
     是掌背或桌面，参考价值最低。

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
import itertools
import hashlib
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from datetime import datetime, timezone

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

# ================= 1. 规范基础配置 (对应 SJ/T 具身智能数据采集规范) =================
CODE_VERSION = "v5.9"          # 启动横幅会打印，用来确认跑的是哪一版
CODE_FILE = "standard_embodied_collector_v5.py"

# ---- 数据字典与精度声明（真值验证结果，完整报告见 _diag/ACCURACY_REPORT.md）----
JOINT_SEMANTICS = (
    "joints_pose 为 21 个关节点中心（腕 1 + 五指 × 4 级）；Tip 关节取指尖几何中心，"
    "比真实接触面缩进约 10mm（指腹厚度），用于接触/抓取约束时需按此偏移补偿；"
    "joints_inferred_mask=1 表示该关节由骨长/上一帧维持（被遮挡），非当帧实测"
)
ACCURACY_STATEMENT = {
    "method": "ChArUco 标定板真值(5x7, 方格 34.4mm) + 留一视角三角化(LOO)",
    "extrinsic_consistency_mm": {"median": 4.0, "max": 8.8,
                                 "note": "同一块板三机各自解出的世界位姿差，含板角检测噪声"},
    "depth_along_ray_bias_mm": {"median": 0.7, "p95": 21.3,
                                "note": "深度只在沿视线方向做标量约束，尾部大误差已被拒权"},
    "pure_2d_triangulation_mm": {"median": 1.1, "p95": 2.2,
                                 "note": "仅用三路 2D + 标定，代表几何链路的能力上限"},
    "hand_loo_3d_mm": {"median": 4.7, "p95": 18.6,
                       "note": "手部管线 LOO，与几何上限的差距来自 MediaPipe 跨视角 2D 标注不一致"},
    "hand_reprojection_px": {"per_cam_median": [6.6, 8.4, 5.5], "p95": 13.3},
    "jitter_mm": {"median": 1.4, "p95": 3.4},
    "bone_length_cv_pct": {"median": 7.5, "max": 24.0},
    "reference_episode": "episode_0001_20261029_192346 (真值) / episode_0001_20261029_185927 (指标)",
}

DATASET_ROOT = "Embodied_Hand_Grasp_Dataset"
CALIB_FILE = "extrinsics_3cams.npz"

DATASET_DECLARATION = {
    "dataset_name": "MultiView_DexHand_Grasp",
    "author_name": "Researcher",
    "author_organization": "Robotics_Lab",
    "author_email": "researcher@university.edu.cn",
    "dataset_license": "CC BY 4.0",
    "dataset_version": "1.4.0"
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
# 三台相机没有硬触发线，自由运行必然存在一个"恒定相位差"(实测中位 30ms)。
# 这个偏差由启动相位和晶振决定、逐帧重复，是硬件属性而不是采集失败；
# 软件真正该管的是相位差【之外】的抖动，所以这里拆成两个量统计。
SYNC_TOLERANCE_MS = 25.0     # 相位差绝对值参考线(仅用于显示，不作为通过判据)
SYNC_JITTER_TOL_MS = 12.0    # (v5.5 起不再使用，保留常量避免旧脚本 import 失败)
SYNC_RESAMPLE_SPAN_MAX_MS = 100.0  # v5.5 判据：有效插值跨度超过它说明某一路缺样本/掉帧
# ---- v5.3 按时间戳配对取帧 ----
# 事故原因：采集循环 13.7fps、相机流 15fps，两者存在 6.1ms/帧的拍频，
# 每 6 帧就会有一路"落后一整帧"，该帧三视角最多相差 63ms，重投影比对齐帧差 19%。
# 改法：每路缓存最近几帧，按时间戳挑极差最小的三元组，而不是谁先到就用谁。
PAIR_BUF_LEN = 3             # 每路缓存帧数(同时设为 RealSense frames_queue_size)
PAIR_MAX_WAIT_MS = 60.0      # 只在"某一路还没出帧"时短暂等待；等不到就用现有最好的组合
PAIR_DEAD_MS = 400.0         # 某路超过这么久没出帧，就把它从"必须参与"里摘掉
# ---- v5.5 时间戳重采样（SJ/T 5.2.2 同步要求）----
PAIR_HIST_LEN = 8            # 每路保留的观测样本数，用于按时间戳插值
PAIR_MAX_AGE_MS = 250.0      # 样本超过这个年龄就不再参与插值（别拿半秒前的点凑）
PAIR_MAX_DEPTH_GAP_M = 0.040  # 插值两端深度差 >4cm 视为跳变（遮挡/漏检），只取更近的一端
PAIR_MAX_UV_GAP_PX = 30.0    # (保留字段，速度判据优先)
# 跳变判据用"速度"而不是"帧间差"：手正常挥动 66ms 也能走 30px，
# 用绝对差会把正常动作误判成跳变；遮挡造成的深度跳变则有很陡的速度。
PAIR_MAX_UV_SPEED_PX_S = 4000.0   # 2D 速度上限(0.6m 处约 2.6m/s，正常手动作到不了)
PAIR_MAX_DEPTH_SPEED_M_S = 0.5    # 深度变化速度上限(0.5=每秒50cm，遮挡边缘才会这么陡)
MAX_LOSS_RATE = 0.001
SAVE_RAW_IMAGES = True

# ---------------- 手部检测 ----------------
MP_MODEL_COMPLEXITY = 1      # 0=lite；实测 300 帧同批数据：留一验证中位 7.70px -> 5.57px(-28%)，
                             # 掌部只改善 6%(见下)；代价 3 路线程池 10.5ms -> 19.9ms/帧，仍在 66.7ms 预算内

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
SIG_BONE_REL = 0.030        # 骨长标准差 = 3% 骨长（v5.1：0.06→0.03，CV 9.6%→5.0%，重投影只差 2%）
SIG_BONE_REL_WARMUP = 0.080 # 骨长模型还没学出来时的"软"尺度，避免用半成品骨长锁死手型
BONE_WARMUP_N = 12          # 骨长模型至少有这么多段才切成硬约束
HUBER_BONE_REL = 0.25       # 骨长残差超过 25% 就降权
W_PREV = 0.2 / 0.05 ** 2    # 上一帧先验权重(σ=5cm，再乘 0.2)
# ---------------- 解算稳定性护栏 (v5.2) ----------------
# 事故复盘：episode_20261029_182545 第 513–537 帧，环指指尖被瓶子完全遮住，
# MediaPipe 把它外推到瓶子边缘，2D 残差把点沿视线推向相机；投影深度 z 趋近 0 时
# 雅可比里 1/z 放大，一步就冲过相机(z 变负)，之后该视角 2D 全被判为无效，
# 点只能靠骨长项慢慢拉回，于是连续 25 帧以 ~1m/帧 的速度飞出去(最远 10m)，
# 而且全程 state 仍标记为"实测"。三道护栏：
MAX_GN_STEP_M = 0.020       # 每次迭代单关节最大位移(信任域)，手一帧最多动 2cm
MIN_PROJ_Z = 0.10           # 投影深度下限：比这更近的行不参与 2D 残差(避免 1/z 爆炸)
BONE_GROSS_REL = 0.25       # 骨长偏离模型 25% 以上判定为"解崩了"，强制拉回并转推断

# ---------------- v5: 观测权重重构 ----------------
# 离线评测结论(663 帧)：纯 2D 三角化 LOO 中位 7.2mm；深度反投影跨视角分歧
# 中位 28.5mm —— 深度这一路比 2D 差约 4 倍，且抓握段会飙到 90mm+。
# 所以 v5 把三角化升为主观测，深度降为"只约束沿视线距离"的弱约束。
SIGMA_TRI = 0.010           # 三角化先验标准差(m)
W_TRI = 1.0                 # 三角化先验权重
TRI_MAX_RMS_PX = 6.0        # 三角化内点判据(px)，超过则该关节本轮不用先验
DEPTH_W = 0.50              # 深度项整体权重
DEPTH_PALM_W = 0.15         # 腕/掌部关节深度额外降权(评测里它们最差)
DEPTH_SIGMA_FLOOR = 0.020   # 深度标准差下限：局部 std 低估了"测到的不是手"的系统误差
PALM_JOINTS = (0, 1, 5, 9, 13, 17)

# ---------------- v5.4 掌部刚性化 ----------------
# 腕 + 5 个掌指关节在解剖上是刚体（掌骨之间不会相对滑动）。
# 之前把它们当 6 个自由点各自去拟合 2D，跨视角的标注分歧就直接变成掌部抖动，
# 这也是评测里 Wrist/Middle_MCP/Index_MCP/Ring_MCP 长期霸占误差榜首的原因。
# 现在改为：维护一个掌部模板点集，每帧用 Kabsch 求一个最优刚体变换，
# 再用它把 6 个点往同一个刚体上拉。
PALM_IDX = np.array(PALM_JOINTS, dtype=np.int64)
PALM_RIGID_SIGMA = 0.006        # 掌部刚性残差 σ(米)：6mm，比骨长(3%)更硬
PALM_RIGID_SIGMA_WARMUP = 0.020 # 模板刚建好时的放宽值
PALM_TEMPLATE_N = 90            # 模板用最近多少帧的中位数
PALM_TEMPLATE_MIN = 5           # 至少攒这么多帧才启用
# 实测结论(episode_0001_20261029_185927, 557 帧, _diag/ab_palm.py)：
#   关掉 → 重投影 5.41/7.45px、掌部 6.03/11.97px、骨长CV 4.74%、掌形CV 6.23%
#   σ=6mm → 5.56/7.49、6.03/12.54、5.24%、6.37%      (全部略差)
#   σ=25mm → 5.41/7.45、6.02/12.03、4.88%、6.27%     (趋近基线，即"没有增益")
# 原因：掌部本来就被 4 条腕-掌骨 + 3 条掌骨横撑钉住了，再加刚体先验只是重复约束，
#       而残余误差是 MediaPipe 对腕/MCP 的跨视角标注不一致，几何先验消不掉它。
# 所以默认关闭，代码保留以便将来换更稳的 2D 源后重测。
USE_PALM_RIGID = False

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
    """
    把观测整理成便于向量化的列表，并统计每个关节有几路观测。

    v5 关键改动：深度观测不再当"3D 点"用，只保留【沿视线方向】的标量。
    同一个 (2D, 深度) 观测里，2D 已经完整约束了横向位置，深度唯一额外
    提供的信息是"这个点沿这条视线有多远"。v4 把深度点做全向残差，等于
    让一条噪声大 4 倍的测量去拉扯横向位置——抓握段骨架被拽偏就是这么来的。
    """
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
                dvec = np.asarray(o.p3d, dtype=np.float64) - o.cam["o"]
                nd = float(np.linalg.norm(dvec))
                if nd < 1e-6:
                    continue
                wd = (0.15 + 0.85 * o.tissue) * DEPTH_W
                if j in PALM_JOINTS:
                    wd *= DEPTH_PALM_W
                dep_obs.append((j, o.p3d, dvec / nd,
                                max(float(o.sigma), DEPTH_SIGMA_FLOOR), wd))
                n_obs[j] += 1
    return uv_obs, dep_obs, n_obs


def _triangulated_priors(uv_obs, cams):
    """
    把每帧所有 2D 观测按关节分组做多视角三角化，得到"2D 的最优融合点"。
    只保留重投影 RMS <= TRI_MAX_RMS_PX 的关节（自洽才值得信）。
    返回 {j: (X, rms_px)}
    """
    by_j = {}
    for (c, j, u, v, w) in uv_obs:
        by_j.setdefault(j, []).append((c, u, v))
    out = {}
    for j, items in by_j.items():
        if len(items) < 2:
            continue
        cs = [cams[c] for (c, u, v) in items]
        uvs = [(u, v) for (c, u, v) in items]
        X, errs = triangulate(cs, uvs)
        if X is None or errs is None:
            continue
        rms = float(np.sqrt(np.mean(np.square(errs))))
        if not np.isfinite(rms) or rms > TRI_MAX_RMS_PX:
            continue
        out[int(j)] = (np.asarray(X, dtype=np.float64), rms)
    return out


def _init_pose(uv_obs, dep_obs, prev, cams, tri=None):
    """
    初值优先级(v5)：三角化点 > 深度点均值 > 上一帧 > 沿视线放到手中心深度 > 手中心。
    三角化放最前，是因为它就是 2D 的最优融合，量级比深度点靠谱得多；
    从它出发，GN 不会先被一条错深度带进沟里。
    """
    X = np.full((21, 3), np.nan)
    if tri:
        for j, (Xt, rms) in tri.items():
            if np.isfinite(Xt).all():
                X[j] = Xt
    for (j, p, dvec, s, w) in dep_obs:
        if np.isfinite(X[j]).all():
            continue
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


def kabsch(src, dst):
    """求刚体变换 (R, t)，使 R@src_i + t ≈ dst_i。无镜像(det=+1)。"""
    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return np.eye(3), cd - cs
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, 1.0 if d == 0 else d])
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs


def solve_pose(obs_by_cam, cams, prev=None, boneL=None, boneX=None,
               iters=None, sig_bone=None, palmP=None, sig_palm=None):
    """
    P1：21 个关节一起做加权最小二乘（整手刚性）。

    代价 = Σ_关节 Huber(三角化点残差)        <-- v5 主观测，1cm 量级
          + Σ_视角 Σ_关节 Huber(2D 重投影残差)
          + Σ_视线 Σ_关节 Huber(沿视线深度残差) <-- v5：只约束距离，不碰横向
          + Σ_骨长 (长度 - 模型骨长)²      <-- 让手保持刚性
          + Σ_掌部 (点 - 刚体模板最近点)²   <-- v5.4：腕+5MCP 作为一个刚体
          + 上一帧弱先验

    掌部刚性采用 ICP 形式：每次迭代先用 Kabsch 求"模板 -> 当前掌部"的最优刚体变换，
    再把 6 个掌点往变换后的模板上拉。这样跨视角的标注分歧由刚性约束吸收，
    而不是直接变成掌部抖动。
    返回 pts(21x3), scores(每个关节的观测路数), state(1实测/2推断/0缺失)
    """
    if iters is None:
        iters = GN_ITERS
    if sig_bone is None:
        sig_bone = SIG_BONE_REL
    if sig_palm is None:
        sig_palm = PALM_RIGID_SIGMA

    uv_obs, dep_obs, n_obs = _collect_obs(obs_by_cam)
    state = np.where(n_obs > 0, 1, 2).astype(np.int32)
    if not uv_obs and not dep_obs:
        return np.full((21, 3), np.nan), n_obs.astype(np.float64), \
               np.zeros(21, dtype=np.int32)

    tri = _triangulated_priors(uv_obs, cams)
    X = _init_pose(uv_obs, dep_obs, prev, cams, tri)

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

        # ---------- 掌部刚体：先求"模板 -> 当前掌部"的最优刚体变换 (ICP) ----------
        palmR = None
        palm_t = None
        if palmP is not None:
            cur = X[PALM_IDX]
            if np.isfinite(cur).all():
                palmR, palm_t = kabsch(palmP, cur)

        # ---------- 2D 重投影 ----------
        for c, items in by_cam.items():
            P = cams[c]["P"]
            idx = np.array([it[1] for it in items], dtype=np.int64)
            uv = np.array([[it[2], it[3]] for it in items])
            w0 = np.array([it[4] for it in items])

            Xs = X[idx]
            hom = np.hstack([Xs, np.ones((len(idx), 1))]) @ P.T     # k,3
            z = hom[:, 2]
            good = z > MIN_PROJ_Z
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

        # ---------- 三角化先验（v5 主观测：2D 的最优融合） ----------
        for j, (Xt, rms) in tri.items():
            r = X[j] - Xt
            e = float(np.linalg.norm(r))
            sig = SIGMA_TRI * (1.0 + rms / 3.0)   # 自洽度差的关节自动降权
            w = W_TRI * float(_huber(np.array([e]), 3.0 * sig)[0]) / (sig ** 2)
            s = j * 3 + ar3
            H[np.ix_(s, s)] += w * I3
            g[s] += w * r

        # ---------- 深度（v5：只约束沿视线的距离，横向交给 2D） ----------
        if dep_obs:
            jj = np.array([d[0] for d in dep_obs], dtype=np.int64)
            pp = np.array([d[1] for d in dep_obs])
            dd = np.array([d[2] for d in dep_obs])      # 单位视线方向 k,3
            ss = np.array([d[3] for d in dep_obs])
            w0 = np.array([d[4] for d in dep_obs])
            r = np.einsum("kj,kj->k", dd, X[jj] - pp)   # 沿视线的标量残差 k
            e = np.abs(r)
            w = w0 * _huber(e, HUBER_DEP) / np.maximum(ss, 0.002) ** 2
            for aa in range(3):
                for bb in range(3):
                    np.add.at(H, (jj * 3 + aa, jj * 3 + bb),
                              w * dd[:, aa] * dd[:, bb])
                np.add.at(g, jj * 3 + aa, w * r * dd[:, aa])

        # ---------- 骨长 ----------
        for (a, b, L) in bones:
            d = X[b] - X[a]
            l = float(np.linalg.norm(d))
            if l < 1e-6:
                continue
            u = d / l
            err = l - L
            ww = float(_huber(np.array([abs(err)]), HUBER_BONE_REL * L)[0]) \
                / (sig_bone * L) ** 2
            ia = a * 3 + ar3
            ib = b * 3 + ar3
            uu = ww * np.outer(u, u)
            H[np.ix_(ia, ia)] += uu
            H[np.ix_(ib, ib)] += uu
            H[np.ix_(ia, ib)] -= uu
            H[np.ix_(ib, ia)] -= uu
            g[ia] += ww * u * err
            g[ib] -= ww * u * err

        # ---------- 掌部刚性（v5.4：腕 + 5 MCP 视作一个刚体） ----------
        if palmR is not None:
            pred = palmP @ palmR.T + palm_t
            for k, j in enumerate(PALM_JOINTS):
                r = X[j] - pred[k]
                e = float(np.linalg.norm(r))
                w = float(_huber(np.array([e]), 3.0 * sig_palm)[0]) / (sig_palm ** 2)
                s = j * 3 + ar3
                H[np.ix_(s, s)] += w * I3
                g[s] += w * r

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
        dX = dx.reshape(21, 3)
        step = float(np.linalg.norm(dX, axis=1).max())
        if step > MAX_GN_STEP_M:                 # 信任域：一步最多走 2cm
            dX *= MAX_GN_STEP_M / step
        X = X + dX
        if float(np.linalg.norm(dX)) < 1e-7:
            break

    # ---------- 兜底：骨长被严重破坏的关节沿父骨方向拉回 ----------
    # 只处理"明显解崩"(>25% 骨长)的情况，正常帧一律不动，避免刚性化吃掉 2D 拟合
    if boneL:
        for j in range(1, 21):
            L = boneL.get(j)
            if not L or L <= 1e-4 or not np.isfinite(X[j]).all():
                continue
            pa = PARENT[j]
            if not np.isfinite(X[pa]).all():
                continue
            d = X[j] - X[pa]
            l = float(np.linalg.norm(d))
            if l < 1e-6 or abs(l - L) <= BONE_GROSS_REL * L:
                continue
            X[j] = X[pa] + d / l * L
            state[j] = 2          # 这一帧是几何硬拉回来的，不算实测

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


class PalmModel(object):
    """
    在线学习"掌部刚体模板"（v5.4）。

    做法：每帧把当前 6 个掌点去掉质心，用 Kabsch 对齐到模板朝向，再入窗取逐点中位数。
    只喂 6 点全部为实测(state==1) 的帧，避免推断点把模板带偏。
    """

    def __init__(self):
        self.sets = []
        self.P = None

    def reset(self):
        self.sets.clear()
        self.P = None

    def ready(self):
        return self.P is not None

    def update(self, pts, state, scores=None):
        if any(int(state[j]) != 1 for j in PALM_JOINTS):
            return
        if scores is not None and any(float(scores[j]) < 2.0 for j in PALM_JOINTS):
            return          # 只有单视角支持的帧不进模板，避免把噪声学成形状
        cur = np.asarray(pts, dtype=np.float64)[PALM_IDX]
        if not np.isfinite(cur).all():
            return
        span = float(np.linalg.norm(cur[2] - cur[0]))    # 腕 -> 食指MCP
        if not (0.05 < span < 0.13):                      # 不合理的手直接丢弃
            return
        cur0 = cur - cur.mean(axis=0)
        if self.P is None:
            self.sets.append(cur0.copy())
            self.P = cur0.copy()
            return
        R, _ = kabsch(cur0, self.P)                        # 当前形状 -> 模板朝向
        self.sets.append(cur0 @ R.T)
        if len(self.sets) > PALM_TEMPLATE_N:
            self.sets.pop(0)
        if len(self.sets) >= PALM_TEMPLATE_MIN:
            self.P = np.median(np.stack(self.sets), axis=0)


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

    # 快路径：窗口里只有一个深度面（绝大多数关节都是这样），不必排序/切簇/打分。
    # 21 个关节 × 3 路相机，这一条能省掉约 40% 的 Python 时间；而 Python 时间在
    # 三个工作线程里是被 GIL 串行的，正是它把主循环拖到 12fps。
    if zs.size >= 8 and float(zs.max() - zs.min()) <= DEPTH_CLUSTER_GAP_M:
        return (float(np.median(zs)),
                max(0.003, min(float(np.std(zs)), 0.02)),
                float(ts.mean()), int(zs.size))

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

    # 链路速率：三机同时跑 848x480@15 的彩色+深度，USB2 会直接掉帧。
    # cam3 一直比另外两路少产帧（412 vs 550），先排除这个可能性。
    try:
        usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
    except Exception:
        usb_type = "unknown"
    try:
        physical_port = dev.get_info(rs.camera_info.physical_port)
    except Exception:
        physical_port = "unknown"
    print(f"    Cam{i + 1} {serial} | USB {usb_type} | 端口 {physical_port}")

    if not usb_type.startswith("3"):
        print(f"[警告] Cam{i + 1} ({serial}) 链路是 USB {usb_type}，不是 3.x —— "
              f"三机同跑会严重掉帧，请换到独立 USB3 口/控制器！")

    for sensor in dev.query_sensors():
        if sensor.supports(rs.option.frames_queue_size):
            try:
                # v5.3: 队列留 3 帧，给"按时间戳配对"留挑选余量（原先是 1，取到即丢）
                sensor.set_option(rs.option.frames_queue_size, PAIR_BUF_LEN)
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
        "usb_link": usb_type,
        "physical_port": physical_port,
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
            "sync_note": "三台相机无硬触发线，自由运行存在随启动相位变化的相位差"
                         "(本回合原始值见 manifest_checksum.json 的 sync_raw_spread_ms_*)；"
                         "采集端按硬件时间戳把各路观测线性插值到同一时刻 t_pair，"
                         "故同一帧内所有传感器数据共享同一时间戳(sync_resampled_time_error_ms=0)，"
                         "残余为插值二阶误差 sync_interp_residual_mm_*；"
                         f"判据：插值跨度 p95 ≤ {SYNC_RESAMPLE_SPAN_MAX_MS}ms"
        },
        "cam_relations": "3台D435i通过ChArUco标定板统一至桌面世界坐标系(+Z垂直桌面向上)",
        "reconstruction_method": "每视角 MediaPipe 2D + 对齐深度 -> 反投影; "
                                 "跨视角: 2D 三角化作为主观测(σ=1cm) + 深度沿视线标量约束 + 骨长刚性约束，"
                                 "21 关节整手高斯-牛顿联合求解; "
                                 "融合前按硬件时间戳把各路观测重采样到公共时刻(SJ/T 5.2.2 同步); "
                                 "被物体遮挡的关节用骨长/上一帧维持并标记(见 joints_inferred_mask)",
        "coordinate_note": "pose_trajectory.joints_pose 与 global_pose 已转换为仿真系(y、z取反)，"
                           "collection_devices.cam_extrinsic_parameters 仍为世界系",
        "joint_semantics": JOINT_SEMANTICS,
        "accuracy_statement": ACCURACY_STATEMENT
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

    # ---- 丢失率（SJ/T 5.2.3 e)）：分母用相机自己产出的帧号，不是名义 FPS ----
    # 名义 FPS 会把"循环跑得慢"误判成"丢帧"；帧号差才是真的断流/丢帧。
    cam_loss = {}
    for c in range(num_cams):
        fns = [f["cam_fn"][c] for f in frames_buffer
               if f.get("cam_fn") and f["cam_fn"][c] >= 0]
        if len(fns) >= 2:
            produced = int(max(fns) - min(fns) + 1)
            f0, t0, f1, t1 = cam_ext[c]
            span_s = (t1 - t0) * 1e-3 if (f1 > f0 and t1 > t0) else 0.0
            dev_fps = ((f1 - f0) / span_s) if span_s > 1e-6 else 0.0
            cam_loss[serials[c]] = {
                "frames_saved": int(len(fns)),
                "frames_produced": produced,
                "device_fps": round(dev_fps, 2),
                "measure_span_s": round(span_s, 3),
                "loss_rate": round(1.0 - len(fns) / max(1, produced), 6)
            }
    loss_rate = max([v["loss_rate"] for v in cam_loss.values()], default=0.0)
    is_loss_qualified = (loss_rate <= MAX_LOSS_RATE)
    nominal_expected = int(total_expected)
    nominal_loss = max(0.0, 1.0 - valid_frames / max(1, nominal_expected))

    # ---- 同步（SJ/T 5.2.2）：重采样后同一帧所有观测共享 t_pair，时间戳差恒为 0 ----
    _span = np.array([f["interp_span_ms"] for f in frames_buffer], dtype=np.float64)
    _raw = np.array([f["raw_spread_ms"] for f in frames_buffer
                     if np.isfinite(f.get("raw_spread_ms", float("nan")))], dtype=np.float64)
    if _span.size:
        span_med = float(np.median(_span))
        span_p95 = float(np.percentile(_span, 95))
        span_max = float(_span.max())
    else:
        span_med = span_p95 = span_max = None
    if _raw.size:
        raw_med = float(np.median(_raw))
        raw_p95 = float(np.percentile(_raw, 95))
    else:
        raw_med = raw_p95 = None

    # 插值残余估计：线性插值在弯曲轨迹上的误差 ≈ (1/8)·|二阶差分|·(span/周期)²
    _res = []
    if len(frames_buffer) >= 3:
        Pw = np.array([f["joints_3d"][0] for f in frames_buffer], dtype=np.float64)
        okw = np.isfinite(Pw).all(axis=1)
        idxs = np.where(okw)[0]
        for k in range(1, len(idxs) - 1):
            i0, i1, i2 = int(idxs[k - 1]), int(idxs[k]), int(idxs[k + 1])
            d2 = float(np.linalg.norm(Pw[i0] - 2.0 * Pw[i1] + Pw[i2])) * 1000.0
            period = (frames_buffer[i2]["rel_ts"] - frames_buffer[i0]["rel_ts"]) * 500.0
            span = float(frames_buffer[i1].get("interp_span_ms") or 0.0)
            if period > 1e-6:
                _res.append(0.125 * d2 * (span / period) ** 2)
    res_med = float(np.median(_res)) if _res else None
    res_p95 = float(np.percentile(_res, 95)) if _res else None

    # 把残余空间误差折算成"时间当量"：eq_ms = 残余(mm) / 腕速(mm/ms)。
    # SJ/T 5.2.2 的 20ms 是时间量，这样才有可比的口径。
    _spd = []
    for k in range(1, len(frames_buffer)):
        dt_s = frames_buffer[k]["rel_ts"] - frames_buffer[k - 1]["rel_ts"]
        a_ = frames_buffer[k]["joints_3d"][0]
        b_ = frames_buffer[k - 1]["joints_3d"][0]
        if dt_s > 1e-6 and np.isfinite(a_).all() and np.isfinite(b_).all():
            _spd.append(float(np.linalg.norm(a_ - b_)) * 1000.0 / (dt_s * 1000.0))
    speed_mm_per_ms = float(np.median(_spd)) if _spd else 0.0
    if res_med is not None and speed_mm_per_ms > 1e-4:
        eq_med = res_med / speed_mm_per_ms
        eq_p95 = (res_p95 / speed_mm_per_ms) if res_p95 is not None else None
    else:
        eq_med = eq_p95 = None
    pass_sync = (eq_med is not None and eq_med <= 20.0)
    sync_viol = int(np.sum(_span > SYNC_RESAMPLE_SPAN_MAX_MS)) if _span.size else 0

    manifest = {
        "episode": ep_name,
        "quality_inspection": {
            "collect_duration_s": round(float(elapsed_s), 3),
            "realized_fps": round(valid_frames / max(1e-6, elapsed_s), 2),
            "total_frames_expected": nominal_expected,
            "total_frames_collected": valid_frames,
            "missing_frames_count": int(max(0, nominal_expected - valid_frames)),
            "data_loss_rate": round(loss_rate, 6),
            "data_loss_rate_method": "camera_frame_number",
            "data_loss_rate_per_camera": cam_loss,
            "pass_loss_rate_0.1_percent": bool(is_loss_qualified),
            "data_loss_rate_vs_nominal_fps": round(nominal_loss, 6),
            "data_loss_rate_vs_nominal_fps_note":
                "仅供参考：按 录制时长×FPS 推算，会把循环变慢误判成丢帧，不作为合规判据",
            "sync_method": "timestamp_linear_resample",
            "sync_resampled_time_error_ms": 0.0,
            "sync_note": "三相机无硬触发，原始相位差见 sync_raw_spread_*；每帧已把各路观测"
                         "按硬件时间戳线性插值到同一时刻 t_pair，故同一帧内所有传感器数据"
                         "共享同一时间戳。残余误差是插值的二阶项，见 sync_interp_residual_mm_*。",
            "sync_phase_offset_ms_median": opt_float(raw_med, 2),
            "sync_phase_offset_ms_p95": opt_float(raw_p95, 2),
            "sync_raw_spread_ms_median": opt_float(raw_med, 2),
            "sync_raw_spread_ms_p95": opt_float(raw_p95, 2),
            "sync_interp_span_ms_median": opt_float(span_med, 2),
            "sync_interp_span_ms_p95": opt_float(span_p95, 2),
            "sync_interp_span_ms_max": opt_float(span_max, 2),
            "sync_interp_residual_mm_median": opt_float(res_med, 3),
            "sync_interp_residual_mm_p95": opt_float(res_p95, 3),
            "sync_equivalent_error_ms_median": opt_float(eq_med, 2),
            "sync_equivalent_error_ms_p95": opt_float(eq_p95, 2),
            "sync_equivalent_speed_mm_per_ms": opt_float(speed_mm_per_ms, 4),
            "sync_span_tolerance_ms": SYNC_RESAMPLE_SPAN_MAX_MS,
            "sync_violation_frames": int(sync_viol),
            "pass_sync_20ms": bool(pass_sync),
            "stale_camera_frames": int(stale_frames),
            "mean_valid_joints_per_frame": round(float(np.mean(joints_pool)), 2) if joints_pool else 0.0,
            "mean_measured_joints_per_frame": round(float(np.mean(measured_pool)), 2) if measured_pool else 0.0,
            "mean_views_per_frame": round(float(np.mean(views_pool)), 2) if views_pool else 0.0,
            "profile_ms_per_frame": (
                {**{k: round(_prof_total[k] / _prof_total["n"], 2)
                    for k in ("wait", "front", "fuse", "render")},
                 "other": round((_prof_total["frame"]
                                 - sum(_prof_total[k] for k in ("wait", "front",
                                                                "fuse", "render")))
                                / _prof_total["n"], 2),
                 "frame": round(_prof_total["frame"] / _prof_total["n"], 2),
                 "front_detail": {k: round(_prof_total.get(k, 0.0) / _prof_total["n"], 2)
                                  for k in ("align", "mp", "core", "depth")}}
                if _prof_total["n"] else None),
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
    print(f">>> [保存完毕] {valid_frames} 帧 / 名义期望 {nominal_expected} 帧 | "
          f"丢失率(按相机帧号) {loss_rate * 100:.3f}% "
          f"(合规 <=0.1%: {'通过' if is_loss_qualified else '未通过'})")
    print(f">>> [同步] 重采样后帧内时间戳差 0ms | 插值跨度 {q['sync_interp_span_ms_median']}ms(中位) "
          f"| 残余估计 {q['sync_interp_residual_mm_median']}mm | 原始相位差 {q['sync_raw_spread_ms_median']}ms")
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
cam_fn = [-1] * num_cams                                # 本帧的相机内部帧号(用于丢失率)
cam_stamp = [-1] * num_cams
cam_has_hand = [False] * num_cams
last_z = [[np.nan] * 21 for _ in range(num_cams)]      # 本视角上次可信深度
last_z_frame = [[-999] * 21 for _ in range(num_cams)]

bone_model = BoneModel()
palm_model = PalmModel()
smoothed_hand_3d = None

is_recording = False
debug_tissue = False
episode_counter = 1
frames_buffer = []
sync_violation_count = 0
stale_frame_count = 0
sync_hist = deque(maxlen=90)          # 最近 90 帧相位差，用于算"抖动"基线
record_start_perf = 0.0
frame_id = 0

window_name = "SJ/T Standard Embodied Data Collector v5 (SPACE/R: record, Q: quit, D: debug)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("\n================== 具身智能标准数据采集系统 %s ==================" % CODE_VERSION)
print("文件: %s | 启动目录: %s" % (CODE_FILE, os.getcwd()))
print("配置: complexity=%d  骨长σ=%.3f  三角化门限=%.1fpx  深度权重=%.2f  步长上限=%.0fmm"
      % (MP_MODEL_COMPLEXITY, SIG_BONE_REL, TRI_MAX_RMS_PX, DEPTH_W, MAX_GN_STEP_M * 1000))
print("取帧: 每路缓冲 %d 帧，按硬件时间戳配对 + 重采样到公共时刻" % PAIR_BUF_LEN)
print("丢失率: 按相机内部帧号统计(不再用 录制时长×FPS)")
print("掌部刚性: %s (实测无增益，默认关闭)" % ("开" if USE_PALM_RIGID else "关"))
print("1. 【重要】请先用鼠标左键点击弹出的相机画面窗口，使其获得键盘焦点！")
print("2. 按 【空格键 (Space)】 或 【r】 键：开始录制当前回合(Episode)，左上角出现红点 ● REC。")
print("3. 抓取完成后，再按一次 【空格键】 或 【r】 键：停止录制并自动生成全套规范数据集。")
print("4. 按 【d】 切换调试显示(红圈 = 该关键点处检测不到手，多半被物体挡住)。")
print("5. 按 【q】 或 【ESC】 键退出系统。")
print("画面上 reproj 为当前融合骨架与本视角绿点的像素残差，正常应 < 4px；")
print("骨架颜色：品红=实测，橙色=推断(被遮挡，靠骨长/上一帧维持)。")
print("精度声明: 几何链路(2D三角化) 1.1mm 中位 / 手部 LOO 4.7mm / 抖动 1.4mm；")
print("          Tip 为指尖中心，比接触面缩进约 10mm（详见 config_metadata 的 accuracy_statement）。")
print("=================================================================\n")

# ---- v5.3 取帧器：按时间戳配对（替代"谁先到就用谁"）----
cam_buf = [deque(maxlen=PAIR_BUF_LEN) for _ in range(num_cams)]
last_used_ts = [-1.0] * num_cams
# 每路：首帧号、首帧时间戳、末帧号、末帧时间戳（算设备真实出帧速率，不受启动相位影响）
cam_ext = [[-1, 0.0, -1, 0.0] for _ in range(num_cams)]
cam_seen_wall = [-1.0] * num_cams       # 每路最后一次收到新帧的墙钟时刻


def _drain_buffers():
    """把三路相机的新帧都收进缓冲"""
    now = time.perf_counter()
    for idx in range(num_cams):
        fs = pipelines[idx].poll_for_frames()
        if fs:
            cam_buf[idx].append((float(fs.get_timestamp()), fs))
            cam_seen_wall[idx] = now


def _best_combo():
    """
    在缓冲里挑一组帧：每路取一帧，且时间戳必须比上次用过的更新。
    优先级 = 参与路数多 > 时间戳极差小 > 时间戳新。
    返回 (相机下标列表, [(ts, frameset), ...]) 或 None。
    """
    avail = []
    for c in range(num_cams):
        lst = [x for x in cam_buf[c] if x[0] > last_used_ts[c] + 1e-6]
        if lst:
            avail.append((c, lst))
    if len(avail) < 2:
        return None
    for r in range(len(avail), 1, -1):
        best = None
        for subset in itertools.combinations(avail, r):
            for combo in itertools.product(*[lst for _, lst in subset]):
                ts = [x[0] for x in combo]
                key = (-(max(ts) - min(ts)), max(ts))
                if best is None or key > best[0]:
                    best = (key, [c for c, _ in subset], list(combo))
        if best is not None:
            return best[1], best[2]
    return None


# ---- v5.5 时间戳重采样 ----
# 为什么不再"取最近帧"：三台相机没有硬触发，同一物理时刻在三路里落在不同的
# 采样时刻上（实测相位差 15–55ms，手在 30cm/s 时 40ms 就是 12mm 的位置差）。
# 这在数据里表现为"同一关节三路的 3D 对不上"，正好是 SJ/T 5.2.2 要管的事。
# 做法：每路保留最近若干个观测样本(2D + 深度标量)，融合前把它们按时间戳
# 线性插值到同一个公共时刻 t_pair，于是"同一帧内所有观测共享同一时间戳"，
# 残余误差从"帧相位差导致的位置差"降为"插值的二阶项"（毫秒级跨度 → 亚毫米）。
cam_hist = [deque(maxlen=PAIR_HIST_LEN) for _ in range(num_cams)]


def _push_sample(c, ts, fn, uv, z, sig, tis):
    """把一路相机的观测样本存入历史（按时间递增）"""
    cam_hist[c].append({"ts": float(ts), "fn": int(fn),
                        "uv": uv, "z": z, "sigma": sig, "tissue": tis})


def _pick_bracket(hist, t_pair):
    """在按时间升序的样本里找包住 t_pair 的相邻两帧"""
    before = None
    after = None
    for s in hist:
        if s["ts"] <= t_pair + 1e-6:
            before = s
        elif after is None:
            after = s
    if before is None:
        before = after if after is not None else (hist[-1] if hist else None)
    if after is None:
        after = before
    return before, after


def _resample_at(t_pair):
    """
    把每路最近两帧的关节观测线性插值到 t_pair。
    返回 (obs_by_cam, uv_res, used_cams, span_ms)。
    """
    obs_by_cam = [[None] * 21 for _ in range(num_cams)]
    uv_res = [[None] * 21 for _ in range(num_cams)]
    used = []
    span_ms = 0.0

    for c in range(num_cams):
        hist = cam_hist[c]
        if not hist:
            continue
        if (t_pair - hist[-1]["ts"]) > PAIR_MAX_AGE_MS:
            continue
        b, a = _pick_bracket(hist, t_pair)
        if b is None:
            continue

        dt = a["ts"] - b["ts"]
        w = float(np.clip((t_pair - b["ts"]) / dt, 0.0, 1.0)) if dt > 1e-6 else 0.0
        # 有效插值跨度：w*(1-w)*interval 的 4 倍（w=0.5 时等于整段区间，w=0/1 时为 0）
        # 这样"t_pair 正好落在某路样本上"的相机不会把跨度算大。
        span_ms = max(span_ms, 4.0 * w * (1.0 - w) * abs(a["ts"] - b["ts"]))

        cam = cams[c]
        got = 0
        for j in range(21):
            zb = b["z"][j]
            za = a["z"][j]
            if zb is None and za is None:
                continue
            ub = b["uv"][j]
            ua = a["uv"][j]

            if zb is None:
                z, u, sg = za, ua, a["sigma"][j]
            elif za is None:
                z, u, sg = zb, ub, b["sigma"][j]
            else:
                _duv = float(np.hypot(ua[0] - ub[0], ua[1] - ub[1]))
                _dz = abs(za - zb)
                _dts = dt * 1e-3                      # 时间戳是毫秒，速度要按秒算
                if _dts > 1e-9:
                    jump = (_duv / _dts > PAIR_MAX_UV_SPEED_PX_S
                            or _dz / _dts > PAIR_MAX_DEPTH_SPEED_M_S)
                else:
                    jump = (_dz > PAIR_MAX_DEPTH_GAP_M)
                if jump:
                    # 两端对不上：多半是中间被遮断/漏检，取离 t_pair 更近的一端并放大不确定度
                    if w < 0.5:
                        z, u = zb, ub
                    else:
                        z, u = za, ua
                    sg = max(b["sigma"][j], a["sigma"][j]) * 2.0
                else:
                    z = zb + w * (za - zb)
                    u = (ub[0] + w * (ua[0] - ub[0]),
                         ub[1] + w * (ua[1] - ub[1]))
                    sg = max(b["sigma"][j], a["sigma"][j])

            obs_by_cam[c][j] = Obs(cam, u, backproject(cam, u[0], u[1], z),
                                   sigma=sg, tissue=b["tissue"][j])
            uv_res[c][j] = u
            got += 1

        if got:
            used.append(c)

    return obs_by_cam, uv_res, used, span_ms


def _pick_view(idx, frameset, ts, fn, gid):
    _tv0 = time.perf_counter()
    """
    单相机的完整前端，跑在线程池里：对齐 -> 取色/深度 -> MediaPipe -> 深度采样 -> 观测。

    为什么连 rs.align 也搬进线程：对齐单路要 5–10ms，三路串行就是主循环 1/5 的预算。
    之前只有 MediaPipe 在线程里，主循环仍被三次 align 串行卡住，跑不到相机帧率，
    结果每帧都要跨 66ms 插值，既丢帧又拖慢。这里把整条前端并行掉，主循环只做融合。
    """
    out = {"idx": idx, "ts": ts, "fn": fn, "ok": False,
           "uv": [None] * 21, "z": [None] * 21, "sigma": [np.inf] * 21, "tissue": [0.0] * 21,
           "has_hand": False, "error": None}
    try:
        aligned = align_objects[idx].process(frameset)
        cf = aligned.get_color_frame()
        df = aligned.get_depth_frame()
        if not cf or not df:
            return out
        color_image = np.asanyarray(cf.get_data()).copy()
        depth_image = np.asanyarray(df.get_data())
        out["rgb"] = color_image.copy()
        out["depth"] = depth_image.copy()
        out["vis"] = color_image

        out["t_align"] = time.perf_counter() - _tv0
        _tv0 = time.perf_counter()
        image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands_trackers[idx].process(image_rgb)
        image_rgb.flags.writeable = True

        out["t_mp"] = time.perf_counter() - _tv0
        _tv0 = time.perf_counter()
        if not results.multi_hand_landmarks:
            for j in range(21):
                last_z[idx][j] = np.nan
                last_z_frame[idx][j] = -999
            out["ok"] = True
            return out

        cam = cams[idx]
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
        out["t_core"] = time.perf_counter() - _tv0
        _tv0 = time.perf_counter()

        # 掩膜整体失效(极端光照/深肤色)时不要用它惩罚所有关节。
        # 判据只用掌心/腕部：它们几乎不可能被物体完全挡住。
        core = [0, 5, 9, 13, 17]
        tf_core = [bool(tissue[int(uvi[j, 1]), int(uvi[j, 0])]) for j in core]
        tissue_ok = int(np.sum(tf_core)) >= 4

        for j in range(21):
            u, v = float(uv[j, 0]), float(uv[j, 1])
            if not (0 <= u < WIDTH and 0 <= v < HEIGHT):
                continue
            out["uv"][j] = (u, v)

            prior = None
            if (gid - last_z_frame[idx][j]) <= DEPTH_PRIOR_MAX_AGE:
                prior = last_z[idx][j]

            res = sample_joint_depth(depth_image, int(uvi[j, 0]), int(uvi[j, 1]),
                                     tissue, prior, hand_zc)
            if res is None:
                continue
            z, sd, tf, npix = res
            t_use = tf if tissue_ok else 1.0
            out["z"][j] = float(z)
            out["sigma"][j] = float(max(sd, 1e-3))
            out["tissue"][j] = float(t_use)

            if t_use >= TISSUE_MIN_FRAC:      # 只有"像手"的采样才更新先验
                last_z[idx][j] = z
                last_z_frame[idx][j] = gid

            if debug_tissue and t_use < TISSUE_MIN_FRAC:
                cv2.circle(color_image, (int(uvi[j, 0]), int(uvi[j, 1])), 5, (0, 0, 255), 2)
            else:
                cv2.circle(color_image, (int(uvi[j, 0]), int(uvi[j, 1])), 4, (0, 255, 0), -1)

        out["t_depth"] = time.perf_counter() - _tv0
        out["has_hand"] = True
        out["ok"] = True
    except Exception as exc:                 # 单路异常不能带崩整个采集
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
    return out


def _choose_t_pair():
    """
    在"每路都能插值"的时间窗里挑一个让最大插值跨度最小的公共时刻。

    v5.5 里取的是 min(各路最新时间戳)，那正好让最慢那一路 w=0、其余各路 w≈0.5，
    跨度统计里 4w(1-w)Δt 取到峰值，等于每次都按"最坏的那一路"插值。
    这里改成在可行域 [max(次新), min(最新)] 上做一维 minimax：候选点只有各路样本
    时间戳和它们的中点，几十个，够用了。
    """
    lo, hi = -np.inf, np.inf
    for c in range(num_cams):
        h = cam_hist[c]
        if not h:
            continue
        hi = min(hi, h[-1]["ts"])                       # 每路都至少有一帧不晚于 hi
        if len(h) >= 2:
            lo = max(lo, h[-2]["ts"])
    if not np.isfinite(hi):
        return 0.0
    if not np.isfinite(lo) or lo > hi - 1e-6:
        lo = hi

    def cost(t):
        worst = 0.0
        for c in range(num_cams):
            b, a = _pick_bracket(cam_hist[c], t)
            if b is None:
                return float("inf")
            dt = a["ts"] - b["ts"]
            if dt <= 1e-6:
                continue
            w = float(np.clip((t - b["ts"]) / dt, 0.0, 1.0))
            worst = max(worst, 4.0 * w * (1.0 - w) * dt)
        return worst

    cand = [lo, hi, 0.5 * (lo + hi)]
    for c in range(num_cams):
        for smp in cam_hist[c]:
            if lo - 1e-6 <= smp["ts"] <= hi + 1e-6:
                cand.append(smp["ts"])
    t_best, best = 0.5 * (lo + hi), cost(0.5 * (lo + hi))
    for t in cand:
        v = cost(t)
        if v < best:
            t_best, best = t, v
    return float(t_best)


# ---- 分阶段计时：只用于回答"78ms 到底花在哪"，不影响数据 ----
_prof = {"wait": 0.0, "front": 0.0, "fuse": 0.0, "render": 0.0, "frame": 0.0, "n": 0}
# 全程累计：每回合落盘时写进 manifest，这样不用再让人从控制台里抄数字
_prof_total = {"wait": 0.0, "front": 0.0, "fuse": 0.0, "render": 0.0, "frame": 0.0, "n": 0}


def _prof_add(key, t0):
    dt = (time.perf_counter() - t0) * 1000.0
    _prof[key] += dt
    _prof_total[key] += dt


def _prof_line():
    n = max(1, _prof["n"])
    seg = ("wait", "front", "fuse", "render")
    parts = " ".join("%s %.0f" % (k, _prof[k] / n) for k in seg)
    other = (_prof["frame"] - sum(_prof[k] for k in seg)) / n
    tot = _prof_total["n"]
    if tot:
        det = " ".join("%s %.0f" % (k, _prof_total.get(k, 0.0) / tot)
                       for k in ("align", "mp", "core", "depth"))
    else:
        det = ""
    return "ms/f: %s other %.0f = %.0f | front 细分: %s" % (
        parts, other, _prof["frame"] / n, det)


try:
    while True:
        _t0 = time.perf_counter()
        _t_frame = _t0
        # ---- 取帧：按时间戳在当前缓冲里挑极差最小的三元组 ----
        # 不按 spread 死等：三机无硬触发时的固有相位差(实测 ~23ms)是等不出来的，
        # 死等只会把帧率拖垮。等待只用于"某一路还没出帧"这一种情况。
        t_wait0 = time.perf_counter()
        picked = None
        while True:
            _drain_buffers()
            comb = _best_combo()
            if comb is not None:
                # v5.8: 只要存在可用组合就走，不再"等慢相机凑齐"。
                # cam3 实测只有 ~10.8fps，而旧逻辑要等它凑满三路、上限 60ms；
                # 每个"缺 cam3"的循环都被这 60ms 空等拖掉，帧率因此卡在 13.7。
                picked = comb
                break
            if (time.perf_counter() - t_wait0) * 1000.0 >= PAIR_MAX_WAIT_MS:
                break
            time.sleep(0.001)

        if picked is None:
            time.sleep(0.002)
            continue
        _prof_add("wait", _t0)

        subset, frames = picked
        raw_framesets = [None] * num_cams
        grab_ts = {}
        domains = set()
        for c, (ts, fs) in zip(subset, frames):
            raw_framesets[c] = fs
            grab_ts[c] = ts
            domains.add(fs.get_frame_timestamp_domain())
            last_used_ts[c] = ts
            cam_fn[c] = int(fs.get_frame_number())
            if cam_ext[c][0] < 0:
                cam_ext[c][0] = cam_fn[c]
                cam_ext[c][1] = ts
            cam_ext[c][2] = cam_fn[c]
            cam_ext[c][3] = ts
            while cam_buf[c] and cam_buf[c][0][0] <= ts + 1e-6:
                cam_buf[c].popleft()

        # ---- 原始相位差（重采样前，仅供参考统计） ----
        sync_known = (len(grab_ts) >= 2 and len(domains) == 1)
        raw_spread_pre_ms = ((max(grab_ts.values()) - min(grab_ts.values()))
                             if sync_known else float("nan"))
        if np.isfinite(raw_spread_pre_ms):
            sync_hist.append(float(raw_spread_pre_ms))

        # ---- 阶段 A：把整条前端(对齐+MediaPipe+深度采样)三路并行丢进线程池 ----
        # 之前只有 MediaPipe 在线程里、rs.align 仍串行，三路各 5–10ms 就把主循环拖到 12fps，
        # 追不上 15fps 的相机，于是一半时间在做跨帧插值。现在对齐也并行。
        pending = []
        for idx in range(num_cams):
            frames = raw_framesets[idx]
            if not frames:
                continue
            pending.append((idx, mp_pool.submit(_pick_view, idx, frames,
                                                grab_ts[idx], cam_fn[idx], frame_id + 1)))

        # 三路都没有新帧：说明循环跑得比相机快，这一轮不占帧号
        if not pending:
            time.sleep(0.002)
            continue

        frame_id += 1

        # ---- 阶段 B：收结果（前端已在线程里算完，这里只做汇总） ----
        for idx, fut in pending:
            try:
                r = fut.result()
            except Exception as exc:
                print(f"[警告] Cam{idx+1} 前端异常: {exc}")
                continue
            if r.get("error"):
                print(f"[警告] Cam{idx+1} 前端异常: {r['error']}")

            if r.get("rgb") is not None:
                latest_raw_rgb[idx] = r["rgb"]
                latest_raw_depth[idx] = r["depth"]
                latest_color_imgs[idx] = r["vis"]

            cam_uv[idx] = r["uv"]
            cam_obs[idx] = [None] * 21
            cam_stamp[idx] = frame_id
            cam_has_hand[idx] = r["has_hand"]
            if r["has_hand"]:
                for j in range(21):
                    if r["z"][j] is None:
                        continue
                    u = r["uv"][j]
                    cam_obs[idx][j] = Obs(cams[idx], u,
                                          backproject(cams[idx], u[0], u[1], r["z"][j]),
                                          sigma=r["sigma"][j], tissue=r["tissue"][j])
            # 没手的帧也入历史：插值时能知道"这一路此刻看不到手"，不会拿旧点硬凑
            for _k in ("t_align", "t_mp", "t_core", "t_depth"):
                if r.get(_k):
                    _prof_total.setdefault(_k[2:], 0.0)
                    _prof_total[_k[2:]] += r[_k] * 1000.0
            _push_sample(idx, r["ts"], r["fn"], r["uv"], r["z"], r["sigma"], r["tissue"])

        # ---- v5.6 时间戳重采样：在"每路都能插值"的窗口里挑最小跨度时刻 ----
        _prof_add("front", _t0)
        _t0 = time.perf_counter()
        t_pair = _choose_t_pair()
        obs_by_cam, uv_res, fresh, interp_span_ms = _resample_at(t_pair)
        if not fresh:                       # 还没有任何可用观测，跳过这一轮
            time.sleep(0.002)
            continue
        has_stale = len(fresh) < num_cams
        raw_spread_ms = raw_spread_pre_ms
        # 同步判据：重采样后同一帧内所有观测共享 t_pair，时间戳差恒为 0；
        # 残余的是"插值跨度"带来的二阶误差，这里记录下来供 manifest 使用。
        sync_err_ms = interp_span_ms

        pts, scores, state = solve_pose(
            obs_by_cam, cams,
            prev=smoothed_hand_3d,
            boneL=bone_model.L if bone_model.L else None,
            boneX=bone_model.Lx if bone_model.Lx else None,
            sig_bone=(SIG_BONE_REL_WARMUP
                      if len(bone_model.L) < BONE_WARMUP_N else SIG_BONE_REL),
            palmP=(palm_model.P if (USE_PALM_RIGID and palm_model.ready()) else None),
            sig_palm=(PALM_RIGID_SIGMA_WARMUP
                      if len(palm_model.sets) < 3 * PALM_TEMPLATE_MIN
                      else PALM_RIGID_SIGMA)
        )

        # ---- 骨长/掌部模板在线更新(只用高置信实测点) ----
        bone_model.update(pts, state, scores)
        palm_model.update(pts, state, scores)

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
        _prof_add("fuse", _t0)
        _t0 = time.perf_counter()
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
                    # 与"重采样到 t_pair 后的 2D"比：这才是同一时刻的偏差
                    if idx in fresh and uv_res[idx][j] is not None:
                        errs.append(float(np.hypot(p[0] - uv_res[idx][j][0],
                                                   p[1] - uv_res[idx][j][1])))
                reproj_uv[idx] = uvj
                reproj_err_px[idx] = float(np.mean(errs)) if errs else None

        # ---- 录制落盘缓存 ----
        if is_recording:
            valid_mask = np.zeros(21, dtype=bool)
            inferred_mask = np.zeros(21, dtype=bool)
            if out_pts is not None:
                valid_mask = np.isfinite(out_pts).all(axis=1)
                inferred_mask = valid_mask & (state == 2)

            if sync_err_ms > SYNC_RESAMPLE_SPAN_MAX_MS:
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
                    "pair_ts_ms": float(t_pair),
                    "interp_span_ms": float(interp_span_ms),
                    "raw_spread_ms": raw_spread_ms,
                    "cam_ts_ms": [grab_ts.get(c, float("nan")) for c in range(num_cams)],
                    "cam_fn": [cam_fn[c] if c in fresh else -1 for c in range(num_cams)],
                    "sync_err_ms": sync_err_ms,
                    "n_views": int(sum(1 for c in fresh if cam_has_hand[c])),
                    "valid_mask": valid_mask.copy(),
                    "inferred_mask": inferred_mask.copy(),
                    "state": state.copy(),
                    "joints_3d": sim_pts,
                    "reproj_px": float(np.mean(errs_all)) if errs_all else None,
                    "stale": [cam_stamp[c] != frame_id for c in range(num_cams)],
                    "rgb_list": list(latest_raw_rgb) if SAVE_RAW_IMAGES else [],
                    "depth_list": list(latest_raw_depth) if SAVE_RAW_IMAGES else []
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

            # v5.5：显示"原始相位差"与"插值跨度"两个量；插值跨度才是本帧同步质量的判据
            raw_txt = "%.1fms" % raw_spread_pre_ms if np.isfinite(raw_spread_pre_ms) else "n/a"
            span_ok = (not np.isfinite(interp_span_ms)) or interp_span_ms <= SYNC_RESAMPLE_SPAN_MAX_MS
            sync_txt = "raw:%s span:%.1fms" % (raw_txt, interp_span_ms)
            sync_color = (0, 255, 0) if span_ok else (0, 0, 255)
            cv2.putText(vis, f"Cam {idx + 1} | Sync {sync_txt}",
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

            if idx == 0:
                cv2.putText(vis, _prof_line(), (15, HEIGHT - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

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
            scale = 1600.0 / combined.shape[1]
            combined = cv2.resize(combined, (0, 0), fx=scale, fy=scale,
                                  interpolation=cv2.INTER_NEAREST)
        cv2.imshow(window_name, combined)
        _prof_add("render", _t0)
        _t0 = time.perf_counter()
        _prof["n"] += 1
        _prof_total["n"] += 1
        if _prof["n"] >= 120:
            print("[帧率剖析] " + _prof_line(), flush=True)
            for _k in ("wait", "front", "fuse", "render"):
                _prof[_k] = 0.0
            _prof["n"] = 0

        key = cv2.waitKey(1) & 0xFF
        _prof["frame"] += (time.perf_counter() - _t_frame) * 1000.0
        _prof_total["frame"] += (time.perf_counter() - _t_frame) * 1000.0
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