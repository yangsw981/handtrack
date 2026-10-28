import cv2
import numpy as np

# 1. 定义 A4 纸 (300 DPI) 的像素尺寸: 2480 x 3508
A4_W, A4_H = 2480, 3508
margin = 200  # 留白边距

# 2. 创建 5x7 的 ChArUco 板 (使用 DICT_4X4_50 字典，识别最远最清晰)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((5, 7), 0.035, 0.026, aruco_dict)

# 3. 渲染图像并保存
board_img = board.generateImage((A4_W - 2 * margin, A4_H - 2 * margin))
a4_canvas = np.ones((A4_H, A4_W), dtype=np.uint8) * 255
a4_canvas[margin:A4_H - margin, margin:A4_W - margin] = board_img

cv2.imwrite("charuco_A4.png", a4_canvas)
print("已生成标定板图片：charuco_A4.png，请用 A4 纸按【实际大小/100%比例】打印！")