import cv2
import numpy as np

# 参数设定
input_path = '/home/jovyan/work/gs_07/gaussian-splatting/output/orggs_deepblending/truck/test/ours_30000/gt/00023.png'  # 请将此处替换为你的图片路径
output_path = '/home/jovyan/work/gs_07/picture/gt_truck_processed.png'
crop_x, crop_y = 290, 0
crop_w, crop_h = 200, 100
zoom_factor = 2.5

# 读取图像
image = cv2.imread(input_path)
if image is None:
    raise FileNotFoundError(f'找不到图像: {input_path}')
h, w = image.shape[:2]

# 裁剪区域
patch = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

# 放大区域
zoomed_patch = cv2.resize(patch, (int(crop_w * zoom_factor), int(crop_h * zoom_factor)), interpolation=cv2.INTER_CUBIC)

# 计算嵌入位置（右下角）
zp_h, zp_w = zoomed_patch.shape[:2]
paste_x = w - zp_w - 10  # 留10px边距
paste_y = h - zp_h - 10

# 将缩放后的区域粘贴到原图上（复制原图防止修改原图）
result = image.copy()
result[paste_y:paste_y+zp_h, paste_x:paste_x+zp_w] = zoomed_patch

# 可选：绘制红框标注裁剪区域
cv2.rectangle(result, (crop_x, crop_y), (crop_x+crop_w, crop_y+crop_h), (0, 0, 255), 2)
cv2.rectangle(result, (paste_x, paste_y), (paste_x+zp_w, paste_y+zp_h), (0, 0, 255), 2)

# 保存结果
cv2.imwrite(output_path, result)
print(f'已保存结果图像: {output_path}')
