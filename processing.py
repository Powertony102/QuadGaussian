import cv2
import numpy as np

# 路径设置
input_path = '/home/jovyan/work/gs_07/gaussian-splatting/output/orggs/treehill/test/ours_30000/gt/00010.png'  # 请将此处替换为你的图片路径
output_path = '/home/jovyan/work/gs_07/gaussian-splatting/output/orggs/treehill/test/ours_30000/gt/00010_process.png'

# 读取彩色图像
image = cv2.imread(input_path, cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError(f'未找到文件: {input_path}')

# 检查图像尺寸
if image.shape[:2] != (1036, 1600):
    print(f'警告：图像尺寸为 {image.shape[1]}x{image.shape[0]}，不是1600x1036')

# 转换为 HSV 颜色空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

# 反转亮度（V 通道），保留色调（H）和饱和度（S）
h, s, v = cv2.split(hsv)
v = 255 - v  # 负片亮度

# 可选：添加 gamma 压缩提升暗部细节
gamma = 0.8
v = np.power(v / 255.0, gamma) * 255.0
v = np.clip(v, 0, 255)

# 合并回 HSV
fake_neg_hsv = cv2.merge([h, s, v]).astype(np.uint8)

# 转换回 BGR 空间
fake_negative_bgr = cv2.cvtColor(fake_neg_hsv, cv2.COLOR_HSV2BGR)

# 保存结果
cv2.imwrite(output_path, fake_negative_bgr)
print(f'感知友好负片图像已保存为 {output_path}')
