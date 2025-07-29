#!/usr/bin/env python3
"""
快速FPS测试 - 验证相机矩阵修复
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fps_test import test_fps_from_lookat

def main():
    """快速测试修复后的FPS功能"""
    print("快速FPS测试 - 验证相机矩阵修复")
    print("=" * 50)
    
    # 使用较小的参数进行快速测试
    test_fps_from_lookat(
        lookat_path="viewer/data/TEST.lookat",
        model_path="eval/flowers",  # 请根据实际模型路径调整
        n_frames=10,  # 每视角只渲染10帧
        output_csv="fps_test_fix.csv",
        width=800,
        height=600,
        background_color=[0, 0, 0],
        max_views=3,  # 只测试前3个视角
        save_images=True,  # 保存图片以便对比
        image_save_interval=1,  # 每个视角都保存图片
        output_dir="viewer/test_fix"
    )

if __name__ == "__main__":
    main() 