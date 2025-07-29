#!/usr/bin/env python3
"""
测试Bundle格式的相机参数解析
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fps_test import test_fps_from_camera_file

def main():
    """测试Bundle格式的相机参数解析"""
    print("测试Bundle格式的相机参数解析")
    print("=" * 50)
    
    # 使用Bundle格式进行测试
    test_fps_from_camera_file(
        camera_file="viewer/data/TEST.out",  # 使用.out文件
        model_path="eval/flowers",  # 请根据实际模型路径调整
        n_frames=1,  # 每视角只渲染1帧
        output_csv="fps_test_bundle.csv",
        width=800,
        height=600,
        background_color=[0, 0, 0],
        max_views=1,  # 只测试第1个视角
        save_images=True,  # 保存图片以便对比
        image_save_interval=1,  # 每个视角都保存图片
        output_dir="viewer/test_bundle",
        use_bundle=True  # 明确使用Bundle格式
    )
    
    print("\n测试完成！请检查生成的图片是否与Viewer中的视角一致。")

if __name__ == "__main__":
    main() 