#!/usr/bin/env python3
"""
测试COLMAP格式的相机参数解析
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fps_test import test_fps_from_camera_file

def main():
    """测试COLMAP格式的相机参数解析"""
    print("测试COLMAP格式的相机参数解析")
    print("=" * 50)
    
    # 使用COLMAP格式进行测试
    # 请根据实际的COLMAP文件路径调整
    model_path = "eval/flowers"  # 请根据实际模型路径调整
    cameras_file = os.path.join(model_path, "sparse/0/cameras.txt")
    images_file = os.path.join(model_path, "sparse/0/images.txt")
    
    # 检查COLMAP文件是否存在
    if not os.path.exists(cameras_file) or not os.path.exists(images_file):
        print(f"错误: 找不到COLMAP文件")
        print(f"  相机内参文件: {cameras_file}")
        print(f"  相机外参文件: {images_file}")
        print("请确保模型路径正确且包含COLMAP格式的相机文件")
        return
    
    test_fps_from_camera_file(
        camera_file=None,  # COLMAP格式不需要这个参数
        model_path=model_path,
        n_frames=1,  # 每视角只渲染1帧
        output_csv="fps_test_colmap.csv",
        width=800,
        height=600,
        background_color=[0, 0, 0],
        max_views=1,  # 只测试第1个视角
        save_images=True,  # 保存图片以便对比
        image_save_interval=1,  # 每个视角都保存图片
        output_dir="viewer/test_colmap",
        use_colmap=True,  # 使用COLMAP格式
        cameras_file=cameras_file,
        images_file=images_file
    )
    
    print("\n测试完成！请检查生成的图片是否与Viewer中的视角一致。")

if __name__ == "__main__":
    main() 