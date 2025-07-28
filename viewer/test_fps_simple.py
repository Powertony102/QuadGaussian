#!/usr/bin/env python3
"""
简化的3DGS FPS测试脚本
用于快速验证功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fps_test import test_fps_from_lookat

if __name__ == "__main__":
    # 快速测试：只测试前10个视角，每视角渲染50帧
    print("运行简化FPS测试...")
    
    test_fps_from_lookat(
        lookat_path="viewer/images/TEST.lookat",
        model_path="output",  # 请根据实际模型路径修改
        n_frames=50,
        output_csv="fps_test_simple.csv",
        width=800,
        height=600,
        max_views=10  # 只测试前10个视角
    ) 