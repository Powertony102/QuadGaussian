#!/usr/bin/env python3
"""
快速FPS测试脚本
用于验证工具是否能正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fps_test import test_fps_from_lookat

if __name__ == "__main__":
    print("运行快速FPS测试...")
    
    # 使用默认参数进行快速测试
    test_fps_from_lookat(
        lookat_path="viewer/data/TEST.lookat",
        model_path="eval/flowers",  # 请根据实际模型路径修改
        n_frames=10,  # 减少帧数用于快速测试
        output_csv="fps_test_quick.csv",
        width=800,
        height=600,
        max_views=3  # 只测试前3个视角
    ) 