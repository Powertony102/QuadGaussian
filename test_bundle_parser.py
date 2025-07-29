#!/usr/bin/env python3
"""
测试Bundle文件解析功能
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fps_test import parse_bundle_file

def test_bundle_parser():
    """测试Bundle文件解析"""
    print("测试Bundle文件解析功能")
    print("=" * 50)
    
    bundle_file = "viewer/data/TEST.out"
    
    if not os.path.exists(bundle_file):
        print(f"错误: 找不到Bundle文件 {bundle_file}")
        print("请确保Viewer已经生成了TEST.out文件")
        return
    
    try:
        # 解析Bundle文件
        cameras = parse_bundle_file(bundle_file)
        
        print(f"成功解析了 {len(cameras)} 个相机")
        
        # 显示前3个相机的参数
        for i, cam in enumerate(cameras[:3]):
            print(f"\n相机 {i}:")
            print(f"  焦距: {cam['focal']:.6f}")
            print(f"  旋转矩阵:")
            print(f"    {cam['R'][0]}")
            print(f"    {cam['R'][1]}")
            print(f"    {cam['R'][2]}")
            print(f"  平移向量: {cam['T']}")
            
        # 验证旋转矩阵的正交性
        print(f"\n验证旋转矩阵的正交性:")
        for i, cam in enumerate(cameras[:3]):
            R = cam['R']
            RRT = R @ R.T
            det_R = np.linalg.det(R)
            print(f"  相机 {i}: R*R^T ≈ I: {np.allclose(RRT, np.eye(3), atol=1e-6)}, det(R) = {det_R:.6f}")
            
    except Exception as e:
        print(f"解析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bundle_parser() 