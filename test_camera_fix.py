#!/usr/bin/env python3
"""
测试相机矩阵修复
"""

import sys
import os
import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fps_test import parse_lookat_file, create_camera_from_lookat

def test_camera_matrix():
    """测试相机矩阵构建"""
    print("测试相机矩阵修复...")
    
    # 使用第一个相机参数进行测试
    lookat_path = "viewer/data/TEST.lookat"
    
    if not os.path.exists(lookat_path):
        print(f"错误: 找不到文件 {lookat_path}")
        return False
    
    try:
        viewpoints = parse_lookat_file(lookat_path)
        if not viewpoints:
            print("错误: 没有解析到相机参数")
            return False
        
        # 测试第一个相机
        params = viewpoints[0]
        print(f"测试相机参数:")
        print(f"  Origin: {params['origin']}")
        print(f"  Target: {params['target']}")
        print(f"  Up: {params['up']}")
        print(f"  FOVy: {params['fovy']:.6f}")
        
        # 创建相机
        camera = create_camera_from_lookat(params, 800, 600)
        
        print(f"\n相机矩阵信息:")
        print(f"  R shape: {camera.R.shape}")
        print(f"  T shape: {camera.T.shape}")
        print(f"  FoVx: {camera.FoVx:.6f}")
        print(f"  FoVy: {camera.FoVy:.6f}")
        
        # 验证旋转矩阵的正交性
        R = camera.R
        orthogonality_error = np.abs(R @ R.T - np.eye(3)).max()
        print(f"  旋转矩阵正交性误差: {orthogonality_error:.2e}")
        
        # 验证行列式
        det = np.linalg.det(R)
        print(f"  旋转矩阵行列式: {det:.6f} (应该接近1)")
        
        # 验证相机中心
        camera_center = camera.camera_center
        print(f"  相机中心: {camera_center.cpu().numpy()}")
        
        # 验证世界到视图变换矩阵
        world_view_transform = camera.world_view_transform
        print(f"  世界到视图变换矩阵形状: {world_view_transform.shape}")
        
        # 验证投影矩阵
        projection_matrix = camera.projection_matrix
        print(f"  投影矩阵形状: {projection_matrix.shape}")
        
        # 验证完整投影变换矩阵
        full_proj_transform = camera.full_proj_transform
        print(f"  完整投影变换矩阵形状: {full_proj_transform.shape}")
        
        print("\n相机矩阵构建成功！")
        return True
        
    except Exception as e:
        print(f"相机矩阵构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_glm_lookat():
    """与GLM lookAt函数比较"""
    print("\n与GLM lookAt函数比较...")
    
    # 简单的测试用例
    origin = np.array([0.0, 0.0, 5.0])
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    
    params = {
        "origin": tuple(origin),
        "target": tuple(target),
        "up": tuple(up),
        "fovy": 0.7,
        "clip": (0.01, 100.0)
    }
    
    try:
        camera = create_camera_from_lookat(params, 800, 600)
        
        print(f"测试参数:")
        print(f"  Origin: {origin}")
        print(f"  Target: {target}")
        print(f"  Up: {up}")
        
        print(f"生成的旋转矩阵:")
        print(camera.R)
        
        print(f"生成的平移向量:")
        print(camera.T)
        
        # 验证相机朝向
        forward_expected = (target - origin) / np.linalg.norm(target - origin)
        forward_actual = -camera.R[:, 2]  # Z轴方向
        
        print(f"期望的前向向量: {forward_expected}")
        print(f"实际的前向向量: {forward_actual}")
        print(f"前向向量误差: {np.linalg.norm(forward_expected - forward_actual):.2e}")
        
        # 验证相机位置
        camera_pos_expected = origin
        camera_pos_actual = camera.camera_center.cpu().numpy()
        
        print(f"期望的相机位置: {camera_pos_expected}")
        print(f"实际的相机位置: {camera_pos_actual}")
        print(f"相机位置误差: {np.linalg.norm(camera_pos_expected - camera_pos_actual):.2e}")
        
        return True
        
    except Exception as e:
        print(f"GLM比较失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_cameras():
    """测试多个相机"""
    print("\n测试多个相机...")
    
    lookat_path = "viewer/data/TEST.lookat"
    
    if not os.path.exists(lookat_path):
        print(f"错误: 找不到文件 {lookat_path}")
        return False
    
    try:
        viewpoints = parse_lookat_file(lookat_path)
        
        # 测试前5个相机
        test_count = min(5, len(viewpoints))
        success_count = 0
        
        for i in range(test_count):
            params = viewpoints[i]
            try:
                camera = create_camera_from_lookat(params, 800, 600)
                
                # 基本验证
                R = camera.R
                orthogonality_error = np.abs(R @ R.T - np.eye(3)).max()
                det = np.linalg.det(R)
                
                if orthogonality_error < 1e-10 and abs(det - 1.0) < 1e-6:
                    success_count += 1
                    print(f"  相机 {i}: 通过")
                else:
                    print(f"  相机 {i}: 失败 (正交性误差: {orthogonality_error:.2e}, 行列式: {det:.6f})")
                    
            except Exception as e:
                print(f"  相机 {i}: 异常 - {e}")
        
        print(f"成功创建 {success_count}/{test_count} 个相机")
        return success_count == test_count
        
    except Exception as e:
        print(f"多相机测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("相机矩阵修复测试")
    print("=" * 50)
    
    success = True
    
    # 测试1: 基本相机矩阵构建
    success &= test_camera_matrix()
    
    # 测试2: 与GLM lookAt比较
    success &= compare_with_glm_lookat()
    
    # 测试3: 多相机测试
    success &= test_multiple_cameras()
    
    print("\n" + "=" * 50)
    if success:
        print("所有测试通过！相机矩阵修复成功。")
    else:
        print("部分测试失败，需要进一步检查。")
    print("=" * 50) 