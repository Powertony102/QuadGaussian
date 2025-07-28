#!/usr/bin/env python3
"""
测试lookat文件解析功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fps_test import parse_lookat_file

def test_parser():
    """测试解析器功能"""
    lookat_path = "viewer/images/TEST.lookat"
    
    if not os.path.exists(lookat_path):
        print(f"错误: 找不到文件 {lookat_path}")
        return
    
    print("测试lookat文件解析...")
    
    try:
        viewpoints = parse_lookat_file(lookat_path)
        
        print(f"成功解析 {len(viewpoints)} 个视角")
        
        if viewpoints:
            print("\n前3个视角的示例:")
            for i, vp in enumerate(viewpoints[:3]):
                print(f"View {i}:")
                print(f"  Origin: {vp['origin']}")
                print(f"  Target: {vp['target']}")
                print(f"  Up: {vp['up']}")
                print(f"  FOVy: {vp['fovy']:.6f}")
                print(f"  Clip: {vp['clip']}")
                print()
        
        return True
        
    except Exception as e:
        print(f"解析失败: {e}")
        return False

if __name__ == "__main__":
    test_parser() 