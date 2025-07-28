#!/usr/bin/env python3
"""
简单的FPS测试验证脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_simple():
    """简单测试"""
    try:
        from fps_test import test_fps_from_lookat
        
        print("开始简单FPS测试...")
        
        # 只测试前2个视角，每视角5帧
        test_fps_from_lookat(
            lookat_path="viewer/data/TEST.lookat",
            model_path="eval/flowers",
            n_frames=5,
            output_csv="test_simple.csv",
            width=800,
            height=600,
            max_views=2
        )
        
        print("✅ 简单测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple()
    
    if success:
        print("\n🎉 FPS测试工具工作正常！")
    else:
        print("\n❌ FPS测试工具有问题，请检查错误信息。") 