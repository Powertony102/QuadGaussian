#!/usr/bin/env python3
"""
测试模型加载功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_loading():
    """测试模型加载功能"""
    try:
        from fps_test import init_scene
        
        print("测试模型加载...")
        
        # 尝试加载模型
        gaussians, pipeline = init_scene("eval/flowers")
        
        print("✅ 模型加载成功！")
        print(f"高斯模型参数数量: {sum(p.numel() for p in gaussians.parameters())}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    
    if success:
        print("\n🎉 模型加载测试通过！可以运行FPS测试。")
    else:
        print("\n❌ 模型加载测试失败，请检查模型路径和文件。") 