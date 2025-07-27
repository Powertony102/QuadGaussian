#!/usr/bin/env python3

import subprocess
import sys
import os

def test_compute_metrics():
    """测试 compute_scene_metrics.py 是否正常工作"""
    
    print("🧪 测试 compute_scene_metrics.py...")
    
    # 测试帮助信息
    try:
        result = subprocess.run(
            ["python", "compute_scene_metrics.py", "--help"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode == 0:
            print("✅ compute_scene_metrics.py 可以正常启动")
        else:
            print("❌ compute_scene_metrics.py 启动失败")
            return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    # 检查CUDA是否可用
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用，设备数量: {torch.cuda.device_count()}")
            print(f"✅ 当前设备: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA 不可用，将使用CPU（会很慢）")
    except ImportError:
        print("❌ PyTorch 未安装")
        return False
    
    # 检查必要的文件是否存在
    required_files = [
        "compute_scene_metrics.py",
        "scene/__init__.py",
        "gaussian_renderer/__init__.py",
        "utils/loss_utils.py"
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} 存在")
        else:
            print(f"❌ {file} 不存在")
            return False
    
    print("🎉 所有测试通过！")
    return True

if __name__ == "__main__":
    test_compute_metrics() 