#!/usr/bin/env python3
"""
检查模型路径是否正确
"""

import os
import sys

def check_model_path(model_path):
    """检查模型路径是否包含必要的文件"""
    print(f"检查模型路径: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return False
    
    # 检查point_cloud目录
    point_cloud_dir = os.path.join(model_path, "point_cloud")
    if not os.path.exists(point_cloud_dir):
        print(f"❌ 找不到point_cloud目录: {point_cloud_dir}")
        return False
    
    # 查找最新的迭代目录
    iteration_dirs = []
    for item in os.listdir(point_cloud_dir):
        if item.startswith("iteration_"):
            try:
                iter_num = int(item.split("_")[1])
                iteration_dirs.append((iter_num, item))
            except:
                continue
    
    if not iteration_dirs:
        print(f"❌ 在 {point_cloud_dir} 中找不到迭代目录")
        return False
    
    # 按迭代次数排序，取最新的
    iteration_dirs.sort()
    latest_iter, latest_dir = iteration_dirs[-1]
    latest_path = os.path.join(point_cloud_dir, latest_dir)
    
    print(f"✅ 找到最新迭代: {latest_iter}")
    
    # 检查PLY文件
    ply_file = os.path.join(latest_path, "point_cloud.ply")
    if not os.path.exists(ply_file):
        print(f"❌ 找不到PLY文件: {ply_file}")
        return False
    
    print(f"✅ 找到PLY文件: {ply_file}")
    
    # 检查其他可选文件
    cameras_file = os.path.join(model_path, "cameras.json")
    if os.path.exists(cameras_file):
        print(f"✅ 找到cameras.json: {cameras_file}")
    else:
        print(f"⚠️  未找到cameras.json (可选)")
    
    input_ply = os.path.join(model_path, "input.ply")
    if os.path.exists(input_ply):
        print(f"✅ 找到input.ply: {input_ply}")
    else:
        print(f"⚠️  未找到input.ply (可选)")
    
    print(f"✅ 模型路径检查通过！")
    return True

def main():
    """主函数"""
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "eval/flowers"
    
    success = check_model_path(model_path)
    
    if success:
        print("\n🎉 模型路径正确，可以运行FPS测试！")
        print(f"使用命令: python fps_test.py --model {model_path}")
    else:
        print("\n❌ 模型路径有问题，请检查路径是否正确")
        print("正确的模型路径应该包含:")
        print("  - point_cloud/iteration_XXXXX/point_cloud.ply")
        print("  - cameras.json (可选)")
        print("  - input.ply (可选)")

if __name__ == "__main__":
    main() 