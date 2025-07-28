#!/usr/bin/env python3
"""
详细调试模型加载过程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_model_loading():
    """详细调试模型加载过程"""
    try:
        from fps_test import init_scene
        
        print("=== 开始详细调试模型加载 ===")
        
        # 1. 检查模型路径
        model_path = "eval/flowers"
        print(f"1. 检查模型路径: {model_path}")
        if not os.path.exists(model_path):
            print(f"❌ 模型路径不存在: {model_path}")
            return False
        print(f"✅ 模型路径存在")
        
        # 2. 检查point_cloud目录
        point_cloud_dir = os.path.join(model_path, "point_cloud")
        print(f"2. 检查point_cloud目录: {point_cloud_dir}")
        if not os.path.exists(point_cloud_dir):
            print(f"❌ point_cloud目录不存在")
            return False
        print(f"✅ point_cloud目录存在")
        
        # 3. 查找迭代目录
        print("3. 查找迭代目录...")
        iteration_dirs = []
        for item in os.listdir(point_cloud_dir):
            if item.startswith("iteration_"):
                try:
                    iter_num = int(item.split("_")[1])
                    iteration_dirs.append((iter_num, item))
                except:
                    continue
        
        if not iteration_dirs:
            print("❌ 找不到迭代目录")
            return False
        
        iteration_dirs.sort()
        latest_iter, latest_dir = iteration_dirs[-1]
        print(f"✅ 找到最新迭代: {latest_iter}")
        
        # 4. 检查PLY文件
        ply_path = os.path.join(point_cloud_dir, latest_dir, "point_cloud.ply")
        print(f"4. 检查PLY文件: {ply_path}")
        if not os.path.exists(ply_path):
            print(f"❌ PLY文件不存在")
            return False
        print(f"✅ PLY文件存在")
        
        # 5. 测试参数设置
        print("5. 测试参数设置...")
        from arguments import ModelParams, PipelineParams
        import argparse
        
        parser = argparse.ArgumentParser(description="Debug Test")
        model_params = ModelParams(parser, sentinel=True)
        pipeline_params = PipelineParams(parser)
        
        args = parser.parse_args([])
        args.model_path = model_path
        args.iteration = -1
        args.source_path = model_path
        args.images = "images"
        args.depths = ""
        args.resolution = -1
        args.white_background = False
        args.train_test_exp = False
        args.data_device = "cuda"
        args.eval = False
        args.sh_degree = 3  # 关键：设置sh_degree
        
        model = model_params.extract(args)
        pipeline = pipeline_params.extract(args)
        
        print(f"✅ 参数设置完成")
        print(f"   - model.sh_degree: {model.sh_degree}")
        print(f"   - model.model_path: {model.model_path}")
        
        # 6. 测试GaussianModel初始化
        print("6. 测试GaussianModel初始化...")
        from gaussian_renderer import GaussianModel
        
        gaussians = GaussianModel(model.sh_degree)
        print(f"✅ GaussianModel初始化成功")
        print(f"   - max_sh_degree: {gaussians.max_sh_degree}")
        print(f"   - active_sh_degree: {gaussians.active_sh_degree}")
        
        # 7. 测试PLY文件加载
        print("7. 测试PLY文件加载...")
        use_train_test_exp = False
        gaussians.load_ply(ply_path, use_train_test_exp)
        print(f"✅ PLY文件加载成功")
        
        # 8. 初始化必要属性
        print("8. 初始化必要属性...")
        if not hasattr(gaussians, 'exposure_mapping') or gaussians.exposure_mapping is None:
            gaussians.exposure_mapping = {}
        if not hasattr(gaussians, '_exposure') or gaussians._exposure is None:
            gaussians._exposure = torch.eye(3, 4, device="cuda")[None]
        print(f"✅ 属性初始化完成")
        
        print("\n🎉 所有测试通过！模型加载成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ 调试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_model_loading()
    
    if success:
        print("\n✅ 调试完成，模型加载功能正常！")
    else:
        print("\n❌ 调试失败，请检查错误信息。") 