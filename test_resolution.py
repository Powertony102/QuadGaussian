#!/usr/bin/env python3

import os
import sys
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from gaussian_renderer import GaussianModel

def test_resolution():
    # 模拟命令行参数
    sys.argv = [
        "test_resolution.py",
        "-m", "eval_4K/bonsai",  # 使用您实际的模型路径
        "-i", "images_test",
        "-r", "2"
    ]
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--kernel_times", action="store_true")
    parser.add_argument("--no-kernel", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    try:
        args = get_combined_args(parser)
        print("✅ 成功解析参数")
        print(f"Model path: {args.model_path}")
        print(f"Images: {args.images}")
        print(f"Resolution: {args.resolution}")
        
        # 提取模型参数
        model_args = model.extract(args)
        print(f"Extracted images: {model_args.images}")
        print(f"Extracted resolution: {model_args.resolution}")
        
        # 测试 Scene 初始化
        gaussians = GaussianModel(model_args.sh_degree)
        
        # 根据 resolution 参数动态设置 resolution_scales
        if model_args.resolution in [1, 2, 4, 8]:
            resolution_scales = [1.0 / model_args.resolution]
        else:
            resolution_scales = [1.0]
        
        print(f"Using resolution scales: {resolution_scales} (resolution={model_args.resolution})")
        
        # 尝试初始化 Scene（可能需要实际的模型路径）
        try:
            scene = Scene(model_args, gaussians, load_iteration=30000, shuffle=False, resolution_scales=resolution_scales)
            print("✅ Scene 初始化成功")
            
            # 检查相机分辨率
            test_cameras = scene.getTestCameras()
            if test_cameras:
                first_camera = test_cameras[0]
                print(f"相机分辨率: {first_camera.image_width} x {first_camera.image_height}")
                print(f"原始图像尺寸: {first_camera.original_image.shape}")
            else:
                print("❌ 没有找到测试相机")
                
        except Exception as e:
            print(f"❌ Scene 初始化失败: {e}")
            print("这可能是因为模型路径不存在或格式不正确")
        
    except Exception as e:
        print(f"❌ 解析失败: {e}")

if __name__ == "__main__":
    test_resolution() 