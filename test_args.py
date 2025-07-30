#!/usr/bin/env python3

import os
import sys
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, get_combined_args

def test_args():
    # 模拟命令行参数
    sys.argv = [
        "test_args.py",
        "-m", "test_model_path",
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
        
    except Exception as e:
        print(f"❌ 解析失败: {e}")
        # 尝试直接解析
        args = parser.parse_args()
        print(f"直接解析 - Images: {args.images}")
        print(f"直接解析 - Resolution: {args.resolution}")

if __name__ == "__main__":
    test_args() 