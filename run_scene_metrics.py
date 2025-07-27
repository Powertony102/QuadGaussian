#!/usr/bin/env python3
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import subprocess
import argparse
from pathlib import Path

# 从 full_eval.py 导入场景定义
mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

def get_all_scenes():
    """获取所有场景列表"""
    all_scenes = []
    all_scenes.extend(mipnerf360_outdoor_scenes)
    all_scenes.extend(mipnerf360_indoor_scenes)
    all_scenes.extend(tanks_and_temples_scenes)
    all_scenes.extend(deep_blending_scenes)
    return all_scenes

def run_compute_scene_metrics(model_path, iteration=30000, skip_train=False, skip_test=False, 
                             kernel_times=False, suffix="", quiet=False):
    """
    运行 compute_scene_metrics.py 对指定模型路径计算指标
    
    Args:
        model_path: 模型路径
        iteration: 迭代次数
        skip_train: 是否跳过训练集
        skip_test: 是否跳过测试集
        kernel_times: 是否计算内核时间
        suffix: 后缀
        quiet: 是否静默模式
    """
    cmd = [
        "python", "compute_scene_metrics.py",
        "-m", model_path,
        "--iteration", str(iteration)
    ]
    
    if skip_train:
        cmd.append("--skip_train")
    if skip_test:
        cmd.append("--skip_test")
    if kernel_times:
        cmd.append("--kernel_times")
    if suffix:
        cmd.extend(["--suffix", suffix])
    if quiet:
        cmd.append("--quiet")
    
    print(f"运行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ 成功处理模型: {model_path}")
        if result.stdout:
            print(f"输出: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 处理模型失败: {model_path}")
        print(f"错误: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="运行 compute_scene_metrics.py 测试所有场景")
    parser.add_argument("--output_path", default="./eval", help="输出路径")
    parser.add_argument("--iteration", type=int, default=30000, help="迭代次数")
    parser.add_argument("--skip_train", action="store_true", help="跳过训练集")
    parser.add_argument("--skip_test", action="store_true", help="跳过测试集")
    parser.add_argument("--kernel_times", action="store_true", help="计算内核时间")
    parser.add_argument("--suffix", type=str, default="", help="后缀")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    parser.add_argument("--scenes", nargs="+", help="指定要处理的场景（可选）")
    parser.add_argument("--scene_types", nargs="+", 
                       choices=["mipnerf360_outdoor", "mipnerf360_indoor", "tanks_and_temples", "deep_blending"],
                       help="指定要处理的场景类型（可选）")
    
    args = parser.parse_args()
    
    # 确定要处理的场景
    if args.scenes:
        target_scenes = args.scenes
    elif args.scene_types:
        target_scenes = []
        if "mipnerf360_outdoor" in args.scene_types:
            target_scenes.extend(mipnerf360_outdoor_scenes)
        if "mipnerf360_indoor" in args.scene_types:
            target_scenes.extend(mipnerf360_indoor_scenes)
        if "tanks_and_temples" in args.scene_types:
            target_scenes.extend(tanks_and_temples_scenes)
        if "deep_blending" in args.scene_types:
            target_scenes.extend(deep_blending_scenes)
    else:
        target_scenes = get_all_scenes()
    
    print(f"🎯 将处理以下场景: {target_scenes}")
    print(f"📁 输出路径: {args.output_path}")
    print(f"🔄 迭代次数: {args.iteration}")
    print(f"⚙️  参数: skip_train={args.skip_train}, skip_test={args.skip_test}, kernel_times={args.kernel_times}")
    
    # 检查输出路径是否存在
    output_path = Path(args.output_path)
    if not output_path.exists():
        print(f"❌ 输出路径不存在: {output_path}")
        return
    
    # 统计信息
    total_scenes = len(target_scenes)
    successful_scenes = 0
    failed_scenes = []
    
    print(f"\n🚀 开始处理 {total_scenes} 个场景...")
    
    for i, scene in enumerate(target_scenes, 1):
        model_path = output_path / scene
        
        if not model_path.exists():
            print(f"⚠️  跳过场景 {scene}: 模型路径不存在 {model_path}")
            failed_scenes.append((scene, "模型路径不存在"))
            continue
        
        print(f"\n📊 [{i}/{total_scenes}] 处理场景: {scene}")
        
        success = run_compute_scene_metrics(
            str(model_path),
            iteration=args.iteration,
            skip_train=args.skip_train,
            skip_test=args.skip_test,
            kernel_times=args.kernel_times,
            suffix=args.suffix,
            quiet=args.quiet
        )
        
        if success:
            successful_scenes += 1
        else:
            failed_scenes.append((scene, "处理失败"))
    
    # 输出统计结果
    print(f"\n📈 处理完成!")
    print(f"✅ 成功: {successful_scenes}/{total_scenes}")
    print(f"❌ 失败: {len(failed_scenes)}/{total_scenes}")
    
    if failed_scenes:
        print(f"\n❌ 失败的场景:")
        for scene, reason in failed_scenes:
            print(f"  - {scene}: {reason}")

if __name__ == "__main__":
    main() 