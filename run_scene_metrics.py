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
from tqdm import tqdm

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
                             kernel_times=False, no_kernel=False, suffix="", quiet=False, verbose=False, pbar=None,
                             images="images", resolution=-1):
    """
    运行 compute_scene_metrics.py 对指定模型路径计算指标
    
    Args:
        model_path: 模型路径
        iteration: 迭代次数
        skip_train: 是否跳过训练集
        skip_test: 是否跳过测试集
        kernel_times: 是否计算内核时间
        no_kernel: 是否使用 torch.event 计算 FPS
        suffix: 后缀
        quiet: 是否静默模式
        verbose: 是否详细模式
        images: 图片文件夹名称
        resolution: 分辨率缩放因子
    """
    cmd = [
        "python", "compute_scene_metrics.py",
        "-m", model_path,
        "--iteration", str(iteration),
        "-i", images,
        "-r", str(resolution)
    ]
    
    if skip_train:
        cmd.append("--skip_train")
    if skip_test:
        cmd.append("--skip_test")
    if kernel_times:
        cmd.append("--kernel_times")
    if no_kernel:
        cmd.append("--no-kernel")
    if suffix:
        cmd.extend(["--suffix", suffix])
    # 只有在非详细模式且明确要求静默时才添加 --quiet
    if quiet and not verbose and not pbar:
        cmd.append("--quiet")
    
    if verbose:
        print(f"运行命令: {' '.join(cmd)}")
    
    try:
        if verbose and pbar:
            # 详细模式：实时显示输出并更新进度条
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                     text=True, bufsize=1, universal_newlines=True)
            
            # 实时读取输出并更新进度条
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line:
                    # 解析输出中的进度信息
                    if "Progress" in line:
                        pbar.set_postfix_str(f"进度: {line}", refresh=True)
                    elif "Evaluation" in line:
                        pbar.set_postfix_str(f"评估: {line}", refresh=True)
                    elif "L1" in line and "PSNR" in line:
                        pbar.set_postfix_str(f"完成: {line}", refresh=True)
                    elif "Rendering" in line:
                        pbar.set_postfix_str(f"渲染: {line}", refresh=True)
                    elif "Image Metric Progress" in line:
                        pbar.set_postfix_str(f"图像指标计算: {line}", refresh=True)
                    elif "Kernel Time Progress" in line:
                        pbar.set_postfix_str(f"内核时间计算: {line}", refresh=True)
                    elif "Processing" in line:
                        pbar.set_postfix_str(f"处理中: {line}", refresh=True)
                    elif "[DEBUG]" in line:
                        # 显示调试信息
                        print(f"  {line}")
                    
                    if verbose:
                        print(f"  {line}")
            
            process.wait()
            return process.returncode == 0
        else:
            # 静默模式：只捕获输出
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
    except subprocess.CalledProcessError as e:
        if not quiet:
            print(f"\n❌ 处理模型失败: {model_path}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"错误: {e.stderr}")
            else:
                print(f"命令执行失败: {' '.join(cmd)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="运行 compute_scene_metrics.py 测试所有场景的测试集")
    parser.add_argument("--output_path", default="eval", help="输出路径")
    parser.add_argument("--iteration", type=int, default=30000, help="迭代次数")
    parser.add_argument("--skip_train", action="store_true", default=True, help="跳过训练集（默认启用）")
    parser.add_argument("--skip_test", action="store_true", help="跳过测试集")
    parser.add_argument("--kernel_times", action="store_true", help="计算内核时间")
    parser.add_argument("--no-kernel", action="store_true", help="使用 torch.event 计算 FPS 而不是 kernel_times")
    parser.add_argument("--suffix", type=str, default="", help="后缀")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    parser.add_argument("--verbose", action="store_true", help="显示详细输出")
    parser.add_argument("--show_progress", action="store_true", help="显示实时进度信息")
    parser.add_argument("--scenes", nargs="+", help="指定要处理的场景（可选）")
    parser.add_argument("--scene_types", nargs="+", 
                       choices=["mipnerf360_outdoor", "mipnerf360_indoor", "tanks_and_temples", "deep_blending"],
                       help="指定要处理的场景类型（可选）")
    parser.add_argument("-i", "--images", type=str, default="images", help="图片文件夹名称")
    parser.add_argument("-r", "--resolution", type=int, default=-1, help="分辨率缩放因子")
    
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
    
    print(f"🎯 将处理以下场景的测试集: {target_scenes}")
    print(f"📁 输出路径: {args.output_path}")
    print(f"🔄 迭代次数: {args.iteration}")
    print(f"🖼️  图片文件夹: {args.images}")
    print(f"📏 分辨率缩放: {args.resolution}")
    print(f"⚙️  参数: skip_train={args.skip_train}, skip_test={args.skip_test}, kernel_times={args.kernel_times}, no_kernel={args.no_kernel}")
    
    # 检查输出路径是否存在
    output_path = Path(args.output_path)
    if not output_path.exists():
        print(f"❌ 输出路径不存在: {output_path}")
        print(f"💡 请先运行训练脚本生成模型，或者指定正确的输出路径")
        print(f"💡 例如: python run_scene_metrics.py --output_path /path/to/your/models")
        return
    
    # 统计信息
    total_scenes = len(target_scenes)
    successful_scenes = 0
    failed_scenes = []
    
    print(f"\n🚀 开始处理 {total_scenes} 个场景...")
    
    # 创建进度条
    with tqdm(total=total_scenes, desc="处理场景", unit="场景", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}") as pbar:
        
        for scene in target_scenes:
            model_path = output_path / scene
            
            if not model_path.exists():
                pbar.set_postfix_str(f"跳过 {scene}: 路径不存在", refresh=True)
                failed_scenes.append((scene, "模型路径不存在"))
                pbar.update(1)
                continue
            
            pbar.set_postfix_str(f"处理 {scene}", refresh=True)
            
            success = run_compute_scene_metrics(
                str(model_path),
                iteration=args.iteration,
                skip_train=args.skip_train,
                skip_test=args.skip_test,
                kernel_times=args.kernel_times,
                no_kernel=args.no_kernel,
                suffix=args.suffix,
                quiet=args.quiet or not args.verbose,
                verbose=args.verbose or args.show_progress,
                pbar=pbar if args.show_progress else None,
                images=args.images,
                resolution=args.resolution
            )
            
            if success:
                successful_scenes += 1
                pbar.set_postfix_str(f"✅ {scene} 完成", refresh=True)
            else:
                failed_scenes.append((scene, "处理失败"))
                pbar.set_postfix_str(f"❌ {scene} 失败", refresh=True)
            
            pbar.update(1)
    
    # 最终进度条更新
    pbar.set_postfix_str(f"完成! 成功: {successful_scenes}/{total_scenes}", refresh=True)
    
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