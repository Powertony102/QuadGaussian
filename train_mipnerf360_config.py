#!/usr/bin/env python3
"""
可配置的 orggs (原版3dgs) 批量训练脚本
支持自定义路径和训练参数
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path

# mipnerf360 数据集的所有场景
SCENES = [
    "bicycle",
    "bonsai", 
    "counter",
    "flowers",
    "garden",
    "kitchen",
    "room",
    "stump",
    "treehill"
]

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="orggs (原版3dgs) 批量训练脚本")
    
    # 路径参数
    parser.add_argument("--data_path", type=str, 
                       default="/home/jovyan/work/gs_compression/HAC-main/data/mipnerf360",
                       help="mipnerf360 数据集的基础路径")
    parser.add_argument("--output_path", type=str, 
                       default="output/3dgs",
                       help="输出结果的基础路径")
    
    # wandb 参数
    parser.add_argument("--wandb_project", type=str, default="OG-Quad",
                       help="wandb 项目名称")
    parser.add_argument("--disable_wandb", action="store_true",
                       help="禁用 wandb 日志记录")
    
    # 其他参数
    parser.add_argument("--scenes", nargs="+", type=str, default=None,
                       help="指定要训练的场景列表，默认训练所有场景")
    parser.add_argument("--start_from", type=str, default=None,
                       help="从指定场景开始训练（跳过之前的场景）")
    parser.add_argument("--dry_run", action="store_true",
                       help="只打印命令，不实际执行训练")

    # 训练参数
    # Optimizer 
    parser.add_argument("--sparse_adam", action="store_true",
                       help="使用稀疏 Adam 优化器")
    
    return parser.parse_args()

def run_training(scene_name, args):
    """
    运行单个场景的训练
    
    Args:
        scene_name (str): 场景名称
        args: 命令行参数
    """
    print(f"\n{'='*60}")
    print(f"开始训练场景: {scene_name}")
    print(f"{'='*60}")
    
    # 构建路径
    data_path = os.path.join(args.data_path, scene_name)
    output_path = os.path.join(args.output_path, scene_name)
    
    # 检查数据路径是否存在
    if not os.path.exists(data_path):
        print(f"错误: 数据路径不存在: {data_path}")
        return False
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)


    """
    # 构建训练命令
    cmd = [
        "python", "train_dash.py",
        "-s", data_path,
        "-i", args.image_folder, 
        "-m", output_path,
        "--disable_viewer",
        "--eval",
        "--quiet",
        "--densify_mode", args.densify_mode,
        "--resolution_mode", args.resolution_mode, 
        "--densify_until_iter", str(args.densify_until_iter),
    ]
    """

    # 构建训练命令
    cmd = [
        "python", "train.py",
        "-s", data_path,
        "-i", "images", 
        "-m", output_path,
        "--eval",
        "--disable_viewer",
        "--quiet",
    ]
    # 添加稀疏 Adam 优化器参数
    if args.sparse_adam:
        cmd.append("--sparse_adam")
    
    # 添加 wandb 参数
    if not args.disable_wandb:
        cmd.extend([
            "--use_wandb",
            "--wandb_project", args.wandb_project,
            "--wandb_name", scene_name + "_sota"
        ])
    
    print(f"执行命令: {' '.join(cmd)}")
    print(f"数据路径: {data_path}")
    print(f"输出路径: {output_path}")
    
    if args.dry_run:
        print("干运行模式：跳过实际执行")
        return True
    
    try:
        # 执行训练命令
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"场景 {scene_name} 训练完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"场景 {scene_name} 训练失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n用户中断了场景 {scene_name} 的训练")
        return False

def main():
    """主函数"""
    args = parse_args()
    
    # 确定要训练的场景列表
    scenes_to_train = args.scenes if args.scenes else SCENES
    
    # 如果指定了起始场景，跳过之前的场景
    if args.start_from:
        try:
            start_index = scenes_to_train.index(args.start_from)
            scenes_to_train = scenes_to_train[start_index:]
            print(f"从场景 '{args.start_from}' 开始训练")
        except ValueError:
            print(f"错误: 找不到起始场景 '{args.start_from}'")
            sys.exit(1)
    
    print("OrgGS mipnerf360 批量训练脚本")
    print(f"将依次训练以下场景: {', '.join(scenes_to_train)}")
    print(f"数据基础路径: {args.data_path}")
    print(f"输出基础路径: {args.output_path}")
    print(f"Wandb 项目: {args.wandb_project}")
    print(f"Wandb 启用: {not args.disable_wandb}")
    
    if args.dry_run:
        print("模式: 干运行（只打印命令）")
    
    # 检查基础路径
    if not os.path.exists(args.data_path):
        print(f"错误: 数据基础路径不存在: {args.data_path}")
        sys.exit(1)
    
    # 检查 train_org_gs.py 是否存在
    if not os.path.exists("train.py"):
        print("错误: train_org_gs.py 不存在，请确保在正确的目录下运行此脚本")
        sys.exit(1)
    
    # 统计信息
    total_scenes = len(scenes_to_train)
    successful_scenes = 0
    failed_scenes = []
    
    # 依次训练每个场景
    for i, scene in enumerate(scenes_to_train, 1):
        print(f"\n进度: {i}/{total_scenes}")
        
        success = run_training(scene, args)
        if success:
            successful_scenes += 1
        else:
            failed_scenes.append(scene)
    
    # 打印最终统计
    print(f"\n{'='*60}")
    print("训练完成统计:")
    print(f"{'='*60}")
    print(f"总场景数: {total_scenes}")
    print(f"成功训练: {successful_scenes}")
    print(f"失败场景: {len(failed_scenes)}")
    
    if failed_scenes:
        print(f"失败的场景: {', '.join(failed_scenes)}")    
    else:
        print("所有场景训练成功!")
    
    print(f"\n输出目录: {args.output_path}")

if __name__ == "__main__":
    main() 