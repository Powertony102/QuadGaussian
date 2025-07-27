import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
from collections import defaultdict

# 从 full_eval.py 导入场景定义
mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

def get_dataset_name(scene_name):
    """根据场景名称返回数据集名称"""
    if scene_name in mipnerf360_outdoor_scenes:
        return "MipNeRF360-Outdoor"
    elif scene_name in mipnerf360_indoor_scenes:
        return "MipNeRF360-Indoor"
    elif scene_name in tanks_and_temples_scenes:
        return "Tanks&Temples"
    elif scene_name in deep_blending_scenes:
        return "DeepBlending"
    else:
        return "Unknown"

def load_metrics_from_path(base_path, iteration=30000, split="test"):
    """
    从指定路径加载所有场景的指标数据
    
    Args:
        base_path: 基础路径，如 /home/jovyan/work/gs_07/QuadGaussian/eval
        iteration: 迭代次数
        split: 数据集分割（train/test）
    
    Returns:
        dict: 包含所有场景指标数据的字典
    """
    metrics_data = {}
    base_path = Path(base_path)
    
    # 获取所有场景
    all_scenes = []
    all_scenes.extend(mipnerf360_outdoor_scenes)
    all_scenes.extend(mipnerf360_indoor_scenes)
    all_scenes.extend(tanks_and_temples_scenes)
    all_scenes.extend(deep_blending_scenes)
    
    for scene in all_scenes:
        metrics_file = base_path / scene / split / f"ours_{iteration}" / "metrics.csv"
        
        if metrics_file.exists():
            try:
                df = pd.read_csv(metrics_file)
                metrics_data[scene] = df.iloc[0].to_dict()  # 取第一行数据
                print(f"✅ 加载 {scene}: {metrics_file}")
            except Exception as e:
                print(f"❌ 加载 {scene} 失败: {e}")
        else:
            print(f"⚠️  文件不存在: {metrics_file}")
    
    return metrics_data

def calculate_dataset_averages(metrics_data):
    """
    计算每个数据集的平均值
    
    Args:
        metrics_data: 包含所有场景指标数据的字典
    
    Returns:
        dict: 每个数据集的平均值
    """
    dataset_metrics = defaultdict(list)
    
    # 按数据集分组
    for scene, metrics in metrics_data.items():
        dataset_name = get_dataset_name(scene)
        dataset_metrics[dataset_name].append(metrics)
    
    # 计算每个数据集的平均值
    dataset_averages = {}
    for dataset_name, metrics_list in dataset_metrics.items():
        if metrics_list:
            # 转换为DataFrame进行计算
            df = pd.DataFrame(metrics_list)
            averages = df.mean().to_dict()
            dataset_averages[dataset_name] = averages
            print(f"📊 {dataset_name}: {len(metrics_list)} 个场景")
    
    # 计算MipNeRF360总体平均值（Indoor + Outdoor）
    mipnerf360_indoor_metrics = dataset_metrics.get("MipNeRF360-Indoor", [])
    mipnerf360_outdoor_metrics = dataset_metrics.get("MipNeRF360-Outdoor", [])
    
    if mipnerf360_indoor_metrics or mipnerf360_outdoor_metrics:
        all_mipnerf360_metrics = mipnerf360_indoor_metrics + mipnerf360_outdoor_metrics
        if all_mipnerf360_metrics:
            df = pd.DataFrame(all_mipnerf360_metrics)
            averages = df.mean().to_dict()
            dataset_averages["MipNeRF360-All"] = averages
            print(f"📊 MipNeRF360-All: {len(all_mipnerf360_metrics)} 个场景")
    
    return dataset_averages

def print_results(metrics_data, dataset_averages, output_format="table"):
    """
    打印结果
    
    Args:
        metrics_data: 原始指标数据
        dataset_averages: 数据集平均值
        output_format: 输出格式 (table/csv)
    """
    if output_format == "table":
        print("\n" + "="*80)
        print("📈 数据集平均值统计")
        print("="*80)
        
        # 定义数据集显示顺序
        dataset_order = [
            "MipNeRF360-Outdoor",
            "MipNeRF360-Indoor", 
            "MipNeRF360-All",
            "Tanks&Temples",
            "DeepBlending"
        ]
        
        # 按顺序打印每个数据集的平均值
        for dataset_name in dataset_order:
            if dataset_name in dataset_averages:
                averages = dataset_averages[dataset_name]
                print(f"\n🎯 {dataset_name}")
                print("-" * 50)
                print(f"L1 Loss:     {averages.get('l1', 'N/A'):.6f}")
                print(f"PSNR:        {averages.get('psnr', 'N/A'):.2f}")
                print(f"SSIM:        {averages.get('ssim', 'N/A'):.4f}")
                print(f"LPIPS:       {averages.get('lpips', 'N/A'):.4f}")
                print(f"FPS:         {averages.get('fps', 'N/A'):.1f}")
        
        # 打印其他可能的数据集
        for dataset_name, averages in dataset_averages.items():
            if dataset_name not in dataset_order:
                print(f"\n🎯 {dataset_name}")
                print("-" * 50)
                print(f"L1 Loss:     {averages.get('l1', 'N/A'):.6f}")
                print(f"PSNR:        {averages.get('psnr', 'N/A'):.2f}")
                print(f"SSIM:        {averages.get('ssim', 'N/A'):.4f}")
                print(f"LPIPS:       {averages.get('lpips', 'N/A'):.4f}")
                print(f"FPS:         {averages.get('fps', 'N/A'):.1f}")
        
        # 打印所有场景的详细数据
        print("\n" + "="*80)
        print("📋 所有场景详细数据")
        print("="*80)
        
        for scene, metrics in metrics_data.items():
            dataset_name = get_dataset_name(scene)
            print(f"\n{scene} ({dataset_name}):")
            print(f"  L1: {metrics.get('l1', 'N/A'):.6f}, PSNR: {metrics.get('psnr', 'N/A'):.2f}, "
                  f"SSIM: {metrics.get('ssim', 'N/A'):.4f}, LPIPS: {metrics.get('lpips', 'N/A'):.4f}, "
                  f"FPS: {metrics.get('fps', 'N/A'):.1f}")
    
    elif output_format == "csv":
        # 保存为CSV文件
        output_file = "dataset_averages.csv"
        
        # 创建数据集平均值DataFrame，按指定顺序排列
        dataset_order = [
            "MipNeRF360-Outdoor",
            "MipNeRF360-Indoor", 
            "MipNeRF360-All",
            "Tanks&Temples",
            "DeepBlending"
        ]
        
        # 按顺序创建DataFrame
        ordered_data = {}
        for dataset_name in dataset_order:
            if dataset_name in dataset_averages:
                ordered_data[dataset_name] = dataset_averages[dataset_name]
        
        # 添加其他数据集
        for dataset_name, data in dataset_averages.items():
            if dataset_name not in dataset_order:
                ordered_data[dataset_name] = data
        
        avg_df = pd.DataFrame(ordered_data).T
        avg_df.index.name = "Dataset"
        
        # 保存数据集平均值
        avg_df.to_csv(output_file)
        print(f"💾 数据集平均值已保存到: {output_file}")
        
        # 创建所有场景详细数据DataFrame
        detailed_df = pd.DataFrame(metrics_data).T
        detailed_df.index.name = "Scene"
        detailed_df['Dataset'] = [get_dataset_name(scene) for scene in detailed_df.index]
        
        # 重新排列列顺序
        cols = ['Dataset', 'l1', 'psnr', 'ssim', 'lpips', 'fps']
        detailed_df = detailed_df[cols]
        
        # 保存详细数据
        detailed_file = "scene_details.csv"
        detailed_df.to_csv(detailed_file)
        print(f"💾 场景详细数据已保存到: {detailed_file}")

def main():
    parser = argparse.ArgumentParser(description="分析metrics.csv文件，计算数据集平均值")
    parser.add_argument("--base_path", default="eval", help="基础路径")
    parser.add_argument("--iteration", type=int, default=30000, help="迭代次数")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="数据集分割")
    parser.add_argument("--output_format", default="table", choices=["table", "csv"], help="输出格式")
    parser.add_argument("--scenes", nargs="+", help="指定要分析的场景（可选）")
    
    args = parser.parse_args()
    
    print(f"🔍 分析路径: {args.base_path}")
    print(f"🔄 迭代次数: {args.iteration}")
    print(f"📊 数据集分割: {args.split}")
    
    # 加载指标数据
    metrics_data = load_metrics_from_path(args.base_path, args.iteration, args.split)
    
    if not metrics_data:
        print("❌ 没有找到任何指标数据")
        return
    
    print(f"\n✅ 成功加载 {len(metrics_data)} 个场景的数据")
    
    # 如果指定了特定场景，只分析这些场景
    if args.scenes:
        filtered_data = {}
        for scene in args.scenes:
            if scene in metrics_data:
                filtered_data[scene] = metrics_data[scene]
            else:
                print(f"⚠️  场景 {scene} 不在数据中")
        metrics_data = filtered_data
    
    # 计算数据集平均值
    dataset_averages = calculate_dataset_averages(metrics_data)
    
    if not dataset_averages:
        print("❌ 无法计算数据集平均值")
        return
    
    # 打印结果
    print_results(metrics_data, dataset_averages, args.output_format)
    
    # 计算总体平均值
    all_metrics = list(metrics_data.values())
    if all_metrics:
        overall_df = pd.DataFrame(all_metrics)
        overall_averages = overall_df.mean().to_dict()
        
        print(f"\n" + "="*80)
        print("🌍 总体平均值")
        print("="*80)
        print(f"L1 Loss:     {overall_averages.get('l1', 'N/A'):.6f}")
        print(f"PSNR:        {overall_averages.get('psnr', 'N/A'):.2f}")
        print(f"SSIM:        {overall_averages.get('ssim', 'N/A'):.4f}")
        print(f"LPIPS:       {overall_averages.get('lpips', 'N/A'):.4f}")
        print(f"FPS:         {overall_averages.get('fps', 'N/A'):.1f}")

if __name__ == "__main__":
    main() 