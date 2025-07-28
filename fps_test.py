#!/usr/bin/env python3
"""
3DGS FPS测试工具
基于相机轨迹文件进行渲染性能测试
"""

import time
import re
import argparse
import csv
import os
import sys
import numpy as np
import torch
from typing import List, Dict, Optional
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 延迟导入，避免循环依赖
import utils.graphics_utils as graphics_utils
import utils.general_utils as general_utils
import utils.system_utils as system_utils


def parse_lookat_file(path: str) -> List[Dict]:
    """
    解析TEST.lookat文件，提取相机参数
    
    Args:
        path: lookat文件路径
        
    Returns:
        包含相机参数的字典列表
    """
    viewpoints = []
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # 解析格式: CamXXX -D origin=X,Y,Z -D target=X,Y,Z -D up=X,Y,Z -D fovy=θ -D clip=near,far
            pattern = r'Cam\d+\s+-D\s+origin=([^,\s]+),([^,\s]+),([^,\s]+)\s+-D\s+target=([^,\s]+),([^,\s]+),([^,\s]+)\s+-D\s+up=([^,\s]+),([^,\s]+),([^,\s]+)\s+-D\s+fovy=([^,\s]+)\s+-D\s+clip=([^,\s]+),([^,\s]+)'
            
            match = re.match(pattern, line)
            if match:
                origin = tuple(float(x) for x in match.groups()[:3])
                target = tuple(float(x) for x in match.groups()[3:6])
                up = tuple(float(x) for x in match.groups()[6:9])
                fovy = float(match.groups()[9])
                clip = (float(match.groups()[10]), float(match.groups()[11]))
                
                viewpoints.append({
                    "origin": origin,
                    "target": target,
                    "up": up,
                    "fovy": fovy,
                    "clip": clip
                })
    
    print(f"解析了 {len(viewpoints)} 个相机视角")
    return viewpoints


def create_camera_from_lookat(params: Dict, width: int = 800, height: int = 600):
    """
    从lookat参数创建Camera对象
    
    Args:
        params: 包含origin, target, up, fovy, clip的字典
        width: 图像宽度
        height: 图像高度
        
    Returns:
        Camera对象
    """
    # 延迟导入，避免循环依赖
    from scene.cameras import Camera
    
    origin = np.array(params["origin"])
    target = np.array(params["target"])
    up = np.array(params["up"])
    
    # 计算相机坐标系
    forward = target - origin
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # 构建旋转矩阵
    R = np.eye(3)
    R[:, 0] = right
    R[:, 1] = -up  # 注意Y轴方向
    R[:, 2] = forward
    
    # 平移向量
    T = -R @ origin
    
    # 计算FOV
    fovy = params["fovy"]
    fovx = fovy * width / height
    
    # 创建虚拟图像（用于Camera构造函数）
    dummy_image = np.zeros((height, width, 3), dtype=np.uint8)
    from PIL import Image
    pil_image = Image.fromarray(dummy_image)
    
    # 创建Camera对象
    camera = Camera(
        resolution=(width, height),
        colmap_id=0,
        R=R,
        T=T,
        FoVx=fovx,
        FoVy=fovy,
        depth_params=None,
        image=pil_image,
        invdepthmap=None,
        image_name="fps_test",
        uid=0,
        data_device="cuda"
    )
    
    return camera


def init_scene(model_path: str, iteration: int = -1):
    """
    初始化场景和渲染器
    
    Args:
        model_path: 模型路径
        iteration: 模型迭代次数
        
    Returns:
        (gaussian_model, pipeline_params)
    """
    # 延迟导入，避免循环依赖
    from arguments import ModelParams, PipelineParams, get_combined_args
    from gaussian_renderer import GaussianModel
    from utils.system_utils import searchForMaxIteration
    import torch
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="FPS Test")
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iteration", default=iteration, type=int)
    
    # 设置默认参数
    args = parser.parse_args([])
    args.model_path = model_path
    args.source_path = model_path
    args.images = "images"
    args.depths = ""
    args.resolution = -1
    args.white_background = False
    args.train_test_exp = False
    args.data_device = "cuda"
    args.eval = False
    args.sh_degree = 3  # 设置sh_degree参数
    
    # 提取参数
    model = model_params.extract(args)
    pipeline = pipeline_params.extract(args)
    
    # 初始化高斯模型
    gaussians = GaussianModel(model.sh_degree)
    
    # 直接加载训练好的模型，跳过场景数据加载
    if iteration == -1:
        loaded_iter = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
    else:
        loaded_iter = iteration
    
    if loaded_iter is None:
        raise ValueError(f"在 {model_path}/point_cloud 中找不到训练好的模型")
    
    print(f"加载训练好的模型，迭代次数: {loaded_iter}")
    
    # 加载PLY文件
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{loaded_iter}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"找不到模型文件: {ply_path}")
    
    # 确保train_test_exp是布尔值
    use_train_test_exp = bool(model.train_test_exp) if model.train_test_exp is not None else False
    gaussians.load_ply(ply_path, use_train_test_exp)
    
    # 初始化必要的属性，避免在渲染时出错
    if not hasattr(gaussians, 'exposure_mapping') or gaussians.exposure_mapping is None:
        gaussians.exposure_mapping = {}
    if not hasattr(gaussians, '_exposure') or gaussians._exposure is None:
        # 创建一个默认的exposure矩阵
        gaussians._exposure = torch.eye(3, 4, device="cuda")[None]
    
    print(f"场景初始化完成，模型路径: {model_path}")
    print(f"加载迭代次数: {loaded_iter}")
    
    return gaussians, pipeline


def render_and_time(camera, gaussians, pipeline, 
                   n_frames: int = 100, background_color: List[float] = [0, 0, 0]) -> float:
    """
    渲染并计时
    
    Args:
        camera: 相机对象
        gaussians: 高斯模型
        pipeline: 渲染管线参数
        n_frames: 渲染帧数
        background_color: 背景颜色
        
    Returns:
        FPS值
    """
    # 延迟导入，避免循环依赖
    from gaussian_renderer import render
    
    background = torch.tensor(background_color, dtype=torch.float32, device="cuda")
    
    # 预热
    with torch.no_grad():
        for _ in range(5):
            render(camera, gaussians, pipeline, background)
    
    # 同步GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 开始计时
    start_time = time.time()
    
    with torch.no_grad():
        # 为单帧渲染添加进度条（当帧数较多时）
        if n_frames > 10:
            for _ in tqdm(range(n_frames), desc="渲染帧", leave=False, unit="帧"):
                render(camera, gaussians, pipeline, background)
        else:
            for _ in range(n_frames):
                render(camera, gaussians, pipeline, background)
    
    # 同步GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    fps = n_frames / elapsed_time
    
    return fps


def report_results(results: List[tuple], output_csv: Optional[str] = None):
    """
    报告测试结果
    
    Args:
        results: 结果列表，每个元素为(view_id, fps)
        output_csv: 可选的CSV输出文件路径
    """
    if not results:
        print("没有测试结果")
        return
    
    fps_values = [r[1] for r in results]
    
    print("\n" + "="*50)
    print("FPS测试结果")
    print("="*50)
    
    for view_id, fps in results:
        print(f"View {view_id:03d}: {fps:.2f} FPS")
    
    print("-"*50)
    print(f"平均FPS: {np.mean(fps_values):.2f}")
    print(f"最小FPS: {np.min(fps_values):.2f}")
    print(f"最大FPS: {np.max(fps_values):.2f}")
    print(f"标准差:  {np.std(fps_values):.2f}")
    print("="*50)
    
    # 保存CSV
    if output_csv:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["view_id", "fps"])
            writer.writerows(results)
        print(f"结果已保存到: {output_csv}")


def test_fps_from_lookat(
    lookat_path: str = "viewer/data/TEST.lookat",
    model_path: str = "eval/flowers",
    n_frames: int = 100,
    output_csv: str = "fps_report.csv",
    width: int = 800,
    height: int = 600,
    background_color: List[float] = [0, 0, 0],
    max_views: Optional[int] = None
):
    """
    主测试函数
    
    Args:
        lookat_path: lookat文件路径
        model_path: 模型路径
        n_frames: 每个视角渲染的帧数
        output_csv: 输出CSV文件路径
        width: 渲染宽度
        height: 渲染高度
        background_color: 背景颜色
        max_views: 最大测试视角数（用于快速测试）
    """
    print("开始3DGS FPS测试")
    print(f"相机轨迹文件: {lookat_path}")
    print(f"模型路径: {model_path}")
    print(f"渲染分辨率: {width}x{height}")
    print(f"每视角渲染帧数: {n_frames}")
    
    # 1. 解析相机列表
    print("\n1. 解析相机轨迹文件...")
    views = parse_lookat_file(lookat_path)
    
    if max_views:
        views = views[:max_views]
        print(f"限制测试视角数为: {max_views}")
    
    # 2. 初始化场景
    print("\n2. 初始化场景和渲染器...")
    try:
        gaussians, pipeline = init_scene(model_path)
    except Exception as e:
        print(f"场景初始化失败: {e}")
        print("请确保模型路径正确且包含训练好的模型")
        return
    
    # 3. 执行FPS测试
    print(f"\n3. 开始FPS测试，共{len(views)}个视角...")
    results = []
    
    # 使用tqdm进度条显示渲染进度
    for idx, view_params in tqdm(enumerate(views), total=len(views), desc="渲染进度", unit="视角"):
        try:
            # 创建相机
            camera = create_camera_from_lookat(view_params, width, height)
            
            # 渲染并计时
            fps = render_and_time(camera, gaussians, pipeline, n_frames, background_color)
            results.append((idx, fps))
            
            # 更新进度条描述，显示当前FPS
            tqdm.write(f"View {idx:03d}: {fps:.2f} FPS")
            
        except Exception as e:
            tqdm.write(f"View {idx:03d}: 渲染失败 - {e}")
            continue
    
    # 4. 报告结果
    print("\n4. 生成测试报告...")
    report_results(results, output_csv)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="3DGS FPS测试工具")
    parser.add_argument("--lookat", default="viewer/data/TEST.lookat", 
                       help="相机轨迹文件路径")
    parser.add_argument("--model", default="eval/flowers", 
                       help="模型路径")
    parser.add_argument("--frames", type=int, default=100, 
                       help="每个视角渲染的帧数")
    parser.add_argument("--output", default="fps_report.csv", 
                       help="输出CSV文件路径")
    parser.add_argument("--width", type=int, default=800, 
                       help="渲染宽度")
    parser.add_argument("--height", type=int, default=600, 
                       help="渲染高度")
    parser.add_argument("--background", nargs=3, type=float, default=[0, 0, 0], 
                       help="背景颜色 (R G B)")
    parser.add_argument("--max-views", type=int, default=None, 
                       help="最大测试视角数（用于快速测试）")
    
    args = parser.parse_args()
    
    test_fps_from_lookat(
        lookat_path=args.lookat,
        model_path=args.model,
        n_frames=args.frames,
        output_csv=args.output,
        width=args.width,
        height=args.height,
        background_color=args.background,
        max_views=args.max_views
    )


if __name__ == "__main__":
    main() 