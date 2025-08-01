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
import matplotlib.pyplot as plt
import torchvision

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


def parse_bundle_file(path: str) -> List[Dict]:
    """
    解析Bundle格式的.out文件，提取相机参数
    
    Args:
        path: .out文件路径
        
    Returns:
        包含相机参数的字典列表
    """
    viewpoints = []
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # 跳过第一行注释
    if not lines or not lines[0].startswith('# Bundle file'):
        raise ValueError("不是有效的Bundle文件格式")
    
    # 解析相机数量
    header_parts = lines[1].strip().split()
    num_cameras = int(header_parts[0])
    
    print(f"Bundle文件包含 {num_cameras} 个相机")
    
    # 解析每个相机的参数
    line_idx = 2  # 从第3行开始
    for cam_idx in range(num_cameras):
        if line_idx >= len(lines):
            break
            
        # 解析焦距和畸变参数
        focal_line = lines[line_idx].strip().split()
        focal = float(focal_line[0])
        # 跳过畸变参数
        line_idx += 1
        
        # 解析旋转矩阵（3行）
        R = []
        for i in range(3):
            if line_idx >= len(lines):
                break
            row = [float(x) for x in lines[line_idx].strip().split()]
            R.append(row)
            line_idx += 1
        
        # 解析平移向量
        if line_idx >= len(lines):
            break
        T = [float(x) for x in lines[line_idx].strip().split()]
        line_idx += 1
        
        viewpoints.append({
            "focal": focal,
            "R": np.array(R),
            "T": np.array(T),
            "cam_id": cam_idx
        })
    
    print(f"成功解析了 {len(viewpoints)} 个相机参数")
    return viewpoints


def parse_colmap_format(cameras_file: str, images_file: str) -> List[Dict]:
    """
    解析COLMAP格式的相机参数
    
    Args:
        cameras_file: cameras.txt或cameras.bin文件路径
        images_file: images.txt或images.bin文件路径
        
    Returns:
        包含相机参数的字典列表
    """
    # 延迟导入，避免循环依赖
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_extrinsics_text, read_intrinsics_text, qvec2rotmat
    from utils.graphics_utils import focal2fov
    
    viewpoints = []
    
    try:
        # 尝试读取二进制文件
        cam_extrinsics = read_extrinsics_binary(images_file)
        cam_intrinsics = read_intrinsics_binary(cameras_file)
    except:
        # 如果二进制文件不存在，读取文本文件
        cam_extrinsics = read_extrinsics_text(images_file)
        cam_intrinsics = read_intrinsics_text(cameras_file)
    
    print(f"COLMAP文件包含 {len(cam_extrinsics)} 个相机")
    
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        
        # 四元数转旋转矩阵
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        
        # 计算FOV
        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, intr.height)
            FovX = focal2fov(focal_length_x, intr.width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, intr.height)
            FovX = focal2fov(focal_length_x, intr.width)
        else:
            raise ValueError(f"不支持的相机模型: {intr.model}")
        
        viewpoints.append({
            "R": R,
            "T": T,
            "FovX": FovX,
            "FovY": FovY,
            "width": intr.width,
            "height": intr.height,
            "cam_id": idx
        })
    
    print(f"成功解析了 {len(viewpoints)} 个相机参数")
    return viewpoints


def create_camera_from_lookat(params: Dict, width: int = 800, height: int = 600, is_speedy_splat: bool = False):
    """
    从lookat参数创建Camera对象
    
    Args:
        params: 包含origin, target, up, fovy, clip的字典
        width: 图像宽度
        height: 图像高度
        is_speedy_splat: 是否为speedy-splat模型
        
    Returns:
        Camera对象
    """
    # 延迟导入，避免循环依赖
    from scene.cameras import Camera
    
    origin = np.array(params["origin"])
    target = np.array(params["target"])
    up = np.array(params["up"])
    
    # 使用Blender兼容的lookAt矩阵构建方法
    # 计算相机坐标系
    forward = target - origin
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # 构建Blender兼容的lookAt矩阵（世界到相机变换）
    # Blender约定：Z轴指向相机前方，Y轴向上
    R = np.eye(3)
    R[:, 0] = right
    R[:, 1] = up
    R[:, 2] = forward  # Z轴指向相机前方（Blender约定）
    
    # 计算平移向量
    T = -R @ origin
    
    # 计算FOV
    fovy = params["fovy"]
    fovx = fovy * width / height
    
    # 创建虚拟图像（用于Camera构造函数）
    dummy_image = np.zeros((height, width, 3), dtype=np.uint8)
    from PIL import Image
    pil_image = Image.fromarray(dummy_image)
    
    if is_speedy_splat:
        # Speedy-splat模型的Camera构造函数
        camera = Camera(
            colmap_id=0,
            R=R,
            T=T,
            FoVx=fovx,
            FoVy=fovy,
            image=torch.from_numpy(dummy_image).float().permute(2, 0, 1) / 255.0,  # 转换为torch tensor并归一化
            gt_alpha_mask=None,
            image_name="fps_test",
            uid=0,
            data_device="cuda"
        )
    else:
        # 默认模型的Camera构造函数
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


def create_camera_from_bundle(params: Dict, width: int = 800, height: int = 600, is_speedy_splat: bool = False):
    """
    从Bundle格式参数创建Camera对象
    
    Args:
        params: 包含focal, R, T的字典
        width: 图像宽度
        height: 图像高度
        is_speedy_splat: 是否为speedy-splat模型
        
    Returns:
        Camera对象
    """
    # 延迟导入，避免循环依赖
    from scene.cameras import Camera
    
    R = params["R"]
    T = params["T"]
    focal = params["focal"]
    
    # 计算FOV
    # Bundle格式中的focal是像素焦距，需要转换为角度
    fovy = 2 * np.arctan(height / (2 * focal))
    fovx = 2 * np.arctan(width / (2 * focal))
    
    # 创建虚拟图像（用于Camera构造函数）
    dummy_image = np.zeros((height, width, 3), dtype=np.uint8)
    from PIL import Image
    pil_image = Image.fromarray(dummy_image)
    
    if is_speedy_splat:
        # Speedy-splat模型的Camera构造函数
        camera = Camera(
            colmap_id=0,
            R=R,
            T=T,
            FoVx=fovx,
            FoVy=fovy,
            image=torch.from_numpy(dummy_image).float().permute(2, 0, 1) / 255.0,  # 转换为torch tensor并归一化
            gt_alpha_mask=None,
            image_name="fps_test",
            uid=0,
            data_device="cuda"
        )
    else:
        # 默认模型的Camera构造函数
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


def create_camera_from_colmap(params: Dict, width: int = 800, height: int = 600, is_speedy_splat: bool = False):
    """
    从COLMAP格式参数创建Camera对象
    
    Args:
        params: 包含R, T, FovX, FovY, width, height的字典
        width: 图像宽度（如果为None则使用params中的width）
        height: 图像高度（如果为None则使用params中的height）
        is_speedy_splat: 是否为speedy-splat模型
        
    Returns:
        Camera对象
    """
    # 延迟导入，避免循环依赖
    from scene.cameras import Camera
    
    R = params["R"]
    T = params["T"]
    FovX = params["FovX"]
    FovY = params["FovY"]
    
    # 使用params中的分辨率，如果没有指定的话
    if width is None:
        width = params["width"]
    if height is None:
        height = params["height"]
    
    # 创建虚拟图像（用于Camera构造函数）
    dummy_image = np.zeros((height, width, 3), dtype=np.uint8)
    from PIL import Image
    pil_image = Image.fromarray(dummy_image)
    
    if is_speedy_splat:
        # Speedy-splat模型的Camera构造函数
        camera = Camera(
            colmap_id=0,
            R=R,
            T=T,
            FoVx=FovX,
            FoVy=FovY,
            image=torch.from_numpy(dummy_image).float().permute(2, 0, 1) / 255.0,  # 转换为torch tensor并归一化
            gt_alpha_mask=None,
            image_name="fps_test",
            uid=0,
            data_device="cuda"
        )
    else:
        # 默认模型的Camera构造函数
        camera = Camera(
            resolution=(width, height),
            colmap_id=0,
            R=R,
            T=T,
            FoVx=FovX,
            FoVy=FovY,
            depth_params=None,
            image=pil_image,
            invdepthmap=None,
            image_name="fps_test",
            uid=0,
            data_device="cuda"
        )
    
    return camera


def init_scene(model_path: str, iteration: int = -1, skip_train_test_exp: bool = False):
    """
    初始化场景和渲染器
    
    Args:
        model_path: 模型路径
        iteration: 模型迭代次数
        skip_train_test_exp: 是否跳过train_test_exp参数设置
        
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
    
    # 加载PLY文件（这个模型的load_ply只接受2个参数）
    gaussians.load_ply(ply_path)
    
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
                   n_frames: int = 100, background_color: List[float] = [0, 0, 0],
                   save_image: bool = False, save_path: str = None) -> float:
    """
    渲染并计时
    
    Args:
        camera: 相机对象
        gaussians: 高斯模型
        pipeline: 渲染管线参数
        n_frames: 渲染帧数
        background_color: 背景颜色
        save_image: 是否保存渲染图片
        save_path: 图片保存路径
        
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
    
    # 创建CUDA事件用于精确计时
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # 同步GPU并开始计时
        torch.cuda.synchronize()
        start_event.record()
    else:
        start_time = time.time()
    
    with torch.no_grad():
        # 为单帧渲染添加进度条（当帧数较多时）
        if n_frames > 10:
            for _ in tqdm(range(n_frames), desc="渲染帧", leave=False, unit="帧"):
                render(camera, gaussians, pipeline, background)
        else:
            for _ in range(n_frames):
                render(camera, gaussians, pipeline, background)
    
    # 结束计时
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
    else:
        end_time = time.time()
        elapsed_time = end_time - start_time
    
    fps = n_frames / elapsed_time
    
    # 保存渲染图片
    if save_image and save_path:
        with torch.no_grad():
            rendered_image = render(camera, gaussians, pipeline, background)["render"]
            # 确保图片在正确的范围内
            rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
            torchvision.utils.save_image(rendered_image, save_path)
    
    return fps


def report_results(results: List[tuple], output_csv: Optional[str] = None, plot_path: Optional[str] = None):
    """
    报告测试结果
    
    Args:
        results: 结果列表，每个元素为(view_id, fps)
        output_csv: 可选的CSV输出文件路径
        plot_path: 可选的图表保存路径
    """
    if not results:
        print("没有测试结果")
        return
    
    fps_values = [r[1] for r in results]
    view_ids = [r[0] for r in results]
    
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
    
    # 生成FPS变化图表
    if plot_path:
        plt.figure(figsize=(12, 6))
        plt.plot(view_ids, fps_values, 'b-o', linewidth=2, markersize=6, label='FPS')
        plt.axhline(y=np.mean(fps_values), color='r', linestyle='--', alpha=0.7, label=f'Mean FPS: {np.mean(fps_values):.2f}')
        
        plt.xlabel('View ID', fontsize=12)
        plt.ylabel('FPS', fontsize=12)
        plt.title('3DGS Rendering FPS Performance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 添加统计信息
        plt.text(0.02, 0.98, f'Mean FPS: {np.mean(fps_values):.2f}\nMin FPS: {np.min(fps_values):.2f}\nMax FPS: {np.max(fps_values):.2f}\nStd Dev: {np.std(fps_values):.2f}', 
                transform=plt.gca().transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"FPS chart saved to: {plot_path}")


def test_fps_from_camera_file(
    camera_file: str = "viewer/data/TEST.out",
    model_path: str = "eval/flowers",
    n_frames: int = 100,
    output_csv: str = "fps_report.csv",
    width: int = 800,
    height: int = 600,
    background_color: List[float] = [0, 0, 0],
    max_views: Optional[int] = None,
    save_images: bool = False,
    image_save_interval: int = 10,
    output_dir: str = None,
    skip_train_test_exp: bool = False,
    is_speedy_splat: bool = False,
    even_only: bool = False,
    use_bundle: bool = True,
    use_colmap: bool = False,
    cameras_file: str = None,
    images_file: str = None
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
        save_images: 是否保存渲染图片
        image_save_interval: 图片保存间隔（每隔多少个视角保存一次）
        output_dir: 输出目录，如果为None则使用viewer/{model_name}
        skip_train_test_exp: 是否跳过train_test_exp参数设置
    """
    print("开始3DGS FPS测试")
    if use_colmap:
        print(f"相机内参文件: {cameras_file}")
        print(f"相机外参文件: {images_file}")
        print(f"使用格式: COLMAP")
    else:
        print(f"相机轨迹文件: {camera_file}")
        print(f"使用格式: {'Bundle (.out)' if use_bundle else 'Lookat (.lookat)'}")
    print(f"模型路径: {model_path}")
    print(f"渲染分辨率: {width}x{height}")
    print(f"每视角渲染帧数: {n_frames}")
    
    # 设置输出目录
    if output_dir is None:
        model_name = os.path.basename(model_path)
        output_dir = os.path.join("viewer", model_name)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 如果启用图片保存，创建图片目录
    images_dir = None
    if save_images:
        images_dir = os.path.join(output_dir, "rendered_images")
        os.makedirs(images_dir, exist_ok=True)
        print(f"图片保存间隔: 每{image_save_interval}个视角")
    
    # 1. 解析相机列表
    print("\n1. 解析相机轨迹文件...")
    if use_colmap:
        views = parse_colmap_format(cameras_file, images_file)
    elif use_bundle:
        views = parse_bundle_file(camera_file)
    else:
        views = parse_lookat_file(camera_file)
    
    if max_views:
        views = views[:max_views]
        print(f"限制测试视角数为: {max_views}")
    
    # 如果只测试偶数视角
    if even_only:
        views = views[::2]  # 取偶数索引的视角
        print(f"只测试偶数视角，共 {len(views)} 个视角")
    
    # 2. 初始化场景
    print("\n2. 初始化场景和渲染器...")
    try:
        gaussians, pipeline = init_scene(model_path, skip_train_test_exp=skip_train_test_exp)
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
            if use_colmap:
                camera = create_camera_from_colmap(view_params, width, height, is_speedy_splat)
            elif use_bundle:
                camera = create_camera_from_bundle(view_params, width, height, is_speedy_splat)
            else:
                camera = create_camera_from_lookat(view_params, width, height, is_speedy_splat)
            
            # 决定是否保存图片
            save_image = save_images and (idx % image_save_interval == 0)
            save_path = None
            if save_image:
                save_path = os.path.join(images_dir, f"view_{idx:03d}.png")
            
            # 渲染并计时
            fps = render_and_time(camera, gaussians, pipeline, n_frames, background_color, save_image, save_path)
            results.append((idx, fps))
            
            # 更新进度条描述，显示当前FPS
            status_msg = f"View {idx:03d}: {fps:.2f} FPS"
            if save_image:
                status_msg += f" [已保存图片]"
            tqdm.write(status_msg)
            
        except Exception as e:
            tqdm.write(f"View {idx:03d}: 渲染失败 - {e}")
            continue
    
    # 4. 报告结果
    print("\n4. 生成测试报告...")
    
    # 设置输出文件路径
    csv_path = os.path.join(output_dir, output_csv)
    plot_path = os.path.join(output_dir, "fps_plot.png")
    
    report_results(results, csv_path, plot_path)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="3DGS FPS测试工具")
    parser.add_argument("--camera-file", default="viewer/data/TEST.out", 
                       help="相机轨迹文件路径 (.out 或 .lookat)")
    parser.add_argument("--model", default="eval/flowers", 
                       help="模型路径")
    parser.add_argument("--frames", type=int, default=100, 
                       help="每个视角渲染的帧数")
    parser.add_argument("--output", default="fps_report.csv", 
                       help="输出CSV文件路径")
    parser.add_argument("--width", type=int, default=1256, 
                       help="渲染宽度")
    parser.add_argument("--height", type=int, default=828, 
                       help="渲染高度")
    parser.add_argument("--background", nargs=3, type=float, default=[0, 0, 0], 
                       help="背景颜色 (R G B)")
    parser.add_argument("--max-views", type=int, default=None, 
                       help="最大测试视角数（用于快速测试）")
    parser.add_argument("--save-images", action="store_true", 
                       help="保存渲染图片")
    parser.add_argument("--image-interval", type=int, default=40, 
                       help="图片保存间隔（每隔多少个视角保存一次）")
    parser.add_argument("--output-dir", type=str, default=None, 
                       help="输出目录，默认使用viewer/{model_name}")
    parser.add_argument("--skip-train-test-exp", action="store_true", 
                       help="跳过train_test_exp参数设置（已废弃，保留用于兼容性）")
    parser.add_argument("--speedy-splat", action="store_true", 
                       help="使用speedy-splat模型的Camera构造函数")
    parser.add_argument("--even-only", action="store_true", 
                       help="只测试偶数索引的相机视角")
    parser.add_argument("--use-lookat", action="store_true", 
                       help="使用lookat格式而不是bundle格式")
    parser.add_argument("--use-colmap", action="store_true", 
                       help="使用COLMAP格式")
    parser.add_argument("--cameras-file", type=str, default=None, 
                       help="COLMAP相机内参文件路径 (cameras.txt或cameras.bin)")
    parser.add_argument("--images-file", type=str, default=None, 
                       help="COLMAP相机外参文件路径 (images.txt或images.bin)")
    
    args = parser.parse_args()
    
    # 根据参数判断格式
    use_colmap = args.use_colmap
    use_bundle = not args.use_lookat and not use_colmap and args.camera_file.endswith('.out')
    
    test_fps_from_camera_file(
        camera_file=args.camera_file,
        model_path=args.model,
        n_frames=args.frames,
        output_csv=args.output,
        width=args.width,
        height=args.height,
        background_color=args.background,
        max_views=args.max_views,
        save_images=args.save_images,
        image_save_interval=args.image_interval,
        output_dir=args.output_dir,
        skip_train_test_exp=args.skip_train_test_exp,
        is_speedy_splat=args.speedy_splat,
        even_only=args.even_only,
        use_bundle=use_bundle,
        use_colmap=use_colmap,
        cameras_file=args.cameras_file,
        images_file=args.images_file
    )


if __name__ == "__main__":
    main() 