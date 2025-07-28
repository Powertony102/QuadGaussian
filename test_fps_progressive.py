#!/usr/bin/env python3
"""
渐进式FPS测试工具
测试从远到近的渲染性能变化
"""

import time
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

def generate_progressive_cameras(n_cameras: int = 100, 
                               start_distance: float = 10.0,
                               end_distance: float = 1.0,
                               center_point: tuple = (0, 0, 0),
                               fov: float = 0.7) -> List[Dict]:
    """
    生成从远到近的相机轨迹
    
    Args:
        n_cameras: 相机数量
        start_distance: 起始距离
        end_distance: 结束距离
        center_point: 目标中心点
        fov: 视场角
        
    Returns:
        相机参数列表
    """
    cameras = []
    
    for i in range(n_cameras):
        # 计算当前距离（指数衰减，让变化更明显）
        t = i / (n_cameras - 1)
        distance = start_distance * (end_distance / start_distance) ** t
        
        # 相机位置（在Z轴上移动）
        origin = (center_point[0], center_point[1], center_point[2] + distance)
        target = center_point
        
        # 相机朝向
        up = (0, 1, 0)
        
        cameras.append({
            "origin": origin,
            "target": target,
            "up": up,
            "fovy": fov,
            "clip": (0.01, 100.0)
        })
    
    print(f"生成了 {n_cameras} 个渐进式相机视角")
    print(f"距离范围: {start_distance:.2f} -> {end_distance:.2f}")
    
    return cameras

def create_camera_from_params(params: Dict, width: int = 800, height: int = 600, is_speedy_splat: bool = False):
    """从参数创建Camera对象"""
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
    R[:, 1] = -up
    R[:, 2] = forward
    
    # 平移向量
    T = -R @ origin
    
    # 计算FOV
    fovy = params["fovy"]
    fovx = fovy * width / height
    
    # 创建虚拟图像
    dummy_image = np.zeros((height, width, 3), dtype=np.uint8)
    from PIL import Image
    pil_image = Image.fromarray(dummy_image)
    
    if is_speedy_splat:
        camera = Camera(
            colmap_id=0,
            R=R,
            T=T,
            FoVx=fovx,
            FoVy=fovy,
            image=torch.from_numpy(dummy_image).float().permute(2, 0, 1) / 255.0,
            gt_alpha_mask=None,
            image_name="progressive_test",
            uid=0,
            data_device="cuda"
        )
    else:
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
            image_name="progressive_test",
            uid=0,
            data_device="cuda"
        )
    
    return camera

def init_scene(model_path: str, iteration: int = -1):
    """初始化场景"""
    from arguments import ModelParams, PipelineParams, get_combined_args
    from gaussian_renderer import GaussianModel
    from utils.system_utils import searchForMaxIteration
    import torch
    
    parser = argparse.ArgumentParser(description="Progressive FPS Test")
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iteration", default=iteration, type=int)
    
    args = parser.parse_args([])
    args.model_path = model_path
    args.source_path = model_path
    args.images = "images"
    args.depths = ""
    args.resolution = -1
    args.white_background = False
    args.data_device = "cuda"
    args.eval = False
    args.sh_degree = 3
    
    model = model_params.extract(args)
    pipeline = pipeline_params.extract(args)
    
    gaussians = GaussianModel(model.sh_degree)
    
    if iteration == -1:
        loaded_iter = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
    else:
        loaded_iter = iteration
    
    if loaded_iter is None:
        raise ValueError(f"在 {model_path}/point_cloud 中找不到训练好的模型")
    
    print(f"加载训练好的模型，迭代次数: {loaded_iter}")
    
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{loaded_iter}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"找不到模型文件: {ply_path}")
    
    gaussians.load_ply(ply_path)
    
    if not hasattr(gaussians, 'exposure_mapping') or gaussians.exposure_mapping is None:
        gaussians.exposure_mapping = {}
    if not hasattr(gaussians, '_exposure') or gaussians._exposure is None:
        gaussians._exposure = torch.eye(3, 4, device="cuda")[None]
    
    return gaussians, pipeline

def render_and_time(camera, gaussians, pipeline, 
                   n_frames: int = 100, background_color: List[float] = [0.5, 0.5, 0.5],
                   save_image: bool = False, save_path: str = None) -> float:
    """渲染并计时"""
    from gaussian_renderer import render
    
    background = torch.tensor(background_color, dtype=torch.float32, device="cuda")
    
    # 预热
    with torch.no_grad():
        for _ in range(5):
            render(camera, gaussians, pipeline, background)
    
    # 计时
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
    else:
        start_time = time.time()
    
    with torch.no_grad():
        for _ in range(n_frames):
            render(camera, gaussians, pipeline, background)
    
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0
    else:
        end_time = time.time()
        elapsed_time = end_time - start_time
    
    fps = n_frames / elapsed_time
    
    # 保存渲染图片
    if save_image and save_path:
        with torch.no_grad():
            rendered_image = render(camera, gaussians, pipeline, background)["render"]
            rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
            torchvision.utils.save_image(rendered_image, save_path)
    
    return fps

def test_progressive_fps(
    model_path: str = "eval/flowers",
    n_cameras: int = 50,
    start_distance: float = 8.0,
    end_distance: float = 2.0,
    n_frames: int = 100,
    width: int = 800,
    height: int = 600,
    background_color: List[float] = [0.5, 0.5, 0.5],
    save_images: bool = False,
    image_save_interval: int = 10,
    output_dir: str = None,
    is_speedy_splat: bool = False
):
    """主测试函数"""
    print("开始渐进式FPS测试")
    print(f"模型路径: {model_path}")
    print(f"相机数量: {n_cameras}")
    print(f"距离范围: {start_distance:.2f} -> {end_distance:.2f}")
    print(f"渲染分辨率: {width}x{height}")
    
    # 设置输出目录
    if output_dir is None:
        model_name = os.path.basename(model_path)
        output_dir = os.path.join("viewer", f"{model_name}_progressive")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 创建图片目录
    images_dir = None
    if save_images:
        images_dir = os.path.join(output_dir, "rendered_images")
        os.makedirs(images_dir, exist_ok=True)
    
    # 1. 生成相机轨迹
    print("\n1. 生成渐进式相机轨迹...")
    cameras = generate_progressive_cameras(n_cameras, start_distance, end_distance)
    
    # 2. 初始化场景
    print("\n2. 初始化场景和渲染器...")
    try:
        gaussians, pipeline = init_scene(model_path)
    except Exception as e:
        print(f"场景初始化失败: {e}")
        return
    
    # 3. 执行FPS测试
    print(f"\n3. 开始渐进式FPS测试...")
    results = []
    
    for idx, camera_params in tqdm(enumerate(cameras), total=len(cameras), desc="渲染进度", unit="视角"):
        try:
            # 创建相机
            camera = create_camera_from_params(camera_params, width, height, is_speedy_splat)
            
            # 计算当前距离
            origin = np.array(camera_params["origin"])
            target = np.array(camera_params["target"])
            distance = np.linalg.norm(origin - target)
            
            # 决定是否保存图片
            save_image = save_images and (idx % image_save_interval == 0)
            save_path = None
            if save_image:
                save_path = os.path.join(images_dir, f"progressive_{idx:03d}_dist_{distance:.2f}.png")
            
            # 渲染并计时
            fps = render_and_time(camera, gaussians, pipeline, n_frames, background_color, save_image, save_path)
            results.append((idx, fps, distance))
            
            # 显示进度
            status_msg = f"View {idx:03d}: {fps:.2f} FPS (距离: {distance:.2f})"
            if save_image:
                status_msg += f" [已保存图片]"
            tqdm.write(status_msg)
            
        except Exception as e:
            tqdm.write(f"View {idx:03d}: 渲染失败 - {e}")
            continue
    
    # 4. 报告结果
    print("\n4. 生成测试报告...")
    
    csv_path = os.path.join(output_dir, "progressive_fps_report.csv")
    plot_path = os.path.join(output_dir, "progressive_fps_plot.png")
    
    # 保存CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["view_id", "fps", "distance"])
        writer.writerows(results)
    
    # 生成图表
    distances = [r[2] for r in results]
    fps_values = [r[1] for r in results]
    
    plt.figure(figsize=(12, 8))
    
    # FPS vs 距离
    plt.subplot(2, 1, 1)
    plt.plot(distances, fps_values, 'b-o', linewidth=2, markersize=4, label='FPS')
    plt.xlabel('Distance', fontsize=12)
    plt.ylabel('FPS', fontsize=12)
    plt.title('Progressive FPS Test: FPS vs Distance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 统计信息
    plt.subplot(2, 1, 2)
    plt.hist(fps_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('FPS', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('FPS Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    plt.figtext(0.02, 0.02, f'Mean FPS: {np.mean(fps_values):.2f}\nMin FPS: {np.min(fps_values):.2f}\nMax FPS: {np.max(fps_values):.2f}\nStd Dev: {np.std(fps_values):.2f}', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"结果已保存到: {csv_path}")
    print(f"图表已保存到: {plot_path}")
    
    # 打印统计信息
    print("\n" + "="*50)
    print("渐进式FPS测试结果")
    print("="*50)
    print(f"平均FPS: {np.mean(fps_values):.2f}")
    print(f"最小FPS: {np.min(fps_values):.2f}")
    print(f"最大FPS: {np.max(fps_values):.2f}")
    print(f"标准差:  {np.std(fps_values):.2f}")
    print("="*50)

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="渐进式FPS测试工具")
    parser.add_argument("--model", default="eval/flowers", help="模型路径")
    parser.add_argument("--cameras", type=int, default=50, help="相机数量")
    parser.add_argument("--start-distance", type=float, default=8.0, help="起始距离")
    parser.add_argument("--end-distance", type=float, default=2.0, help="结束距离")
    parser.add_argument("--frames", type=int, default=100, help="每个视角渲染的帧数")
    parser.add_argument("--width", type=int, default=800, help="渲染宽度")
    parser.add_argument("--height", type=int, default=600, help="渲染高度")
    parser.add_argument("--background", nargs=3, type=float, default=[0.5, 0.5, 0.5], help="背景颜色 (R G B)")
    parser.add_argument("--save-images", action="store_true", help="保存渲染图片")
    parser.add_argument("--image-interval", type=int, default=10, help="图片保存间隔")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--speedy-splat", action="store_true", help="使用speedy-splat模型")
    
    args = parser.parse_args()
    
    test_progressive_fps(
        model_path=args.model,
        n_cameras=args.cameras,
        start_distance=args.start_distance,
        end_distance=args.end_distance,
        n_frames=args.frames,
        width=args.width,
        height=args.height,
        background_color=args.background,
        save_images=args.save_images,
        image_save_interval=args.image_interval,
        output_dir=args.output_dir,
        is_speedy_splat=args.speedy_splat
    )

if __name__ == "__main__":
    main() 