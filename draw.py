import cv2
import numpy as np
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageProcessor:
    """图像处理器类，支持批量处理和自动化裁切"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化图像处理器
        
        Args:
            config_file: 配置文件路径，如果为None则使用默认配置
        """
        self.config = self._load_config(config_file)
        
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """加载配置文件"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 默认配置
        return {
            "scenes": {
                "bicycle": {
                    "crop_params": {
                        "00006": {"x": 540, "y": 445, "w": 190, "h": 130, "zoom": 2.5}
                    },
                    "methods": {
                        "gt": "/home/jovyan/work/gs_07/QuadGaussian/eval/bicycle/test/ours_30000/gt/{frame_id}.png",
                        "speedy": "/home/jovyan/work/gs_07/picture/OG/speedy_bicycle_{frame_id}.png",
                        "adr": "/home/jovyan/work/gs_07/adrgaussian/eval/bicycle/test/ours_30000/renders/{frame_id}.png",
                        "3dgs": "/home/jovyan/work/gs_07/orggs/gaussian-splatting/eval/bicycle/test/ours_30000/renders/{frame_id}.png",
                        "ours": "/home/jovyan/work/gs_07/QuadGaussian/eval/bicycle/test/ours_30000/renders/{frame_id}.png"
                    }
                },
                "playroom": {
                    "crop_params": {
                        "00023": {"x": 640, "y": 50, "w": 200, "h": 100, "zoom": 2.5}
                    },
                    "methods": {
                        "gt": "/home/jovyan/work/gs_07/QuadGaussian/eval/playroom/test/ours_30000/gt/{frame_id}.png",
                        "speedy": "/home/jovyan/work/gs_07/picture/OG/speedy_playroom_{frame_id}.png",
                        "adr": "/home/jovyan/work/gs_07/adrgaussian/eval/playroom/test/ours_30000/renders/{frame_id}.png",
                        "3dgs": "/home/jovyan/work/gs_07/orggs/gaussian-splatting/eval/playroom/test/ours_30000/renders/{frame_id}.png",
                        "ours": "/home/jovyan/work/gs_07/QuadGaussian/eval/playroom/test/ours_30000/renders/{frame_id}.png"
                    }
                },
                "truck": {
                    "crop_params": {
                        "00023": {"x": 290, "y": 0, "w": 200, "h": 100, "zoom": 2.5}
                    },
                    "methods": {
                        "gt": "/home/jovyan/work/gs_07/QuadGaussian/eval/truck/test/ours_30000/gt/{frame_id}.png",
                        "speedy": "/home/jovyan/work/gs_07/picture/OG/speedy_truck_{frame_id}.png",
                        "adr": "/home/jovyan/work/gs_07/adrgaussian/eval/truck/test/ours_30000/renders/{frame_id}.png",
                        "3dgs": "/home/jovyan/work/gs_07/orggs/gaussian-splatting/eval/truck/test/ours_30000/renders/{frame_id}.png",
                        "ours": "/home/jovyan/work/gs_07/QuadGaussian/eval/truck/test/ours_30000/renders/{frame_id}.png"
                    }
                }
            },
            "output_dir": "/home/jovyan/work/gs_07/picture/processed",
            "draw_rectangles": True,
            "margin": 10
        }
    
    def process_single_image(self, input_path: str, output_path: str, 
                           crop_x: int, crop_y: int, crop_w: int, crop_h: int, 
                           zoom_factor: float = 2.5, draw_rectangles: bool = True,
                           margin: int = 10) -> bool:
        """
        处理单张图像
        
        Args:
            input_path: 输入图像路径
            output_path: 输出图像路径
            crop_x, crop_y: 裁剪区域左上角坐标
            crop_w, crop_h: 裁剪区域宽高
            zoom_factor: 放大倍数
            draw_rectangles: 是否绘制矩形框
            margin: 边距
            
        Returns:
            bool: 处理是否成功
        """
        try:
            # 读取图像
            image = cv2.imread(input_path)
            if image is None:
                logger.error(f'找不到图像: {input_path}')
                return False
                
            h, w = image.shape[:2]
            
            # 检查裁剪参数是否有效
            if (crop_x < 0 or crop_y < 0 or 
                crop_x + crop_w > w or crop_y + crop_h > h):
                logger.warning(f'裁剪参数超出图像范围: {input_path}')
                return False
            
            # 裁剪区域
            patch = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            
            # 放大区域
            zoomed_patch = cv2.resize(patch, 
                                    (int(crop_w * zoom_factor), int(crop_h * zoom_factor)), 
                                    interpolation=cv2.INTER_CUBIC)
            
            # 计算嵌入位置（右下角）
            zp_h, zp_w = zoomed_patch.shape[:2]
            paste_x = w - zp_w - margin
            paste_y = h - zp_h - margin
            
            # 确保粘贴位置有效
            if paste_x < 0 or paste_y < 0:
                logger.warning(f'放大后的图像太大，无法粘贴到右下角: {input_path}')
                return False
            
            # 将缩放后的区域粘贴到原图上
            result = image.copy()
            result[paste_y:paste_y+zp_h, paste_x:paste_x+zp_w] = zoomed_patch
            
            # 可选：绘制红框标注裁剪区域
            if draw_rectangles:
                cv2.rectangle(result, (crop_x, crop_y), 
                            (crop_x+crop_w, crop_y+crop_h), (0, 0, 255), 2)
                cv2.rectangle(result, (paste_x, paste_y), 
                            (paste_x+zp_w, paste_y+zp_h), (0, 0, 255), 2)
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存结果
            cv2.imwrite(output_path, result)
            logger.info(f'已保存结果图像: {output_path}')
            return True
            
        except Exception as e:
            logger.error(f'处理图像时出错 {input_path}: {str(e)}')
            return False
    
    def process_scene_method(self, scene: str, method: str, frame_id: str) -> bool:
        """
        处理特定场景、方法和帧的图像
        
        Args:
            scene: 场景名称
            method: 方法名称
            frame_id: 帧ID
            
        Returns:
            bool: 处理是否成功
        """
        if scene not in self.config["scenes"]:
            logger.error(f'未知场景: {scene}')
            return False
            
        scene_config = self.config["scenes"][scene]
        
        if frame_id not in scene_config["crop_params"]:
            logger.error(f'场景 {scene} 中未找到帧 {frame_id} 的裁剪参数')
            return False
            
        if method not in scene_config["methods"]:
            logger.error(f'场景 {scene} 中未找到方法 {method}')
            return False
        
        # 获取输入路径
        input_path = scene_config["methods"][method].format(frame_id=frame_id)
        
        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            logger.warning(f'输入文件不存在: {input_path}')
            return False
        
        # 获取裁剪参数
        crop_params = scene_config["crop_params"][frame_id]
        
        # 生成输出路径
        output_filename = f"{method}_{scene}_{frame_id}_processed.png"
        output_path = os.path.join(self.config["output_dir"], output_filename)
        
        # 处理图像
        return self.process_single_image(
            input_path=input_path,
            output_path=output_path,
            crop_x=crop_params["x"],
            crop_y=crop_params["y"],
            crop_w=crop_params["w"],
            crop_h=crop_params["h"],
            zoom_factor=crop_params["zoom"],
            draw_rectangles=self.config["draw_rectangles"],
            margin=self.config["margin"]
        )
    
    def batch_process(self, scenes: List[str] = None, methods: List[str] = None, 
                     frame_ids: List[str] = None) -> Dict[str, int]:
        """
        批量处理图像
        
        Args:
            scenes: 要处理的场景列表，如果为None则处理所有场景
            methods: 要处理的方法列表，如果为None则处理所有方法
            frame_ids: 要处理的帧ID列表，如果为None则处理所有帧
            
        Returns:
            Dict: 处理结果统计
        """
        results = {"success": 0, "failed": 0, "skipped": 0}
        
        # 如果没有指定，则使用所有可用的
        if scenes is None:
            scenes = list(self.config["scenes"].keys())
        if methods is None:
            methods = ["gt", "speedy", "adr", "3dgs", "ours"]
        if frame_ids is None:
            frame_ids = []
            for scene in scenes:
                if scene in self.config["scenes"]:
                    frame_ids.extend(list(self.config["scenes"][scene]["crop_params"].keys()))
            frame_ids = list(set(frame_ids))  # 去重
        
        logger.info(f'开始批量处理: 场景={scenes}, 方法={methods}, 帧={frame_ids}')
        
        for scene in scenes:
            for method in methods:
                for frame_id in frame_ids:
                    try:
                        if self.process_scene_method(scene, method, frame_id):
                            results["success"] += 1
                        else:
                            results["failed"] += 1
                    except Exception as e:
                        logger.error(f'处理 {scene}/{method}/{frame_id} 时出错: {str(e)}')
                        results["failed"] += 1
        
        logger.info(f'批量处理完成: 成功={results["success"]}, 失败={results["failed"]}')
        return results
    
    def create_config_template(self, output_path: str = "config_template.json"):
        """创建配置文件模板"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        logger.info(f'配置文件模板已保存到: {output_path}')
    
    def list_available_options(self):
        """列出所有可用的选项"""
        print("可用场景:")
        for scene in self.config["scenes"].keys():
            print(f"  - {scene}")
            print(f"    帧ID: {list(self.config['scenes'][scene]['crop_params'].keys())}")
            print(f"    方法: {list(self.config['scenes'][scene]['methods'].keys())}")
            print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='图像批量处理工具')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--scene', type=str, help='要处理的场景')
    parser.add_argument('--method', type=str, help='要处理的方法')
    parser.add_argument('--frame', type=str, help='要处理的帧ID')
    parser.add_argument('--scenes', nargs='+', help='要处理的场景列表')
    parser.add_argument('--methods', nargs='+', help='要处理的方法列表')
    parser.add_argument('--frames', nargs='+', help='要处理的帧ID列表')
    parser.add_argument('--batch', action='store_true', help='批量处理模式')
    parser.add_argument('--create-config', type=str, help='创建配置文件模板')
    parser.add_argument('--list', action='store_true', help='列出所有可用选项')
    parser.add_argument('--input', type=str, help='自定义输入路径')
    parser.add_argument('--output', type=str, help='自定义输出路径')
    parser.add_argument('--crop', nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'), 
                       help='自定义裁剪参数: X Y W H')
    parser.add_argument('--zoom', type=float, default=2.5, help='放大倍数')
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = ImageProcessor(args.config)
    
    # 列出可用选项
    if args.list:
        processor.list_available_options()
        return
    
    # 创建配置文件模板
    if args.create_config:
        processor.create_config_template(args.create_config)
        return
    
    # 自定义单张图像处理
    if args.input and args.output and args.crop:
        crop_x, crop_y, crop_w, crop_h = args.crop
        success = processor.process_single_image(
            input_path=args.input,
            output_path=args.output,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_w=crop_w,
            crop_h=crop_h,
            zoom_factor=args.zoom
        )
        if success:
            print("处理成功!")
        else:
            print("处理失败!")
        return
    
    # 单张图像处理
    if args.scene and args.method and args.frame:
        success = processor.process_scene_method(args.scene, args.method, args.frame)
        if success:
            print("处理成功!")
        else:
            print("处理失败!")
        return
    
    # 批量处理
    if args.batch:
        results = processor.batch_process(
            scenes=args.scenes,
            methods=args.methods,
            frame_ids=args.frames
        )
        print(f"批量处理完成: 成功={results['success']}, 失败={results['failed']}")
        return
    
    # 如果没有参数，显示帮助信息
    parser.print_help()


if __name__ == "__main__":
    main()