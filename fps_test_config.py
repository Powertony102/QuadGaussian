#!/usr/bin/env python3
"""
FPS测试配置文件
用户可以修改此文件来自定义测试参数
"""

# 基本配置
BASIC_CONFIG = {
    "lookat_path": "viewer/images/TEST.lookat",  # 相机轨迹文件路径
    "model_path": "output",                      # 模型路径
    "n_frames": 100,                             # 每个视角渲染的帧数
    "output_csv": "fps_report.csv",              # 输出CSV文件路径
    "width": 800,                                # 渲染宽度
    "height": 600,                               # 渲染高度
    "background_color": [0, 0, 0],              # 背景颜色 [R, G, B]
    "max_views": None,                           # 最大测试视角数（None表示测试所有视角）
}

# 高分辨率测试配置
HIGH_RES_CONFIG = {
    "lookat_path": "viewer/images/TEST.lookat",
    "model_path": "output",
    "n_frames": 100,
    "output_csv": "fps_report_highres.csv",
    "width": 1920,
    "height": 1080,
    "background_color": [0, 0, 0],
    "max_views": None,
}

# 快速测试配置
QUICK_TEST_CONFIG = {
    "lookat_path": "viewer/images/TEST.lookat",
    "model_path": "output",
    "n_frames": 50,
    "output_csv": "fps_report_quick.csv",
    "width": 800,
    "height": 600,
    "background_color": [0, 0, 0],
    "max_views": 10,  # 只测试前10个视角
}

# 白色背景测试配置
WHITE_BG_CONFIG = {
    "lookat_path": "viewer/images/TEST.lookat",
    "model_path": "output",
    "n_frames": 100,
    "output_csv": "fps_report_whitebg.csv",
    "width": 800,
    "height": 600,
    "background_color": [1, 1, 1],  # 白色背景
    "max_views": None,
}

# 性能测试配置（多分辨率）
PERFORMANCE_CONFIGS = [
    {
        "name": "800x600",
        "width": 800,
        "height": 600,
        "n_frames": 100,
        "output_csv": "fps_report_800x600.csv",
    },
    {
        "name": "1280x720",
        "width": 1280,
        "height": 720,
        "n_frames": 100,
        "output_csv": "fps_report_1280x720.csv",
    },
    {
        "name": "1920x1080",
        "width": 1920,
        "height": 1080,
        "n_frames": 100,
        "output_csv": "fps_report_1920x1080.csv",
    },
    {
        "name": "4K",
        "width": 3840,
        "height": 2160,
        "n_frames": 50,  # 4K分辨率下减少帧数
        "output_csv": "fps_report_4k.csv",
    }
]

def get_config(config_name: str = "basic"):
    """
    获取指定配置
    
    Args:
        config_name: 配置名称 ("basic", "high_res", "quick", "white_bg")
        
    Returns:
        配置字典
    """
    configs = {
        "basic": BASIC_CONFIG,
        "high_res": HIGH_RES_CONFIG,
        "quick": QUICK_TEST_CONFIG,
        "white_bg": WHITE_BG_CONFIG,
    }
    
    return configs.get(config_name, BASIC_CONFIG)

def run_performance_test():
    """
    运行多分辨率性能测试
    """
    from fps_test import test_fps_from_lookat
    
    print("开始多分辨率性能测试...")
    
    for config in PERFORMANCE_CONFIGS:
        print(f"\n测试分辨率: {config['name']}")
        
        test_config = BASIC_CONFIG.copy()
        test_config.update(config)
        
        test_fps_from_lookat(**test_config)

if __name__ == "__main__":
    # 示例：运行快速测试
    from fps_test import test_fps_from_lookat
    
    config = get_config("quick")
    print(f"使用配置: {config}")
    
    test_fps_from_lookat(**config) 