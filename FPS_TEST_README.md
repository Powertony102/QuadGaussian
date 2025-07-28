# 3DGS FPS测试工具

这是一个基于3D Gaussian Splatting (3DGS) 的渲染性能测试工具，可以测试不同相机视角下的渲染帧率。

## 功能特性

- 📊 **多视角FPS测试**: 支持从相机轨迹文件批量测试多个视角的渲染性能
- 📈 **详细统计报告**: 提供平均、最小、最大FPS和标准差等统计信息
- 💾 **CSV结果导出**: 支持将测试结果保存为CSV文件，便于后续分析
- ⚙️ **灵活配置**: 支持自定义渲染分辨率、背景颜色、测试帧数等参数
- 🚀 **GPU优化**: 包含GPU预热和同步机制，确保测试结果准确性

## 文件结构

```
├── fps_test.py              # 主测试工具
├── test_fps_quick.py        # 快速测试脚本（验证功能）
├── test_fps_simple.py       # 简化测试脚本
├── fps_test_config.py       # 配置文件
├── FPS_TEST_README.md       # 使用说明
└── viewer/data/
    ├── TEST.lookat          # 相机轨迹文件
    └── TEST.out             # Bundle文件（可选）
```

## 安装依赖

确保已安装项目所需的所有依赖：

```bash
pip install torch torchvision numpy pillow opencv-python tqdm
```

## 使用方法

### 1. 基本用法

```bash
# 首先检查模型路径是否正确
python check_model.py [model_path]

# 简单测试（推荐）
python test_simple.py

# 使用默认参数运行完整测试
python fps_test.py

# 指定模型路径
python fps_test.py --model /path/to/your/model

# 快速测试（只测试前3个视角，每视角10帧）
python test_fps_quick.py

# 使用简化测试脚本
python test_fps_simple.py
```

### 2. 命令行参数

```bash
python fps_test.py [选项]

选项:
  --lookat PATH          相机轨迹文件路径 (默认: viewer/data/TEST.lookat)
  --model PATH           模型路径 (默认: eval/flowers)
  --frames N             每个视角渲染的帧数 (默认: 100)
  --output PATH          输出CSV文件路径 (默认: fps_report.csv)
  --width N              渲染宽度 (默认: 800)
  --height N             渲染高度 (默认: 600)
  --background R G B     背景颜色 (默认: 0 0 0)
  --max-views N          最大测试视角数（用于快速测试）
  --save-images          保存渲染图片
  --image-interval N     图片保存间隔（每隔多少个视角保存一次，默认: 10）
  --output-dir PATH      输出目录，默认使用viewer/{model_name}
```

### 3. 示例用法

```bash
# 测试高分辨率渲染性能
python fps_test.py --width 1920 --height 1080 --frames 200

# 使用白色背景
python fps_test.py --background 1 1 1

# 快速测试前20个视角
python fps_test.py --max-views 20 --frames 50

# 自定义输出文件
python fps_test.py --output my_fps_results.csv

# 保存渲染图片（每隔10个视角保存一次）
python fps_test.py --save-images --image-interval 10

# 完整示例：保存图片并生成可视化
python fps_test.py --model eval/flowers --save-images --image-interval 5 --max-views 20
```

## 输入文件格式

### TEST.lookat 文件格式

每行包含一个相机视角的参数：

```
CamXXX -D origin=X,Y,Z -D target=X,Y,Z -D up=X,Y,Z -D fovy=θ -D clip=near,far
```

示例：
```
Cam000 -D origin=0.10681,1.94477,5.25577 -D target=0.164741,1.72807,4.28125 -D up=0.0109114,-0.975962,0.217666 -D fovy=0.737495 -D clip=0.009,1100
```

参数说明：
- `origin`: 相机位置 (X, Y, Z)
- `target`: 目标点位置 (X, Y, Z)
- `up`: 上方向向量 (X, Y, Z)
- `fovy`: 垂直视场角（弧度）
- `clip`: 近远裁剪平面 (near, far)

## 输出结果

### 输出文件结构

当启用图片保存功能时，会在 `viewer/{model_name}` 目录下生成以下文件：

```
viewer/{model_name}/
├── fps_report.csv          # FPS测试结果CSV文件
├── fps_plot.png           # FPS变化图表
└── rendered_images/       # 渲染图片目录
    ├── view_000.png
    ├── view_010.png
    ├── view_020.png
    └── ...
```

### 控制台输出

```
开始3DGS FPS测试
相机轨迹文件: viewer/data/TEST.lookat
模型路径: eval/flowers
渲染分辨率: 800x600
每视角渲染帧数: 100

1. 解析相机轨迹文件...
解析了 669 个相机视角

2. 初始化场景和渲染器...
场景初始化完成，模型路径: output
加载迭代次数: 30000

3. 开始FPS测试，共669个视角...
View 000: 62.45 FPS
View 001: 61.23 FPS
View 002: 63.12 FPS
...

4. 生成测试报告...

==================================================
FPS测试结果
==================================================
View 000: 62.45 FPS
View 001: 61.23 FPS
View 002: 63.12 FPS
...
--------------------------------------------------
平均FPS: 61.85
最小FPS: 58.92
最大FPS: 64.78
标准差:  1.23
==================================================
结果已保存到: fps_report.csv
```

### CSV输出格式

```csv
view_id,fps
0,62.45
1,61.23
2,63.12
...
```

## 性能优化建议

1. **GPU预热**: 工具会自动进行5帧预热，确保GPU达到稳定状态
2. **内存管理**: 使用 `torch.no_grad()` 减少内存占用
3. **批量测试**: 对于大量视角，建议使用 `--max-views` 分批测试
4. **分辨率调整**: 根据硬件性能调整渲染分辨率

## 故障排除

### 常见问题

1. **模型路径错误**
   ```
   场景初始化失败: [Errno 2] No such file or directory: 'output'
   ```
   解决：确保模型路径正确，包含训练好的模型文件
   
   **正确的模型路径结构：**
   ```
   your_model_path/
   ├── point_cloud/
   │   └── iteration_30000/
   │       └── point_cloud.ply
   ├── cameras.json
   └── input.ply
   ```
   
   **示例：**
   ```bash
   # 如果您的模型在 eval/flowers 目录
   python fps_test.py --model eval/flowers
   
   # 如果您的模型在其他位置
   python fps_test.py --model /path/to/your/trained/model
   ```

2. **CUDA内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决：减少渲染分辨率或测试帧数

3. **相机轨迹文件格式错误**
   ```
   解析了 0 个相机视角
   ```
   解决：检查TEST.lookat文件格式是否正确

### 调试模式

如需调试，可以修改 `fps_test.py` 中的参数：

```python
# 在 test_fps_from_lookat 函数中添加调试信息
print(f"相机参数: {view_params}")
```

## 扩展功能

### 支持Bundle文件

工具预留了Bundle文件解析接口，可以扩展支持更精细的相机参数：

```python
def parse_bundle_file(path: str) -> List[Dict]:
    """解析TEST.out Bundle文件"""
    # TODO: 实现Bundle文件解析
    pass
```

### 多线程渲染

对于大量视角测试，可以考虑添加多线程支持：

```python
import concurrent.futures

def test_viewpoint_parallel(view_params, gaussians, pipeline):
    """并行测试单个视角"""
    # TODO: 实现并行渲染
    pass
```

## 许可证

本工具遵循项目的LICENSE.md文件中的许可证条款。

## 贡献

欢迎提交Issue和Pull Request来改进这个工具！ 