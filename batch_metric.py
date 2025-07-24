import os
import subprocess

# 所有场景名
scenes = [
    # "bicycle",
    # "bonsai",
    # "counter",
    # "flowers",
    # "garden",
    # "kitchen",
    # "room",
    # "stump",
    "treehill"
]

# scenes = [
#     "train",
#     "truck"
# ]

# scenes = [
#     "drjohnson",
#     "playroom"
# ]

# 输出根目录
output_root = "output/orggs"  # 替换为你的实际路径

# 渲染迭代次数
iteration = 30000  # 可根据需要修改

for scene in scenes:
    model_path = os.path.join(output_root, scene)
    cmd = [
        "python", "metrics.py",
        "--model_path", model_path
    ]
    print(f"开始渲染场景: {scene}")
    subprocess.run(cmd)
    print(f"完成渲染场景: {scene}\n")