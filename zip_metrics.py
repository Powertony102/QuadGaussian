#!/usr/bin/env python3
import os
import zipfile

# 根 eval 目录
root_dir = '/home/jovyan/work/gs_07/QuadGaussian/eval'
# 输出 zip 文件名
output_zip = os.path.join(root_dir, 'all_metrics.zip')

with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    # 遍历根目录下所有子文件夹
    for scene in os.listdir(root_dir):
        scene_dir = os.path.join(root_dir, scene)
        if not os.path.isdir(scene_dir):
            continue

        # 构造 metrics.csv 的路径
        metrics_path = os.path.join(scene_dir, 'test', 'ours_30000', 'metrics.csv')
        if os.path.isfile(metrics_path):
            # 在 zip 内保持相对目录结构：scene_name/test/ours_30000/metrics.csv
            arcname = os.path.relpath(metrics_path, root_dir)
            zf.write(metrics_path, arcname)
            print(f"Added: {arcname}")
        else:
            print(f"Skipped (not found): {scene}/test/ours_30000/metrics.csv")

print(f"\nAll done! Created zip: {output_zip}")
