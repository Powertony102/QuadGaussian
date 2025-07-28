import os
import zipfile

# 根目录（你贴的目录结构的根）
root_dir = '/home/jovyan/work/gs_07/QuadGaussian/eval'
output_zip = os.path.join(root_dir, 'TRAIN_INFO.zip')

# 创建 Zip 文件
with zipfile.ZipFile(output_zip, 'w') as zipf:
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            results_path = os.path.join(subdir_path, 'TRAIN_INFO')
            if os.path.exists(results_path):
                # 将文件添加到 zip，并指定存档中的名称
                zipf.write(results_path, arcname=f'{subdir}/TRAIN_INFO')

print(f"✅ 所有 TRAIN_INFO 已压缩为: {output_zip}")