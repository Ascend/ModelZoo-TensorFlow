# -*- coding: utf-8 -*-
import os
import imageio
import numpy as np
from skimage.transform import resize

precision_mode = 32

def preprocessing(x):
    x = x/127.5 - 1 # -1 ~ 1
    return x

def load_test_data(image_path, size_h=256, size_w=256):
    img = imageio.imread(image_path, pilmode= 'RGB')
    img = resize(img, output_shape=(size_h, size_w))
    img = preprocessing(img)

    return img

def files2bin(src_path, dst_path):
    files = os.listdir(src_path)
    for file in files:
        print(f"start to process {os.path.join(src_path, file)}")
        sample_image = np.asarray(load_test_data(os.path.join(src_path, file), size_h=256, size_w=256),dtype=f"float{precision_mode}")

        sample_image.tofile(os.path.join(dst_path, f"{file.split('.')[0]}_{precision_mode}") + ".bin") # 处理后的图片保存为bin文件

src_path_A = "val/valA"
src_path_B = "val/valB"

dst_path = "image_test"
dst_path_A = os.path.join(dst_path, "valA")
dst_path_B = os.path.join(dst_path, "valB")
dst_path_style = os.path.join(dst_path, "style")

os.makedirs(dst_path, exist_ok=True)
os.makedirs(dst_path_A, exist_ok=True)
os.makedirs(dst_path_B, exist_ok=True)
os.makedirs(dst_path_style, exist_ok=True)

# 生成图片的bin文件
files2bin(src_path_A, dst_path_A)
files2bin(src_path_B, dst_path_B)

# 生成style的bin文件
files = os.listdir(src_path_A)
for file in files:
    test_style = np.random.normal(loc=0.0, scale=1.0, size=[1, 1, 1, 8]).astype(f"float{precision_mode}")
    test_style.tofile(os.path.join(dst_path_style, f"{file.split('.')[0]}_{precision_mode}") + ".bin")