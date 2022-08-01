import numpy as np
from PIL import Image
import imageio
import os

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def imsave(images, size, path):
    return imageio.imwrite(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image
    return img

bin_path = "inference/2022728_14_55_36_141275" # msame推理得到的结果路径
dst_path = "inference_images"
dst_path_A = os.path.join(dst_path, "inferA")
dst_path_B = os.path.join(dst_path, "inferB")

os.makedirs(dst_path, exist_ok=True)
os.makedirs(dst_path_A, exist_ok=True)
os.makedirs(dst_path_B, exist_ok=True)

files = os.listdir(bin_path)

for file in files:
    file_name = file.split(".")[0]
    out_image = np.fromfile(os.path.join(bin_path, file),dtype="float32").reshape(1,256,256,3)

    if file_name[-1] == '0':
        save_images(out_image, [1,1],f"{dst_path_A}/{file_name}.jpg")
    elif file_name[-1] == '1':
        save_images(out_image, [1,1],f"{dst_path_B}/{file_name}.jpg")
