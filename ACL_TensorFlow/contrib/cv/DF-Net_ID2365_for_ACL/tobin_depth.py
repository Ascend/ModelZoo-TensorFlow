import PIL.Image as pil
import numpy as np
import os
import tensorflow as tf

# 使用PIL读入png文件为numpy数组；
# 进行和在线推理时一致的图像处理操作（例如：减均值、归一化、调整大小、中心裁剪等）
# 利用numpy的.tofile()函数将处理好的数组导出为bin文件，注意文件命名（例如：ILSVRC2012_val_00000001.JPEG --> ILSVRC2012_val_00000001.JPEG.bin）

dataset_dir = "/data-val/dataset/KITTI/raw/data/"
txt_file = './data/kitti/mytest_files_eigen.txt'
img_width = 576
img_height = 160

with open(txt_file, 'r') as f:
    test_files = f.readlines()
    test_files = [dataset_dir + t[:-1] for t in test_files]

session = tf.Session()
for t in range(0, len(test_files)):
    print("start to process %s" % test_files[t])
    img_org = pil.open(test_files[t])  # 读入数据
    scaled_im = img_org.resize((img_width, img_height), pil.ANTIALIAS)  # 对原始图片进行需要的预处理
    res = np.array(scaled_im)
    image = tf.image.convert_image_dtype(res, dtype=tf.float32)
    image = image * 2. - 1.
    image_input = image.eval(session=session)

    save_path = test_files[t].replace('raw', 'bin', 1)[:-4]
    if not os.path.exists(save_path[:-10]):
        os.makedirs(save_path[:-10])
    image_input.tofile(save_path + ".bin")  # 处理后的图片保存为bin文件
    '''
    # 确保是相同的变换
    depth_input = np.fromfile(save_path + ".bin", dtype=np.float32)
    depth_input = depth_input.reshape([160, 576, 3])
    print(depth_input == image_input)
    '''


