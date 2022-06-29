import PIL.Image as pil
import numpy as np
import os
import cv2

#使用PIL读入png文件为numpy数组；
#进行和在线推理时一致的图像处理操作（例如：减均值、归一化、调整大小、中心裁剪等）
#利用numpy的.tofile()函数将处理好的数组导出为bin文件，注意文件命名（例如：ILSVRC2012_val_00000001.JPEG --> ILSVRC2012_val_00000001.JPEG.bin）

use_mixed_precision = False
pyr_lvls = 6

def pick_frame(path):
    new_files = []
    for i in range(200):
        frame1 = os.path.join(path, 'image_2', '{:06d}'.format(i) + '_10.png')
        frame2 = os.path.join(path, 'image_2', '{:06d}'.format(i) + '_11.png')
        new_files.append([frame1, frame2])
    return new_files

def adapt_x(x):
    """Preprocess the input samples to adapt them to the network's requirements
    Here, x, is the actual data, not the x TF tensor.
    Args:
        x: input samples in list[(2,H,W,3)] or (N,2,H,W,3) np array form
    Returns:
        Samples ready to be given to the network (w. same shape as x)
        Also, return adaptation info in (N,2,H,W,3) format
    """
    # Ensure we're dealing with RGB image pairs
    assert (isinstance(x, np.ndarray) or isinstance(x, list))
    if isinstance(x, np.ndarray):
        assert (len(x.shape) == 5)
        assert (x.shape[1] == 2 and x.shape[4] == 3)
    else:
        assert (len(x[0].shape) == 4)
        assert (x[0].shape[0] == 2 or x[0].shape[3] == 3)

    # Bring image range from 0..255 to 0..1 and use floats (also, list[(2,H,W,3)] -> (batch_size,2,H,W,3))
    if use_mixed_precision is True:
        x_adapt = np.array(x, dtype=np.float16) if isinstance(x, list) else x.astype(np.float16)
    else:
        x_adapt = np.array(x, dtype=np.float32) if isinstance(x, list) else x.astype(np.float32)
    x_adapt /= 255.

    # Make sure the image dimensions are multiples of 2**pyramid_levels, pad them if they're not
    _, pad_h = divmod(x_adapt.shape[2], 2 ** pyr_lvls)
    if pad_h != 0:
        pad_h = 2 ** pyr_lvls - pad_h
    _, pad_w = divmod(x_adapt.shape[3], 2 ** pyr_lvls)
    if pad_w != 0:
        pad_w = 2 ** pyr_lvls - pad_w
    x_adapt_info = None
    if pad_h != 0 or pad_w != 0:
        padding = [(0, 0), (0, 0), (0, pad_h), (0, pad_w), (0, 0)]
        x_adapt_info = x_adapt.shape  # Save original shape
        x_adapt = np.pad(x_adapt, padding, mode='constant', constant_values=0.)

    return x_adapt, x_adapt_info

dataset_dir = "E:/datasets/DF-Net_dataset/data_scene_flow_2015/training/"
img_width = 1280
img_height = 384

new_files = pick_frame(dataset_dir)

for t in range(0, len(new_files)):
    print("start to process %s" % new_files[t][0])
    # 对原始图片进行需要的预处理
    raw_im0 = cv2.imread(new_files[t][0])
    raw_im1 = cv2.imread(new_files[t][1])  # (375, 1242, 3)
    x = np.stack([raw_im0, raw_im1])
    x = np.expand_dims(x, 0)
    x_adapt, x_adapt_info = adapt_x(x) #x_adapt:(1, 2, 384, 1280, 3)

    # 保存为bin格式
    save_path = new_files[t][0][:-14] + '_bin' + new_files[t][0][-14:-7] + '.bin'
    #if not os.path.exists(save_path[:-6]):
    #    os.makedirs(save_path[:-6])
    x_adapt.tofile(save_path)


