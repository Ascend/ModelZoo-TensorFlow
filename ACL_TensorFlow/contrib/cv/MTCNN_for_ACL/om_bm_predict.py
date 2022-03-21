# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import graph_util
import sys,os
import numpy as np
import six
import math
import cv2
import subprocess


test_dir='picture/'
min_face=20
threshold=[0.6,0.7,0.7]
scale_factor=0.79

FLAGS = flags.FLAGS
FLAGS(sys.argv)

boxcshape = 0

def py_nms(dets, thresh):
    '''剔除太相似的box'''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 将概率值从大到小排列
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)

        # 保留小于阈值的下标，因为order[0]拿出来做比较了，所以inds+1是原来对应的下标
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def generate_bbox(cls_map, reg, scale, threshold):
    """
     得到对应原图的box坐标，分类分数，box偏移量
    """
    # pnet大致将图像size缩小2倍
    stride = 2

    cellsize = 12

    # 将置信度高的留下
    t_index = np.where(cls_map > threshold)

    # 没有人脸
    if t_index[0].size == 0:
        return np.array([])
    # 偏移量
    dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]

    reg = np.array([dx1, dy1, dx2, dy2])
    score = cls_map[t_index[0], t_index[1]]
    # 对应原图的box坐标，分类分数，box偏移量
    boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                             np.round((stride * t_index[0]) / scale),
                             np.round((stride * t_index[1] + cellsize) / scale),
                             np.round((stride * t_index[0] + cellsize) / scale),
                             score,
                             reg])
    # shape[n,9]
    return boundingbox.T

def processed_image(img, scale):
    '''预处理数据，转化图像尺度并对像素归一到[-1,1]
    '''
    height, width, channels = img.shape
    new_height = int(height * scale)
    new_width = int(width * scale)
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
    img_resized = (img_resized - 127.5) / 128
    return img_resized

def do_bm_predict(ompath,data_in_om,data_out_om,ascend,pbname):

    path = test_dir
    indx = 0
    for item in os.listdir(path):

        img_path = os.path.join(path, item)
        img = cv2.imread(img_path)
        h, w, c = img.shape
        net_size = 12
        # 人脸和输入图像的比率
        current_scale = float(net_size) / min_face
        im_resized = processed_image(img, current_scale)
        current_height, current_width, _ = im_resized.shape

        all_boxes = list()
        size = 0
        # 图像金字塔
        while min(current_height, current_width) > net_size:
            # 类别和box
            image_reshape = np.reshape(im_resized, [1, current_height, current_width, 3])
            print("height", current_height)
            print("width", current_width)

            image_reshape = image_reshape.astype(np.float32)
            os.system('rm -f *.txt')
            print(
                'atc --model={} --framework=3  --output={}/pnet_{}_{} --output_type=FP32 --soc_version={} --input_shape="input_image:1,{},{},3"  --input_format=NHWC'.format(
                    pbname, ompath, indx, size, ascend, current_height, current_width))
            os.system('atc --model={} --framework=3  --output={}/pnet_{}_{} --output_type=FP32 --soc_version={} --input_shape="input_image:1,{},{},3"  --input_format=NHWC --buffer_optimize=off_optimize'.format(pbname, ompath, indx, size, ascend,current_height,current_width))

            os.system("mkdir -p {}/{}/{}".format(data_out_om, indx, size))
            os.system("mkdir {}".format(data_in_om))
            os.system("mkdir {}".format(ompath))
            image_reshape.tofile(os.path.join("{}".format(data_in_om), "{}_{}_pnet_input.bin".format(indx, size)))
            os.system("mkdir -p {}/{}/{}".format(data_out_om,indx,size))
            print("./xacl_fmk-master/out/xacl_fmk  -m {}/pnet_{}_{}.om -i {}/{}_{}_pnet_input.bin -o {}/{}/{}/".format(ompath,indx, size, data_in_om,indx, size, data_out_om,indx,size))
            os.system("./xacl_fmk-master/out/xacl_fmk  -m {}/pnet_{}_{}.om -i {}/{}_{}_pnet_input.bin -o {}/{}/{}/".format(ompath,indx, size, data_in_om, indx, size, data_out_om,indx,size))

            os.system('grep -A 300 \'type: "NetOutput"\'  *_Build.txt|grep -A 5 \'shape {\' > shapep.log')

            shape1 = subprocess.getoutput("sed -n '2p' shapep.log |awk '{print$2}'")
            shape2 = subprocess.getoutput("sed -n '3p' shapep.log |awk '{print$2}'")
            print("output shape:",shape1,shape2)

            cls_cls_map = np.fromfile(os.path.join("{}/{}/{}".format(data_out_om,indx,size), "_output_00_000.bin"),np.float32)
            cls_cls_map = cls_cls_map.reshape(int(shape1),int(shape2),2)
            reg = np.fromfile(os.path.join("{}/{}/{}".format(data_out_om,indx,size), "_output_01_000.bin"),np.float32)
            reg = reg.reshape(int(shape1), int(shape2), 4)
            boxes = generate_bbox(cls_cls_map[:, :, 1], reg, current_scale, threshold[0])
            current_scale *= scale_factor  # 继续缩小图像做金字塔
            im_resized = processed_image(img, current_scale)
            current_height, current_width, _ = im_resized.shape
            size = size + 1
            if boxes.size == 0:
                continue
            # 非极大值抑制留下重复低的box
            keep = py_nms(boxes[:, :5], 0.5)
            boxes = boxes[keep]
            all_boxes.append(boxes)
        if len(all_boxes) == 0:
            print("all_boxes is None")
            break
        all_boxes = np.vstack(all_boxes)
        # 将金字塔之后的box也进行非极大值抑制
        keep = py_nms(all_boxes[:, 0:5], 0.7)
        all_boxes = all_boxes[keep]
        boxes = all_boxes[:, :5]
        # box的长宽
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] \
              - all_boxes[:, 1] + 1
        # 对应原图的box坐标和分数
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T

        boxes.tofile(os.path.join("{}".format(data_out_om), "{}_{}_pnet_output0.bin".format('boxes',indx)))
        boxes_c.tofile(os.path.join("{}".format(data_out_om), "{}_{}_pnet_output1.bin".format('boxes_c', indx)))


        boxcshape0 = boxes_c.shape[0]
        boxcshape1 = boxes_c.shape[1]

        with open('pnet_boxc_shape.log', 'w') as f:
            f.write(str(boxcshape0))
            f.write('\r\n')
            f.write(str(boxcshape1))
       
        indx = indx + 1

if __name__ == '__main__':
    do_bm_predict(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])



