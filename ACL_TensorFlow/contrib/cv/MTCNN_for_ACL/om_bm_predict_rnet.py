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
batchsize = [2048,256,16]
threshold=[0.6,0.7,0.7]

FLAGS = flags.FLAGS
FLAGS(sys.argv)


def calibrate_box(bbox, reg):
    '''校准box
    参数：
      bbox:pnet生成的box

      reg:rnet生成的box偏移值
    返回值：
      调整后的box是针对原图的绝对坐标
    '''

    bbox_c = bbox.copy()
    w = bbox[:, 2] - bbox[:, 0] + 1
    w = np.expand_dims(w, 1)
    h = bbox[:, 3] - bbox[:, 1] + 1
    h = np.expand_dims(h, 1)
    reg_m = np.hstack([w, h, w, h])
    aug = reg_m * reg
    bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
    return bbox_c

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

def convert_to_square(box):
    '''将box转换成更大的正方形
    参数：
      box：预测的box,[n,5]
    返回值：
      调整后的正方形box，[n,5]
    '''
    square_box = box.copy()
    h = box[:, 3] - box[:, 1] + 1
    w = box[:, 2] - box[:, 0] + 1
    # 找寻正方形最大边长
    max_side = np.maximum(w, h)

    square_box[:, 0] = box[:, 0] + w * 0.5 - max_side * 0.5
    square_box[:, 1] = box[:, 1] + h * 0.5 - max_side * 0.5
    square_box[:, 2] = square_box[:, 0] + max_side - 1
    square_box[:, 3] = square_box[:, 1] + max_side - 1
    return square_box


def pad(bboxes, w, h):
    '''将超出图像的box进行处理
    参数：
      bboxes:人脸框
      w,h:图像长宽
    返回值：
      dy, dx : 为调整后的box的左上角坐标相对于原box左上角的坐标
      edy, edx : n为调整后的box右下角相对原box左上角的相对坐标
      y, x : 调整后的box在原图上左上角的坐标
      ex, ex : 调整后的box在原图上右下角的坐标
      tmph, tmpw: 原始box的长宽
    '''
    # box的长宽
    tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
    num_box = bboxes.shape[0]

    dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
    edx, edy = tmpw.copy() - 1, tmph.copy() - 1
    # box左上右下的坐标
    x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    # 找到超出右下边界的box并将ex,ey归为图像的w,h
    # edx,edy为调整后的box右下角相对原box左上角的相对坐标
    tmp_index = np.where(ex > w - 1)
    edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
    ex[tmp_index] = w - 1

    tmp_index = np.where(ey > h - 1)
    edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
    ey[tmp_index] = h - 1
    # 找到超出左上角的box并将x,y归为0
    # dx,dy为调整后的box的左上角坐标相对于原box左上角的坐标
    tmp_index = np.where(x < 0)
    dx[tmp_index] = 0 - x[tmp_index]
    x[tmp_index] = 0

    tmp_index = np.where(y < 0)
    dy[tmp_index] = 0 - y[tmp_index]
    y[tmp_index] = 0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
    return_list = [item.astype(np.int32) for item in return_list]

    return return_list

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
        dets = np.fromfile(os.path.join("{}".format(data_out_om), "{}_{}_pnet_output1.bin".format('boxes_c',indx)),np.float64)


        detshape1 = subprocess.getoutput("sed -n '1p' pnet_boxc_shape.log ")
        detshape2 = subprocess.getoutput("sed -n '2p' pnet_boxc_shape.log ")
        print('det shape :', detshape1,detshape2)
        dets = dets.reshape(int(detshape1),int(detshape2))
        h, w, c = img.shape
        # 调整超出图像的box
       
        dets = convert_to_square(dets)
       
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        
        delete_size = np.ones_like(tmpw) * 20
        ones = np.ones_like(tmpw)
        zeros = np.zeros_like(tmpw)
        num_boxes = np.sum(np.where((np.minimum(tmpw, tmph) >= delete_size), ones, zeros))
        cropped_ims = np.zeros((num_boxes, 24, 24, 3), dtype=np.float32)
       
        for i in range(num_boxes):
            # 将pnet生成的box相对与原图进行裁剪，超出部分用0补
            if tmph[i] < 20 or tmpw[i] < 20:
                continue
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = img[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (24, 24)) - 127.5) / 128

        minibatch = []
        cur = 0
        batch_size = batchsize[1]
        # 所有数据总数
        n = cropped_ims.shape[0]
        print("curcur1", cur, n)
        # 将数据整理成固定batch
        while cur < n:
            print("22ddd", cur,n)
            minibatch.append(cropped_ims[cur:min(cur + batch_size, n), :, :, :])
            cur += batch_size
        cls_prob_list = []
        bbox_pred_list = []
        landmark_pred_list = []
        
        for idx, data in enumerate(minibatch):
            
            m = data.shape[0]
            real_size = batch_size
            # 最后一组数据不够一个batch的处理
            if m < batch_size:
                keep_inds = np.arange(m)
                gap = batch_size - m
                while gap >= len(keep_inds):
                    gap -= len(keep_inds)
                    keep_inds = np.concatenate((keep_inds, keep_inds))
                if gap != 0:
                    keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
                data = data[keep_inds]
                real_size = m
                
            
            data.tofile(os.path.join("{}".format(data_in_om), "{}_{}_rnet_input.bin".format(indx,idx)))
            os.system("mkdir {}".format(ompath))

            os.system('rm -f *.txt')
            print(
                'atc --model={} --framework=3  --output={}/rnet_{}_{} --output_type=FP32 --soc_version={} --input_shape="Placeholder:{},24,24,3"  --input_format=NHWC'.format(
                    pbname, ompath, indx, idx, ascend, m))
            os.system(
                'atc --model={} --framework=3  --output={}/rnet_{}_{} --output_type=FP32 --soc_version={} --input_shape="Placeholder:{},24,24,3"  --input_format=NHWC'.format(
                    pbname, ompath, indx, idx, ascend, m))
            os.system("mkdir -p {}/{}/{}".format(data_out_om, indx, idx))

            print("./xacl_fmk-master/out/xacl_fmk  -m {}/rnet_{}_{}.om -i {}/{}_{}_rnet_input.bin -o {}/{}/{}/rnet".format(
                ompath, indx, idx, data_in_om, indx, idx, data_out_om, indx, idx))
            os.system(
                "./xacl_fmk-master/out/xacl_fmk  -m {}/rnet_{}_{}.om -i {}/{}_{}_rnet_input.bin -o {}/{}/{}/rnet".format(
                    ompath, indx, idx, data_in_om, indx, idx, data_out_om, indx, idx))

            os.system('grep -A 300 \'type: "NetOutput"\'  *_Build.txt|grep -A 5 \'shape {\' > shaper.log')

            shape1 = subprocess.getoutput("sed -n '2p' shaper.log |awk '{print$2}'")
            shape2 = subprocess.getoutput("sed -n '3p' shaper.log |awk '{print$2}'")
            cls_prob = np.fromfile(os.path.join("{}/{}/{}".format(data_out_om,indx,idx), "rnet_output_00_000.bin"),np.float32)
            cls_prob = cls_prob.reshape(int(shape1), int(shape2))
            bbox_pred = np.fromfile(os.path.join("{}/{}/{}".format(data_out_om,indx,idx), "rnet_output_01_000.bin"),np.float32)
            bbox_pred = bbox_pred.reshape(int(shape1), 4)
            landmark_pred = np.fromfile(os.path.join("{}/{}/{}".format(data_out_om,indx,idx), "rnet_output_02_000.bin"),np.float32)
           
            cls_prob_list.append(cls_prob[:real_size])
            bbox_pred_list.append(bbox_pred[:real_size])
            landmark_pred_list.append(landmark_pred[:real_size])
 

        ouput0 = np.concatenate(cls_prob_list, axis=0)
        ouput1 = np.concatenate(bbox_pred_list, axis=0)
        ouput2 = np.concatenate(landmark_pred_list, axis=0)

        cls_scores = ouput0[:, 1]
        keep_inds = np.where(cls_scores > threshold[1])[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            ouput1 = ouput1[keep_inds]
        keep = py_nms(boxes, 0.6)
        boxes = boxes[keep]
        # 对pnet截取的图像的坐标进行校准，生成rnet的人脸框对于原图的绝对坐标
        boxes_c = calibrate_box(boxes, ouput1[keep])
        boxes.tofile(os.path.join("{}/".format(data_out_om), "{}_{}_rnet_output0.bin".format('boxes',indx)))
        boxes_c.tofile(os.path.join("{}/".format(data_out_om), "{}_{}_rnet_output1.bin".format('boxes_c', indx)))

        boxcshape0 = boxes_c.shape[0]
        boxcshape1 = boxes_c.shape[1]

        with open('rnet_boxc_shape.log', 'w') as f:
            f.write(str(boxcshape0))
            f.write('\r\n')
            f.write(str(boxcshape1))

        indx = indx + 1

if __name__ == '__main__':
    do_bm_predict(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])



