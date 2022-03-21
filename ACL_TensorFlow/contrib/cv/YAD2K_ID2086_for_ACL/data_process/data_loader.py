# Copyright 2021 Huawei Technologies Co., Ltd
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

#构建dataset，方便在训练过程中读取数据
from keras.utils import Sequence
import math
import cv2
import numpy as np
from data_process.config import classes,input_shape,  anchors

class SequenceData(Sequence):

    # 初始化数据发生器
    def __init__(self, path, input_shape, batch_size, anchors, num_classes):
        # path: 数据路径； input_shape: 模型输入图片大小； batch_size: 一个批次大小

        self.datasets = []
        with open(path, "r")as f:
            self.datasets = f.readlines()  # 读取训练数据集
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.datasets))
        self.anchors = anchors
        self.shuffle = True
        self.num_anchors = len(self.anchors)
        self.num_classes = num_classes
        self.max_boxes = 20  # 一张图像中最多的box数量，不足的补充0， 超出的截取前max_boxes个

    def __len__(self):
        num_images = len(self.datasets)  # 得到总的训练图片的个数
        return math.ceil(num_images / float(self.batch_size))  # 计算每一个epoch的迭代次数

    def __getitem__(self, idx):  # 获取第index个batch的数据
        print("调用 getitem算法")
        # 生成batch_size个索引
        batch_indexs = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据（包含：图片路径和对应的标签）
        batch = [self.datasets[k] for k in batch_indexs]
        X, y = self.data_generation(batch)  # 生成一个batch的数据
        # print("X",X)
        # print("y",y)
        return X, y

    def get_epochs(self):
        return self.__len__()

    def on_epoch_end(self):
        print("执行一下shuffle")
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        np.random.shuffle(self.indexes)

    # 根据box来计算 detectors_mask 和 matching_true_boxes
    def preprocess_true_boxes(self, boxes):
        print("我正在运行 preprocess_true_boxes方法 ")
        print("根据box来计算 detectors_mask 和 matching_true_boxes")
        # 输入boxes：box列表[x, y, w, h, class]，x,y,w,h都是经过归一化后的如 [[0.65234375  0.48541667  0.0328125   0.04583333 29.]]

        # 返回值：detectors_mask，shape: [13, 13, 5, 1],表示由哪个anchor负责预测目前
        #       matching_true_boxes，shape 同上 ，编码后的真实label

        height, width = self.input_shape  # 训练输入的图片大小（416,416）
        num_anchors = len(self.anchors)  # anchor 个数 5
        conv_height = height // 32  # 从输入图片到最后的网络输出，进行了32倍下采样，得到13*13的特征图
        conv_width = width // 32
        num_box_params = boxes.shape[1]
        # 创建掩码 (13, 13, 5, 1)
        detectors_mask = np.zeros((conv_height, conv_width, num_anchors, 1), dtype=np.float32)
        # 与真实值匹配的box (13, 13, 5, 5)
        matching_true_boxes = np.zeros((conv_height, conv_width, num_anchors, num_box_params), dtype=np.float32)

        for box in boxes:
            box_class = box[4:5]  # 类别索引
            # 得到box在特征图上的大小
            box = box[0:4] * np.array([conv_width, conv_height, conv_width, conv_height])
            i = np.floor(box[1]).astype('int')  # 得到当前box是在特征图中哪个像素点上
            j = np.floor(box[0]).astype('int')
            best_iou = 0
            best_anchor = 0
            for k, anchor in enumerate(self.anchors):
                box_maxes = box[2:4] / 2.  # 以目标中心点为坐标原点，得到目标框的右下角
                box_mins = -box_maxes  # 以目标中心点为坐标原点，得到目标框的左上角
                anchor_maxes = (anchor / 2.)  # anchor ，同上两行
                anchor_mins = -anchor_maxes

                # 计算实际box和anchor的iou，找到iou最大的anchor
                intersect_mins = np.maximum(box_mins, anchor_mins)
                intersect_maxes = np.minimum(box_maxes, anchor_maxes)
                intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
                intersect_area = intersect_wh[0] * intersect_wh[1]  # box 与anchor的交集的面积

                box_area = box[2] * box[3]  # box面积
                anchor_area = anchor[0] * anchor[1]  # anchor面积
                iou = intersect_area / (box_area + anchor_area - intersect_area)  # iou
                if iou > best_iou:  # 寻找iou最大的anchor
                    best_iou = iou
                    best_anchor = k

            if best_iou > 0:
                detectors_mask[i, j, best_anchor] = 1  # 置位iou最大的anchor，这个anchor将负责预测目标
                adjusted_box = np.array(
                    [
                        box[0] - j, box[1] - i,  # -j,-i后就得到了当前box相对于像素点左侧的位置
                        np.log(box[2] / self.anchors[best_anchor][0]),  # 对宽、高进行编码
                        np.log(box[3] / self.anchors[best_anchor][1]), box_class
                    ], dtype=np.float32)
                matching_true_boxes[i, j, best_anchor] = adjusted_box  # 对应的真实label
        # print("my detectors_mask:",detectors_mask)
        # print("my matching_true_boxes:",matching_true_boxes)
        print("detectors_mask.shape",detectors_mask.shape)
        print("my matching_true_boxes.shape",matching_true_boxes.shape)
        return detectors_mask, matching_true_boxes

    def get_detector_mask(self, true_boxes):
        detectors_mask = [0 for i in range(len(true_boxes))]
        matching_true_boxes = [0 for i in range(len(true_boxes))]
        for i, box in enumerate(true_boxes):
            detectors_mask[i], matching_true_boxes[i] = self.preprocess_true_boxes(box)
        return np.array(detectors_mask), np.array(matching_true_boxes)

    def read(self, dataset):
        dataset = dataset.strip().split()
        image_path = dataset[0]  # 得到一个图片的路径
        image = cv2.imread(image_path)
        orig_size = np.array([image.shape[1], image.shape[0]])  # 获取图片原尺寸
        orig_size = np.expand_dims(orig_size, axis=0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # opencv读取通道顺序为BGR，转为RGB
        image = cv2.resize(image, self.input_shape)  # 将图片resize 到模型要求输入大小 416*416
        image = image / 255.  # 图片归一化

        boxes = np.array([np.array(box.split(","), dtype=np.int) for box in dataset[1:]])  # 图片对应的 标签

        boxes_xy = 0.5 * (boxes[:, 0:2] + boxes[:, 2:4])  # 转化为中心点
        boxes_wh = boxes[:, 2:4] - boxes[:, 0:2]  # 宽、高

        boxes_xy = boxes_xy / orig_size  # 计算相对原尺寸的中点坐标和宽高
        boxes_wh = boxes_wh / orig_size

        # 拼接x，y，w，h，class，为N*5矩阵, N为图像中实际的box数量
        boxes = np.concatenate((boxes_xy, boxes_wh, boxes[:, 4:]), axis=1)  # (N,5)
        box_data = np.zeros((self.max_boxes, 5))  # 填充boxes，保证上述的N=20 ，（20,5）
        if len(boxes) > self.max_boxes:  # 超过20个框的话，只取20个
            boxes = boxes[:self.max_boxes]
        box_data[:len(boxes)] = boxes  # （20,5）
        return image, box_data  # （416,416,3），（20,5）

    # # 改变dataset的大小，变成batch_size的倍数
    # def change_dataset_size(x, y, batch_size):
    #     length = len(x)
    #     if (length % batch_size != 0):
    #         remainder = length % batch_size
    #         x = x[:(length - remainder)]
    #         y = y[:(length - remainder)]
    #         print(len(x))
    #         print(len(y))
    #     return x, y

    def data_generation(self, batch):
        print("正在运行 data_generation代码")
        images = []
        true_boxes = []
        for dataset in batch:  # 遍历这个batch中的所有的数据
            image, box = self.read(dataset)
            print("image", image.shape)
            print("box", box.shape)
            images.append(image)  # 把图片组合成到一个batch中
            true_boxes.append(box)  # 把标签组合到一个batch中
        images = np.array(images)  #
        true_boxes = np.array(true_boxes)  # （B,N,5） B为batchsize一个批次图像的数量，N=20表示一个图像中允许的最大目标数量

        # 根据true_boxes 生成detectors_mask和matching_true_boxes
        detectors_mask, matching_true_boxes = self.get_detector_mask(true_boxes)

        # 返回值：[图片， 真实标签， 哪个anchor复杂预测，转化到特诊上对一个的标签],
        # print("data_generation detectors_mask:",detectors_mask)
        # print("data_generation matching_true_boxes:",matching_true_boxes)
        print("data_generation detectors_mask.shape",detectors_mask.shape)
        print("data_generation matching_true_boxes.shape",matching_true_boxes.shape)
        return [images, true_boxes, detectors_mask, matching_true_boxes], np.zeros(self.batch_size)

if __name__ == "__main__":
    # 测试一个batchsize的数据
    train_sequence_test = SequenceData("2007_train.txt", input_shape, 8, anchors, len(classes))
    for traindata in train_sequence_test:
        print("图片shape:", np.array(traindata[0][0]).shape)
        print("标签shape:", np.array(traindata[0][1]).shape)
        break
