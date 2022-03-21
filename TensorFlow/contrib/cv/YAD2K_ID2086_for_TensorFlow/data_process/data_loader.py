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
# 构建dataset，方便在训练过程中读取数据
from keras.utils import Sequence
import math
import cv2
import numpy as np
from data_process.config import classes, input_shape, anchors


class SequenceData(Sequence):

    # data init
    def __init__(self, path, input_shape, batch_size, anchors, num_classes, config):
        # path: data path； input_shape: size of input； batch_size:
        self.config = config
        self.datasets = []
        with open(path, "r") as f:
            self.datasets = f.readlines()  # read train dataset
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.datasets))
        self.anchors = anchors
        self.shuffle = True
        self.num_anchors = len(self.anchors)
        self.num_classes = num_classes
        self.max_boxes = 20  # the max_boxes in one image.

    def __len__(self):
        num_images = len(self.datasets)  # the total number of train images
        return math.ceil(num_images / float(self.batch_size))  # iterition in one epoch

    def __getitem__(self, idx):  # gain  batch  in index size
        # batch_size index
        batch_indexs = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        # find the images
        batch = [self.datasets[k] for k in batch_indexs]
        X, y = self.data_generation(batch)  # generate a batch data
        return X, y

    def get_epochs(self):
        return self.__len__()

    def on_epoch_end(self):
        # shuffle index
        np.random.shuffle(self.indexes)

    # According to box  compute detectors_mask and matching_true_boxes

    def preprocess_true_boxes(self, boxes):
        # input boxes：box list [x, y, w, h, class]，x,y,w,h  guiyihua  [[0.65234375  0.48541667  0.0328125   0.04583333 class]]

        # return ：detectors_mask，shape: [13, 13, 5, 1], which anchor for pre
        #       matching_true_boxes，shape  the same as below

        height, width = self.input_shape  # (416,416)
        num_anchors = len(self.anchors)  # anchor=5
        conv_height = height // 32  # input through network to output, use 32size of unpooling,gain 13*13 feature maps .Then gird cell divide'
        conv_width = width // 32  # grid cell divide
        # second dimension： [x, y, w, h, class],x,y ,w,h.    x,y,w,h are in [0,1]。
        num_box_params = boxes.shape[1]
        # create mask (13, 13, 5, 1)  for pre  anchor box
        detectors_mask = np.zeros((conv_height, conv_width, num_anchors, 1), dtype=np.float32)
        # box (13, 13, 5, 5)     GT
        matching_true_boxes = np.zeros((conv_height, conv_width, num_anchors, num_box_params), dtype=np.float32)

        for box in boxes:  # for all boxes
            box_class = box[4:5]  # which class
            # grid cell value
            box = box[0:4] * np.array([conv_width, conv_height, conv_width, conv_height])
            i = np.floor(box[1]).astype('int')  #  y direction belongs to  which grid cell
            j = np.floor(box[0]).astype('int')  # x direction belongs to grid cell
            best_iou = 0
            best_anchor = 0
            # 'compute anchor boxes and true boxes IOU, find the best anchor boxes'
            for k, anchor in enumerate(self.anchors):
                box_maxes = box[2:4] / 2.  # center is the O , gain the right down corner
                box_mins = -box_maxes  # center is the O , gain the left up corner
                anchor_maxes = (anchor / 2.)  # anchor the same as below
                anchor_mins = -anchor_maxes

                # compute real box and anchor IOU,find the best iou's anchor
                intersect_mins = np.maximum(box_mins, anchor_mins)
                intersect_maxes = np.minimum(box_maxes, anchor_maxes)
                intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
                intersect_area = intersect_wh[0] * intersect_wh[1]  # IOU area

                box_area = box[2] * box[3]  # box area
                anchor_area = anchor[0] * anchor[1]  # anchor area
                iou = intersect_area / (box_area + anchor_area - intersect_area)  # iou
                if iou > best_iou:  # find the biggest iou's anchor
                    best_iou = iou
                    best_anchor = k

            if best_iou > 0:
                detectors_mask[i, j, best_anchor] = 1  # the best anchor for pre
                # (j,i)this grid cell 's best_anchor. It's fourth dimension change to 1,as for mask.
                adjusted_box = np.array(
                    [
                        box[0] - j, box[1] - i,  # -j,-i  find the left up corner(0,0)
                        np.log(box[2] / self.anchors[best_anchor][0]),  # w,h  anchor boxes w,h  log
                        np.log(box[3] / self.anchors[best_anchor][1]), box_class  # class
                    ], dtype=np.float32)
                matching_true_boxes[i, j, best_anchor] = adjusted_box  # real label
        return detectors_mask, matching_true_boxes

    def get_detector_mask(self, true_boxes):
        detectors_mask = [0 for i in range(len(true_boxes))]
        matching_true_boxes = [0 for i in range(len(true_boxes))]
        for i, box in enumerate(true_boxes):
            detectors_mask[i], matching_true_boxes[i] = self.preprocess_true_boxes(box)
        return np.array(detectors_mask), np.array(matching_true_boxes)

    def read(self, dataset):
        dataset = dataset.strip().split()
        image_path = dataset[0]
        image = cv2.imread(image_path)
        orig_size = np.array([image.shape[1], image.shape[0]])  # origin size
        orig_size = np.expand_dims(orig_size, axis=0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # opencv BGR change to RGB
        image = cv2.resize(image, self.input_shape)  # resize image to 416*416
        image = image / 255.  # normlize

        boxes = np.array([np.array(box.split(","), dtype=np.int) for box in dataset[1:]])  # label

        boxes_xy = 0.5 * (boxes[:, 0:2] + boxes[:, 2:4])  # center
        boxes_wh = boxes[:, 2:4] - boxes[:, 0:2]  # w、h

        boxes_xy = boxes_xy / orig_size  # compute the origin
        boxes_wh = boxes_wh / orig_size

        # concatenate x，y，w，h，class,  N*5  m ,N is the real box counts in images
        boxes = np.concatenate((boxes_xy, boxes_wh, boxes[:, 4:]), axis=1)  # (N,5)
        box_data = np.zeros((self.max_boxes, 5))  # fill boxes, hold N=20 ,(20,5)
        if len(boxes) > self.max_boxes:  # over 20, use only 20
            boxes = boxes[:self.max_boxes]
        box_data[:len(boxes)] = boxes  # (20,5)
        return image, box_data  # (416,416,3),(20,5)

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

        images = []
        true_boxes = []
        for dataset in batch:
            image, box = self.read(dataset)
            images.append(image)  # batch
            true_boxes.append(box)  # batch
        images = np.array(images)  #
        true_boxes = np.array(true_boxes)  # (B,N,5) B is the number of batchsize images in a batch, N=20 represents the maximum number of targets allowed in an image

        # Generate detectors_mask and matching_true_boxes according to true_boxes
        detectors_mask, matching_true_boxes = self.get_detector_mask(true_boxes)

        # Return value: [images, real label, which anchor complex prediction, converted to a label on the special diagnosis],
        return [images, true_boxes, detectors_mask, matching_true_boxes], np.zeros(self.batch_size)


if __name__ == "__main__":
    # Test a batchsize data
    train_sequence_test = SequenceData("2007_train.txt", input_shape, 8, anchors, len(classes))
    print("outout dimension ")
    for traindata in train_sequence_test:
        print("images shape:", np.array(traindata[0][0]).shape)
        print("labels shape:", np.array(traindata[0][1]).shape)
        break
