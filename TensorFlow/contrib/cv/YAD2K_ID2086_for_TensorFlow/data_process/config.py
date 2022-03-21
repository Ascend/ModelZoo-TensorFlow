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
import numpy as np

anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
anchors = np.array(anchors).reshape(-1, 2)  # (5,2)
input_shape = (416, 416)
batch_size = 10
epochs = 100
colors = [[255, 0, 255], [218, 112, 214], [100, 149, 237], [95, 158, 160], [0, 255, 255], [0, 255, 127], [107, 142, 35],
          [255, 255, 0], [184, 134, 11], [255, 165, 0], [255, 0, 0], [224, 255, 255], [70, 130, 180], [255, 192, 203],
          [255, 240, 245], [0, 255, 0],
          [240, 128, 128], [220, 220, 220], [0, 0, 0], [169, 169, 169]]
VOCdevkit_path = 'Dataset/VOCdevkit'
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
