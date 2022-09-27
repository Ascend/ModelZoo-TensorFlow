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
from tqdm import tqdm
from PIL import Image
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input


def cvtColor(img):
    if len(np.shape(img)) == 3 and np.shape(img)[2] == 3:
        return img
    else:
        img = img.convert('RGB')
        return img

image_ids = open("/root/FasterRCNN/VOCdevkit/VOC2012/ImageSets/Main/val.txt").read().strip().split()


for image_id in tqdm(image_ids):
    image_path = '/root/FasterRCNN/VOCdevkit/VOC2012/JPEGImages/' + image_id + '.jpg'
    image = Image.open(image_path)
    image = cvtColor(image)
    image = np.expand_dims(preprocess_input(np.array(image, dtype='float32')), 0)
    if image.shape[1] == 600:
        image.tofile('/root/FasterRCNN/FasterRCNN-PR/input_bins_x600/' + image_id + '.bin')
    else:
        image.tofile('/root/FasterRCNN/FasterRCNN-PR/input_bins_x800/' + image_id + '.bin')
