# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
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
from npu_bridge.npu_init import *
import h5py
from progressbar import ProgressBar
from PIL import Image

"""
This model is used to convert data files to h5 database to facilitate training
"""


# IMG_MEAN for Pascal dataset
IMG_MEAN = np.array(
    (122.67891434, 116.66876762, 104.00698793), dtype=np.float32)  # RGB


def read_images(data_list):
    with open(data_list, 'r') as f:
        data = [line.strip("\n").split(' ') for line in f]
    return data


def process_image(image, shape, resize_mode=Image.BILINEAR):
    img = Image.open(image)
    img = img.resize(shape, resize_mode)
    img.load()
    img = np.asarray(img, dtype="float32")
    if len(img.shape) < 3:
        return img.T
    else:
        return np.transpose(img, (1,0,2))


def build_h5_dataset(data_dir, list_path, out_dir, shape, name, norm=False):
    images = read_images(list_path)
    images_size = len(images)
    dataset = h5py.File(out_dir+name+'.h5', 'w')
    dataset.create_dataset('X', (images_size, *shape, 3), dtype='f')
    dataset.create_dataset('Y', (images_size, *shape), dtype='f')
    pbar = ProgressBar()
    for index, (image, label) in pbar(enumerate(images)):
        image = process_image(data_dir+image, shape)
        label = process_image(data_dir+label, shape, Image.NEAREST)
        image -= IMG_MEAN
        image = image / 255. if norm else image
        dataset['X'][index], dataset['Y'][index] = image, label
    dataset.close()


if __name__ == '__main__':
    shape = (256, 256)
    data_dir = './dataset'
    list_dir = './dataset/'
    output_dir = './dataset/'

    data_files = {
        'training': 'train.txt',
        'validation': 'val.txt',
        'testing': 'test.txt'
    }
    for name, list_path in data_files.items():
        build_h5_dataset(data_dir, list_dir+list_path, output_dir, shape, name)
