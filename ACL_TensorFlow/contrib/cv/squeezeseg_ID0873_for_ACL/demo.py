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
#
# Author: Bichen Wu (bichen@berkeley.edu) 03/07/2017

"""Evaluation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
from six.moves import xrange
from PIL import Image
import glob

CLS_COLOR_MAP = np.array([[0.00, 0.00, 0.00],
                          [0.12, 0.56, 0.37],
                          [0.66, 0.55, 0.71],
                          [0.58, 0.72, 0.88]])

NUM_CLASS = 4
INPUT_MEAN = np.array([[[10.88, 0.23, -1.04, 0.21, 12.12]]])
INPUT_STD = np.array([[[11.47, 6.91, 0.86, 0.16, 12.32]]])

out_dir = "./inference_result"


def visualize_seg(label_map, one_hot=False):
    if one_hot:
        label_map = np.argmax(label_map, axis=-1)
    out = np.zeros(
        (label_map.shape[0], label_map.shape[1], label_map.shape[2], 3))

    for l in range(1, NUM_CLASS):
        out[label_map == l, :] = CLS_COLOR_MAP[l]

    return out


def bgr_to_rgb(ims):
    """Convert a list of images from BGR format to RGB format."""
    out = []
    for im in ims:
        out.append(im[:, :, ::-1])
    return out


def _normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def detect(use_eval=True):
    if use_eval == True:

        num_images = len(os.listdir("./bin_out/lidar"))
        lidar = os.listdir("./bin_out/lidar")
        lidar_mask = os.listdir("./bin_out/lidar_mask")
        pred_cls_total = os.listdir(os.path.join("./bin_out/pred_cls", os.listdir("./bin_out/pred_cls")[-1]))
        label = os.listdir("./bin_out/label")

        for i in range(0, 5):

            lidar = np.fromfile(
                os.path.join("./bin_out/lidar", lidar[i]),
                dtype=np.float32).reshape(64, 512, 5)

            lidar_mask = np.fromfile(
                os.path.join("./bin_out/lidar_mask", lidar_mask[i]),
                dtype=np.float32).reshape(64, 512, 1)

            pred_cls = np.fromfile(
                os.path.join(os.path.join("./bin_out/pred_cls", os.listdir("./bin_out/pred_cls")[-1]),
                             pred_cls_total[i]),
                dtype=np.int32).reshape(1, 64, 512)

            # save the data   
            np.save(os.path.join(out_dir, 'pred_' + "{}".format(i) + '.npy'), pred_cls[0])
            # save the plot
            depth_map = Image.fromarray((255 * _normalize(lidar[:, :, 3])).astype(np.uint8))
            label_map = Image.fromarray((255 * visualize_seg(pred_cls)[0]).astype(np.uint8))

            blend_map = Image.blend(
                depth_map.convert('RGBA'),
                label_map.convert('RGBA'),
                alpha=0.4)

            blend_map.save(os.path.join(out_dir, 'plot_' + "{}".format(i) + '.png'))

    else:

        demo_lidar_path = "./demo_data/demo_lidar"
        demo_lidar_mask_path = "./demo_data/demo_lidar_mask"
        demo_pred_cls_path = "./demo_data/demo_pred_cls"
        demo_output = "./demo_data/demo_output/"
        demo_npy = "./demo_data/demo_npy/"

        if (not os.path.exists(demo_lidar_path)):
            os.makedirs(demo_lidar_path)

        if (not os.path.exists(demo_lidar_mask_path)):
            os.makedirs(demo_lidar_mask_path)

        if (not os.path.exists(demo_pred_cls_path)):
            os.makedirs(demo_pred_cls_path)

        if (not os.path.exists(demo_output)):
            os.makedirs(demo_output)

        for f in os.listdir(demo_npy):
            lidar = np.load(os.path.join(demo_npy, f)).astype(np.float32, copy=False)[:, :, :5]

            lidar_mask = np.reshape((lidar[:, :, 4] > 0), [1, 64, 512, 1]).astype(np.float32)

            lidar = ((lidar - INPUT_MEAN) / INPUT_STD).reshape(1, 64, 512, 5).astype(np.float32)

            lidar.tofile(os.path.join(demo_lidar_path, f[:-3]) + "bin")

            lidar_mask.tofile(os.path.join(demo_lidar_mask_path, f[:-3]) + "bin")

            os.system(
                "$HOME/AscendProjects/tools/msame/out/msame --model squeezeseg_acc.om --input './demo_data/demo_lidar','./demo_data/demo_lidar_mask' --output ./demo_data/demo_pred_cls")

            pred_cls = np.fromfile(demo_pred_cls_path + '/' + os.listdir(demo_pred_cls_path)[0] + '/' +
                                   os.listdir(os.path.join(demo_pred_cls_path, os.listdir(demo_pred_cls_path)[0]))[0],
                                   dtype=np.int32).reshape(1, 64, 512)

            lidar = lidar.reshape(64, 512, 5)
            lidar_mask = lidar_mask.reshape(64, 512, 1)

            # save the data   

            np.save(os.path.join(demo_output, 'pred_' + f + '.npy'), pred_cls[0])
            # save the plot
            depth_map = Image.fromarray((255 * _normalize(lidar[:, :, 3])).astype(np.uint8))
            label_map = Image.fromarray((255 * visualize_seg(pred_cls)[0]).astype(np.uint8))

            blend_map = Image.blend(
                depth_map.convert('RGBA'),
                label_map.convert('RGBA'),
                alpha=0.4)

            blend_map.save(os.path.join(demo_output, 'plot_' + f + '.png'))


def main(argv=None):
    detect(False)


if __name__ == '__main__':
    main()
