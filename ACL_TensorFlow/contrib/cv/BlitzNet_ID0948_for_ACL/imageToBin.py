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
import tensorflow as tf
from config import args
from getData.voc_loader import VOCLoader

import progressbar
import logging
log = logging.getLogger()
import numpy as np

def main(argv=None):
    if args.dataset == 'voc07' or args.dataset == 'voc07+12':
        loader = VOCLoader('07', 'test')
    if args.dataset == 'voc12-val':
        loader = VOCLoader('12', 'val', segmentation=args.segment)

    filenames = loader.get_filenames()
    image_list = []

    inputs = tf.placeholder(tf.float32, shape=[None, None, 3], name="input")
    img_ph = tf.image.resize_bilinear(tf.expand_dims(inputs, 0), (300, 300))# 增加一维，并reshape

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess0:
        bar = progressbar.ProgressBar()# 显示进度条
        for i in bar(range(len(filenames))):
            name = filenames[i]
            img = loader.load_image(name) # 获取图片
            image = sess0.run(img_ph, feed_dict={inputs: img})

            image_list.append(image)
            gt_bboxes, seg_gt, gt_cats, w, h, difficulty = loader.read_annotations(name) # 获取图片信息
            image.tofile("./binFile/img/{0:05d}.bin".format(i))
            # im = np.fromfile("./binFile/img/{0:05d}.bin".format(i), dtype=np.float32)
            # print(im)
            gt_bboxes.tofile("./binFile/gt_bboxes/{0:05d}.bin".format(i))
            seg_gt.tofile("./binFile/seg_gt/{0:05d}.bin".format(i))
            gt_cats.tofile("./binFile/gt_cats/{0:05d}.bin".format(i))
            # w.tofile("./binFile/w/{0:05d}.bin".format(i))
            # h.tofile("./binFile/h/{0:05d}.bin".format(i))
            difficulty.tofile("./binFile/difficulty/{0:05d}.bin".format(i))


if __name__ == '__main__':
    tf.app.run()