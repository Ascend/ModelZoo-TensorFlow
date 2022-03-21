#
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
import math
import os

from pathlib import Path

from lxml import etree
import numpy as np
import tensorflow as tf

from ..util.tfrecord import bytes_feature


def xml_to_example(xmlpath, imgpath, name_to_id, prefix):
    if isinstance(xmlpath, Path):
        xmlpath = str(xmlpath)
    xml = etree.parse(xmlpath)
    root = xml.getroot()
    imgname = root.find('filename').text
    imgname = os.path.join(imgpath, imgname)
    image = tf.io.gfile.GFile(imgname, 'rb').read()
    size = root.find('size')
    height = int(size.find('height').text)
    width = int(size.find('width').text)
    depth = int(size.find('depth').text)
    shape = np.asarray([height, width, depth], np.int32)
    xpath = xml.xpath('//object')
    ground_truth = np.zeros([len(xpath), 5], np.float32)
    for i in range(len(xpath)):
        obj = xpath[i]
        classid = name_to_id[obj.find('name').text]
        bndbox = obj.find('bndbox')
        ymin = float(bndbox.find('ymin').text)
        ymax = float(bndbox.find('ymax').text)
        xmin = float(bndbox.find('xmin').text)
        xmax = float(bndbox.find('xmax').text)
        ground_truth[i, :] = np.asarray([ymin, ymax, xmin, xmax, classid], np.float32)
    features = {
        'id': bytes_feature((prefix + os.path.split(imgname)[1]).encode()),
        'image': bytes_feature(image),
        'shape': bytes_feature(shape.tobytes()),
        'ground_truth': bytes_feature(ground_truth.tobytes())
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def gen_get_examples(xml_paths, img_dir_path, name_to_id, prefix='voc_', num_shards=1):
    total = len(xml_paths)
    print('[INFO] total: %d' % total)
    num_examples_per_shard = math.ceil(total / num_shards)
    for shard_id in range(num_shards):
        start_idx = shard_id * num_examples_per_shard
        end_idx = min(total, start_idx + num_examples_per_shard)
        print('[INFO] Shard %d, %d examples' % (shard_id + 1, end_idx - start_idx))

        def get_examples():
            for i in range(start_idx, end_idx):
                yield xml_to_example(xml_paths[i], img_dir_path, name_to_id, prefix)

        yield get_examples()
