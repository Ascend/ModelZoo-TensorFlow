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
import colorsys
from math import ceil, floor, sqrt

import tensorflow as tf
from PIL import ImageDraw

from . import image_augmentor
# from .image_augmentor import image_augmentor


def preprocess(data, augmentor_config, is_training=True):
    features = {
        'id': tf.FixedLenFeature([1], tf.string),
        'image': tf.FixedLenFeature([], tf.string),
        'shape': tf.FixedLenFeature([], tf.string)
    }
    if is_training:
        features['ground_truth'] = tf.FixedLenFeature([], tf.string)

    features = tf.parse_single_example(data, features=features)
    shape = tf.decode_raw(features['shape'], tf.int32)
    shape = tf.reshape(shape, [3])
    if is_training:
        ground_truth = tf.decode_raw(features['ground_truth'], tf.float32)
        ground_truth = tf.reshape(ground_truth, [-1, 5])
    images = tf.image.decode_jpeg(features['image'], channels=3)
    images = tf.cast(tf.reshape(images, shape), tf.float32)

    if is_training:
        images, ground_truth = image_augmentor.basic(image=images,
                                                     input_shape=shape,
                                                     ground_truth=ground_truth,
                                                     **augmentor_config)
    else:
        images = image_augmentor.basic(image=images,
                                       input_shape=shape,
                                       **augmentor_config)
    if is_training:
        return features['id'], images, shape, ground_truth
    return features['id'], images, shape


def bbox_converter(bbox, bbox_type='coco', to_type='array'):
    # center: ycenter, xcenter, height, width
    # coco: xmin, ymin, width, height
    # array: ymin, ymax, xmin, xmax
    # two_points: ymin, xmin, ymax, xmax
    func = {
        # (? -> array, array -> ?)
        'coco': (lambda x: [x[1], x[1] + x[3], x[0], x[0] + x[2]],
                 lambda x: [x[2], x[0], x[3] - x[2], x[1] - x[0]]),
        'center':
        (lambda x:
         [x[0] - x[2] / 2, x[0] + x[2] / 2, x[1] - x[3] / 2, x[1] + x[3] / 2],
         lambda x: [(x[0] + x[1]) / 2,
                    (x[2] + x[3]) / 2, x[1] - x[0], x[3] - x[2]]),
        'array': (lambda x: x, lambda x: x),
        'two_points': (lambda x: [x[0], x[2], x[1], x[3]],
                       lambda x: [x[0], x[2], x[1], x[3]])
    }
    return func[to_type][1](func[bbox_type][0](bbox))


def get_id_to_color(ids):
    num_ids = len(ids)
    num_h = ceil(sqrt(num_ids))
    num_s = ceil(sqrt(num_ids / num_h))
    num_v = ceil(num_ids / num_h / num_s)
    assert num_h * num_s * num_v >= num_ids

    colors_plan = [(h / num_h, s / num_s, v / num_v) for h in range(num_h)
                   for s in range(1, num_s + 1) for v in range(1, num_v + 1)]
    colors_plan = list(map(lambda x: colorsys.hsv_to_rgb(*x), colors_plan))
    colors_plan = list(
        map(lambda x: tuple(map(lambda y: floor(y * 255.), x)), colors_plan))
    return dict(zip(ids, colors_plan))


def draw_bboxes_on_img(img,
                       bboxes,
                       id_to_name,
                       id_to_color,
                       bbox_type='array'):
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        if min(bbox[:4]) < 0.0:
            continue
        category_id = bbox[4]
        text = id_to_name[category_id]
        if len(bbox) > 5:
            score = bbox[5]
            text = '%s  %s' % (text, score)
        ymin, ymax, xmin, xmax = bbox_converter(bbox[:4], bbox_type=bbox_type)
        draw.rectangle([(xmin, ymin), (xmax, ymax)],
                       outline=id_to_color[category_id])
        draw.text([xmin, ymin], text, id_to_color[category_id])
    return img