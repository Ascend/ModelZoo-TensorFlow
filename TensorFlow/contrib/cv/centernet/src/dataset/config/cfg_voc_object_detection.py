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
import os
from pathlib import Path

NAME = 'voc_object_detection'

# PATH
RAW_PATH = Path(os.environ['PRJPATH']) / 'dataset' / 'VOC/raw'
TFRECORD_PATH = Path(os.environ['PRJPATH']) / 'dataset' / 'VOC/tfrecord'

# Dataset
TRAIN_DATASET_PATHS = list((TFRECORD_PATH / 'train2007').glob('*.tfrecord')) + list(
    (TFRECORD_PATH / 'train2012').glob('*.tfrecord'))
VALIDATION_DATASET_PATHS = list((TFRECORD_PATH / 'test2007').glob('*.tfrecord'))
NUM_TRAIN_EXAMPLES = 5011 + 11540
NUM_VALIDATION_EXAMPLES = 4952
NUM_CLASSES = 20
NAME_TO_ID = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}
ID_TO_NAME = {value: key for key, value in NAME_TO_ID.items()}

# Build TFRecords
PARSER = ('dataset.parser.voc_object_detection', 'gen_get_examples')


def get_xml_paths(dataset_path, image_set):
    with open(dataset_path / 'ImageSets/Main' / image_set) as image_set:
        return [dataset_path / 'Annotations' / (x[:-1] + '.xml') for x in image_set.readlines()]


BUILD_TFRECORDS = {
    'train2007': {
        'prefix': 'voc2007train_',
        'xml_paths': get_xml_paths(RAW_PATH / 'train/VOCdevkit/VOC2007', 'trainval.txt'),
        'img_dir_path': RAW_PATH / 'train/VOCdevkit/VOC2007/JPEGImages',
        'name_to_id': NAME_TO_ID,
        'num_shards': 1
    },
    'train2012': {
        'prefix': 'voc2012train_',
        'xml_paths': get_xml_paths(RAW_PATH / 'train/VOCdevkit/VOC2012', 'trainval.txt'),
        'img_dir_path': RAW_PATH / 'train/VOCdevkit/VOC2012/JPEGImages',
        'name_to_id': NAME_TO_ID,
        'num_shards': 1
    },
    'test2007': {
        'prefix': 'voc2007test_',
        'xml_paths': get_xml_paths(RAW_PATH / 'test/VOCdevkit/VOC2007', 'test.txt'),
        'img_dir_path': RAW_PATH / 'test/VOCdevkit/VOC2007/JPEGImages',
        'name_to_id': NAME_TO_ID,
        'num_shards': 1
    }
}
