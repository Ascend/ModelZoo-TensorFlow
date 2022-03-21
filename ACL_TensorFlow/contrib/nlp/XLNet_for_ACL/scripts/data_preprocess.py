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
# =============================================================================

import os
import argparse
import numpy as np
import tensorflow as tf
from os.path import join
from tqdm import tqdm

seq_length = 128

name_to_features = {
    "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
    "input_mask": tf.FixedLenFeature([seq_length], tf.float32),
    "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    "label_ids": tf.FixedLenFeature([], tf.float32),
    "is_real_example": tf.FixedLenFeature([], tf.int64),
}


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    return  example

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./proc_data/sts-b/spiece.model.len-128.dev.eval.tf_record")
    parser.add_argument("--output_path", type=str, default="./input_bins/")
    parser.add_argument("--sample_num", type=int, default=1500)
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(join(args.output_path, 'input_ids')):
        os.makedirs(join(args.output_path, 'input_ids'))
    if not os.path.exists(join(args.output_path, 'input_mask')):
        os.makedirs(join(args.output_path, 'input_mask'))
    if not os.path.exists(join(args.output_path, 'label_ids')):
        os.makedirs(join(args.output_path, 'label_ids'))
    if not os.path.exists(join(args.output_path, 'segment_ids')):
        os.makedirs(join(args.output_path, 'segment_ids'))
    if not os.path.exists(join(args.output_path, 'is_real_example')):
        os.makedirs(join(args.output_path, 'is_real_example'))

    input_file = args.input_file

    d = tf.data.TFRecordDataset(input_file)
    iterator = d.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        for i in tqdm(range(args.sample_num)):
            value = sess.run(next_element)
            example = _decode_record(value,name_to_features)
            example['input_ids'].eval().tofile(join(args.output_path, 'input_ids', '{}.bin'.format(str(i).zfill(6))))
            example['input_mask'].eval().tofile(join(args.output_path, 'input_mask', '{}.bin'.format(str(i).zfill(6))))
            example['label_ids'].eval().tofile(join(args.output_path, 'label_ids', '{}.bin'.format(str(i).zfill(6))))
            example['segment_ids'].eval().tofile(join(args.output_path, 'segment_ids', '{}.bin'.format(str(i).zfill(6))))
            example['is_real_example'].eval().tofile(join(args.output_path, 'is_real_example', '{}.bin'.format(str(i).zfill(6))))
