#!/usr/bin/env python

# Copyright 2019 Google LLC (original)
# Copyright 2019 Uizard Technologies (small modifications)
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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


"""Script to create SSL splits from a dataset.
"""
from npu_bridge.npu_init import *

from collections import defaultdict
import json
import os

from absl import app
from absl import flags
from libml import utils
from libml.data import DATA_DIR
import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm
from typing import List, Any

flags.DEFINE_integer('seed', 0, 'Random seed to use, 0 for no shuffling.')
flags.DEFINE_integer('label_split_size', 0, 'Size of labeled data in the split.')
flags.DEFINE_bool('custom', False, 'True if custom unlabeled dataset is in use.')
flags.DEFINE_integer('label_size', 1000, 'Total size of labeled data in the original dataset.')
flags.DEFINE_bool('class_balance', False, 'True if classes need to be balanced.')
FLAGS = flags.FLAGS


def get_class(serialized_example):
    return tf.parse_single_example(serialized_example, features={'label': tf.FixedLenFeature([], tf.int64)})['label']


def main(argv):
    assert FLAGS.label_split_size and FLAGS.label_split_size > 0
    if FLAGS.custom:
        assert (FLAGS.label_size >= FLAGS.label_split_size), "The size of labeled images in the split cannot exceed the total number of labeled \
            images in the dataset."

    argv.pop(0)
    if any(not os.path.exists(f) for f in argv[1:]):
        raise FileNotFoundError(argv[1:])
    target = '%s.%d@%d' % (argv[0], FLAGS.seed, FLAGS.label_split_size)

    # if os.path.exists(target + '-label.tfrecord') and os.path.exists(target + '-unlabel.tfrecord'):
    #     raise FileExistsError('For safety overwriting is not allowed', target)

    input_files = argv[1:]
    count = 0
    id_class = []
    class_id = defaultdict(list)

    print('Computing class distribution')
    dataset = tf.data.TFRecordDataset(input_files)
    # 以前是1024除不开 24并行
    dataset = dataset.map(get_class, 24).batch(250, drop_remainder=True)
    it = dataset.make_one_shot_iterator().get_next()
    try:
        # Store each image in a dict by its class number and image id.
        with tf.Session(config=npu_config_proto()) as session, tqdm(leave=False) as t:
            while 1:
                old_count = count
                for i in session.run(it):
                    id_class.append(i)

                    # In a custom dataset, if the count has exceeded the total_size
                    # of the labeled dataset, the remaining images are unlabeled images.
                    if count < FLAGS.label_size or not FLAGS.custom:
                        class_id[i].append(count)

                    count += 1
                t.update(count - old_count)
    except tf.errors.OutOfRangeError:
        pass
    print('%d records found' % count)
    print(set(id_class))

    nclass = len(class_id)
    train_stats = np.array([len(class_id[i]) for i in range(nclass)], np.float64)
    train_stats /= train_stats.max()

    if 'stl10' in argv[1]:
        # All of the unlabeled data is given label 0, but we know that
        # STL has equally distributed data among the 10 classes.
        train_stats[:] *= 0
        train_stats[:] += 1

    # Compute the class distribution statistics for the labeled part of the dataset.
    print('  Stats', ' '.join(['%.2f' % (100 * x) for x in train_stats]))
    assert min(class_id.keys()) == 0 and max(class_id.keys()) == (nclass - 1)
    class_id = [np.array(class_id[i], dtype=np.int64) for i in range(nclass)]

    if FLAGS.seed:
        np.random.seed(FLAGS.seed)
        for i in range(nclass):
            np.random.shuffle(class_id[i])

    # Distribute labels to match the input distribution.
    npos = np.zeros(nclass, np.int64)
    label = []
    print(npos)

    if FLAGS.class_balance:
        c = 0
        for i in range(FLAGS.label_split_size):
            if c > nclass - 1:
                c = 0
            while True:
                try:
                    label.append(class_id[c][npos[c]])
                    npos[c]+=1
                    c+=1
                    break
                except:
                    if c > nclass - 1:
                        c = 0
                    else:
                        c+=1
                    continue
    else:
        for i in range(FLAGS.label_split_size):
            c = np.argmax(train_stats - npos / max(npos.max(), 1))
            label.append(class_id[c][npos[c]])
            npos[c] += 1

    print(npos)
    del npos, class_id

    label = frozenset([int(x) for x in label])

    if 'stl10' in argv[1] and FLAGS.label_split_size == 1000:
        data = open(os.path.join(DATA_DIR, 'stl10_fold_indices.txt'), 'r').read()
        label = frozenset(list(map(int, data.split('\n')[FLAGS.seed].split())))

    print('Creating split in %s' % target)
    npos = np.zeros(nclass, np.int64)
    class_data = [[] for _ in range(nclass)]
    unlabel = []
    unlabel_writes = 0
    label_writes = 0
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with tf.python_io.TFRecordWriter(target + '-label.tfrecord') as writer_label, tf.python_io.TFRecordWriter(
            target + '-unlabel.tfrecord') as writer_unlabel:
        pos, loop = 0, trange(count, desc='Writing records')
        for input_file in input_files:
            for record in tf.python_io.tf_record_iterator(input_file):
                if pos in label:
                    if pos in [0,1,2,3,4]:
                        print()
                    writer_label.write(record)
                    label_writes += 1
                else:
                    try:
                        # 有爆列表的数据忽略
                        class_data[id_class[pos]].append((pos, record))
                    except IndexError:
                        print("skip data",pos)
                        pass
                    while True:
                        c = np.argmax(train_stats - npos / max(npos.max(), 1))
                        if class_data[c]:
                            p, v = class_data[c].pop(0)
                            unlabel.append(p)
                            writer_unlabel.write(v)
                            unlabel_writes += 1
                            npos[c] += 1
                        else:
                            break
                pos += 1
                loop.update()
        for remain in class_data:
            for p, v in remain:
                unlabel.append(p)
                writer_unlabel.write(v)
                unlabel_writes += 1
        loop.close()
        
    print(label_writes, unlabel_writes, str(label_writes + unlabel_writes))
    with open(target + '-map.json', 'w') as writer:
        writer.write(json.dumps(
            dict(label=sorted(label), unlabel=unlabel), indent=2, sort_keys=True))


if __name__ == '__main__':
    utils.setup_tf()
    app.run(main)

