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
import numpy as np
import tensorflow as tf

from util import log
import datasets.hdf5_loader as dataset

def check_data_id(dataset, data_id):
    if not data_id:
        return

    wrong = []
    for id in data_id:
        if id in dataset.data:
            pass
        else:
            wrong.append(id)

    if len(wrong) > 0:
        raise RuntimeError("There are %d invalid ids, including %s" % (
            len(wrong), wrong[:5]
        ))


def create_input_ops(dataset,
                     batch_size,
                     num_threads=16,           # for creating batches
                     is_training=False,
                     data_id=None,
                     scope='inputs',
                     shuffle=True,
                     ):
    '''
    Return a batched tensor for the inputs from the dataset.
    '''
    input_ops = {}

    if data_id is None:
        data_id = dataset.ids
        log.info("input_ops [%s]: Using %d IDs from dataset", scope, len(data_id))
    else:
        log.info("input_ops [%s]: Using specified %d IDs", scope, len(data_id))

    # single operations
    with tf.device('/cpu:0'), tf.name_scope(scope):
        input_ops['id'] = tf.train.string_input_producer(
           tf.convert_to_tensor(data_id),
            capacity=128
        ).dequeue(name='input_ids_dequeue')

        m, label = dataset.get_data(data_id[0])

        def load_fn(id):
            # image [n, n], label: [m]
            image, label = dataset.get_data(id)
            return (id,
                    image.astype(np.float32),
                    label.astype(np.float32))

        input_ops['id'], input_ops['image'], input_ops['label'] = tf.py_func(
            load_fn, inp=[input_ops['id']],
            Tout=[tf.string, tf.float32, tf.float32],
            name='func_hp'
        )
        
        input_ops['id'].set_shape([])
        input_ops['image'].set_shape(list(m.shape))
        input_ops['label'].set_shape(list(label.shape))

    # batchify
    capacity = 2 * batch_size * num_threads
    min_capacity = min(int(capacity * 0.75), 1024)

    if shuffle:
        batch_ops = tf.train.shuffle_batch(
            input_ops,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity,
            min_after_dequeue=min_capacity,
        )
    else:
        batch_ops = tf.train.batch(
            input_ops,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity,
        )

    return input_ops, batch_ops


###### MY_CREATE_INPUT_OPS

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def hdf2tfrecord(out_tfrecord_path, is_train=True):
    print("GENERATING TFRECORD...")
    # dataset_path = os.path.join('./datasets', "MNIST".lower())
    dataset_path = "/cache/data/mnist"
    train_set, test_set = dataset.create_default_splits(dataset_path)
    if is_train:
        choosen_dataset = train_set
    else:
        choosen_dataset = test_set
    # DEBUG USE
    # img, label = train_set.get_data(train_set.ids[0])
    # print(img,type(img),img.shape) # <class 'numpy.ndarray'> (28, 28, 1)
    # print(label,type(label)) # <class 'numpy.ndarray'>
    # print(train_set.ids[0],type(train_set.ids[0])) # <class 'str'>
    # input()
    tf_write = tf.python_io.TFRecordWriter(out_tfrecord_path)
    for id in train_set.ids:
        img, label = train_set.get_data(id)
        # img = img.tobytes()
        # img = tf.io.serialize_tensor(img)
        # label = label.tobytes()
        # label = tf.io.serialize_tensor(label)

        # DEBUG USE
        # print(img,type(img))
        # input()
        id = str.encode(id)
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'image': float_list_feature(img.flatten()),
                'label': float_list_feature(label.flatten()),
                'id': bytes_feature(id)
            }
            )
        )
        # DEBUG USE
        # print(example)
        # input()
        tf_write.write(example.SerializeToString())
    print("TFRECORD GENERATED!!!")


def parse_record(example):
    features = {
        'image': tf.FixedLenFeature((28, 28, 1), tf.float32),
        'label': tf.FixedLenFeature((10,), tf.float32),
        'id': tf.FixedLenFeature((), tf.string, default_value=''),
    }
    parsed = tf.parse_single_example(example, features=features)
    # image = tf.decode_raw(parsed['image'], out_type=tf.uint8)
    # image = tf.reshape(image, shape=(28,28,1))
    # image = tf.cast(image, tf.float32)
    image = parsed['image']

    # print(image)
    # input()
    # label = tf.decode_raw(parsed['label'], out_type=tf.uint8)
    # label = tf.reshape(label, shape=(10,))
    # label = tf.cast(label, tf.float32)
    label = parsed['label']
    id = parsed['id']
    # DEBUG USE
    # print(image,type(image)) #Tensor("DecodeRaw:0", shape=(?,), dtype=float32) <class 'tensorflow.python.framework.ops.Tensor'>
    # print(label,type(label))
    # print(id,type(id))
    # input()

    return {'image': image, 'label': label, 'id': id}


def read_input(in_tfrecord_path):
    batch_size = 32
    input_ops = {}
    batch_ops = {}
    with tf.device('/cpu:0'):
        dataset = tf.data.TFRecordDataset([in_tfrecord_path]).repeat()  # 数据读取
        dataset = dataset.map(lambda value: parse_record(value), num_parallel_calls=64)  # 数据预处理
        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(64)  # 取batch并进行预取操作
        iterator = dataset.make_initializable_iterator()  # 初始化迭代器
        batch_ops = iterator.get_next()  # 从迭代器中取出数据，getnext输出tensor的顺序取决于map函数的输出

    # batch_ops: {'image': <tf.Tensor 'shuffle_batch:1' shape=(32, 28, 28, 1) dtype=float32>, 'label': <tf.Tensor 'shuffle_batch:2' shape=(32, 10) dtype=float32>, 'id': <tf.Tensor 'shuffle_batch:0' shape=(32,) dtype=string>}
    return input_ops, batch_ops


def my_create_input_ops(in_tfrecord_path, batch_size):
    '''
    Return a batched tensor for the inputs from the dataset.
    '''

    input_ops = {}
    batch_ops = {}
    iterator = None
    with tf.device('/cpu:0'):
        dataset = tf.data.TFRecordDataset([in_tfrecord_path]).repeat()  # 数据读取
        dataset = dataset.map(lambda value: parse_record(value), num_parallel_calls=64)  # 数据预处理
        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(64)  # 取batch并进行预取操作
        iterator = dataset.make_initializable_iterator()  # 初始化迭代器
        batch_ops = iterator.get_next()  # 从迭代器中取出数据，getnext输出tensor的顺序取决于map函数的输出

    # batch_ops: {'image': <tf.Tensor 'shuffle_batch:1' shape=(32, 28, 28, 1) dtype=float32>, 'label': <tf.Tensor 'shuffle_batch:2' shape=(32, 10) dtype=float32>, 'id': <tf.Tensor 'shuffle_batch:0' shape=(32,) dtype=string>}

    return input_ops, batch_ops, iterator


