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
from scipy import misc
import numpy as np
# import mxnet as mx
import argparse
import pickle
import cv2
import os
from npu_bridge.npu_init import *
from npu_bridge.estimator.npu import npu_scope

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='data path information'
    )
    parser.add_argument('--bin_path', default='../datasets/faces_ms1m_112x112/train.rec', type=str,
                        help='path to the binary image file')
    parser.add_argument('--idx_path', default='../datasets/faces_ms1m_112x112/train.idx', type=str,
                        help='path to the image index path')
    parser.add_argument('--tfrecords_file_path', default='../datasets/tfrecords', type=str,
                        help='path to the output of tfrecords file path')
    args = parser.parse_args()
    return args


def mx2tfrecords(imgidx, imgrec, args):
    output_path = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
    if not os.path.exists(args.tfrecords_file_path):
        os.makedirs(args.tfrecords_file_path)
    writer = tf.python_io.TFRecordWriter(output_path)
    for i in imgidx:
        img_info = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(img_info)
        label = int(header.label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())  # Serialize To String
        if i % 10000 == 0:
            print('%d num image processed' % i)
    writer.close()

def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')


def parse_function(example_proto):
    features = {'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example_proto, features)
    # You can do more image distortion here for training data
    img = tf.image.decode_jpeg(features['image_raw'])
    img = tf.reshape(img, shape=(112, 112, 3))

    # img = tf.py_func(random_rotate_image, [img], tf.uint8)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.subtract(img, 127.5)
    img = tf.multiply(img,  0.0078125)
    img = tf.image.random_flip_left_right(img)
    label = tf.cast(features['label'], tf.int64)
    return img, label


def create_tfrecords():
    '''convert mxnet data to tfrecords.'''
    id2range = {}
    args = parse_args()

    imgrec = mx.recordio.MXIndexedRecordIO(args.idx_path, args.bin_path, 'r')
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    print(header.label)
    imgidx = list(range(1, int(header.label[0])))
    seq_identity = range(int(header.label[0]), int(header.label[1]))
    for identity in seq_identity:
        s = imgrec.read_idx(identity)
        header, _ = mx.recordio.unpack(s)
        a, b = int(header.label[0]), int(header.label[1])
        id2range[identity] = (a, b)
    print('id2range', len(id2range))
    # print('Number of examples in training set: {}'.format(imgidx))

    # generate tfrecords
    mx2tfrecords(imgidx, imgrec, args)


def load_bin(db_name, image_size, args):
    bins, issame_list = pickle.load(open(os.path.join(args.eval_db_path, db_name+'.bin'), 'rb'), encoding='bytes')
    data_list = []
    for _ in [0,1]:
        data = np.empty((len(issame_list)*2, image_size[0], image_size[1], 3))
        data_list.append(data)
    for i in range(len(issame_list)*2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for flip in [0, 1]:
            if flip == 1:
                img = np.fliplr(img)
            data_list[flip][i, ...] = img
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data_list[0].shape)

    return data_list, issame_list


def load_data(db_name, image_size, args):
    # python3
    bins, issame_list = pickle.load(open(os.path.join(args.eval_db_path, db_name+'.bin'), 'rb'), encoding='bytes')
    # python2
    # bins, issame_list = pickle.load(open(os.path.join(args.eval_db_path, db_name+'.bin'), 'rb'))
    datasets = np.empty((len(issame_list)*2, image_size[0], image_size[1], 3))

    for i in range(len(issame_list)*2):
        _bin = bins[i]
#         img = mx.image.imdecode(_bin).asnumpy()
        
        nparr = np.fromstring(_bin,np.uint8)
        img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img - 127.5
        img = img * 0.0078125
#         print(img)
#         print('writing imgae 1')
#         cv2.imwrite('1.jpg', img)
#         break;
        datasets[i, ...] = img
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(datasets.shape)

    return datasets, issame_list


def test_tfrecords():
    args = parse_args()

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=npu_config_proto(config_proto=config))
    # training datasets api config
    tfrecords_f = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
    dataset = tf.data.TFRecordDataset(tfrecords_f)
    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(buffer_size=20000)
    dataset = dataset.batch(32, drop_remainder=True)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    # begin iteration
    for i in range(1000):
        sess.run(iterator.initializer)
        while True:
            try:
                images, labels = sess.run(next_element)
                cv2.imshow('test', images[1, ...])
                cv2.waitKey(0)
            except tf.errors.OutOfRangeError:
                print("End of dataset")

# def parse_exmp(serial_exmp):
#     feats = tf.parse_single_example(serial_exmp, features={'feature':tf.FixedLenFeature([], tf.string),\
#         'label':tf.FixedLenFeature([],tf.int64), 'shape':tf.FixedLenFeature([], tf.int64)})
    
#     img = tf.decode_raw(feats['feature'], tf.float32)
#     img = tf.reshape(img, shape=(112, 112, 3))
#     img = tf.cast(img, dtype=tf.float32)
#     img = tf.subtract(img, 127.5)
#     img = tf.multiply(img, 0.0078125)
#     img = tf.image.random_flip_left_right(img)
#     label = tf.cast(feats['label'], tf.int64)

#     shape = tf.cast(feats['shape'], tf.int32)
#     return img, label, shape


# def get_dataset(fname):
#     dataset = tf.data.TFRecordDataset(fname)
#     return dataset.map(parse_exmp)

def next_batch(batch_size, path):
    # args = parse_args()
    reader = tf.TFRecordReader()
    filename_tensorlist = tf.train.match_filenames_once(path)
    filename_queue = tf.train.string_input_producer(filename_tensorlist)
    _, serialized_example = reader.read(filename_queue)

    features = {'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}

    features = tf.parse_single_example(serialized=serialized_example,
                                       features=features)
    
    img = tf.image.decode_jpeg(features['image_raw'])
    img = tf.reshape(img, shape=(112, 112, 3))

    # img = tf.py_func(random_rotate_image, [img], tf.uint8)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.subtract(img, 127.5)
    img = tf.multiply(img, 0.0078125)
    img = tf.image.random_flip_left_right(img)
    label = tf.cast(features['label'], tf.int64)
   
    img_batch, label_batch = \
        tf.train.shuffle_batch(
            [img, label],
            batch_size=batch_size,
            capacity=40000,
            num_threads=16,
            min_after_dequeue=20000)

    return img_batch, label_batch
    


if __name__ == '__main__':
    '''data process'''
#     load_data('lfw', [112,112], './datasets/faces_ms1m_112x112')
    create_tfrecords()
    test_tfrecords()

    args = parse_args()
        
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    path = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
    img_batch, label_batch = next_batch(batch_size=32, path=path)

#     init_op = tf.group(
#         tf.global_variables_initializer(),
#         tf.local_variables_initializer()
#     )
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap
    # config.gpu_options.allow_growth = True

    with tf.Session(config=npu_config_proto(config_proto=config)) as sess:
#         sess.run(init_op)
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        img_batch_, label_batch_ \
            = sess.run([img_batch, label_batch])

        # cv2.imshow('test', img_batch_[1, ...])
        cv2.imwrite('./test.jpg', img_batch_[1, ...])
        # cv2.waitKey(0)

        print(img_batch_)

        print('debug')

        coord.request_stop()
        coord.join(threads)















