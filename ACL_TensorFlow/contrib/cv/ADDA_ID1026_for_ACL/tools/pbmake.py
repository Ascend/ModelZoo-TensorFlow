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
import logging
import os
import time
from collections import OrderedDict

import click
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sys

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
import adda
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.tools import freeze_graph


def format_array(arr):
    return '  '.join(['{:.3f}'.format(x) for x in arr])


@click.command()
@click.argument('dataset')
@click.argument('split')
@click.argument('model')
@click.argument('weights')
@click.option('--gpu', default='0')
def main(dataset, split, model, weights, gpu):
    adda.util.config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

    dataset_name = dataset
    split_name = split
    dataset = adda.data.get_dataset(dataset, shuffle=False)
    split = getattr(dataset, split)
    model_fn = adda.models.get_model_fn(model)
    im, label = split.tf_ops(capacity=2)
    im = adda.models.preprocessing(im, model_fn)
    im_batch, label_batch = tf.train.batch([im, label], batch_size=1)

    x = tf.placeholder(tf.float32, shape=[1, 28, 28, 1], name='input_image')
    y = tf.placeholder(tf.float32, shape=[1], name='input_label')
    net, layers = model_fn(x, is_training=False)
    net = tf.argmax(net, -1, name='output')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())
    var_dict = adda.util.collect_vars(model)
    restorer = tf.train.Saver(var_list=var_dict)
    if os.path.isdir(weights):
        weights = tf.train.latest_checkpoint(weights)
    logging.info('Evaluating {}'.format(weights))
    restorer.restore(sess, weights)

    class_correct = np.zeros(dataset.num_classes, dtype=np.int32)
    class_counts = np.zeros(dataset.num_classes, dtype=np.int32)
    for i in tqdm(range(len(split))):
        im_batch_x, label_batch_y = sess.run([im_batch, label_batch])
        predictions = sess.run(net, feed_dict={x: im_batch_x, y: label_batch_y})
        gt = label_batch_y
        class_counts[gt[0]] += 1
        if predictions[0] == gt[0]:
            class_correct[gt[0]] += 1
    logging.info('Class accuracies:')
    logging.info('    ' + format_array(class_correct / class_counts))
    logging.info('Overall accuracy:')
    logging.info('    ' + str(np.sum(class_correct) / np.sum(class_counts)))

    coord.request_stop()
    coord.join(threads)
    graph_def = tf.get_default_graph()
    graph_def = sess.graph_def
    output_graph_def = graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=graph_def,
        output_node_names=['output'])
    with tf.gfile.GFile('./pb_model/svhn_mnist.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))
    print('done')
    sess.close()


if __name__ == '__main__':
    main()