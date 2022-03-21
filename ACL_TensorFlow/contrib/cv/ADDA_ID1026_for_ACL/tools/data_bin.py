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
def main(dataset, split, model):
    dataset_name = dataset
    split_name = split
    dataset = adda.data.get_dataset(dataset, shuffle=False)
    split = getattr(dataset, split)
    model_fn = adda.models.get_model_fn(model)
    im, label = split.tf_ops(capacity=32)
    im = adda.models.preprocessing(im, model_fn)
    im_batch, label_batch = tf.train.batch([im, label], batch_size=1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if not os.path.exists('mnist2000bin_im'):  # bin数据保存路径，自行替换
        os.makedirs('mnist2000bin_im')
    if not os.path.exists('mnist2000bin_label'):  # bin数据保存路径，自行替换
        os.makedirs('mnist2000bin_label')
    for i in tqdm(range(len(split))):
        im_batch_x, label_batch_y = sess.run([im_batch, label_batch])
        im_batch_x.astype(np.float32).tofile(f"mnist2000bin_im/{i}.bin")
        label_batch_y.astype(np.float32).tofile(f"mnist2000bin_label/{i}.bin")

    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    main()