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
from read_data import *
import numpy as np
import tensorflow as tf
import os
from RCNN import *
import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--snapshot_name', required=True)
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()

snapshot_name = args.snapshot_name
tst_image_set = args.test_split

gpu_id = args.gpu_id  # set GPU id to use
import os; os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test():
    log_dir = './logdir'
    snapshot_interval = 10000
    snapshot_dir = './snapshot_dir'
    max_iter = 100000
    log_interval = 100

    lr = 0.0005

    file = 'data/test_32x32.mat'
    X_raw, y_raw = getData(filename=file)
    n_test = X_raw.shape[0]
    y_raw[y_raw == 10] = 0
    y_raw = np.reshape(y_raw, (n_test,))

    snapshot_file = './snapshot_dir/%s' % (snapshot_name)

    with tf.Session(config=npu_config_proto()) as sess:
        X = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        y = tf.placeholder(tf.int32, shape=(None,))
        rcnn = RCNN(time=3, K=192, p=0.9, numclass=10, is_training=True)
        _, _, _, preds = rcnn.buile_model(X, y)

        snapshot_saver = tf.train.Saver(max_to_keep=None)  # keep all snapshots
        snapshot_saver.restore(sess, snapshot_file)

        np.random.seed(0)
        count = 0
        start = datetime.datetime.now()
        for i in range(n_test):
            image = X_raw[i]
            labels = y_raw[i]
            # print(image.shape)
            preds_each = sess.run(preds, feed_dict={X:image})
            if preds_each == labels:
                count += 1
            else:
                continue
    acc = count/(n_test*1.0)
    end = datetime.datetime.now()
    print("sum time: {}, acc: {}".format(end - start, acc))

if __name__ == '__main__':
    test()
