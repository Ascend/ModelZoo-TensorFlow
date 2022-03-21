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
"""
This is the boot file for ModelArts platform.
"""
from npu_bridge.npu_init import *

import numpy as np
import tensorflow as tf
import argparse
#import moxing as mox

from utils import data_generator
from tcn import compiled_tcn, tcn_full_summary


def parse_args():
    """Add some parameters"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--train_data_n', type=int, default=200000, help='number of train data')
    parser.add_argument('--validation_data_n', type=int, default=40000, help='number of validation data')
    parser.add_argument('--seq_length', type=int, default=600, help='sequence length')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--tensorboard_path', default='/cache/tfboard1', type=str, help='tensorboard path')
    parser.add_argument('--h5_obs_path', default='obs://kyq/mytestnew/mytestnew1h5/tcn.h5', type=str,
                        help='h5 obs path')
    parser.add_argument('--tensorboard_obs_path', default='obs://kyq/mytestnew/tfboard1', type=str,
                        help='tensorboard obs path')
    args, unknown_args = parser.parse_known_args()
    return args


args = parse_args()
x_train, y_train = data_generator(n=args.train_data_n, seq_length=args.seq_length)
x_test, y_test = data_generator(n=args.validation_data_n, seq_length=args.seq_length)


class PrintSomeValues(tf.keras.callbacks.Callback):
    """Define a callback function"""
    def on_epoch_begin(self, epoch, logs=None):
        """Print some information"""
        print('y_true, y_pred')
        print(np.hstack([y_test[:5], self.model.predict(x_test[:5])]))


def run_task():
    """Run the task: the adding problem"""
    model = compiled_tcn(
        return_sequences=False,
        num_feat=x_train.shape[2],
        num_classes=0,
        nb_filters=24,
        kernel_size=8,
        dilations=[2 ** i for i in range(9)],
        nb_stacks=1,
        max_len=x_train.shape[1],
        use_skip_connections=False,
        use_batch_norm=True,
        regression=True,
        dropout_rate=args.dropout_rate,
        lr=args.lr
    )

    tcn_full_summary(model)

    batch_size = args.batch_size
    steps_per_epoch = int(np.ceil(x_train.shape[0] / batch_size))
    test_batch_size = int(np.ceil(x_test.shape[0] / batch_size))

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.batch(steps_per_epoch, drop_remainder=True).repeat()
    train_it = train_ds.make_one_shot_iterator()
    x_train_it, y_train_it = train_it.get_next()

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(test_batch_size, drop_remainder=True).repeat()
    test_it = test_ds.make_one_shot_iterator()
    x_test_it, y_test_it = test_it.get_next()

    tbCallBack = tf.keras.callbacks.TensorBoard(write_graph=False, log_dir=args.tensorboard_path)

    with tf.Session(config=npu_config_proto()) as sess:
        x_train_it, y_train_it = sess.run([x_train_it, y_train_it])
        x_test_it, y_test_it = sess.run([x_test_it, y_test_it])
        model.fit(x_train_it, y_train_it, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
                  validation_data=(x_test_it, y_test_it), validation_steps=test_batch_size,
                  callbacks=[PrintSomeValues(), tbCallBack], verbose=1)
        model.save('./tcn.h5')


if __name__ == '__main__':
    npu_keras_sess = set_keras_session_npu_config()
    run_task()
    close_session(npu_keras_sess)

   # mox.file.copy_parallel(args.tensorboard_path, args.tensorboard_obs_path)
   # mox.file.copy('./tcn.h5', args.h5_obs_path)
