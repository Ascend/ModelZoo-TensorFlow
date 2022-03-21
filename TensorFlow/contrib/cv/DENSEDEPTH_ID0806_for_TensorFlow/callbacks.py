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

from evaluate import ckpt_evaluation
import threading
import os
from npu_bridge.npu_init import *
from tensorflow.keras.callbacks import Callback
from tensorflow.python.keras import backend as K
import tensorflow as tf


def evaluate_threading(test_data, evaluate_ckptdir, bs, logdir, is_distributed):
    ckpt_evaluation(test_data, evaluate_ckptdir, bs, logdir, is_distributed)


def get_nyu_callbacks(test_set=None, runPath=None, checkpoint=None, lr=None, lr_tf=None, is_distributed=False):
    callbacks = []

    class ModelCheckpoint(Callback):
        def __init__(self, checkpoint, savedir, period, test_data, bs, is_distributed):
            super(ModelCheckpoint, self).__init__()
            self.checkpoint = checkpoint
            self.savedir = savedir
            self.period = period
            self.test_data = test_data
            self.bs = bs
            self.is_distributed = is_distributed

        def on_epoch_end(self, epoch, logs=None):
            epoch = epoch + 1
            if epoch % self.period == 0 or epoch > 10:
                graph = tf.get_default_graph()
                sess = K.get_session()
                with sess.as_default():
                    with graph.as_default():
                        # checkpoint_manager = tf.train.CheckpointManager(
                        #     self.checkpoint, directory=self.savedir,
                        #     checkpoint_name="model.callback.epoch{}.ckpt".format(epoch)
                        #     , max_to_keep=15)
                        # checkpoint_manager.save()

                        checkpoint.save(file_prefix=os.path.join(self.savedir, 'epoch{}/model.ckpt'.format(epoch)))
                        print("Epoch {}:save checkpoint...".format(epoch))

                if self.test_data is not None:
                    print("Epoch {}:begin to evaluate checkpoint...".format(epoch))
                    evaluate_ckptdir = os.path.join(self.savedir, 'epoch{}'.format(epoch))
                    t1 = threading.Thread(target=evaluate_threading,
                                          args=(self.test_data, evaluate_ckptdir,
                                                self.bs, self.savedir, self.is_distributed))
                    t1.start()
                    t1.join()

    callbacks.append(ModelCheckpoint(checkpoint=checkpoint, savedir=os.path.join(runPath, "ckpt_npu"), period=5,
                                     test_data=test_set, bs=1, is_distributed=is_distributed))

    class Exponential_Decay(Callback):
        def __init__(self, lr, learning_rate_tf, decay_rate, decay_steps, staircase):
            super(Exponential_Decay, self).__init__()
            self.learning_rate_tf = learning_rate_tf
            self.lr = lr
            self.decay_rate = decay_rate
            self.decay_steps = decay_steps
            self.staircase = staircase
            self.epoch = 0

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch
            sess = K.get_session()
            graph = tf.get_default_graph()
            with sess.as_default():
                with graph.as_default():
                    learning_rate_tf = tf.train.exponential_decay(tf.constant(self.lr, dtype=tf.float32, shape=[]),
                                                                  self.epoch,
                                                                  decay_steps=self.decay_steps,
                                                                  decay_rate=self.decay_rate,
                                                                  staircase=self.staircase)
                    K.set_value(self.learning_rate_tf, K.eval(learning_rate_tf))
                    print('Epoch {}:lr:{}'.format(epoch + 1, K.get_value(learning_rate_tf)))

    callbacks.append(Exponential_Decay(lr, lr_tf, decay_rate=0.90, decay_steps=1,
                                       staircase=False))
    if is_distributed:
        callbacks.append(NPUBroadcastGlobalVariablesCallback(0))

    return callbacks
