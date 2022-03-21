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
# Copyright 2020 Huawei Technologies Co., Ltd
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

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from util.dataset import get_dataset
from random import shuffle
from pathlib import Path
import tensorflow as tf
from config import config
import os
import math
import shutil
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from models.squeezenet import SqueezeNet

tf.keras.backend.set_learning_phase(True)


tf.logging.set_verbosity(tf.logging.ERROR)


class CopyCheckPoint(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        pass


class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        tf.summary.scalar('lr', self.model.optimizer.lr)
        super().on_batch_end(batch, logs)


DATASET_DIR = '/media/data1/haoyiqing/imagenet_tfrecords'

train_dataset = get_dataset(DATASET_DIR, 'train', batch_size=config.batch_size)
val_dataset = get_dataset(DATASET_DIR, 'validation',
                          batch_size=config.batch_size)


def step_decay(epoch):
    # set lr
    initial_lrate = config.lr
    drop = 0.83
    epochs_drop = 500
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    if lrate < 1e-9:
        return 1e-9
    return lrate


top_5_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=5)


strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1", "GPU:2"])
with strategy.scope():

    Squeezenet = SqueezeNet(1000)

    Squeezenet.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=1e-3, momentum=0.09),
        loss='categorical_crossentropy',
        metrics=['accuracy', top_5_acc],
    )

# Squeezenet.load_weights('./ckpt/squeeze_imagenet_2__best.h5')


checkpoint = ModelCheckpoint(filepath='./ckpt/squeeze_imagenet_2__best.h5',
                             monitor='val_acc', mode='auto', save_best_only='True')

cpy_ckpt_cb = CopyCheckPoint()
lr_cb = callback = tf.keras.callbacks.LearningRateScheduler(step_decay)
tensorbord_cb = LRTensorBoard(
    log_dir='./logs', histogram_freq=1, update_freq=1)

callbacks_list = [checkpoint, cpy_ckpt_cb, tensorbord_cb, lr_cb]

Squeezenet.fit(train_dataset, steps_per_epoch=1281167 // config.batch_size, epochs=config.epoch,
               validation_data=val_dataset, validation_steps=50000 // config.batch_size, callbacks=callbacks_list)

Squeezenet.evaluate(val_dataset)
Squeezenet.save("./ckpt/epoch_60.h5")
