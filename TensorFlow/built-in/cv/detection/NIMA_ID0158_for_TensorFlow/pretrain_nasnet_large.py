#
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
#
import os

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras import backend as K

from utils.data_loader import features_generator

'''
Below is a modification to the TensorBoard callback to perform 
batchwise writing to the tensorboard, instead of only at the end
of the batch.
'''
class TensorBoardBatch(TensorBoard):
    def __init__(self, *args, **kwargs):
        super(TensorBoardBatch, self).__init__(*args, **kwargs)

        # conditionally import tensorflow iff TensorBoardBatch is created
        self.tf = __import__('tensorflow')

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, batch)

        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch * self.batch_size)

        self.writer.flush()

def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)

NUM_FEATURES = 1056

image_size = 224
ip = Input(shape=(1056,))
x = Dropout(0.75)(ip)
x = Dense(10, activation='softmax')(x)

model = Model(ip, x)
model.summary()
optimizer = Adam(lr=1e-4)
model.compile(optimizer, loss=earth_mover_loss)

# load weights from trained model if it exists
if os.path.exists('weights/nasnet_large_pretrained_weights.h5'):
    model.load_weights('weights/nasnet_large_pretrained_weights.h5')

checkpoint = ModelCheckpoint('weights/nasnet_large_pretrained_weights.h5', monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True,
                             mode='min')
tensorboard = TensorBoardBatch(log_dir='./nasnet_logs/')
callbacks = [checkpoint, tensorboard]

batchsize = 200
epochs = 20

TRAIN_RECORD_PATH = 'weights/nasnet_large_train.tfrecord'
VAL_RECORD_PATH = 'weights/nasnet_large_val.tfrecord'

model.fit_generator(features_generator(TRAIN_RECORD_PATH, NUM_FEATURES, batchsize=batchsize, shuffle=True),
                    steps_per_epoch=(500000. // batchsize),
                    epochs=epochs, verbose=1, callbacks=callbacks,
                    validation_data=features_generator(VAL_RECORD_PATH, NUM_FEATURES, batchsize=batchsize, shuffle=False),
                    validation_steps=(5000. // batchsize))
