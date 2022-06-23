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

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K

from utils.data_loader import train_generator, val_generator
from npu_bridge.npu_init import *
from npu_bridge.estimator.npu import npu_convert_dropout
'''
Below is a modification to the TensorBoard callback to perform 
batchwise writing to the tensorboard, instead of only at the end
of the batch.
'''
class TensorBoardBatch(TensorBoard):
    def __init__(self, *args, **kwargs):
        super(TensorBoardBatch, self).__init__(*args)

        # conditionally import tensorflow iff TensorBoardBatch is created
        self.tf = __import__('tensorflow')
        self.writer = self.tf.summary.FileWriter(logdir="logs")

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
            #self.writer.add_summary(summary, epoch * self.batch_size)

        self.writer.flush()

def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)

image_size = 224

base_model = MobileNet((image_size, image_size, 3), alpha=1, include_top=False, pooling='avg')
for layer in base_model.layers:
    layer.trainable = False

x = Dropout(0.75)(base_model.output)
x = Dense(10, activation='softmax')(x)

model = Model(base_model.input, x)
model.summary()
optimizer = Adam(lr=1e-3)
model.compile(optimizer, loss=earth_mover_loss)

# load weights from trained model if it exists
if os.path.exists('weights/mobilenet_weights.h5'):
    model.load_weights('weights/mobilenet_weights.h5')

checkpoint = ModelCheckpoint('weights/mobilenet_weights.h5', monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True,
                             mode='min')
tensorboard = TensorBoardBatch()
callbacks = [checkpoint, tensorboard]

batchsize = 200
epochs = 1
###### add for run on npu #####
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.estimator import npu_ops
import tensorflow as tf
sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess = tf.Session(config=sess_config)
K.set_session(sess)
################################
train_generator = train_generator(batchsize=batchsize)
validation_generator = val_generator(batchsize=batchsize)
model.fit_generator(train_generator,
                    steps_per_epoch=(250000. // batchsize),
                    epochs=epochs, verbose=1, callbacks=callbacks,
                    validation_data=validation_generator,
                    validation_steps=(5000. // batchsize))
###### add for run on npu #####
sess.close()
################################
