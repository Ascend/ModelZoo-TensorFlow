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

import tensorflow as tf
tf.enable_eager_execution()

import os
from pathlib import Path
from random import shuffle
from data.read_tfrecord import create_tiny_image_dataset, read_test_dataset
from tensorflow.keras.callbacks import ModelCheckpoint

# from models.squeezenet import SqueezeNet 


from model import SqueezeNet
from config import _C as cfg
from tensorflow.keras.models import load_model
import shutil

tf.logging.set_verbosity(tf.logging.ERROR)

class CopyCheckPoint(tf.keras.callbacks.Callback):
      def on_train_batch_end(self, batch, logs=None):
          shutil.copyfile('./ckpt/squeeze_best.h5', './ckpt_cpy/squeeze_best.h5')



training_dataset = read_test_dataset(
    [str(x) for x in Path(cfg.TRAINING_FILENAMES).rglob('*.tfrecords')] + \
    [str(x) for x in Path(cfg.VALIDATION_FILENAMES).rglob('*.tfrecords')],
    batch_size=cfg.BATCH_SIZE)


test_dataset = read_test_dataset([str(x) for x in Path(cfg.TEST_FILENAMES).rglob('*.tfrecords')], batch_size=cfg.BATCH_SIZE)


top_5_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1", "GPU:2"])
with strategy.scope():
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-1,
    decay_steps=10000,
    decay_rate=0.9)
    Squeezenet = SqueezeNet(200)
    Squeezenet.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.09),
        loss='categorical_crossentropy',
        metrics=['accuracy', top_5_acc],
    )
    # Squeezenet.load_weights('./ckpt/squeeze_imagenet_best.h5')




# checkpoint = ModelCheckpoint(filepath='./ckpt/squeeze_imagenet_best.h5',monitor='val_acc',mode='auto' ,save_best_only='False')
# cpy_ckpt = CopyCheckPoint()
# callbacks_list = [checkpoint, cpy_ckpt]

# Squeezenet.summary()


# Squeezenet.fit(training_dataset, steps_per_epoch=cfg.STEPS_PER_EPOCH, epochs=cfg.EPOCHS, validation_data=test_dataset,validation_steps=cfg.VALIDATION_STEPS,callbacks=callbacks_list)

# Squeezenet.evaluate(test_dataset)


input_data = training_dataset.take(1)
test_input_data = test_dataset.take(1)


for (img, label) in input_data:
    print(img)
    print(label)

for (img, label) in test_input_data:
    print(img)
    print(label)


