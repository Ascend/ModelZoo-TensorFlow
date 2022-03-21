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

import os
import math
import shutil
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.estimator import npu_ops

sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess = tf.Session(config=sess_config)
K.set_session(sess)


# os.environ['PRINT_MODEL'] = '1'
# os.environ['SLOG_PRINT_TO_STDOUT'] = "1"
# os.environ['DUMP_GE_GRAPH'] = "2"
# os.environ['DUMP_GRAPH_LEVEL'] = "3"
# os.environ['ENABLE_NETWORK_ANALYSIS_DEBUG'] = "1"
# os.environ['EXPERIMENTAL_DYNAMIC_PARTITION'] = "1"
# os.environ['GE_USE_STATIC_MEMORY'] = "1"

from utils.dataset import get_dataset
from config import _C as cfg

# # model arts 
#import moxing as mox

os.system('pip install easydict')

from models.squeezenet import SqueezeNet as Squeezenet


train_dataset = get_dataset(cfg.CACHE_DATA_FILENAMES_CACHE_PATH, 'train', batch_size=cfg.BATCH_SIZE)
val_dataset = get_dataset(cfg.CACHE_DATA_FILENAMES_CACHE_PATH, 'validation',batch_size=cfg.BATCH_SIZE)


def step_decay(epoch):
    # set lr
    initial_lrate = cfg.LR
    drop = 0.83
    epochs_drop = 500
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    if lrate < 1e-5:
        return 1e-5
    return lrate


top_5_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
Squeezenet = Squeezenet()
Squeezenet.compile(
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=1e-3, momentum=0.09),
    loss='categorical_crossentropy',
    metrics=['accuracy', top_5_acc],
)

checkpoint = ModelCheckpoint(filepath=cfg.TRAINING_CKPT_FILE_DIR_NAMES+'squeeze_imagenet_2__best.h5',
                             monitor='val_acc', mode='auto', save_best_only='True')

#cpy_ckpt_cb = CopyCheckPoint()
lr_cb = callback = tf.keras.callbacks.LearningRateScheduler(step_decay)

callbacks_list = []

Squeezenet.summary()

print("start training!")

Squeezenet.fit(train_dataset, steps_per_epoch=6400 // cfg.BATCH_SIZE, epochs=cfg.EPOCH,
               validation_data=val_dataset, validation_steps=6400 // cfg.BATCH_SIZE, callbacks=callbacks_list)

#Squeezenet.evaluate(val_dataset)
Squeezenet.save("./ckpt/epoch_80.h5")
