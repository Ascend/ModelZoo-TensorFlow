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
import os
from keras_efficientnets.efficientnet import EfficientNetB0
from keras.optimizers import SGD
from util.data_load import train_generator, val_generator
from keras.callbacks import ModelCheckpoint, TensorBoard

import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras import backend as K
from npu_bridge.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig



sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess = tf.Session(config=sess_config)
K.set_session(sess)

def tf_data_list(tf_data_path):
    filepath = tf_data_path
    tf_data_list = []
    file_list = os.listdir(filepath)
    for i in file_list:
        tf_data_list.append(os.path.join(filepath,i))
    print("-----------------------------------------------------")
    print(tf_data_list)
    return tf_data_list

TMP_DATA_PATF = './data'
TMP_MODEL_PATF = './model'
TMP_LOG_PATH = './log'


val_size = 50000
val_batch_size = 50
val_generators = val_generator(tf_data_list(TMP_DATA_PATF+"/valid_tf"), val_batch_size)

#
model = EfficientNetB0((224,224,3), include_top=True, weights=None, classes=1000, data_format='channels_last')
# model.summary()
model.load_weights(TMP_MODEL_PATF + "/model_001.h5")
optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
loss, acc =model.evaluate_generator(val_generators, steps=1000)
print("loss:  %f     acc:  %f"%(loss, acc))

