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

# epochs = 1
# train_size = 1281167
# #val_size = 50000  #del
# train_batch_size = 70
# #val_batch_size = 50  #del

# TMP_DATA_PATF = './data'
# TMP_MODEL_PATF = './model'
# TMP_LOG_PATH = './log'
# TMP_WEIGHTS_PATH = '/data1/NRE_Check/wx1056345/ID1057_Efficientnet/weights'

#add
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epochs",type=int, default="1")
parser.add_argument("--train_size", type=int, default=1281167)
parser.add_argument("--train_batch_size", type=int, default=70)
parser.add_argument("--TMP_DATA_PATF", type=str, default='./data')
parser.add_argument("--TMP_MODEL_PATF", type=str, default='./model')
parser.add_argument("--TMP_LOG_PATH", type=str, default='./log')
parser.add_argument("--TMP_WEIGHTS_PATH", type=str, default='/data1/NRE_Check/wx1056345/ID1057_Efficientnet/weights')

FLAGS = parser.parse_args()


model = EfficientNetB0((224, 224, 3), include_top=True, weights=None, classes=1000, data_format='channels_last')
# model.summary()
#print("222222",os.path.join(TMP_WEIGHTS_PATH + "/efficientnet-b0.h5"))
model.load_weights(os.path.join(FLAGS.TMP_WEIGHTS_PATH + "/efficientnet-b0.h5"))

train_generators = train_generator(tf_data_list(FLAGS.TMP_DATA_PATF+"/train_tf"), FLAGS.train_batch_size)
#val_generators = val_generator(tf_data_list(TMP_DATA_PATF+"/valid_tf"), val_batch_size) #del
optimizer = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(os.path.join(FLAGS.TMP_MODEL_PATF, 'model_{epoch:03d}.h5'),
                               verbose=1, save_weights_only=False, period=1)
tensorboard = TensorBoard(log_dir=FLAGS.TMP_LOG_PATH + '/', write_graph=True, update_freq='batch')
#tensorboard_print = TensorBoardBatch(model = model ,log_dir=TMP_LOG_PATH)  #add
callbacks = [checkpointer, tensorboard]



model.fit_generator(train_generators,
                    steps_per_epoch=(FLAGS.train_size // FLAGS.train_batch_size),  #1281167/70=18302
                    epochs=FLAGS.epochs,
                    callbacks=callbacks,
                    max_queue_size=2)
