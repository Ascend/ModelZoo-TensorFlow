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

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from dcn.model import DCN
from data_process.criteo import create_criteo_dataset

import os



if __name__ == '__main__':
    # =============================== NPU ==============================
    from tensorflow.python.keras import backend as K
    from npu_bridge.npu_init import *

    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    sess = tf.Session(config=sess_config)
    K.set_session(sess)
    # ========================= Hyper Parameters =======================
    file = './dataset/Criteo/train.txt'
    read_part = True
    sample_num = 5000000
    test_size = 0.2

    embed_dim = 8
    dnn_dropout = 0.5
    dnn_hidden_units = [1024, 1024]
    dcn_layer_number = 6

    learning_rate = 0.0001
    batch_size = 4000
    epochs = 10
    # ========================== Create dataset =======================
    feature_columns, train, test = create_criteo_dataset(file=file,
                                                         embed_dim=embed_dim,
                                                         read_part=read_part,
                                                         sample_num=sample_num,
                                                         test_size=test_size)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    model = DCN(feature_columns,dcn_layer_number, dnn_hidden_units, dnn_dropout=dnn_dropout)
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate, clipnorm=100.),
                    metrics=[AUC()])
    # ============================Restore============================
    model.load_weights('checkpoints/keras/dcn_weights')
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])
    # sess.close()
    # # ===========================Save as Session for offline model==============================
    # session = tensorflow.keras.backend.get_session()
    # saver = tensorflow.train.Saver()
    # saver.save(session,"checkpoints/session/session_model.ckpt")