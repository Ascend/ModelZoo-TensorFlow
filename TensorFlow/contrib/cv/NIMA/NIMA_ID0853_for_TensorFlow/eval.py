
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

from keras import backend as K
from utils.data_loader import train_generator, val_generator
from utils.score_utils import srcc
import keras
import numpy as np
from tqdm import tqdm

from keras.utils.generic_utils import CustomObjectScope

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

'''
Below is a modification to the TensorBoard callback to perform 
batchwise writing to the tensorboard, instead of only at the end
of the batch.
'''


def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)



def calc_srcc(model, gen, test_size, batch_size):
    y_test = []
    y_pred = []

    for i in tqdm(range(test_size // batch_size)):
        batch = next(gen)
        y_test.append(batch[1])
        y_pred.append(model.predict_on_batch(batch[0]))
    y_test = np.concatenate(y_test)
    y_pred = np.concatenate(y_pred)
    rho = srcc(y_test, y_pred)
    print("----------------------",y_test.shape)
    print("----------------------",y_pred.shape)
    print("srcc = {}".format(rho))
    return rho


image_size = 224
batchsize = 10
val_size = 5000
TMP_MODEL_PATF = './model/'


# with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
model = keras.models.load_model(TMP_MODEL_PATF+"model_007.h5", custom_objects={'earth_mover_loss': earth_mover_loss})

val_generators = val_generator(batchsize=batchsize)
val_generators_srcc = val_generator(batchsize=batchsize)

loss = model.evaluate_generator(val_generators,
                    steps=(val_size // batchsize),
                    max_queue_size=2,)

print("calculating spearman's rank correlation coefficient")
rho = calc_srcc(model=model, gen=val_generators_srcc, test_size=5000,
        batch_size=batchsize)
print("loss  %f     srcc %f    "%(loss,rho))

