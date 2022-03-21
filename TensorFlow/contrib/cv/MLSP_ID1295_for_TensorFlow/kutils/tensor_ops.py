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
from npu_bridge.npu_init import *
import tensorflow as tf
# import tensorflow.python.keras as keras
# from tensorflow.python.keras import backend as K
# from keras.engine.topology import Layer
from keras import backend as K
from keras.engine.topology import Layer
import os
# tf = K.tf

def Startnpu():
    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.parameter_map["dynamic_input"].b = True
    custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
    custom_op.name = "NpuOptimizer"
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    sess = tf.Session(config=sess_config)
    K.set_session(sess)

def endnpu():
    sess_config = tf.ConfigProto()
    sess = tf.Session(config=sess_config)
    sess.close()


# Keras configuration directives

# def SetActiveGPU(number=0):
#     """
#     Set visibility of GPUs to the Tensorflow engine.
#
#     :param number: scalar or list of GPU indices
#                    e.g. 0 for the 1st GPU, or [0,2] for the 1st and 3rd GPU
#     """
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     if not isinstance(number,list): number=[number]
#     os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(map(str,number))
#     print('Visible GPU(s):', os.environ["CUDA_VISIBLE_DEVICES"])

# def GPUMemoryCap(fraction=1):
#     """
#     Limit the amount of GPU memory that can be used by an active kernel.
#
#     :param fraction: in [0, 1], 1 = the entire available GPU memory.
#     """
#     config = tf.ConfigProto()
#     config.gpu_options.per_process_gpu_memory_fraction = fraction
#     K.set_session(K.tf.Session(config=npu_config_proto(config_proto=config)))


# Metrics and losses
    
def plcc_tf(x, y):
    """PLCC metric"""
    xc = x - K.mean(x)
    yc = y - K.mean(y)
    return K.mean(xc*yc) / (K.std(x)*K.std(y) + K.epsilon())

def earth_mover_loss(y_true, y_pred):
    """
    Earth Mover's Distance loss.

    Reproduced from https://github.com/titu1994/neural-image-assessment/blob/master/train_inception_resnet.py
    """
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)

def make_loss(loss, **params_defa):
    def custom_loss(*args, **kwargs):
        kwargs.update(params_defa)
        return loss(*args, **kwargs)
    return custom_loss


