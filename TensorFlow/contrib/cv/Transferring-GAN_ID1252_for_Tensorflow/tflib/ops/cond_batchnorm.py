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


import tflib as lib

import numpy as np
import tensorflow as tf

def Batchnorm(name, axes, inputs, is_training=None, stats_iter=None, update_moving_stats=True, fused=True, labels=None, n_labels=None):
    """conditional batchnorm (dumoulin et al 2016) for BCHW conv filtermaps"""
    if axes != [0,2,3]:
        raise Exception('unsupported')
    mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
    shape = mean.get_shape().as_list() # shape is [1,n,1,1]
    offset_m = lib.param(name+'.offset', np.zeros([n_labels,shape[1]], dtype='float32'))
    scale_m = lib.param(name+'.scale', np.ones([n_labels,shape[1]], dtype='float32'))
    offset = tf.nn.embedding_lookup(offset_m, labels)
    scale = tf.nn.embedding_lookup(scale_m, labels)
    result = tf.nn.batch_normalization(inputs, mean, var, offset[:,:,None,None], scale[:,:,None,None], 1e-5)
    return result