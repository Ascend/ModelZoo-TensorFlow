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
"""
Model initialization functions
"""
import tensorflow as tf
import os
import numpy as np
slim = tf.contrib.slim
def load_model_from_numpy(sess, ckpt_path, src_scope, dst_scope):
    weights = np.load(ckpt_path).tolist()
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=dst_scope)
    matched_variables = []
    variable_assigns = []
    for item in var_list:
        item_name = item.name.split('/')[-2]# "osmn/seg/convN" at beginning, ":0" at last
        item_type = item.name.split('/')[-1].split(':')[0] # weights or bias 
        if not item_name in weights.keys():
            print item.name, 'not in weights'
            continue
        matched_variables.append(item_name)
                                        
        sess.run(tf.assign(item, weights[item_name][item_type]))
    print('matched variables from ckpt', ckpt_path)
    print(matched_variables)

def load_model(ckpt_path, src_scope, dst_scope):
    """Initialize the network parameters from an existing model with replaced scope names
    Args:
    Path to the checkpoint
    Returns:
    Function that takes a session and initilaizes the network
    """
    if ckpt_path[-4:] == '.npy':
        init_fn = lambda sess: load_model_from_numpy(sess, ckpt_path, src_scope, dst_scope)
    else:
        reader = tf.train.NewCheckpointReader(ckpt_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        vars_corresp = dict()
        for v in var_to_shape_map:
            v_name = v.split('/') 
            # deeplab model is a special case
            if len(v_name[-1]) == 1:
                if v_name[-1] == 'b':
                    v_new = v + 'iases'
                elif v_name[-1] == 'w':
                    v_new = v + 'eights'
                if v_name[0].startswith('conv'):
                    v_new = 'deeplab/' + v_name[0][:-2] + '/' + v_new
            else:
                v_new = v
            print(v_new)
            corr_var = slim.get_model_variables(v_new.replace(src_scope, dst_scope))
            if len(corr_var) > 0 and var_to_shape_map[v] == corr_var[0].get_shape().as_list():
                vars_corresp[v] = corr_var[0]
        print('matched variables from ckpt', ckpt_path)
        print(vars_corresp.keys())
        init_fn = slim.assign_from_checkpoint_fn(
                ckpt_path,
                vars_corresp)
    return init_fn

