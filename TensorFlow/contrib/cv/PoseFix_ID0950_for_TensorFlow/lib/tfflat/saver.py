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


from npu_bridge.npu_init import *
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
from config import cfg

import os
import os.path as osp

def get_variables_in_checkpoint_file(file_name):
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return reader, var_to_shape_map
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print(
                "It's likely that your checkpoint file has been compressed "
                "with SNAPPY.")

class Saver(object):
    def __init__(self, sess, var_list, model_dump_dir, name_prefix='snapshot'):
        self.sess = sess
        self.var_list = var_list
        self.model_dump_dir = model_dump_dir
        self._name_prefix = name_prefix
        
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=100000)

    def save_model(self, iter):
        filename = '{}_{:d}'.format(self._name_prefix, iter) + '.ckpt'
        if not os.path.exists(self.model_dump_dir):
            os.makedirs(self.model_dump_dir)
        filename = os.path.join(self.model_dump_dir, filename)
        self.saver.save(self.sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

def load_model(sess, model_path, load_ImageNet):
    #TODO(global variables ?? how about _adam weights)
    variables = tf.global_variables()
    reader, var_keep_dic = get_variables_in_checkpoint_file(model_path)
    if 'global_step' in var_keep_dic:
        var_keep_dic.pop('global_step')
    
    # vis_var_keep_dic = []
    variables_to_restore = {}
    changed_variables = {}
    for v in variables:
        
        v_name = v.name.split(':')[0]
        if load_ImageNet:
            if v_name == 'resnet_v1_50/conv1_leo/weights' \
                    or v_name == 'resnet_v1_101/conv1/weights' \
                    or v_name == 'resnet_v1_152/conv1/weights':
                changed_variables[v_name] = v
                continue
 
        if v_name in var_keep_dic:
            # print('Varibles restored: %s' % v.name)
            #variables_to_restore.append(v)
            variables_to_restore[v_name] = v
            # vis_var_keep_dic.append(v.name.split(':')[0])
        else:
            # print('Unrestored Variables: %s' % v.name)
            pass
    # print('Extra Variables in ckpt', set(var_keep_dic) - set(vis_var_keep_dic))
    f=open("variables.txt", "w")
    
    f.write("enter")
    f.write(str(len(variables_to_restore)))
    f.write("\n")
    f.write(str(len(changed_variables)))
    f.write("\n")
    f.write(str(variables_to_restore))
    f.write("\n")
    f.write(str(changed_variables))
    f.write("\n")
    if len(variables_to_restore) > 0:
#         for v_name, v in variables_to_restore.items():
#             sess.run(tf.assign(v,reader.get_tensor(v_name)))
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, model_path)

        for v_name,v in changed_variables.items():
            original_v = reader.get_tensor("resnet_v1_50/conv1/weights")
            sess.run(tf.assign(v[0:7,0:7,0:3,0:64],original_v))
            
    else:
        print('No variables in {} fits the network'.format(model_path))
        


