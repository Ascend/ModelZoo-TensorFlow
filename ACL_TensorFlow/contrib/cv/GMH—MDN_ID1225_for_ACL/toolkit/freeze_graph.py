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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf
import os
import argparse
# import moxing as mox
import os
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

class LinearModel(object):
  def __init__(self,
               linear_size =1024,
               num_layers = 2,
               residual = True,
               dtype=tf.float32):

    self.HUMAN_2D_SIZE = 16 * 2
    self.HUMAN_3D_SIZE = 16 * 3

    self.input_size  = self.HUMAN_2D_SIZE
    self.output_size = self.HUMAN_3D_SIZE
    self.num_models = 5     # specify the number of gaussian kernels in the mixture model
    max_norm = True
    batch_norm = True
    # === Transform the inputs ===
    with vs.variable_scope("inputs"):

      # === fix the batch size in order to introdoce uncertainty into loss ===#
      enc_in  = tf.placeholder(dtype, shape=[None, self.input_size], name="enc_in")
      self.encoder_inputs  = enc_in

    # === Create the linear + relu combos ===
    with vs.variable_scope( "linear_model" ):

      # === First layer, brings dimensionality up to linear_size ===
      w1 = tf.get_variable( name="w1", initializer=kaiming, shape=[self.HUMAN_2D_SIZE, linear_size], dtype=dtype )
      b1 = tf.get_variable( name="b1", initializer=kaiming, shape=[linear_size], dtype=dtype )
      w1 = tf.clip_by_norm(w1, 1) if max_norm else w1
      y3 = tf.matmul( enc_in, w1 ) + b1
      if batch_norm:
        y3 = tf.layers.batch_normalization(y3,training=False, name="batch_normalization")

      y3 = tf.nn.relu(y3)

      # === Create multiple bi-linear layers ===
      for idx in range( num_layers ):
        y3 = self.two_linear( y3, linear_size, residual, dtype, idx )

      # === Last linear layer has HUMAN_3D_SIZE in output ===
      w4 = tf.get_variable( name="w4", initializer=kaiming, shape=[linear_size, self.HUMAN_3D_SIZE*self.num_models], dtype=dtype )
      b4 = tf.get_variable( name="b4", initializer=kaiming, shape=[self.HUMAN_3D_SIZE*self.num_models], dtype=dtype )
      w4 = tf.clip_by_norm(w4, 1) if max_norm else w4
      y_mu = tf.matmul(y3, w4) + b4

      w5 = tf.get_variable( name="w5", initializer=kaiming, shape=[linear_size, self.num_models], dtype=dtype )
      b5 = tf.get_variable( name="b5", initializer=kaiming, shape=[self.num_models], dtype=dtype )
      w5 = tf.clip_by_norm(w5, 1) if max_norm else w5
      y_sigma = tf.matmul(y3, w5) + b5
      y_sigma = tf.nn.elu(y_sigma)+1

      w6 = tf.get_variable( name="w6", initializer=kaiming, shape=[linear_size, self.num_models], dtype=dtype )
      b6 = tf.get_variable( name="b6", initializer=kaiming, shape=[self.num_models], dtype=dtype )
      y_alpha = tf.matmul(y3, w6) + b6
      y_alpha = tf.nn.softmax(y_alpha, dim=1)

      # === End linear model ===

      components = tf.concat([y_mu, y_sigma, y_alpha], axis=1)
      self.outputs = components


  def two_linear( self, xin, linear_size, residual, dtype, idx ):
    max_norm = True
    batch_norm = True
    with vs.variable_scope( "two_linear_"+str(idx) ) as scope:
      input_size = int(xin.get_shape()[1])

      # Linear 1
      w2 = tf.get_variable( name="w2_"+str(idx), initializer=kaiming, shape=[input_size, linear_size], dtype=dtype)
      b2 = tf.get_variable( name="b2_"+str(idx), initializer=kaiming, shape=[linear_size], dtype=dtype)
      w2 = tf.clip_by_norm(w2, 1) if max_norm else w2
      y = tf.matmul(xin, w2) + b2
      if  batch_norm:
        y = tf.layers.batch_normalization(y,training=False,name="batch_normalization1"+str(idx))
      y = tf.nn.relu( y )

      # Linear 2
      w3 = tf.get_variable( name="w3_"+str(idx), initializer=kaiming, shape=[linear_size, linear_size], dtype=dtype)
      b3 = tf.get_variable( name="b3_"+str(idx), initializer=kaiming, shape=[linear_size], dtype=dtype)
      w3 = tf.clip_by_norm(w3, 1) if max_norm else w3
      y = tf.matmul(y, w3) + b3

      if  batch_norm:
        y = tf.layers.batch_normalization(y,training=False,name="batch_normalization2"+str(idx))

      y = tf.nn.relu( y )

      # Residual every 2 blocks
      y = (xin + y) if residual else y

    return y

def kaiming(shape, dtype, partition_info=None):
  return(tf.truncated_normal(shape, dtype=dtype)*tf.sqrt(2/float(shape[0])))

tf.disable_eager_execution()

def rebuild(args_opt,input_path):
  data_path = '/cache/data'
  mox.file.copy_parallel(src_url=args_opt.data_url, dst_url=data_path)
  output_name = os.path.join(data_path,"save/mdm_5_prior")
  print('重建checkpoint模型')
  model = LinearModel(linear_size =1024,
                 num_layers = 2,
                 residual = True,
                 dtype=tf.float32)
  saver = tf.train.Saver( tf.global_variables(), max_to_keep=None )
  tf.train.init_from_checkpoint(input_path, {'/':'/'})
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run('linear_model/w1:0')[0])
    saver.save(sess,output_name)
  mox.file.copy_parallel(src_url=data_path, dst_url=args_opt.train_url)
  print('重建checkpoint模型结束')

def ckpt2pb(args_opt,output_name):
    data_path = '/cache/data'
    mox.file.copy_parallel(src_url=args_opt.data_url, dst_url=data_path)
    print('读取checkpoint模型')
    input_path = os.path.join(data_path,"save/mdm_5_prior")
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    sess = tf.Session()
    saver = tf.train.import_meta_graph( input_path+ ".meta")
    saver.restore( sess, input_path )
    output_graph_def = convert_variables_to_constants(sess, sess.graph_def, output_node_names=['linear_model/concat'])
    with tf.gfile.FastGFile(output_name, mode='wb') as f:
        f.write(output_graph_def.SerializeToString())
    mox.file.copy_parallel(src_url=data_path, dst_url=args_opt.train_url)
    print('保存pb模型')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--data_url', type=str, default=None, help='the data url')
    parser.add_argument('--train_url', type=str, default=None, help='the data url')
    data_path = '/cache/data'
    input_path = os.path.join(data_path, 'checkpoint-4508635')
    output_name = os.path.join(data_path, "save/mdm_5_prior.pb")
    args_opt = parser.parse_args()
    rebuild(args_opt,input_path)
    ckpt2pb(args_opt,output_name)