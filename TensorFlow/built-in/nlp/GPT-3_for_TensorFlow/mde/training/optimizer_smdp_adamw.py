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
import six
import re
import tensorflow as tf
import os

from npu_bridge.hccl import hccl_ops

from . import smdp_gradients
from mde.distribute.dist import Allgather
from mde.distribute.mix_parallel_init import get_data_parallel_world_size, get_data_parallel_rank, get_data_parallel_group

WORLD_SIZE = get_data_parallel_world_size()
RANK_INDEX = get_data_parallel_rank()


class AdamWeightDecayOptimizer_with_smdp(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999, 
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer_with_smdp"):
    """Constructs a AdamWeightDecayOptimizer_with_smdp."""
    super(AdamWeightDecayOptimizer_with_smdp, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def compute_gradients(self, loss, *args, **kwargs):
    grads_and_vars = smdp_gradients.compute_gradients( loss, recompute=True, var_partition=False )
    return grads_and_vars


  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    global_step = tf.train.get_global_step()
    increase_global_step = tf.assign_add( global_step, tf.constant(1, dtype=tf.int64))

    split_dim = []
    split_flag = []
    allreduce_num = len(grads_and_vars) 
    for i, (grad, param) in enumerate(grads_and_vars):
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      # --- smdp , create split m and v -----
      ori_shape = param.shape.as_list()
      can_split=False
      each_split_dim = 0
      for dim, s in enumerate(ori_shape):
          if s % WORLD_SIZE == 0:
              s_split = s // WORLD_SIZE
              ori_shape[dim] = s_split
              split_dim.append(dim)
              can_split=True
              each_split_dim = dim
              break
          else:
              continue
      split_flag.append(can_split)
    #   print ('--- in apply gradient, ori_shape, split_shape:',param.shape.as_list(), ori_shape )

    #   if USE_NPU:    #????????????????????????main???????????????????????????allreduce
    #      grad = hccl_ops.allreduce(grad, "sum", fusion=0)/float(WORLD_SIZE)
    #   else:
    #      grad = hvd.allreduce(grad, average=True)  

      if can_split:
          grad_split = tf.split( grad, WORLD_SIZE, axis = each_split_dim )[RANK_INDEX]  #fp32
          param_split_tensor = tf.cast(tf.split( param, WORLD_SIZE, axis = each_split_dim )[RANK_INDEX], tf.float32)
          param_split = tf.get_variable(param_name + "/fp32_partial", initializer=param_split_tensor, dtype=tf.float32, trainable=False)

      else:
          grad_split = grad
        #   param_split = param
          param_split = tf.get_variable(param_name + "/fp32_partial", initializer=tf.cast(param.initialized_value(), tf.float32), dtype=tf.float32, trainable=False)


      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=ori_shape,
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=ori_shape, 
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      # next_m = (
      #     tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad_split))
      # next_v = (
      #     tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
      #                                               tf.square(grad_split)))


      # ???????????????
      beta_1_grad = tf.multiply(1.0 - self.beta_1, grad_split)
      beta_2_grad = tf.multiply(1.0 - self.beta_2, tf.square(grad_split))

      beta_grad = tf.group(beta_1_grad, beta_2_grad)

      with tf.control_dependencies([beta_grad]):
          next_m = tf.multiply(self.beta_1, m) + beta_1_grad
          next_v = tf.multiply(self.beta_2, v) + beta_2_grad



      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      
      # if self._do_use_weight_decay(param_name):
      #   update += self.weight_decay_rate * param_split

      # ???????????????
      with tf.control_dependencies([update]):
          if self._do_use_weight_decay(param_name):
              update = update + self.weight_decay_rate * param_split




      #original
      update_with_lr = self.learning_rate * update

      param_split = tf.assign_add(param_split, -1*update_with_lr)

      fp16_param_split = tf.cast(param_split, param.dtype)  #tensor 


      if can_split and WORLD_SIZE>1:  #???????????????????????????1??????????????????allgather
          with tf.name_scope('SMDP_Allgather'):
              split_num = allreduce_num // 8 +1   #allreduce?????????8???
              next_param = Allgather(fp16_param_split, dim=each_split_dim, group=get_data_parallel_group(),fusion=2, fusion_id= i//split_num+20)
      else:
          next_param = fp16_param_split

      #next_param = Allgather(next_param_split, dim=each_split_dim)
    #   print ('---- next params:', next_param, param.shape.as_list() )

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, increase_global_step)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
