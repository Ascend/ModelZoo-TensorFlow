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
import copy
import math
import os

import numpy as np
import tensorflow as tf
from models.base_model import BaseModel

from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

FLAGS = tf.flags.FLAGS

if FLAGS.chip == 'npu':
  from npu_bridge.npu_init import *

# 与模型训练相关的参数
tf.flags.DEFINE_string('bsrn_model_scales', '-1', 'Supported scales of the model. Use the \',\' character to specify multiple scales (e.g., 2,3,4). This parameter is involved in constructing the multi-scale structure of the model.')
tf.flags.DEFINE_integer('bsrn_conv_features', 64, 'The number of convolutional features (\'c\' in the paper).')
tf.flags.DEFINE_integer('bsrn_state_features', 64, 'The number of state features (\'s\' in the paper).')
tf.flags.DEFINE_integer('bsrn_recursions', 16, 'The number of recursions of the recursive residual block (\'R\' in the paper).')
tf.flags.DEFINE_integer('bsrn_recursion_frequency', 1, 'The frequency of upscaling features to obtain an ensembled image (\'r\' in the paper).')
tf.flags.DEFINE_string('bsrn_rgb_mean', '127.5,127.5,127.5', 'Mean R, G, and B values of the training images (e.g., 127.5,127.5,127.5).')

tf.flags.DEFINE_float('bsrn_learning_rate', 1e-4, 'Initial learning rate.')
tf.flags.DEFINE_float('bsrn_learning_rate_decay', 0.5, 'Learning rate decay factor.')
tf.flags.DEFINE_integer('bsrn_learning_rate_decay_steps', 200000, 'The number of training steps to perform learning rate decay.')
tf.flags.DEFINE_float('bsrn_adam_epsilon', 1e-8, 'Epsilon of the Adam optimizer.')
tf.flags.DEFINE_float('bsrn_clip_norm', -1, 'Clipping ratio of gradient clipping. Specify < 0 to disable clipping.')

tf.flags.DEFINE_boolean('bsrn_intermediate_outputs', False, 'Specify this to retrieve intermediate outputs instead of the last ensembled output on upscale().')


def create_model():
  return BSRN()

class BSRN(BaseModel):
  def __init__(self):
    super().__init__()
  

  def prepare(self, is_training, global_step=0):
    # config. parameters
    # is_training specifies whether the model is for training or not.
    # 保存全局步数，设置此参数可以从保存点继续训练
    self.global_step = global_step

    # 使用lambda匿名函数处理字符串
    self.scale_list = list(map(lambda x: int(x), FLAGS.scales.split(',')))
    self.model_scale_list = list(map(lambda x: int(x), FLAGS.bsrn_model_scales.split(',')))

    if (self.model_scale_list[0] == -1):
      self.model_scale_list = copy.deepcopy(self.scale_list)

    for scale in self.scale_list:
      if (not scale in self.model_scale_list):
        raise ValueError('Unsupported scale is provided.')
    for scale in self.model_scale_list:
      if (not scale in [2, 3, 4, 8]):
        raise ValueError('Unsupported scale is provided.')

    self.num_expected_residual_blocks = 1

    # Mean R, G, and B values of the training images
    self.shift_mean_list = list(map(lambda x: float(x), FLAGS.bsrn_rgb_mean.split(',')))

    if (is_training):
      self.initial_learning_rate = FLAGS.bsrn_learning_rate
      self.adam_epsilon = FLAGS.bsrn_adam_epsilon
      self.learning_rate_decay = FLAGS.bsrn_learning_rate_decay
      self.learning_rate_decay_steps = FLAGS.bsrn_learning_rate_decay_steps


    # tensorflow graph
    self.tf_graph = tf.Graph()
    with self.tf_graph.as_default():
      # placeholder is a tensor that will be always fed.
      self.tf_input = tf.placeholder(tf.float32,[None, None, None, 3],name=BaseModel.TF_INPUT_NAME)
      self.tf_scale = tf.placeholder(tf.float32, [], name=BaseModel.TF_INPUT_SCALE_NAME)
      
      if (is_training):

        self.tf_truth = tf.placeholder(tf.float32, [None, None, None, 3])

        if FLAGS.chip == 'npu':
          sess_config = tf.compat.v1.ConfigProto()
          custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
          custom_op.name = "NpuOptimizer"
          sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
          sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
          custom_op.parameter_map['dynamic_input'].b = True
          custom_op.parameter_map["dynamic_inputs_shape_range"].s = tf.compat.as_bytes("data:[-1,-1,-1, 3]")
          # custom_op.parameter_map["dynamic_input"].b = True
          # custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
          self.tf_session = tf.compat.v1.Session(config=sess_config)

        # Clips tensor values to a specified min and max.
        input_summary = tf.cast(tf.clip_by_value(self.tf_input, 0.0, 255.0), tf.uint8)
        # tensor which must be 4-D with shape [batch_size, height, width, channels]
        tf.summary.image('input', input_summary)
        truth_summary = tf.cast(tf.clip_by_value(self.tf_truth, 0.0, 255.0), tf.uint8)
        tf.summary.image('truth', truth_summary)

        tf_output_list = self._generator(input_list=self.tf_input, num_modules=FLAGS.bsrn_recursions, scale=self.tf_scale, intermediate_outputs=True, recursion_frequency=FLAGS.bsrn_recursion_frequency, reuse=False)

        for i in range(FLAGS.bsrn_recursions // FLAGS.bsrn_recursion_frequency):
          num_modules = i + 1
          output_summary = tf.cast(tf.clip_by_value(tf_output_list[i], 0.0, 255.0), tf.uint8)
          tf.summary.image('output_m%d' % (num_modules * FLAGS.bsrn_recursion_frequency), output_summary)

        self.tf_output = tf_output_list[-1]

        self.tf_global_step = tf.placeholder(tf.int64, [])
        self.tf_train_op, self.tf_loss = self._optimize(
            output_list=tf_output_list,
            truth_list=self.tf_truth,
            scale=self.tf_scale,
            global_step=self.tf_global_step
        )

        for key, loss in self.loss_dict.items():
          tf.summary.scalar(('loss/%s' % (key)), loss)

        self.tf_saver = tf.train.Saver(max_to_keep=FLAGS.save_max_keep)
        self.tf_summary_op = tf.summary.merge_all()

      else:
        self.tf_output = self._generator(input_list=self.tf_input, num_modules=FLAGS.bsrn_recursions, scale=self.tf_scale, intermediate_outputs=FLAGS.bsrn_intermediate_outputs, recursion_frequency=FLAGS.bsrn_recursion_frequency, reuse=False)

      
      self.tf_init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
      # print num trainable parameters
      print('trainable parameters')
      total_variable_parameters = 0
      for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
          variable_parameters *= dim.value
        print(' - %s: %d' % (variable.name, variable_parameters))
        total_variable_parameters += variable_parameters
      print('total: %d' % (total_variable_parameters))

      # tensorflow session

      if FLAGS.chip == 'gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_device  # set GPU:0
        # 设置set_session,与GPU有关
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.3
        # 设置GPU显存按需增长
        config.gpu_options.allow_growth = True
        self.tf_session = tf.compat.v1.Session(config=config)

      elif FLAGS.chip == 'npu':
        # os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = "1"
        # sess_config = tf.compat.v1.ConfigProto()
        # custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
        # custom_op.name = "NpuOptimizer"
        # 设置自动调优
        # custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")
        # if FLAGS.profiling:
        #   custom_op.parameter_map["profiling_mode"].b = True
        #   custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(
        #     '{"output":"/home/HwHiAiUser/output","task_trace":"on"}')
        # sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        # sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
        # self.tf_session = tf.compat.v1.Session(config=sess_config)
        sess_config = tf.compat.v1.ConfigProto()
        custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
        # custom_op.parameter_map['dynamic_input'].b = True
        # custom_op.parameter_map["dynamic_inputs_shape_range"].s = tf.compat.as_bytes("data:[-1,-1,-1, 3]")
        custom_op.parameter_map["dynamic_input"].b = True
        custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
        self.tf_session = tf.compat.v1.Session(config=sess_config)
      else:
        config = tf.compat.v1.ConfigProto()
        self.tf_session = tf.Session(config=config)

      self.tf_session.run(self.tf_init_op)





  
  def save(self, base_path):
    save_path = os.path.join(base_path, 'model.ckpt')
    self.tf_saver.save(sess=self.tf_session, save_path=save_path, global_step=self.global_step)
    tf.io.write_graph(self.tf_session.graph, save_path, 'graph.pbtxt', as_text=True)

  def restore(self, ckpt_path, target=None):
    with self.tf_graph.as_default():
      if (target == 'generator'):
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        restorer = tf.train.Saver(var_list=var_list)
      else:
        restorer = tf.train.Saver()
      restorer.restore(sess=self.tf_session, save_path=ckpt_path)
  

  def get_session(self):
    return self.tf_session
  

  def get_next_train_scale(self):
    scale = self.scale_list[np.random.randint(len(self.scale_list))]
    return scale


  def train_step(self, input_list, scale, truth_list, with_summary=False):

    # feed dict
    feed_dict = {}
    feed_dict[self.tf_input] = input_list
    feed_dict[self.tf_scale] = scale
    feed_dict[self.tf_truth] = truth_list

    feed_dict[self.tf_global_step] = self.global_step

    summary = None

    if (with_summary):
      _, loss, summary = self.tf_session.run([self.tf_train_op, self.tf_loss, self.tf_summary_op], feed_dict=feed_dict)
    else:
      _, loss = self.tf_session.run([self.tf_train_op, self.tf_loss], feed_dict=feed_dict)

    self.global_step += 1

    return loss, summary


  def upscale(self, input_list, scale):
    feed_dict = {}
    feed_dict[self.tf_input] = input_list
    feed_dict[self.tf_scale] = scale

    output_list = self.tf_session.run(self.tf_output, feed_dict=feed_dict)

    return output_list
  

  def _get_randomly_cropped_patches(self, image_list, num_patches, patch_size):
    image_list_shape = tf.shape(image_list)

    def _batch_patches(image):
      def _batch_patch(i):
        return tf.random_crop(image, size=[patch_size, patch_size, image_list_shape[3]])
      
      return tf.map_fn(_batch_patch, tf.zeros([num_patches]))
    
    return tf.map_fn(_batch_patches, image_list)

  
  def _mean_shift(self, image_list):
    image_list = image_list - self.shift_mean_list
    return image_list
  
  def _mean_inverse_shift(self, image_list):
    image_list = image_list + self.shift_mean_list
    return image_list
  

  def _conv2d(self, x, num_features, kernel_size, strides=(1, 1), kernel_initializer=None):
    return tf.layers.conv2d(x, filters=num_features, kernel_size=kernel_size, strides=strides, kernel_initializer=kernel_initializer, padding='same')
  
  def _conv2d_for_residual_block(self, x, num_features, kernel_size, strides=(1, 1)):
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1e-4/self.num_expected_residual_blocks, mode='FAN_IN', uniform=False)
    return self._conv2d(x, num_features=num_features, kernel_size=kernel_size, strides=strides, kernel_initializer=kernel_initializer)
  

  def _local_residual_block(self, x, num_features, state, num_state_features, kernel_size, weight=1.0):
    num_total_features = num_features + num_state_features

    if (num_state_features > 0):
      res = tf.concat([x, state], axis=-1)
    else:
      res = x
    
    res = self._conv2d_for_residual_block(res, num_features=num_total_features, kernel_size=kernel_size)
    res = tf.nn.relu(res)
    res = self._conv2d_for_residual_block(res, num_features=num_total_features, kernel_size=kernel_size)
    
    if (num_state_features > 0):
      res_x, state = tf.split(res, [num_features, num_state_features], axis=-1)
    else:
      res_x = res
    
    res_x *= weight
    x = x + res_x

    return x, state
  
  def _residual_module(self, x, num_features, state, num_state_features, num_blocks):
    num_total_features = num_features + num_state_features

    for block_index in range(num_blocks):
      x, state = self._local_residual_block(x, num_features=num_features, state=state, num_state_features=num_state_features, kernel_size=(3, 3))

    if (num_state_features > 0):
      res = tf.concat([x, state], axis=-1)
    else:
      res = x
    res = self._conv2d(res, num_features=num_total_features, kernel_size=(3, 3))
    if (num_state_features > 0):
      res_x, state = tf.split(res, [num_features, num_state_features], axis=-1)
    else:
      res_x = res
    x = x + res_x

    return x, state
  
  def _2x_upscaling_module(self, x, num_features):
    x = self._conv2d(x, num_features=num_features * 4, kernel_size=(3, 3))
    x = tf.nn.relu(x)
    x = tf.depth_to_space(x, 2)

    return x
  
  def _3x_upscaling_module(self, x, num_features):
    x = self._conv2d(x, num_features=num_features * 9, kernel_size=(3, 3))
    x = tf.nn.relu(x)
    x = tf.depth_to_space(x, 3)

    return x
  
  def _scale_specific_upscaling(self, x, scale, num_features):
    with tf.variable_scope('x%d' % (scale)):
      if (scale == 3):
        x = self._3x_upscaling_module(x, num_features=num_features)
      else:
        for i in range(int(math.log(scale, 2))):
          x = self._2x_upscaling_module(x, num_features=num_features)
    
    return x
  

  def _generator(self, input_list, num_modules, scale, intermediate_outputs=False, recursion_frequency=1, reuse=False):
    # Variable scope allows you to create new variables and to share already created ones
    # while providing checks to not create or share by accident.
    with tf.variable_scope('generator', reuse=reuse):
      # pre-process
      input_list = tf.cast(input_list, tf.float32)
      input_list = self._mean_shift(input_list)
      x = input_list

      # initial feature extraction
      with tf.variable_scope('first_conv'):
        x = self._conv2d(x, num_features=FLAGS.bsrn_conv_features, kernel_size=(3, 3))
      
      # initial state
      if (FLAGS.bsrn_state_features > 0):
        state = tf.zeros([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], FLAGS.bsrn_state_features], tf.float32)
      else:
        state = None

      # residual modules
      x_intermediate_list = []
      for i in range(num_modules):
        with tf.variable_scope('residual_modules', reuse=(reuse or (i > 0))):
          x, state = self._residual_module(x, num_features=FLAGS.bsrn_conv_features, state=state, num_state_features=FLAGS.bsrn_state_features, num_blocks=1)
          if (intermediate_outputs):
            if ((i+1) % recursion_frequency == 0):
              x_intermediate_list.append(x)

      # if there is no need to collect intermediate output x
      if (not intermediate_outputs):
        x_intermediate_list.append(x)
      
      # upscaling
      x_upscale_list = []
      for (i, x) in enumerate(x_intermediate_list):
        with tf.variable_scope('upscale', reuse=(reuse or (i > 0))):
          pred_fn_pairs = []
          if (2 in self.model_scale_list):
            pred_fn_pairs.append((tf.equal(scale, 2), lambda: self._scale_specific_upscaling(x, scale=2, num_features=FLAGS.bsrn_conv_features)))
          if (3 in self.model_scale_list):
            pred_fn_pairs.append((tf.equal(scale, 3), lambda: self._scale_specific_upscaling(x, scale=3, num_features=FLAGS.bsrn_conv_features)))
          if (4 in self.model_scale_list):
            pred_fn_pairs.append((tf.equal(scale, 4), lambda: self._scale_specific_upscaling(x, scale=4, num_features=FLAGS.bsrn_conv_features)))
          if (8 in self.model_scale_list):
            pred_fn_pairs.append((tf.equal(scale, 8), lambda: self._scale_specific_upscaling(x, scale=8, num_features=FLAGS.bsrn_conv_features)))
          x = tf.case(pred_fn_pairs, exclusive=True)
          x_upscale_list.append(x)

      
      # last feature extraction
      x_last_conv_list = []
      for (i, x) in enumerate(x_upscale_list):
        with tf.variable_scope('last_conv', reuse=(reuse or (i > 0))):
          if (len(self.model_scale_list) > 1):
            def _last_conv(x, scale):
              with tf.variable_scope('x%d' % (scale)):
                x = self._conv2d(x, num_features=3, kernel_size=(3, 3))
              return x

            pred_fn_pairs = []
            if (2 in self.model_scale_list):
              pred_fn_pairs.append((tf.equal(scale, 2), lambda: _last_conv(x, scale=2)))
            if (3 in self.model_scale_list):
              pred_fn_pairs.append((tf.equal(scale, 3), lambda: _last_conv(x, scale=3)))
            if (4 in self.model_scale_list):
              pred_fn_pairs.append((tf.equal(scale, 4), lambda: _last_conv(x, scale=4)))
            if (8 in self.model_scale_list):
              pred_fn_pairs.append((tf.equal(scale, 8), lambda: _last_conv(x, scale=8)))
            x = tf.case(pred_fn_pairs, exclusive=True)
          else:
            x = self._conv2d(x, num_features=3, kernel_size=(3, 3))
          x_last_conv_list.append(x)
      
      # post-process
      output_list = []
      for (i, x) in enumerate(x_last_conv_list):
        x_final = self._mean_inverse_shift(x)
        output_list.append(x_final)
    # why here ?
    if (not intermediate_outputs):
      output_list = output_list[-1]
    
    return output_list


  def _optimize(self, output_list, truth_list, scale, global_step):

    loss = 0.0
    total_loss_weights = 0.0

    # reconstruction loss
    for (i, each_output_list) in enumerate(output_list):
      num_modules = (i + 1) * FLAGS.bsrn_recursion_frequency
      current_loss_weight = 2.0 ** (num_modules-1)
      loss_l1 = tf.reduce_mean(tf.losses.absolute_difference(each_output_list, truth_list))
      self.loss_dict['recon_l1_m%d' % (num_modules)] = loss_l1
      loss += loss_l1 * current_loss_weight
      total_loss_weights += current_loss_weight

    if (total_loss_weights > 0):
      loss /= total_loss_weights

    self.loss_dict['final'] = loss

    learning_rate = tf.train.exponential_decay(self.initial_learning_rate, global_step=global_step, decay_steps=self.learning_rate_decay_steps, decay_rate=self.learning_rate_decay, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    g_optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=self.adam_epsilon)

    # A list of (gradient, variable) pairs
    g_grad_vars = g_optimizer.compute_gradients(loss, var_list=g_variables)

    if (FLAGS.bsrn_clip_norm > 0):
      # Clips tensor values to a maximum L2-norm.进行梯度裁剪
      g_grad_vars = [(tf.clip_by_norm(g, FLAGS.bsrn_clip_norm), v) for g, v in g_grad_vars]

    # return an Operation that applies the specified gradients
    g_train_op = g_optimizer.apply_gradients(g_grad_vars)
    
    train_op = g_train_op

    return train_op, loss





