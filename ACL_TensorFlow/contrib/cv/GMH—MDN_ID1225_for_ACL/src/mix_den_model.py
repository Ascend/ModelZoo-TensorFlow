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

"""Simple model to regress 3d human poses from 2d joint locations"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

from tensorflow.python.ops import variable_scope as vs

import os
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import data_utils
import cameras as cam

def kaiming(shape, dtype, partition_info=None):
  """Kaiming initialization as described in https://arxiv.org/pdf/1502.01852.pdf

  Args
    shape: dimensions of the tf array to initialize
    dtype: data type of the array
    partition_info: (Optional) info about how the variable is partitioned.
      See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/init_ops.py#L26
      Needed to be used as an initializer.
  Returns
    Tensorflow array with initial weights
  """
  return(tf.truncated_normal(shape, dtype=dtype)*tf.sqrt(2/float(shape[0])))

class LinearModel(object):
  """ A simple Linear+RELU model """

  def __init__(self,
               linear_size,
               num_layers,
               residual,
               batch_norm,
               max_norm,
               batch_size,
               learning_rate,
               summaries_dir,
               predict_14=False,
               dtype=tf.float32):
    """Creates the linear + relu model

    Args
      linear_size: integer. number of units in each layer of the model
      num_layers: integer. number of bilinear blocks in the model
      residual: boolean. Whether to add residual connections
      batch_norm: boolean. Whether to use batch normalization
      max_norm: boolean. Whether to clip weights to a norm of 1
      batch_size: integer. The size of the batches used during training
      learning_rate: float. Learning rate to start with
      summaries_dir: String. Directory where to log progress
      predict_14: boolean. Whether to predict 14 instead of 17 joints
      dtype: the data type to use to store internal variables
    """

    # There are in total 17 joints in H3.6M and 16 in MPII (and therefore in stacked
    # hourglass detections). We settled with 16 joints in 2d just to make models
    # compatible (e.g. you can train on ground truth 2d and test on SH detections).
    # This does not seem to have an effect on prediction performance.
    self.HUMAN_2D_SIZE = 16 * 2

    # In 3d all the predictions are zero-centered around the root (hip) joint, so
    # we actually predict only 16 joints. The error is still computed over 17 joints,
    # because if one uses, e.g. Procrustes alignment, there is still error in the
    # hip to account for!
    # There is also an option to predict only 14 joints, which makes our results
    # directly comparable to those in https://arxiv.org/pdf/1611.09010.pdf
    self.HUMAN_3D_SIZE = 14 * 3 if predict_14 else 16 * 3

    self.input_size  = self.HUMAN_2D_SIZE
    self.output_size = self.HUMAN_3D_SIZE

    self.isTraining = tf.placeholder(tf.bool,name="isTrainingflag")
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    # Summary writers for train and test runs
    self.train_writer = tf.summary.FileWriter( os.path.join(summaries_dir, 'train' ))
    self.test_writer  = tf.summary.FileWriter( os.path.join(summaries_dir, 'test' ))

    self.linear_size   = linear_size
    self.batch_size    = batch_size
    self.learning_rate = tf.Variable( float(learning_rate), trainable=False, dtype=dtype, name="learning_rate")
    self.global_step   = tf.Variable(0, trainable=False, name="global_step")
    decay_steps = 100000  # empirical
    decay_rate = 0.96     # empirical
    self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, decay_steps, decay_rate)
    self.num_models = 5     # specify the number of gaussian kernels in the mixture model


    # === Transform the inputs ===
    with vs.variable_scope("inputs"):

      # === fix the batch size in order to introdoce uncertainty into loss ===#

      enc_in  = tf.placeholder(dtype, shape=[None, self.input_size], name="enc_in")
      dec_out = tf.placeholder(dtype, shape=[None, self.output_size], name="dec_out")


      self.encoder_inputs  = enc_in
      self.decoder_outputs = dec_out

    # === Create the linear + relu combos ===
    with vs.variable_scope( "linear_model" ):

      # === First layer, brings dimensionality up to linear_size ===
      w1 = tf.get_variable( name="w1", initializer=kaiming, shape=[self.HUMAN_2D_SIZE, linear_size], dtype=dtype )
      b1 = tf.get_variable( name="b1", initializer=kaiming, shape=[linear_size], dtype=dtype )
      w1 = tf.clip_by_norm(w1,1) if max_norm else w1
      y3 = tf.matmul( enc_in, w1 ) + b1

      if batch_norm:
        y3 = tf.layers.batch_normalization(y3,training=self.isTraining, name="batch_normalization")
      y3 = tf.nn.relu( y3 )
      y3 = npu_ops.dropout( y3, self.dropout_keep_prob )

      # === Create multiple bi-linear layers ===
      for idx in range( num_layers ):
        y3 = self.two_linear( y3, linear_size, residual, self.dropout_keep_prob, max_norm, batch_norm, dtype, idx )



      # === Last linear layer has HUMAN_3D_SIZE in output ===
      w4 = tf.get_variable( name="w4", initializer=kaiming, shape=[linear_size, self.HUMAN_3D_SIZE*self.num_models], dtype=dtype )
      b4 = tf.get_variable( name="b4", initializer=kaiming, shape=[self.HUMAN_3D_SIZE*self.num_models], dtype=dtype )
      w4 = tf.clip_by_norm(w4,1) if max_norm else w4
      y_mu = tf.matmul(y3, w4) + b4


      w5 = tf.get_variable( name="w5", initializer=kaiming, shape=[linear_size, self.num_models], dtype=dtype )
      b5 = tf.get_variable( name="b5", initializer=kaiming, shape=[self.num_models], dtype=dtype )
      w5 = tf.clip_by_norm(w5,1) if max_norm else w5
      y_sigma = tf.matmul(y3, w5) + b5
      y_sigma = tf.nn.elu(y_sigma)+1

      w6 = tf.get_variable( name="w6", initializer=kaiming, shape=[linear_size, self.num_models], dtype=dtype )
      b6 = tf.get_variable( name="b6", initializer=kaiming, shape=[self.num_models], dtype=dtype )
      y_alpha = tf.matmul(y3, w6) + b6
      y_alpha = tf.nn.softmax(y_alpha, dim=1)

      # === End linear model ===

      components = tf.concat([y_mu, y_sigma, y_alpha], axis=1)
      self.outputs = components

    # add dirichlet conjucate prior to the mixing coefficents
    prior = tf.constant([2.0, 2.0, 2.0, 2.0, 2.0], dtype=tf.float32)
    loss_prior = Dirichlet_loss(components, self.HUMAN_3D_SIZE, self.num_models, prior)

    with vs.variable_scope('loss'):

        loss_gaussion = mean_log_Gaussian_like(dec_out, components, self.HUMAN_3D_SIZE, self.num_models)   # Mixture density network based on gaussian kernel
        self.loss = loss_gaussion + loss_prior

    tf.summary.scalar('loss', self.loss, collections=['train', 'test'])
    self.loss_summary = tf.summary.merge_all('train')




    # To keep track of the loss in mm
    self.err_mm = tf.placeholder( tf.float32, name="error_mm" )
    self.err_mm_summary = tf.summary.scalar( "loss/error_mm", self.err_mm )

    # Gradients and update operation for training the model.
    opt = tf.train.AdamOptimizer( self.learning_rate )
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):

      # Update all the trainable parameters
      gradients = opt.compute_gradients(self.loss)
      self.gradients = [[] if i==None else i for i in gradients]
      self.updates = opt.apply_gradients(gradients, global_step=self.global_step)

    # Keep track of the learning rate
    self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)

    # To save the model
    self.saver = tf.train.Saver( tf.global_variables(), max_to_keep=None )


  def two_linear( self, xin, linear_size, residual, dropout_keep_prob, max_norm, batch_norm, dtype, idx ):
    """
    Make a bi-linear block with optional residual connection

    Args
      xin: the batch that enters the block
      linear_size: integer. The size of the linear units
      residual: boolean. Whether to add a residual connection
      dropout_keep_prob: float [0,1]. Probability of dropping something out
      max_norm: boolean. Whether to clip weights to 1-norm
      batch_norm: boolean. Whether to do batch normalization
      dtype: type of the weigths. Usually tf.float32
      idx: integer. Number of layer (for naming/scoping)
    Returns
      y: the batch after it leaves the block
    """

    with vs.variable_scope( "two_linear_"+str(idx) ) as scope:

      input_size = int(xin.get_shape()[1])

      # Linear 1
      w2 = tf.get_variable( name="w2_"+str(idx), initializer=kaiming, shape=[input_size, linear_size], dtype=dtype)
      b2 = tf.get_variable( name="b2_"+str(idx), initializer=kaiming, shape=[linear_size], dtype=dtype)
      w2 = tf.clip_by_norm(w2,1) if max_norm else w2
      y = tf.matmul(xin, w2) + b2
      if  batch_norm:
        y = tf.layers.batch_normalization(y,training=self.isTraining,name="batch_normalization1"+str(idx))

      y = tf.nn.relu( y )
      y = npu_ops.dropout( y, dropout_keep_prob )

      # Linear 2
      w3 = tf.get_variable( name="w3_"+str(idx), initializer=kaiming, shape=[linear_size, linear_size], dtype=dtype)
      b3 = tf.get_variable( name="b3_"+str(idx), initializer=kaiming, shape=[linear_size], dtype=dtype)
      w3 = tf.clip_by_norm(w3,1) if max_norm else w3
      y = tf.matmul(y, w3) + b3

      if  batch_norm:
        y = tf.layers.batch_normalization(y,training=self.isTraining,name="batch_normalization2"+str(idx))

      y = tf.nn.relu( y )
      y = npu_ops.dropout( y, dropout_keep_prob )

      # Residual every 2 blocks
      y = (xin + y) if residual else y

    return y

  def step(self, session, encoder_inputs, decoder_outputs, dropout_keep_prob, isTraining=True):
    """Run a step of the model feeding the given inputs.

    Args
      session: tensorflow session to use
      encoder_inputs: list of numpy vectors to feed as encoder inputs
      decoder_outputs: list of numpy vectors that are the expected decoder outputs
      dropout_keep_prob: (0,1] dropout keep probability
      isTraining: whether to do the backward step or only forward

    Returns
      if isTraining is True, a 4-tuple
        loss: the computed loss of this batch
        loss_summary: tf summary of this batch loss, to log on tensorboard
        learning_rate_summary: tf summary of learnign rate to log on tensorboard
        outputs: predicted 3d poses
      if isTraining is False, a 3-tuple
        (loss, loss_summary, outputs) same as above
    """

    input_feed = {self.encoder_inputs: encoder_inputs,
                  self.decoder_outputs: decoder_outputs,
                  self.isTraining: isTraining,
                  self.dropout_keep_prob: dropout_keep_prob}

    # Output feed: depends on whether we do a backward step or not.
    if isTraining:
      output_feed = [self.updates,       # Update Op that does SGD
                     self.loss,
                     self.loss_summary,
                     self.learning_rate_summary,
                     self.outputs
                     ]

      outputs = session.run( output_feed, input_feed )
      return outputs[1], outputs[2], outputs[3], outputs[4]

    else:
      output_feed = [self.loss, # Loss for this batch.
                     self.loss_summary,
                     self.outputs]

      outputs = session.run(output_feed, input_feed)
      return outputs[0], outputs[1], outputs[2]  # No gradient norm

  def get_all_batches(self, data_x, data_y, camera_frame, training=True):
      """
      Obtain a list of all the batches, randomly permutted
      Args
        data_x: dictionary with 2d inputs
        data_y: dictionary with 3d expected outputs
        camera_frame: whether the 3d data is in camera coordinates
        training: True if this is a training batch. False otherwise.

      Returns
        encoder_inputs: list of 2d batches
        decoder_outputs: list of 3d batches
      """

      # Figure out how many frames we have
      n = 0
      repre = {}

      for key2d in sorted(data_x.keys()):
          n2d, _ = data_x[key2d].shape
          n = n + n2d
          repre[key2d] = n2d

      encoder_inputs = np.zeros((n, self.HUMAN_2D_SIZE), dtype=float)
      decoder_outputs = np.zeros((n, self.HUMAN_3D_SIZE), dtype=float)

      # Put all the data into big arrays
      idx = 0
      for key2d in sorted(data_x.keys()):
          (subj, b, fname) = key2d
          # keys should be the same if 3d is in camera coordinates
          key3d = key2d if (camera_frame) else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
          key3d = (subj, b, fname[:-3]) if fname.endswith('-sh') and camera_frame else key3d

          n2d, _ = data_x[key2d].shape
          encoder_inputs[idx:idx + n2d, :] = data_x[key2d]
          decoder_outputs[idx:idx + n2d, :] = data_y[key3d]
          idx = idx + n2d

      if training:
          # Randomly permute everything
          idx = np.random.permutation(n)
          encoder_inputs = encoder_inputs[idx, :]
          decoder_outputs = decoder_outputs[idx, :]

      # Make the number of examples a multiple of the batch size
      n_extra = n % self.batch_size
      if n_extra > 0:  # Otherwise examples are already a multiple of batch size
          encoder_inputs = encoder_inputs[:-n_extra, :]
          decoder_outputs = decoder_outputs[:-n_extra, :]

      n_batches = n // self.batch_size
      encoder_inputs = np.split(encoder_inputs, n_batches)
      decoder_outputs = np.split(decoder_outputs, n_batches)
      repre[sorted(data_x.keys())[-1]] = repre[sorted(data_x.keys())[-1]] - n_extra   ## track how many frames are used in each video,

      return encoder_inputs, decoder_outputs, repre


def mean_log_Gaussian_like(y_true, parameters,c,m ):
    """Mean Log Gaussian Likelihood distribution
    y_truth: ground truth 3d pose
    parameters: output of hypotheses generator, which conclude the mean, variance and mixture coeffcient of the mixture model
    c: dimension of 3d pose
    m: number of kernels
    """
    components = tf.reshape(parameters, [-1, c + 2, m])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    sigma = tf.clip_by_value(sigma, 1e-15,1e15)
    alpha = components[:, c + 1, :]
    alpha = tf.clip_by_value(alpha, 1e-8, 1.)

    exponent = tf.log(alpha) - 0.5 * c * tf.log(2 * np.pi) \
               - c * tf.log(sigma) \
               - tf.reduce_sum((tf.expand_dims(y_true, 2) - mu) ** 2, axis=1) / (2.0 * (sigma) ** 2.0)

    log_gauss = log_sum_exp(exponent, axis=1)
    res = - tf.reduce_mean(log_gauss)
    return res


def Dirichlet_loss(parameters, c, m, prior):
    '''
    add dirichlet conjucate prior to the loss function to prevent all data fitting into single kernel
    '''

    components = tf.reshape(parameters, [-1, c + 2, m])
    alpha = components[:, c + 1, :]
    alpha = tf.clip_by_value(alpha, 1e-8, 1.)

    loss = tf.reduce_sum((prior-1.0) * tf.log(alpha), axis=1)
    res = -tf.reduce_mean(loss)
    return res




def log_sum_exp(x, axis=None):
    """Log-sum-exp trick implementation"""
    x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
    return tf.log(tf.reduce_sum(tf.exp(x - x_max),
                       axis=axis, keep_dims=True))+x_max




def mean_log_LaPlace_like(y_true, parameters, c, m):
    """Mean Log Laplace Likelihood distribution
    parameters refer to mean_log_Gaussian_like
    """
    components = tf.reshape(parameters, [-1, c + 2, m])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    sigma = tf.clip_by_value(sigma, 1e-15, 1e15)
    alpha = components[:, c + 1, :]
    alpha = tf.clip_by_value(alpha, 1e-8, 1.)

    exponent = tf.log(alpha) - c * tf.log(2.0 * sigma) \
               - tf.reduce_sum(tf.abs(tf.expand_dims(y_true, 2) - mu), axis=1) / (sigma)

    log_gauss, _ = log_sum_exp(exponent, axis=1)
    res = - tf.reduce_mean(log_gauss)
    return res


