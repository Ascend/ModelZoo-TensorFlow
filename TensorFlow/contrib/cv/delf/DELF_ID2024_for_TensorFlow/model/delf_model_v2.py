# Lint as: python3
# coding=utf-8
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
# ==============================================================================
"""DELF model implementation based on the following paper.

  Large-Scale Image Retrieval with Attentive Deep Local Features
  https://arxiv.org/abs/1612.06321
"""
from npu_bridge.npu_init import *

import tensorflow as tf

from delf.python.training.model import resnet50_ops as ops
from delf.python.training.model import resnet50 as resnet

### layers = tf.keras.layers
### reg = tf.keras.regularizers

_DECAY = 0.0001


def _learning_rate_schedule(global_step, max_steps, initial_lr):
  """Calculates learning_rate with linear decay.
  Args:
    global_step: int, global step.
    max_steps: int, maximum iterations.
    initial_lr: float, initial learning rate.
  Returns:
    lr: float, learning rate.
  """
  ### lr = initial_lr * (1.0 - global_step / max_steps)
  global_step_value = tf.cast(global_step, tf.float32)
  lr = initial_lr * (1.0 - global_step_value / float(max_steps))
  return lr


class AttentionModel(object):
  """Instantiates attention model.

  Uses two [kernel_size x kernel_size] convolutions and softplus as activation
  to compute an attention map with the same resolution as the featuremap.
  Features l2-normalized and aggregated using attention probabilites as weights.
  The features (targets) to be aggregated can be the input featuremap, or a
  different one with the same resolution.
  """

  def __init__(self, kernel_size=1, decay=_DECAY, name='attention'):
    """Initialization of attention model.

    Args:
      kernel_size: int, kernel size of convolutions.
      decay: float, decay for l2 regularization of kernel weights.
      name: str, name to identify model.
    """
    self.kernel_size = kernel_size
    self.decay = decay
    self.name = name

    ### # First convolutional layer (called with relu activation).
    ### self.conv1 = layers.Conv2D(
    ###     512,
    ###     kernel_size,
    ###     kernel_regularizer=reg.l2(decay),
    ###     padding='same',
    ###     name='attn_conv1')
    ### self.bn_conv1 = layers.BatchNormalization(axis=3, name='bn_conv1')

    ### # Second convolutional layer, with softplus activation.
    ### self.conv2 = layers.Conv2D(
    ###     1,
    ###     kernel_size,
    ###     kernel_regularizer=reg.l2(decay),
    ###     padding='same',
    ###     name='attn_conv2')
    ### self.activation_layer = layers.Activation('softplus')

  def build_call(self, inputs, targets=None, training=True):
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      x = ops.conv_layer(inputs, self.kernel_size, 1024, 512, 1, "attn_conv1")  # TODO: kernel_regularizer
      x = ops.bn(x, is_training=training, name="bn_conv1")
      x = tf.nn.relu(x)
      score = ops.conv_layer(x, self.kernel_size, 512, 1, 1, "attn_conv2")  # TODO: kernel_regularizer
      prob = tf.nn.softplus(score)

      # Aggregate inputs if targets is None.
      if targets is None:
        targets = inputs

      # L2-normalize the featuremap before pooling.
      targets = tf.nn.l2_normalize(targets, axis=-1)
      feat = tf.reduce_mean(tf.multiply(targets, prob), [1, 2], keepdims=False)

    return feat, prob, score


class AutoencoderModel(object):
  """Instantiates the Keras Autoencoder model."""

  def __init__(self, reduced_dimension, expand_dimension, kernel_size=1, name='autoencoder'):
    """Initialization of Autoencoder model.

    Args:
      reduced_dimension: int, the output dimension of the autoencoder layer.
      expand_dimension: int, the input dimension of the autoencoder layer.
      kernel_size: int or tuple, height and width of the 2D convolution window.
      name: str, name to identify model.
    """
    self.name = name
    self.reduced_dimension = reduced_dimension
    self.expand_dimension = expand_dimension
    self.kernel_size = kernel_size

    ### self.conv1 = layers.Conv2D(
    ###     reduced_dimension,
    ###     kernel_size,
    ###     padding='same',
    ###     name='autoenc_conv1')
    ### self.conv2 = layers.Conv2D(
    ###     expand_dimension,
    ###     kernel_size,
    ###     activation=tf.keras.activations.relu,
    ###     padding='same',
    ###     name='autoenc_conv2')

  def build_call(self, inputs):
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      dim_reduced_features = ops.conv_layer(inputs, self.kernel_size, 1024, self.reduced_dimension, 1, "autoenc_conv1")
      dim_expanded_features = ops.conv_layer(dim_reduced_features, self.kernel_size, 128, self.expand_dimension, 1, "autoenc_conv2")
      dim_expanded_features = tf.nn.relu(dim_expanded_features)
    return dim_expanded_features, dim_reduced_features


class Delf(object):
  """Instantiates Keras DELF model using ResNet50 as backbone.

  This class implements the [DELF](https://arxiv.org/abs/1612.06321) model for
  extracting local features from images. The backbone is a ResNet50 network
  that extracts featuremaps from both conv_4 and conv_5 layers. Activations
  from conv_4 are used to compute an attention map of the same resolution.
  """

  def __init__(self,
               block3_strides=True,
               name='DELF',
               pooling='avg',
               gem_power=3.0,
               embedding_layer=False,
               embedding_layer_dim=2048,
               use_dim_reduction=False,
               reduced_dimension=128,
               dim_expand_channels=1024,
               num_classes=1000,
               initial_lr=0.001,
               attention_loss_weight=1.0,
               reconstruction_loss_weight=10.0):
    """Initialization of DELF model.

    Args:
      block3_strides: bool, whether to add strides to the output of block3.
      name: str, name to identify model.
      pooling: str, pooling mode for global feature extraction; possible values
        are 'None', 'avg', 'max', 'gem.'
      gem_power: float, GeM power for GeM pooling. Only used if pooling ==
        'gem'.
      embedding_layer: bool, whether to create an embedding layer (FC whitening
        layer).
      embedding_layer_dim: int, size of the embedding layer.
      use_dim_reduction: Whether to integrate dimensionality reduction layers.
        If True, extra layers are added to reduce the dimensionality of the
        extracted features.
      reduced_dimension: int, only used if use_dim_reduction is True. The output
        dimension of the autoencoder layer.
      dim_expand_channels: int, only used if use_dim_reduction is True. The
        number of channels of the backbone block used. Default value 1024 is the
        number of channels of backbone block 'block3'.
    """
    self.name = name
    self.num_classes = num_classes
    self.initial_lr = initial_lr
    self.attention_loss_weight = attention_loss_weight
    self.reconstruction_loss_weight = reconstruction_loss_weight

    # Backbone using Keras ResNet50.
    self.backbone = resnet.ResNet50(
        'channels_last',
        name='backbone',
        include_top=False,
        pooling=pooling,
        block3_strides=block3_strides,
        average_pooling=False,
        gem_power=gem_power,
        embedding_layer=embedding_layer,
        embedding_layer_dim=embedding_layer_dim)

    # Attention model.
    self.attention = AttentionModel(name='attention')

    # Autoencoder model.
    self._use_dim_reduction = use_dim_reduction
    if self._use_dim_reduction:
      self.autoencoder = AutoencoderModel(reduced_dimension,
                                          dim_expand_channels,
                                          name='autoencoder')

  def global_and_local_forward_pass(self, images, training=True):
    """Run a forward to calculate global descriptor and attention prelogits.

    Args:
      images: Tensor containing the dataset on which to run the forward pass.
      training: Indicator of wether the forward pass is running in training mode
        or not.

    Returns:
      Global descriptor prelogits, attention prelogits, attention scores,
        backbone weights.
    """
    backbone_blocks = {}
    desc_prelogits = self.backbone.build_call(
        images, intermediates_dict=backbone_blocks, training=training)
    # Prevent gradients from propagating into the backbone. See DELG paper:
    # https://arxiv.org/abs/2001.05027.
    block3 = backbone_blocks['block3']  # pytype: disable=key-error
    block3 = tf.stop_gradient(block3)
    if self._use_dim_reduction:
      (dim_expanded_features, dim_reduced_features) = self.autoencoder.build_call(block3)
      attn_prelogits, attn_scores, _ = self.attention.build_call(
          block3,
          targets=dim_expanded_features,
          training=training)
    else:
      attn_prelogits, attn_scores, _ = self.attention.build_call(block3, training=training)
      dim_expanded_features = None
      dim_reduced_features = None
    return (desc_prelogits, attn_prelogits, attn_scores, backbone_blocks,
            dim_expanded_features, dim_reduced_features)

  def build_call(self, input_image, training=True):
    (global_feature, _, attn_scores, backbone_blocks, _,
     dim_reduced_features) = self.global_and_local_forward_pass(input_image,
                                                                training)
    if self._use_dim_reduction:
      features = dim_reduced_features
    else:
      features = backbone_blocks['block3']  # pytype: disable=key-error
    return global_feature, attn_scores, features

  def step_fn(self, images, labels, global_batch_size, max_steps):
    global_step = tf.train.get_or_create_global_step()
    learning_rate = _learning_rate_schedule(global_step, max_steps, self.initial_lr)
    self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                momentum=0.9,
                                                use_nesterov=True,
                                                use_locking=True)

    # Temporary workaround to avoid some corrupted labels.
    labels = tf.clip_by_value(labels, 0, self.num_classes)

    def _compute_loss(labels, predictions):
      per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=predictions)
      return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

    # Record gradients and loss through backbone.
    with tf.GradientTape() as gradient_tape:
      # Make a forward pass to calculate prelogits.
      (desc_prelogits, attn_prelogits, attn_scores, backbone_blocks,
       dim_expanded_features, _) = self.global_and_local_forward_pass(images)

      # Calculate global loss by applying the descriptor classifier.
      with tf.variable_scope("desc_fc", reuse=tf.AUTO_REUSE):
        desc_logits = ops.fc_layer(desc_prelogits, self.num_classes, name='desc_fc')
        desc_loss = _compute_loss(labels, desc_logits)

      # Calculate attention loss by applying the attention block classifier.
      with tf.variable_scope("attn_fc", reuse=tf.AUTO_REUSE):
        attn_logits = ops.fc_layer(attn_prelogits, self.num_classes, name='attn_fc')
        attn_loss = _compute_loss(labels, attn_logits)

      # Calculate reconstruction loss between the attention prelogits and the backbone.
      block3 = tf.stop_gradient(backbone_blocks['block3'])
      reconstruction_loss_ori = tf.losses.mean_squared_error(block3, dim_expanded_features)
      reconstruction_loss = tf.math.reduce_mean(reconstruction_loss_ori)

      # Cumulate global loss, attention loss and reconstruction loss.
      total_loss = (
          desc_loss + self.attention_loss_weight * attn_loss +
          self.reconstruction_loss_weight * reconstruction_loss)

    variables = tf.trainable_variables()
    gradients = tf.gradients(total_loss, variables)
    clip_val = tf.constant(10.0)
    clipped, _ = tf.clip_by_global_norm(gradients, clip_norm=clip_val)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = self.optimizer.apply_gradients(zip(clipped, variables),
                                                global_step=global_step)
    with tf.control_dependencies([train_op]):
      # Input image-related summaries.
      tf.summary.image('batch_images', (images + 1.0) / 2.0)
      tf.summary.scalar(
          'image_range/max', tf.reduce_max(images))
      tf.summary.scalar(
          'image_range/min', tf.reduce_min(images))

      # Record statistics of the attention score
      tf.summary.image('batch_attention', attn_scores / tf.reduce_max(attn_scores + 1e-3))
      tf.summary.scalar('attention/max', tf.reduce_max(attn_scores))
      tf.summary.scalar('attention/min', tf.reduce_min(attn_scores))
      tf.summary.scalar('attention/mean', tf.reduce_mean(attn_scores))
      #TODO: tf.summary.scalar('attention/percent_25', tfp.stats.percentile(attn_scores, 25.0))
      # tf.summary.scalar('attention/percent_50', tfp.stats.percentile(attn_scores, 50.0))
      # tf.summary.scalar('attention/percent_75', tfp.stats.percentile(attn_scores, 75.0))

      # Record desc accuracy
      desc_accuracy = tf.equal(tf.argmax(desc_logits, 1), labels)
      desc_accuracy = tf.reduce_mean(tf.cast(desc_accuracy, tf.float32))
      tf.summary.scalar('desc_accuracy', desc_accuracy)

      # Record attn accuracy
      attn_accuracy = tf.equal(tf.argmax(attn_logits, 1), labels)
      attn_accuracy= tf.reduce_mean(tf.cast(attn_accuracy, tf.float32))
      tf.summary.scalar('attn_accuracy', attn_accuracy)
      merged_summary_op = tf.summary.merge_all()
    return merged_summary_op, desc_loss, attn_loss, reconstruction_loss 

  def eval_fn(self, images, labels, eval_batch_size):
    labels = tf.clip_by_value(labels, 0, self.num_classes)

    backbone_blocks = {}
    desc_prelogits = self.backbone.build_call(images, intermediates_dict=backbone_blocks, training=False)

    ## desc predictions 
    with tf.variable_scope("desc_fc", reuse=True):
      desc_logits = ops.fc_layer(desc_prelogits, self.num_classes, name='desc_fc')
      # desc_pref = tf.nn.softmax(desc_logits, axis=-1)
      desc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=desc_logits)
      desc_loss = tf.nn.compute_average_loss(desc_loss, global_batch_size=eval_batch_size)
      desc_accuracy = tf.equal(tf.argmax(desc_logits, 1), labels)
      desc_accuracy = tf.reduce_mean(tf.cast(desc_accuracy, tf.float32))

    ## attention predictions
    block3 = backbone_blocks['block3']
    attn_prelogits, _, _ = self.attention.build_call(block3, training=False)
    with tf.variable_scope("attn_fc", reuse=True):
      attn_logits = ops.fc_layer(attn_prelogits, self.num_classes, name='attn_fc')
      attn_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=attn_logits)
      attn_loss = tf.nn.compute_average_loss(attn_loss, global_batch_size=eval_batch_size)
      attn_accuracy = tf.equal(tf.argmax(attn_logits, 1), labels)
      attn_accuracy = tf.reduce_mean(tf.cast(attn_accuracy, tf.float32))

    return desc_loss, desc_accuracy, attn_loss, attn_accuracy

