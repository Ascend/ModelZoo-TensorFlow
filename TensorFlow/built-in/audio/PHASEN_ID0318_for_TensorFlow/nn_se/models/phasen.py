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
import tensorflow as tf
import collections
import numpy as np

from nn_se.FLAGS import PARAM
from nn_se.utils import losses
from nn_se.utils import misc_utils


# def dense(o, n, act=None, use_bias=True, np=None, name="dense"):
#     with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
#        if np is None:
#           w = get_variable("W", (o.get_shape().as_list()[-1],n))
#           o = tf.tensordot(o,w,[(-1),(-2)])
#        else:
#           w1 = get_variable("W1", (o.get_shape().as_list()[-1],np))
#           w2 = get_variable("W2", (np,n))
#           w = tf.matmul(w1,w2, name="W")
#           o = tf.tensordot(o,w,[(-1),(-2)])
#        if use_bias:
#           o = o + get_variable("b",(n,))
#        o = act(o) if act else o
#        return o


# def conv2dx(o, n, act=None, h=[-1,0,1], w=[-1,0,1], name='conv2d'):
#  '''
#  :param o: input of shape [B, H, W, C]
#  :param n: output channel number
#  :param h: kernel index for H
#  :param w: kernel index for W
#  :return: shape [B, H, W, n]
#  '''
#  B,H,W,C = o.get_shape().as_list()
#  with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
#     hn1,hn2 = -np.min(h+[0]),np.max(h+[0])
#     wn1,wn2 = -np.min(w+[0]),np.max(w+[0])
#     o = tf.pad(o,[(0,0),(hn1,hn2),(wn1,wn2),(0,0)],mode="SYMMETRIC")
#     o = tf.concat([ o[:, hn1+i:hn1+i+H, wn1+j:wn1+j+W, :] for j in w for i in h], axis=-1)
#     o = dense(o,n,None,name="conv2dx")
#  return act(o) if act else o

class SelfConv2d(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size,
               activation=None, use_bias=True, trainable=True,
               name='conv2d', padding='same'):
    super(SelfConv2d, self).__init__()
    assert padding == 'same', 'only same is surpported.'
    assert kernel_size[0] & 1 == 1 and kernel_size[1] & 1 == 1, "filters dime must be odd"
    self._h = [i for i in range(-(kernel_size[0] // 2), (kernel_size[0] // 2)+1)]
    self._w = [i for i in range(-(kernel_size[1] // 2), (kernel_size[1] // 2)+1)]
    self._name = name
    self.dense_kernel = tf.keras.layers.Dense(filters,
                                              activation=activation,
                                              use_bias=use_bias,
                                              trainable=trainable,
                                              name='D')

  def call(self, feature_in):
    fea_shape = tf.shape(feature_in)
    H = fea_shape[1]
    W = fea_shape[2]
    hn1,hn2 = -np.min(self._h+[0]),np.max(self._h+[0])
    wn1,wn2 = -np.min(self._w+[0]),np.max(self._w+[0])
    out = tf.pad(feature_in,[(0,0),(hn1,hn2),(wn1,wn2),(0,0)],mode="SYMMETRIC")
    out = tf.concat([out[:, hn1+i:hn1+i+H, wn1+j:wn1+j+W, :] for j in self._w for i in self._h], axis=-1)
    out = self.dense_kernel(out)
    return out


class BatchNormAndActivate(tf.keras.Model):
  def __init__(self, bn_axis=-1, activation=tf.keras.activations.relu, name='BN_activate'):
    super(BatchNormAndActivate, self).__init__()
    self._name = name
    self.bn_layer = tf.keras.layers.BatchNormalization(bn_axis, name="bn")
    self.activate_fn = activation

  def call(self, fea_in, training):
    out = self.bn_layer(fea_in, training=training)
    if self.activate_fn is not None:
      out = self.activate_fn(out)
    # print(training)
    # # if training:
    # for update_op in self.bn_layer.updates:
    #   tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)
    return out

  def get_bn_weight(self):
    moving_mean, moving_variance = None, None
    for var in self.bn_layer.variables:
      name = var.name.lower()
      if "variance" in name:
        moving_variance = var
      if "mean" in name:
        moving_mean = var
    if moving_mean is not None and moving_variance is not None:
      return [moving_mean, moving_variance]
    raise ValueError("Unable to find moving mean and variance")


class Stream_PreNet(tf.keras.Model):
  def __init__(self, channel_out, kernels=[[1, 7], [7, 1]],
               conv2d_activation=None, conv2d_bn=False,
               name='streamA_or_P_prenet'):
    '''
    channel_out: output channel
    kernels: kernel for layers
    '''
    super(Stream_PreNet, self).__init__()
    self._name = name
    self.nn_layers = []
    for i, kernel in enumerate(kernels):
      conv_name = ("conv2d_%d" % i)
      conv2d = SelfConv2d(filters=channel_out,
                          kernel_size=kernel,
                          activation=(
                              None if conv2d_bn else conv2d_activation),
                          padding="same", name=conv_name)
      self.nn_layers.append({'name':'conv2d', 'fn':conv2d})
      if conv2d_bn:
        bn_fn = BatchNormAndActivate(bn_axis=-1, activation=conv2d_activation, name='bna_%d' % i)
        self.nn_layers.append({'name':'batchNorm', 'fn':bn_fn})


  def call(self, feature_in, training):
    '''
    feature_in : [batch, T, F, channel_in]
    return : [batch, T, F, channel_out]
    '''
    if len(self.nn_layers) == 0:
      return feature_in
    out = feature_in
    for layer_dict in self.nn_layers:
      layer_name = layer_dict['name']
      layer_fn = layer_dict['fn']
      if layer_name is 'batchNorm':
        out = layer_fn(out, training)
      else:
        out = layer_fn(out)
    return out


class FrequencyTransformationBlock(tf.keras.Model):
  def __init__(self, frequency_dim, channel_in_out, channel_attention=5, name="FTB"):
    super(FrequencyTransformationBlock, self).__init__()
    self._name = name
    self.frequency_dim = frequency_dim
    self.channel_out = channel_in_out
    self.att_conv2d_1 = SelfConv2d(channel_attention, [1, 1], padding="same",
                                   name="att_conv2d_1")  # [batch, T, F * channel_attention]
    self.att_conv2d_1_bna = BatchNormAndActivate(name="att_conv2d_1_bna")

    self.att_inner_reshape = tf.keras.layers.Reshape(
        [-1, frequency_dim * channel_attention])
    self.att_conv1d_2 = tf.keras.layers.Conv1D(frequency_dim, 9, padding="same",
                                               name="att_conv1d_2")  # [batch, T, F]
    self.att_conv1d_2_bna = BatchNormAndActivate(name="att_conv1d_2_bna")
    self.att_out_reshape = tf.keras.layers.Reshape([-1, frequency_dim, 1])
    self.frequencyFC = tf.keras.layers.Dense(frequency_dim, name="FFC")
    self.concat_FFCout_and_In = tf.keras.layers.Concatenate(-1)

    self.out_conv2d = SelfConv2d(channel_in_out, [1, 1], padding="same",
                                 name="out_conv2d")
    self.out_conv2d_bna = BatchNormAndActivate(name="out_conv2d_bna")

  def call(self, feature_in, training):
    '''
    feature_n: [batch, T, F, channel_in_out]
    '''
    att_out = self.att_conv2d_1(feature_in)
    att_out = self.att_conv2d_1_bna(att_out, training=training)

    att_out = self.att_inner_reshape(att_out)
    att_out = self.att_conv1d_2(att_out)
    att_out = self.att_conv1d_2_bna(att_out, training=training)
    att_out = self.att_out_reshape(att_out)

    # [batch, T, F, channel_in_out]
    atted_out = tf.multiply(feature_in, att_out)
    # [batch, T, channel_in_out, F]
    atted_out_T = tf.transpose(atted_out, perm=[0, 1, 3, 2])
    ffc_out_T = self.frequencyFC(atted_out_T)
    # [batch, T, F, channel_in_out]
    ffc_out = tf.transpose(ffc_out_T, perm=[0, 1, 3, 2])
    feature_in.set_shape([None, None, self.frequency_dim, self.channel_out])
    # print(feature_in.get_shape().as_list(), ffc_out.get_shape().as_list())
    concated_out = self.concat_FFCout_and_In([feature_in, ffc_out])

    out = self.out_conv2d(concated_out)
    out = self.out_conv2d_bna(out, training=training)
    return out


class InfoCommunicate(tf.keras.Model):
  def __init__(self, channel_out, activate_fn=tf.keras.activations.tanh, name='InfoC'):
    super(InfoCommunicate, self).__init__()
    self._name = name
    self.conv2d = SelfConv2d(
        channel_out, [1, 1], padding="same", name="conv2d")
    self.activate_fn = activate_fn

  def call(self, feature_x1, feature_x2):
    # feature_x1: [batch, T, F, channel_out]
    # feature_x2: [batch, T, F, Cp or Ca]
    # return: [batch, T, F, channel_out]
    conv_out = self.conv2d(feature_x2)
    act_out = self.activate_fn(conv_out)

    out_multiply = tf.multiply(feature_x1, act_out)
    return out_multiply


class TwoStreamBlock(tf.keras.Model):
  def __init__(self, frequency_dim, channel_in_out_A, channel_in_out_P, name="TSB"):
    super(TwoStreamBlock, self).__init__()
    self._name = name
    self.sA1_pre_FTB = FrequencyTransformationBlock(
        frequency_dim, channel_in_out_A, name="sA1_pre_FTB")
    self.sA2_conv2d = SelfConv2d(
        channel_in_out_A, [5, 5], padding="same", name="sA2_conv2d")
    self.sA2_conv2d_bna = BatchNormAndActivate(name="sA2_conv2d_bna")
    self.sA3_conv2d = SelfConv2d(
        channel_in_out_A, [25, 1], padding="same", name="sA3_conv2d")
    self.sA3_conv2d_bna = BatchNormAndActivate(name="sA3_conv2d_bna")
    self.sA4_conv2d = SelfConv2d(
        channel_in_out_A, [5, 5], padding="same", name="sA4_conv2d")
    self.sA4_conv2d_bna = BatchNormAndActivate(name="sA4_conv2d_bna")
    self.sA5_post_FTB = FrequencyTransformationBlock(
        frequency_dim, channel_in_out_A, name="sA5_post_FTB")
    self.sA6_info_communicate = InfoCommunicate(
        channel_in_out_A, name="InfoC_A")

    self.sP1_conv2d_before_LN = tf.keras.layers.LayerNormalization(
        -1, name="sP1_conv2d_before_LN")
    self.sP1_conv2d = SelfConv2d(
        channel_in_out_P, [5, 3], padding="same", name="sP1_conv2d")
    self.sP2_conv2d_before_LN = tf.keras.layers.LayerNormalization(
        -1, name="sP2_conv2d_before_LN")
    self.sP2_conv2d = SelfConv2d(
        channel_in_out_P, [25, 1], padding="same", name="sP2_conv2d")
    self.sP3_info_communicate = InfoCommunicate(
        channel_in_out_P, name="InfoC_P")

  def call(self, feature_sA, feature_sP, training):
    # Stream A
    sA_out = feature_sA
    sA_out = self.sA1_pre_FTB(sA_out, training=training)
    sA_out = self.sA2_conv2d(sA_out)
    sA_out = self.sA2_conv2d_bna(sA_out, training=training)
    sA_out = self.sA3_conv2d(sA_out)
    sA_out = self.sA3_conv2d_bna(sA_out, training=training)
    sA_out = self.sA4_conv2d(sA_out)
    sA_out = self.sA4_conv2d_bna(sA_out, training=training)
    sA_out = self.sA5_post_FTB(sA_out, training=training)

    # Stream P
    sP_out = feature_sP
    sP_out = self.sP1_conv2d_before_LN(sP_out)
    sP_out = self.sP1_conv2d(sP_out)
    sP_out = self.sP2_conv2d_before_LN(sP_out)
    sP_out = self.sP2_conv2d(sP_out)

    # information communication
    sA_fin_out = self.sA6_info_communicate(sA_out, sP_out)
    sP_fin_out = self.sP3_info_communicate(sP_out, sA_out)

    return sA_fin_out, sP_fin_out


class StreamAmplitude_PostNet(tf.keras.Model):
  def __init__(self, frequency_dim, name="sA_PostNet"):
    super(StreamAmplitude_PostNet, self).__init__()
    self._name = name
    self.p1_conv2d = SelfConv2d(8, [1, 1],
                                activation=tf.keras.activations.sigmoid,
                                padding="same", name="p1_conv2d")
    self.p1_reshape = tf.keras.layers.Reshape([-1, frequency_dim * 8])

    fw_lstm = tf.keras.layers.LSTM(600,
                                    #  dropout=0.2,
                                    implementation=2,
                                    return_sequences=True, name='fwlstm')
    bw_lstm = tf.keras.layers.LSTM(600,
                                    #  dropout=0.2,
                                    implementation=2,
                                    return_sequences=True, name='bwlstm', go_backwards=True)
    #fw_lstm = tf.compat.v1.keras.layers.CuDNNLSTM(600,
                                                  #  dropout=0.2,
                                                  # implementation=2,
                                                  #return_sequences=True, name='fwlstm')
    #bw_lstm = tf.compat.v1.keras.layers.CuDNNLSTM(600,
                                                  #  dropout=0.2,
                                                  # implementation=2,
                                                  #return_sequences=True, name='bwlstm', go_backwards=True)
    self.p2_blstm = tf.keras.layers.Bidirectional(layer=fw_lstm, backward_layer=bw_lstm,
                                                  merge_mode='concat', name='p2_blstm')

    self.p3_dense = tf.keras.layers.Dense(
        600, activation=tf.keras.activations.relu, name="p3_dense")
    self.p4_dense = tf.keras.layers.Dense(
        600, activation=tf.keras.activations.relu, name="p4_dense")
    self.out_dense = tf.keras.layers.Dense(frequency_dim, activation=tf.keras.activations.sigmoid,
                                           name="out_dense")

  def call(self, feature_sA, training):
    '''
    return [batch, T, F]
    '''
    out = feature_sA
    out = self.p1_conv2d(out)
    out = self.p1_reshape(out)
    #out = self.p2_blstm(out, training=training)
    out = self.p3_dense(out)
    out = self.p4_dense(out)
    out = self.out_dense(out)
    return out


class StreamPhase_PostNet(tf.keras.Model):
  def __init__(self, name="sP_PostNet"):
    super(StreamPhase_PostNet, self).__init__()
    self._name = name
    self._layers.append(SelfConv2d(
        2, [1, 1], padding="same", name="conv2d"))

  def call(self, feature_sP, training):
    '''
    return [batch, T, F]->complex
    '''
    out = feature_sP
    for layer_fn in self._layers:
      out = layer_fn(out)
    # out: [batch, T, F, 2]
    out_complex = tf.complex(out[..., 0], out[..., 1])
    out_angle = tf.angle(out_complex)
    normed_out = tf.exp(tf.complex(0.0, out_angle))
    return normed_out


class NET_PHASEN_OUT(
    collections.namedtuple("NET_PHASEN_OUT",
                           ("mag_mask", "normalized_complex_phase"))):
  pass


class NetPHASEN(tf.keras.Model):
  def __init__(self, name="PHASEN"):
    super(NetPHASEN, self).__init__()
    self._name = name
    self.streamA_prenet = Stream_PreNet(
        PARAM.channel_A, kernels=PARAM.prenet_A_kernels, conv2d_bn=True,
        conv2d_activation=tf.keras.activations.relu, name="streamA_prenet")
    self.streamP_prenet = Stream_PreNet(
        PARAM.channel_P, PARAM.prenet_P_kernels, name="streanP_prenet")
    self.layers_TSB = []
    for i in range(1, PARAM.n_TSB+1):
      tsb_t = TwoStreamBlock(
          PARAM.frequency_dim, PARAM.channel_A, PARAM.channel_P, name="TSB_%d" % i)
      self.layers_TSB.append(tsb_t)
    self.streamA_postnet = StreamAmplitude_PostNet(
        PARAM.frequency_dim, name="sA_postnet")
    self.streamP_postnet = StreamPhase_PostNet(name="sP_postnet")

  def call(self, feature_in, training):
    '''
    return mag_batch[batch, time, fre]->real, normalized_complex_phase[batch, time, fre]->complex
    '''
    sA_out = self.streamA_prenet(feature_in, training=training)  # [batch, t, f, Ca]
    sP_out = self.streamP_prenet(feature_in, training=training)  # [batch, t, f, Cp]
    for tsb in self.layers_TSB:
      sA_out, sP_out = tsb(sA_out, sP_out, training=training)
    sA_out = self.streamA_postnet(sA_out, training=training)  # [batch, t, f]
    sP_out = self.streamP_postnet(sP_out, training=training)  # [batch, t, f, 2]

    est_mask = sA_out  # [batch, t, f]
    normed_complex_phase = sP_out  # [batch, t, f], complex value
    return NET_PHASEN_OUT(mag_mask=est_mask,
                          normalized_complex_phase=normed_complex_phase)


class FrowardOutputs(
    collections.namedtuple("FrowardOutputs",
                           ("est_clean_stft_batch", "est_clean_mag_batch",
                            "est_clean_wav_batch", "est_complexPhase_batch"))):
  pass


class Losses(
    collections.namedtuple("Losses",
                           ("sum_loss", "show_losses", "stop_criterion_loss"))):
  pass


def if_grads_is_nan_or_inf(grads):
  grads_is_nan_or_inf_lst = []
  for grad in grads:
    nan_or_inf = tf.reduce_max(tf.cast(tf.math.is_nan(grad), tf.int32)) + \
        tf.reduce_max(tf.cast(tf.math.is_inf(grad), tf.int32))
    grads_is_nan_or_inf_lst.append(nan_or_inf)
  grads_is_nan_or_inf = tf.stack(grads_is_nan_or_inf_lst)
  # True if grads_is_nan_or_inf > 0 else False
  return grads_is_nan_or_inf


class PHASEN(object):
  def __init__(self,
               mode,
               net_model: NetPHASEN,
               mixed_wav_batch,
               clean_wav_batch=None,
               noise_wav_batch=None):
    del noise_wav_batch
    self.mode = mode
    self.net_model = net_model
    self.net_model.build(input_shape=(None, None, PARAM.frequency_dim, 2))
    # print(net_model.summary())
    # misc_utils.show_variables(net_model.trainable_variables)

    # global_step, lr, notrainable variables
    with tf.compat.v1.variable_scope("notrain_vars", reuse=tf.compat.v1.AUTO_REUSE):
      self._global_step = tf.compat.v1.get_variable("global_step", dtype=tf.int32,
                                                    initializer=tf.constant(1), trainable=False)
      self._lr = tf.compat.v1.get_variable("lr", dtype=tf.float32, trainable=False,
                                           initializer=tf.constant(PARAM.learning_rate))

    # save variables
    self.save_variables = [self._lr, self._global_step]
    self.save_variables.extend(net_model.variables)
    if mode == PARAM.MODEL_TRAIN_KEY:
      print("\nsave PARAMs")
      misc_utils.show_variables(self.save_variables)
    self.saver = tf.compat.v1.train.Saver(self.save_variables,
                                          max_to_keep=PARAM.max_keep_ckpt,
                                          save_relative_paths=True)

    self.mixed_wav_batch = mixed_wav_batch
    self.mixed_stft_batch = misc_utils.tf_wav2stft(self.mixed_wav_batch,
                                                   PARAM.frame_length,
                                                   PARAM.frame_step,
                                                   PARAM.fft_length)
    self.mixed_mag_batch = tf.abs(self.mixed_stft_batch)
    self.mixed_angle_batch = tf.angle(self.mixed_stft_batch)
    self.batch_size = tf.shape(self.mixed_wav_batch)[0]

    if clean_wav_batch is not None:
      self.clean_wav_batch = clean_wav_batch
      self.clean_stft_batch = misc_utils.tf_wav2stft(self.clean_wav_batch,
                                                     PARAM.frame_length,
                                                     PARAM.frame_step,
                                                     PARAM.fft_length)
      self.clean_mag_batch = tf.abs(self.clean_stft_batch)
      self.clean_angle_batch = tf.angle(self.clean_stft_batch)

    # nn forward
    training = (self.mode == PARAM.MODEL_TRAIN_KEY)
    print('%s model phase is_training:' % self.mode, training, flush=True)
    self._forward_outputs = self._forward(training=training)
    self._est_clean_wav_batch = self._forward_outputs.est_clean_wav_batch

    # for lr halving
    self._new_lr = tf.compat.v1.placeholder(tf.float32, name='new_lr')
    self._assign_lr = tf.compat.v1.assign(self._lr, self._new_lr)

    # for lr warmup
    if PARAM.use_lr_warmup:
      self._lr = misc_utils.noam_scheme(
          self._lr, self.global_step, warmup_steps=PARAM.warmup_steps)

    # get loss
    if mode != PARAM.MODEL_INFER_KEY:
      # losses
      self._losses = self._get_losses()

    # get specific variables
    self.se_net_vars = self.net_model.trainable_variables

    if mode == PARAM.MODEL_VALIDATE_KEY or mode == PARAM.MODEL_INFER_KEY:
      return

    # region get specific grads
    # se_net grads
    se_loss_grads = tf.gradients(
        self._losses.sum_loss,
        self.se_net_vars,
        colocate_gradients_with_ops=True
    )
    # endregion

    all_grads = se_loss_grads
    all_params = self.se_net_vars

    if PARAM.clip_grads:
      all_grads, _ = tf.clip_by_global_norm(all_grads, PARAM.max_gradient_norm)

    # choose optimizer
    if PARAM.optimizer == "Adam":
      self._optimizer = tf.compat.v1.train.AdamOptimizer(self._lr)
    elif PARAM.optimizer == "RMSProp":
      self._optimizer = tf.compat.v1.train.RMSPropOptimizer(self._lr)

    self._grads_bad_lst = if_grads_is_nan_or_inf(all_grads)
    self._grads_bad = tf.greater(tf.reduce_max(self._grads_bad_lst), 0)
    self._grads_coef = tf.cond(self._grads_bad,
                               lambda: tf.constant(0.0),
                               lambda: tf.constant(1.0))
    checked_grads = [tf.math.multiply_no_nan(
        grad, self._grads_coef) for grad in all_grads]
    update_ops = self.net_model.updates
    # for ops in update_ops:
    #   print(ops, flush=True)
    print("%s model update_ops:" % self.mode, len(update_ops), flush=True)
    with tf.control_dependencies(update_ops):
      self._train_op = self._optimizer.apply_gradients(zip(checked_grads, all_params),
                                                       global_step=self.global_step)

  def _forward(self, training):
    mixed_stft_batch_real = tf.real(self.mixed_stft_batch)
    mixed_stft_batch_imag = tf.imag(self.mixed_stft_batch)
    mixed_stft_batch_real = tf.expand_dims(mixed_stft_batch_real, -1)
    mixed_stft_batch_imag = tf.expand_dims(mixed_stft_batch_imag, -1)
    feature_in = tf.concat(
        [mixed_stft_batch_real, mixed_stft_batch_imag], axis=-1)

    net_phasen_out = self.net_model(feature_in, training=training)

    est_clean_mag_batch = tf.multiply(
        self.mixed_mag_batch, net_phasen_out.mag_mask)  # [batch, t, f]
    # [batch, t, f], complex value
    est_complexPhase_batch = net_phasen_out.normalized_complex_phase
    est_clean_stft_batch = tf.multiply(tf.complex(
        est_clean_mag_batch, 0.0), est_complexPhase_batch)
    est_clean_wav_batch = misc_utils.tf_stft2wav(est_clean_stft_batch, PARAM.frame_length,
                                                 PARAM.frame_step, PARAM.fft_length)
    _mixed_wav_length = tf.shape(self.mixed_wav_batch)[1]
    est_clean_wav_batch = est_clean_wav_batch[:, :_mixed_wav_length]

    return FrowardOutputs(est_clean_stft_batch,
                          est_clean_mag_batch,
                          est_clean_wav_batch,
                          est_complexPhase_batch)

  def _get_losses(self):
    est_clean_mag_batch = self._forward_outputs.est_clean_mag_batch
    est_clean_stft_batch = self._forward_outputs.est_clean_stft_batch
    est_clean_wav_batch = self._forward_outputs.est_clean_wav_batch
    # est_complexPhase_batch = self._forward_outputs.est_complexPhase_batch

    # region losses
    self.loss_compressedMag_mse = losses.batch_time_compressedMag_mse(est_clean_mag_batch,
                                                                      self.clean_mag_batch,
                                                                      PARAM.loss_compressedMag_idx)
    self.loss_compressedStft_mse = losses.batch_time_compressedStft_mse(est_clean_stft_batch,
                                                                        self.clean_stft_batch,
                                                                        PARAM.loss_compressedMag_idx)
    self.loss_mag_mse = losses.batch_time_fea_real_mse(
        est_clean_mag_batch, self.clean_mag_batch)
    self.loss_mag_reMse = losses.batch_real_relativeMSE(est_clean_mag_batch, self.clean_mag_batch,
                                                        PARAM.relative_loss_epsilon, PARAM.RL_idx)
    self.loss_stft_mse = losses.batch_time_fea_complex_mse(
        est_clean_stft_batch, self.clean_stft_batch)
    self.loss_stft_reMse = losses.batch_complex_relativeMSE(est_clean_stft_batch, self.clean_stft_batch,
                                                            PARAM.relative_loss_epsilon, PARAM.RL_idx)

    self.loss_wav_L1 = losses.batch_wav_L1_loss(
        est_clean_wav_batch, self.clean_wav_batch)*10.0
    self.loss_wav_L2 = losses.batch_wav_L2_loss(
        est_clean_wav_batch, self.clean_wav_batch)*100.0
    self.loss_wav_reL2 = losses.batch_wav_relativeMSE(est_clean_wav_batch, self.clean_wav_batch,
                                                      PARAM.relative_loss_epsilon, PARAM.RL_idx)

    self.loss_CosSim = losses.batch_CosSim_loss(
        est_clean_wav_batch, self.clean_wav_batch)
    self.loss_SquareCosSim = losses.batch_SquareCosSim_loss(
        est_clean_wav_batch, self.clean_wav_batch)
    self.loss_stCosSim = losses.batch_short_time_CosSim_loss(est_clean_wav_batch, self.clean_wav_batch,
                                                             PARAM.st_frame_length_for_loss,
                                                             PARAM.st_frame_step_for_loss)
    self.loss_stSquareCosSim = losses.batch_short_time_SquareCosSim_loss(est_clean_wav_batch, self.clean_wav_batch,
                                                                         PARAM.st_frame_length_for_loss,
                                                                         PARAM.st_frame_step_for_loss)
    loss_dict = {
        'loss_compressedMag_mse': self.loss_compressedMag_mse,
        'loss_compressedStft_mse': self.loss_compressedStft_mse,
        'loss_mag_mse': self.loss_mag_mse,
        'loss_mag_reMse': self.loss_mag_reMse,
        'loss_stft_mse': self.loss_stft_mse,
        'loss_stft_reMse': self.loss_stft_reMse,
        'loss_wav_L1': self.loss_wav_L1,
        'loss_wav_L2': self.loss_wav_L2,
        'loss_wav_reL2': self.loss_wav_reL2,
        'loss_CosSim': self.loss_CosSim,
        'loss_SquareCosSim': self.loss_SquareCosSim,
        'loss_stCosSim': self.loss_stCosSim,
        'loss_stSquareCosSim': self.loss_stSquareCosSim,
    }
    # endregion losses

    # region sum_loss
    sum_loss = tf.constant(0, dtype=tf.float32)
    sum_loss_names = PARAM.sum_losses
    for i, name in enumerate(sum_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.sum_losses_w) > 0:
        loss_t *= PARAM.sum_losses_w[i]
      sum_loss += loss_t
    # endregion sum_loss

    # region show_losses
    show_losses = []
    show_loss_names = PARAM.show_losses
    for i, name in enumerate(show_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.show_losses_w) > 0:
        loss_t *= PARAM.show_losses_w[i]
      show_losses.append(loss_t)
    show_losses = tf.stack(show_losses)
    # endregion show_losses

    # region stop_criterion_losses
    stop_criterion_losses_sum = tf.constant(0, dtype=tf.float32)
    stop_criterion_loss_names = PARAM.stop_criterion_losses
    for i, name in enumerate(stop_criterion_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.stop_criterion_losses_w) > 0:
        loss_t *= PARAM.stop_criterion_losses_w[i]
      stop_criterion_losses_sum += loss_t
    # endregion stop_criterion_losses

    return Losses(sum_loss=sum_loss,
                  show_losses=show_losses,
                  stop_criterion_loss=stop_criterion_losses_sum)

  def change_lr(self, sess, new_lr):
    sess.run(self._assign_lr, feed_dict={self.new_lr: new_lr})

  @property
  def global_step(self):
    return self._global_step

  @property
  def train_op(self):
    return self._train_op

  @property
  def losses(self):
    return self._losses

  @property
  def optimizer_lr(self):
    return self._optimizer._lr

  @property
  def est_clean_wav_batch(self):
    return self._est_clean_wav_batch

  @property
  def mixed_wav_batch_in(self):
    return self.mixed_wav_batch
