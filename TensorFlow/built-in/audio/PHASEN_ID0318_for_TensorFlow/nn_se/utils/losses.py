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

def vec_dot_mul(y1, y2):
  dot_mul = tf.reduce_sum(tf.multiply(y1, y2), -1)
  return dot_mul

def vec_normal(y):
  normal_ = tf.sqrt(tf.reduce_sum(tf.square(y), -1))
  return normal_

def batch_time_compressedMag_mse(y1, y2, compress_idx):
  """
  y1>=0: real, [batch, time, feature_dim]
  y2>=0: real, [batch, time, feature_dim]
  """
  y1 = tf.pow(y1, compress_idx)
  y2 = tf.pow(y2, compress_idx)
  loss = tf.square(y1-y2)
  loss = tf.reduce_mean(tf.reduce_sum(loss, 0))
  return loss


def batch_time_compressedStft_mse(y1, y2, compress_idx):
  """
  y1: complex, [batch, time, feature_dim]
  y2: complex, [batch, time, feature_dim]
  """
  y1_abs_cpr = tf.pow(tf.abs(y1), compress_idx)
  y2_abs_cpr = tf.pow(tf.abs(y2), compress_idx)
  y1_angle = tf.angle(y1)
  y2_angle = tf.angle(y2)
  y1_cpr = tf.complex(y1_abs_cpr, 0.0) * tf.exp(tf.complex(0.0, y1_angle))
  y2_cpr = tf.complex(y2_abs_cpr, 0.0) * tf.exp(tf.complex(0.0, y2_angle))
  y1_con = tf.concat([tf.real(y1_cpr), tf.imag(y1_cpr)], -1)
  y2_con = tf.concat([tf.real(y2_cpr), tf.imag(y2_cpr)], -1)
  loss = tf.square(y1_con-y2_con)
  loss = tf.reduce_mean(tf.reduce_sum(loss, 0))
  return loss


def batch_time_fea_real_mse(y1, y2):
  """
  y1: real, [batch, time, feature_dim]
  y2: real, [batch, time, feature_dim]
  """
  loss = tf.square(y1-y2)
  loss = tf.reduce_mean(tf.reduce_sum(loss, 0))
  return loss

def batch_time_fea_complex_mse(y1, y2):
  """
  y1: complex, [batch, time, feature_dim]
  y2: conplex, [batch, time, feature_dim]
  """
  y1_real = tf.math.real(y1)
  y1_imag = tf.math.imag(y1)
  y2_real = tf.math.real(y2)
  y2_imag = tf.math.imag(y2)
  loss_real = batch_time_fea_real_mse(y1_real, y2_real)
  loss_imag = batch_time_fea_real_mse(y1_imag, y2_imag)
  loss = loss_real + loss_imag
  return loss

def batch_real_relativeMSE(y1, y2, RL_epsilon, index_=2.0):
  # y1, y2 : [batch, time, feature]
  # refer_sum = tf.maximum(tf.abs(y1)+tf.abs(y2),1e-12)
  # small_val_debuff = tf.pow(refer_sum*RL_epsilon*1.0,-1.0)+1.0-tf.pow(RL_epsilon*1.0,-1.0)
  # relative_loss = tf.abs(y1-y2)/refer_sum/small_val_debuff
  relative_loss = tf.abs(y1-y2)/(tf.abs(y1)+tf.abs(y2)+RL_epsilon)
  cost = tf.reduce_mean(tf.reduce_sum(tf.pow(relative_loss, index_), 0))
  return cost

def batch_complex_relativeMSE(y1, y2, RL_epsilon, index_=2.0):
  """
  y1: complex, [batch, time, feature_dim]
  y2: conplex, [batch, time, feature_dim]
  """
  y1_real = tf.math.real(y1)
  y1_imag = tf.math.imag(y1)
  y2_real = tf.math.real(y2)
  y2_imag = tf.math.imag(y2)
  loss_real = batch_real_relativeMSE(y1_real, y2_real, RL_epsilon)
  loss_imag = batch_real_relativeMSE(y1_imag, y2_imag, RL_epsilon)
  loss = 0.5*loss_real+0.5*loss_imag
  return loss

def batch_wav_L1_loss(y1, y2):
  loss = tf.reduce_mean(tf.reduce_sum(tf.abs(y1-y2), 0))
  return loss

def batch_wav_L2_loss(y1, y2):
  loss = tf.reduce_mean(tf.reduce_sum(tf.square(y1-y2), 0))
  return loss

def batch_wav_relativeMSE(y1, y2, RL_epsilon, index_=2.0):
  loss = batch_real_relativeMSE(y1, y2, RL_epsilon, index_=index_)
  return loss

def batch_CosSim_loss(est, ref): # -cos
  '''
  est, ref: [batch, ..., n_sample]
  '''
  cos_sim = - tf.divide(vec_dot_mul(est, ref), # [batch, ...]
                        tf.multiply(vec_normal(est), vec_normal(ref)))
  loss = tf.reduce_mean(cos_sim)
  return loss

def batch_SquareCosSim_loss(est, ref): # -cos^2
  loss_s1 = - tf.divide(tf.square(vec_dot_mul(est, ref)),  # [batch, ...]
                        tf.multiply(vec_dot_mul(est, est),
                                    vec_dot_mul(ref, ref)))
  loss = tf.reduce_mean(loss_s1)
  return loss

def batch_short_time_CosSim_loss(est, ref, st_frame_length, st_frame_step): # -cos
  st_est = tf.signal.frame(est, frame_length=st_frame_length, # [batch, frame, st_wav]
                           frame_step=st_frame_step, pad_end=True)
  st_ref = tf.signal.frame(ref, frame_length=st_frame_length,
                           frame_step=st_frame_step, pad_end=True)
  loss = batch_CosSim_loss(st_est, st_ref)
  return loss

def batch_short_time_SquareCosSim_loss(est, ref, st_frame_length, st_frame_step): # -cos^2
  st_est = tf.signal.frame(est, frame_length=st_frame_length, # [batch, frame, st_wav]
                           frame_step=st_frame_step, pad_end=True)
  st_ref = tf.signal.frame(ref, frame_length=st_frame_length,
                           frame_step=st_frame_step, pad_end=True)
  loss = batch_SquareCosSim_loss(st_est, st_ref)
  return loss
