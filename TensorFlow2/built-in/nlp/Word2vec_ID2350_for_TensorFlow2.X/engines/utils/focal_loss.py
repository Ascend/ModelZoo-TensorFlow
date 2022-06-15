# -*- coding: utf-8 -*-
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
# @Time : 2020/11/25 7:51 下午 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : focal_loss.py 
# @Software: PyCharm
# import npu_device
# npu_device.open().as_default()
import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    """
    focal loss for multi-class classification
    fl(pt) = -alpha*(1-pt)^(gamma)*log(pt)
    :param alpha:
    :param gamma:
    :param epsilon:
    """
    def __init__(self, gamma=2.0, alpha=0.25, epsilon=1.e-9):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        # To avoid divided by zero
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        """
        :param y_true: ground truth one-hot vector shape of [batch_size, nb_class]
        :param y_pred: prediction after softmax shape of [batch_size, nb_class]
        :return:
        """
        y_pred = tf.add(y_pred, self.epsilon)
        # Cross entropy
        ce = -y_true * tf.math.log(y_pred)
        # Not necessary to multiply y_true(cause it will multiply with CE which has set unconcerned index to zero ),
        # but refer to the definition of p_t, we do it
        weight = tf.math.pow(1 - y_pred, self.gamma) * y_true
        # Now fl has a shape of [batch_size, nb_class]
        # alpha should be a step function as paper mentioned, but it doesn't matter like reason mentioned above
        # (CE has set unconcerned index to zero)
        # alpha_step = tf.where(y_true, alpha*np.ones_like(y_true), 1-alpha*np.ones_like(y_true))
        fl = ce * weight * self.alpha
        loss = tf.reduce_sum(fl, axis=1)
        return loss

