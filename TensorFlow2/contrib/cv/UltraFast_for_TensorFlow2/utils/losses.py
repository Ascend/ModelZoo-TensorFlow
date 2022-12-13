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
# Copyright 2022 Huawei Technologies Co., Ltd
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


import tensorflow as tf
import numpy as np


def focal_loss(gamma=2., alpha=4.):
    # https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/
    
    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        #y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        
        #y_pred = tf.nn.softmax(y_pred, axis=1)

        model_out = y_pred + epsilon
        ce = y_true * -tf.math.log(model_out)
        weight = y_true * tf.math.pow(1. - model_out, gamma)
        fl = alpha * weight * ce
        reduced_fl = tf.math.reduce_max(fl, axis=1)
        return tf.math.reduce_mean(reduced_fl)
    return focal_loss_fixed


def sim_loss(y_true, y_pred):
    #assert y_pred.ndim == 4
    n, c, h, w = y_pred.shape
    losses = []
    
    for i in range(0, w-1):
        losses.append(y_pred[:,:,:,i] - y_pred[:,:,:,i+1])
    
    loss = tf.concat(losses, axis=0)
    
    # huber is smooth l1 loss
    huber = tf.keras.losses.Huber()
    return huber(loss, tf.zeros_like(loss))


def shape_loss(y_true, y_pred):
    n, c, h, w = y_pred.shape
    x = tf.math.softmax(y_pred[:,:c-1,:,:], axis=1)
    embedding = tf.reshape(tf.range(c-1, dtype=tf.float32), (1,-1,1,1))
    pos = tf.math.reduce_sum(x * embedding, axis=1)

    diff_list1 = []
    for i in range(0, h // 2):
        diff_list1.append(pos[:,i,:] - pos[:,i+1,:])
    
    # mae is not exaclty l1, but lets try
    l1 = tf.keras.losses.mae

    loss = 0
    for i in range(len(diff_list1)-1):
        loss += l1(diff_list1[i],diff_list1[i+1])
    loss /= len(diff_list1) - 1
    
    return loss


def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * tf.keras.backend.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * tf.keras.backend.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return tf.keras.backend.mean(tf.keras.backend.sum(loss, axis=-1))

    return categorical_focal_loss_fixed


def ultrafast_loss(y_true, y_pred):
    a = 0.0
    b = 0.0
    
    fl = focal_loss(gamma=2.0, alpha=4.0)
    
    cl = fl(y_true, y_pred)
    sim = sim_loss(y_true, y_pred)
    shp = shape_loss(y_true, y_pred)
    
    return cl #+ b * shp + a * sim
