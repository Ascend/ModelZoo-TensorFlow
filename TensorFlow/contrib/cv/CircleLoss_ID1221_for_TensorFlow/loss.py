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

import tensorflow as tf
# from config import *
from train_npuv3 import *
from model import weight_variable
import numpy as np
def feature2similaryMatrix(y_norm):
    return tf.matmul(y_norm,tf.transpose(y_norm))

def reducelogsumexp(x):
    x = tf.math.exp(x)
    x = tf.math.reduce_sum(x)
    x = tf.math.log(x)
    return x

def PairWiseCircleLossv2(y_norm, y_label):
    label_matrix = tf.equal(tf.expand_dims(y_label, 1), tf.expand_dims(y_label, 0))
    # label_matrix = tf.cast(label_matrix, tf.int64)

    similarity_matrix = tf.matmul(y_norm, tf.transpose(y_norm))
    # negative_matrix = tf.linalg.band_part(tf.logical_not(tf.cast(label_matrix, tf.bool)), 0, -1)
    # positive_matrix_1 = label_matrix - tf.linalg.band_part(label_matrix, 0, 0)
    # positive_matrix = tf.linalg.band_part(positive_matrix_1, 0, -1)
    # positive_matrix = tf.cast(positive_matrix, tf.bool)
    # positive_matrix = 

    similarity_matrix_flat = tf.reshape(similarity_matrix, [-1])
    same = tf.reshape(label_matrix, [-1])
    dif = tf.logical_not(same)
    sp = similarity_matrix_flat[same]
    sn = similarity_matrix_flat[dif]

    # 计算代价函数
    op = 1 + m
    on = -m
    ap = tf.clip_by_value(op - sp, clip_value_min=0., clip_value_max=float('inf'))
    an = tf.clip_by_value(sn - on, clip_value_min=0., clip_value_max=float('inf'))
    delta_p = 1 - m
    delta_n = m
    logit_p = - ap * (sp - delta_p) * gamma
    logit_n = an * (sn - delta_n) * gamma
    loss = tf.math.softplus(tf.math.reduce_logsumexp(logit_n) + tf.math.reduce_logsumexp(logit_p))

    recall, precision, f1_score = get_metric(m, similarity_matrix, label_matrix)

    return loss, sp, sn, recall, precision, f1_score

def PairWiseCircleLoss(y_norm, y_label):
    label_matrix = tf.equal(tf.expand_dims(y_label, 1), tf.expand_dims(y_label, 0))
    label_matrix = tf.cast(label_matrix, tf.int64)

    similarity_matrix = tf.matmul(y_norm, tf.transpose(y_norm))
    negative_matrix = tf.linalg.band_part(tf.logical_not(tf.cast(label_matrix, tf.bool)), 0, -1)
    positive_matrix_1 = label_matrix - tf.linalg.band_part(label_matrix, 0, 0)
    positive_matrix = tf.linalg.band_part(positive_matrix_1, 0, -1)
    positive_matrix = tf.cast(positive_matrix, tf.bool)

    similarity_matrix_flat = tf.reshape(similarity_matrix, [-1])
    positive_matrix = tf.reshape(positive_matrix, [-1])
    negative_matrix = tf.reshape(negative_matrix, [-1])

    sp = tf.cond(tf.equal(tf.size(positive_matrix), 0), lambda: 0.0, lambda: similarity_matrix_flat[positive_matrix])
    sn = tf.cond(tf.equal(tf.size(negative_matrix), 0), lambda: 0.0, lambda: similarity_matrix_flat[negative_matrix])

    # 计算代价函数
    op = 1 + m
    on = -m
    ap = tf.clip_by_value(op - sp, clip_value_min=0., clip_value_max=float('inf'))
    an = tf.clip_by_value(sn - on, clip_value_min=0., clip_value_max=float('inf'))
    delta_p = 1 - m
    delta_n = m
    logit_p = - ap * (sp - delta_p) * gamma
    logit_n = an * (sn - delta_n) * gamma
    loss = tf.math.softplus(reducelogsumexp(logit_n) + reducelogsumexp(logit_p))

    recall, precision, f1_score = get_metric(m, similarity_matrix, label_matrix)

    return loss, sp, sn, recall, precision, f1_score


# 取矩阵的上三角矩阵，对角线置False，转bool类型
def get_uppertrimatrix(tensor):
    tensor = tf.cast(tensor,tf.float32)
    tensor = tensor - tf.linalg.band_part(tensor,0,0)
    tensor = tf.linalg.band_part(tensor,0,-1)
    tensor = tf.cast(tensor,tf.bool)
    return tensor

def get_metric(m,similarity_matrix,label_matrix):
    thresh = 0.5
    K = get_uppertrimatrix(similarity_matrix>=thresh)
    K_bar = get_uppertrimatrix(similarity_matrix<thresh)
    K = tf.reshape(K,[-1])
    K_bar = tf.reshape(K_bar,[-1])
    label_matrix_bool = tf.cast(label_matrix,tf.bool)
    F = tf.reshape(label_matrix_bool,[-1])
    F_bar = tf.reshape(tf.logical_not(label_matrix_bool),[-1])
    tp = tf.reduce_sum(tf.cast(F[K],tf.int64)) 
    fp = tf.reduce_sum(tf.cast(F_bar[K],tf.int64))
    fn = tf.reduce_sum(tf.cast(F[K_bar],tf.int64))
    tp = tf.cast(tp,dtype=tf.float32)+ 1e-10
    fp = tf.cast(fp,dtype=tf.float32)+ 1e-10
    fn = tf.cast(fn,dtype=tf.float32)+ 1e-10
    recall = tp / (tp + fn )
    precision = tp /(tp + fp) 
    f1_score = 2 * (recall*precision)/(precision+recall)

    return recall,precision,f1_score

    
def SparseCircleLoss(y_,y_label):
    W = weight_variable([y_.shape[1],num_classes])
    w_norm = tf.math.l2_normalize(W,axis=1,epsilon=1e-12)
    y_similar = tf.matmul(y_, w_norm) # [B,num_classes]
    y_label = tf.cast(y_label,dtype=tf.int64)
    y_mask = tf.one_hot(y_label,num_classes)
    y_mask = tf.cast(y_mask,dtype=tf.bool)
    sp = y_similar[y_mask]
    y_mask_not = tf.logical_not(y_mask)
    sn = y_similar[y_mask_not]
    sp = tf.reshape(sp,(-1,))
    sn = tf.reshape(sn,(-1,))

    # 计算代价函数
    op = 1+m
    on = -m
    ap = tf.clip_by_value(op-sp,clip_value_min=0.,clip_value_max=float('inf'))
    an = tf.clip_by_value(sn-on,clip_value_min=0.,clip_value_max=float('inf'))
    delta_p = 1 - m
    delta_n = m
    logit_p = - ap * (sp - delta_p) * gamma
    logit_n = an * (sn - delta_n) * gamma
    loss = tf.math.softplus(tf.reduce_logsumexp(logit_n) + tf.reduce_logsumexp(logit_p))

    return loss/batch_size,sn,sp

def compute_R1acc(query_features,gallery_features,query_label,gallery_label):
    s = np.matmul(query_features,np.transpose(gallery_features))
    indx = np.argmax(s,1)
    acc = query_label == gallery_label[indx]
    return acc.mean()


