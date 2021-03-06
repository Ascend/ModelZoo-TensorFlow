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
# Author: Qingyong Hu (huqingyong15@outlook.com) 15/11/2019


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import helper_tf_util

from tensorflow.python.framework import graph_util


def inference(xyz0,xyz1,xyz2,xyz3,xyz4,
              neigh_idx0,neigh_idx1,neigh_idx2,neigh_idx3,neigh_idx4,
              sub_idx0,sub_idx1,sub_idx2,sub_idx3,sub_idx4,
            interp_idx0,interp_idx1,interp_idx2,interp_idx3,interp_idx4,
              features,is_training):

    inputs=dict()
    inputs['xyz']=[xyz0,xyz1,xyz2,xyz3,xyz4]
    inputs['neigh_idx']=[neigh_idx0,neigh_idx1,neigh_idx2,neigh_idx3,neigh_idx4]
    inputs['sub_idx']=[sub_idx0,sub_idx1,sub_idx2,sub_idx3,sub_idx4]
    inputs['interp_idx']=[interp_idx0,interp_idx1,interp_idx2,interp_idx3,interp_idx4]


    d_out = [16,64,128,256,512]  # ??????????????????????????????
    # ???????????????????????????6(x,y,z,r,g,b)??????????????????3(x,y,z)
    feature = features
    # ???????????????mlp???????????????????????????8
    feature = tf.layers.dense(feature, 8, activation=None, name='fc0')
    feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
    feature = tf.expand_dims(feature, axis=2)

    # ###########################Encoder############################
    # ?????????
    f_encoder_list = []
    for i in range(5):  # layers???5
        # ??????LFA??????
        f_encoder_i = dilated_res_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i],
                                             'Encoder_layer_' + str(i), is_training)
        # ?????????????????????dataset???????????????????????????????????????????????????id
        f_sampled_i = random_sample(f_encoder_i, inputs['sub_idx'][i])
        # ????????????????????????????????????????????????????????????
        feature = f_sampled_i
        if i == 0:
            f_encoder_list.append(f_encoder_i)
        f_encoder_list.append(f_sampled_i)
    # ###########################Encoder############################

    feature = helper_tf_util.conv2d(f_encoder_list[-1], f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                    'decoder_0',
                                    [1, 1], 'VALID', True, is_training)

    # ###########################Decoder############################
    # ?????????
    f_decoder_list = []
    for j in range(5):
        # ??????????????????????????????
        f_interp_i = nearest_interpolation(feature, inputs['interp_idx'][-j - 1])
        # ??????
        f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([f_encoder_list[-j - 2], f_interp_i], axis=3),
                                                      f_encoder_list[-j - 2].get_shape()[-1].value, [1, 1],
                                                      'Decoder_layer_' + str(j), [1, 1], 'VALID', bn=True,
                                                      is_training=is_training)
        feature = f_decoder_i
        f_decoder_list.append(f_decoder_i)
    # ###########################Decoder############################
    # ????????????????????????3???mlp+dropout??????
    f_layer_fc1 = helper_tf_util.conv2d(f_decoder_list[-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
    f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
    f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
    f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, 13, [1, 1], 'fc', [1, 1], 'VALID', False,
                                        is_training, activation_fn=None)
    f_out = tf.squeeze(f_layer_fc3, [2])
    return f_out


def dilated_res_block(feature, xyz, neigh_idx, d_out, name, is_training):
    f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
    f_pc = building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training)
    f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                 activation_fn=None)
    shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                     activation_fn=None, bn=True, is_training=is_training)
    return tf.nn.leaky_relu(f_pc + shortcut)


def building_block(xyz, feature, neigh_idx, d_out, name, is_training):
    print(neigh_idx)
    print(xyz)
    d_in = feature.get_shape()[-1].value
    f_xyz = relative_pos_encoding(xyz, neigh_idx)
    f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
    f_neighbours = gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
    f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
    f_pc_agg = att_pooling(f_concat, d_out // 2, name + 'att_pooling_1', is_training)

    f_xyz = helper_tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
    f_neighbours = gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
    f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
    f_pc_agg = att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training)
    return f_pc_agg


def relative_pos_encoding(xyz, neigh_idx):
    neighbor_xyz = gather_neighbour(xyz, neigh_idx)
    xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
    relative_xyz = xyz_tile - neighbor_xyz
    relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
    relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
    return relative_feature


def random_sample(feature, pool_idx):
    feature = tf.squeeze(feature, axis=2)
    num_neigh = tf.shape(pool_idx)[-1]
    d = feature.get_shape()[-1]
    batch_size = tf.shape(pool_idx)[0]
    pool_idx = tf.reshape(pool_idx, [batch_size, -1])
    pool_features = tf.batch_gather(feature, pool_idx)
    pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
    pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
    return pool_features


def nearest_interpolation(feature, interp_idx):
    feature = tf.squeeze(feature, axis=2)
    batch_size = tf.shape(interp_idx)[0]
    up_num_points = tf.shape(interp_idx)[1]
    interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
    interpolated_features = tf.batch_gather(feature, interp_idx)
    interpolated_features = tf.expand_dims(interpolated_features, axis=2)
    return interpolated_features


def gather_neighbour(pc, neighbor_idx):
    batch_size = tf.shape(pc)[0]
    num_points = tf.shape(pc)[1]
    d = pc.get_shape()[2].value
    index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
    features = tf.batch_gather(pc, index_input)
    features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
    return features


def att_pooling(feature_set, d_out, name, is_training):
    batch_size = tf.shape(feature_set)[0]
    num_points = tf.shape(feature_set)[1]
    num_neigh = tf.shape(feature_set)[2]
    d = feature_set.get_shape()[3].value
    f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
    att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')
    att_scores = tf.nn.softmax(att_activation, axis=1)
    f_agg = f_reshaped * att_scores
    f_agg = tf.reduce_sum(f_agg, axis=1)
    f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
    f_agg = helper_tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
    return f_agg


def generate_placeholder_ckpt(model_path,new_model_path):

    with tf.Graph().as_default():
        points_xyz=[]
        neigh_idx=[]
        sub_idx=[]
        interp_idx=[]
        ration_1 = [1, 4, 4, 4, 4]
        ration_2 = [4, 4, 4, 4, 2]
        for i in range(5):
            points_xyz.append(tf.placeholder(tf.float32,(3,40960//ration_1[i],3),name='xyz_'+str(i)))
            neigh_idx.append(tf.placeholder(tf.int32,(3,40960//ration_1[i],16),name='neigh_idx_'+str(i)))
            sub_idx.append(tf.placeholder(tf.int32,(3,40960//ration_2[i],16),name='sub_idx_'+str(i)))
            interp_idx.append(tf.placeholder(tf.int32,(3,40960//ration_1[i],1),name='interp_idx_'+str(i)))

        # xyz+??????????????????6
        points_rgb=tf.placeholder(tf.float32,(3,40960,6),name='rgb')

        is_training=tf.constant(False,dtype=tf.bool)

        with tf.variable_scope('layers'):
            logits = inference(
                points_xyz[0],points_xyz[1],points_xyz[2],points_xyz[3],points_xyz[4],
                neigh_idx[0],neigh_idx[1],neigh_idx[2],neigh_idx[3],neigh_idx[4],
                sub_idx[0],sub_idx[1],sub_idx[2],sub_idx[3],sub_idx[4],
                interp_idx[0],interp_idx[1],interp_idx[2],interp_idx[3],interp_idx[4],
                points_rgb,is_training
            )

        probs=tf.nn.softmax(logits,name='probs')


        saver=tf.train.Saver()

        with tf.Session() as sess:
            if model_path:
                saver.restore(sess,model_path)

            saver.save(sess,new_model_path)

model_path='D:/Log/snap-23501'   #randlanet?????????????????????
new_model_path='D:/Log/snap-23501_final' #dataset????????????placeholder??????
generate_placeholder_ckpt(model_path,new_model_path)
