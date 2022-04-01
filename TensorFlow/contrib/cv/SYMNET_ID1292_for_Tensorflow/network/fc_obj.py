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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import os, logging, torch
from collections import OrderedDict

from network.base_network import *
from utils.utils import Embedder
from utils import config as cfg



class Network(BaseNetwork):
    root_logger = logging.getLogger("network %s"%__file__)

    def __init__(self, dataloader, args, feat_dim=None):
        super(Network, self).__init__(dataloader, args, feat_dim)

        self.pos_obj_id   = tf.placeholder(tf.int32, shape=[None])
        self.pos_image_feat = tf.placeholder(tf.float32, shape=[None, self.feat_dim])
        self.lr = tf.placeholder(tf.float32)
        
        self.test_attr_id   = tf.placeholder(tf.int32, shape=[None], name='test_attr_id')
        self.test_obj_id    = tf.placeholder(tf.int32, shape=[None], name='test_obj_id')


    def build_network(self, test_only=False):
        logger = self.logger('create_train_arch')


        ########################## classifier losses ##########################
        score_pos_O = self.MLP(self.pos_image_feat, self.num_obj, 
            is_training=True, name='obj_cls', 
            hidden_layers=self.args.fc_cls)
        prob_pos_O = tf.nn.softmax(score_pos_O, 1)
        
        loss = self.cross_entropy(prob_pos_O, self.pos_obj_id, 
            depth=self.num_obj, weight=self.obj_weight)


        ################################ test #################################
        
        score_pos_O = self.MLP(self.pos_image_feat, self.num_obj,
            is_training=False, name='obj_cls', 
            hidden_layers=self.args.fc_cls)
        prob_pos_O = tf.nn.softmax(score_pos_O, 1)

        batchsize = tf.shape(self.pos_image_feat)[0]
        prob_pos_A = tf.zeros([batchsize, self.num_attr],
            dtype=self.pos_image_feat.dtype)
        score_original = tf.zeros([batchsize, self.num_pair],
            dtype=self.pos_image_feat.dtype)

        score_res = OrderedDict([
            ("score_fc", [score_original, prob_pos_A, prob_pos_O]),
        ])
        

        # summary
        
        with tf.device("/cpu:0"):
            tf.summary.scalar('loss_total', loss)
            tf.summary.scalar('lr', self.lr)
            
        
        train_summary_op = tf.summary.merge_all()


        if test_only:
            return prob_pos_O
        else:
            return loss, score_res, train_summary_op



    def train_step(self, sess, blobs, lr, train_op, train_summary_op):
        summary, _ = sess.run(
            [train_summary_op, train_op],
            feed_dict={
                self.pos_obj_id      : blobs[2],
                self.pos_image_feat  : blobs[9],
                self.lr: lr,
            })

        return summary

    def test_step_no_postprocess(self, sess, blobs, score_op):
        score = sess.run(
            score_op,
            feed_dict={
                self.pos_image_feat: blobs[4],
            })

        return score