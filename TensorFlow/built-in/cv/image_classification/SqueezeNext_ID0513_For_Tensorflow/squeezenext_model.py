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

import tensorflow as tf

slim = tf.contrib.slim
metrics = tf.contrib.metrics
import squeezenext_architecture as squeezenext
from optimizer  import PolyOptimizer
from dataloader import ReadTFRecords
import tools
import os
metrics = tf.contrib.metrics

class Model(object):
    def __init__(self, config, batch_size):
        self.image_size = config["image_size"]
        self.num_classes = config["num_classes"]
        self.batch_size = batch_size
        self.read_tf_records = ReadTFRecords(self.image_size, self.batch_size, self.num_classes)

    def define_batch_size(self, features, labels):
        """
        Define batch size of dictionary
        :param features:
            Feature dict
        :param labels:
            Labels dict
        :return:
            (features,label)
        """
        features = tools.define_first_dim(features, self.batch_size)
        labels = tools.define_first_dim(labels, self.batch_size)
        return (features, labels)

    def input_fn(self, file_pattern,training):
        """
        Input fn of model
        :param file_pattern:
            Glob file pattern
        :param training:
            Whether or not the model is training
        :return:
            Input generator
        """
        return self.define_batch_size(*self.read_tf_records(file_pattern,training=training))

    def model_fn(self, features, labels, mode, params):
        """
        Function to create squeezenext model and setup training environment
        :param features:
            Feature dict from estimators input fn
        :param labels:
            Label dict from estimators input fn
        :param mode:
            What mode the model is in tf.estimator.ModeKeys
        :param params:
            Dictionary of parameters used to configurate the network
        :return:
            Train op, predictions, or eval op depening on mode
        """

        training = mode == tf.estimator.ModeKeys.TRAIN
        # init model class
        model = squeezenext.SqueezeNext(self.num_classes, params["block_defs"], params["input_def"], params["groups"],params["seperate_relus"])
        # create model inside the argscope of the model
        with slim.arg_scope(squeezenext.squeeze_next_arg_scope(training)):
            predictions,endpoints = model(features["image"], training)

        # output predictions
        if mode == tf.estimator.ModeKeys.PREDICT:
            _,top_5 =  tf.nn.top_k(predictions,k=5)
            predictions = {
                'top_1': tf.argmax(predictions, -1),
                'top_5': top_5,
                 'probabilities': tf.nn.softmax(predictions),
                'logits': predictions,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # create loss (should be equal to caffe softmaxwithloss)
        loss = tf.losses.softmax_cross_entropy(tf.squeeze(labels["class_vec"],axis=1), predictions)

        # create histogram of class spread
        tf.summary.histogram("classes",labels["class_idx"])

        if training:
            # init poly optimizer
            optimizer = PolyOptimizer(params)
            # define train op
            train_op = optimizer.optimize(loss, training, params["total_steps"])

            # if params["output_train_images"] is true output images during training
            if params["output_train_images"]:
                tf.summary.image("training", features["image"])
            
            from utils import stats, fine_tune           
            stats_hook = stats.ModelStats("squeezenext", params["model_dir"], self.batch_size)
            # setup fine tune scaffold
            scaffold = tf.train.Scaffold(init_op=None, init_fn=fine_tune.init_weights("squeezenext", params["fine_tune_ckpt"]))

            # create estimator training spec, which also outputs the model_stats of the model to params["model_dir"]
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[stats_hook], scaffold=scaffold)



        if mode == tf.estimator.ModeKeys.EVAL:
            exit(0)
            # Define the metrics:
            metrics_dict = {
                'Recall@1': tf.metrics.accuracy(tf.argmax(predictions, axis=-1), labels["class_idx"][:, 0]),
                'Recall@5': metrics.streaming_sparse_recall_at_k(predictions, tf.cast(labels["class_idx"], tf.int64),
                                                                 5)
            }
            # output eval images
            eval_summary_hook = tf.train.SummarySaverHook(
                save_steps=100,
                output_dir=os.path.join(params["model_dir"],"eval"),
                summary_op=tf.summary.image("validation", features["image"]))

            #return eval spec
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics_dict,
                evaluation_hooks=[eval_summary_hook])
