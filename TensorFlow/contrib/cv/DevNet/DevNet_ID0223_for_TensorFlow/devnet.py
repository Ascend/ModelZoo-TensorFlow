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

import numpy as np
import tensorflow as tf

tf.set_random_seed(42)
np.random.seed(42)


class DeviationNetwork(object):
    """
    Definition of DeviationNetwork
    """

    def __init__(self, input_shape, network_depth, learning_rate, label_shape=None, use_npu=False):
        """
        Constructor of DeviationNetwork
        """
        tf.reset_default_graph()
        self.x_input = tf.placeholder(shape=input_shape, dtype=tf.float32, name="input")
        if label_shape:
            self.y_true = tf.placeholder(shape=label_shape, dtype=tf.float32)
        if network_depth == 4:
            self.pred = dev_network_d(self.x_input)
        elif network_depth == 2:
            self.pred = dev_network_s(self.x_input)
        elif network_depth == 1:
            self.pred = dev_network_linear(self.x_input)
        if label_shape:
            self.loss_tensor = deviation_loss(self.pred, self.y_true)
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_ops = opt.minimize(self.loss_tensor)

        self.saver = tf.train.Saver(max_to_keep=1)
        if use_npu :
            from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
            from npu_bridge.estimator import npu_ops
            config = tf.ConfigProto()
            custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
            custom_op.name = "NpuOptimizer"
            custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
            custom_op.parameter_map["use_off_line"].b = True
            config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
            self.sess = tf.Session(config=config)
        else :
            self.sess = tf.Session()
        
        self.sess.run(tf.global_variables_initializer())

    def predict(self, input_data):
        """
        In forward prediction
        """
        return self.sess.run(self.pred, feed_dict={self.x_input: input_data})

    def fit_generator(self, data_generator, epochs, steps_per_epoch, model_path):
        """
        Training By Data Generator
        """
        best_loss = 99.0
        for i in range(epochs):
            for j in range(steps_per_epoch):
                input_data, labels = next(data_generator)
                _, loss = self.sess.run([self.train_ops, self.loss_tensor],
                                        feed_dict={self.x_input: input_data,
                                                   self.y_true: np.expand_dims(labels, axis=-1)})
            if loss < best_loss:
                self.save_weights(model_path)
                best_loss = loss
            print('epoch {}/{}   loss:{:.4f}'.format(i + 1, epochs, loss))
        return loss

    def save_weights(self, path):
        """
        Save Model Weights
        """
        self.saver.save(self.sess, path)

    def load_weights(self, path):
        """
        Load Model Weights
        """   
        self.saver.restore(self.sess, path)


def dev_network_d(input_tensor):
    """
    deeper network architecture with three hidden layers
    """
    intermediates = tf.layers.dense(input_tensor, 1000, activation=tf.nn.relu,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
    intermediates = tf.layers.dense(intermediates, 250, activation=tf.nn.relu,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
    intermediates = tf.layers.dense(intermediates, 20, activation=tf.nn.relu,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
    output = tf.layers.dense(intermediates, 1, name="output")

    return output


def dev_network_s(input_tensor):
    """
    network architecture with one hidden layer
    """
    intermediates = tf.layers.dense(input_tensor, 20, activation=tf.nn.relu,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
    output = tf.layers.dense(intermediates, 1, name="output")

    return output


def dev_network_linear(input_tensor):
    """
    network architecture with no hidden layer, equivalent to linear mapping from
    raw inputs to anomaly scores
    """
    output = tf.layers.dense(input_tensor, 1, name="output")
    return output


def deviation_loss(y_pred, y_true):
    """z-score-based deviation loss"""
    confidence_margin = 5.
    ref = tf.random.normal(mean=0., stddev=1.0, shape=[5000])
    dev = (y_pred - tf.reduce_mean(ref)) / tf.math.reduce_std(ref)
    inlier_loss = tf.math.abs(dev)
    outlier_loss = tf.math.abs(tf.math.maximum(confidence_margin - dev, 0.))
    return tf.reduce_mean((1 - y_true) * inlier_loss + y_true * outlier_loss)
