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

from npu_bridge.npu_init import *

import codecs

import numpy as np
import tensorflow as tf
import os
import util


class ModelHelper(object):
    VALID_MODEL_TYPE = ["FastText", "TextCNN", "TextRNN", "TextVDCNN",
                        "RegionEmbedding", "AttentiveConvNet", "TextDPCNN",
                        "TextDRNN"]
    VALID_CELL_TYPE = ["gru", "lstm"]
    """Define some common parameters and layers for model
    """

    def __init__(self, config, logger=None):
        self.config = config
        if logger:
            self.logger = logger
        else:
            self.logger = util.Logger(config)

    @staticmethod
    def self_attention(embedding, attention_dimension):
        """Self attention
        Reference "Hierarchical Attention Networks for Document Classification"
        Input shape [n, dimension]
        Output shape [dimension]
        Args:
            embedding: Embeddings to attention to.
            attention_dimension: Attention dimension.
        Returns:
            Embedding after attention
        """
        attention_matrix = tf.contrib.layers.fully_connected(
            inputs=embedding, num_outputs=attention_dimension,
            activation_fn=tf.nn.tanh)
        # TODO(marvinmu): try more init strategy
        attention_vector = tf.Variable(
            tf.random.uniform([attention_dimension], -1.0 / attention_dimension,
                              1.0 / attention_dimension))

        alpha = tf.nn.softmax(
            tf.reduce_sum(tf.multiply(attention_matrix, attention_vector),
                          axis=2, keepdims=True), axis=1)
        attention_embedding = tf.reduce_sum(tf.multiply(embedding, alpha),
                                            axis=1, keepdims=False)
        return attention_embedding

    @staticmethod
    def dropout(layer, dropout_keep_prob):
        """Dropout of layer
        Args:
            layer: Layer to dropout.
            dropout_keep_prob: Dropout keep prob
        Return:
            layer after dropout
        """
        layer = npu_ops.dropout(layer, keep_prob=dropout_keep_prob)
        return layer

    def get_run_config(self):
        """Get run config of tensorflow
        Return:
            tf.contrib.learn.RunConfig
        """
        # session_config = tf.ConfigProto(log_device_placement=True)
        # session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
        # config = tf.estimator.RunConfig().replace(session_config=session_config)


        # gpu_option = tf.GPUOptions(
        #     allow_growth=True,
        #     per_process_gpu_memory_fraction=0.8,
        #     visible_device_list=self.config.train.visible_device_list)
        # session_config = tf.ConfigProto(
        #     gpu_options=gpu_option)
        session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=False)

        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        # 配置1：选择在昇腾AI处理器上执行推理
        custom_op.parameter_map["use_off_line"].b = True

        custom_op.parameter_map["mix_compile_mode"].b = False  #关闭混合计算，根据实际情况配置，默认关闭
        # custom_op.parameter_map["enable_data_pre_proc"].b = True # getnext算子下沉是迭代循环下沉的必要条件
        custom_op.parameter_map["enable_data_pre_proc"].b = False

        # force-fp32
        # custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp32")
        # session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
        # session_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭
 
        # custom_op.parameter_map["iterations_per_loop"].i = 1 #此处设置的值和set_iteration_per_loop设置的iterations_per_loop值保持一致，用于判断是否进行训练迭代下沉

        custom_op.parameter_map["dynamic_input"].b = True
        custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")

        # custom_op.parameter_map["profiling_mode"].b = True  # 开启profiling采集
        #仅采集任务轨迹数据
        # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test_user08/adq/output", "task_trace":"on"}')
        #采集任务轨迹数据和迭代轨迹数据。可先仅采集任务轨迹数据，如果仍然无法分析到具体问题，可再采集迭代轨迹数据
        #custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test_user08/adq/output","task_trace":"on","training_trace":"on","fp_point":"","bp_point":""}')

        # Dump溢出数据
        # custom_op = npu_tf_config.update_custom_op(custom_op, action='overflow')

        # 关闭融合规则
        # session_config = npu_tf_config.session_dump_config(session_config, action='fusion_off')
        # session_config = tf.ConfigProto(
        # device_count={'CPU': 32, 'GPU': 0},
        #              inter_op_parallelism_threads=32,
        #              intra_op_parallelism_threads=32,
        #              use_per_session_threads=32,
        #              gpu_options=None)

        # profiling_options = '{"output":"/home/test_user08/adq/dt","task_trace":"on"}'
        # profiling_config = ProfilingConfig(enable_profiling=True, profiling_options= profiling_options)

        config = NPURunConfig(
            session_config=session_config,
            # mix_compile_mode=True, 
            iterations_per_loop=1,
            # precision_mode="force_fp32",
            precision_mode="allow_fp32_to_fp16",
            enable_data_pre_proc=False,
            # profiling_config=profiling_config,
            )
        return config

    @staticmethod
    def _get_exports_dict(probability):
        """Get export dict, used in serving.
        Args:
            probability: The predict probability by the model.
        Returns:
            exports dict
        """
        key = \
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        exports = {
            "probs": tf.estimator.export.ClassificationOutput(
                scores=probability),
            key: tf.estimator.export.ClassificationOutput(scores=probability)
        }
        return exports

    def get_train_op(self, loss, learning_rate, static_embedding=False,
                     optimizer="Adam"):
        def _get_train_op_func(variables):
            if self.config.optimizer.optimizer == "Adadelta":
                return tf.train.AdadeltaOptimizer(
                    learning_rate, self.config.optimizer.adadelta_decay_rate,
                    self.config.optimizer.adadelta_epsilon).minimize(
                    loss, global_step=tf.compat.v1.train.get_global_step(),
                    var_list=variables)
            return tf.contrib.layers.optimize_loss(
                loss, global_step=tf.compat.v1.train.get_global_step(),
                learning_rate=learning_rate, optimizer=optimizer,
                clip_gradients=self.config.train.clip_gradients,
                variables=variables)

        variables = tf.compat.v1.trainable_variables()
        if static_embedding:
            # For some optimizer such as Adam, tf need cache params.
            # This statement will build all params cache in case of
            # embedding will be updated in the later epochs
            _get_train_op_func(variables)
            variables = [var for var in variables if
                         'EmbeddingLookupTable' not in var.name
                         or 'NonStatic' in var.name]

        return _get_train_op_func(variables)

    @staticmethod
    def softmax_focal_loss(labels, logits, label_size, gamma=2.0, alpha=0.25,
                           epsilon=1.e-9):
        """Softmax focal loss
        references: Focal Loss for Dense Object Detection
        Args:
            labels: ground truth labels, shape of [batch_size, 1]
            logits: model's output, shape of [batch_size, 1, num_cls]
            label_size:
            gamma:
            alpha:
            epsilon:
        Returns:
            shape of [batch_size, 1]
        """
        probs = tf.clip_by_value(tf.nn.softmax(logits), epsilon, 1.)
        one_hot_labels = tf.one_hot(labels, label_size)
        fl = -one_hot_labels * tf.math.log(probs) * tf.pow(1. - probs, gamma) * alpha
        reduced_fl = tf.reduce_sum(fl, axis=2)
        return reduced_fl

    def get_softmax_estimator_spec(self, hidden_layer, mode, labels,
                                   label_size, static_embedding=False,
                                   label_file=None):
        """Build softmax layer after hidden_layer.
        Args:
            hidden_layer: Hidden_layer
            mode: Can be tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
            labels: Golden truth.
            label_size: Candidate label size.
            static_embedding: If true, embedding will not updated.
            label_file: Label file for fixed uniform sampler.
        Returns:
            NPUEstimatorSpec
        """

        weights = tf.compat.v1.get_variable(
            name="hidden_layer_weight",
            shape=[hidden_layer.shape[1], label_size],
            initializer=tf.contrib.layers.xavier_initializer())
        np_prob_biases = np.zeros([label_size])
        # When using init trick of focal loss, try the following code
        # np_prob_biases[0] = (label_size - 1) * (1 - 0.01) / 0.01
        biases = tf.Variable(name="hidden_layer_bias",
                             initial_value=tf.convert_to_tensor(
                                 np_prob_biases, dtype=tf.float32))
        if self.config.model_common.activation == "relu":
            weights = tf.compat.v1.get_variable(
                name="hidden_layer_weight",
                shape=[hidden_layer.shape[1], label_size],
                initializer=tf.random_normal_initializer(stddev=0.01))
            logits = tf.nn.relu(tf.matmul(hidden_layer, weights) + biases)
        elif self.config.model_common.activation == "sigmoid":
            logits = tf.nn.sigmoid(tf.matmul(hidden_layer, weights) + biases)
        elif self.config.model_common.activation == "tanh":
            logits = tf.nn.tanh(tf.matmul(hidden_layer, weights) + biases)
        elif self.config.model_common.activation == "none":
            logits = tf.matmul(hidden_layer, weights) + biases
        else:
            raise TypeError(
                "Unknown active method: %s" %
                self.config.model_common.activation)
        logits = tf.expand_dims(logits, -2)
        predictions = tf.argmax(logits, axis=-1)
        loss, train_op = None, None
        metrics = {}
        if mode != tf.estimator.ModeKeys.PREDICT:
            
            if self.config.train.loss_type == "Sigmoid":
                one_hot_label = tf.one_hot(labels, logits.shape[-1])
                loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=one_hot_label, logits=logits))
            elif self.config.train.loss_type == "Weighted_Sigmoid":
                one_hot_label = tf.one_hot(labels, logits.shape[-1])
                
                sigmoid_out = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=one_hot_label, logits=logits)
                label_num = []
                total = 0
                for line in codecs.open(label_file, "r",
                                    encoding=util.CHARSET):
                    num = int(line.strip("\n").split("\t")[1])
                    label_num.append(num)
                    total += num
                weight = []
                for i, _ in enumerate(label_num):
                    if label_num[i] == 0:
                        label_num[i] = 1
                    weight.append((total-label_num[i])/label_num[i])
                weight = np.array(weight)
                weight = tf.constant(np.log2(weight+1)*0.2, dtype=tf.float32)
                weight = tf.exp(tf.multiply(weight, one_hot_label))
                loss = tf.reduce_mean(tf.multiply(weight, sigmoid_out))

            elif self.config.train.loss_type == "Softmax":
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=labels, logits=logits))
            elif self.config.train.loss_type == "NCE":
                if self.config.train.sampler == "fixed":
                    unigram_probs = []
                    total = 0
                    for line in codecs.open(label_file, "r",
                                            encoding=util.CHARSET):
                        norm = pow(int(line.strip("\n").split("\t")[1]), 1)
                        unigram_probs.append(norm)
                        total += norm
                    for i, _ in enumerate(unigram_probs):
                        unigram_probs[i] = unigram_probs[i] / total
                    sampler = tf.nn.fixed_unigram_candidate_sampler(
                        labels, 1, self.config.train.num_sampled, False,
                        label_size,
                        unigrams=unigram_probs)
                elif self.config.train.sampler == "learned":
                    sampler = tf.nn.learned_unigram_candidate_sampler(
                        labels, 1, self.config.train.num_sampled, False,
                        label_size)
                elif self.config.train.sampler == "log":
                    sampler = tf.nn.log_uniform_candidate_sampler(
                        labels, 1, self.config.train.num_sampled, False,
                        label_size)
                elif self.config.train.sampler == "uniform":
                    sampler = tf.nn.uniform_candidate_sampler(
                        labels, 1, self.config.train.num_sampled, False,
                        label_size)
                else:
                    raise TypeError(
                        "unknown sampler: " + self.config.train.sampler)
                loss = tf.reduce_mean(tf.nn.nce_loss(
                    tf.transpose(weights), biases, labels, hidden_layer,
                    self.config.train.num_sampled, label_size, num_true=1,
                    sampled_values=sampler, remove_accidental_hits=True))
            elif self.config.train.loss_type == "SoftmaxFocalLoss":
                loss = tf.reduce_mean(
                    self.softmax_focal_loss(labels, logits, label_size))
            else:
                raise TypeError(
                    "unknown loss type: " + self.config.train.loss_type)

            if self.config.train.l2_lambda > 0:
                l2_losses = tf.add_n(
                    [tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables() if
                     'bias' not in v.name and 'global_step' not in v.name and
                     'GLOBAL_STEP' not in v.name]) * self.config.train.l2_lambda
                loss = loss + l2_losses

            if abs(self.config.train.decay_rate - 1) > util.EPS:
                learning_rate = tf.compat.v1.train.exponential_decay(
                    self.config.train.learning_rate, tf.compat.v1.train.get_global_step(),
                    self.config.train.decay_steps, self.config.train.decay_rate,
                    staircase=True, name="learning_rate")
            else:
                learning_rate = self.config.train.learning_rate

            train_op = self.get_train_op(
                loss, learning_rate, static_embedding,
                optimizer=self.config.optimizer.optimizer)
            labels = tf.squeeze(labels, -1)
            metrics = {
                "accuracy": tf.compat.v1.metrics.accuracy(labels, predictions)
            }
        probability = tf.sigmoid(logits)
        exports = self._get_exports_dict(probability)
        return NPUEstimatorSpec(
            mode, predictions=probability, loss=loss, train_op=train_op,
            eval_metric_ops=metrics, export_outputs=exports)
