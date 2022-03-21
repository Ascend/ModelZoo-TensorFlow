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
import os
from absl import app
from absl import flags
import modeling
import tensorflow as tf

from tensorflow.python.tools import freeze_graph

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model will be written.")

flags.DEFINE_string(
    "ckpt_dir", None,
    "Initial checkpoint.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

FLAGS = flags.FLAGS


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    final_hidden = model.get_sequence_output()

    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    output_weights = tf.get_variable(
        "cls/squad/output_weights", [2, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

    final_hidden_matrix = tf.reshape(final_hidden,
                                     [batch_size * seq_length, hidden_size])
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, seq_length, 2])
    logits = tf.transpose(logits, [2, 0, 1], name="logits")

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    return (start_logits, end_logits)


def main(_):
    ckpt_path = os.path.join(FLAGS.ckpt_dir)
    max_seq_length = FLAGS.max_seq_length
    input_ids = tf.placeholder(tf.int32, [None, max_seq_length], "input_ids")
    input_mask = tf.placeholder(tf.int32, [None, max_seq_length], "input_mask")
    segment_ids = tf.placeholder(tf.int32, [None, max_seq_length], "segment_ids")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    (start_logits, end_logits) = create_model(
        bert_config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=False)
    # 定义网络的输出节点
    # predict_class = tf.argmax(logits, axis=1, output_type=tf.int32, name="output")

    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, FLAGS.output_dir, 'model.pb')  # 通过write_graph生成模型文件
        freeze_graph.freeze_graph(
            input_graph=os.path.join(FLAGS.output_dir, "model.pb"),  # 传入write_graph生成的模型文件
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,  # 传入训练生成的checkpoint文件
            output_node_names="logits",
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=os.path.join(FLAGS.output_dir, "roberta.pb"),  # 改为需要生成的推理网络的名称
            clear_devices=False,
            initializer_nodes='')


if __name__ == "__main__":
    app.run(main)
