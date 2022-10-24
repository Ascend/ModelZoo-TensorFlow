# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

from absl import flags
from absl import app
import tensorflow.compat.v1 as tf
from smith import experiment_config_pb2
from smith import utils
from npu_bridge.npu_init import *
from smith.modeling import build_smith_dual_encoder
from tensorflow.python.framework import graph_util

FLAGS = flags.FLAGS

flags.DEFINE_string("dual_encoder_config_file", None, "The proto config file for dual encoder SMITH models.")
flags.DEFINE_string("ckpt_path", None, "The NPU ckpt file.")
flags.DEFINE_string("output_graph", "smith.pb", "The output path of pb file.")

ckpt_path = FLAGS.ckpt_path
output_graph = FLAGS.output_graph


def main(_argv):

    input_ids_1 = tf.placeholder(tf.int32, shape=(32, 2048), name="input_ids_1")
    input_mask_1 = tf.placeholder(tf.int32, shape=(32, 2048), name="input_mask_1") 
    input_ids_2 = tf.placeholder(tf.int32, shape=(32, 2048), name="input_ids_2")
    input_mask_2 = tf.placeholder(tf.int32, shape=(32, 2048), name="input_mask_2")
    exp_config = utils.load_config_from_file(FLAGS.dual_encoder_config_file, experiment_config_pb2.DualEncoderConfig())
    tf.logging.info("*** Features ***")
    masked_lm_positions_1 = tf.zeros([1])
    masked_lm_ids_1 = tf.zeros([1])
    masked_lm_weights_1 = tf.zeros([1])
    masked_lm_positions_2 = tf.zeros([1])
    masked_lm_ids_2 = tf.zeros([1])
    masked_lm_weights_2 = tf.zeros([1])
    documents_match_labels = tf.placeholder(tf.float32, shape=(32, 1), name="documents_match_labels")
    (masked_lm_loss_1, masked_lm_loss_2, masked_lm_example_loss_1,
     masked_lm_example_loss_2, masked_lm_weights_1, masked_lm_weights_2,
     masked_sent_lm_loss_1, masked_sent_lm_loss_2,
     masked_sent_per_example_loss_1, masked_sent_per_example_loss_2,
     masked_sent_weight_1, masked_sent_weight_2, seq_embed_1, seq_embed_2,
     input_sent_embed_1, input_sent_embed_2, output_sent_embed_1,
     output_sent_embed_2, siamese_loss,
     siamese_example_loss, siamese_logits) = build_smith_dual_encoder(
        exp_config, "finetune", True, input_ids_1,
        input_mask_1, masked_lm_positions_1, masked_lm_ids_1,
        masked_lm_weights_1, input_ids_2, input_mask_2,
        masked_lm_positions_2, masked_lm_ids_2, masked_lm_weights_2,
        False, documents_match_labels, False)
    predicted_score = tf.sigmoid(siamese_logits)
    predicted_class = tf.round(predicted_score)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=["seq_rep_from_bert_doc_dense/l2_normalize_1", "Sigmoid", "Round"])
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    app.run(main)
