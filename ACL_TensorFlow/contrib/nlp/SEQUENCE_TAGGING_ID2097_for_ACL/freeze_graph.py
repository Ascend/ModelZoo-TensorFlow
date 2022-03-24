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
from tensorflow_core.python.platform import gfile

from model.data_utils import pad_sequences
from model.ner_model import NERModel
from model.config import Config


def infer_pb():
    """Inference test of pb."""

    sess = tf.Session()
    with gfile.FastGFile('./pb_model/SEQUENCE_TAGGING.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    # 输入
    words_raw = "I love Paris".strip().split(" ")
    words = [config.processing_word(w) for w in words_raw]
    if type(words[0]) == tuple:
        words = zip(*words)
    char_ids, word_ids = zip(*[words])
    word_ids, sequence_lengths = pad_sequences(word_ids, 0,
                                               max_sequence_length=config.max_sequence_length,
                                               max_word_length=config.max_word_length)
    char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2,
                                           max_sequence_length=config.max_sequence_length,
                                           max_word_length=config.max_word_length)
    input_word_ids = sess.graph.get_tensor_by_name('word_ids:0')
    input_sequence_length = sess.graph.get_tensor_by_name('sequence_lengths:0')
    input_char_ids = sess.graph.get_tensor_by_name('char_ids:0')

    fd = {input_word_ids: word_ids, input_sequence_length: sequence_lengths,
          input_char_ids: char_ids}
    logits = sess.graph.get_tensor_by_name('dense/BiasAdd:0')
    trans_params = sess.graph.get_tensor_by_name('transitions:0')
    logits, trans_params = sess.run([logits, trans_params], feed_dict=fd)
    print(logits)
    print(trans_params)


def ckpt_to_pb(config):
    """Convert ckpt to pb.

    Args:
        config: configuration of pb.
    """
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_ckpt)

    sess = model.sess
    output_nodes = 'dense/BiasAdd,transitions'

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=input_graph_def,
        output_node_names=output_nodes.split(","))

    with tf.gfile.GFile("./pb_model/SEQUENCE_TAGGING.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))

    print("done")


if __name__ == '__main__':
    # create instance of config
    config = Config()
    config.batch_size = 1
    ckpt_to_pb(config)
