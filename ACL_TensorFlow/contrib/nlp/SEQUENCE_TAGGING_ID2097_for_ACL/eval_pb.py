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

if type(tf.contrib) != type(tf): tf.contrib._warning = None
tf.get_logger().setLevel('ERROR')

from tensorflow_core.python.platform import gfile
from model.data_utils import get_chunks, load_bin
from model.config import Config


def eval_pb(config):
    """Evaluate the performance of pb on test set.

    Args:
        config: configuration of eval.

    Returns:
        metrics: (dict) metrics["acc"] = 98.4, ...
    """
    dir_bin_input = './bin_data'
    dir_bin_label = './bin_data/labels'

    word_ids = load_bin(dir_bin_input + '/word_ids', data_type=np.int)
    char_ids = load_bin(dir_bin_input + '/char_ids', data_type=np.int)
    sequence_lengths = load_bin(dir_bin_input + '/sequence_lengths', data_type=np.int)
    labels = load_bin(dir_bin_label, data_type=np.int)

    # import graph
    sess = tf.Session()
    with gfile.FastGFile('./pb_model/SEQUENCE_TAGGING.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    correct_preds, total_correct, total_preds = 0., 0., 0.
    for word_id, char_id, sequence_length, label in zip(word_ids, char_ids, sequence_lengths, labels):
        word_id = word_id.reshape((config.batch_size, config.max_sequence_length))
        char_id = char_id.reshape((config.batch_size, config.max_sequence_length, config.max_word_length))

        input_word_ids = sess.graph.get_tensor_by_name('word_ids:0')
        input_sequence_length = sess.graph.get_tensor_by_name('sequence_lengths:0')
        input_char_ids = sess.graph.get_tensor_by_name('char_ids:0')

        fd = {input_word_ids: word_id, input_sequence_length: sequence_length, input_char_ids: char_id}
        logits = sess.graph.get_tensor_by_name('dense/BiasAdd:0')
        trans_params = sess.graph.get_tensor_by_name('transitions:0')
        logit, trans_param = sess.run([logits, trans_params], feed_dict=fd)
        logit = logit.squeeze()[:len(label)]

        lab_pred, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_param)

        lab_chunks = set(get_chunks(label, config.vocab_tags))
        lab_pred_chunks = set(get_chunks(lab_pred, config.vocab_tags))

        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

    return {"precision": 100 * p, "recall": 100 * r, "f1": 100 * f1}


if __name__ == "__main__":
    # create instance of config
    config = Config()
    config.batch_size = 1
    metrics = eval_pb(config)
    msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in metrics.items()])
    print(msg)
