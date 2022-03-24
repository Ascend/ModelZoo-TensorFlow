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

from model.data_utils import get_chunks, load_bin
from model.config import Config


def eval_om(config):
    """Evaluate the performance of om on test set.

    Args:
        config: configuration of eval.

    Returns:
        metrics: (dict) metrics["acc"] = 98.4, ...
    """

    dir_bin_label = './bin_data/labels'

    crf_params = load_bin(config.dir_om_output + '/output_1')
    logits = load_bin(config.dir_om_output + '/output_0')
    labels = load_bin(dir_bin_label, data_type=np.int)

    correct_preds, total_correct, total_preds = 0., 0., 0.
    for label, logit, crf_param in zip(labels, logits, crf_params):
        length = len(label)
        logit = logit.reshape((config.max_sequence_length, 9))[:length]
        crf_param = crf_param.reshape((9, 9))
        lab_pred, viterbi_score = tf.contrib.crf.viterbi_decode(logit, crf_param)

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
    metrics = eval_om(config)
    msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in metrics.items()])
    print(msg)
