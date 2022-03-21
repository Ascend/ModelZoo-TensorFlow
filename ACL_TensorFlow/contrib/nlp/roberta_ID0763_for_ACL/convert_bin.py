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

import pickle
import tensorflow as tf
import json
from tensorflow.contrib import data as contrib_data
import numpy as np

import os

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "base_dir", None,
    "The inputput directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "tf_dir", None,
    "The inputput directory where the model checkpoints will be written.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")


class SquadExample(object):
    """A single training/test example for simple sequence classification.

       For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def input_fn_builder(input_file, seq_length, bsz):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    batch_size = bsz

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    d = d.apply(
        contrib_data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=False))
    return d


def data_from_record(max_seq_length, predict_file):
    # with tf.gfile.Open(predict_feature_left_file, "rb") as fin:
    #    eval_features = pickle.load(fin)
    predict_input = input_fn_builder(
        input_file=predict_file,
        seq_length=max_seq_length,
        bsz=1)
    predict_iterator = predict_input.make_initializable_iterator()
    predict_next_element = predict_iterator.get_next()
    input_id = []
    input_mask = []
    segment_id = []
    unique_ids = []
    with tf.Session() as sess:
        sess.run(predict_iterator.initializer)
        idx = 0
        while True:
            try:
                feature = sess.run(predict_next_element)
                input_ids = feature["input_ids"]
                input_masks = feature["input_mask"]
                segment_ids = feature["segment_ids"]
                unique_id = feature["unique_ids"]
                input_id = np.array(input_ids)
                input_mask = np.array(input_masks)
                segment_id = np.array(segment_ids)
                unique_ids.append(unique_id)

                input_id.tofile(FLAGS.base_dir + "/input_ids/{0:05d}.bin".format(idx))
                input_mask.tofile(FLAGS.base_dir + "/input_masks/{0:05d}.bin".format(idx))
                segment_id.tofile(FLAGS.base_dir + "/segment_ids/{0:05d}.bin".format(idx))
                idx += 1
                '''
                input_id.extend(input_ids)
                input_mask.extend(input_masks)
                segment_id.extend(segment_ids)
                '''
            except:
                print(idx)
                break
    idx_file = os.path.join(FLAGS.base_dir, "idx.txt")
    with open(idx_file, "w") as f:
        for id in unique_ids:
            f.write(str(id[0]) + '\n')
    f.close()


def main():
    """main function to receive params them change data to bin.
    """
    predict_feature_file = os.path.join(FLAGS.tf_dir, "eval.tf_record")
    data_from_record(FLAGS.max_seq_length, predict_feature_file)


if __name__ == '__main__':
    main()
