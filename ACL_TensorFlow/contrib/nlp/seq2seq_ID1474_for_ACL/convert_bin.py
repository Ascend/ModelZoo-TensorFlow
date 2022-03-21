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
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import os
import random
import logging
import data_utils

tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("bin_data_dir", "./bin_data", "Bin Data directory")
tf.app.flags.DEFINE_integer("from_vocab_size", 160000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 80000, "French vocabulary size.")

FLAGS = tf.app.flags.FLAGS

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def get_batch_data(data, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = _buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(1):
        encoder_input, decoder_input = random.choice(data[bucket_id])

        # Encoder inputs are padded and then reversed.
        encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
        encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

        # Decoder inputs get an extra "GO" symbol, and are padded then.
        decoder_pad_size = decoder_size - len(decoder_input) - 1
        decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                              [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
        batch_encoder_inputs.append(
            np.array([encoder_inputs[batch_idx][length_idx]
                      for batch_idx in xrange(1)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
        batch_decoder_inputs.append(
            np.array([decoder_inputs[batch_idx][length_idx]
                      for batch_idx in xrange(1)], dtype=np.int32))

        # Create target_weights to be 0 for targets that are padding.
        batch_weight = np.ones(1, dtype=np.float32)
        for batch_idx in xrange(1):
            # We set weight to 0 if the corresponding target is a PAD symbol.
            # The corresponding target is decoder_input shifted by 1 forward.
            if length_idx < decoder_size - 1:
                target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights


def save_bin_data(encoder_inputs, decoder_inputs, target_weights, bucket_id, sample_idx):
    # Check if the sizes match.
    encoder_size, decoder_size = _buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
        raise ValueError("Encoder length must be equal to the one in bucket,"
                         " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
        raise ValueError("Decoder length must be equal to the one in bucket,"
                         " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
        raise ValueError("Weights length must be equal to the one in bucket,"
                         " %d != %d." % (len(target_weights), decoder_size))

    for l in xrange(encoder_size):
        current_encoder_path = os.path.join(FLAGS.bin_data_dir, "encoder{0}".format(l))
        if not os.path.exists(current_encoder_path):
            os.makedirs(current_encoder_path)
        encoder_inputs[l].tofile(os.path.join(current_encoder_path, '{0:05d}.bin'.format(sample_idx)))

    # Complement the following input, so that each input encoder has a value
    for l in range(encoder_size, _buckets[-1][0]):
        current_encoder_path = os.path.join(FLAGS.bin_data_dir, "encoder{0}".format(l))
        if not os.path.exists(current_encoder_path):
            os.makedirs(current_encoder_path)
        # Complement 0,0 is _PAD in the vocabulary
        np.zeros([1], dtype=np.int32).tofile(os.path.join(current_encoder_path, '{0:05d}.bin'.format(sample_idx)))
        # Complement 2,2 is _EOS in the vocabulary
        # np.array([2], dtype=np.int32).tofile(os.path.join(current_encoder_path, '{0:05d}.bin'.format(sample_idx)))

    for l in xrange(decoder_size):
        current_decoder_path = os.path.join(FLAGS.bin_data_dir, "decoder{0}".format(l))
        current_weight_path = os.path.join(FLAGS.bin_data_dir, "weight{0}".format(l))
        if not os.path.exists(current_decoder_path):
            os.makedirs(current_decoder_path)
        if not os.path.exists(current_weight_path):
            os.makedirs(current_weight_path)
        decoder_inputs[l].tofile(os.path.join(current_decoder_path, '{0:05d}.bin'.format(sample_idx)))
        target_weights[l].tofile(os.path.join(current_weight_path, '{0:05d}.bin'.format(sample_idx)))

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target_path = os.path.join(FLAGS.bin_data_dir, "decoder{0}".format(decoder_size))
    if not os.path.exists(last_target_path):
        os.makedirs(last_target_path)
    last_target = np.zeros([1], dtype=np.int32)
    last_target.tofile(os.path.join(last_target_path, '{0:05d}.bin'.format(sample_idx)))


def main():
    """main function to receive params them change data to bin.
    """
    dev_set_str_list = data_utils.extract_dev_set(FLAGS.data_dir)

    # Load vocabularies.
    en_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.from" % FLAGS.from_vocab_size)
    fr_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.to" % FLAGS.to_vocab_size)
    en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
    _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)
    for sample_idx, dev_set_str in enumerate(dev_set_str_list):
        en_sentence = dev_set_str[0]
        fr_sentence_references = [dev_set_str[1].split()]
        # print(fr_sentence_references)

        # Get token-ids for the input sentence.
        token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(en_sentence), en_vocab)
        # Which bucket does it belong to?
        bucket_id = len(_buckets) - 1
        for i, bucket in enumerate(_buckets):
            if bucket[0] >= len(token_ids):
                bucket_id = i
                break
        else:
            logging.warning("Sentence truncated: %s", en_sentence)

        # Get a 1-element batch to feed the sentence to the model.
        encoder_inputs, decoder_inputs, target_weights = get_batch_data(
            {bucket_id: [(token_ids, [])]}, bucket_id)
        save_bin_data(encoder_inputs, decoder_inputs, target_weights, bucket_id, sample_idx)


if __name__ == '__main__':
    main()
