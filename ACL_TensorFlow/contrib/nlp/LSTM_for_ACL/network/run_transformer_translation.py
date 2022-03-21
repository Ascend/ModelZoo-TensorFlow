# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

# Copyright 2020 Huawei Technologies Co., Ltd
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

import codecs
import datetime
import operator
import os

import numpy as np
import tensorflow.compat.v1 as tf
from absl import flags

from network.tensor2tensor import text_encoder, bleu_hook

FLAGS = flags.FLAGS

__all__ = 'Wmt32kProcessor'


class Wmt32kProcessor(object):
    def __init__(self):
        self.predict_batch_size = FLAGS.predict_batch_size
        self.max_seq_length = FLAGS.max_seq_length
        self.vocab_file = FLAGS.vocab_file

    def get_features(self, preprocess=False):

        def _get_sorted_inputs(filename, delimiter="\n"):
            """Returning inputs sorted according to decreasing length.

            This causes inputs of similar lengths to be processed in the same batch,
            facilitating early stopping for short sequences.

            Longer sequences are sorted first so that if you're going to get OOMs,
            you'll see it in the first batch.

            Args:
                filename: path to file with inputs, 1 per line.
                delimiter: str, delimits records in the file.

            Returns:
                a sorted list of inputs

            """
            print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                            "I", "Getting sorted inputs"))
            with tf.gfile.Open(filename) as f:
                text = f.read()
                records = text.split(delimiter)
                inputs = [record.strip() for record in records]
                # Strip the last empty line.
                if not inputs[-1]:
                    inputs.pop()
            input_lens = [(i, -len(line.split())) for i, line in enumerate(inputs)]
            sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1))
            # We'll need the keys to rearrange the inputs back into their original order
            sorted_keys = {}
            sorted_inputs = []
            for i, (index, _) in enumerate(sorted_input_lens):
                sorted_inputs.append(inputs[index])
                sorted_keys[index] = i
            return sorted_inputs, sorted_keys

        def _decode_batch_input_fn(num_decode_batches, sorted_inputs, vocabulary, batch_size, input_path,
                                   max_seq_length, preprocess):
            """Generator to produce batches of inputs."""
            print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                            "I", "Batch %d" % num_decode_batches))
            input_path = os.path.join(input_path, "input_ids")
            if not os.path.exists(input_path):
                os.makedirs(input_path)
            for b in range(num_decode_batches):
                if (b + 1) % 1000 == 0 or (b + 1) == num_decode_batches:
                    print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3], "I",
                                                    "Start to load result files: %d/%d" % (b + 1, num_decode_batches)))

                batch_inputs = []
                for inputs in sorted_inputs[b * batch_size:(b + 1) * batch_size]:
                    input_ids = vocabulary.encode(inputs)
                    final_id = text_encoder.EOS_ID
                    input_ids.append(final_id)
                    batch_inputs.append(input_ids)
                final_batch_inputs = []
                for input_ids in batch_inputs:

                    if len(input_ids) > max_seq_length:
                        print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                                        "E", "Length of input ids: %d" % len(input_ids)))
                        print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                                        "E", "Length of max sequence length: %d" % max_seq_length))
                        raise AssertionError

                    x = input_ids + [0] * (max_seq_length - len(input_ids))
                    final_batch_inputs.append(x)
                if len(final_batch_inputs) < batch_size:
                    for loop in range(batch_size - len(final_batch_inputs)):
                        x = [1] + [0] * (max_seq_length - 1)
                        final_batch_inputs.append(x)
                if preprocess:
                    np.array(final_batch_inputs).astype(np.int32).tofile(os.path.join(input_path,
                                                                                      "input_ids_%05d.bin" % b))
            print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                            "I", "Preprocess finished, bin file in %s" % input_path))

        preprocess_file = os.path.join(FLAGS.data_dir, "newstest2014.en")
        inputs_vocab = text_encoder.SubwordTextEncoder(self.vocab_file)
        sorted_inputs, sorted_keys = _get_sorted_inputs(preprocess_file)
        num_sentences = len(sorted_inputs)
        num_decode_batches = (num_sentences - 1) // self.predict_batch_size + 1
        _decode_batch_input_fn(
            num_decode_batches,
            sorted_inputs,
            inputs_vocab,
            self.predict_batch_size,
            FLAGS.data_dir,
            self.max_seq_length,
            preprocess
        )

    def calc_precision(self):
        def _read_from_bin(f):
            np_data = np.fromfile(f, np.int32)
            return np_data

        def _load_vocab(s):
            with codecs.open(s, 'r', encoding='UTF-8') as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            return content

        try:
            model_name = (FLAGS.om_model_file.split('/')[-1]).split('.')[0]
        except AttributeError:
            model_name = (FLAGS.pb_model_file.split('/')[-1]).split('.')[0]
        output_pre = os.path.join(FLAGS.output_dir, FLAGS.task_name, model_name)
        output_file_list = []
        for root, dirs, files in os.walk(output_pre):
            for bin_file in files:
                if bin_file.endswith(".bin"):
                    output_file_list.append(os.path.join(root, bin_file))
        output_file_list.sort()

        vocab = _load_vocab(self.vocab_file)

        idx = 0
        output_dir = os.path.join(FLAGS.output_dir, FLAGS.task_name)
        decode_to_file = os.path.join(output_dir, "infer_result.de")
        reference_file = os.path.join(FLAGS.data_dir, "sorted_newstest2014.de")

        with codecs.open(decode_to_file, "w", "utf-8") as target:
            for file in output_file_list:
                if (idx + 1) % 1000 == 0 or (idx + 1) == len(output_file_list):
                    print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3], "I",
                                                    "Start to load result files: %d/%d" % (
                                                        idx + 1, len(output_file_list))))
                output_values = _read_from_bin(file)
                sentence = ""
                for word_value in output_values:
                    if word_value == 1:
                        break
                    sentence += vocab[word_value].split("'")[1].replace("_", " ")
                target.write(sentence + "\n")
                idx += 1

        bleu = 100 * bleu_hook.bleu_wrapper(reference_file, decode_to_file, case_sensitive=False)
        print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                        "I", "BLEU_uncased = %6.2f" % bleu))
        bleu = 100 * bleu_hook.bleu_wrapper(reference_file, decode_to_file, case_sensitive=True)
        print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                        "I", "BLEU_cased = %6.2f" % bleu))

        result_save_file = os.path.join(output_pre, "%s_precision.txt" % model_name)
        fp = open(result_save_file, "w")
        fp.write("BLEU_uncased = %6.2f" % bleu)
        fp.write("BLEU_cased = %6.2f" % bleu)
        fp.close()
