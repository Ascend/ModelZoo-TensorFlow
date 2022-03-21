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
import os
import logging
import data_utils
import compute_bleu

tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("msame_out_dir", "./msame_out/2022114_11_39_53_737140", "msame out directory")
tf.app.flags.DEFINE_integer("from_vocab_size", 160000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 80000, "French vocabulary size.")

FLAGS = tf.app.flags.FLAGS

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

# The subscript of the bucket corresponds to the serial number of the inference output file, and also corresponds to the serial number of the output node add
bucket_id_to_output_num = {
    0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    1: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    2: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    3: [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
        78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
}

def save_candidates_file(fr_sentence_candidate_list):
    candidatesFilePath = './newstest2013_candidate.fr'
    with open(candidatesFilePath, 'w', encoding='utf-8') as f:
        for fr_sentence_candidate in fr_sentence_candidate_list:
            f.write(fr_sentence_candidate + '\n')

def main():
    dev_set_str_list = data_utils.extract_dev_set(FLAGS.data_dir)
    bleu = compute_bleu.Bleu()
    bleu_score_list = []
    fr_sentence_candidate_list = []

    # Load vocabularies.
    en_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.from" % FLAGS.from_vocab_size)
    fr_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.to" % FLAGS.to_vocab_size)
    en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
    _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)
    for sample_idx, dev_set_str in enumerate(dev_set_str_list):
        en_sentence = dev_set_str[0]
        fr_sentence_references = [dev_set_str[1].split()]

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

        outputs = []
        for output_num in bucket_id_to_output_num[bucket_id]:
            current_out_file_name = '{0:05d}_output_{1}.txt'.format(sample_idx, output_num)
            current_out_file_path = os.path.join(FLAGS.msame_out_dir, current_out_file_name)
            with open(current_out_file_path, 'r', encoding='utf-8') as f:
                l = f.readlines()
                logit = l[0].split(' ')
            logit = logit[0:-1]  # Remove end-of-list newline element
            logit = [np.float32(i) for i in logit]
            output_idx = int(np.argmax(logit))
            outputs.append(output_idx)
        # If there is an EOS symbol in outputs, cut them at that point.
        if data_utils.EOS_ID in outputs:
            outputs = outputs[:outputs.index(data_utils.EOS_ID)]
        fr_sentence_candidate = [tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]
        fr_sentence_candidate_list.append(" ".join(fr_sentence_candidate))
        # Calculate and save the Bleu scores of predicted and real sentences
        bleu_score_list.append(bleu.score(fr_sentence_references, fr_sentence_candidate))
        print("The %d sentence is calculated" % len(bleu_score_list))
    save_candidates_file(fr_sentence_candidate_list)
    bleu_score_avg = np.mean(bleu_score_list)
    print("The average score of Bleu : ", bleu_score_avg)
    print("The average score of Bleu*100 : ", bleu_score_avg * 100)


if __name__ == '__main__':
    main()
