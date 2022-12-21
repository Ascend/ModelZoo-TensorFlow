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
import json
import os
import sys
from collections import Counter

import tensorflow as tf

import util
from config import Config

from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig


class DataProcessor(object):
    VOCAB_UNKNOWN = "_UNK"
    VOCAB_PADDING = "_PAD"

    # Text format: label\t[(token )+]\t[(char )+]\t[(custom_feature )+].
    LINE_SPLIT_NUMBER = 4

    # Index of line, also index in dict list
    LABEL_INDEX = 0
    TOKEN_INDEX = 1
    CHAR_INDEX = 2
    CUSTOM_FEATURE_INDEX = 3
    USE_SPECIAL_LABEL = False
    SPECIAL_LABEL_LIST = ["社会", "时政", "国际", "军事"]
    SPECIAL_LABEL = "|".join(SPECIAL_LABEL_LIST)

    # TODO(marvinmu): check config
    def __init__(self, config, logger=None):
        self.config = config
        if logger:
            self.logger = logger
        else:
            self.logger = util.Logger(config)
        self.dict_names = ["label", "token", "char", "custom_feature",
                           "token_ngram", "char_ngram", "char_in_token"]
        self.dict_files = []
        for dict_name in self.dict_names:
            self.dict_files.append(
                self.config.data.dict_dir + "/" + dict_name + ".dict")
        self.label_dict_file = self.dict_files[0]

        # Should keep all labels
        self.min_count = [0, self.config.feature_common.min_token_count,
                          self.config.feature_common.min_char_count,
                          self.config.var_len_feature.min_custom_feature_count,
                          self.config.var_len_feature.min_token_ngram_count,
                          self.config.var_len_feature.min_char_ngram_count,
                          self.config.feature_common.min_char_count_in_token]
        # Should keep all labels
        self.max_dict_size = \
            [1000 * 1000, self.config.feature_common.max_token_dict_size,
             self.config.feature_common.max_char_dict_size,
             self.config.var_len_feature.max_custom_feature_dict_size,
             self.config.var_len_feature.max_token_ngram_dict_size,
             self.config.var_len_feature.max_char_ngram_dict_size,
             self.config.feature_common.max_char_in_token_dict_size]
        # Label and custom feature has no max_sequence_length.
        self.max_sequence_length = \
            [0, self.config.fixed_len_feature.max_token_sequence_length,
             self.config.fixed_len_feature.max_char_sequence_length, 0]
        # Label and custom feature has no ngram.
        self.ngram_list = [0, self.config.var_len_feature.token_ngram,
                           self.config.var_len_feature.char_ngram, 0]
        self.label_map = dict()
        self.token_map = dict()
        self.char_map = dict()
        self.custom_feature_map = dict()
        self.token_gram_map = dict()
        self.char_gram_map = dict()
        self.char_in_token_map = dict()
        self.dict_list = [self.label_map, self.token_map, self.char_map,
                          self.custom_feature_map, self.token_gram_map,
                          self.char_gram_map, self.char_in_token_map]

        self.id_to_label_map = dict()
        self.id_to_token_map = dict()
        self.id_to_char_map = dict()
        self.id_to_custom_feature_map = dict()
        self.id_to_token_gram_map = dict()
        self.id_to_char_gram_map = dict()
        self.id_to_char_in_token_map = dict()
        self.id_to_vocab_dict_list = [
            self.id_to_label_map, self.id_to_token_map,
            self.id_to_char_map, self.id_to_custom_feature_map,
            self.id_to_token_gram_map, self.id_to_char_gram_map,
            self.id_to_char_in_token_map]

        self.train_text_file, self.validate_text_file, self.test_text_file = \
            self.config.data.train_text_file, \
            self.config.data.validate_text_file, \
            self.config.data.test_text_file

        self.tfrecord_files = [
            self.config.data.tfrecord_dir + "/" + os.path.basename(
                self.train_text_file) + ".tfrecord",
            self.config.data.tfrecord_dir + "/" + os.path.basename(
                self.validate_text_file) + ".tfrecord",
            self.config.data.tfrecord_dir + "/" + os.path.basename(
                self.test_text_file) + ".tfrecord"]
        self.train_file, self.validate_file, self.test_file = \
            self.tfrecord_files

        self.feature_files = [
            self.config.data.tfrecord_dir + "/" + os.path.basename(
                self.train_text_file) + ".feature",
            self.config.data.tfrecord_dir + "/" + os.path.basename(
                self.validate_text_file) + ".feature",
            self.config.data.tfrecord_dir + "/" + os.path.basename(
                self.test_text_file) + ".feature"]
        (self.train_feature_file, self.validate_feature_file,
         self.test_feature_file) = self.feature_files

        self.pretrained_embedding_files = [
            "", config.feature_common.token_pretrained_embedding_file,
            config.feature_common.char_pretrained_embedding_file,
            config.var_len_feature.custom_feature_pretrained_embedding_file, ]

        self.int_list_column = ["fixed_len_token", "var_len_token",
                                "char_in_token", "char_in_token_real_len",
                                "fixed_len_char", "var_len_char",
                                "var_len_token_ngram", "var_len_char_ngram",
                                "var_len_custom_feature"]
        self.int_column = ["token_fixed_real_len", "char_fixed_real_len"]
        self.float_column = ["token_var_real_len", "char_var_real_len",
                             "token_ngram_var_real_len",
                             "char_ngram_var_real_len",
                             "custom_feature_var_real_len"]

    def _save_dict(self, dict_file, counter, name):
        """Save all vocab to file.
        Args:
            dict_file: File to save to.
            counter: Vocab counts.
            name: Dict name.
        """
        dict_list = counter.most_common()
        dict_file = codecs.open(dict_file, "w", encoding=util.CHARSET)
        # Save _UNK for vocab not in dict and _PAD for padding
        count = 1000 * 1000  # Must bigger than min count
        if name != "label":
            dict_list = [(self.VOCAB_PADDING, count),
                         (self.VOCAB_UNKNOWN, count)] + dict_list
        for vocab in dict_list:
            dict_file.write("%s\t%d\n" % (vocab[0], vocab[1]))
        dict_file.close()
        self.logger.info("Total count of %s: %d" % (name, len(dict_list)))

    def _load_dict(self, dict_map, id_to_vocab_dict_map, dict_file, min_count,
                   max_dict_size, name):
        """Load dict according to params.
        Args:
            dict_map: Vocab dict map.
            id_to_vocab_dict_map: Id to vocab dict map.
            dict_file: File to load.
            min_count: Vocab whose count is equal or greater than min_count
                       will be loaded.
            max_dict_size: Load top max_dict_size vocabs sorted by count.
            name: Dict name.
        Returns:
            dict.
        """
        if not os.path.exists(dict_file):
            self.logger.warn("Not exists %s for %s" % (dict_file, name))
        else:
            for line in codecs.open(dict_file, "r", encoding=util.CHARSET):
                vocab = line.strip("\n").split("\t")
                try:
                    temp = vocab[1]
                except IndexError:
                    continue
                if int(temp) >= min_count:
                    index = len(dict_map)
                    dict_map[vocab[0]] = index
                    id_to_vocab_dict_map[index] = vocab[0]
                if len(dict_map) >= max_dict_size:
                    self.logger.warn(
                        "Reach the max size(%d) of %s, ignore the rest" % (
                            max_dict_size, name))
                    break
            self.logger.info("Load %d vocab of %s" % (len(dict_map), name))

    def load_all_dict(self):
        """Load all dict.
        """
        for i, dict_name in enumerate(self.dict_names):
            self._load_dict(self.dict_list[i], self.id_to_vocab_dict_list[i],
                            self.dict_files[i], self.min_count[i],
                            self.max_dict_size[i], dict_name)

    def _generate_dict(self, text_file_list):
        """Generate dict and label dict given train text file.
        Save all vocab to files and load dicts.
        Text format: label\t[(token )+]\t[(char )+]\t[(feature )+].
        Label could be flattened or hierarchical which is separated by "--".
        Args:
            text_file_list:
                Text file list, usually only contain train text file.
        """
        sample_size = 0
        label_counter = Counter()
        token_counter = Counter()
        char_in_token_counter = Counter()
        token_ngram_counter = Counter()
        char_counter = Counter()
        char_ngram_counter = Counter()
        custom_feature_counter = Counter()
        counters = [label_counter, token_counter, char_counter,
                    custom_feature_counter, token_ngram_counter,
                    char_ngram_counter, char_in_token_counter]
        for text_file in text_file_list:
            self.logger.info("Generate dict using text file %s" % text_file)
            for line in codecs.open(text_file, "r", encoding='utf8'):
                content = line.strip("\n").split('\t')
                if len(content) != self.LINE_SPLIT_NUMBER:
                    self.logger.error("Wrong line: %s" % line)
                    continue
                sample_size += 1

                for i, _ in enumerate(content):
                    vocabs = content[i].strip().split(" ")
                    counters[i].update(vocabs)
                    if i == self.LABEL_INDEX:
                        # 增加特殊类处理
                        if vocabs[0] in self.SPECIAL_LABEL_LIST and self.USE_SPECIAL_LABEL:
                            vocabs = [self.SPECIAL_LABEL]
                            counters[i].update(vocabs)
                    # If vocab is token, extract char info of each token
                    if i == self.TOKEN_INDEX:
                        char_in_token = []
                        for vocab in vocabs:
                            char_in_token.extend(vocab)
                        char_in_token_counter.update(char_in_token)
                    if self.ngram_list[i] > 1:
                        ngram_list = []
                        for j in range(2, self.ngram_list[i] + 1):
                            ngram_list.extend(["".join(vocabs[k:k + j]) for k in
                                               range(len(vocabs) - j + 1)])
                        counters[i + 3].update(ngram_list)
        for counter in counters:
            if "" in counter.keys():
                counter.pop("")
        self.logger.info("sample size: %d" % sample_size)

        for i, dict_name in enumerate(self.dict_names):
            self._save_dict(self.dict_files[i], counters[i], self.dict_names[i])

    def _get_vocab_id_list(self, dict_map, vocabs, ngram, sequence_length,
                           max_var_length, ngram_dict_map=None,
                           char_in_token_map=None,
                           max_char_sequence_length_per_token=-1):
        """Convert vocab string list to vocab id list.
        Args:
            dict_map: Dict used to map string to id.
            vocabs: Vocab string list.
            ngram: Ngram to use (if bigger than 1).
            sequence_length: List length for fixed length vocab id list.
            max_var_length: List length for var length vocab id list.
            ngram_dict_map: Ngram dict map.
            max_char_sequence_length_per_token:
                    Useful when using char to get token embedding.
        Returns:
            fixed length vocab id list,
            real length of fixed vocab id list,
            var length vocab id list,
            ngram string list.
        """
        if len(dict_map) == 0 or len(vocabs) == 0:
            return [], 0, [], [], [], []
        vocabs_iter = [x for x in vocabs if x in dict_map]
        var_len_vocabs = [dict_map[x] for x in vocabs_iter]
        if len(var_len_vocabs) > max_var_length:
            var_len_vocabs = var_len_vocabs[0:max_var_length]
        if not var_len_vocabs:
            var_len_vocabs.append(dict_map[self.VOCAB_UNKNOWN])

        if len(vocabs) > sequence_length:
            vocabs = vocabs[0:sequence_length]

        fixed_len_vocabs = []
        fixed_len_vocabs.extend(
            [dict_map[x] if x in dict_map else dict_map[self.VOCAB_UNKNOWN]
             for x in vocabs])
        fixed_real_len = len(fixed_len_vocabs)
        if fixed_real_len < sequence_length:
            fixed_len_vocabs.extend([dict_map[self.VOCAB_PADDING]] * (
                sequence_length - len(fixed_len_vocabs)))

        ngram_list = []
        if ngram > 1:
            ngram_list_str = []
            for i in range(2, ngram + 1):
                ngram_list_str.extend(["".join(vocabs[j:j + i]) for j in
                                       range(len(vocabs) - i + 1)])
            ngram_iter = [x for x in ngram_list_str if x in ngram_dict_map]
            ngram_list = [ngram_dict_map[x] for x in ngram_iter]
            if not ngram_list:
                ngram_list.append(ngram_dict_map[self.VOCAB_UNKNOWN])
        char_in_token = []
        char_in_token_real_len = []
        if max_char_sequence_length_per_token > 0:
            length = 0
            for vocab in vocabs:
                length += 1
                chars = []
                chars.extend(
                    [char_in_token_map[x] if x in char_in_token_map else
                     char_in_token_map[self.VOCAB_UNKNOWN] for x in vocab])
                if len(chars) > max_char_sequence_length_per_token:
                    chars = chars[0:max_char_sequence_length_per_token]
                char_in_token_real_len.append(len(chars))
                if len(chars) < max_char_sequence_length_per_token:
                    chars.extend([char_in_token_map[self.VOCAB_PADDING]] * (
                        max_char_sequence_length_per_token - len(chars)))
                char_in_token.extend(chars)
            while length < sequence_length:
                length += 1
                char_in_token.extend(
                    [char_in_token_map[self.VOCAB_PADDING]] *
                    max_char_sequence_length_per_token)
                char_in_token_real_len.append(0)
        return (fixed_len_vocabs, fixed_real_len, var_len_vocabs, ngram_list,
                char_in_token, char_in_token_real_len)

    def _get_features_from_text(self, text, has_label=True):
        """Parse text to features that model can use.
        Args:
            text: Input text
            has_label: If true, result will contain label.
        Returns:
            Features that model can use.
        """
        content = text.split('\t')
        if len(content) != self.LINE_SPLIT_NUMBER:
            self.logger.error("Wrong format line: %s" % text)
            return None

        label_string = content[self.LABEL_INDEX]
        if has_label and label_string not in self.label_map:
            self.logger.error("Wrong label of line: %s" % text)
            return None

        token = content[self.TOKEN_INDEX].strip().split(" ")
        (fixed_len_token, token_fixed_real_len, var_len_token,
         var_len_token_ngram, char_in_token, char_in_token_real_len) = \
            self._get_vocab_id_list(
                self.token_map, token, self.config.var_len_feature.token_ngram,
                self.config.fixed_len_feature.max_token_sequence_length,
                self.config.var_len_feature.max_var_token_length,
                self.token_gram_map, self.char_in_token_map,
                self.config.fixed_len_feature.max_char_length_per_token)

        chars = content[self.CHAR_INDEX].strip().split(" ")
        (fixed_len_char, char_fixed_real_len, var_len_char, var_len_char_ngram,
         _, _) = self._get_vocab_id_list(
            self.char_map, chars, self.config.var_len_feature.char_ngram,
            self.config.fixed_len_feature.max_char_sequence_length,
            self.config.var_len_feature.max_var_char_length,
            self.char_gram_map)

        custom_features = content[self.CUSTOM_FEATURE_INDEX].strip().split(
            " ")
        _, _, var_len_custom_feature, _, _, _ = self._get_vocab_id_list(
            self.custom_feature_map, custom_features, 0, 0, 0,
            self.config.var_len_feature.max_var_custom_feature_length, None)

        feature_sample = dict({
            "fixed_len_token": fixed_len_token,
            "token_fixed_real_len": token_fixed_real_len,
            "var_len_token": var_len_token,
            "token_var_real_len": len(var_len_token),
            "char_in_token": char_in_token,
            "char_in_token_real_len": char_in_token_real_len,
            "var_len_token_ngram": var_len_token_ngram,
            "token_ngram_var_real_len": len(var_len_token_ngram),

            "fixed_len_char": fixed_len_char,
            "char_fixed_real_len": char_fixed_real_len,
            "var_len_char": var_len_char,
            "char_var_real_len": len(var_len_char),
            "var_len_char_ngram": var_len_char_ngram,
            "char_ngram_var_real_len": len(var_len_char_ngram),

            "var_len_custom_feature": var_len_custom_feature,
            "custom_feature_var_real_len": len(var_len_custom_feature)
        })
        if has_label:
            label = self.label_map[content[0]]
            if content[self.LABEL_INDEX] in self.SPECIAL_LABEL_LIST and self.USE_SPECIAL_LABEL:
                label = self.label_map[self.SPECIAL_LABEL]
            feature_sample["label"] = label
        return feature_sample

    def _convert_features_to_tfexample(self, feature_sample,
                                       has_label=True):
        """Convert feature sample to tf.example
        Args:
            feature_sample: Feature sample.
            has_label: If true, result will contain label
        Returns:
            tf.example
        """
        if not feature_sample:
            return None
        tfexample = tf.train.Example()
        for name in self.int_list_column:
            tfexample.features.feature[name].int64_list.value.extend(
                feature_sample[name])
        for name in self.int_column:
            tfexample.features.feature[name].int64_list.value.append(
                feature_sample[name])
        for name in self.float_column:
            tfexample.features.feature[name].float_list.value.append(
                feature_sample[name])
        if has_label:
            tfexample.features.feature["label"].int64_list.value.append(
                feature_sample["label"])
        return tfexample

    def get_tfexample_from_text(self, text, has_label=True):
        feature_sample = self._get_features_from_text(text, has_label)
        tfexample = self._convert_features_to_tfexample(feature_sample,
                                                        has_label)
        return tfexample, feature_sample

    def _get_tfrecord_from_text_file(self, text_file, tfrecord_file,
                                     feature_file):
        """Get tfrecord from text file.
        Text format: label\t[(token )+]\t[(char )+]\t[(feature )+].
        Label could be flattened or hierarchical which is separated by "--".
        Args:
            text_file: Text file.
            tfrecord_file: Tfrecord file to write.
            feature_file: Feature file, will save feature sample for debug.
                          For validate and test evaluation
        """
        self.logger.info("Get tfrecord from text file %s" % text_file)
        writer = tf.io.TFRecordWriter(tfrecord_file)

        sample_size = 0
        with codecs.open(feature_file, "w",
                         encoding=util.CHARSET) as label_file:
            for line in codecs.open(text_file, "r", encoding='utf8'):
                tfexample, feature_sample = self.get_tfexample_from_text(line)
                if tfexample is not None:
                    feature_str = json.dumps(feature_sample, ensure_ascii=False)
                    label_file.write(
                        self.id_to_label_map[feature_sample["label"]] + "\t" +
                        feature_str + "\n")
                    writer.write(tfexample.SerializeToString())
                    sample_size += 1
        writer.close()
        self.logger.info(
            "Text file %s has sample %d" % (text_file, sample_size))

    def process_from_text_file(self, use_exists_dict=True):
        """Process text data to tfrecord for training and generate dicts.
        """
        if not os.path.exists(self.config.data.tfrecord_dir):
            os.makedirs(self.config.data.tfrecord_dir)
        if not os.path.exists(self.config.data.dict_dir):
            os.makedirs(self.config.data.dict_dir)
        if use_exists_dict:
            self.load_all_dict()
        else:
            self._generate_dict([self.config.data.train_text_file])
            # If using pretrained embedding, dict can be generated by all text file.
            # when repeating the result in the paper of textcnn, the following code
            # should be used.
            # self._generate_dict([self.config.data.train_text_file,
            #                      self.config.data.validate_text_file,
            #                      self.config.data.test_text_file])

            self.load_all_dict()
        text_files = [self.config.data.train_text_file,
                      self.config.data.validate_text_file,
                      self.config.data.test_text_file]
        for i, text_file in enumerate(text_files):
            self._get_tfrecord_from_text_file(text_file, self.tfrecord_files[i],
                                              self.feature_files[i])

    @staticmethod
    def _get_feature_spec(has_label):
        """Feature map to parse tf.example
        Args:
            has_label: If true, feature map include label
        Return:
            feature map
        """
        feature_spec = dict({
            "fixed_len_token": tf.io.VarLenFeature(dtype=tf.int64),
            "token_fixed_real_len": tf.io.FixedLenFeature(shape=(1,),
                                                       dtype=tf.int64),
            "var_len_token": tf.io.VarLenFeature(dtype=tf.int64),
            "token_var_real_len": tf.io.FixedLenFeature(shape=(1,),
                                                     dtype=tf.float32),
            "char_in_token": tf.io.VarLenFeature(dtype=tf.int64),
            "char_in_token_real_len": tf.io.VarLenFeature(dtype=tf.int64),
            "var_len_token_ngram": tf.io.VarLenFeature(dtype=tf.int64),
            "token_ngram_var_real_len": tf.io.FixedLenFeature(shape=(1,),
                                                           dtype=tf.float32),

            "fixed_len_char": tf.io.VarLenFeature(dtype=tf.int64),
            "char_fixed_real_len": tf.io.FixedLenFeature(shape=(1,),
                                                      dtype=tf.int64),
            "var_len_char": tf.io.VarLenFeature(dtype=tf.int64),
            "char_var_real_len": tf.io.FixedLenFeature(shape=(1,),
                                                    dtype=tf.float32),
            "var_len_char_ngram": tf.io.VarLenFeature(dtype=tf.int64),
            "char_ngram_var_real_len": tf.io.FixedLenFeature(shape=(1,),
                                                          dtype=tf.float32),

            "var_len_custom_feature": tf.io.VarLenFeature(dtype=tf.int64),
            "custom_feature_var_real_len": tf.io.FixedLenFeature(shape=(1,),
                                                              dtype=tf.float32),
        })

        if has_label:
            feature_spec["label"] = tf.io.FixedLenFeature(shape=(1,),
                                                       dtype=tf.int64)
        return feature_spec

    def check_tfrecord(self, file_names, field_name, dtype=tf.int32):
        """Check one field of tfrecord
        Args:
            file_names: List of file names.
            field_name: Field to check.
            dtype: Field data type.
        """
        filename_queue = tf.train.string_input_producer(file_names,
                                                        shuffle=False)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.io.parse_single_example(serialized_example,
                                           self._get_feature_spec(True))
        feature = tf.cast(features[field_name], dtype)
        check_file = codecs.open("tf_check.txt", "w", encoding=util.CHARSET)
        # with tf.Session(config=tf.compat.v1.ConfigProto(
        # device_count={"CPU":12},
        # inter_op_parallelism_threads=1,
        # intra_op_parallelism_threads=1,
        # gpu_options=gpu_options,
        # )) as sess:


        config_npu = tf.ConfigProto(allow_soft_placement=True)
        custom_op = config_npu.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        config_npu.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
        config_npu.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭

        with tf.Session(config=config_npu) as sess:          
            init_op = tf.global_variable_initializer()
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            check_file.write(feature.eval())
            coord.request_stop()
            coord.join(threads)
        check_file.close()

        #with tf.Session() as sess:

        #     init_op = tf.global_variables_initializer()
        #     sess.run(init_op)
        #     coord = tf.train.Coordinator()
        #     threads = tf.train.start_queue_runners(coord=coord)
        #     check_file.write(feature.eval())
        #     coord.request_stop()
        #     coord.join(threads)
        # check_file.close()

    def _parse_tfexample(self, example, mode=tf.estimator.ModeKeys.TRAIN):
        """Parse input example.
        Args:
            example: Tf.example.
            mode: Estimator mode.
        Return:
            parsed feature and label.
        """
        parsed = tf.io.parse_single_example(example, self._get_feature_spec(True))
        parsed = self._sparse_to_dense(parsed)
        label = None
        if mode != tf.estimator.ModeKeys.PREDICT:
            label = parsed.pop("label")
        return parsed, label

    def _sparse_to_dense(self, parsed_example):
        for key in self.int_list_column:
            if "var" not in key:
                parsed_example[key] = tf.sparse.to_dense(
                    parsed_example[key])
        return parsed_example

    def dataset_input_fn(self, mode, input_file, batch_size, num_epochs=1):
        """Input function using tf.dataset for estimator
        Args:
            mode: input mode of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}.
            input_file: Input tfrecord file.
            batch_size: Batch size for model.
            num_epochs: Number epoch.
        Returns:
            tf.dataset
        """
        dataset = tf.data.TFRecordDataset(input_file)
        dataset = dataset.map(self._parse_tfexample)
        if mode != tf.estimator.ModeKeys.PREDICT:
            dataset = dataset.shuffle(
                buffer_size=self.config.data.shuffle_buffer)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset

    def serving_input_receiver_fn(self):
        """Input function for session server
        Returns:
            input_receiver_fn for session server
        """
        serialized_tf_example = tf.placeholder(
            dtype=tf.string, name='input_example_tensor')
        receiver_tensors = {'examples': serialized_tf_example}
        parsed_example = tf.parse_example(serialized_tf_example,
                                          self._get_feature_spec(False))
        parsed_example = self._sparse_to_dense(parsed_example)
        return tf.estimator.export.ServingInputReceiver(parsed_example,
                                                        receiver_tensors)


def main(_):
    config = Config(config_file=sys.argv[1])
    data_processor = DataProcessor(config)
    data_processor.process_from_text_file()


if __name__ == '__main__':
    tf.app.run()
