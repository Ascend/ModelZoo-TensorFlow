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


import codecs

import numpy as np
import tensorflow as tf

import util
from model import model_layer


class EmbeddingLayer(object):
    """Define several type of embedding layer.
    """

    def __init__(self, config, logger=None):
        self.config = config
        if logger:
            self.logger = logger
        else:
            self.logger = util.Logger(self.config)

    def get_lookup_table(self, name, vocab_size, dimension, epoch,
                         dict_map=None,
                         pretrained_embedding_file="",
                         mode=tf.estimator.ModeKeys.TRAIN):
        """Get embedding lookup table.
        If pretrained_embedding_file is empty, embedding lookup table will be
        randomly init else embedding lookup table will be init using
        pretrained_embedding_file
        Args:
            name: Name of the lookup table.
            vocab_size: Size of vocab.
            dimension: Dimension of embedding.
            epoch: Training epoch.
            dict_map: Useful when load from embedding file.
            pretrained_embedding_file: Init embedding lookup table
                                       with pre-trained embedding
            mode: Mode of estimator.
        Returns:
            embedding_lookup_table
        """
        bound = 1.0 / pow(dimension, 0.5)
        if self.config.embedding_layer.embedding_uniform_bound != 0:
            bound = self.config.embedding_layer.embedding_uniform_bound
        if dict_map and pretrained_embedding_file and epoch == 1 and \
                mode == tf.estimator.ModeKeys.TRAIN:
            pre_trained_size = 0
            if self.config.embedding_layer.embedding_initializer == "uniform":
                word_embedding = np.random.uniform(
                    -bound, bound, [vocab_size, dimension])
            elif self.config.embedding_layer.embedding_initializer == "normal":
                word_embedding = np.random.normal(
                    0, self.config.embedding_layer.embedding_random_stddev,
                    [vocab_size, dimension])
            else:
                raise TypeError(
                    "Unknown embedding_initializer: %s" %
                    self.config.embedding_layer.embedding_initializer)

            embedding_file = codecs.open(pretrained_embedding_file, "r",
                                         encoding=util.CHARSET)
            pretrained_embedding_dim = len(
                embedding_file.readline().split("\t")[1].split(' '))
            
            assert (pretrained_embedding_dim == dimension)
            for line in embedding_file:
                
                line = line.strip("\n")
                embeddings = line.split("\t")
                embeddings_vec = embeddings[1].split(" ")
                if len(embeddings_vec) != dimension:
                    self.logger.error("Wrong embedding line: %s" % line)
                word = embeddings[0]
                if word not in dict_map:
                    continue
                pre_trained_size += 1
                index = dict_map[word]
                vector = []
                for i in range(len(embeddings_vec)):
                    vector.append(float(embeddings_vec[i]))
                word_embedding[index] = np.array(vector)
            embedding_file.close()
            embedding_lookup_table = tf.Variable(
                name=name + "EmbeddingLookupTable",
                initial_value=tf.convert_to_tensor(word_embedding,
                                                   dtype=tf.float32))

            self.logger.info(
                "Load %s embedding from %s" % (
                    name, pretrained_embedding_file))
            self.logger.info(
                "Total dict size of %s is %d" % (name, vocab_size))
            self.logger.info("Size of pre-trained %s embedding is %d" % (
                name, pre_trained_size))
            self.logger.info(
                "Size of randomly initialize %s embedding is %d" % (
                    name, vocab_size - pre_trained_size))
        else:
            if epoch == 1 and mode == tf.estimator.ModeKeys.TRAIN:
                self.logger.info("Initialize %s embedding randomly" % name)
            if self.config.embedding_layer.embedding_initializer == "uniform":
                tf_initializer = tf.random.uniform_initializer(-bound, bound)
            elif self.config.embedding_layer.embedding_initializer == "xavier":
                tf_initializer = tf.contrib.layers.xavier_initializer()
            elif self.config.embedding_layer.embedding_initializer == "truncated_normal":
                tf_initializer = tf.truncated_normal_initializer(stddev=0.02)
            else:
                tf_initializer = tf.random_normal_initializer(
                    mean=0,
                    stddev=self.config.embedding_layer.embedding_random_stddev)
            embedding_lookup_table = tf.compat.v1.get_variable(
                name=name + "EmbeddingLookupTable",
                shape=[vocab_size, dimension], initializer=tf_initializer)

        return embedding_lookup_table

    def get_vocab_embedding(self, name, vocab_ids, vocab_size, epoch,
                            mode=tf.estimator.ModeKeys.TRAIN,
                            pretrained_embedding_file=None, dict_map=None,
                            begin_padding_size=0, end_padding_size=0,
                            padding_id=0):
        """Get vocab embedding of the vocab_ids
        Args:
            name: Name of the lookup table.
            vocab_ids: Vocab id list.
            vocab_size: Vocab size.
            epoch: Training epoch.
            mode: Mode of estimator.
            pretrained_embedding_file: Init embedding lookup table
                                       with pre-trained embedding.
            dict_map: Vocab dict map.
            begin_padding_size: Begin padding size.
            end_padding_size: End padding size.
            padding_id: Id to padding.
        Returns:
            vocab_embedding of the vocab_ids
        """
        vocab_lookup_table = self.get_lookup_table(
            name, vocab_size, self.config.embedding_layer.embedding_dimension,
            epoch,
            pretrained_embedding_file=pretrained_embedding_file,
            mode=mode, dict_map=dict_map)
        if begin_padding_size > 0 or end_padding_size > 0:
            shapes = vocab_ids.shape.as_list()
            if len(shapes) > 3:
                raise NotImplementedError
            elif len(shapes) == 3:
                padding = [[0, 0], [0, 0],
                           [begin_padding_size, end_padding_size]]
            else:
                padding = [[0, 0], [begin_padding_size, end_padding_size]]
            vocab_ids = tf.pad(vocab_ids, tf.constant(padding),
                               constant_values=padding_id)
        vocab_embedding = tf.nn.embedding_lookup(
            vocab_lookup_table, vocab_ids)
        return vocab_embedding

    def get_vocab_embedding_sparse(self, name, vocab_ids, vocab_size,
                                   epoch, vocab_weights=None,
                                   mode=tf.estimator.ModeKeys.TRAIN,
                                   pretrained_embedding_file=None,
                                   dict_map=None):
        """Get vocab embedding of the vocab_ids
        Args:
            name: Name of the lookup table.
            vocab_ids: Vocab id list.
            vocab_size: Vocab size.
            epoch: Training epoch.
            vocab_weights: Weights for vocab ids.
            mode: Mode of estimator.
            pretrained_embedding_file: Init embedding lookup table
                                       with pre-trained embedding.
            dict_map: Vocab dict map.
        Returns:
            vocab_embedding of the vocab_ids
        """
        vocab_lookup_table = self.get_lookup_table(
            name, vocab_size, self.config.embedding_layer.embedding_dimension,
            epoch, pretrained_embedding_file=pretrained_embedding_file,
            mode=mode, dict_map=dict_map)
        vocab_embedding = tf.nn.embedding_lookup_sparse(
            vocab_lookup_table, vocab_ids, vocab_weights, combiner="sum")
        return vocab_embedding

    def get_context_lookup_table(self, name, dimension, shape,
                                 epoch, mode=tf.estimator.ModeKeys.TRAIN,
                                 initializer=None):
        """Get context embedding lookup table.
        Args:
            name: Name of the lookup table.
            dimension: Dimension of embedding.
            shape: shape of lookup table.
            epoch: Training epoch.
            mode: Mode of estimator.
            initializer: random initializer
        Returns:
            context_embedding_table
        """
        if epoch == 1 and mode == tf.estimator.ModeKeys.TRAIN:
            self.logger.info("Initialize %s context embedding randomly" % name)
        if not initializer:
            initializer = tf.random.uniform_initializer(
                - 1.0 / pow(dimension, 0.5),
                1.0 / pow(dimension, 0.5))
        context_embedding_table = tf.compat.v1.get_variable(
            name + 'ContextEmbedLookupTable',
            shape=shape,
            initializer=initializer)
        return context_embedding_table

    def _get_alignment_embedding(self, vocab_ids, region_size,
                                 sequence_length, lookup_table,
                                 unit_id_bias=None):
        """Get context/word alignment embedding.
        Args:
            vocab_ids: vocab ids.
            region_size: region size.
            sequence_length: max sequence length
            lookup_table: context/word lookup table
            unit_id_bias: context offset
        Returns:
            word/context aligned embedding
        """
        region_radius = int(region_size / 2)
        aligned_seq = map(lambda i:
                          tf.slice(vocab_ids, [0, i - region_radius],
                                   [-1, region_size]),
                          range(region_radius, sequence_length - region_radius))
        aligned_seq = tf.reshape(tf.concat(list(aligned_seq), 1),
                                 [-1, sequence_length - region_radius * 2,
                                  region_size])
        if unit_id_bias is not None:
            aligned_seq = aligned_seq + unit_id_bias
        return tf.nn.embedding_lookup(lookup_table, aligned_seq)

    def get_region_embedding(self, name, vocab_ids, vocab_size, epoch,
                             sequence_length, region_size,
                             region_embedding_mode="WC",
                             mode=tf.estimator.ModeKeys.TRAIN,
                             pretrained_embedding_file="",
                             initializer=None,
                             dict_map=None):
        """Compute Region Embedding
        Args:
            name: Name
            vocab_ids: Vocab id list.
            vocab_size: Vocab size
            epoch: Epoch
            sequence_length: Sequence length
            region_size: Region size
            region_embedding_mode: Can be WC,CW,multi_region
            mode: Can be tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
            pretrained_embedding_file: Init embedding lookup table
                                       with pre-trained embedding
            initializer: Initializer for randomly init.
            dict_map: Useful when load from embedding file.
        Returns:
            region embedding
        """

        region_radius = int(region_size / 2)
        if region_embedding_mode == "WC":
            # get word aligned embedding
            vocab_lookup_table = self.get_lookup_table(
                name, vocab_size,
                self.config.embedding_layer.embedding_dimension, epoch,
                pretrained_embedding_file=pretrained_embedding_file,
                mode=mode, dict_map=dict_map)
            word_aligned_emb = self._get_alignment_embedding(
                vocab_ids, region_size, sequence_length, vocab_lookup_table)
            # get context embedding
            context_lookup_table = self.get_context_lookup_table(
                name, self.config.embedding_layer.embedding_dimension,
                [vocab_size, region_size,
                 self.config.embedding_layer.embedding_dimension],
                epoch, mode, initializer)
            trimmed_seq = \
                vocab_ids[:, region_radius:sequence_length - region_radius]
            context_emb = \
                tf.nn.embedding_lookup(context_lookup_table, trimmed_seq)
            # compute projected embedding
            projected_emb = word_aligned_emb * context_emb
            # max pooling
            region_emb = tf.reduce_max(projected_emb, axis=2)
        elif region_embedding_mode == "CW":
            # get word embedding
            word_lookup_table = self.get_lookup_table(
                name, vocab_size,
                self.config.embedding_layer.embedding_dimension, epoch,
                pretrained_embedding_file=pretrained_embedding_file,
                mode=mode, dict_map=dict_map)
            word_emb = tf.nn.embedding_lookup(
                word_lookup_table,
                tf.slice(vocab_ids, [0, region_radius], [-1, tf.cast(
                    sequence_length - 2 * region_radius, tf.int32)]))
            word_emb = tf.expand_dims(word_emb, 2)
            # get context aligned embedding
            context_lookup_table = self.get_context_lookup_table(
                name, self.config.embedding_layer.embedding_dimension,
                [vocab_size * region_size,
                 self.config.embedding_layer.embedding_dimension],
                epoch, mode, initializer)
            unit_id_bias = \
                np.array([i * vocab_size for i in range(region_size)])
            context_aligned_emb = self._get_alignment_embedding(
                vocab_ids, region_size, sequence_length,
                context_lookup_table, unit_id_bias)
            # compute projected embedding
            projected_emb = context_aligned_emb * word_emb
            # max pooling
            region_emb = tf.reduce_max(projected_emb, axis=2)
        else:
            raise TypeError("Invalid region embedding mode: %s" %
                            region_embedding_mode)
        return region_emb

    def char_embedding_to_token(self, char_embedding, generate_type="cnn",
                                cnn_filter_size=None, cnn_num_filters=None,
                                rnn_cell_type="gru",
                                rnn_sequence_length=None,
                                rnn_cell_dimension=None,
                                rnn_cell_hidden_keep_prob=1.):
        """Char embedding to token embedding
        Args:
            char_embedding: Char embedding, shape is
                [batch_size x token_sequence_length, max_char_length_per_token,
                char embedding size].
            generate_type: How to generate token embedding use char embedding,
                can be: sum, avg, max, cnn, rnn.
            cnn_filter_size: Useful if generate_type is cnn.
            cnn_num_filters: Useful if generate_type is cnn.
            rnn_cell_type: Useful if generate_type is rnn.
            rnn_sequence_length: Useful if generate_type is rnn.
            rnn_cell_dimension: Useful if generate_type is rnn.
            rnn_cell_hidden_keep_prob: Useful if generate_type is rnn.
        Returns:
            Token embedding generated by char embedding, whose first n_dim is 2
                , the first dim is batch_size x token_sequence_length and the
                last dim depend on the generate type and correspond params.
        """
        if generate_type == "sum":
            return tf.reduce_sum(char_embedding, axis=1)
        elif generate_type == "avg":
            return tf.reduce_mean(char_embedding, axis=1)
        elif generate_type == "max":
            return tf.reduce_max(char_embedding, axis=1)
        elif generate_type == "cnn":
            char_embedding = tf.expand_dims(char_embedding, axis=-1)
            filter_shape = \
                [cnn_filter_size, char_embedding.shape[-2], 1,
                 cnn_num_filters]
            char_embedding_cnn = model_layer.convolution(
                char_embedding, filter_shape, use_bias=True,
                activation=tf.nn.relu, name="convolution")
            return tf.reduce_max(char_embedding_cnn, axis=1)
        elif generate_type == "rnn":
            _, output_states = model_layer.recurrent(
                char_embedding, rnn_cell_dimension, rnn_sequence_length,
                cell_type=rnn_cell_type,
                cell_hidden_keep_prob=rnn_cell_hidden_keep_prob,
                name="char_embedding_to_token_rnn",
                use_bidirectional=False)
            return output_states
        else:
            raise TypeError("Wrong generate type in char_embedding_to_token: " +
                            generate_type)
