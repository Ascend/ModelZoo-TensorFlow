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

import time
import tensorflow as tf

if type(tf.contrib) != type(tf): tf.contrib._warning = None
tf.get_logger().setLevel('ERROR')

from .data_utils import minibatches, pad_sequences, get_chunks
from .base_model import BaseModel

# from npu_bridge.npu_init import *
# from npu_bridge.estimator import npu_ops


class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}

    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.compat.v1.placeholder(tf.int32,
                                                 shape=[self.config.batch_size, self.config.max_sequence_length],
                                                 name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.compat.v1.placeholder(tf.int32, shape=[self.config.batch_size],
                                                         name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.compat.v1.placeholder(tf.int32, shape=[self.config.batch_size,
                                                                  self.config.max_sequence_length,
                                                                  self.config.max_word_length],
                                                 name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.compat.v1.placeholder(tf.int32,
                                                     shape=[self.config.batch_size, self.config.max_sequence_length],
                                                     name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.compat.v1.placeholder(tf.int32,
                                               shape=[self.config.batch_size, self.config.max_sequence_length],
                                               name="labels")

        # hyper parameters
        self.dropout = tf.compat.v1.placeholder_with_default(1.0, shape=[],
                                                             name="dropout")
        self.lr = tf.compat.v1.placeholder(dtype=tf.float32, shape=[],
                                           name="lr")

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0,
                                                       max_sequence_length=self.config.max_sequence_length,
                                                       max_word_length=self.config.max_word_length)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2,
                                                   max_sequence_length=self.config.max_sequence_length,
                                                   max_word_length=self.config.max_word_length)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0,
                                                       max_sequence_length=self.config.max_sequence_length,
                                                       max_word_length=self.config.max_word_length)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0,
                                      max_sequence_length=self.config.max_sequence_length,
                                      max_word_length=self.config.max_word_length)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def add_embeddings_op(self):
        """Defines self.embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.compat.v1.variable_scope("words"):
            if self.config.embeddings is None:
                print("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.compat.v1.get_variable(
                    name="_word_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                    self.config.embeddings,
                    name="_word_embeddings",
                    dtype=tf.float32,
                    trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                                                     self.word_ids, name="word_embeddings")

        with tf.compat.v1.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.compat.v1.get_variable(
                    name="_char_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                         self.char_ids, name="char_embeddings")

                char_embeddings = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Conv1D(kernel_size=self.config.conv_kernel_size,
                                           filters=self.config.conv_filter_num,
                                           padding='same', activation='tanh', strides=1))(char_embeddings)
                char_embeddings = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D
                                                                  (self.config.max_word_length))(char_embeddings)
                char_embeddings = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(char_embeddings)

                word_embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)

        self.word_embeddings = tf.nn.dropout(word_embeddings, rate=1-self.dropout)

    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.compat.v1.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.word_embeddings,
                sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)

            self.word_embeddings = tf.nn.dropout(output, rate=1-self.dropout)

        self.logits = tf.layers.dense(output, self.config.ntags)

    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With the CRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params  # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.compat.v1.summary.scalar("loss", self.loss)

    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                          self.config.clip)
        self.initialize_session()  # now self.sess is defined and vars are init

    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words)
        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length]  # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)
                viterbi_sequences += [viterbi_seq]
            return viterbi_sequences, sequence_lengths
        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)
            return labels_pred, sequence_lengths

    def run_epoch(self, train, test, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            test: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        epoch_start = time.time()

        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = len(train) // batch_size

        # iterate over dataset
        start_time = time.time()
        step = 0
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr, self.config.dropout)

            _, train_loss, summary = self.sess.run(
                [self.train_op, self.loss, self.merged], feed_dict=fd)
            step += 1

            # tensorboard
            if i % 100 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)
                cost_time = time.time() - start_time
                print("epoch : {}----step : {}----loss : {}----sec/step : {}"
                      .format(epoch + 1, i, train_loss, cost_time / step))
                start_time = time.time()
                step = 0

        epoch_end = time.time()
        print("epoch time: %.8s s" % (epoch_end - epoch_start))

        metrics = self.run_evaluate(test)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in metrics.items()])
        print(msg)

        return metrics["f1"]

    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a == b for (a, b) in zip(lab, lab_pred)]

                lab_chunks = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

        return {"precision": 100 * p, "recall": 100 * r, "f1": 100 * f1}

    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds
