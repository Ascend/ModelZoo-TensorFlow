#! -*- coding:utf-8 -*-

#                                 Apache License
#                           Version 2.0, January 2004
#                        http://www.apache.org/licenses/

#   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.core.protobuf import config_pb2
import collections
import math
import random
import jieba
import numpy as np
from six.moves import xrange
import tensorflow as tf
import time
import sys
import os

tf.app.flags.DEFINE_string('dataset_dir', './cnews', '')
tf.app.flags.DEFINE_string('precision_mode', 'allow_mix_precision', '')
FLAGS = tf.app.flags.FLAGS


def npu_tf_optimizer(opt):
    npu_opt = NPUDistributedOptimizer(opt)
    # if FLAGS.precision_mode == "allow_mix_precision":
    #    loss_scale_manager = ExponentialUpdateLossScaleManager(
    #        init_loss_scale=2**32,
    #        incr_every_n_steps=1000,
    #        decr_every_n_nan_or_inf=2,
    #        decr_ratio=0.5)
    #    npu_opt = NPULossScaleOptimizer(npu_opt, loss_scale_manager)
    return npu_opt


def npu_session_config_init(session_config=None):
    if ((not isinstance(session_config, config_pb2.ConfigProto)) and (
    not issubclass(type(session_config), config_pb2.ConfigProto))):
        session_config = config_pb2.ConfigProto()
    if (isinstance(session_config, config_pb2.ConfigProto) or issubclass(type(session_config), config_pb2.ConfigProto)):
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        custom_op.parameter_map["enable_data_pre_proc"].b = True
        custom_op.parameter_map["iterations_per_loop"].i = 10
        if FLAGS.precision_mode == "allow_mix_precision":
            custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    return session_config


def read_data():
    global FLAGS
    '\n    对要训练的文本进行处理，最后把文本的内容的所有词放在一个列表中\n    '
    stop_words = []
    with open('stop_words.txt', 'r', encoding='UTF-8') as f:
        line = f.readline()
        while line:
            stop_words.append(line[:(- 1)])
            line = f.readline()
    stop_words = set(stop_words)
    print('停用词读取完毕，共{n}个词'.format(n=len(stop_words)))
    raw_word_list = []
    #    with open('doupocangqiong.txt', 'r', encoding='UTF-8') as f:
    val_txt = os.path.join(FLAGS.dataset_dir, "./cnews.val.txt")
    with open(val_txt, 'r', encoding='UTF-8') as f:
        line = f.readline()
        while line:
            while ('\n' in line):
                line = line.replace('\n', '')
            while (' ' in line):
                line = line.replace(' ', '')
            if (len(line) > 0):
                raw_words = list(jieba.cut(line, cut_all=False))
                raw_word_list.extend(raw_words)
            line = f.readline()
    return raw_word_list


words = read_data()
print('Data size', len(words))
vocabulary_size = 50000


def build_dataset(words):
    count = [['UNK', (- 1)]]
    count.extend(collections.Counter(words).most_common((vocabulary_size - 1)))
    print('count', len(count))
    dictionary = dict()
    for (word, _) in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if (word in dictionary):
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return (data, count, dictionary, reverse_dictionary)


(data, count, dictionary, reverse_dictionary) = build_dataset(words)
del words
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
data_index = 0


# def generate_batch(batch_size, num_skips, skip_window):
#    global data_index
#    assert ((batch_size % num_skips) == 0)
#    assert (num_skips <= (2 * skip_window))
#    batch = np.ndarray(shape=batch_size, dtype=np.int32)
#    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
#    span = ((2 * skip_window) + 1)
#    buffer = collections.deque(maxlen=span)
#    for _ in range(span):
#        buffer.append(data[data_index])
#        data_index = ((data_index + 1) % len(data))
#    for i in range((batch_size // num_skips)):
#        target = skip_window
#        targets_to_avoid = [skip_window]
#        for j in range(num_skips):
#            while (target in targets_to_avoid):
#                target = random.randint(0, (span - 1))
#            targets_to_avoid.append(target)
#            batch[((i * num_skips) + j)] = buffer[skip_window]
#            labels[(((i * num_skips) + j), 0)] = buffer[target]
#        buffer.append(data[data_index])
#        data_index = ((data_index + 1) % len(data))
#    return (batch, labels)
# (batch, labels) = generate_batch(batch_size=8, num_skips=2, skip_window=1)
def generate_batch(data):
    batch = []
    labels = []
    for i in range(1, len(data) - 1):
        batch.append(data[i])
        batch.append(data[i])
        left_labels = []
        right_labels = []
        left_labels.append(data[i-1])
        right_labels.append(data[i+1])
        labels.append(right_labels)
        labels.append(left_labels)
    print("batch,labels")
    print(np.array(batch))
    print(np.array(labels))
    return (np.array(batch), np.array(labels))

(batch, labels) = generate_batch(data)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[(i, 0)], reverse_dictionary[labels[(i, 0)]])
batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2
valid_size = 9
valid_window = 100
num_sampled = 64
# valid_word = ['萧炎', '灵魂', '火焰', '萧薰儿', '药老', '天阶', '云岚宗', '乌坦城', '惊诧']
valid_word = ['城市', '记者', '体育', '教练', '足球', '赛季', '奥运会', '丑闻', '足协']
valid_examples = [dictionary[li] for li in valid_word]
graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int64, shape=[batch_size])
    train_labels = tf.placeholder(tf.int64, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], (- 1.0), 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size], stddev=(1.0 / math.sqrt(embedding_size))))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]), dtype=tf.float32)
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, inputs=embed, labels=train_labels,
                                         num_sampled=num_sampled, num_classes=vocabulary_size))
    optimizer = npu_tf_optimizer(tf.train.GradientDescentOptimizer(1.0)).minimize(loss)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = (embeddings / norm)
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    init = tf.global_variables_initializer()

if len(sys.argv) < 3:
    num_steps = 100000
else:
    num_steps = int(sys.argv[2])

with tf.Session(graph=graph, config=npu_session_config_init()) as session:
    init.run()
    print('Initialized')
    average_loss = 0
    duration = 0

    dataset = tf.data.Dataset.from_tensor_slices((batch, labels))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    nextelement = iterator.get_next()
    train_op = util.set_iteration_per_loop(session,optimizer,10)
    num_steps = int(num_steps / 10)
    session.run(iterator.initializer)
    for step in xrange(num_steps):
        # if FLAGS.precision_mode == "allow_mix_precision":
        #     lossScale = tf.get_default_graph().get_tensor_by_name("loss_scale:0")
        #     overflow_status_reduce_all = tf.get_default_graph().get_tensor_by_name("overflow_status_reduce_all:0")
        start_time = time.time()
        (batch_inputs, batch_labels) = session.run(nextelement)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        # start_time = time.time()
        #(_, loss_val) = session.run([optimizer, loss], feed_dict=feed_dict)
        (_, loss_val) = session.run([train_op, loss], feed_dict=feed_dict)
        # if FLAGS.precision_mode == "allow_mix_precision":
        #    (l_s,overflow_status_reduce_all,_, loss_val) = session.run([lossScale,overflow_status_reduce_all,optimizer, loss], feed_dict=feed_dict)
        #    print("loss_scale_is:",l_s)
        #    print("overflow_status_reduce_all:",overflow_status_reduce_all)
        # else:
        #     (_, loss_val) = session.run([train_op, loss], feed_dict=feed_dict)

        average_loss += loss_val
        duration += (time.time() - start_time)
        # print('step = ', step, 'loss = {:3f}, time cost = {:4f} s'.format(loss_val, duration))
        # duration = 0
        if ((step % 200) == 0):
            if (step > 0):
                average_loss /= 200
            #            print('Average loss at step ', step, ': ', average_loss)
            print('step = ', step, 'loss = {:3f}, time cost = {:4f} s'.format(average_loss, duration))
            average_loss = 0
            duration = 0
        if ((step % 10000) == 0):
            tf.train.Saver().save(session, "ckpt_npu/model.ckpt")
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (- sim[i, :]).argsort()[:top_k]
                log_str = ('Nearest to %s:' % valid_word)
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = ('%s %s,' % (log_str, close_word))
                print(log_str)
    final_embeddings = normalized_embeddings.eval()


def plot_with_labels(low_dim_embs, labels, filename='images/tsne_npu.png', fonts=None):
    assert (low_dim_embs.shape[0] >= len(labels)), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))
    for (i, label) in enumerate(labels):
        (x, y) = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, fontproperties=fonts, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right',
                     va='bottom')
    plt.savefig(filename, dpi=800)


try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    # font = FontProperties(fname='c:\\windows\\fonts\\simsun.ttc', size=14)
    f_name = os.path.join(FLAGS.dataset_dir, "./SIMSUN.TTC")
    font = FontProperties(
        fname=f_name, size=14)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    #plot_with_labels(low_dim_embs, labels, fonts=font)
except ImportError:
    print('Please install sklearn, matplotlib, and scipy to visualize embeddings.')
