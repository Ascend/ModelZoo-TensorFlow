#!usr/bin/env python
# -*- coding:utf-8 -*-
#
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
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
import argparse
import ast

import npu_device
from npu_device.compat.v1.npu_init import *
npu_device.compat.enable_v1()

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--dataset_dir', type=str, default='./cnews', help='dataset dir')
parser.add_argument('--step', type=int, default=30000, help='train steps')
parser.add_argument('--precision_mode', type=str, default='allow_mix_precision', help='precision mode')

#===============================NPU Migration========================================= 
parser.add_argument('--over_dump', dest='over_dump', type=ast.literal_eval,
                        help='if or not over detection, default is False')
parser.add_argument('--data_dump_flag', dest='data_dump_flag', type=ast.literal_eval,
                    help='data dump flag, default is False')
parser.add_argument('--data_dump_step', default="10",
                    help='data dump step, default is 10')
parser.add_argument('--profiling', dest='profiling', type=ast.literal_eval,help='if or not profiling for performance debug, default is False')
parser.add_argument('--profiling_dump_path', default="/home/data", type=str,help='the path to save profiling data')
parser.add_argument('--over_dump_path', default="/home/data", type=str,help='the path to save over dump data')
parser.add_argument('--data_dump_path', default="/home/data", type=str,help='the path to save dump data')
parser.add_argument('--use_mixlist', dest='use_mixlist', type=ast.literal_eval,
                    help='use_mixlist flag, default is False')
parser.add_argument('--fusion_off_flag', dest='fusion_off_flag', type=ast.literal_eval,
                    help='fusion_off flag, default is False')
parser.add_argument('--mixlist_file', default="ops_info.json", type=str,help='mixlist file name, default is ops_info.json')
parser.add_argument('--fusion_off_file', default="fusion_switch.cfg", type=str,help='fusion_off file name, default is fusion_switch.cfg')
parser.add_argument('--auto_tune', dest='auto_tune', type=ast.literal_eval, help='autotune, default is False')
#===============================NPU Migration========================================= 

FLAGS = parser.parse_args()

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
    session_config = tf.compat.v1.ConfigProto()
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    custom_op.parameter_map["enable_data_pre_proc"].b = True
    custom_op.parameter_map["iterations_per_loop"].i = 10
    if FLAGS.data_dump_flag:
        custom_op.parameter_map["enable_dump"].b = True
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(FLAGS.data_dump_path)
        custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes(FLAGS.data_dump_step)
        custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
    if FLAGS.over_dump:
        custom_op.parameter_map["enable_dump_debug"].b = True
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(FLAGS.over_dump_path)
        custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")  
    if FLAGS.profiling:
        custom_op.parameter_map["precision_mode"].b = True
        profiling_options = '{"output":"' + FLAGS.profiling_dump_path + '", \
                          "training_trace":"on", \
                          "task_trace":"on", \
                          "aicpu":"on", \
                          "aic_metrics":"PipeUtilization",\
                          "fp_point":"", \
                          "bp_point":""}'
        custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(profiling_options)
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(FLAGS.precision_mode)
    if FLAGS.use_mixlist and FLAGS.precision_mode=='allow_mix_precision':
        custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes(FLAGS.mixlist_file)
    if FLAGS.fusion_off_flag:
        custom_op.parameter_map["sfusion_switch_file"].s = tf.compat.as_bytes(FLAGS.fusion_off_file)
    if FLAGS.auto_tune:
        custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")
    session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    session_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    return session_config

# Step 1: Download the data.
# Read the data into a list of strings.
def read_data():
    global FLAGS
    """
    对要训练的文本进行处理，最后把文本的内容的所有词放在一个列表中
    """
    #读取停用词
    stop_words = []
    with open('stop_words.txt',"r",encoding="UTF-8") as f:
        line = f.readline()
        while line:
            stop_words.append(line[:(-1)])
            line = f.readline()
    stop_words = set(stop_words)
    print('停用词读取完毕，共{n}个词'.format(n=len(stop_words)))

    # 读取文本，预处理，分词，得到词典
    raw_word_list = []
    # with open('doupocangqiong.txt',"r", encoding='UTF-8') as f:
    val_txt = os.path.join(FLAGS.dataset_dir, './cnews.val.txt')
    with open(val_txt, 'r', encoding='UTF-8') as f:
        line = f.readline()
        while line:
            while ('\n' in line):
                line = line.replace('\n','')
            while (' ' in line):
                line = line.replace(' ','')
            if (len(line)>0): # 如果句子非空
                raw_words = list(jieba.cut(line,cut_all=False))
                raw_word_list.extend(raw_words)
            line=f.readline()
    return raw_word_list

#step 1:读取文件中的内容组成一个列表
words = read_data()
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000

def build_dataset(words):
    count = [['UNK', (-1)]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    print("count",len(count))
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
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
#删除words节省内存
del words  
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],'->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.
batch_size = 128
embedding_size = 128  
skip_window = 1       
num_skips = 2         
valid_size = 9      #切记这个数字要和len(valid_word)对应，要不然会报错哦    
valid_window = 100  
num_sampled = 64    # Number of negative examples to sample.

#验证集
# valid_word = ['萧炎','灵魂','火焰','萧薰儿','药老','天阶',"云岚宗","乌坦城","惊诧"]
valid_word = ['城市', '记者', '体育', '教练', '足球', '赛季', '奥运会', '丑闻', '足协']
valid_examples =[dictionary[li] for li in valid_word]

graph = tf.Graph()
with graph.as_default():
    # Input data.
    train_inputs = tf.compat.v1.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random.uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(params=embeddings, ids=train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.random.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]),dtype=tf.float32)

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(input_tensor=tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases, 
                                         inputs=embed, 
                                         labels=train_labels,
                                         num_sampled=num_sampled, 
                                         num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = npu_tf_optimizer(tf.compat.v1.train.GradientDescentOptimizer(1.0)).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(embeddings), axis=1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(params=normalized_embeddings, ids=valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.compat.v1.global_variables_initializer()

# Step 5: Begin training.
num_steps = FLAGS.step

with tf.compat.v1.Session(graph=graph, config=npu_session_config_init()) as session:
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")

    average_loss = 0
    duration = 0

    for step in xrange(num_steps):
        start_time = time.time()
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
        duration += (time.time() - start_time)

        if step % 200 == 0:
            if step > 0:
                average_loss /= 200
            # The average loss is an estimate of the loss over the last 2000 batches.
            # print("Average loss at step ", step, ": ", average_loss)
            print('step= ', step, 'loss = {:3f}, time cost = {:4f} s'.format(average_loss, duration))
            average_loss = 0
            duration=0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[:top_k]
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()

# Step 6: Visualize the embeddings.
def plot_with_labels(low_dim_embs, labels, filename='images/tsne3.png',fonts=None):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                    fontproperties=fonts,
                    xy=(x, y),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')

    plt.savefig(filename,dpi=800)

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    
    #为了在图片上能显示出中文
    # font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    f_name = os.path.join(FLAGS.dataset_dir, "./SIMSUN.TTC")
    font = FontProperties(fname=f_name, size=14)
    
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels,fonts=font)
    
    
except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")