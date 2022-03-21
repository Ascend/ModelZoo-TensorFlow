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
import numpy as np
import os
import re
from math import ceil

'''
接收原始数据，包含数据中的每一个评论句子字符串
返回格式化后的数据（也就是对每个句子进行格式化之后的结果）
'''
def clean_string(string_list):
    ret_list = []
    for string in string_list:
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string) #不在"[]"里面的其他字符全部替换成空格
        string = re.sub(r"\'s", " \'s", string) #将空白替换成另一种格式的空白？？
        string = re.sub(r"\'ve", " \'ve", string) # ？？
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string) # ？？
        string = re.sub(r"\'d", " \'d", string) #数字
        string = re.sub(r"\'ll", " \'ll", string) #
        #标点、符号
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = re.sub(r"[\.,-]", "", string)
        ret_list.append(string)
    return ret_list

'''
接收句子格式化后结果
返回“{词：index}”字典
'''
def build_word_index(string_list):
    string_list = clean_string(string_list)
    word2idx = {}
    for line in string_list:
        for word in line.split():
            if not word in word2idx:
                word2idx[word] = len(word2idx) + 1 #每增加一个词长度增加一，构建索引的方式值得学习
    return word2idx

'''
接收原始句子，填充pading大小，词索引word2idx
返回 arr([['word',...,'word'],...,[]])
'''
def tokenizer(string_list, padding, word2idx):
    string_list = clean_string(string_list)
    tokenized = []
    for line in string_list:
        tokenized_line = []
        for word in line.split(): #对每个一行进行划分（按照空格）
            tokenized_line.append(word2idx[word])
        k = padding - len(tokenized_line)
        tokenized_line += [0] * k #填充k个零
        tokenized.append(tokenized_line) #[['word',...,'word'],...,[]]
    return np.asarray(tokenized)

#数据的没一行代表一个评论，
def get_data(paths):
    PATH_POS = paths[0]
    PATH_NEG = paths[1]
    with open(PATH_NEG, 'r', encoding='UTF-8') as f:
        neg_texts = f.read().splitlines() #按行进行划分，返回由每个句子构成的列表
    with open(PATH_POS,'r', encoding='UTF-8') as f:
        pos_texts = f.read().splitlines() #按行进行划分，返回由每个句子构成的列表

    #为每一个词构建索引
    word2idx = build_word_index(
        string_list=(clean_string(pos_texts) + clean_string(neg_texts)) #将正负样本合在一起
    ) #通过clean_string()过滤之后得到格式化的字符串
    t_pos = tokenizer(pos_texts, 54, word2idx)
    t_neg = tokenizer(neg_texts, 54, word2idx)

    #构造数据标签正向为1，负向为为零
    pos_labels = np.ones([t_pos.shape[0],], dtype='int32')
    neg_labels = np.zeros([t_neg.shape[0], ], dtype='int32')

    #连接两个类别的数据
    data = np.concatenate((t_pos, t_neg))
    labels = np.concatenate((pos_labels, neg_labels))
    return data, labels, word2idx

'''
训练集、验证集四六开
'''
def generate_split(data, labels, val_split):
    j = np.concatenate((data, labels.reshape([-1, 1])), 1) #将数据和标签对应起来, [-1]??
    np.random.shuffle(j) #将数据集打乱
    split_point = int(ceil(data.shape[0]*(1-val_split))) #取整函数
    train_data = j[:split_point,:-1]
    val_data = j[split_point:,:-1]
    train_labels = j[:split_point,-1]
    val_labels = j[split_point:, -1]
    return train_data, train_labels, val_data, val_labels
'''
数据batch划分，
'''
def generate_batch(data, labels, batch_size):
    j = np.concatenate((data, labels.reshape([-1, 1])), 1)
    mark = np.random.randint(batch_size, j.shape[0])
    batch_data = j[mark-batch_size : mark]
    return batch_data[:,:-1], batch_data[:,-1]