# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
#
#    http:/www.apache.org/license/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,either express or implies.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


max_len = 123


def preprocess():
    """
    数据集前处理
    :return: 空
    """
    src_vocab_file = './dataset/vocab.vi'
    src_dataset_file = './dataset/tst2013.vi'
    tgt_dict = {}
    vocab_file = open(src_vocab_file)
    vocab_line_number = 0
    for vocab_line in vocab_file:
        vocab_line = vocab_line.strip('\n')
        tgt_dict[vocab_line] = vocab_line_number
        vocab_line_number = vocab_line_number + 1
    vocab_file.close()
    data_file = open(src_dataset_file)
    data_line_number = 0
    for data_line in data_file:
        data_list = data_line.strip('\n').split()
        tgt_src_ids = []
        for vocab in data_list:
            if vocab not in tgt_dict:
                tgt_src_ids.append(0)
            else:
                tgt_src_ids.append(tgt_dict[vocab])
        tgt_array = np.array(tgt_src_ids)
        if tgt_array.shape != max_len:
            zero_len = max_len - len(tgt_src_ids)
            tgt_array = np.pad(tgt_array, (0, zero_len), 'constant', constant_values=(0, 2))
        tgt_array = tgt_array.astype(np.int32)
        tgt_name = '../dataset/data/src_ids/input_src_ids_%05d.bin' % data_line_number
        tgt_array.tofile(tgt_name)
        tgt_src_len = [len(tgt_src_ids)]
        tgt_len_array = np.array(tgt_src_len)
        tgt_len_array = tgt_len_array.astype(np.int32)
        tgt_len_name = '../dataset/data/src_len/input_src_len_%05d.bin' % data_line_number
        tgt_len_array.tofile(tgt_len_name)
        data_line_number = data_line_number + 1
    data_file.close()


preprocess()
