# -*- coding: utf-8 -*-
#
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

import os
import numpy as np
import codecs
import tensorflow as tf
from nmt.utils import misc_utils as utils
from nmt.utils import evaluation_utils as evaluation_utils

'''
翻译之后得到的shape可能与原shape不同，请根据文件大小自行计算。句子个数不会发生改变。
单句子文件大小 = 句子长度 * 4 最大句子长度请修改max_len, 句子个数请修改sentence_number
此后处理为一次处理所有batch,其余batch需根据源码进行字符串的扩充，拼接。
'''
output_bin = './outputs'
max_len = 124
sentence_number = 1268
accuracy_file = './accuracy_result'


def concat_bin(path):
    """
    拼接输出Bin文件
    :param path: 离线推理输出文件目录
    :return: 拼接文件路径
    """
    files = os.listdir(path)
    files.sort()
    out_list = []
    out_array = np.array(out_list)
    for file in files:
        if file.endswith('./bin'):
            new_array = np.fromfile(os.path.join(os.getcwd(), path, file), np.int32)
            out_array = np.concatenate((out_array, new_array))
    out_array_name = './dataset.om_infer.bin'
    out_array = out_array.astype(np.int32)
    out_array.tofile(out_array_name)
    return out_array_name


def write_accuracy(result_file, result_content):
    """
    将精度结果写入到文件
    :param result_file: 指定需要写入的文件
    :param result_content: 精度结果json
    :return: 空
    """
    import json
    encode_json = json.dumps(result_content, sort_keys=False, indent=4)
    with open(result_file, 'W') as json_file:
        json_file.write(encode_json)


def get_translation(nmt_outputs, sent_id, tgt_eos, subword_option):
    """Given batch decoding outputs, select a sentence and turn to text"""
    if tgt_eos: tgt_eos = tgt_eos.encode('UTF-8')
    output = nmt_outputs[sent_id, :].tolist()
    if tgt_eos and tgt_eos in output:
        output = output[:output.index(tgt_eos)]
    if subword_option == 'bpe':
        translation = utils.format_bpe_text(output)
    elif subword_option == 'spm':
        translation = utils.format_spm_text(output)
    else:
        translation = utils.format_text(output)
    return translation


def postprocess():
    """
    计算精度结果
    :return: 空
    """
    out_bin = concat_bin(output_bin)
    tgt_vocab_file = './dataset/vocab.en'
    output = np.fromfile(out_bin, np.int32)
    nmt_outputs = output.reshape(max_len, sentence_number)
    nmt_outputs = nmt_outputs.transpose()
    dict = {}
    file = open(tgt_vocab_file)
    i = 0
    for line in file:
        line = line.strip('\n')
        line = line.encode("UTF-8")
        dict[i] = line
        i = i + 1
    file.close()
    sample_words = []
    for line_number in range(sentence_number):
        for vocab_number in range(max_len):
            sample_words.append(dict[nmt_outputs[line_number][vocab_number]])
    sample_words = np.array(sample_words)
    nmt_outputs = sample_words.reshape(sentence_number, max_len)
    # Decode
    num_sentences = 0
    trans_file = './dataset/om_infer.en'
    with codecs.getwriter("utf-8")(
        tf.gfile.GFile(trans_file, mode='wb')) as trans_f:
        trans_f.write("")
    num_translations_per_input = 1
    nmt_outputs = np.expand_dims(nmt_outputs, 0)
    batch_size = nmt_outputs.shape[1]
    num_sentences += batch_size
    for sent_id in range(batch_size):
        for beam_id in range(num_translations_per_input):
            translation = get_translation(
                nmt_outputs[beam_id],
                sent_id,
                tgt_eos='</s>',
                subword_option=''
            )
            trans_f.write((translation + b"\n").decode("utf-8"))
    ref_file = './dataset/tst2013.en'
    evaluation_scores = {}
    if ref_file and tf.gfile.Exists(trans_file):
        for metric in ['bleu']:
            score = evaluation_utils.evaluate(
                ref_file,
                trans_file,
                metric,
                subword_option=''
            )
            evaluation_scores[metric] = score
            utils.print_out(" %s %s: %.5f" % (metric, "infer", score))
            accuracy_results = {}
            accuracy_results['bleu'] = score
            write_accuracy(accuracy_file, accuracy_results)