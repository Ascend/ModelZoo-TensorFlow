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
#==============================================================================

from __future__ import print_function
from npu_bridge.npu_init import *
from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import re
import os
import unicodedata
from num2words import num2words
from random import randint
import pandas as pd
import random

def keep_pho():
    return (random.random() > hp.phon_drop)

def text_normalize(sent):
    'Minimum text preprocessing'

    def _strip_accents(s):
        return ''.join((c for c in unicodedata.normalize('NFD', s) if (unicodedata.category(c) != 'Mn')))
    normalized = []
    for word in sent.split():
        word = _strip_accents(word.lower())
        srch = re.match('\\d[\\d,.]*$', word)
        if srch:
            word = num2words(float(word.replace(',', '')))
        word = re.sub('[-—-]', ' ', word)
        word = re.sub("[^ a-z'.?]", '', word)
        normalized.append(word)
    normalized = ' '.join(normalized)
    normalized = re.sub('[ ]{2,}', ' ', normalized)
    normalized = normalized.strip()
    return normalized

def text_normalize_cmu(sent):
    'Remove accents and upper strings.'

    def _strip_accents(s):
        return ''.join((c for c in unicodedata.normalize('NFD', s) if (unicodedata.category(c) != 'Mn')))
    normalized = []
    for word in sent.split():
        word = _strip_accents(word.lower())
        srch = re.match('\\d[\\d,.]*$', word)
        if srch:
            word = num2words(float(word.replace(',', '')))
        word = re.sub('[-—-]', ' ', word)
        word = re.sub("[^ a-z'.?]", '', word)
        normalized.append(word)
    normalized = ' '.join(normalized)
    normalized = re.sub('[ ]{2,}', ' ', normalized)
    normalized = normalized.strip()
    normalized = re.sub('[^ A-Z,;.]', '', _strip_accents(sent).upper())
    if (normalized[(- 1)] in ['.', ',', '?', ';']):
        normalized = normalized[0:(- 1)]
    normalized = re.sub("'", ' ', normalized)
    normalized = re.sub(' ', '@', normalized)
    normalized = re.sub(',', '@@', normalized)
    normalized = re.sub(';', '@@@', normalized)
    normalized = re.sub('\\.', '@@@@', normalized)
    normalized = normalized.strip()
    return normalized

def break_to_phonemes(cmu, strin):
    strin = re.sub('([A-Z])@', '\\1 @', strin)
    strin = re.sub('([A-Z])\\*', '\\1 *', strin)
    strin = re.sub('@([A-Z])', '@ \\1', strin)
    strin = re.sub('\\s+', ' ', strin)
    strin = re.split('\\s', strin)
    strout = ''
    for word_in in strin:
        word_in = word_in.upper()
        wpd = wwd = ''
        if ('@' in word_in):
            wpd = word_in
        else:
            if (word_in in cmu):
                if keep_pho():
                    wwd = cmu[word_in].split(' ')
                else:
                    wwd = list(word_in)
                for kl in range(0, len(wwd)):
                    if (len(wwd[kl]) == 3):
                        wwd[kl] = wwd[kl][0:2]
            else:
                wwd = list(word_in)
            for kl in range(0, len(wwd)):
                if (kl != (len(wwd) - 1)):
                    wpd = ((wpd + wwd[kl]) + ' ')
                else:
                    wpd = (wpd + wwd[kl])
        strout = (strout + wpd)
    return strout

def load_vocab():
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"
    char2idx = {char: idx for (idx, char) in enumerate(vocab)}
    idx2char = {idx: char for (idx, char) in enumerate(vocab)}
    return (char2idx, idx2char)

def load_vocab_cmu():
    valid_symbols = ['#', '@', 'A', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'C', 'CH', 'D', 'DH', 'E', 'EH', 'ER', 'EY', 'F', 'G', 'H', 'HH', 'I', 'IH', 'IY', 'J', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'O', 'OY', 'P', 'Q', 'R', 'S', 'SH', 'T', 'TH', 'U', 'UH', 'UW', 'V', 'W', 'X', 'Y', 'Z', 'ZH', '*', "'"]
    _valid_symbol_set = set(valid_symbols)
    char2idx = {char: idx for (idx, char) in enumerate(_valid_symbol_set)}
    idx2char = {idx: char for (idx, char) in enumerate(_valid_symbol_set)}
    return (char2idx, idx2char)

def str_to_ph(strin):
    strin = re.sub('([A-Z])@', '\\1 @', strin)
    strin = re.sub('([A-Z])\\*', '\\1 *', strin)
    strin = re.sub('@([A-Z])', '@ \\1', strin)
    strin = re.sub('@', ' @', strin)
    strin = re.sub('\\s+', ' ', strin)
    strin = re.sub('@\\*', '*', strin)
    strin = re.split('\\s', strin)
    return strin

def invert_text(txt):
    if (not hp.run_cmu):
        (char2idx, idx2char) = load_vocab()
        pstring = [idx2char[char] for char in txt]
        pstring = ''.join(pstring)
        pstring = pstring.replace('E', '')
        pstring = pstring.replace('P', '')
    else:
        (char2idx, idx2char) = load_vocab_cmu()
        pstring = [idx2char[char] for char in txt]
        pstring = ''.join(pstring)
        pstring = pstring.replace('@', ' ')
        pstring = pstring.replace('#', '')
        pstring = pstring.replace('*', '')
    return pstring

def load_test_data(config):
    if (not hp.run_cmu):
        (char2idx, idx2char) = load_vocab()
    else:
        cmudict = os.path.join(config.data_paths, 'cmudict.dict.txt')
        cmu = pd.read_csv(cmudict, header=None, names=['name'])
        (cmu['word'], cmu['phone']) = cmu['name'].str.split(' ', 1).str
        cmu['word'] = cmu['word'].str.upper()
        cmu.drop(['name'], axis=1, inplace=True)
        cmu = list(cmu.set_index('word').to_dict().values()).pop()
        (char2idx, idx2char) = load_vocab_cmu()
    texts = []
    for line in codecs.open('test_sents.txt', 'r', 'utf-8'):
        if (not hp.run_cmu):
            sent = (text_normalize(line).strip() + 'E')
        else:
            sent = (text_normalize_cmu(line) + '*')
            sent = break_to_phonemes(cmu, sent)
            sent = str_to_ph(sent)
        if (len(sent) <= hp.T_x):
            if (not hp.run_cmu):
                sent += ('P' * (hp.T_x - len(sent)))
            else:
                sent.extend((['#'] * (hp.T_x - len(sent))))
            texts.append([char2idx[char] for char in sent])
    texts = np.array(texts, np.int32)
    return texts

def load_data(config, train_form, training=True):
    if (not hp.run_cmu):
        (char2idx, idx2char) = load_vocab()
    else:
        cmudict = os.path.join(config.data_paths, 'cmudict.dict.txt')
        cmu = pd.read_csv(cmudict, header=None, names=['name'])
        (cmu['word'], cmu['phone']) = cmu['name'].str.split(' ', 1).str
        cmu['word'] = cmu['word'].str.upper()
        cmu.drop(['name'], axis=1, inplace=True)
        cmu = list(cmu.set_index('word').to_dict().values()).pop()
        (char2idx, idx2char) = load_vocab_cmu()
    (texts, _texts_test, mels, mags, dones) = ([], [], [], [], [])
    num_samples = 1
    metadata = os.path.join(config.data_paths, 'metadata.csv')
    for line in codecs.open(metadata, 'r', 'utf-8'):
        (fname, _, sent) = line.strip().split('|')
        if (not hp.run_cmu):
            sent = (text_normalize(sent) + 'E')
        else:
            sent = (text_normalize_cmu(sent) + '*')
            sent = break_to_phonemes(cmu, sent)
            sent = str_to_ph(sent)
        if (len(sent) <= hp.T_x):
            if (not hp.run_cmu):
                sent += ('P' * (hp.T_x - len(sent)))
            else:
                sent.extend((['#'] * (hp.T_x - len(sent))))
            pstring = [char2idx[char] for char in sent]
            texts.append(np.array(pstring, np.int32).tostring())
            _texts_test.append(np.array(pstring, np.int32).tostring())
            mels.append(os.path.join(config.data_paths, 'mels', (fname + '.npy')))
            if hp.include_dones:
                dones.append(os.path.join(config.data_paths, 'dones', (fname + '.npy')))
            if (train_form != 'Encoder'):
                mags.append(os.path.join(config.data_paths, 'mags', (fname + '.npy')))
    return (texts, _texts_test, mels, mags, dones)

def parse_function_with_mags(texts, texts_tests, mels, mags):

    text = tf.decode_raw(texts, tf.int32)
    texts_test = tf.decode_raw(texts_tests, tf.int32)
    mel = tf.py_func((lambda x: np.load(x)), [mels], tf.float32)
    mag = tf.py_func((lambda x: np.load(x)), [mags], tf.float32)

    text = tf.pad(text, ((0, hp.T_x),))[:hp.T_x]
    texts_test = tf.pad(texts_test, ((0, hp.T_x),))[:hp.T_x]
    mel = tf.pad(mel, ((0, hp.T_y), (0, 0)))[:hp.T_y]
    mag = tf.pad(mag, ((0, hp.T_y), (0, 0)))[:hp.T_y]

    return text, texts_test, mel, mag

def reshape_fn_with_mags(text, texts_test, mel, mag):

    text = tf.reshape(text, [hp.T_x, ])
    texts_test = tf.reshape(texts_test, [hp.T_x, ])
    mel = tf.reshape(mel, [(hp.T_y // hp.r), (hp.n_mels * hp.r)])
    mag = tf.reshape(mag, [hp.T_y, (1 + (hp.n_fft // 2))])

    return text, texts_test, mel, mag

def parse_function_with_dones(texts, texts_tests, mels, dones):

    text = tf.decode_raw(texts, tf.int32)
    texts_test = tf.decode_raw(texts_tests, tf.int32)
    mel = tf.py_func((lambda x: np.load(x)), [mels], tf.float32)
    done = tf.py_func((lambda x: np.load(x)), [dones], tf.int32)

    text = tf.pad(text, ((0, hp.T_x),))[:hp.T_x]
    texts_test = tf.pad(texts_test, ((0, hp.T_x),))[:hp.T_x]
    mel = tf.pad(mel, ((0, hp.T_y), (0, 0)))[:hp.T_y]
    done = tf.pad(done, ((0, hp.T_y),))[:hp.T_y]
    done = done[::hp.r]

    return text, texts_test, mel, done

def reshape_fn_with_dones(text, texts_test, mel, done):

    text = tf.reshape(text, [hp.T_x, ])
    texts_test = tf.reshape(texts_test, [hp.T_x, ])
    mel = tf.reshape(mel, [(hp.T_y // hp.r), (hp.n_mels * hp.r)])
    done = tf.reshape(done, [(hp.T_y // hp.r), ])

    return text, texts_test, mel, done

def parse_function_full(texts, texts_tests, mels, mags, dones):

    text = tf.decode_raw(texts, tf.int32)
    texts_test = tf.decode_raw(texts_tests, tf.int32)
    mel = tf.py_func((lambda x: np.load(x)), [mels], tf.float32)
    mag = tf.py_func((lambda x: np.load(x)), [mags], tf.float32)
    done = tf.py_func((lambda x: np.load(x)), [dones], tf.int32)
    done = done[::hp.r]

    text = tf.pad(text, ((0, hp.T_x),))[:hp.T_x]
    texts_test = tf.pad(texts_test, ((0, hp.T_x),))[:hp.T_x]
    mel = tf.pad(mel, ((0, hp.T_y), (0, 0)))[:hp.T_y]
    mag = tf.pad(mag, ((0, hp.T_y), (0, 0)))[:hp.T_y]
    done = tf.pad(done, ((0, hp.T_y),))[:hp.T_y]

    return text, texts_test, mel, mag, done

def reshape_fn_full(text, texts_test, mel, mag, done):

    text = tf.reshape(text, [hp.T_x, ])
    texts_test = tf.reshape(texts_test, [hp.T_x, ])
    mel = tf.reshape(mel, [(hp.T_y // hp.r), (hp.n_mels * hp.r)])
    mag = tf.reshape(mag, [hp.T_y, (1 + (hp.n_fft // 2))])
    done = tf.reshape(done, [(hp.T_y // hp.r), ])

    return text, texts_test, mel, mag, done

def parse_function(texts, texts_tests, mels):

    text = tf.decode_raw(texts, tf.int32)
    texts_test = tf.decode_raw(texts_tests, tf.int32)
    mel = tf.py_func((lambda x: np.load(x)), [mels], tf.float32)

    text = tf.pad(text, ((0, hp.T_x),))[:hp.T_x]
    texts_test = tf.pad(texts_test, ((0, hp.T_x),))[:hp.T_x]
    mel = tf.pad(mel, ((0, hp.T_y), (0, 0)))[:hp.T_y]
    # mel = tf.reshape(mel, ((hp.T_y // hp.r), (- 1)))

    return text, texts_test, mel

def reshape_fn(text, texts_test, mel):

    text = tf.reshape(text, [hp.T_x, ])
    texts_test = tf.reshape(texts_test, [hp.T_x, ])
    mel = tf.reshape(mel, [(hp.T_y // hp.r), (hp.n_mels * hp.r)])

    return text, texts_test, mel

def make_dataset(config, train_form):
    'Loads training data and put them in queues'
    (_texts, _texts_tests, _mels, _mags, _dones) = load_data(config, train_form)
    num_batch = (len(_texts) // config.batch_size)

    if (train_form == 'Both'):
        if hp.include_dones:
            ds = tf.data.Dataset.from_tensor_slices((_texts, _texts_tests, _mels, _mags, _dones))
            ds = ds.shuffle(buffer_size=10000)
            ds = ds.repeat(config.epoch)
            ds = ds.map(lambda texts, texts_tests, mels, mags, dones: parse_function_full(texts=texts,
                                                                                     texts_tests=texts_tests,
                                                                                     mels=mels,
                                                                                     mags=mags,
                                                                                     dones=dones),
                        num_parallel_calls=14)
            ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE).map(reshape_fn_full)

        else:
            ds = tf.data.Dataset.from_tensor_slices((_texts, _texts_tests, _mels, _mags))
            ds = ds.shuffle(buffer_size=10000)
            ds = ds.repeat(config.epoch)
            ds = ds.map(lambda texts, texts_tests, mels, mags: parse_function_with_mags(texts=texts,
                                                                              texts_tests=texts_tests,
                                                                              mels=mels,
                                                                              mags=mags),
                        num_parallel_calls=14)
            ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE).map(reshape_fn_with_mags)

    elif (train_form == 'Encoder'):
        if hp.include_dones:
            ds = tf.data.Dataset.from_tensor_slices((_texts, _texts_tests, _mels, _dones))
            ds = ds.shuffle(buffer_size=10000)
            ds = ds.repeat(config.epoch)
            ds = ds.map(lambda texts, texts_tests, mels, dones: parse_function_with_dones(texts=texts,
                                                                               texts_tests=texts_tests,
                                                                               mels=mels,
                                                                               dones=dones),
                        num_parallel_calls=14)
            ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE).map(reshape_fn_with_dones)

        else:
            ds = tf.data.Dataset.from_tensor_slices((_texts, _texts_tests, _mels))
            ds = ds.shuffle(buffer_size=10000)
            ds = ds.repeat(config.epoch)
            ds = ds.map(lambda texts, texts_tests, mels: parse_function(texts=texts,
                                                                        texts_tests=texts_tests,
                                                                        mels=mels),
                        num_parallel_calls=14)
            ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE).map(reshape_fn)

    else:
        ds = tf.data.Dataset.from_tensor_slices((_texts, _texts_tests, _mels, _mags))
        ds = ds.shuffle(buffer_size=10000)
        ds = ds.repeat(config.epoch)
        ds = ds.map(lambda texts, texts_tests, mels, mags: parse_function_with_mags(texts=texts,
                                                                          texts_tests=texts_tests,
                                                                          mels=mels,
                                                                          mags=mags),
                    num_parallel_calls=14)
        ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE).map(reshape_fn_with_mags)

    if config.dynamic_bs:
        ds = ds.batch(config.batch_size)
    else:
        ds = ds.batch(config.batch_size, drop_remainder=True)

    return ds, num_batch

def get_batch(config, train_form):
    'Loads training data and put them in queues'
    with tf.device('/cpu:0'):
        (_texts, _texts_tests, _mels, _mags, _dones) = load_data(config, train_form)
        num_batch = (len(_texts) // config.batch_size)
        texts = tf.convert_to_tensor(_texts)
        texts_tests = tf.convert_to_tensor(_texts_tests)
        mels = tf.convert_to_tensor(_mels)
        if hp.include_dones:
            dones = tf.convert_to_tensor(_dones)
        if (train_form != 'Encoder'):
            mags = tf.convert_to_tensor(_mags)
        shuffleFlag = True
        with npu_scope.without_npu_compile_scope():
            if (train_form == 'Both'):
                if hp.include_dones:
                    (text, texts_test, mel, mag, done) = tf.train.slice_input_producer([texts, texts_tests, mels, mags, dones], shuffle=shuffleFlag)
                else:
                    (text, texts_test, mel, mag) = tf.train.slice_input_producer([texts, texts_tests, mels, mags], shuffle=shuffleFlag)
            elif (train_form == 'Encoder'):
                if hp.include_dones:
                    (text, texts_test, mel, done) = tf.train.slice_input_producer([texts, texts_tests, mels, dones], shuffle=shuffleFlag)
                else:
                    (text, texts_test, mel) = tf.train.slice_input_producer([texts, texts_tests, mels], shuffle=shuffleFlag)
            else:
                (text, texts_test, mel, mag) = tf.train.slice_input_producer([texts, texts_tests, mels, mags], shuffle=shuffleFlag)

        text = tf.decode_raw(text, tf.int32)
        texts_test = tf.decode_raw(texts_test, tf.int32)
        mel = tf.py_func((lambda x: np.load(x)), [mel], tf.float32)
        if hp.include_dones:
            done = tf.py_func((lambda x: np.load(x)), [done], tf.int32)
        if (train_form != 'Encoder'):
            mag = tf.py_func((lambda x: np.load(x)), [mag], tf.float32)
        text = tf.pad(text, ((0, hp.T_x),))[:hp.T_x]
        texts_test = tf.pad(texts_test, ((0, hp.T_x),))[:hp.T_x]
        mel = tf.pad(mel, ((0, hp.T_y), (0, 0)))[:hp.T_y]
        if hp.include_dones:
            done = tf.pad(done, ((0, hp.T_y),))[:hp.T_y]
        if (train_form != 'Encoder'):
            mag = tf.pad(mag, ((0, hp.T_y), (0, 0)))[:hp.T_y]
        mel = tf.reshape(mel, ((hp.T_y // hp.r), (- 1)))
        if hp.include_dones:
            done = done[::hp.r]
        num_threadsSet = 1
        batch_size = config.batch_size
        with npu_scope.without_npu_compile_scope():
            if (train_form == 'Both'):
                if hp.include_dones:
                    (texts, texts_tests, mels, mags, dones) = tf.train.batch([text, texts_test, mel, mag, done], shapes=[(hp.T_x,), (hp.T_x,), ((hp.T_y // hp.r), (hp.n_mels * hp.r)), (hp.T_y, (1 + (hp.n_fft // 2))), ((hp.T_y // hp.r),)], num_threads=num_threadsSet, batch_size=batch_size, capacity=(batch_size * 8), dynamic_pad=False)
                    return (texts_tests, texts, mels, dones, mags, num_batch)
                else:
                    (texts, texts_tests, mels, mags) = tf.train.batch([text, texts_test, mel, mag], shapes=[(hp.T_x,), (hp.T_x,), ((hp.T_y // hp.r), (hp.n_mels * hp.r)), (hp.T_y, (1 + (hp.n_fft // 2)))], num_threads=num_threadsSet, batch_size=batch_size, capacity=(batch_size * 8), dynamic_pad=False)
                    return (texts_tests, texts, mels, None, mags, num_batch)
            elif (train_form == 'Encoder'):
                if hp.include_dones:
                    (texts, texts_tests, mels, dones) = tf.train.batch([text, texts_test, mel, done], shapes=[(hp.T_x,), (hp.T_x,), ((hp.T_y // hp.r), (hp.n_mels * hp.r)), ((hp.T_y // hp.r),)], num_threads=num_threadsSet, batch_size=batch_size, capacity=(batch_size * 8), dynamic_pad=False)
                    return (texts_tests, texts, mels, dones, None, num_batch)
                else:
                    (texts, texts_tests, mels) = tf.train.batch([text, texts_test, mel], shapes=[(hp.T_x,), (hp.T_x,), ((hp.T_y // hp.r), (hp.n_mels * hp.r))], num_threads=num_threadsSet, batch_size=batch_size, capacity=(batch_size * 8), dynamic_pad=False)
                    return (texts_tests, texts, mels, None, None, num_batch)
            else:
                (texts, texts_tests, mels, mags) = tf.train.batch([text, texts_test, mel, mag], shapes=[(hp.T_x,), (hp.T_x,), ((hp.T_y // hp.r), (hp.n_mels * hp.r)), (hp.T_y, (1 + (hp.n_fft // 2)))], num_threads=num_threadsSet, batch_size=batch_size, capacity=(batch_size * 8), dynamic_pad=False)
                return (texts_tests, texts, mels, None, mags, num_batch)
