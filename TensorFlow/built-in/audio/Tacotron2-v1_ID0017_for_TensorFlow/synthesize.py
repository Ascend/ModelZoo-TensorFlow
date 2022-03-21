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
import os, sys
from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
from utils import *
from data_load import load_test_data, invert_text
from scipy.io.wavfile import write
import random
import librosa
from tqdm import tqdm

def create_write_files(ret, sess, g, x, mname, cdir, typeS):
    x = np.expand_dims(x, axis=0)
    x = np.append(x, np.zeros(((hp.batch_size - 1), hp.T_x)), axis=0)
    mel_output = np.zeros((hp.batch_size, (hp.T_y // hp.r), (hp.n_mels * hp.r)), np.float32)
    (_gs, mel_output) = sess.run([g.global_step, g.mel_output], {g.x: x, g.y1: mel_output})
    mag_output = sess.run(g.mag_output, {g.converter_input: mel_output})
    x = x[0]
    txt = invert_text(x)
    mag_output = np.squeeze(mag_output[0])
    try:
        wav = spectrogram2wav(mag_output)
        (wav, _) = librosa.effects.trim(wav)
        write((cdir + '/{}mag.wav'.format(mname)), hp.sr, wav)
        ret.append([txt, wav, (typeS + '_world'), mel_output, mag_output])
    except Exception:
        sys.exc_clear()
    return ret

def create_mel(sess, g, x):
    x = np.expand_dims(x, axis=0)
    x = np.append(x, np.zeros(((hp.batch_size - 1), hp.T_x)), axis=0)
    mel_output = np.zeros((hp.batch_size, (hp.T_y // hp.r), (hp.n_mels * hp.r)), np.float32)
    (_gs, mel_output) = sess.run([g.global_step, g.mel_output], {g.x: x, g.y1: mel_output})
    return mel_output

def create_write_files_conv(ret, sess, mel_in, g, x, mname, cdir, typeS):
    mag_output = sess.run(g.mag_output, {g.converter_input: mel_in})
    txt = invert_text(x)
    mag_output = np.squeeze(mag_output[0])
    try:
        wav = spectrogram2wav(mag_output)
        (wav, _) = librosa.effects.trim(wav)
        write((cdir + '/{}mag.wav'.format(mname)), hp.sr, wav)
        ret.append([txt, wav, (typeS + '_world'), mel_in, mag_output])
    except Exception:
        sys.exc_clear()
    return ret

def synthesize_part(grp, config, gs, x_train, g_conv):
    if (len(x_train) > hp.batch_size):
        x_train = random.sample(x_train, hp.batch_size)
    else:
        x_train = x_train[0]
    x_test = load_test_data()
    rand = random.randint(0, (hp.batch_size - 1))
    x_train = x_train[rand]
    x_test = x_test[rand]
    wavs = []
    if (g_conv is None):
        with grp.graph.as_default():
            sv = tf.train.Supervisor(logdir=config.log_dir)
            with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                print(('Restoring checkpoint : ' + tf.train.latest_checkpoint(config.log_dir)))
                sv.saver.restore(sess, tf.train.latest_checkpoint(config.log_dir))
                wavs = create_write_files(wavs, sess, grp, x_train, (('sample_' + str(gs)) + '_train_'), config.log_dir, 'train')
                wavs = create_write_files(wavs, sess, grp, x_test, (('sample_' + str(gs)) + '_test_'), config.log_dir, 'test')
                sess.close()
    else:
        with grp.graph.as_default():
            sv = tf.train.Supervisor(logdir=config.log_dir)
            with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                print(('Restoring checkpoint : ' + tf.train.latest_checkpoint(config.log_dir)))
                sv.saver.restore(sess, tf.train.latest_checkpoint(config.log_dir))
                mel_out1 = create_mel(sess, grp, x_train)
                mel_out2 = create_mel(sess, grp, x_test)
                sess.close()
        with g_conv.graph.as_default():
            sv_conv = tf.train.Supervisor(logdir=config.load_converter)
            with sv_conv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess_conv:
                print(('Restoring checkpoint : ' + tf.train.latest_checkpoint(config.load_converter)))
                sv_conv.saver.restore(sess_conv, tf.train.latest_checkpoint(config.load_converter))
                wavs = create_write_files_conv(wavs, sess_conv, mel_out1, g_conv, x_train, (('sample_' + str(gs)) + '_train_'), config.log_dir, 'train')
                wavs = create_write_files_conv(wavs, sess_conv, mel_out2, g_conv, x_test, (('sample_' + str(gs)) + '_test_'), config.log_dir, 'test')
                sess_conv.close()
    return wavs
