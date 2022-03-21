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
import tensorflow as tf
import sys
import tensorflow.contrib.slim as slim
import time
from distutils import version
from pathlib import Path
import os

from nn_se.FLAGS import PARAM

def tf_wav2mag(batch_wav, frame_length, frame_step, n_fft):
  cstft = tf.signal.stft(batch_wav, frame_length, frame_step, fft_length=n_fft, pad_end=True)
  feature = tf.math.abs(cstft)
  return feature


def tf_wav2stft(batch_wav, frame_length, frame_step, n_fft):
  cstft = tf.signal.stft(batch_wav, frame_length, frame_step, fft_length=n_fft, pad_end=True)
  return cstft


def tf_stft2wav(batch_stft, frame_length, frame_step, n_fft):
  signals = tf.signal.inverse_stft(batch_stft, frame_length, frame_step, fft_length=n_fft,
                                   window_fn=tf.signal.inverse_stft_window_fn(frame_step)
                                   )
  return signals


# def tf_wav2feature(batch_wav, frame_length, frame_step):
#   if PARAM.feature_type == "WAV":
#     feature = tf.signal.frame(batch_wav, frame_length, frame_step, pad_end=True)
#   elif PARAM.feature_type == "MAG":
#     feature = tf_wav2mag(batch_wav, frame_length, frame_step)
#   elif PARAM.feature_type == "DCT":
#     frames = tf.signal.frame(batch_wav, frame_length, frame_step, pad_end=True) # [batch,time,frame_len]
#     hann_win = tf.reshape(tf.signal.hann_window(frame_length), [1,1,-1])
#     frames = frames*hann_win
#     feature = tf.signal.dct(frames, norm='ortho')
#     feature = feature * 10.0
#   elif PARAM.feature_type == "ComplexDFT":
#     cstft = tf.signal.stft(batch_wav, frame_length, frame_step, pad_end=True) # [batch, time, frequency]
#     stft_real = tf.real(cstft)
#     stft_imag = tf.imag(cstft)
#     feature = tf.concat([stft_real, stft_imag], axis=-1)

#   return feature


# def tf_feature2wav(batch_feature, frame_length, frame_step):
#   if PARAM.feature_type == "WAV":
#     signals = tf.signal.overlap_and_add(batch_feature, frame_step)
#   elif PARAM.feature_type == "AbsDFT":
#     batch_mag, batch_angle = batch_feature
#     cstft = tf.complex(batch_mag, 0.0) * tf.exp(tf.complex(0.0, batch_angle))
#     signals = tf.signal.inverse_stft(cstft, frame_length, frame_step,
#                                      window_fn=tf.signal.inverse_stft_window_fn(frame_step))
#   elif PARAM.feature_type == "DCT":
#     # hann_win = tf.reshape(tf.signal.hann_window(frame_length), [1,1,-1])
#     # frames = frames*hann_win
#     batch_feature = batch_feature / 10.0
#     itrans = tf.signal.idct(batch_feature, norm='ortho')
#     signals = tf.signal.overlap_and_add(itrans, frame_step)
#   elif PARAM.feature_type == "ComplexDFT":
#     batch_feature_real, batch_feature_imag = tf.split(batch_feature, 2, axis=-1)
#     cstft = tf.complex(batch_feature_real, batch_feature_imag)
#     signals = tf.signal.inverse_stft(cstft, frame_length, frame_step,
#                                      window_fn=tf.signal.inverse_stft_window_fn(frame_step))

#   return signals


def initial_run(config_name):
  assert config_name == PARAM().config_name(), (
      "config name error: dir.%s|FLAG.%s." % (config_name, PARAM().config_name()))
  check_tensorflow_version()
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
  print_hparams()


def test_code_out_dir():
  _dir = Path(PARAM.root_dir).joinpath("exp", "test")
  return _dir


def enhanced_testsets_save_dir(testset_name):
  exp_config_name_dir = exp_configName_dir()
  return exp_config_name_dir.joinpath('enhanced_testsets', testset_name)


def hparams_file_dir():
  exp_config_name_dir = exp_configName_dir()
  return exp_config_name_dir.joinpath('hparam')


def ckpt_dir():
  exp_config_name_dir = exp_configName_dir()
  return exp_config_name_dir.joinpath('ckpt')


def test_log_file_dir(testset_name):
  str_snr = "%s.test.log" % testset_name
  log_dir_ = log_dir()
  return log_dir_.joinpath(str_snr)


def train_log_file_dir():
  log_dir_ = log_dir()
  return log_dir_.joinpath('train.log')


def log_dir():
  exp_config_name_dir = exp_configName_dir()
  return exp_config_name_dir.joinpath('log')


def exp_configName_dir():
  return Path(PARAM.root_dir).joinpath('exp', PARAM().config_name())


def datasets_dir():
  return Path(PARAM.root_dir).joinpath(PARAM.datasets_name)


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
  '''Noam scheme learning rate decay
  init_lr: initial learning rate. scalar.
  global_step: scalar.
  warmup_steps: scalar. During warmup_steps, learning rate increases
      until it reaches init_lr.
  '''
  step = tf.cast(global_step + 1, dtype=tf.float32)
  return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def show_variables(vars_):
  slim.model_analyzer.analyze_vars(vars_, print_info=True)
  sys.stdout.flush()


def show_all_variables():
  model_vars = tf.trainable_variables()
  show_variables(model_vars)


def print_log(msg, log_file=None, no_time=False, no_prt=False):
  if log_file is not None:
    log_file = str(log_file)
  if not no_time:
      time_stmp = "%s | " % time.ctime()
      msg = time_stmp+msg
  if not no_prt:
    print(msg, end='', flush=True)
  if log_file:
    with open(log_file, 'a+') as f:
        f.write(msg)


def check_tensorflow_version():
  # LINT.IfChange
  min_tf_version = PARAM.min_TF_version
  # LINT.ThenChange(<pwd>/nmt/copy.bara.sky)
  if not (version.LooseVersion(tf.__version__) == version.LooseVersion(min_tf_version)):
    raise EnvironmentError("Tensorflow version must be '%s'" % min_tf_version)


def save_hparams(f):
  f = open(f, 'a+')
  import nn_se.FLAGS as FLAGS
  self_dict = FLAGS.PARAM.__dict__
  self_dict_keys = self_dict.keys()
  f.writelines('FLAGS.PARAM:\n')
  supper_dict = FLAGS.BaseConfig.__dict__
  for key in sorted(supper_dict.keys()):
    if key in self_dict_keys:
      f.write('%s:%s\n' % (key,self_dict[key]))
    else:
      f.write('%s:%s\n' % (key,supper_dict[key]))
  f.write('--------------------------\n\n')

  f.write('Short hparams:\n')
  [f.write("%s:%s\n" % (key, self_dict[key])) for key in sorted(self_dict_keys)]
  f.write('--------------------------\n\n')


def print_hparams(short=True):
  from nn_se import FLAGS
  self_dict = FLAGS.PARAM.__dict__
  self_dict_keys = self_dict.keys()
  print('\n--------------------------\n')
  print('Short hparams:')
  [print("%s:%s" % (key, self_dict[key])) for key in sorted(self_dict_keys)]
  print('--------------------------\n')
  if not short:
    print('FLAGS.PARAM:')
    supper_dict = FLAGS.BaseConfig.__dict__
    for key in sorted(supper_dict.keys()):
      if key in self_dict_keys:
        print('%s:%s' % (key,self_dict[key]))
      else:
        print('%s:%s' % (key,supper_dict[key]))
    print('--------------------------\n')
    print('Short hparams:')
    [print("%s:%s" % (key, self_dict[key])) for key in sorted(self_dict_keys)]
    print('--------------------------\n')
