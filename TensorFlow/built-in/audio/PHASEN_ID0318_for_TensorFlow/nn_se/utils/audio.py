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
from numpy import linalg
import numpy as np
import soundfile as sf
import librosa

from nn_se.FLAGS import PARAM

'''
soundfile.info(file, verbose=False)
soundfile.available_formats()
soundfile.available_subtypes(format=None)
soundfile.read(file, frames=-1, start=0, stop=None, dtype='float64', always_2d=False, fill_value=None, out=None, samplerate=None, channels=None, format=None, subtype=None, endian=None, closefd=True)
soundfile.write(file, data, samplerate, subtype=None, endian=None, format=None, closefd=True)
'''

def read_audio(file):
  data, sr = sf.read(file)
  if sr != PARAM.sampling_rate:
    data = librosa.resample(data, sr, PARAM.sampling_rate, res_type='kaiser_fast')
    print('resample wav(%d to %d) :' % (sr, PARAM.sampling_rate), file)
    # librosa.output.write_wav(file, data, PARAM.sampling_rate)
  return data, PARAM.sampling_rate

def write_audio(file, data, sr):
  return sf.write(file, data, sr)

def repeat_to_len(wave, repeat_len, random_trunc_long_wav=False):
  wave_len = len(wave)
  if random_trunc_long_wav and wave_len > repeat_len:
    random_s = np.random.randint(wave_len-repeat_len+1)
    wave = wave[random_s:random_s+repeat_len]
    return wave

  while len(wave) < repeat_len:
    wave = np.tile(wave, 2)
  wave = wave[0:repeat_len]
  return wave


def repeat_to_len_2(wave1, wave2, repeat_len, random_trunc_long_wav=False):
  wave_len1 = len(wave1)
  wave_len2 = len(wave2)
  assert wave_len1 == wave_len2, 'wav length not match.'
  wave_len = wave_len1
  if random_trunc_long_wav and wave_len > repeat_len:
    random_s = np.random.randint(wave_len-repeat_len+1)
    wave1 = wave1[random_s:random_s+repeat_len]
    wave2 = wave2[random_s:random_s+repeat_len]
    return wave1, wave2

  while len(wave1) < repeat_len:
    wave1 = np.tile(wave1, 2)
    wave2 = np.tile(wave2, 2)
  wave1 = wave1[0:repeat_len]
  wave2 = wave2[0:repeat_len]
  return wave1, wave2


def mix_wav_by_SNR(waveData, noise, snr):
  As = linalg.norm(waveData)
  An = linalg.norm(noise)

  alpha = As/(An*(10**(snr/20))) if An != 0 else 0
  waveMix = (waveData+alpha*noise)/(1.0+alpha)
  # return mixed, speech_weight, noise_weight
  return waveMix, 1.0/(1.0+alpha), alpha/(1.0+alpha)
