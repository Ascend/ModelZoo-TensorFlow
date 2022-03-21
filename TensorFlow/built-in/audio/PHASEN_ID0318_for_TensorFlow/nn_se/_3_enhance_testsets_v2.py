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
import os
import tensorflow as tf
import collections
from pathlib import Path
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import sys
import math

from .utils import misc_utils
from .utils import audio
from .inference import build_SMG
from .inference import enhance_one_wav
from .inference import SMG

from .FLAGS import PARAM

test_processor = 1
smg = None

slice_len = 10000
strides = slice_len // 16


def enhance_mini_process(noisy_dir, enhanced_save_dir):
  global smg
  if smg is None:
    smg = build_SMG()

  noisy_wav, sr = audio.read_audio(noisy_dir)

  n_samples = noisy_wav.shape[0]
  slice_num = math.ceil((n_samples - slice_len) / strides) + 1
  out_wav = np.zeros(n_samples)
  win = np.zeros(n_samples)
  for j in range(slice_num):
    # When the last frame is less than a long time, some sample
    # points are obtained from the previous frame to fill
    if j == slice_num - 1:
      slice_noise = noisy_wav[-slice_len:]
    else:
      slice_noise = noisy_wav[j * strides: j * strides + slice_len]
    #slice_noise *= window
    output_slice = enhance_one_wav(smg, slice_noise)
    #output_slice /= window

    if j == slice_num - 1:
      output_slice = output_slice[j * strides - n_samples:]
      out_wav[j * strides:] += output_slice
      win[j * strides:] += np.ones(output_slice.shape[0])
    else:
      out_wav[j * strides: j * strides + slice_len] += output_slice
      win[j * strides: j * strides + slice_len] += np.ones(slice_len)
  out_wav /= win
  assert out_wav.shape[0] == n_samples

  enhanced_wav = enhance_one_wav(smg, noisy_wav)
  noisy_name = Path(noisy_dir).stem
  audio.write_audio(os.path.join(enhanced_save_dir, noisy_name+'_enhanced.wav'),
                    enhanced_wav, PARAM.sampling_rate)


def enhance_one_testset(testset_dir, enhanced_save_dir):
  testset_path = Path(testset_dir)
  noisy_path_list = list(map(str, testset_path.glob("*.wav")))
  func = partial(enhance_mini_process, enhanced_save_dir=enhanced_save_dir)
  job = Pool(test_processor).imap(func, noisy_path_list)
  list(tqdm(job, "Enhancing", len(noisy_path_list), unit="test wav", ncols=60))


def main():
  for testset_name in PARAM.test_noisy_sets:
    print("Enhancing %s:" % testset_name, flush=True)
    _dir = misc_utils.enhanced_testsets_save_dir(testset_name+'_v2')
    if _dir.exists():
      import shutil
      shutil.rmtree(str(_dir))
    _dir.mkdir(parents=True)
    testset_dir = str(misc_utils.datasets_dir().joinpath(testset_name))
    enhanced_save_dir = str(_dir)
    enhance_one_testset(testset_dir, enhanced_save_dir)

if __name__ == "__main__":
  misc_utils.initial_run(sys.argv[0].split("/")[-2])

  if len(sys.argv) > 1:
    test_processor = int(sys.argv[1])
  main()

  """
  run cmd:
  `OMP_NUM_THREADS=1 python -m xx._3_enhance_testsets 3`
  [csig,cbak,cvol,pesq,snr,ssnr]=evaluate_all('/home/lhf/worklhf/PHASEN/noisy_datasets_16k/clean_testset_wav','/home/lhf/worklhf/PHASEN/exp/se_reMagMSE_cnn/enhanced_testsets/noisy_testset_wav')
  """
