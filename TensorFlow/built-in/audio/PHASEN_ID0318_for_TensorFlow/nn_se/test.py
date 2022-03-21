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
from .dataloader.dataloader import get_batch_inputs_from_nosiyCleanDataset
from .utils import audio
from .utils.assess import core as assess_core
from .FLAGS import PARAM
import tensorflow as tf
import numpy as np
import os
from .utils import misc_utils


def wav_through_stft_istft():
  frame_length = 400
  frame_step = 160
  n_fft = 512
  wav_dir = os.path.join("exp", "test", "p232_001.wav")
  wav, sr = audio.read_audio(str(wav_dir))
  print("sr", sr)
  wav_batch = np.array([wav], dtype=np.float32)
  spec = misc_utils.tf_wav2stft(wav_batch, frame_length, frame_step, n_fft)

  # mag = tf.math.abs(spec)
  # phase = tf.math.angle(spec)
  # spec2 = tf.complex(mag, 0.0) * tf.exp(tf.complex(0.0, phase))

  wav2 = misc_utils.tf_stft2wav(spec, frame_length, frame_step, n_fft)

  sess = tf.compat.v1.Session()
  wav_np = sess.run(wav2)
  print(np.shape(wav_batch), np.shape(wav_np))
  wav_np = wav_np[0][:len(wav)]

  pesq = assess_core.calc_pesq(wav, wav_np, sr)
  sdr = assess_core.calc_sdr(wav, wav_np, sr)
  stoi = assess_core.calc_stoi(wav, wav_np, sr)
  print(pesq, sdr, stoi)

  # audio.write_audio(os.path.join(PARAM.root_dir,
  #                                "exp/test/p265_002_reconstructed_step%d.wav" % step),
  #                   wav_np, PARAM.sampling_rate)


def test_dataloader_from_noisy_clean_datasets():
  batch=get_batch_inputs_from_nosiyCleanDataset(PARAM.train_noisy_path, PARAM.train_clean_path)
  sess=tf.compat.v1.Session()
  sess.run(batch.initializer)
  clean, mixed=sess.run([batch.clean, batch.mixed])
  print(np.shape(clean), np.shape(mixed))
  if not os.path.exists("exp/test"):
    os.makedirs("exp/test/")
  audio.write_audio(os.path.join(PARAM.root_dir,"exp/test/clean.wav"),clean[0],PARAM.sampling_rate)
  audio.write_audio(os.path.join(PARAM.root_dir,"exp/test/mixed.wav"),mixed[0],PARAM.sampling_rate)


if __name__ == "__main__":
  # test_dataloader_py()
  wav_through_stft_istft()
  # wav_through_stft_istft_noreconstructed()
  # test_dataloader_from_noisy_clean_datasets()
