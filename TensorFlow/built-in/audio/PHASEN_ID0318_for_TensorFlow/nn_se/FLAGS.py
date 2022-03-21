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
import sys

class StaticKey(object):
  MODEL_TRAIN_KEY = 'train'
  MODEL_VALIDATE_KEY = 'validation'
  MODEL_INFER_KEY = 'infer'

  def config_name(self): # config_name
    return self.__class__.__name__

class BaseConfig(StaticKey):
  VISIBLE_GPU = "0"
  root_dir = os.path.dirname(os.path.abspath(__file__))
  # datasets_name = 'vctk_musan_datasets'
  datasets_name = 'noisy_datasets_16k'
  '''
  # dir to store log, model and results files:
  $root_dir/$datasets_name: datasets dir
  $root_dir/exp/$config_name/log: logs(include tensorboard log)
  $root_dir/exp/$config_name/ckpt: ckpt
  $root_dir/exp/$config_name/enhanced_testsets: enhanced results
  $root_dir/exp/$config_name/hparams
  '''

  min_TF_version = "1.14.0"


  train_noisy_set = 'noisy_trainset_wav'
  train_clean_set = 'clean_trainset_wav'
  validation_noisy_set = 'noisy_testset_wav'
  validation_clean_set = 'clean_testset_wav'
  test_noisy_sets = ['noisy_testset_wav']
  test_clean_sets = ['clean_testset_wav']

  # n_train_set_records = 11572
  # n_val_set_records = 824
  # n_test_set_records = 824
  n_train_set_records = 80
  n_val_set_records = 88
  n_test_set_records = 88

  train_val_wav_seconds = 3.0

  batch_size = 4
  n_processor_tfdata = 4

  model_name = "PHASEN"

  relative_loss_epsilon = 0.1
  RL_idx = 2.0
  st_frame_length_for_loss = 512
  st_frame_step_for_loss = 256
  sampling_rate = 16000
  frame_length = 400
  frame_step = 160
  fft_length = 512
  max_keep_ckpt = 30
  optimizer = "Adam" # "Adam" | "RMSProp"
  learning_rate = 0.0005
  max_gradient_norm = 5.0

  GPU_RAM_ALLOW_GROWTH = True
  GPU_PARTION = 0.97

  s_epoch = 1
  max_epoch = int(sys.argv[1])
  batches_to_logging = 200000

  max_model_abandon_time = 3
  no_abandon = True
  use_lr_warmup = True # true: lr warmup; false: lr halving
  warmup_steps = 6000. # for (use_lr_warmup == true)
  start_halving_impr = 0.01 # no use for (use_lr_warmup == true)
  lr_halving_rate = 0.7 # no use for (use_lr_warmup == true)

  # melMat: tf.contrib.signal.linear_to_mel_weight_matrix(129,129,8000,125,3900)
  # plt.pcolormesh
  # import matplotlib.pyplot as plt

  """
  @param not_transformed_losses/transformed_losses[add FT before loss_name]:
  loss_mag_mse, loss_spec_mse, loss_wav_L1, loss_wav_L2,
  loss_mag_reMse, loss_reSpecMse, loss_reWavL2,
  """
  sum_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  sum_losses_w = []
  show_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  show_losses_w = []
  stop_criterion_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  stop_criterion_losses_w = []

  # just for "DISCRIMINATOR_AD_MODEL"

  channel_A = 96
  channel_P = 48
  prenet_A_kernels = [[1,7], [7, 1]]
  prenet_P_kernels = [[5,3], [25,1]]
  n_TSB = 3
  frequency_dim = 257
  loss_compressedMag_idx = 0.3

  clip_grads = False


class p40(BaseConfig):
  n_processor_gen_tfrecords = 56
  n_processor_tfdata = 8
  # GPU_PARTION = 0.27
  root_dir = os.path.dirname(os.path.abspath(__file__))


class se_phasen_001(p40): # done p40
  '''
  phasen 001
  loss_compressedMag_mse + loss_compressedStft_mse
  '''
  sum_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  sum_losses_w = []
  show_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse", "loss_CosSim"]
  show_losses_w = []
  stop_criterion_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  stop_criterion_losses_w = []
  channel_A = 96
  channel_P = 48
  n_TSB = 3

class se_phasen_002(p40): # done p40
  '''
  phasen 002
  loss_mag_reMse|0050 + loss_CosSim
  '''
  sum_losses = ["loss_mag_reMse", "loss_CosSim"]
  sum_losses_w = []
  show_losses = ["loss_mag_reMse", "loss_CosSim"]
  show_losses_w = []
  stop_criterion_losses = ["loss_mag_reMse", "loss_CosSim"]
  stop_criterion_losses_w = []
  relative_loss_epsilon = 0.05
  channel_A = 96
  channel_P = 48
  n_TSB = 3

class se_phasen_003(p40): # pendding p40
  '''
  phasen 003
  loss_mag_mse + loss_stft_mse
  '''
  sum_losses = ["loss_mag_mse", "loss_stft_mse"]
  sum_losses_w = []
  show_losses = ["loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  show_losses_w = []
  stop_criterion_losses = ["loss_mag_mse", "loss_stft_mse"]
  stop_criterion_losses_w = []
  channel_A = 96
  channel_P = 48
  n_TSB = 3

class se_phasen_004_selfConv2d(p40): # running p40
  '''
  phasen 004
  loss_compressedMag_mse + loss_compressedStft_mse
  Ca = 24, Cp = 12
  '''
  sum_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  sum_losses_w = []
  show_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse", "loss_CosSim"]
  show_losses_w = []
  stop_criterion_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  stop_criterion_losses_w = []
  channel_A = 24
  channel_P = 12
  n_TSB = 3

class se_phasen_004_clipGrads(BaseConfig): # pendding 15123
  '''
  phasen 004_clipGrads
  loss_compressedMag_mse + loss_compressedStft_mse
  Ca = 12, Cp = 8
  '''
  sum_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  sum_losses_w = []
  show_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse", "loss_stft_mse"]
  show_losses_w = []
  stop_criterion_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  stop_criterion_losses_w = []
  channel_A = 12
  channel_P = 8
  n_TSB = 3
  clip_grads = True

class se_phasen_005(p40): # pendding p40
  '''
  phasen 005
  loss_compressedMag_mse + loss_compressedStft_mse
  Ca = 12, Cp = 8
  '''
  sum_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  sum_losses_w = []
  show_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse", "loss_stft_mse"]
  show_losses_w = []
  stop_criterion_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  stop_criterion_losses_w = []
  channel_A = 12
  channel_P = 8
  n_TSB = 3
  learning_rate = 1e-3


PARAM = se_phasen_004_selfConv2d ###

# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 python -m xxx._2_train
