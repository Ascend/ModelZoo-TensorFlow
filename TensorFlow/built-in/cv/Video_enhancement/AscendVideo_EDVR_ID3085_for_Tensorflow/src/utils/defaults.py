# Copyright 2022 Huawei Technologies Co., Ltd
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

from yacs.config import CfgNode as CN


cfg = CN(new_allowed=True)

cfg.mode = 'train'
cfg.task = 'vsr'  # 'denoise', 'vsr'

# ---------------------------------------------------------------------------- #
# Model (common)
# ---------------------------------------------------------------------------- #
cfg.model = CN(new_allowed=True)
cfg.model.name = 'EDVR'     # Key for model
cfg.model.scope = 'G'       # Scope key for parameters

# for vfi
cfg.model.frame_rate = 2

# for vsr
cfg.model.scale = 4

# The input size as well as the placeholder will be adapted automatically.
# See base_model.py `_cal_input_size` function, and inferencer.py `adapt_input` function.
# The output will be reversed-adapted by the inferencer.
cfg.model.factor_for_adapt_input = 4

# The following num of frames are standalone defined to generalize to model configuration, 
# e.g., extend to temporal, or extend to cascading of models: 
#    num_net_input_frames: the num frames of input LQ when model inference
#    num_net_output_frames: the num frames of output SR when model inference
#    num_data_lq_frames: the num frames of input LQ when dataloader in training
#    num_data_gt_frames: the num frames of target GT when dataloader in training for supervision
cfg.model.num_net_input_frames = 5
cfg.model.num_net_output_frames = 1

# Options for the input dimension
# 4: 4D tensor, with shape [b*frames, h, w, c], used when model frozen
# 5: 5D tensor, with shape [b, frames, h, w, c]
cfg.model.input_format_dimension = 5
cfg.model.convert_output_to_uint8 = False

# ---------------------------------------------------------------------------- #
# Loss (common)
# ---------------------------------------------------------------------------- #
cfg.loss = CN(new_allowed=True)
cfg.loss.content = CN(new_allowed=True)
cfg.loss.content.loss_type = 'L1Loss' 
cfg.loss.content.loss_reduction = 'mean'
cfg.loss.content.loss_margin = 1e-6

# Loss (edge)
cfg.loss.edge = CN(new_allowed=True)
cfg.loss.edge.loss_weight = 0.
cfg.loss.edge.method = 'sobel'  # ['sobel', 'laplacian']

# Loss (perceptual)
# perceptual module
cfg.loss.perceptual = CN(new_allowed=True)
cfg.loss.perceptual.loss_weight = 0.
cfg.loss.perceptual.module = 'vgg_19'
cfg.loss.perceptual.layers = ['vgg_19/conv2/conv2_2', 
                              'vgg_19/conv3/conv3_4',
                              'vgg_19/conv4/conv4_4',
                              'vgg_19/conv5/conv5_4']
cfg.loss.perceptual.layers_weights = [1.0, 1.0, 1.0, 1.0]
# full ckpt file should be '${ckpt_dir}/${module}.ckpt'
cfg.loss.perceptual.ckpt_dir = './pretrained_modules'

# Loss (adv)
cfg.loss.adversarial = CN(new_allowed=True)
cfg.loss.adversarial.loss_weight = 0.
cfg.loss.adversarial.adaptive_strategy = False
cfg.loss.adversarial.d_balance = 0.4
cfg.loss.adversarial.gan_type = 'VanillaGAN'
cfg.loss.adversarial.grad_penalty_weight = 0.1
cfg.loss.adversarial.g_update_interval = 1
cfg.loss.adversarial.d_update_interval = 1
cfg.loss.adversarial.loss_type = 'VanillaAdvLoss'
cfg.loss.adversarial.loss_reduction = 'mean'
cfg.loss.adversarial.norm_type = 'in'
cfg.loss.adversarial.mid_channels = 64
cfg.loss.adversarial.parameter_clip = False
cfg.loss.adversarial.parameter_clip_range = [-0.01, 0.01]

# ---------------------------------------------------------------------------- #
# Data (common)
# ---------------------------------------------------------------------------- #
cfg.data = CN(new_allowed=True)
# For mixture datasets, each should be separated with ':'
cfg.data.data_dir = 'data/reds'

cfg.data.num_data_lq_frames = 5
cfg.data.num_data_gt_frames = 1
# File extension. For HDR, it should be 'exr'. For others, it would be 'png'
# Note: it's only used in inference dataset for now.
cfg.data.extension = 'png'
# ['bgr', 'rgb', 'lab'], default to `rgb`
cfg.data.color_space = 'rgb'
cfg.data.normalized = True

# training
cfg.data.train = CN(new_allowed=True)
cfg.data.train.degradation = CN(new_allowed=True)
cfg.data.train.degradation.online = False
cfg.data.train.degradation.options = \
"""
GaussianNoise:
    input_dim: 4
    noise_level: 20
IsotropicGaussianBlur2D:
    input_dim: 4
    kernel_size: 15
    sigma: 10
BicubicDownsampling:
    input_dim: 4
    scale: 4
batch_apply: False
"""
cfg.data.train.gt_enhancement = False
cfg.data.train.set_file = 'train.json'
cfg.data.train.batch_size = 4
cfg.data.train.input_size = [64, 64]

cfg.data.train.augmentation = CN(new_allowed=True)
cfg.data.train.augmentation.apply = True
cfg.data.train.augmentation.interval_list = [1, 2]
# Augmentation options, should be a doc-string (yml formatted),
# for example, the following. Note that in 'RandomCrop', the
# 'crop_size' and 'scale' will be further provided by the
# _TrainDataset class based on other configurations. Therefore,
# there is no need for users to explicitly provide these two
# parameters. The reason of such design is to avoid duplicate
# configure of the two parameters.
cfg.data.train.augmentation.options = \
"""
RandomCrop:
    input_dim: 4
RandomTemporalReverse:
    input_dim: 4
RandomFlipLeftRight:
    input_dim: 4
RandomFlipUpDown:
    input_dim: 4
shuffle_transformers_order: False
"""

# inference
cfg.data.inference = CN()
cfg.data.inference.auto_adapt_input = True
cfg.data.inference.batch_size = 1
cfg.data.inference.input_size = [180, 320]
cfg.data.inference.eval_using_patch = False
cfg.data.inference.patch_pad_size = [32, 32]

# Specify the max size of the input supported by the network.
# When releasing, the program will adaptively use different strategies on whether do
# inference with the whole image input or stitching.
cfg.data.inference.max_input_size = [540, 960]
cfg.data.inference.best_patch_size = [540, 640]

# A subset of the given dataset for inference, (min_index, max_index).
# One should set the index **in** the file name, instead of the actual index of the
# file order. For example, the files are:
# - samples
#   |- 0001.png     (file list index 0)
#   |- 0002.png     (file list index 1)
#   |- 0003.png     (file list index 2)
#   |- 0004.png     (file list index 3)
#   `- 0005.png     (file list index 4)
# and the frames 0002.png - 0004.png are about to be inferred. Then the value of
# the following key should be [2, 4] (indices **in** the file name), rather than
# [1, 3] (indices of the file list).
cfg.data.inference.subset_range = []
cfg.data.inference.subset_list = []

# ---------------------------------------------------------------------------- #
# Training (common)
# ---------------------------------------------------------------------------- #
cfg.train = CN(new_allowed=True)
cfg.train.training_scope = ''
cfg.train.pretrained_scope_list = []
cfg.train.pretrained_scope_ckpt = []

cfg.train.optimizer = CN(new_allowed=True)
cfg.train.optimizer.type = 'Adam'

# TODO: add options for optimizer
cfg.train.generator = CN(new_allowed=True)
cfg.train.generator.lr_schedule = CN(new_allowed=True)
cfg.train.generator.lr_schedule.type = 'CosineRestart'
cfg.train.generator.lr_schedule.base_lr = 4e-4
cfg.train.generator.lr_schedule.total_steps = [10000]
cfg.train.generator.lr_schedule.restart_weights = [1, 0.5, 0.5, 0.5]
cfg.train.generator.lr_schedule.min_lr = 1e-7

# Discriminator lr schedule
cfg.train.discriminator = CN(new_allowed=True)
cfg.train.discriminator.lr_schedule = CN(new_allowed=True)
cfg.train.discriminator.lr_schedule.type = 'CosineRestart'
cfg.train.discriminator.lr_schedule.base_lr = 4e-4
cfg.train.discriminator.lr_schedule.total_steps = [150000, 150000, 150000, 150000]
cfg.train.discriminator.lr_schedule.restart_weights = [1, 0.5, 0.5, 0.5]
cfg.train.discriminator.lr_schedule.min_lr = 1e-7

cfg.train.checkpoint_interval = 5000
cfg.train.print_interval = 20
cfg.train.loss_scale = 'off'

cfg.train.use_tensorboard = False
cfg.train.dump_intermediate = False
cfg.train.dump_intermediate_interval = 2000
cfg.train.continue_training = False

cfg.train.output_dir = 'outputs/edvr'

# ---------------------------------------------------------------------------- #
# Session
# ---------------------------------------------------------------------------- #
cfg.session = CN()
cfg.session.mix_precision = False
cfg.session.xla = False

# ---------------------------------------------------------------------------- #
# Env
# ---------------------------------------------------------------------------- #
cfg.env = CN(new_allowed=True)
cfg.env.device = 'npu'
cfg.env.device_ids = [0]
cfg.env.rank_size = 1
cfg.env.root_rank = 0

# ---------------------------------------------------------------------------- #
# Misc
# ---------------------------------------------------------------------------- #
cfg.debug_mode = False 
cfg.log_file = ''
cfg.checkpoint = ''

# ---------------------------------------------------------------------------- #
# Inference
# ---------------------------------------------------------------------------- #
cfg.inference = CN(new_allowed=True)
cfg.inference.write_out = True
cfg.inference.io_backend = 'disk'

# disk scenario
cfg.inference.result_dir = ''
cfg.inference.writer_num_threads = 8
cfg.inference.writer_queue_size = 64   # used in both disk and ffmpeg scenario

# ffmpeg stream scenario
cfg.inference.ffmpeg = CN(new_allowed=True)
cfg.inference.ffmpeg.video_filename = 'test'
cfg.inference.ffmpeg.fps = 25.
cfg.inference.ffmpeg.codec_file = './config/codecs/default_x264.json'
