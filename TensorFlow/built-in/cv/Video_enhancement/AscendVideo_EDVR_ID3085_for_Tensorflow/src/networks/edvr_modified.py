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

import math
import os

import numpy as np
import tensorflow as tf

from src.layers import Conv2D, ActLayer
from src.modules import Conv2DNormAct, ResBlockNoBN
from src.ops import depth_to_space, resize, split
from src.modules.edvr_submodules import PCDAlign, TSAFusion, PCWoDCN
from src.networks.base_model import Base
from src.runner.common import name_space
from src.utils.file_io import imwrite


class EDVRVariant(Base):
    """EDVR video super-resolution network.

    Args:
        cfg: yacs node. EDVR configures configured in edvr_config.py. 
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.with_tsa = cfg.edvr.with_tsa
        self.mid_channels = cfg.edvr.mid_channels
        self.num_groups = cfg.edvr.num_groups
        self.num_deform_groups = cfg.edvr.num_deform_groups
        self.num_blocks_extraction = cfg.edvr.num_blocks_extraction
        self.num_blocks_reconstruction = cfg.edvr.num_blocks_reconstruction

        if cfg.edvr.use_dcn:
            self.align_module = PCDAlign(self.mid_channels, 1, self.num_deform_groups,
                                         dcn_impl='npu',
                                         upsample_method=self.cfg.edvr.upsampling,
                                         align_corners=self.cfg.edvr.align_corners)
        else:
            self.align_module = PCWoDCN(self.mid_channels,
                                        upsample_method=self.cfg.edvr.upsampling,
                                        align_corners=self.cfg.edvr.align_corners)

        self.tsa_fusion_module = TSAFusion(self.num_net_input_frames, 
                                           self.mid_channels, 
                                           self.cfg.edvr.upsampling,
                                           align_corners=self.cfg.edvr.align_corners)
        
    def feature_extraction(self, x, act_cfg=dict(type='LeakyRelu', alpha=0.1)):
        # extract LR features
        with tf.variable_scope('extraction', reuse=tf.AUTO_REUSE):
            # L1
            # l1_feat = tf.reshape(x, [-1, x.shape[2], x.shape[3], x.shape[4]])
            l1_feat = Conv2D(self.mid_channels, name='conv_first')(x)
            l1_feat = ActLayer(act_cfg)(l1_feat)
            l1_feat = ResBlockNoBN(num_blocks=self.num_blocks_extraction, mid_channels=self.mid_channels)(l1_feat)
            # L2
            l2_feat = Conv2DNormAct(self.mid_channels, strides=[2, 2], act_cfg=act_cfg, name='feat_l2_conv1')(l1_feat)
            l2_feat = Conv2DNormAct(self.mid_channels, act_cfg=act_cfg, name='feat_l2_conv2')(l2_feat)
            # L3
            l3_feat = Conv2DNormAct(self.mid_channels, strides=[2, 2], act_cfg=act_cfg, name='feat_l3_conv1')(l2_feat)
            l3_feat = Conv2DNormAct(self.mid_channels, act_cfg=act_cfg, name='feat_l3_conv2')(l3_feat)

            return l1_feat, l2_feat, l3_feat

    def reconstruction(self, feat, x_center, act_cfg=dict(type='LeakyRelu', alpha=0.1)):
        # reconstruction
        out_channel = x_center.get_shape().as_list()[-1]
        with tf.variable_scope('reconstruction', reuse=tf.AUTO_REUSE):
            out = ResBlockNoBN(num_blocks=self.num_blocks_reconstruction, mid_channels=self.mid_channels)(feat)

            out = Conv2D(self.mid_channels * (self.scale ** 2), name='upsample')(out)
            out = depth_to_space(out, self.scale)

            out = Conv2D(self.mid_channels, name='conv_hr')(out)
            out = ActLayer(act_cfg)(out)
            out = Conv2D(out_channel, name='conv_last')(out)
            
            base = resize(
                x_center,
                size=[x_center.shape[1] * self.scale, x_center.shape[2] * self.scale],
                align_corners=self.cfg.edvr.align_corners,
                name='img_upsample', method=self.cfg.edvr.upsampling)
            base = tf.cast(base, tf.float32)
            out = tf.cast(out, tf.float32)
            self.residual = out
            out += base

            return out

    def build_generator(self, x):
        # shape of x: [B,T_in,H,W,C]
        with tf.variable_scope(self.generative_model_scope, reuse=tf.AUTO_REUSE):
            if self.cfg.model.input_format_dimension == 4:
                x_shape = x.get_shape().as_list()
                x = tf.reshape(x, [-1, self.num_net_input_frames, *x_shape[1:]])

            x_list = split(x, self.num_net_input_frames, axis=1, keep_dims=False)
            x_center = x_list[self.num_net_input_frames//2]

            l1_feat_list = []
            l2_feat_list = []
            l3_feat_list = []
            for f in range(self.num_net_input_frames):
                l1_feat, l2_feat, l3_feat = self.feature_extraction(x_list[f])
                l1_feat_list.append(l1_feat)
                l2_feat_list.append(l2_feat)
                l3_feat_list.append(l3_feat)

            ref_feats = [  
                l1_feat_list[self.num_net_input_frames//2],
                l2_feat_list[self.num_net_input_frames//2],
                l3_feat_list[self.num_net_input_frames//2]
            ]
            aligned_feat = []

            for i in range(self.num_net_input_frames):
                neighbor_feats = [
                    l1_feat_list[i],
                    l2_feat_list[i],
                    l3_feat_list[i]
                ]
                # aligned_feat.append(self.pcd_align(neighbor_feats, ref_feats))
                aligned_feat.append(self.align_module(neighbor_feats, ref_feats))

            if self.with_tsa:

                feat = self.tsa_fusion_module(aligned_feat)
            else:
                aligned_feat = tf.stack(aligned_feat, axis=3)
                aligned_feat_shape = aligned_feat.shape
                aligned_feat = tf.reshape(aligned_feat, [*aligned_feat_shape[:3], -1])
                feat = Conv2D(self.mid_channels, kernel_size=[1, 1], name='fusion')(aligned_feat)

            # reconstruction
            out = self.reconstruction(feat, x_center)

            return out

    def dump_summary(self, step, summary_dict):
        # Keys of the summary dict correspond to the keys defined base_model "build_generator"
        lr = summary_dict['lr']
        sr = summary_dict['sr']
        hr = summary_dict['hr']

        os.makedirs(os.path.join(self.output_dir, 'intermediate'), exist_ok=True)

        output_file = os.path.join(self.output_dir, 'intermediate', f'step{step:06d}_lr.png')
        imwrite(output_file, np.squeeze(lr[0, self.num_net_input_frames//2]),
                source_color_space=self.cfg.data.color_space)

        output_file = os.path.join(self.output_dir, 'intermediate', f'step{step:06d}_hr.png')
        imwrite(output_file, np.squeeze(hr[0]), source_color_space=self.cfg.data.color_space)

        output_file = os.path.join(self.output_dir, 'intermediate', f'step{step:06d}_sr.png')
        imwrite(output_file, sr[0], source_color_space=self.cfg.data.color_space)
