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

import os
import numpy as np
import tensorflow as tf

from src.layers import Conv2D, Conv3D, ActLayer, DCNPack
from src.ops import resize

from .conv_module import Conv2DNormAct


class PCDAlign(object):
    """
    Pyramid, cascade and deformable alignment module in EDVR.

    Args:
        num_feat: int, number of channels multiplier in the intermediate layers.
        num_conv_groups: int, number of groups in convolution in dcn.
        deformable_groups: int, number of groups in offsets in dcn.
        dcn_impl: str, version of dcn operator. Possible choice in ('tf', 'npu').
        upsample_method: str, method of resize operator. Possible choice in
            ('bilinear', 'bicubic').
        align_corners: boolean, used in resize. Whether to align corners during
            resize.
    """
    def __init__(self, num_feat=64, num_conv_groups=1, deformable_groups=1, 
                 dcn_impl='npu', upsample_method='bilinear', align_corners=True):
        self.mid_channels = num_feat
        self.num_deform_groups = deformable_groups
        self.num_groups = num_conv_groups
        self.upsample_method = upsample_method
        self.dcn_impl = dcn_impl
        self.align_corners = align_corners

    def __call__(self, neighbor_feats, ref_feats, 
                 act_cfg=dict(type='LeakyRelu', alpha=0.1), 
                 name='pcd_align'):
        """Forward pass of PCD module.

        Args:
            neighbor_feats: list[tensor], the multi-scale feature maps of a 
                single neighbor frame.
            ref_feats: list[tensor], the multi-scale feature maps of the center
                frame.
            act_cfg: dict, specify the activation `type` and other parameters.
            name: str, variable scope name.
        
        Returns:
            tensor, aligned multi-frame features.
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # The number of pyramid levels is 3.
            assert len(neighbor_feats) == 3 and len(ref_feats) == 3, (
                'The length of neighbor_feats and ref_feats must be both 3, '
                'but got {} and {}'.format(len(neighbor_feats), len(ref_feats)))

            # Pyramids
            upsampled_offset, upsampled_feat = None, None
            for i in range(3, 0, -1):
                with tf.variable_scope('level{}'.format(i)):
                    offset = tf.concat([neighbor_feats[i - 1], ref_feats[i - 1]], axis=-1)
                    offset = Conv2DNormAct(self.mid_channels, act_cfg=act_cfg, name='offset_conv1')(offset)
                    if i == 3:
                        offset = Conv2DNormAct(self.mid_channels, act_cfg=act_cfg, name='offset_conv2')(offset)
                    else:
                        offset = tf.concat([offset, upsampled_offset], axis=-1)
                        offset = Conv2DNormAct(self.mid_channels, act_cfg=act_cfg, name='offset_conv2')(offset)
                        offset = Conv2DNormAct(self.mid_channels, act_cfg=act_cfg, name='offset_conv3')(offset)

                    feat = DCNPack(self.mid_channels, kernel_size=[3, 3], padding='same',
                                   num_deform_groups=self.num_deform_groups, num_groups=self.num_groups,
                                   name='dcn_l{}'.format(i), impl=self.dcn_impl,
                                   )(neighbor_feats[i - 1], offset)
                    if i == 3:
                        feat = ActLayer(act_cfg)(feat)
                    else:
                        feat = tf.concat([feat, upsampled_feat], axis=-1)
                        feat = Conv2DNormAct(self.mid_channels, act_cfg=act_cfg if i == 2 else None,
                                          name='feat_conv')(feat)

                    if i > 1:
                        # upsample offset and features
                        upsampled_offset = resize(
                            offset, size=[offset.shape[1] * 2, offset.shape[2] * 2], align_corners=self.align_corners,
                            name='upsample_offset{}'.format(i), method=self.upsample_method)
                        upsampled_offset = upsampled_offset * 2
                        upsampled_feat = resize(
                            feat, size=[feat.shape[1] * 2, feat.shape[2] * 2], align_corners=self.align_corners,
                            name='upsample_feat{}'.format(i), method=self.upsample_method)

            # Cascading
            offset = tf.concat([feat, ref_feats[0]], axis=-1)
            offset = Conv2DNormAct(self.mid_channels, act_cfg=act_cfg, name='cas_offset_conv1')(offset)
            offset = Conv2DNormAct(self.mid_channels, act_cfg=act_cfg, name='cas_offset_conv2')(offset)
            feat = DCNPack(self.mid_channels, kernel_size=[3, 3], padding='same',
                           num_deform_groups=self.num_deform_groups, name='dcn_cas', 
                           impl=self.dcn_impl)(feat, offset)
            feat = ActLayer(act_cfg)(feat)

            return feat


class PCWoDCN(object):
    """
    A verbose pyramid and cascade module.

    Args:
        num_feat: int, number of channels multiplier in the intermediate layers.
        upsample_method: str, method of resize operator. Possible choice in
            ('bilinear', 'bicubic').
        align_corners: boolean, used in resize. Whether to align corners during
            resize.
    """

    def __init__(self, num_feat=64, upsample_method='bilinear',
                 align_corners=True):
        self.mid_channels = num_feat
        self.upsample_method = upsample_method
        self.align_corners = align_corners

    def __call__(self, neighbor_feats, ref_feats,
                 act_cfg=dict(type='LeakyRelu', alpha=0.1),
                 name='pcd_align'):
        """Forward pass of PCD module.

        Args:
            neighbor_feats: list[tensor], the multi-scale feature maps of a
                single neighbor frame.
            ref_feats: list[tensor], the multi-scale feature maps of the center
                frame.
            act_cfg: dict, specify the activation `type` and other parameters.
            name: str, variable scope name.

        Returns:
            tensor, aligned multi-frame features.
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # The number of pyramid levels is 3.
            assert len(neighbor_feats) == 3 and len(ref_feats) == 3, (
                'The length of neighbor_feats and ref_feats must be both 3, '
                'but got {} and {}'.format(len(neighbor_feats), len(ref_feats)))

            # Pyramids
            upsampled_offset, upsampled_feat = None, None
            for i in range(3, 0, -1):
                with tf.variable_scope('level{}'.format(i)):
                    feat = Conv2DNormAct(self.mid_channels, kernel_size=[3, 3],
                                   padding='same', name='pc_conv{}'.format(i))(neighbor_feats[i - 1])
                    if i == 3:
                        feat = ActLayer(act_cfg)(feat)
                    else:
                        feat = tf.concat([feat, upsampled_feat], axis=-1)
                        feat = Conv2DNormAct(self.mid_channels,
                                             act_cfg=act_cfg if i == 2 else None,
                                             name='feat_conv')(feat)

                    if i > 1:
                        upsampled_feat = resize(
                            feat, size=[feat.shape[1] * 2, feat.shape[2] * 2],
                            align_corners=self.align_corners,
                            name='upsample_feat{}'.format(i),
                            method=self.upsample_method)

            # Cascading
            feat = Conv2DNormAct(self.mid_channels, kernel_size=[3, 3],
                           padding='same', name=f'pc_final_conv{i}')(feat)
            feat = ActLayer(act_cfg)(feat)

            return feat


class TSAFusion(object):
    """Fusiong of temporal and spatial attention.

    Args:
        num_frames: int, number of input frames.
        num_feat: int, multiplier of the filters number in the middle layers.
        upsample_method: str, resize method. Possible choices in 
            ('bilinear', 'bicubic').
        align_corners: boolean, whether to align with corners when resize.
    """
    def __init__(self, num_frames, num_feat, upsample_method='bilinear', 
                 align_corners=True):
        self.num_frames = num_frames
        self.num_feat = num_feat
        self.upsample_method = upsample_method
        self.align_corners = align_corners

    def __call__(self, aligned_feat, act_cfg=dict(type='LeakyRelu', alpha=0.1)):
        """Forward pass.

        Args:
            aligned_feat: tensor
            act_cfg: dict, specify the activation `type` and other parameters.

        Returns:
            tensor, aggregated multi-frame features.
        """
        with tf.variable_scope('tsa_fusion', reuse=tf.AUTO_REUSE):
            # temporal attention
            embedding_ref = Conv2D(self.num_feat, name='temporal_attn1')(aligned_feat[self.num_frames//2])

            # corr_l = []  # correlation list
            aligned_feat_list = []
            for i in range(self.num_frames):
                emb = Conv2D(self.num_feat, name='temporal_attn2')(aligned_feat[i])
                emb = tf.cast(emb, tf.float32)
                corr = tf.reduce_sum(emb * embedding_ref, axis=-1, keep_dims=True)  # (n, h, w, 1)
                # corr_l.append(corr)

                corr_prob = tf.nn.sigmoid(corr)
                aligned_feat_list.append(corr_prob * aligned_feat[i])
            aligned_feat = tf.concat(aligned_feat_list, axis=-1)  # (n, h, w, t*c)
            feat = Conv2DNormAct(self.num_feat, kernel_size=(1, 1), act_cfg=act_cfg, name='feat_fusion')(aligned_feat)

            # spatial attention
            attn = Conv2DNormAct(self.num_feat, kernel_size=(1, 1), act_cfg=act_cfg, name='spatial_attn1')(aligned_feat)
            attn_max = tf.nn.max_pool2d(attn, 3, 2, 'SAME')
            attn_avg = tf.nn.avg_pool(attn, 3, 2, 'SAME')
            attn = Conv2DNormAct(self.num_feat, kernel_size=(1, 1),
                              act_cfg=act_cfg, name='spatial_attn2')(tf.concat([attn_max, attn_avg], axis=-1))
            # pyramid levels
            attn_level = Conv2DNormAct(self.num_feat, kernel_size=(1, 1), act_cfg=act_cfg, name='spatial_attn_l1')(attn)
            attn_max = tf.nn.max_pool2d(attn_level, 3, 2, 'SAME')
            attn_avg = tf.nn.avg_pool(attn_level, 3, 2, 'SAME')
            attn_level = Conv2DNormAct(self.num_feat, act_cfg=act_cfg, name='spatial_attn_l2')\
                                   (tf.concat([attn_max, attn_avg], axis=-1))
            attn_level = Conv2DNormAct(self.num_feat, act_cfg=act_cfg, name='spatial_attn_l3')(attn_level)

            attn_level = resize(
                attn_level, size=[attn_level.shape[1] * 2, attn_level.shape[2] * 2],
                align_corners=self.align_corners,
                name='upsample1', method=self.upsample_method)

            attn = Conv2DNormAct(self.num_feat, act_cfg=act_cfg, name='spatial_attn3')(attn) + attn_level
            attn = Conv2DNormAct(self.num_feat, kernel_size=(1, 1), act_cfg=act_cfg, name='spatial_attn4')(attn)

            attn = resize(
                attn, size=[attn.shape[1] * 2, attn.shape[2] * 2],
                align_corners=self.align_corners,
                name='upsample2', method=self.upsample_method)
            attn = Conv2D(self.num_feat, name='spatial_attn5')(attn)
            attn = Conv2DNormAct(self.num_feat, kernel_size=(1, 1), act_cfg=act_cfg, name='spatial_attn_add1')(attn)
            attn_add = Conv2D(self.num_feat, kernel_size=(1, 1), name='spatial_attn_add2')(attn)
            
            attn = tf.cast(attn, tf.float32)
            attn = tf.nn.sigmoid(attn)
            
            feat = tf.cast(feat, tf.float32)
            attn_add = tf.cast(attn_add, tf.float32)

            # after initialization, * 2 makes (attn * 2) to be close to 1.
            feat = feat * attn * 2 + attn_add
            return feat
