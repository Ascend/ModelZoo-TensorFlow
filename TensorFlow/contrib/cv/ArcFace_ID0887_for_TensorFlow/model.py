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


import tensorflow as tf

slim = tf.contrib.slim

from backbones import modifiedResNet_v2, ResNet_v2


def get_embd(inputs, is_training_dropout, is_training_bn, config, reuse=False, scope='embd_extractor'):
    with tf.variable_scope(scope, reuse=reuse):
        net = inputs
        end_points = {}
        if config['backbone_type'].startswith('resnet_v2_m'):
            arg_sc = modifiedResNet_v2.resnet_arg_scope(weight_decay=config['weight_decay'],
                                                        batch_norm_decay=config['bn_decay'])
            with slim.arg_scope(arg_sc):
                if config['backbone_type'] == 'resnet_v2_m_50':
                    net, end_points = modifiedResNet_v2.resnet_v2_m_50(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_m_101':
                    net, end_points = modifiedResNet_v2.resnet_v2_m_101(net, is_training=is_training_bn,
                                                                        return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_m_100':
                    # print("begin call resnet_v2_m_100")
                    net, end_points = modifiedResNet_v2.resnet_v2_m_100(net, is_training=is_training_bn,
                                                                        return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_m_152':
                    net, end_points = modifiedResNet_v2.resnet_v2_m_152(net, is_training=is_training_bn,
                                                                        return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_m_200':
                    net, end_points = modifiedResNet_v2.resnet_v2_m_200(net, is_training=is_training_bn,
                                                                        return_raw=True)
                else:
                    raise ValueError('Invalid backbone type.')
        elif config['backbone_type'].startswith('resnet_v2'):
            arg_sc = ResNet_v2.resnet_arg_scope(weight_decay=config['weight_decay'],
                                                batch_norm_decay=config['bn_decay'])
            with slim.arg_scope(arg_sc):
                if config['backbone_type'] == 'resnet_v2_50':
                    net, end_points = ResNet_v2.resnet_v2_50(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_101':
                    net, end_points = ResNet_v2.resnet_v2_101(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_152':
                    net, end_points = ResNet_v2.resnet_v2_152(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_200':
                    net, end_points = ResNet_v2.resnet_v2_200(net, is_training=is_training_bn, return_raw=True)
        else:
            raise ValueError('Invalid backbone type.')

        if config['out_type'] == 'E':
            with slim.arg_scope(arg_sc):
                net = slim.batch_norm(net, activation_fn=None, is_training=is_training_bn)
                # net = slim.dropout(net, keep_prob=config['keep_prob'], is_training=is_training_dropout)
                net = slim.flatten(net)
                net = slim.fully_connected(net, config['embd_size'], normalizer_fn=None, activation_fn=None)
                net = slim.batch_norm(net, scale=False, activation_fn=None, is_training=is_training_bn)
                end_points['embds'] = net
        else:
            raise ValueError('Invalid out type.')

        return net, end_points
