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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial


import pylib
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

import models


def boolean(s):
    return s.lower() in ('true', 't', 'yes', 'y', '1')


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str,
                    default='/home/test_user02/yuxh/STGAN_NPU/output/1024/checkpoints/Epoch_(69)_(473of473).ckpt',
                    help='the path of model(.ckpt)')
parser.add_argument('--pb_path', type=str, default='/home/test_user02/yuxh/STGAN_NPU/output/1024/pb_model',
                    help='the path of model(.pb)')
# model
att_default = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
               'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
parser.add_argument('--atts', default=att_default, nargs='+', help='Attributes to modify by the model')
parser.add_argument('--img_size', type=int, default=128, help='input image size')
parser.add_argument('--shortcut_layers', type=int, default=4,
                    help='# of skip connections between the encoder and the decoder')
parser.add_argument('--inject_layers', type=int, default=4,
                    help='# of attribute vectors applied in the decoder')
parser.add_argument('--enc_dim', type=int, default=64)
parser.add_argument('--dec_dim', type=int, default=64)
parser.add_argument('--enc_layers', type=int, default=5)
parser.add_argument('--dec_layers', type=int, default=5)
# STGAN & STU
parser.add_argument('--use_stu', type=boolean, default=True)
parser.add_argument('--stu_dim', type=int, default=64)
parser.add_argument('--stu_layers', type=int, default=4)
parser.add_argument('--stu_inject_layers', type=int, default=4)
parser.add_argument('--stu_kernel_size', type=int, default=3)
parser.add_argument('--stu_norm', type=str, default='none', choices=['none', 'bn', 'in'])
parser.add_argument('--stu_state', type=str, default='stu', choices=['stu', 'gru', 'direct'],
                    help='gru: gru arch.; stu: stu arch.; direct: directly pass the inner state to the outer layer')
parser.add_argument('--multi_inputs', type=int, default=1,
                    help='# of hierachical inputs (in the first several encoder layers')
parser.add_argument('--rec_loss_weight', type=float, default=100.0)
parser.add_argument('--one_more_conv', type=int, default=0, choices=[0, 1, 3],
                    help='0: no further conv after the decoder; 1: conv(k=1); 3: conv(k=3)')

args = parser.parse_args()

ckpt_path = args.ckpt_path
pb_path = args.pb_path
pylib.mkdir(pb_path)
# model
atts = args.atts
n_att = len(atts)
img_size = args.img_size
shortcut_layers = args.shortcut_layers
inject_layers = args.inject_layers
enc_dim = args.enc_dim
dec_dim = args.dec_dim
enc_layers = args.enc_layers
dec_layers = args.dec_layers
# STU
use_stu = args.use_stu
stu_dim = args.stu_dim
stu_layers = args.stu_layers
stu_inject_layers = args.stu_inject_layers
stu_kernel_size = args.stu_kernel_size
stu_norm = args.stu_norm
stu_state = args.stu_state
multi_inputs = args.multi_inputs
rec_loss_weight = float(args.rec_loss_weight)
one_more_conv = args.one_more_conv


def main():
    tf.reset_default_graph()

    # models
    Genc = partial(models.Genc, dim=enc_dim, n_layers=enc_layers, multi_inputs=multi_inputs)
    Gdec = partial(models.Gdec, dim=dec_dim, n_layers=dec_layers, shortcut_layers=shortcut_layers,
                   inject_layers=inject_layers, one_more_conv=one_more_conv)
    Gstu = partial(models.Gstu, dim=stu_dim, n_layers=stu_layers, inject_layers=stu_inject_layers,
                   kernel_size=stu_kernel_size, norm=stu_norm, pass_state=stu_state)

    # inputs
    xa_sample = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='input1')
    _b_sample = tf.placeholder(tf.float32, shape=[None, n_att], name='input2')
    raw_b_sample = tf.placeholder(tf.float32, shape=[None, n_att], name='input3')

    # sample
    test_label = _b_sample - raw_b_sample
    if use_stu:
        x_sample = Gdec(Gstu(Genc(xa_sample, is_training=False),
                             test_label, is_training=False), test_label, is_training=False)
    else:
        x_sample = Gdec(Genc(xa_sample, is_training=False), test_label, is_training=False)

    x_sample = tf.identity(x_sample, name='output')
    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, pb_path, 'model.pb')
        freeze_graph.freeze_graph(
            input_graph=pb_path + '/model.pb',
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,
            output_node_names='output',
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=pb_path + '/STGAN.pb',
            clear_devices=False,
            initializer_nodes='')
    print("done")


if __name__ == '__main__':
    main()
