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

# -*- coding: UTF-8 -*-
"""
train atlas-based alignment with CVPR2018 version of VoxelMorph
"""

# python imports
import os
import sys
from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import numpy as np
import tensorflow.python.keras as keras
import nibabel as nib

# project imports
import datagenerators
import networks
import losses

sys.path.append('../ext/neuron')
sys.path.append('../ext/medipy')
import neuron.callbacks as nrn_gen
from medipy.metrics import dice

from npu_bridge.npu_init import *


def test_zyh(data_path,
          label,
          load_model_file,
          ):
    """
    model training function
    :param data_dir: folder with npz files for each subject.
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param model_dir: the model directory to save to
    :param gpu_id: integer specifying the gpu to use
    :param lr: learning rate
    :param n_iterations: number of training iterations
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param steps_per_epoch: frequency with which to save models
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param load_model_file: optional h5 model file to initialize with
    :param data_loss: data_loss: 'mse' or 'ncc
    """

    # load atlas from provided files. The atlas we used is 160x192x224.
    atlas_vol = nib.load(os.path.join(data_path, 'atlas_abide_brain_crop.nii.gz')).dataobj[
        np.newaxis, ..., np.newaxis]
    atlas_seg = nib.load(os.path.join(data_path, 'atlas_abide_seg_crop.nii.gz')).dataobj
    vol_size = atlas_vol.shape[1:-1]
    test_path = os.path.join(data_path, 'test/')
    seg_path = os.path.join(data_path, 'seg_affined/')

    # Anatomical labels we want to evaluate
    good_labels = np.array(label)

    # UNET filters for voxelmorph-1 and voxelmorph-2,
    # these are architectures presented in CVPR 2018
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 32, 16, 16]

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("**********")

    # prepare the model
    batch_size = 1
    src = tf.placeholder(dtype=tf.float32, shape=[batch_size, 160, 192, 224, 1])
    tgt = tf.placeholder(dtype=tf.float32, shape=[batch_size, 160, 192, 224, 1])
    y, flow = networks.cvpr2018_net(vol_size, nf_enc, nf_dec, src, tgt)  # vol_size, enc_nf, dec_nf, src, tgt
    
    # NN transfer model
    nn_trf_model = networks.nn_trf(vol_size, indexing='ij')

    # npu config
    config = tf.ConfigProto()
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭

    # 获取当前路径下的文件名，返回List
    file_names = os.listdir(test_path)
    n_batches = len(file_names)
    # prepare a matrix of dice values
    dice_vals = np.zeros((len(good_labels), n_batches))

    with tf.Session(config=config) as sess:
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        sess.run(init_op)

        # saver is used to save the model
        saver = tf.train.Saver()
        print('loading', load_model_file)
        saver.restore(sess, tf.train.latest_checkpoint(load_model_file))

        for k in range(n_batches):
            vol_name = test_path + file_names[k]
            seg_name = seg_path + file_names[k].replace('brain', 'seg')
            # load subject test
            X_vol, X_seg = datagenerators.load_example_by_name(vol_name, seg_name)

            pred_flow = sess.run(flow, feed_dict={src: X_vol, tgt: atlas_vol})

            warp_seg = nn_trf_model.predict([X_seg, pred_flow])[0, ..., 0]
            # OrthoSlicer3D(warp_seg).show()

            dice_vals[:, k] = dice(warp_seg, atlas_seg, labels=good_labels)
            print('%3d %5.3f %5.3f' % (k, np.mean(dice_vals[:, k]), np.mean(np.mean(dice_vals[:, :k + 1]))))

    return np.mean(dice_vals[:]), np.std(dice_vals[:])




if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("data_path", type=str,
                        help="data folder")
    parser.add_argument("--model_path", type=str, dest="model_path",
                        default='../models/')

    args = parser.parse_args()

    dice_all = []
    dice_std_all = []
    model_all = []

    good_labels = np.array([0, 2, 3, 4, 7, 8, 10, 11, 13, 14, 15, 16, 17, 24, 28, 41, 42, 43, 46,
                            47, 49, 50, 53, 54, 60, 251, 252, 253, 254, 255])

    dice_iter, dice_std_iter = test_zyh(args.data_path,  good_labels, args.model_path)
    print('average dice score is %5.3f(%5.3f)' % (dice_iter, dice_std_iter))

