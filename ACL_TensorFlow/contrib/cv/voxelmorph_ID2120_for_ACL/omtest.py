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
# python imports
import os
import sys
from argparse import ArgumentParser
# third-party imports
import tensorflow as tf
import numpy as np
import nibabel as nib
# project imports
import datagenerators
import networks
sys.path.append('./ext/neuron')
sys.path.append('./ext/medipy-lib')
from medipy.metrics import dice
# npu import
from npu_bridge.npu_init import *

def omtest(data_path, test_num, om_outbin_dir):
    # some parameters need to be make manully
    good_labels = np.array([0, 2, 3, 4, 7, 8, 10, 11, 13, 14, 15, 16, 17, 24, 28, 41, 42, 43, 46,
                            47, 49, 50, 53, 54, 60, 251, 252, 253, 254, 255])

    # load tgt image from provided files. 160x192x224.
    atlas_vol = nib.load(os.path.join(data_path, 'atlas_abide_brain_crop.nii.gz')).dataobj[
        np.newaxis, ..., np.newaxis].astype('float32')
    atlas_seg = np.array(nib.load(os.path.join(data_path, 'atlas_abide_seg_crop.nii.gz')).dataobj).astype('float32')
    vol_size = atlas_vol.shape[1:-1]

    # vector of dice values for all the good labels
    dice_vals = np.zeros((len(good_labels), 1))

    # find all the test files
    test_path = os.path.join(data_path, 'test/')
    seg_path = os.path.join(data_path, 'seg_affined/')
    file_names = os.listdir(test_path)

    # NN transfer model
    nn_trf_model = networks.nn_trf(vol_size, indexing='ij')

    # npu config
    config = tf.ConfigProto()
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭

    with tf.Session(config=config) as sess:
        # init
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        sess.run(init_op)

        # load the src image 
        vol_name = test_path + file_names[test_num]
        seg_name = seg_path + file_names[test_num].replace('brain', 'seg')
        _, X_seg = datagenerators.load_example_by_name(vol_name, seg_name)
        
        # load the om output flow tensor
        pred_flow = np.fromfile(om_outbin_dir, dtype=np.float32)[:20643840].reshape((1, 160, 192, 224, 3))

        # register and transfer the src to tgt
        warp_seg = nn_trf_model.predict([X_seg, pred_flow])[0, ..., 0]

        # calculate the dice coefficients
        dice_vals = dice(warp_seg, atlas_seg, labels=good_labels)

        # print
        print('%5.3f' % np.mean(dice_vals))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_path", type=str,
                        dest="data_path", default='../Dataset-ABIDE/')
    parser.add_argument("--test_num", type=int,
                        dest="test_num", default=2)
    parser.add_argument("--om_outbin_dir", type=str,
                        dest="om_outbin_dir", default='./output/2022921_13_56_7_916021/vm_output_1.bin')       
                 
    args = parser.parse_args()

    omtest(args.data_path, args.test_num, args.om_outbin_dir)




