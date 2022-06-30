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
"""Predicting 3d poses from 2d joints"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import src.viz as viz
import src.cameras as cameras
import src.data_utils as data_utils

def sample():
    # """Get samples from data to bin"""
    actions = data_utils.define_actions("All")

    # Load camera parameters
    SUBJECT_IDS = [1, 5, 6, 7, 8, 9, 11]
    rcams = cameras.load_cameras("../data/h36m/cameras.h5", SUBJECT_IDS)
    n_joints = 17 if not False else 14

    # Load 3d data and load (or create) 2d projections
    train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
        actions, "../data/h36m/", True, rcams, False)

    if True:
        train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(
            actions, "../data/h36m/")
    else:
        train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d, _ = data_utils.create_2d_data(
            actions, "../data/h36m/", rcams)

    batch_size = 1

    for key2d in test_set_2d.keys():

        (subj, b, fname) = key2d

        # choose SittingDown action to visualize
        if b == 'SittingDown':
            print("Subject: {}, action: {}, fname: {}".format(subj, b, fname))

            # keys should be the same if 3d is in camera coordinates
            key3d = key2d if True else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
            key3d = (subj, b, fname[:-3]) if (fname.endswith('-sh')) and True else key3d

            enc_in = test_set_2d[key2d]
            n2d, _ = enc_in.shape
            dec_out = test_set_3d[key3d]
            n3d, _ = dec_out.shape
            assert n2d == n3d

            # Split into about-same-size batches

            enc_in = np.array_split(enc_in, n2d // batch_size)
            dec_out = np.array_split(dec_out, n3d // batch_size)

            # store all pose hypotheses in a list
            number = 1
            for bidx in range(len(enc_in)):
                (subj2, b2, fname2) = key3d
                np.array(enc_in[bidx],dtype=np.float32).tofile("../data/2022530_13_36_24_832714/enc_in/" + str(subj2)+"_"+b2+"_"+fname2+"_"+ str(number) + ".bin")
                np.array(dec_out[bidx],dtype=np.float32).tofile("../data/2022530_13_36_24_832714/dec_out/" + str(subj2)+"_"+b2+"_"+fname2+"_"+ str(number) + ".bin")
                number += 1

sample()