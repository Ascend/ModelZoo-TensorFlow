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

def post(filename):
    path = "../experiments/test_git/samples_sh"
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
    b  = filename.split("_")
    key3d = (int(b[0]),b[1],b[2])
    pose_3d_mdm = [[], [], [], [], []]
    out_all_components = np.fromfile("../data/2022530_13_36_24_832714/out_mean/"+filename+"_output_0.bin", dtype=np.float32)

    # denormalize the input 2d pose, ground truth 3d pose as well as 3d pose hypotheses from mdm
    out_all_components = np.reshape(out_all_components, [-1, 48 + 2, 5])
    out_mean = out_all_components[:, : 48, :]

    enc_in = np.fromfile("../data/2022530_13_36_24_832714/enc_in/"+filename+".bin",dtype=np.float32)
    enc_in = enc_in.reshape(-1, 32)

    dec_out = np.fromfile("../data/2022530_13_36_24_832714/dec_out/"+filename+".bin",dtype=np.float32)
    dec_out = dec_out.reshape(-1, 48)

    enc_in = data_utils.unNormalizeData(enc_in, data_mean_2d, data_std_2d, dim_to_ignore_2d)
    dec_out = data_utils.unNormalizeData(dec_out, data_mean_3d, data_std_3d, dim_to_ignore_3d)

    poses3d = np.zeros((out_mean.shape[0], 96, out_mean.shape[-1]))
    for j in range(out_mean.shape[-1]):
        poses3d[:, :, j] = data_utils.unNormalizeData(out_mean[:, :, j], data_mean_3d, data_std_3d, dim_to_ignore_3d)

    # extract the 17 joints
    [pose_3d_mdm[i].append(poses3d[:, :, i]) for i in range(poses3d.shape[-1])]
    # Put all the poses together

    enc_in, dec_out = map(np.vstack, [enc_in, dec_out])
    for i in range(poses3d.shape[-1]):
        pose_3d_mdm[i] = np.vstack(pose_3d_mdm[i])

    # Convert back to world coordinates
    if True:
        N_CAMERAS = 4
        N_JOINTS_H36M = 32

        # Add global position back
        dec_out = dec_out + np.tile(test_root_positions[key3d], [1, N_JOINTS_H36M])
        for i in range(poses3d.shape[-1]):
            pose_3d_mdm[i] = pose_3d_mdm[i] + np.tile(test_root_positions[key3d], [1, N_JOINTS_H36M])

        # Load the appropriate camera
        subj, action, sname = key3d

        cname = sname.split('.')[1]  # <-- camera name
        scams = {(subj, c + 1): rcams[(subj, c + 1)] for c in range(N_CAMERAS)}  # cams of this subject
        scam_idx = [scams[(subj, c + 1)][-1] for c in range(N_CAMERAS)].index(cname)  # index of camera used
        the_cam = scams[(subj, scam_idx + 1)]  # <-- the camera used
        R, T, f, c, k, p, name = the_cam
        assert name == cname


        def cam2world_centered(data_3d_camframe):
            data_3d_worldframe = cameras.camera_to_world_frame(data_3d_camframe.reshape((-1, 3)), R, T)
            data_3d_worldframe = data_3d_worldframe.reshape((-1, N_JOINTS_H36M * 3))
            # subtract root translation
            return data_3d_worldframe - np.tile(data_3d_worldframe[:, :3], (1, N_JOINTS_H36M))


        # Apply inverse rotation and translation
        dec_out = cam2world_centered(dec_out)
        for i in range(poses3d.shape[-1]):
            pose_3d_mdm[i] = cam2world_centered(pose_3d_mdm[i])

    # sample some results to visualize
    np.random.seed(42)
    idx = np.random.permutation(enc_in.shape[0])
    # idx = 0
    enc_in, dec_out = enc_in[idx, :], dec_out[idx, :]
    for i in range(poses3d.shape[-1]):
        pose_3d_mdm[i] = pose_3d_mdm[i][idx, :]
    print(enc_in.shape,dec_out.shape)
    exidx = 0
    nsamples = 1

    for i in np.arange(nsamples):
        fig = plt.figure(figsize=(20, 5))

        subplot_idx = 1
        gs1 = gridspec.GridSpec(1, 7)  # 5 rows, 9 columns
        gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
        plt.axis('off')

        # Plot 2d pose
        ax1 = plt.subplot(gs1[subplot_idx - 1])
        p2d = enc_in[exidx, :]
        viz.show2Dpose(p2d, ax1)
        ax1.invert_yaxis()

        # Plot 3d gt
        ax2 = plt.subplot(gs1[subplot_idx], projection='3d')
        p3d = dec_out[exidx, :]
        viz.show3Dpose(p3d, ax2)

        # Plot 3d pose hypotheses
        # print(poses3d.shape, exidx, pose_3d_mdm,poses3d)
        for i in range(poses3d.shape[-1]):

            ax3 = plt.subplot(gs1[subplot_idx + i + 1], projection='3d')
            p3d = pose_3d_mdm[i][exidx]
            viz.show3Dpose(p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71")
        # plt.show()
        plt.savefig('{}/sample_{}_{}_{}_{}.png'.format(path, subj, action, scam_idx, exidx))
        plt.close(fig)
        exidx = exidx + 1

# sample()
filename = "11_SittingDown_SittingDown 1.54138969.h5_1048"
post(filename)