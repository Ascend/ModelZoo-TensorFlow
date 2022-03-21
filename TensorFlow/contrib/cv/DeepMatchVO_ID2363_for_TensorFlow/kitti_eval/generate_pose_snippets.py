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
from __future__ import division
from npu_bridge.npu_init import *
import os, sys
import math
import scipy.misc
import numpy as np
import argparse
from glob import glob
from pose_evaluation_utils import mat2euler, dump_pose_seq_TUM

CURDIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(CURDIR, '.')))
sys.path.append(os.path.abspath(os.path.join(CURDIR, '..')))
sys.path.append(os.path.abspath(os.path.join(CURDIR, '...')))
from common_utils import is_valid_sample

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help="path to kitti odometry dataset")
parser.add_argument("--output_dir",  type=str, help="path to output pose snippets")
parser.add_argument("--seq_id",      type=int, default=9, help="sequence id to generate groundtruth pose snippets")
parser.add_argument("--seq_length",  type=int, default=5, help="sequence length of pose snippets")
args = parser.parse_args()

def main():
    pose_gt_dir = args.dataset_dir + 'poses/'
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    seq_dir = os.path.join(args.dataset_dir, 'sequences', '%.2d' % args.seq_id)
    img_dir = os.path.join(seq_dir, 'image_2')
    N = len(glob(img_dir + '/*.png'))
    test_frames = ['%.2d %.6d' % (args.seq_id, n) for n in range(N)]
    with open(args.dataset_dir + 'sequences/%.2d/times.txt' % args.seq_id, 'r') as f:
        times = f.readlines()
    times = np.array([float(s[:-1]) for s in times])

    with open(pose_gt_dir + '%.2d.txt' % args.seq_id, 'r') as f:
        poses = f.readlines()
    poses_gt = []
    for pose in poses:
        pose = np.array([float(s) for s in pose[:-1].split(' ')]).reshape((3,4))
        rot = np.linalg.inv(pose[:,:3])
        tran = -np.dot(rot, pose[:,3].transpose())
        rz, ry, rx = mat2euler(rot)
        poses_gt.append(tran.tolist() + [rx, ry, rz])
    poses_gt = np.array(poses_gt)

    max_src_offset = (args.seq_length - 1)//2
    for tgt_idx in range(N):
        if not is_valid_sample(test_frames, tgt_idx, args.seq_length):
            continue
        if tgt_idx % 100 == 0:
            print('Progress: %d/%d' % (tgt_idx, N))
        pred_poses = poses_gt[tgt_idx - max_src_offset:tgt_idx + max_src_offset + 1]
        curr_times = times[tgt_idx - max_src_offset:tgt_idx + max_src_offset + 1]
        out_file = os.path.join(args.output_dir, '%.6d.txt' % (tgt_idx - max_src_offset))
        dump_pose_seq_TUM(out_file, pred_poses, curr_times, is_kitti_format=True)


if __name__ == '__main__':
    main()
