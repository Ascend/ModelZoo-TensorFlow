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
# Mostly based on the code written by Tinghui Zhou: 
# https://github.com/tinghuiz/SfMLearner/blob/master/data/kitti/kitti_odom_loader.py
from __future__ import division
from npu_bridge.npu_init import *
import numpy as np
from glob import glob
import os
import scipy.misc

class kitti_odom_loader(object):
    def __init__(self,
                 dataset_dir,
                 img_height=128,
                 img_width=416,
                 seq_length=5):
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        self.train_seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.test_seqs = [9, 10]

        self.collect_test_frames()
        self.collect_train_frames()

    def collect_test_frames(self):
        self.test_frames = []
        for seq in self.test_seqs:
            seq_dir = os.path.join(self.dataset_dir, 'sequences', '%.2d' % seq)
            img_dir = os.path.join(seq_dir, 'image_0')
            N = len(glob(img_dir + '/*.png'))
            for n in range(N):
                self.test_frames.append('%.2d %.6d' % (seq, n))
        self.num_test = len(self.test_frames)
        
    def collect_train_frames(self):
        self.train_frames = []
        for seq in self.train_seqs:
            seq_dir = os.path.join(self.dataset_dir, 'sequences', '%.2d' % seq)
            img_dir = os.path.join(seq_dir, 'image_0')
            N = len(glob(img_dir + '/*.png'))
            for n in range(N):
                self.train_frames.append('%.2d %.6d' % (seq, n))
        self.num_train = len(self.train_frames)

    def is_valid_sample(self, frames, tgt_idx):
        N = len(frames)
        tgt_drive, _ = frames[tgt_idx].split(' ')
        half_offset = int((self.seq_length - 1)/2)
        min_src_idx = tgt_idx - half_offset
        max_src_idx = tgt_idx + half_offset
        if min_src_idx < 0 or max_src_idx >= N:
            return False
        min_src_drive, _ = frames[min_src_idx].split(' ')
        max_src_drive, _ = frames[max_src_idx].split(' ')
        if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
            return True
        return False

    def load_image_sequence(self, frames, tgt_idx, seq_length):
        half_offset = int((seq_length - 1)/2)
        image_seq = []
        for o in range(-half_offset, half_offset+1):
            curr_idx = tgt_idx + o
            curr_drive, curr_frame_id = frames[curr_idx].split(' ')
            curr_img = self.load_image(curr_drive, curr_frame_id)
            if o == 0:
                zoom_y = self.img_height/curr_img.shape[0]
                zoom_x = self.img_width/curr_img.shape[1]
            curr_img = scipy.misc.imresize(curr_img, (self.img_height, self.img_width))
            image_seq.append(curr_img)
        return image_seq, zoom_x, zoom_y

    def load_example(self, frames, tgt_idx):
        image_seq, zoom_x, zoom_y = self.load_image_sequence(frames, tgt_idx, self.seq_length)
        tgt_drive, tgt_frame_id = frames[tgt_idx].split(' ')
        intrinsics = self.load_intrinsics(tgt_drive, tgt_frame_id)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)        
        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['folder_name'] = tgt_drive
        example['file_name'] = tgt_frame_id
        return example

    def get_train_example_with_idx(self, tgt_idx):
        if not self.is_valid_sample(self.train_frames, tgt_idx):
            return False
        example = self.load_example(self.train_frames, tgt_idx)
        return example

    def load_image(self, drive, frame_id):
        img_file = os.path.join(self.dataset_dir, 'sequences', '%s/image_0/%s.png' % (drive, frame_id))
        img = scipy.misc.imread(img_file)
        return img

    def load_intrinsics(self, drive, frame_id):
        calib_file = os.path.join(self.dataset_dir, 'sequences', '%s/calib.txt' % drive)
        proj_c2p, _ = self.read_calib_file(calib_file)
        intrinsics = proj_c2p[:3, :3]
        return intrinsics

    def read_calib_file(self, filepath, cid=2):
        """Read in a calibration file and parse into a dictionary."""
        with open(filepath, 'r') as f:
            C = f.readlines()
        def parseLine(L, shape):
            data = L.split()
            data = np.array(data[1:]).reshape(shape).astype(np.float32)
            return data
        proj_c2p = parseLine(C[cid], shape=(3,4))
        proj_v2c = parseLine(C[-1], shape=(3,4))
        filler = np.array([0, 0, 0, 1]).reshape((1,4))
        proj_v2c = np.concatenate((proj_v2c, filler), axis=0)
        return proj_c2p, proj_v2c

    def scale_intrinsics(self,mat, sx, sy):
        out = np.copy(mat)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy
        return out

