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
import os
import math
import imageio
import scipy.misc

import tensorflow as tf
import numpy as np
from glob import glob
from geonet_model import *
from kitti_eval.pose_evaluation_utils import dump_pose_seq_TUM
from PIL import Image
def test_pose(opt):

    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    ##### init #####
    input_uint8 = tf.placeholder(tf.uint8, [opt.batch_size, 
        opt.img_height, opt.img_width, opt.seq_length * 3], 
        name='raw_input')
    tgt_image = input_uint8[:,:,:,:3]
    src_image_stack = input_uint8[:,:,:,3:]

    model = GeoNetModel(opt, tgt_image, src_image_stack, None)
    fetches = { "pose": model.pred_poses }

    saver = tf.train.Saver([var for var in tf.model_variables()])

    ##### load test frames #####
    seq_dir = os.path.join(opt.dataset_dir, 'sequences', '%.2d' % opt.pose_test_seq)
    img_dir = os.path.join(seq_dir, 'image_0')
    N = len(glob(img_dir + '/*.png'))
    test_frames = ['%.2d %.6d' % (opt.pose_test_seq, n) for n in range(N)]

    ##### load time file #####
    with open(opt.dataset_dir + 'sequences/%.2d/times.txt' % opt.pose_test_seq, 'r') as f:
        times = f.readlines()
    times = np.array([float(s[:-1]) for s in times])

    ##### Go! #####
    max_src_offset = (opt.seq_length - 1) // 2
    with tf.Session(config=npu_config_proto()) as sess:

        #saver.restore(sess, '/root/GeoNet/GeoNet-master/posenet2/model-85000')

        saver.restore(sess, opt.init_ckpt_file)

        for tgt_idx in range(max_src_offset, N-max_src_offset, opt.batch_size):            
            if (tgt_idx-max_src_offset) % 100 == 0:
                print('Progress: %d/%d' % (tgt_idx-max_src_offset, N))

            inputs = np.zeros((opt.batch_size, opt.img_height,
                     opt.img_width, 3*opt.seq_length), dtype=np.uint8)

            for b in range(opt.batch_size):
                idx = tgt_idx + b
                if idx >= N-max_src_offset:
                    break
                image_seq = load_image_sequence(opt.dataset_dir,
                                                test_frames,
                                                idx,
                                                opt.seq_length,
                                                opt.img_height,
                                                opt.img_width)
                inputs[b] = image_seq

            pred = sess.run(fetches, feed_dict={input_uint8: inputs})
            print(pred)
            #print(pred.shape)
            print("pred")
            pred_poses = pred['pose']
            print(pred_poses)
            print(pred_poses.shape)
            print("pre_posessss")#1*2*6
            # Insert the target pose [0, 0, 0, 0, 0, 0]
            pred_poses = np.insert(pred_poses, max_src_offset, np.zeros((1,6)), axis=1)
            print(pred_poses)
            print(pred_poses.shape)
            print("pred_poses")#1*3*6

            for b in range(opt.batch_size):
                idx = tgt_idx + b
                if idx >=N-max_src_offset:
                    break
                pred_pose = pred_poses[1]
                print(pred_pose)
                print(pred_pose.shape)
                print("pred_pose")
                curr_times = times[idx - max_src_offset:idx + max_src_offset + 1]
                out_file = opt.output_dir + '%.6d.txt' % (idx - max_src_offset)
                dump_pose_seq_TUM(out_file, pred_pose, curr_times)

def load_image_sequence(dataset_dir, 
                        frames, 
                        tgt_idx, 
                        seq_length, 
                        img_height, 
                        img_width):
    half_offset = int((seq_length - 1)/2)
    for o in range(-half_offset, half_offset+1):
        curr_idx = tgt_idx + o
        curr_drive, curr_frame_id = frames[curr_idx].split(' ')
        img_file = os.path.join(
            dataset_dir, 'sequences', '%s/image_0/%s.png' % (curr_drive, curr_frame_id))
        curr_img = imageio.imread(img_file)

        #curr_img = scipy.misc.imresize(curr_img, (img_height, img_width))
        curr_img = np.array(Image.fromarray(curr_img).resize((img_width,img_height)))
        if o == -half_offset:
            image_seq = curr_img
        elif o == 0:
            image_seq = np.dstack((curr_img, image_seq))
        else:
            image_seq = np.dstack((image_seq, curr_img))
    return image_seq

