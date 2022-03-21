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

#wty
#change the data to bin
from __future__ import division
from npu_bridge.npu_init import *
import os
import math
import scipy.misc
import tensorflow as tf
import numpy as np
from glob import glob
from deep_slam import DeepSlam
from data_loader import DataLoader
from kitti_eval.pose_evaluation_utils import dump_pose_seq_TUM
from common_utils import complete_batch_size, is_valid_sample

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_integer("seq_length", 5, "Sequence length for each example")
flags.DEFINE_integer("test_seq", 9, "Sequence id to test")
flags.DEFINE_string("dataset_dir", None, "Raw odometry dataset directory")
flags.DEFINE_string("concat_img_dir", None, "Preprocess image dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
flags.DEFINE_string("bin_dir", None, "bin_dir file")
FLAGS = flags.FLAGS

def load_kitti_image_sequence_names(dataset_dir, frames, seq_length):
    image_sequence_names = []
    target_inds = []
    frame_num = len(frames)
    for tgt_idx in range(frame_num):
        if not is_valid_sample(frames, tgt_idx, FLAGS.seq_length):
            continue
        curr_drive, curr_frame_id = frames[tgt_idx].split(' ')
        img_filename = os.path.join(dataset_dir, '%s/%s.jpg' % (curr_drive, curr_frame_id))
        image_sequence_names.append(img_filename)
        target_inds.append(tgt_idx)
    return image_sequence_names, target_inds

def wty_preprocess_image(image):
    # Assuming input image is uint8
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image * 2. -1.

def wty_select_tensor_or_placeholder_input(input_uint8):
    input_mc = wty_preprocess_image(input_uint8)
    return input_mc


def wty_batch_unpack_image_sequence(image_seq, img_height, img_width, num_source):
    # Assuming the center image is the target frame
    tgt_start_idx = int(img_width * (num_source // 2))
    tgt_image = tf.slice(image_seq,
                         [0, 0, tgt_start_idx, 0],
                         [-1, -1, img_width, -1])
    # Source frames before the target frame
    src_image_1 = tf.slice(image_seq,
                           [0, 0, 0, 0],
                           [-1, -1, int(img_width * (num_source // 2)), -1])
    # Source frames after the target frame
    src_image_2 = tf.slice(image_seq,
                           [0, 0, int(tgt_start_idx + img_width), 0],
                           [-1, -1, int(img_width * (num_source // 2)), -1])
    src_image_seq = tf.concat([src_image_1, src_image_2], axis=2)
    # Stack source frames along the color channels (i.e. [B, H, W, N*3])
    src_image_stack = tf.concat([tf.slice(src_image_seq,
                                          [0, 0, i * img_width, 0],
                                          [-1, -1, img_width, -1])
                                 for i in range(num_source)], axis=3)
    return tgt_image, src_image_stack

def main():
    # get input images
    #if not os.path.isdir(FLAGS.output_dir):
        #os.makedirs(FLAGS.output_dir)
    concat_img_dir = os.path.join(FLAGS.concat_img_dir, '%.2d' % FLAGS.test_seq)
    max_src_offset = int((FLAGS.seq_length - 1)/2)  #1
    N = len(glob(concat_img_dir + '/*.jpg')) + 2*max_src_offset
    test_frames = ['%.2d %.6d' % (FLAGS.test_seq, n) for n in range(N)]

    with open(FLAGS.dataset_dir + 'sequences/%.2d/times.txt' % FLAGS.test_seq, 'r') as f:
        times = f.readlines()
    times = np.array([float(s[:-1]) for s in times])

    with tf.Session(config=npu_config_proto()) as sess:
        # setup input tensor
        loader = DataLoader(FLAGS.concat_img_dir, FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.seq_length-1)
        image_sequence_names, tgt_inds = load_kitti_image_sequence_names(FLAGS.concat_img_dir, test_frames, FLAGS.seq_length)
        image_sequence_names = complete_batch_size(image_sequence_names, FLAGS.batch_size)
        #print("image_sequence_names~~~~~~~~~~~~~~~~~~~~~")
        #print(image_sequence_names)
        tgt_inds = complete_batch_size(tgt_inds, FLAGS.batch_size)
        assert len(tgt_inds) == len(image_sequence_names)
        #print("lenth________________________")
        #print(len(tgt_inds))
        #print(tgt_inds)
        batch_sample = loader.load_test_batch(image_sequence_names)
        print("batch_sample~~~~~~~~~~~~~~~~~~~~~")
        print(batch_sample)
        sess.run(batch_sample.initializer)
        input_batch = batch_sample.get_next()
        
        input_batch.set_shape([FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width * FLAGS.seq_length, 3])
        
        for i in range(len(tgt_inds)//4):
          print(i)
          #sess.run(input_batch)
          input_mc = wty_select_tensor_or_placeholder_input(input_batch)
          tgt_image, src_image_stack = wty_batch_unpack_image_sequence(input_mc, 128, 416, 2)
          
          [tgt_image_numpy,src_image_stack_numpy]=sess.run([tgt_image,src_image_stack])
          #src_image_stack_numpy = sess.run(src_image_stack)
          
          with open(FLAGS.bin_dir + 'tgt/'+ '%06d'%(i) + '.bin', "wb") as f:
            f.write(tgt_image_numpy)
          with open(FLAGS.bin_dir + 'src/' + '%06d'%(i) + '.bin', "wb") as f:
            f.write(src_image_stack_numpy)
        '''
        print("input_batch~~~~~~~~~~~~~~~~~~~~~")
        print(input_batch)
        sess.run(input_batch)
        input_mc = wty_select_tensor_or_placeholder_input(input_batch)
        tgt_image, src_image_stack = wty_batch_unpack_image_sequence(input_mc, 128, 416, 2)
        print("tgt_image~~~~~~~~~~~~~~~~~~~~~")
        print(tgt_image)
        print("src_image_stack~~~~~~~~~~~~~~~~~~~~~")
        print(src_image_stack)
        
        tgt_image_numpy = sess.run(tgt_image)
        batch_cnt = 0
        with open('/home/DeepMatchVo_ID2363_for_TensorFlow/bin/tgt/' + '%06d'%(batch_cnt) + '.bin', "wb") as f:
            f.write(tgt_image_numpy)
        print("tgt_image_numpy~~~~~~~~~~~~~~~~~~~~~")
        print(tgt_image_numpy.shape)
        '''







if __name__ == '__main__':
    main()

