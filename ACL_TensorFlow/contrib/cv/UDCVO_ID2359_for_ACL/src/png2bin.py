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

import os, sys, argparse
import numpy as np
import tensorflow as tf
import data_utils
from dataloader import DataLoader

parser = argparse.ArgumentParser()

# Input paths
parser.add_argument('--image_path',
                    type=str, required=True, help='Path to list of image paths')
parser.add_argument('--interp_depth_path',
                    type=str, required=True, help='Path to list of interpolated depth paths')
parser.add_argument('--validity_map_path',
                    type=str, required=True, help='Path to list of validity map paths')
parser.add_argument('--ground_truth_path',
                    type=str, default='', help='Path to list of ground truth paths')
parser.add_argument('--start_idx',
                    type=int, default=0, help='Start index of the list of paths to evaluate')
parser.add_argument('--end_idx',
                    type=int, default=800, help='Last index of the list of paths to evaluate')
# Batch parameters
parser.add_argument('--n_batch',
                    type=int, default=8, help='Number of samples per batch')
parser.add_argument('--n_height',
                    type=int, default=480, help='Height of of sample')
parser.add_argument('--n_width',
                    type=int, default=640, help='Width of each sample')
# Output options
parser.add_argument('--output_path',
                    type=str, default='output', help='Directory to store output')
# Hardware settings
parser.add_argument('--n_thread',
                    type=int, default=4, help='Number of threads for fetching')

args = parser.parse_args()

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

# Load image paths from file for evaluation
im_paths = sorted(data_utils.read_paths(args.image_path))[args.start_idx:args.end_idx]
iz_paths = sorted(data_utils.read_paths(args.interp_depth_path))[args.start_idx:args.end_idx]
vm_paths = sorted(data_utils.read_paths(args.validity_map_path))[args.start_idx:args.end_idx]
assert (len(im_paths) == len(iz_paths))
assert (len(im_paths) == len(vm_paths))
n_sample = len(im_paths)
nn_sample = int(n_sample/args.n_batch)

# Pad all paths based on batch
im_paths = data_utils.pad_batch(im_paths, args.n_batch)
iz_paths = data_utils.pad_batch(iz_paths, args.n_batch)
vm_paths = data_utils.pad_batch(vm_paths, args.n_batch)
n_step = len(im_paths) // args.n_batch

with tf.Graph().as_default():
    # Initialize dataloader
    dataloader = DataLoader(shape=[args.n_batch, args.n_height, args.n_width, 3],
                            name='dataloader',
                            is_training=False,
                            n_thread=args.n_thread,
                            prefetch_size=2 * args.n_thread)
    # Fetch the input from dataloader
    im0 = dataloader.next_element[0]
    sz0 = dataloader.next_element[3]

    # Initialize Tensorflow session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    # Load image, dense depth, sparse depth
    dataloader.initialize(session,
                          image_paths=im_paths,
                          interp_depth_paths=iz_paths,
                          validity_map_paths=vm_paths)

    im0_arr = np.zeros([n_step * args.n_batch, args.n_height, args.n_width, 3]).astype(np.float32)
    sz0_arr = np.zeros([n_step * args.n_batch, args.n_height, args.n_width, 2]).astype(np.float32)
    step = 0
    while True:
        try:
            sys.stdout.write(
                'Processed {}/{} examples \r'.format(step * args.n_batch, n_sample))
            sys.stdout.flush()

            batch_start = step * args.n_batch
            batch_end = step * args.n_batch + args.n_batch
            step += 1
            [im0_arr[batch_start:batch_end, ...], sz0_arr[batch_start:batch_end, ...]] = session.run([im0, sz0])
        except tf.errors.OutOfRangeError:
            break

    # Store png as binfile
    if args.output_path:
        output_dirpath_im0 = os.path.join(args.output_path, 'im0')
        output_dirpath_sz0 = os.path.join(args.output_path, 'sz0')
        print('Storing PNG as BIN into {}'.format(output_dirpath_im0))
        print('Storing PNG as BIN into {}'.format(output_dirpath_sz0))

        if not os.path.exists(output_dirpath_im0):
            os.makedirs(output_dirpath_im0)
        if not os.path.exists(output_dirpath_sz0):
            os.makedirs(output_dirpath_sz0)

        for idx in range(nn_sample):
            batch_start = idx * args.n_batch
            batch_end = idx * args.n_batch + args.n_batch
            im0 = im0_arr[batch_start:batch_end, ...]
            sz0 = sz0_arr[batch_start:batch_end, ...]
            im0.tofile(output_dirpath_im0 + "/" + str(idx) + ".bin")
            sz0.tofile(output_dirpath_sz0 + "/" + str(idx) + ".bin")
