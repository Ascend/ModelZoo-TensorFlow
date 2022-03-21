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
import data_utils, eval_utils
from data_utils import log


parser = argparse.ArgumentParser()

# File path
parser.add_argument('--restore_path',
    type=str, required=True, help='Path to restore model')
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
parser.add_argument('--min_evaluate_z',
    type=float, default=0.2, help='Minimum depth to evaluate')
parser.add_argument('--max_evaluate_z',
    type=float, default=0.5, help='Maximum depth to evaluate')
# Output options
parser.add_argument('--output_path',
    type=str, default='output', help='Directory to store output')

args = parser.parse_args()

log_path = os.path.join(args.output_path, 'results.txt')
if not os.path.exists(args.output_path):
  os.makedirs(args.output_path)

n_sample = args.end_idx
n_step = args.end_idx//args.n_batch

if args.ground_truth_path != '':
  gt_paths = sorted(data_utils.read_paths(args.ground_truth_path))[args.start_idx:args.end_idx]
  assert(len(gt_paths) == args.end_idx)

gt_arr = []
if args.ground_truth_path != '':
  # Load ground truth
  for idx in range(n_sample):
    sys.stdout.write(
        'Loading {}/{} groundtruth depth maps \r'.format(idx+1, n_sample))
    sys.stdout.flush()

    gt, vm = data_utils.load_depth_with_validity_map(gt_paths[idx])
    gt = np.concatenate([np.expand_dims(gt, axis=-1), np.expand_dims(vm, axis=-1)], axis=-1)
    gt_arr.append(gt)

  print('Completed loading {} groundtruth depth maps'.format(n_sample))

z_arr = np.zeros([n_step*args.n_batch, args.n_height, args.n_width, 1])
step = 0
files = os.listdir(args.restore_path)
files.sort(key=lambda x: int(x.split('_')[0]))
for file in files:
    if file.endswith(".bin"):
        batch_start = step*args.n_batch
        batch_end = step*args.n_batch+args.n_batch
        step += 1
        z_arr[batch_start:batch_end, ...] = np.fromfile(args.restore_path +'/'+file, dtype='float32').reshape(8,480,640,1)

# Run evaluation
if len(gt_arr) > 0:
    mae   = np.zeros(n_sample, np.float32)
    rmse  = np.zeros(n_sample, np.float32)
    imae  = np.zeros(n_sample, np.float32)
    irmse = np.zeros(n_sample, np.float32)

    for idx in range(n_sample):
      z = np.squeeze(z_arr[idx, ...])
      gt = np.squeeze(gt_arr[idx][..., 0])
      vm = np.squeeze(gt_arr[idx][..., 1])

      # Create mask for evaluation
      valid_mask = np.where(vm > 0, 1, 0)
      min_max_mask = np.logical_and(gt > args.min_evaluate_z, gt < args.max_evaluate_z)
      mask = np.where(np.logical_and(valid_mask, min_max_mask) > 0)
      z = z[mask]
      gt = gt[mask]

      # Run evaluations: MAE, RMSE in meters, iMAE, iRMSE in 1/kilometers
      mae[idx] = eval_utils.mean_abs_err(1000.0*z, 1000.0*gt)
      rmse[idx] = eval_utils.root_mean_sq_err(1000.0*z, 1000.0*gt)
      imae[idx] = eval_utils.inv_mean_abs_err(0.001*z, 0.001*gt)
      irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001*z, 0.001*gt)

    # Compute mean error
    mae   = np.mean(mae)
    rmse  = np.mean(rmse)
    imae  = np.mean(imae)
    irmse = np.mean(irmse)
    log('{:>10} {:>10} {:>10} {:>10}'.format('MAE', 'RMSE', 'iMAE', 'iRMSE'), log_path)
    log('{:10.4f} {:10.4f} {:10.4f} {:10.4f}'.format(mae, rmse, imae, irmse), log_path)