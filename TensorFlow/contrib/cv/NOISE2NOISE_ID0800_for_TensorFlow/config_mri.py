# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
# ============================================================================
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

import dnnlib
import argparse
import dnnlib.submission.submit as submit
from tf_config import tf_config

# Submit config
# ------------------------------------------------------------------------------------------

submit_config = dnnlib.SubmitConfig()
submit_config.run_dir_root = "results"
submit_config.run_dir_ignore += ['datasets', 'results']

desc = "n2n-mri"

# ----------------------------------------------------------------------------
# Paths etc.

data_dir = 'datasets'

# ----------------------------------------------------------------------------
# Baseline configuration.

run_desc = desc
random_seed = 1000

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help='Long training runs')
    parser.add_argument('--is-distributed', default=False, type=str2bool, help='Whether to use multi-npu')
    parser.add_argument('--is-loss-scale', default=True, type=str2bool, help='Whether to use loss scale')
    parser.add_argument('--hcom-parallel', default=False, type=str2bool,
                        help='Whether to use parallel allreduce')
    parser.add_argument('--precision-mode', default="allow_mix_precision", type=str,
                        help='Must choose one from (1) allow_fp32_to_fp16 (2) force_fp16 (3) allow_mix_precision')
    parser.add_argument('--op-select-implmode', default="high_precision", type=str,
                        help='Must choose one from (1) high_precision (2) high_performance')
    parser.add_argument('--graph-run-mode', default='1', type=str, help='Whether training(1) or inferring(0)')
    args = parser.parse_args()

    # ----------------------------------------------------------------------------
    # Basic MRI runs.

    run_desc = 'mri'
    train = dict(corrupt_params=dict(), augment_params=dict())

    run_desc += '-ixi'
    train.update(dataset_train=dict(fn='ixi_train.pkl'),
                 augment_params=dict(translate=64))  # 4936 images, lots of augmentation.
    train.update(dataset_test=dict(fn='ixi_valid.pkl'))  # use all images, should be 1000

    train['run_func_name'] = 'train_mri.train'

    train['corrupt_params'].update(type='bspec', p_at_edge=0.025)  # 256x256 avg = 0.10477
    train.update(learning_rate_max=0.001)
    # Noise2noise (corrupt_targets=True) or noise2clean (corrupt_targets=False)
    train.update(corrupt_targets=True)

    # NPU场景下，已改动
    train.update(post_op='fspec')

    train.update(num_epochs=args.epoch)  # Long training runs.

    # Paper cases. Overrides post-op and target corruption modes.
    if train.get('corrupt_targets'):
        run_desc += '_s-n2n_'
    else:
        run_desc += '_s-n2c_'

    train_tf_config = tf_config
    train_tf_config["graph_run_mode"] = args.graph_run_mode
    train_tf_config["precision_mode"] = args.precision_mode
    train_tf_config["op_select_implmode"] = args.op_select_implmode
    train_tf_config["hcom_parallel"] = args.hcom_parallel

    train.update(tf_config=train_tf_config)
    train.update(is_distributed=args.is_distributed)
    train.update(is_loss_scale=args.is_loss_scale)

    # Final inference only. Note: verify that dataset, corruption, and post_op match with loaded network.
    # train.update(load_network='382-mri-ixi_s-n2n_-lr0.001000-Cbs0.025000-At64-Pfspec/network-final.pkl', start_epoch='final')      # N2N
    # train.update(load_network='380-mri-ixi_s-n2c_-lr0.001000-clean-Cbs0.025000-At64-Pfspec/network-final.pkl', start_epoch='final') # N2C

    if train.get('num_epochs'): run_desc += '-ep%d' % train['num_epochs']
    if train.get('learning_rate_max'): run_desc += '-lr%f' % train['learning_rate_max']
    if not train.get('corrupt_targets', True): run_desc += '-clean'
    if train.get('minibatch_size'): run_desc += '-mb%d' % train['minibatch_size']
    if train['corrupt_params'].get('type') == 'gaussian': run_desc += '-Cg%f' % train['corrupt_params']['scale']
    if train['corrupt_params'].get('type') == 'bspec': run_desc += '-Cbs%f' % train['corrupt_params']['p_at_edge']
    if train['corrupt_params'].get('type') == 'bspeclin': run_desc += '-Cbslin%f' % train['corrupt_params']['p_at_edge']
    if train['augment_params'].get('translate', 0) > 0: run_desc += '-At%d' % train['augment_params']['translate']
    if train.get('post_op'): run_desc += '-P%s' % train['post_op']
    if random_seed != 1000: run_desc += '-%d' % random_seed
    if train.get('load_network'): run_desc += '-LOAD%s' % train['load_network'][:3]
    if train.get('start_epoch'): run_desc += '-start%s' % train['start_epoch']

    # Farm submit config
    # ----------------------------------------------------------------

    # Submission target
    run_desc += "-L"
    submit_config.submit_target = dnnlib.SubmitTarget.LOCAL

    submit_config.run_desc = run_desc

    # ----------------------------------------------------------------------------
    print(train)
    submit.submit_run(submit_config, **train)


if __name__ == "__main__":
    main()
