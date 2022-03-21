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
import sys
from tf_config import tf_config

# Submit config
# ------------------------------------------------------------------------------------------

submit_config = dnnlib.SubmitConfig()
submit_config.run_dir_root = 'results'
submit_config.run_dir_ignore += ['datasets', 'results']

desc = "autoencoder"

# ------------------------------------------------------------------------------------------
# Preconfigured validation sets
datasets = {
    'ixi_valid': dnnlib.EasyDict(dataset_dir='datasets/ixi_valid.pkl'),
}

default_validation_config = datasets['ixi_valid']

# Validation run config
# ------------------------------------------------------------------------------------------
validate_config = dnnlib.EasyDict(
    run_func_name="validation_mri_pb.validate",
    data_dir='datasets',
    dataset_test=dict(fn='ixi_valid.pkl'),
    pbdir=None,
    input_tensor_name=None,
    output_tensor_name=None,
    minibatch_size=1,
    post_op=None,
    corrupt_params=dict(type='bspec', p_at_edge=0.025),
    tf_config=tf_config
)


# ------------------------------------------------------------------------------------------

# jhellsten quota group

def error(*print_args):
    print(*print_args)
    sys.exit(1)


def validate_cof(args):
    if submit_config.submit_target != dnnlib.SubmitTarget.LOCAL:
        print('Command line overrides currently supported only in local runs for the validate subcommand')
        sys.exit(1)
    if args.data_dir is None:
        error('Must select dataset with --dataset-dir')
    else:
        validate_config.data_dir = {
            'data_dir': args.data_dir
        }
    if args.input_tensor_name is None:
        error('Must select pb input tensor name with --input_tensor_name')
    if args.output_tensor_name is None:
        error('Must select pb output tensor name with --output_tensor_name')
    if args.pbdir is None:
        error('Must specify pb filename with --noise2noise')
    if args.post_op == 'fspec' or None:
        validate_config.post_op = args.post_op
    else:
        error('--post_op must be fspec or None')
    validate_config.input_tensor_name = args.input_tensor_name
    validate_config.output_tensor_name = args.output_tensor_name
    validate_config.pbdir = args.pbdir

    validate_config.tf_config["graph_run_mode"] = args.graph_run_mode
    validate_config.tf_config["precision_mode"] = args.precision_mode
    validate_config.tf_config["op_select_implmode"] = args.op_select_implmode

    print(validate_config)
    dnnlib.submission.submit.submit_run(submit_config, **validate_config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', default='', help='Append desc to the run descriptor string')
    parser.add_argument('--run-dir-root',
                        help='Working dir for a training or a validation run. Will contain training and validation results.')
    parser.add_argument('--precision-mode', default="allow_mix_precision", type=str,
                                 help='Must choose one from (1) allow_fp32_to_fp16 (2) force_fp16 (3) allow_mix_precision')
    parser.add_argument('--op-select-implmode', default="high_precision", type=str,
                                 help='Must choose one from (1) high_precision (2) high_performance')
    parser.add_argument('--graph-run-mode', default='0', type=str, help='Whether training(1) or inferring(0)')
    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')
    parser_validate = subparsers.add_parser('validate', help='Run a set of images through the network')
    parser_validate.add_argument('--input-tensor-name', default='input:0', help='input tensor name')
    parser_validate.add_argument('--output-tensor-name', default='output:0',
                                 help='output tensor name')
    parser_validate.add_argument('--data-dir', default=r'datasets/ixi_valid.pkl',
                                 help='Load all images from a directory (*.png, *.jpg/jpeg, *.bmp)')
    parser_validate.add_argument('--pbdir', default=r'model/pb/test_mri.pb',
                                 help='Trained network pickle')
    parser_validate.add_argument('--post-op', type=str, default=None)
    parser_validate.set_defaults(func=validate_cof)
    args = parser.parse_args()
    submit_config.run_desc = desc + args.desc
    if args.run_dir_root is not None:
        submit_config.run_dir_root = args.run_dir_root
    if args.command is not None:
        args.func(args)
    else:
        # validate if no subcommand was given
        validate_cof(args)
