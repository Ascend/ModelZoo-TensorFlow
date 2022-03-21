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

import argparse
import sys
from tf_config import tf_config
import dnnlib
import dnnlib.submission.submit as submit
import validation
import validation_pb

# Submit config
# ------------------------------------------------------------------------------------------

submit_config = dnnlib.SubmitConfig()
submit_config.run_dir_root = 'results'
submit_config.run_dir_ignore += ['datasets', 'results', 'ACL', 'test', 'kernel_meta']

desc = "autoencoder"

# Network config
# ------------------------------------------------------------------------------------------

net_config = dnnlib.EasyDict(func_name="network.autoencoder")

# Optimizer config
# ------------------------------------------------------------------------------------------

optimizer_config = dnnlib.EasyDict(beta1=0.9, beta2=0.99, epsilon=1e-8)

# Noise augmentation config
gaussian_noise_config = dnnlib.EasyDict(
    func_name='train.AugmentGaussian',
    train_stddev_rng_range=(0.0, 50.0),
    validation_stddev=25.0
)
poisson_noise_config = dnnlib.EasyDict(
    func_name='train.AugmentPoisson',
    lam_max=50.0
)

# ------------------------------------------------------------------------------------------
# Preconfigured validation sets
datasets = {
    'kodak': dnnlib.EasyDict(dataset_dir='datasets/kodak'),
    'bsd300': dnnlib.EasyDict(dataset_dir='datasets/bsd300'),
    'set14': dnnlib.EasyDict(dataset_dir='datasets/set14')
}

default_validation_config = datasets['kodak']

corruption_types = {
    'gaussian': gaussian_noise_config,
    'poisson': poisson_noise_config
}

# Train config
# ------------------------------------------------------------------------------------------

train_config = dnnlib.EasyDict(
    iteration_count=1000,
    eval_interval=100,
    minibatch_size=4,
    run_func_name="train.train",
    learning_rate=0.0003,
    ramp_down_perc=0.3,
    noise=gaussian_noise_config,
    #    noise=poisson_noise_config,
    noise2noise=True,
    is_distributed=False,
    is_loss_scale=True,
    train_tfrecords='datasets/imagenet_val_raw.tfrecords',
    validation_config=default_validation_config,
    tf_config=tf_config
)

# Validation run config
# ------------------------------------------------------------------------------------------
validate_config = dnnlib.EasyDict(
    run_func_name="validation.validate",
    dataset=default_validation_config,
    network_snapshot=None,
    noise=gaussian_noise_config,
    tf_config=tf_config,
)

# Validation pb run config
# ------------------------------------------------------------------------------------------

validate_pb_config = dnnlib.EasyDict(
    run_func_name="validation_pb.validate_pb",
    dataset=default_validation_config,
    input_tensor_name=None,
    output_tensor_name=None,
    pbdir=None,
    width=None,
    height=None,
    noise=gaussian_noise_config,
    tf_config=tf_config,
)


# ------------------------------------------------------------------------------------------

# jhellsten quota group

def error(*print_args):
    print(*print_args)
    sys.exit(1)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ------------------------------------------------------------------------------------------
examples = '''examples:

  # Train a network using the BSD300 dataset:
  python %(prog)s train --train-tfrecords=datasets/bsd300.tfrecords

  # Run a set of images through a pre-trained network:
  python %(prog)s validate --network-snapshot=results/network_final.pickle --dataset-dir=datasets/kodak
'''

if __name__ == "__main__":
    def train(args):
        if args:
            n2n = args.noise2noise if 'noise2noise' in args else True
            train_config.noise2noise = n2n
            if 'long_train' in args and args.long_train:
                train_config.iteration_count = 500000
                train_config.eval_interval = 5000
                train_config.ramp_down_perc = 0.5
        else:
            print('running with defaults in train_config')
        noise = 'gaussian'
        if 'noise' in args:
            if args.noise not in corruption_types:
                error('Unknown noise type', args.noise)
            else:
                noise = args.noise
        train_config.noise = corruption_types[noise]

        if train_config.noise2noise:
            submit_config.run_desc += "-n2n"
        else:
            submit_config.run_desc += "-n2c"

        if 'train_tfrecords' in args and args.train_tfrecords is not None:
            train_config.train_tfrecords = submit.get_path_from_template(args.train_tfrecords)

        train_config.is_distributed = args.is_distributed,
        train_config.is_loss_scale = args.is_loss_scale,

        train_config.tf_config["graph_run_mode"] = args.graph_run_mode
        train_config.tf_config["precision_mode"] = args.precision_mode
        train_config.tf_config["op_select_implmode"] = args.op_select_implmode
        train_config.tf_config["hcom_parallel"] = args.hcom_parallel

        print(train_config)
        dnnlib.submission.submit.submit_run(submit_config, **train_config)


    def validate(args):
        if submit_config.submit_target != dnnlib.SubmitTarget.LOCAL:
            print('Command line overrides currently supported only in local runs for the validate subcommand')
            sys.exit(1)
        if args.dataset_dir is None:
            error('Must select dataset with --dataset-dir')
        else:
            validate_config.dataset = {
                'dataset_dir': args.dataset_dir
            }
        if args.noise not in corruption_types:
            error('Unknown noise type', args.noise)
        validate_config.noise = corruption_types[args.noise]
        if args.network_snapshot is None:
            error('Must specify trained network filename with --network-snapshot')
        validate_config.network_snapshot = args.network_snapshot

        validate_config.tf_config["graph_run_mode"] = args.graph_run_mode
        validate_config.tf_config["precision_mode"] = args.precision_mode
        validate_config.tf_config["op_select_implmode"] = args.op_select_implmode

        print(validate_config)
        dnnlib.submission.submit.submit_run(submit_config, **validate_config)


    def validate_pb(args):
        if submit_config.submit_target != dnnlib.SubmitTarget.LOCAL:
            print('Command line overrides currently supported only in local runs for the validate subcommand')
            sys.exit(1)
        if args.dataset_dir is None:
            error('Must select dataset with --dataset-dir')
        else:
            validate_pb_config.dataset = {
                'dataset_dir': args.dataset_dir
            }
        if args.noise not in corruption_types:
            error('Unknown noise type', args.noise)
        validate_pb_config.noise = corruption_types[args.noise]
        if args.input_tensor_name is None:
            error('Must select pb input tensor name with --input_tensor_name')
        if args.output_tensor_name is None:
            error('Must select pb output tensor name with --output_tensor_name')
        if args.pbdir is None:
            error('Must specify pb filename with --noise2noise')

        validate_pb_config.input_tensor_name = args.input_tensor_name
        validate_pb_config.output_tensor_name = args.output_tensor_name
        validate_pb_config.pbdir = args.pbdir
        validate_pb_config.width = args.width
        validate_pb_config.height = args.height

        validate_pb_config.tf_config["graph_run_mode"] = args.graph_run_mode
        validate_pb_config.tf_config["precision_mode"] = args.precision_mode
        validate_pb_config.tf_config["op_select_implmode"] = args.op_select_implmode

        print(validate_pb_config)
        dnnlib.submission.submit.submit_run(submit_config, **validate_pb_config)


    def infer_image(args):
        if submit_config.submit_target != dnnlib.SubmitTarget.LOCAL:
            print('Command line overrides currently supported only in local runs for the validate subcommand')
            sys.exit(1)
        if args.image is None:
            error('Must specify image file with --image')
        if args.out is None:
            error('Must specify output image file with --out')
        if args.network_snapshot is None:
            error('Must specify trained network filename with --network-snapshot')
        # Note: there's no dnnlib.submission.submit_run here. This is for quick interactive
        # testing, not for long-running training or validation runs.

        infer_tf_config = tf_config
        infer_tf_config["graph_run_mode"] = args.graph_run_mode
        infer_tf_config["precision_mode"] = args.precision_mode
        infer_tf_config["op_select_implmode"] = args.op_select_implmode

        validation.infer_image(args.network_snapshot, args.image, args.out, infer_tf_config)


    def infer_image_pb(args):
        if submit_config.submit_target != dnnlib.SubmitTarget.LOCAL:
            print('Command line overrides currently supported only in local runs for the validate subcommand')
            sys.exit(1)
        if args.image is None:
            error('Must specify image file with --image')
        if args.out is None:
            error('Must specify output image file with --out')
        if args.input_tensor_name is None:
            error('Must select pb input tensor name with --input_tensor_name')
        if args.output_tensor_name is None:
            error('Must select pb output tensor name with --output_tensor_name')
        if args.pbdir is None:
            error('Must specify pb filename with --noise2noise')
            # Note: there's no dnnlib.submission.submit_run here. This is for quick interactive
            # testing, not for long-running training or validation runs.

        infer_tf_config = tf_config
        infer_tf_config["graph_run_mode"] = args.graph_run_mode
        infer_tf_config["precision_mode"] = args.precision_mode
        infer_tf_config["op_select_implmode"] = args.op_select_implmode

        validation_pb.infer_image_pb(args.pbdir, args.input_tensor_name, args.output_tensor_name, args.image, args.out,
                                     args.height, args.width, infer_tf_config)


    # Train by default
    parser = argparse.ArgumentParser(
        description='Train a network or run a set of images through a trained network.',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--desc', default='', help='Append desc to the run descriptor string')
    parser.add_argument('--run-dir-root',
                        help='Working dir for a training or a validation run. Will contain training and validation results.')
    parser.add_argument('--precision-mode', default="allow_mix_precision", type=str,
                        help='Must choose one from (1) allow_fp32_to_fp16 (2) force_fp16 (3) allow_mix_precision')
    parser.add_argument('--op-select-implmode', default="high_precision", type=str,
                        help='Must choose one from (1) high_precision (2) high_performance')
    parser.add_argument('--graph-run-mode', default='0', type=str, help='Whether training(1) or inferring(0)')
    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')
    parser_train = subparsers.add_parser('train', help='Train a network')
    parser_train.add_argument('--noise2noise', nargs='?', type=str2bool, const=True, default=True,
                              help='Noise2noise (--noise2noise=true) or noise2clean (--noise2noise=false).  Default is noise2noise=true.')
    parser_train.add_argument('--noise', type=str, default='gaussian',
                              help='Type of noise corruption (one of: gaussian, poisson)')
    parser_train.add_argument('--long-train', type=str2bool, default=False,
                              help='Train for a very long time (500k iterations or 500k*minibatch image)')
    parser_train.add_argument('--train-tfrecords', type=str, help='Filename of the training set tfrecords file')
    parser_train.add_argument('--is-distributed', default=False, type=str2bool, help='Whether to use multi-npu')
    parser_train.add_argument('--is-loss-scale', default=True, type=str2bool, help='Whether to use loss scale')
    parser_train.add_argument('--hcom-parallel', default=False, type=str2bool, help='Whether to use parallel allreduce')
    parser_train.set_defaults(func=train)

    parser_validate = subparsers.add_parser('validate', help='Run a set of images through the network')
    parser_validate.add_argument('--dataset-dir', type=str,
                                 help='Load all images from a directory (*.png, *.jpg/jpeg, *.bmp)')
    parser_validate.add_argument('--network-snapshot', type=str, help='Trained network pickle')
    parser_validate.add_argument('--noise', type=str, default='gaussian',
                                 help='Type of noise corruption (one of: gaussian, poisson)')
    parser_validate.set_defaults(func=validate)

    parser_validate_pb = subparsers.add_parser('validate-pb', help='Run a set of images through the pb')
    parser_validate_pb.add_argument('--dataset-dir', help='Load all images from a directory (*.png, *.jpg/jpeg, *.bmp)')
    parser_validate_pb.add_argument('--pbdir', help='pb file path')
    parser_validate_pb.add_argument('--width', type=int, default=768, help='resized image width before inference')
    parser_validate_pb.add_argument('--height', type=int, default=512, help='resized image height before inference')
    parser_validate_pb.add_argument('--input-tensor-name', type=str, default='input:0', help='input tensor name')
    parser_validate_pb.add_argument('--output-tensor-name', type=str, default='output:0', help='output tensor name')
    parser_validate_pb.add_argument('--noise', type=str, default='gaussian',
                                    help='Type of noise corruption (one of: gaussian, poisson)')
    parser_validate_pb.set_defaults(func=validate_pb)

    parser_infer_image = subparsers.add_parser('infer-image',
                                               help='Run one image through the network without adding any noise')
    parser_infer_image.add_argument('--image', type=str, help='Image filename')
    parser_infer_image.add_argument('--out', type=str, help='Output filename')
    parser_infer_image.add_argument('--network-snapshot', type=str, help='Trained network pickle')
    parser_infer_image.set_defaults(func=infer_image)

    parser_infer_image_pb = subparsers.add_parser('infer-image-pb',
                                                  help='Run one image through the network without adding any noise')
    parser_infer_image_pb.add_argument('--image', type=str, help='Image filename')
    parser_infer_image_pb.add_argument('--out', type=str, help='Output filename')
    parser_infer_image_pb.add_argument('--width', type=int, default=768, help='resized image width before inference')
    parser_infer_image_pb.add_argument('--height', type=int, default=512, help='resized image height before inference')
    parser_infer_image_pb.add_argument('--pbdir', type=str, help='pb file path')
    parser_infer_image_pb.add_argument('--input-tensor-name', type=str, default='input:0', help='input tensor name')
    parser_infer_image_pb.add_argument('--output-tensor-name', type=str, default='output:0', help='output tensor name')
    parser_infer_image_pb.set_defaults(func=infer_image_pb)

    args = parser.parse_args()
    submit_config.run_desc = desc + args.desc

    if args.run_dir_root is not None:
        submit_config.run_dir_root = args.run_dir_root
    if args.command is not None:
        args.func(args)
    else:
        # Train if no subcommand was given
        train(args)
