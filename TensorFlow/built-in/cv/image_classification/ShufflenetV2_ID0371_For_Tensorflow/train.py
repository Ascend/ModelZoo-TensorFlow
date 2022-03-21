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
from npu_bridge.npu_init import *
import tensorflow as tf
import os
from model import model_fn, RestoreMovingAverageHook
from input_pipeline import Pipeline
tf.logging.set_verbosity('INFO')
import math
import argparse


"""
The purpose of this script is to train a network.
Evaluation will happen periodically.

To use it just run:
python train.py

Parameters below is for training 0.5x version.
"""

# 1281144/128 = 10008.9375
# so 1 epoch ~ 10000 steps

GPU_TO_USE = '0'
'''
BATCH_SIZE = 128
VALIDATION_BATCH_SIZE = 512
NUM_EPOCHS = 133  # set 166 for 1.0x version
TRAIN_DATASET_SIZE = 1281144
NUM_STEPS = NUM_EPOCHS * (TRAIN_DATASET_SIZE // BATCH_SIZE)
'''
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # mode and parameters related
    parser.add_argument('--mode', default='train',
                        help="""mode to run the program  e.g. train and
                        train_and_evaluate""")
    parser.add_argument('--train_batch_size', default=128, type=int,
                        help="""batch size for one NPU""")
    parser.add_argument('--eval_batch_size', default=512, type=int,
                        help="""batch size for one NPU""")
    parser.add_argument('--max_train_step', default=None, type=int,
                        help="""max train step""")
    parser.add_argument('--iterations_per_loop', default=10, type=int,
                        help="""the number of steps in devices for each iteration""")
    parser.add_argument('--num_epochs', default=133, type=int,
                        help="""total epochs for training""")
    parser.add_argument('--epochs_between_evals', default=5, type=int,
                        help="""the interval between train and evaluation , only meaningful
                        when the mode is train_and_evaluate""")
    parser.add_argument('--model_dir', default='model_dir',
                        help="""directory of model.""")
    # precision mode setting
    parser.add_argument('--precision_mode', default='allow_mix_precision',
                        help="""precision mode setting""")
    # learning rate and momentum
    parser.add_argument('--initial_learning_rate', default=0.0625, type=float,
                        help="""initial learning rate""")
    parser.add_argument('--end_learning_rate', default = 1e-6, type = float,
                        help = """learning rate decay""")
    parser.add_argument('--weight_decay', default=4e-5, type = float, help="""weight decay for regularization""")

    # dataset
    parser.add_argument('--train_data_dir', default='path/data',
                        help="""directory of dataset.""")
    parser.add_argument('--eval_data_dir', default='path/data',
                        help="""directory of dataset.""")
    parser.add_argument('--train_dataset_size', default='1281144', type=int,
                        help="""the length of the imagenet12_train dataset after preprocessing.""")
    parser.add_argument('--eval_dataset_size', default='49999', type=int,
                        help="""the length of the imagenet12_val dataset after preprocessing""")

    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    return args

args = parse_args()
rank_size = int(os.getenv('RANK_SIZE'))
if args.max_train_step is not None:
    args.num_steps = args.max_train_step
else:
    args.num_steps = args.num_epochs * (args.train_dataset_size // args.train_batch_size // rank_size)

PARAMS = {
    'train_dataset_path': args.train_data_dir,
    'val_dataset_path': args.eval_data_dir,
    'weight_decay': args.weight_decay,
    'initial_learning_rate': args.initial_learning_rate,  # 0.5/8
    'decay_steps': args.num_steps,
    'end_learning_rate': args.end_learning_rate,
    'model_dir': args.model_dir,
    'precision_mode': args.precision_mode,
    'num_classes': 1000,
    'depth_multiplier': '0.5'  # set '1.0' for 1.0x version
}

def get_input_fn(is_training):

    dataset_path = PARAMS['train_dataset_path'] if is_training else PARAMS['val_dataset_path']
    filenames = os.listdir(dataset_path)
    filenames = [n for n in filenames if n.endswith('.tfrecords')]
    filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

    batch_size = args.train_batch_size if is_training else args.eval_batch_size
    num_epochs = None if is_training else 1

    def input_fn():
        pipeline = Pipeline(
            filenames, is_training,
            batch_size=batch_size,
            num_epochs=num_epochs
        )
        return pipeline.dataset

    return input_fn


session_config = tf.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.visible_device_list = GPU_TO_USE

args.save_step = args.epochs_between_evals * (args.train_dataset_size // args.train_batch_size // rank_size)
run_config = NPURunConfig(
    model_dir=PARAMS['model_dir'],
    session_config=session_config,
    iterations_per_loop=args.iterations_per_loop,
    hcom_parallel=True,
    precision_mode= args.precision_mode,
	# enable_data_pre_proc=False,
    save_checkpoints_steps=args.save_step,
    save_summary_steps=0,
    log_step_count_steps=1,
)

train_input_fn = get_input_fn(is_training=True)
val_input_fn = get_input_fn(is_training=False)
estimator = NPUEstimator(model_fn, params=PARAMS, config=npu_run_config_init(run_config=run_config))

if args.mode == 'train':
    n_loops = math.ceil(args.num_epochs / args.epochs_between_evals)
    schedule = [args.epochs_between_evals for _ in range(int(n_loops))]
    schedule[-1] = args.num_epochs - sum(schedule[:-1])  # over counting.

    for cycle_index, num_train_epochs in enumerate(schedule):
        tf.compat.v1.logging.info('Starting cycle: %d/%d Num_train_epochs: %d', cycle_index, int(n_loops),
                                  int(num_train_epochs))
        estimator.train(input_fn=get_input_fn(True),
                        steps=num_train_epochs * (args.train_dataset_size // args.train_batch_size // rank_size))

        tf.compat.v1.logging.info('Starting to evaluate.')
        estimator.evaluate(input_fn=get_input_fn(False), hooks=[RestoreMovingAverageHook(PARAMS['model_dir'])])

elif args.mode == 'train_and_evaluate':

    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=args.num_steps, hooks=npu_hooks_append())
    eval_spec = tf.estimator.EvalSpec(
        val_input_fn, steps=None, start_delay_secs=3600 * 2, throttle_secs=3600 * 2,
        hooks=npu_hooks_append(hooks_list=[RestoreMovingAverageHook(PARAMS['model_dir'])])
    )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

else:
    raise ValueError("Invalid mode.")


