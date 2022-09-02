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
import ast
import os
import argparse
import warnings
import yaml
import tensorflow as tf

from utils.data import get_data
from utils.result import generate_result
from utils.test import test_model
from utils.train import train_model
from rich.console import Console
import npu_device

CONSOLE = Console()
warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_args():
    parser = argparse.ArgumentParser(prog='self-attention-har')
    parser.add_argument(
        '--train',
        action='store_true',
        default=False,
        help='Training Mode')
    parser.add_argument(
        '--test',
        action='store_true',
        default=False,
        help='Testing Mode')
    parser.add_argument(
        '--epochs',
        default=150,
        type=int,
        help='Number of Epochs for Training')
    parser.add_argument(
        '--dataset',
        default='zim',
        type=str,
        choices=['zim', 'pamap2', 'opp', 'skoda', 'uschad'],
        help='Name of Dataset for Model Training')
    parser.add_argument('--data_path', default='./', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--log_steps', default=70, type=int)
    parser.add_argument('--static', action='store_true', default=False, help='static input shape, default is False')
    ############维测参数##############
    parser.add_argument('--precision_mode', default="allow_mix_precision", type=str,help='the path to save over dump data')
    parser.add_argument('--over_dump', dest='over_dump', type=ast.literal_eval, help='if or not over detection, default is False')
    parser.add_argument('--data_dump_flag', dest='data_dump_flag', type=ast.literal_eval, help='data dump flag, default is False')
    parser.add_argument('--data_dump_step', default="10", help='data dump step, default is 10')
    parser.add_argument('--profiling', dest='profiling', type=ast.literal_eval,help='if or not profiling for performance debug, default is False')
    parser.add_argument('--profiling_dump_path', default="/home/data", type=str,help='the path to save profiling data')
    parser.add_argument('--over_dump_path', default="/home/data", type=str,help='the path to save over dump data')
    parser.add_argument('--data_dump_path', default="/home/data", type=str,help='the path to save dump data')
    parser.add_argument('--use_mixlist', dest='use_mixlist', type=ast.literal_eval, help='use_mixlist flag, default is False')
    parser.add_argument('--fusion_off_flag', dest='fusion_off_flag', type=ast.literal_eval, help='fusion_off flag, default is False')
    parser.add_argument('--mixlist_file', default="ops_info.json", type=str,help='mixlist file name, default is ops_info.json')
    parser.add_argument('--fusion_off_file', default="fusion_switch.cfg", type=str,help='fusion_off file name, default is fusion_switch.cfg')
    parser.add_argument('--auto_tune', dest='auto_tune', type=ast.literal_eval, help='autotune flag, default is False')
    args = parser.parse_args()
    return args

def npu_config():
  if args.data_dump_flag:
    npu_device.global_options().dump_config.enable_dump = True
    npu_device.global_options().dump_config.dump_path = args.data_dump_path
    npu_device.global_options().dump_config.dump_step = args.data_dump_step
    npu_device.global_options().dump_config.dump_mode = "all"

  if args.over_dump:
    npu_device.global_options().dump_config.enable_dump_debug = True
    npu_device.global_options().dump_config.dump_path = args.over_dump_path
    npu_device.global_options().dump_config.dump_debug_mode = "all"

  if args.profiling:
    npu_device.global_options().profiling_config.enable_profiling = True
    profiling_options = '{"output":"' + args.profiling_dump_path + '", \
                        "training_trace":"on", \
                        "task_trace":"on", \
                        "aicpu":"on", \
                        "L2":"on", \
                        "aic_metrics":"PipeUtilization",\
                        "fp_point":"", \
                        "bp_point":""}'
    npu_device.global_options().profiling_config.profiling_options = profiling_options
  npu_device.global_options().precision_mode=args.precision_mode
  if args.use_mixlist and args.precision_mode=='allow_mix_precision':
    npu_device.global_options().modify_mixlist=args.mixlist_file
  if args.fusion_off_flag:
    npu_device.global_options().fusion_switch_file=args.fusion_off_file
  if args.auto_tune:
    npu_device.global_options().auto_tune_mode="RL,GA"
  npu_device.open().as_default()

args = parse_args()
npu_config()
def main():
    data_path = args.data_path
    batch_size = args.batch_size
    epochs = args.epochs
    log_steps = args.log_steps

    model_config_file = open('configs/model.yaml', mode='r')
    model_cfg = yaml.load(model_config_file, Loader=yaml.FullLoader)
    train_x, train_y, val_x, val_y, test_x, test_y = get_data(
                                            dataset=args.dataset, data_path=data_path)
    
    if args.static:
        train_x , train_y = train_x[:train_x.shape[0]//batch_size * batch_size], train_y[:train_y.shape[0]//batch_size * batch_size]
        val_x , val_y = val_x[:val_x.shape[0]//batch_size * batch_size], val_y[:val_y.shape[0]//batch_size * batch_size]
        test_x , test_y = test_x[:test_x.shape[0]//batch_size * batch_size], test_y[:test_y.shape[0]//batch_size * batch_size]
        
    if args.train:
        CONSOLE.print('\n[MODEL TRAINING]', style='bold green')
        train_model(dataset=args.dataset,
                    model_config=model_cfg,
                    train_x=train_x, train_y=train_y,
                    val_x=val_x, val_y=val_y,
                    epochs=epochs, batch_size=batch_size, log_steps=log_steps)

    if args.test:
        CONSOLE.print('\n[MODEL INFERENCE]', style='bold green')
        pred = test_model(dataset=args.dataset, model_config=model_cfg,
                          test_x=test_x)
        generate_result(dataset=args.dataset, ground_truth=test_y,
                        prediction=pred)


if __name__ == '__main__':
    main()
