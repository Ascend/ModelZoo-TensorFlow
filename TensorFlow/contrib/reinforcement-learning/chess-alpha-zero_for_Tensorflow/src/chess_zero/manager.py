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
"""
Manages starting off each of the separate processes involved in ChessZero -
self play, training, and evaluation.
"""
from npu_bridge.npu_init import *
import argparse

from logging import getLogger,disable

from .lib.logger import setup_logger
from .config import Config

logger = getLogger(__name__)

CMD_LIST = ['self', 'opt', 'eval', 'sl', 'uci']


def create_parser():
    """
    Parses each of the arguments from the command line
    :return ArgumentParser representing the command line arguments that were supplied to the command line:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", help="what to do", choices=CMD_LIST)
    parser.add_argument("--new", help="run from new best model", action="store_true")
    parser.add_argument("--type", help="use normal setting", default="mini")
    parser.add_argument("--total-step", help="set TrainerConfig.start_total_steps", type=int)
    return parser


def setup(config: Config, args):
    """
    Sets up a new config by creating the required directories and setting up logging.

    :param Config config: config to create directories for and to set config from based on the args
    :param ArgumentParser args: args to use to control config.
    """
    config.opts.new = args.new
    if args.total_step is not None:
        config.trainer.start_total_steps = args.total_step
    config.resource.create_directories()
    setup_logger(config.resource.main_log_path)


def start():
    """
    Starts one of the processes based on command line arguments.

    :return : the worker class that was started
    """
    parser = create_parser()
    args = parser.parse_args()
    config_type = args.type

    if args.cmd == 'uci':
        disable(999999) # plz don't interfere with uci

    config = Config(config_type=config_type)
    setup(config, args)

    logger.info(f"config type: {config_type}")

    if args.cmd == 'self':
        from .worker import self_play
        return self_play.start(config)
    elif args.cmd == 'opt':
        from .worker import optimize
        return optimize.start(config)
    elif args.cmd == 'eval':
        from .worker import evaluate
        return evaluate.start(config)
    elif args.cmd == 'sl':
        from .worker import sl
        return sl.start(config)
    elif args.cmd == 'uci':
        from .play_game import uci
        return uci.start(config)