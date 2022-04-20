# Copyright 2022 Huawei Technologies Co., Ltd
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
import os
import sys
import inspect
import importlib
from yacs.config import CfgNode

import src.runner.npu_pkgs

from src.utils.logger import logger as logger
from src.utils.defaults import cfg
from src.utils.world import world
from src.utils.utils import convert_dict_to_list
from src.networks import build_network
from src.engine import build_engine
from src.dataloaders import build_dataloader
from src.utils.constant import VALID_MODE


def get_args():
    """Get external arguments.

    Returns:
        Namespace: the arguements to the whole program.
    """
    parser = argparse.ArgumentParser(description="HiSi Ascend Video Processing Toolkit")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    return args


def dump_cfg(_cfg):
    """Dump config info to log file and stdout.

    Args:
        _cfg: yacs node, the configuration.
    """
    cfg_str = _cfg.dump()
    if not os.path.exists(_cfg.train.output_dir):
        os.makedirs(_cfg.train.output_dir, exist_ok=True)
    dump_file = os.path.join(_cfg.train.output_dir, f"configure_{_cfg.mode}.yaml")
    with open(dump_file, 'w') as f:
        f.write(cfg_str)
    logger.info(_cfg)


def processing(cfg):
    """Processing function.

    This function supports training, inference and freeze engine.

    Args:
        cfg: yacs node, global configuraton.
    """
    world.initialize(device_type=cfg.env.device)
    
    if not cfg.log_file.startswith('/'):
        log_file = os.path.join(cfg.train.output_dir, cfg.log_file)
    else:
        log_file = cfg.log_file
    
    # Silence all nodes other than the root node.
    if world.is_root_rank:
        logger.add_log_file(log_file)
    else:
        logger.silence = True

    if world.is_root_rank:
        dump_cfg(cfg)

    # build networks
    network = build_network(cfg)

    # build dataloader
    dataloader = build_dataloader(cfg)

    # get engine
    engine_type = build_engine(cfg)
    engine = engine_type(dataloader, network, cfg)
    engine.run()


def main():
    """Main entry function.
    """
    args = get_args()
    # Support either python config file with a dict, or a yaml file.
    if args.config_file.endswith('.py'):
        vars = {}
        exec(open(args.config_file).read(), vars)
        cfg.merge_from_other_cfg(CfgNode(vars['cfg']))
    elif config_module_name[-1] == 'yaml':
        cfg.merge_from_file(args.config_file)
    else:
        raise ValueError()
    
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    assert cfg.mode in VALID_MODE
    
    processing(cfg)


if __name__ == '__main__':
    main()
