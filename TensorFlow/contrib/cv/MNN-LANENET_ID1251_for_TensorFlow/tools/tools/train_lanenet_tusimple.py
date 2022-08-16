#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-24 下午9:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : train_lanenet_tusimple.py
# @IDE: PyCharm
"""
Train lanenet script
"""

import argparse

import os

os.system('pip install loguru')
os.system('pip install glog')
os.system('pip install impl')

import sys
# __file__为获取当前执行脚本main.py的绝对路径
# os.path.dirname(__file__)获取main.py的父目录，即project_dir的绝对路径
current_path = os.path.dirname(__file__)
father_path = os.path.dirname(current_path)
sys.path.append(father_path)
grandfather_path = os.path.dirname(father_path)
sys.path.append(grandfather_path)
# 在sys.path.append执行完毕之后再导入其他模块


print("[CANN-Modelzoo] code_dir path is [%s]" % (sys.path[0]))
code_dir = sys.path[0]
os.chdir(code_dir)
print("[CANN-Modelzoo] work_dir path is [%s]" % (os.getcwd()))

from trainner import tusimple_lanenet_single_gpu_trainner as single_gpu_trainner
from local_utils.log_util import init_logger
from local_utils.config_utils import parse_config_utils


from npu_bridge.npu_init import *
import moxing as mox

# mox.file.set_auth(is_secure=False)
# from moxing.framework.file import file_io
# file_io._NUMBER_OF_PROCESSES=1
# mox.file.copy_parallel(threads=0, is_processing=False)

LOG = init_logger.get_logger(log_file_name_prefix='lanenet_train')
CFG = parse_config_utils.lanenet_cfg


data_dir = '/home/ma-user/modelarts/user-job-dir/code/tfrecords'
os.makedirs(data_dir)

OUT_DIR = ''
#
# model_dir = "/cache/result"
# os.makedirs(model_dir)


def train_model():
    """

    :return:
    """

    worker = single_gpu_trainner.LaneNetTusimpleTrainer(cfg=CFG)

    worker.train()

    return


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    # 解析输入参数data_url
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_url", type=str, default="/home/ma-user/modelarts/inputs/data_url_0")
    parser.add_argument("--train_url", type=str, default="/home/ma-user/modelarts/outputs/train_url_0/")

    return parser.parse_args()


if __name__ == '__main__':
    """
    main function
    """

    args = get_arguments()

    mox.file.copy_parallel(args.data_url, data_dir)

    train_model()

    # mox.file.copy_parallel(data_dir, args.data_url)


