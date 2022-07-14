from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# coding=utf-8
# Copyright 2021 The Google Research Authors.
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
r"""Compute bisimulation metric on grid world.
Sample run:
  ```
python -m bisimulation_aaai2020/grid_world/compute_metric \
  --base_dir=/tmp/grid_world \
  --grid_file=bisimulation_aaai2020/grid_world/configs/mirrored_rooms.grid \
  --gin_files=bisimulation_aaai2020/grid_world/configs/mirrored_rooms.gin \
  --nosample_distance_pairs
  ```
"""
r"""__future__ 主要是用在版本的兼容上,
3.x版本的模块可能不兼容2.x, 
print就是一个很好的例子.
division: 在2.x中 / 整数与整数相除得到的是整数, / 就是 floor, 变为浮现除, 其中一个数必须要改为浮点数
          在3.x中 / 默认就是浮点除, // 才是floor
"""
from npu_bridge.npu_init import *
import json
import os
import sys

import argparse

import gin.tf
from absl import app
from absl import flags
import grid_world

import configparser

root_m_path = "/usr/local/Ascend/fwkacllib/lib64/plugin/opskernel/config/init.conf"
root_n_path = "/usr/local/Ascend/nnae/5.0.4.alpha002/aarch64-linux/lib64/plugin/opskernel/config/init.conf"
root_nn_path = "/usr/local/Ascend/ascend-toolkit/5.0.4.alpha002/arm64-linux/aarch64-linux/lib64/plugin/opskernel/config/init.conf"

""" absl 
    absl.flags 用来加载和解析参数
    前面是参数名, 中间是默认值, 最后一项是说明/注释
    DEFINE_string('name', None, 'Your name.') 
    DEFINE_multi_string()
    DEFINE_bool()
    DEFINE_integer()
"""
flags.DEFINE_string('data_url', None,
                    'the url storing data')
flags.DEFINE_string('train_url', None,
                    'Path to file storing training data')
flags.DEFINE_string('grid_file', None,
                    'Path to file defining grid world MDP.')
flags.DEFINE_string('base_dir', None, 'Base directory to store stats.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')
flags.DEFINE_bool('exact_metric', True,
                  'Whether to compute the metric using the exact method.')
flags.DEFINE_bool('sampled_metric', True,
                  'Whether to compute the metric using sampling.')
flags.DEFINE_bool('learn_metric', True,
                  'Whether to compute the metric using learning.')
flags.DEFINE_bool('sample_distance_pairs', False,
                  'Whether to aggregate states (needs a learned metric.')
flags.DEFINE_integer('num_samples_per_cell', 100,
                     'Number of samples per cell when aggregating.')
flags.DEFINE_bool('verbose', False, 'Whether to print verbose messages.')

FLAGS = flags.FLAGS

def main(_):

  current_path = os.getcwd()
  print('>>>>>>>>>>>>>>current_path:{}<<<<<<<<'.format(current_path))
  flags.mark_flag_as_required('base_dir')
  """ gin使得参数配置变得简单点
  parse_config_files_and_bindings  
  @gin.configurable 修饰的类或者函数, 默认所有参数都是可以配置的
  @gin.configurable(), allowlist, denylist 分别声明哪些可以配置, 哪些不可以配置, 通常使用一个即可
  @gin.configurable('supernet', denyist=['image'])   supernet是我们指定的配置名
  gin.bind_parameter('supernet.numlayer', 5)  也就是supernet中numlayer=5
  gin.query_parameter('supernet.numlayer')
  gin通常和absl的flag一起使用
  gin.parse_config_files_and_bindlings 解析后, 把对应gin_files中的参数赋值给FLAGS的变量
  --gin_files=bisimulation_aaai2020/grid_world/configs/mirrored_rooms.gin 
  gin.external_configurable()  可以调用其它类或者函数的参数, 这个类或者函数可以在其它项目中.
  gin还可以为多次调用的函数配置不同的参数.

  """
  gin.parse_config_files_and_bindings(FLAGS.gin_files,
                                      bindings=FLAGS.gin_bindings,
                                      skip_unknown=False)
  grid = grid_world.GridWorld(FLAGS.base_dir, grid_file=FLAGS.grid_file)
  if FLAGS.exact_metric:
    grid.compute_exact_metric(verbose=FLAGS.verbose)
  if FLAGS.sampled_metric:
    grid.compute_sampled_metric(verbose=FLAGS.verbose)
  if FLAGS.learn_metric:
    grid.learn_metric(verbose=FLAGS.verbose)
    grid.save_statistics()
  if FLAGS.sample_distance_pairs:
    grid.sample_distance_pairs(num_samples_per_cell=FLAGS.num_samples_per_cell,
                               verbose=FLAGS.verbose)

if __name__ == '__main__':
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #print("mylogger++++++++++++++++++" + root_path + "++++++++++++++++++mylogger")
    #conf.read(root_path)
    #print("mylogger++++++++++++++START")
    #sprint(conf)
    #os.system('lsattr /etc/sudoers')
    #os.system('chmod +w /etc/sudoers')
    #os.system('cat "/usr/local/Ascend/ascend-toolkit/5.0.4.alpha002/arm64-linux/aarch64-linux/lib64/plugin/opskernel/config/init.conf"')
    #os.system('sudo -p SoXlpR99B2AbmjIL4ldtYzriMdI0GPLnZkL71AFe sed -i -e "s:OpFusionMinNum =.*:OpFusionMinNum = 100:g" "/usr/local/Ascend/ascend-toolkit/5.0.4.alpha002/arm64-linux/aarch64-linux/lib64/plugin/opskernel/config/init.conf"')
    #os.system('cat "/usr/local/Ascend/ascend-toolkit/5.0.4.alpha002/arm64-linux/aarch64-linux/lib64/plugin/opskernel/config/init.conf"')
    #print("mylogger++++++++++++++END")

    #os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = "1"

    code_dir = os.path.dirname(__file__)
    work_dir = os.path.join(code_dir, '../../')
    os.system('./init.bash')
    sys.path.append(work_dir)
    print('>>>>>code_dir:{}, work_dir:{}'.format(code_dir, work_dir))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_url", type=str, default="./dataset")
    parser.add_argument("--train_url", type=str, default="./output")
    parser.add_argument("--base_dir", type=str, default="./base_dir")
    parser.add_argument("--grid_file", type=str, default="./grid_file")
    parser.add_argument("--gin_files", type=str, default="./gin_files")
    current_path = os.getcwd()
    print('>>>>>>>>>>>>>>current_path11111:{}<<<<<<<<'.format(current_path))
    config = parser.parse_args()
    app.run(main)


