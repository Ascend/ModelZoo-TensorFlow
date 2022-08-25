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
Train lanenet script
"""

from trainner import tusimple_lanenet_single_gpu_trainner as single_gpu_trainner
from local_utils.log_util import init_logger
from local_utils.config_utils import parse_config_utils

LOG = init_logger.get_logger(log_file_name_prefix='lanenet_train')
CFG = parse_config_utils.lanenet_cfg


def train_model():
    """

    :return:
    """
    worker = single_gpu_trainner.LaneNetTusimpleTrainer(cfg=CFG)
    worker.train()
    return


if __name__ == '__main__':
    """
    main function
    """
    train_model()


