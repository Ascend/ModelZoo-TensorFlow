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

import inspect
import os
import shutil

from .exceptions import WorldUninitializedError
from .logger import logger



def _setup_npu_env(remove_kernel_meta=True, device_id=None, rank_id=None, rank_size=None):
    """Setup NPU environment variables.
    """
    os.environ['FUSION_TENSOR_SIZE'] = '20000000'
    os.environ['JOB_ID'] = '12345678'
    os.environ['MOX_USE_NPU'] = '1'
    os.environ['MOX_USE_TDT'] = '1'
    os.environ['MOX_USE_TF_ESTIMATOR'] = '0'
    os.environ['HEARTBEAT'] = '1'
    os.environ['CONTINUE_TRAIN'] = 'true'
    os.environ['LOG_DIR'] = './log'
    os.environ['ASCEND_GLOBAL_EVENT_LEVEL'] = '0'
    os.environ['ASCEND_GLOBAL_EVENT_ENABLE'] = '0'
    os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '3'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if device_id is not None:
        os.environ['DEVICE_ID'] = str(device_id)
        os.environ['ASCEND_DEVICE_ID'] = str(device_id)
    if rank_id is not None:
        os.environ['RANK_ID'] = str(rank_id)
    if rank_size is not None:
        os.environ['RANK_SIZE'] = str(rank_size)


class World:
    """A class to keep all the cluster information.

    This class controls how the distributed training and inference is organized,
    when running on multi-device.

    Args:
        root_rank_id: int, the root node of the cluster.

    Properties:
        is_initialized: a boolean flag to indicate whether the cluster information
            is intialized.
        device_type: the type of the devices in the cluster.
        device_id: the physical index of the device used.
        rank_size: the number of the devices used in the cluster.
        rank_id: the index of the device used in the cluster, ranged in [0, rank_size).
        is_root_rank: a boolean value to indicate that the device is regarded as the
            root node. Only root node will print messages and save ckpt during training.
    """
    def __init__(self, root_rank_id=0):
        self.root_rank_id = root_rank_id
        self._device_id = None
        self._rank_id = None
        self._rank_size = None
        self._device_type = None

        self._initialized = False

    def initialize(self, device_type, device_id=None,
                   rank_id=None, rank_size=None, setup_npu_env=True):
        """Initialize cluster information by environment variables or the input.
        """
        if device_id is None or rank_id is None or rank_size is None:
            self.init_by_environ()
        else:
            self._device_id = int(device_id)
            self._rank_id = int(rank_id)
            self._rank_size = int(rank_size)

            # initialize some other env
            if setup_npu_env:
                _setup_npu_env(remove_kernel_meta=True,
                               device_id=device_id,
                               rank_id=rank_id,
                               rank_size=rank_size)

        self._device_type = device_type

        if self._rank_size == 1:
            # Force the single device as root_rank
            self._rank_id = 0

        self._initialized = True

    @property
    def is_initialized(self):
        return self._initialized

    def init_by_environ(self):
        """Initialize cluster using environment variables.
        """
        try:
            self._device_id = int(os.environ['DEVICE_ID'])
        except KeyError:
            logger.error("Environ 'DEVICE_ID' not defined. Use default value DEVICE_ID=0.")
            self._device_id = 0
        except ValueError:
            logger.error(f"Environ 'DEVICE_ID' {os.environ['DEVICE_ID']} cannot converted to int. "
                         "Use default value DEVICE_ID=0.")
            self._device_id = 0

        try:
            self._rank_id = int(os.environ['RANK_ID'])
        except KeyError:
            logger.error("Environ 'RANK_ID' not defined. Use default value RANK_ID=0.")
            self._rank_id = 0
        except ValueError:
            logger.error(f"Environ 'RANK_ID' {os.environ['RANK_ID']} cannot converted to int. "
                         "Use default value RANK_ID=0.")
            self._rank_id = 0

        try:
            self._rank_size = int(os.environ['RANK_SIZE'])
        except KeyError:
            logger.error("Environ 'RANK_SIZE' not defined. Use default value RANK_SIZE=1.")
            self._rank_size = 1
        except ValueError:
            logger.error(f"Environ 'RANK_SIZE' {os.environ['RANK_SIZE']} cannot converted to int. "
                         "Use default value RANK_SIZE=1.")
            self._rank_size = 1

        self._initialized = True

    @property
    def device_type(self):
        if self._device_type is None:
            raise WorldUninitializedError('World not initialized.')
        return self._device_type

    @property
    def device_id(self):
        if self._device_id is None:
            raise WorldUninitializedError('World not initialized.')
        return self._device_id

    @property
    def rank_id(self):
        if self._rank_id is None:
            raise WorldUninitializedError('World not initialized.')
        return self._rank_id

    @property
    def rank_size(self):
        if self._rank_size is None:
            raise WorldUninitializedError('World not initialized.')
        return self._rank_size

    @property
    def is_root_rank(self):
        if self._rank_id is None:
            raise WorldUninitializedError('World not initialized.')
        return self._rank_id == self.root_rank_id


# Global instance.
world = World()
