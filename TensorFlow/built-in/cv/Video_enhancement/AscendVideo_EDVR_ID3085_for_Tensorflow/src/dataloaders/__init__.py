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

import os

from src.dataloaders.dataloader import TfDataloader
from src.utils.exceptions import *
from src.utils.logger import logger
from src.utils.world import world


def build_train_dataloader(config, _world=None):
    """ 
    Build train dataloader given config.

    Args:
        config: yacs node, configuration.
    
    Returns:
        generator of list[tensor].
    """

    # Import the dataset classes only when needed to avoid import error
    from src.dataloaders.train_dataset import (
        OfflineTrainDataset,
        OnlineTrainDataset,
        DummyTrainDataset,
        MixtureDatasets,
    )

    # Support multi-dataset whose path are concated with ':' 
    # in cfg.data.data_dir
    data_dir_list = config.data.data_dir.split(':')
    task = config.task
    online_degradation_mode = config.data.train.degradation.online
    distributed = config.env.rank_size > 1
    device = config.env.device
    batchsize = config.data.train.batch_size

    world_inst = _world or world
    # _world should be initialized
    if not world_inst.is_initialized:
        raise WorldUninitializedError('World not initialized.')

    if config.debug_mode:
        dataset_cls = DummyTrainDataset
    elif online_degradation_mode:
        dataset_cls = OnlineTrainDataset
    else:
        dataset_cls = OfflineTrainDataset

    if len(data_dir_list) > 1:
        dataset = MixtureDatasets.from_datadir(
            dataset_cls, data_dir_list, cfg=config)
    else:
        dataset = dataset_cls(data_dir=data_dir_list[0], cfg=config)
    
    dataloader = TfDataloader(dataset, batchsize,
                              distributed=distributed, 
                              device=device)
    return dataloader.batch_list


def build_test_dataloader(config, _world=None):
    """ 
    Build inference dataloader given config.

    Args:
        config: yacs node, configuration.
    
    Returns:
        dict, contains the data term.
    """
    from src.dataloaders.test_dataset import (
        VSRTestDataset,
        DenoiseTestDataset,
        VFITestDataset,
        HDRTestDataset,
        DummyTestDataset,
        MixtureTestDataset,
        ComposedTestDataset
    )

    TASK_MAP_TO_DATASET = {
        'vsr': VSRTestDataset,
        'denoise': DenoiseTestDataset,
        'vfi': VFITestDataset,
        'hdr': HDRTestDataset,
        'face': DenoiseTestDataset,
    }

    data_dir_list = config.data.data_dir.split(':')
    distributed = config.env.rank_size > 1
    task = config.task
    world_inst = _world or world
    # _world should be initialized
    if not world_inst.is_initialized:
        raise WorldUninitializedError('World not initialized.')

    assert task in TASK_MAP_TO_DATASET

    if config.debug_mode:
        dataset_cls = DummyTestDataset
    else:
        dataset_cls = TASK_MAP_TO_DATASET[task]
    
    if config.debug_mode:
        dataset = dataset_cls(data_dir=data_dir_list[0], cfg=config)
    elif len(data_dir_list) > 1:
        # For multi-dataset
        dataset = MixtureTestDataset.from_datadir(
            dataset_cls, data_dir_list, cfg=config)
    else:
        files = os.listdir(data_dir_list[0])
        if os.path.isdir(os.path.join(data_dir_list[0], files[0])):
            # For dataset with multiple clips
            dataset = ComposedTestDataset.from_datadir(
                dataset_cls, data_dir_list[0], files, cfg=config
            )
        else:
            # Foe a single dataset with frames
            dataset = dataset_cls(data_dir=data_dir_list[0], cfg=config)

    # Manually shard the dataset to inference on multiple devices.
    if distributed:
        dataset.shard(world_inst.rank_size, world_inst.rank_id)
    return dataset


def build_dataloader(cfg, **kwargs):
    """
    Build dataloader given scenario and configurations.
    
    Args:
        cfg: yacs node, global configuration.
        **kwargs: argument dicts.
    """
    if cfg.mode in ['train', 'eval']:
        dataloader = build_train_dataloader(cfg)
    elif cfg.mode in 'inference':
        dataloader = build_test_dataloader(cfg, **kwargs)
    elif cfg.mode == 'freeze':
        dataloader = None
    else:
        raise KeyError
    return dataloader


__all__ = ['build_train_dataloader', 'build_test_dataloader', 'build_dataloader']
