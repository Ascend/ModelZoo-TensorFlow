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

import tensorflow as tf


class TfDataloader:
    """A wrapper of the tensorflow Dataset class.

    This class aims to make the dataset construction more convenient. Users 
    are only required to implement the dataset class and return the specific 
    information, then TfDataloader will wrap the dataset class and load the 
    data, similar to Pytorch dataset and dataloader.

    Args:
            dataset: Dataset class. See `src.dataloaders.train_dataset`
            batch_size: int
            drop_remainder: boolean, whether drop the last remainder terms. 
                Necessary on Ascend NPU. Default is True.
            distributed: boolean, whether to use distribute dataloader.
            shuffle: boolean, whether to shuffle the dataset.
            repeat: boolean, whether to repeat the dataset (usually in training).
            device: str, hardware used for accelerating. Optional in ['npu', 'cpu']
    """
    def __init__(self, dataset, batch_size=2, drop_remainder=True, 
                 distributed=False, shuffle=True, repeat=True, device='npu',
                 _world=None):
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.distributed = distributed
        self.device = device
        self.dataset = dataset
        self.shuffle = shuffle
        self.repeat = repeat
        self.world = _world
        self.sample_indices = list(range(len(self.dataset)))

        self.build_iterator()

    def get_item(self, index):
        """ Tensorflow wrapper of the _get_item method of Dataset class
        
        Args:
            index: int, called by tensorflow.data.Dataset.map function.
        """
        # The dtype and shape are defined by the dataset. Otherwise, 
        # tf does not know the shape.
        data = tf.numpy_function(lambda x: self.dataset[x], 
                                 [index], 
                                 self.dataset.data_dtype)
        
        for d, shape in zip(data, self.dataset.data_shape):
            d.set_shape(tuple(shape))
        return data

    def build_iterator(self):
        """ Build dataloader iterator """
        video_dataset = tf.data.Dataset.from_tensor_slices(self.sample_indices)

        if self.shuffle:
            video_dataset = video_dataset.shuffle(len(self.dataset))

        video_dataset = video_dataset.map(self.get_item,
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        video_dataset = video_dataset.batch(self.batch_size,
                                            drop_remainder=self.drop_remainder)

        if self.repeat:
            video_dataset = video_dataset.repeat()

        if self.distributed:
            video_dataset = video_dataset.shard(self.world.rank_size,
                                                self.world.rank_id)

        video_dataset = video_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        iterator = video_dataset.make_one_shot_iterator()

        # tensorflow will wrap all the numpy.ndarray as tensors
        self.batch_list = iterator.get_next()


def build_dataloader(dataset, batch_size=2, drop_remainder=True, distributed=False, 
                     shuffle=True, repeat=True, device='npu'):
    """ 
    Build dataloader given the dataset.

    Args:
        dataset: Dataset class. See `src.dataloaders.train_dataset`
        batch_size: int
        drop_remainder: boolean, whether drop the last remainder terms. 
            Necessary on Ascend NPU. Default is True.
        distributed: boolean, whether to use distribute dataloader.
        shuffle: boolean, whether to shuffle the dataset.
        repeat: boolean, whether to repeat the dataset (usually in training).
        device: str, hardware used for accelerating. Optional in ['npu', 'cpu']
    
    Returns:
        list[tensorlfow tensor]
    """
    dataloader = TfDataloader(dataset, batch_size, drop_remainder,
                              distributed, shuffle, repeat, device)
    return dataloader.batch_list
