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

import json
import os
import random

import cv2
import numpy as np

import tensorflow as tf
import yaml
from tqdm import tqdm

from src.dataloaders.utils import (
    gen_pattern,
    pad_list,
    get_consecutive_frame_indices,
    batch_dim_squeeze,
    load_batch_image,
    supported_file_format
)

from src.utils.degradation import Degradation
from src.utils.transform import Compose
from src.utils.file_io import imread
from src.utils.logger import logger


class _TrainDataset:
    """The base class for training dataset. 
    
    The derived classes should implement these functions:
        _get_item: an indexing like item fetching method.
        data_shape: returns the shape of each item produced by _get_item. 
            Shapes are like (t, h_lq, w_lq, c), (t, h_gt, w_gt, c).
        data_dtype: returns the tensorflow dtype of each item produced
            by _get_item.

    Supported data folder structure:
        1. All train datasets class support this structure
            data
            `-- reds
                |-- images
                |   |-- lq
                |   |   |-- 000
                |   |   |   |-- 00000000.png
                |   |   |   |-- 00000001.png
                |   |   |   |-- 00000002.png
                |   |   |   |-- ...
                |   |   |   `-- 00000099.png
                |   |   `-- 001
                |   |       |-- 00000000.png
                |   |       |-- 00000001.png
                |   |       |-- 00000002.png
                |   |       |-- ...
                |   |       `-- 00000099.png
                |   `-- gt
                |       |-- 000
                |       |   |-- 00000000.png
                |       |   |-- 00000001.png
                |       |   |-- 00000002.png
                |       |   |-- ...
                |       |   `-- 00000099.png
                |       `-- 001
                |           |-- 00000000.png
                |           |-- 00000001.png
                |           |-- 00000002.png
                |           |-- ...
                |           `-- 00000099.png
                `-- sets
                    |-- train.json
                    `-- val.json
        2. Online datasets class support this:
            reds_gt
                |-- 000
                |   |-- 00000000.png
                |   |-- 00000001.png
                |   |-- 00000002.png
                |   |-- ...
                |   `-- 00000099.png
                `-- 001
                    |-- 00000000.png
                    |-- 00000001.png
                    |-- 00000002.png
                    |-- ...
                    `-- 00000099.png
    
    Args:
        data_dir: str, top data directory of the train dataset.
            Should include the `images` and `sets` sub-folders.
        cfg: yacs node, global configuration.
    """
    def __init__(self, data_dir, cfg):
        """ 
        Initialization of train dataset.
        
        """
        self.cfg = cfg
        self.num_lq_frames = cfg.data.num_data_lq_frames
        self.num_gt_frames = cfg.data.num_data_gt_frames
        self.interval_list = cfg.data.train.augmentation.interval_list
        self.augment = cfg.data.train.augmentation.apply
        self.scale = cfg.model.scale
        self.set_file = os.path.join(data_dir, cfg.data.train.set_file)
        self.crop_size = cfg.data.train.input_size
        self.color_space = cfg.data.color_space
        
        # TODO: change the data structure
        clip_list = self.parse_datafolder_structure(data_dir, self.set_file)
        self.sample_list = []

        # Store all the frame metadata of all the clips in the folder
        for vid in clip_list:
            in_path = self.gt_path_pattern.format(vid)
            file_list = list(
                sorted(
                    filter(
                        lambda x: supported_file_format(x),
                        os.listdir(in_path)
                    )
                )
            )
            num_frames = len(file_list)
            base_index = int(file_list[0].split('.')[0])
            ext = file_list[0].split('.')[1]

            # File metadata includes `clip` name, file name itself, total number
            # of frames in the clip, the starting id of the frames, and file
            # extension (png, exr, .etc).
            file_meta = [dict(clip=vid,
                              filename=f,
                              num_frames=num_frames,
                              base_index=base_index,
                              ext=ext)
                         for f in file_list]

            self.sample_list.extend(file_meta)

        if len(self.sample_list) == 0:
            raise FileNotFoundError(f'Found no files in {data_dir}')
        else:
            logger.info(f'Found {len(self.sample_list)} files in {data_dir}')

    def parse_datafolder_structure(self, data_dir, set_file):
        """ 
        Parse the default dataset structure.

        Args:
            data_dir: str, the top folder of the dataset.
            set_file: str, the json indicating the clips (both lq and gt)

        Returns:
            list of str, the names of the clips (in lq and corresponding gt)
        """
        if not self.cfg.data.train.degradation.online:
            self.lq_path_pattern = os.path.join(data_dir, 'lq', '{}')
        self.gt_path_pattern = os.path.join(data_dir, 'gt', '{}')

        if os.path.exists(set_file):
            clip_list = []
            with open(set_file, 'r') as f:
                for line in f.readlines():
                    clip_list.append(line.strip())
        else:
            clip_list = sorted(os.listdir(os.path.join(data_dir, 'gt')))

        return clip_list

    def __len__(self):
        """ 
        Total number of samples for training.

        Returns:
            int, the number of training samples
        """
        return len(self.sample_list)

    def check_shape(self, data):
        """ 
        Called after '_get_item' to check whether the real shapes are the
        same with the expected in 'data_shape'.

        Args:
            data:

        Returns:

        """
        for index, d, shape in zip(range(len(data)), data, self.data_shape):
            assert tuple(d.shape) == tuple(shape), \
                f'Expect return data at pos {index} to have shape {shape}, ' \
                f'but got {d.shape}'

    @property
    def data_shape(self):
        """ 
        Returns the shape of each item produced by _get_item. Shapes
        are like (t, h_lq, w_lq, c), (t, h_gt, w_gt, c).

        Returns:
            tuple of shapes, each can be like (t, h, w, c)
        """
        raise NotImplementedError

    @property
    def data_dtype(self):
        """ 
        Returns the tensorflow dtype of each item produced by _get_item.

        Returns:
            tuple of data types, each can be like tf.float32
        """
        raise NotImplementedError

    def __getitem__(self, index):
        data = self._get_item(index)

        self.check_shape(data)

        return data

    def _get_item(self, index):
        """ An indexing-like item fetching method

        Args:
            index: int

        Returns:
            tuple of data terms (as numpy.ndarray)
        """
        raise NotImplementedError


# DummyTrainDataset for debug
class DummyTrainDataset(_TrainDataset):
    """ A dummy train dataset for debugging.
    """
    def __init__(self, data_dir, cfg):  # pylint: disable=super-init-not-called
        b = cfg.data.train.batch_size
        h = cfg.data.train.input_size[0]
        w = cfg.data.train.input_size[1]
        c = 1 if (cfg.data.color_space=='gray') else 3
        shape_lq = (cfg.data.num_data_lq_frames, h, w, c)
        self.lq_shape = shape_lq
        self.gt_shape = (cfg.data.num_data_gt_frames, h, w, 3)

        num_samples = 100
        self.sample_list = [np.zeros(self.lq_shape).astype(np.float32)
                            for _ in range(num_samples)]
        self.sample_list_gt = [np.zeros(self.gt_shape).astype(np.float32)
                               for _ in range(num_samples)]

    def _get_item(self, index):
        lq = self.sample_list[index]
        gt = self.sample_list_gt[index]
        return lq, gt

    @property
    def data_dtype(self):
        return tf.float32, tf.float32

    @property
    def data_shape(self):
        return self.lq_shape, self.gt_shape


class OfflineTrainDataset(_TrainDataset):
    """ 
    Offline degradation task training dataset.
    Augmentation is always online.
    """
    def __init__(self, data_dir, cfg):
        super().__init__(data_dir, cfg)
        self.num_channels = 3 if self.color_space != 'gray' else 1

        # Load augmentation options from cfg
        options = yaml.safe_load(cfg.data.train.augmentation.options)
        self.augment_pipeline = Compose.from_cfgs(
            options,
            crop_size=self.crop_size,   # source crop size
            scales=(1, self.scale)      # scale of each crop, corresponds to
                                        #   returned data terms.
        )

    @property
    def data_shape(self):
        h, w = self.crop_size   # this is the input (lq) crop size

        lq_shape = (self.num_lq_frames,
                    h,
                    w,
                    self.num_channels)

        # Squeeze the batch dim if possible. Single image case
        lq_shape = batch_dim_squeeze(lq_shape)

        gt_shape = (self.num_gt_frames,
                    h*self.scale,
                    w*self.scale,
                    self.num_channels)
        gt_shape = batch_dim_squeeze(gt_shape)

        return lq_shape, gt_shape

    @property
    def data_dtype(self):
        return tf.float32, tf.float32

    def _get_item(self, index):
        # Get meta data. We take the `index` frame as the center frame
        center_frame_meta = self.sample_list[index]
        vid = center_frame_meta['clip']
        filename = center_frame_meta['filename']
        num_frames = center_frame_meta['num_frames']
        base_index = center_frame_meta['base_index']
        ext = center_frame_meta['ext']
        center_frame_index = int(filename[:-4])
        num_digits = len(filename[:-4])
        
        # Frames interval augmentation
        if self.augment:
            interval = random.choice(self.interval_list)
        else:
            interval = 1

        # Get the consecutive frame indices
        lq_indices = get_consecutive_frame_indices(
            center_frame_index,
            self.num_lq_frames,
            num_frames,         # total number of frames in the clip
            base_index,
            interval=interval,
            pad_mode='reflect')

        lq_files = [os.path.join(self.lq_path_pattern.format(vid),
                                 f'{ind:0{num_digits}d}.{ext}')
                    for ind in lq_indices]

        gt_indices = get_consecutive_frame_indices(
            center_frame_index,
            self.num_gt_frames,
            num_frames,
            base_index,
            interval=interval,
            pad_mode='reflect')

        gt_files = [os.path.join(self.gt_path_pattern.format(vid),
                                 f'{ind:0{num_digits}d}.{ext}')
                    for ind in gt_indices]

        lq = load_batch_image(lq_files, self.color_space)
        gt = load_batch_image(gt_files, self.color_space)

        if self.augment:
            lq, gt = self.augment_pipeline(lq, gt)

        if self.num_lq_frames == 1 and lq.shape[0] == 1:
            lq = lq[0]
        if self.num_gt_frames == 1 and gt.shape[0] == 1:
            gt = gt[0]

        if self.cfg.data.normalized:
            lq = np.clip(lq / 255., 0, 1)
            gt = np.clip(gt / 255., 0, 1)

        return lq, gt


class OnlineTrainDataset(OfflineTrainDataset):
    """ Online degradation task training dataset.
    """
    def __init__(self, data_dir, cfg):
        super().__init__(data_dir, cfg)
        # Loading degradation from cfg:
        #    add noise, down-sampling, blur, etc.
        options = yaml.safe_load(cfg.data.train.degradation.options)
        assert isinstance(options, dict)

        # TODO: remove preset degradation
        self.degradation_pipeline = get_degradation_model(
            scale=self.scale,
            version=cfg.data.train.degradation.online_version)

        options = yaml.safe_load(cfg.data.train.augmentation.options)
        assert isinstance(options, dict)

        # Loading augmentation from cfg:
        #    random crop, random flip, random interval, etc.
        self.augment_pipeline = transforms.Compose.from_cfgs(
            options,
            crop_size=self.crop_size,
            scales=(self.scale, )
        )

        # Loading gt enhancement from cfg:
        #    usm, etc.
        self.gt_enhancement = cfg.data.train.gt_enhancement
        if cfg.data.gt_enhancement:
            self.gt_enhancement_module = get_degradation_model(
                version='gt_enhancement'
            )

    def load_gt(self, im_files):
        gt_list = []
        for i, _im in enumerate(im_files):
            gt = imread(_im, self.color_space)
            gt_list.append(gt)
        return np.array(gt_list)

    def _get_item(self, index):
        center_frame_meta = self.sample_list[index]
        vid = center_frame_meta['clip']
        filename = center_frame_meta['filename']
        num_frames = center_frame_meta['num_frames']
        base_index = center_frame_meta['base_index']
        ext = center_frame_meta['ext']
        center_frame_index = int(filename[:-4])
        num_digits = len(filename[:-4])

        if self.augment:
            interval = random.choice(self.interval_list)
        else:
            interval = 1

        # Should load num_lq_frames gt images
        gt_indices = get_consecutive_frame_indices(
            center_frame_index,
            self.num_lq_frames,
            num_frames, base_index,
            interval=interval,
            pad_mode='reflect'
        )
        gt_files = [os.path.join(self.gt_path_pattern.format(vid),
                                 f'{ind:0{num_digits}d}.{ext}')
                    for ind in gt_indices]

        gt = self.load_gt(gt_files)
        if self.augment:
            gt = self.augment_pipeline(gt)[0]

        # Do degradation after augmentation to reduce computation.
        lq_list = self.degradation_pipeline.apply_batch(
            np.array(gt),
            allow_quantization=ext != 'exr'
        )

        lq = np.array(lq_list).astype(np.float32)
        gt = gt.astype(np.float32)

        # Select the center num_gt_frames out
        gt = gt[(self.num_lq_frames//2-self.num_gt_frames//2):
                (self.num_lq_frames//2+self.num_gt_frames//2)+1]

        if self.gt_enhancement:
            gt = [self.gt_enhancement_module.apply(_gt) for _gt in gt]
            gt = np.array(gt)

        if self.num_lq_frames == 1:
            lq = lq[0]
        if self.num_gt_frames == 1:
            gt = gt[0]

        if self.cfg.data.normalized:
            lq = np.clip(lq / 255., 0, 1)
            gt = np.clip(gt / 255., 0, 1)
        return lq, gt


class MixtureDatasets(_TrainDataset):
    """ 
    Mixture dataset containing multiple train datasets.
    Could be constructed from a list of folders.
    """
    def __init__(self):  # pylint: disable=super-init-not-called
        self._datasets = None
        self.num_samples_list = []

    @staticmethod
    def from_datasets(*datasets):
        """ 
        Construct mixture dataset from a list of train datasets.

        Args:
            *datasets: list of OfflineTrainDataset or OnlineTrainDataset
                instances. Should be return data terms with the same dtype and
                shape.

        Returns:
            a MixtureDatasets instance
        """
        cls = MixtureDatasets()
        cls._datasets = list(datasets)
        cls.num_samples_list = [len(d) for d in cls._datasets]
        return

    @staticmethod
    def from_datadir(subcls, data_dir_list, cfg):
        """ 
        Construct mixture dataset from a list of data directories.

        Args:
            subcls: OfflineTrainDataset or OnlineTrainDataset type
            data_dir_list: list(str), each is a top directory of a dataset.
                Should be return data terms with the same dtype and shape.
            cfg: yacs Node, global configuration

        Returns:
            a MixtureDatasets instance
        """
        cls = MixtureDatasets()
        datasets = []
        for data_dir in data_dir_list:
            sub_datasets = subcls(data_dir, cfg)
            datasets.append(sub_datasets)
        cls._datasets = list(datasets)
        cls.num_samples_list = [len(d) for d in cls._datasets]
        return cls

    @property
    def data_dtype(self):
        return self._datasets[0].data_dtype

    @property
    def data_shape(self):
        return self._datasets[0].data_shape

    def get_datasets(self, index):
        # Iterate over the datasets to locate the queried sample
        dataset_id = 0
        for dataset_id, num in enumerate(self.num_samples_list):
            if index - num < 0:
                break
            index -= num
        return dataset_id, index

    def _get_item(self, item):
        dataset_id, index = self.get_datasets(item)
        return self._datasets[dataset_id][index]

    def __len__(self):
        return np.sum(self.num_samples_list)
