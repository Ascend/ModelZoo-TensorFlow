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

import numpy as np
from src.dataloaders.utils import (
    get_consecutive_frame_indices,
    load_batch_image,
    supported_file_format
)
from src.utils.file_io import imread
from src.utils.logger import logger


class _TestDataset:
    """ The base class for training dataset. 
    
    The derived classes should implement these functions:
        _get_item: an indexing like item fetching method.
        data_shape: returns the shape of each item produced by _get_item. 
            Shapes are like (t, h_lq, w_lq, c), (t, h_gt, w_gt, c).
        data_dtype: returns the tensorflow dtype of each item produced
            by _get_item.
    
    Args:
        data_dir: str, top data directory of the test data.
            Contains frames.
        cfg: yacs node, global configuration.
    """
    def __init__(self, data_dir, cfg):
        """ Initialization of train dataset
        
        """
        self.cfg = cfg
        self.scale = cfg.model.scale
        
        # Record the frame rate value, which will be used when determining 
        # the output filename.
        self.frame_rate = 1
        self.num_lq_frames = cfg.model.num_net_input_frames
        self.color_space = cfg.data.color_space

        # By default, all the frames will be infered.
        if len(cfg.data.inference.subset_range) == 2:
            # If given the frame index range to infer
            min_index = cfg.data.inference.subset_range[0]
            max_index = cfg.data.inference.subset_range[1]

            # Frame indices between [min_index, max_index] will be infered 
            logger.info(f'Inference range {min_index}, {max_index} in {data_dir}.')
            is_subset = lambda x: min_index <= x <= max_index
        elif cfg.data.inference.subset_list:
            # If given the list of indices to infer
            logger.info(f'Inference list {cfg.data.inference.subset_list} in {data_dir}.')
            is_subset = lambda x: x in cfg.data.inference.subset_list
        elif len(cfg.data.inference.subset_range) > 0:
            # This is an invalid setting
            logger.error(f'cfg.data.inference.subset_range should of length 2, '
                         f'[min_index, max_index], '
                         f'but is given {cfg.data.inference.subset_range}. '
                         f'All the images will be inferred.')
            is_subset = lambda x: True
        else:
            # This comp_fn will not be used, since `always_infer` is True
            is_subset = lambda x: True
            logger.info(f'Inference all images in {data_dir}.')

        def traverse_folder(dir):
            file_list = list(
                            sorted(
                                filter(
                                    lambda x: supported_file_format(x),
                                    os.listdir(dir)
                                )
                            )
                        )
            num_frames = len(file_list)
            base_index = int(file_list[0].split('.')[0])
            ext = file_list[0].split('.')[1]
            file_meta = [dict(source_folder=dir,
                              filename=f,
                              num_frames=num_frames,
                              base_index=base_index,
                              ext=ext)
                         for f in file_list 
                            if is_subset(int(f.split('.')[0]))]
            return file_meta

        # Reseved value for nested fold structure. 
        # Will be used when save the results.
        self._clipname = '' 

        self.sample_list = traverse_folder(data_dir)
        self.num_clips = 1

        if len(self.sample_list) == 0:
            raise FileNotFoundError(f'Found no files in {data_dir}')

        # Preload one lq sample to obtain the input shape
        if self.sample_list:
            # Record the ext. For VSR\Denoise\VFI tasks, the output will follow 
            # the input ext. For HDR, the input ext could be 'png', the output
            # ext will be 'exr'.
            self.ext = self.sample_list[0]['ext']
            self.output_ext = self.ext
    
            center_frame_meta = self.sample_list[0]
            filename = center_frame_meta['filename']
            center_frame_index = int(filename[:-4])
            num_digits = len(filename[:-4])
            lq_path = center_frame_meta['source_folder']
            lq_file = os.path.join(
                lq_path, 
                f'{center_frame_index:0{num_digits}d}.{self.ext}')
            im = imread(lq_file)
            self.lq_size = im.shape[:2]

            self.base_index = center_frame_meta['base_index']
        
    @property
    def raw_image_size(self):
        """ 
        Returns the raw input image size (h, w).

        Returns:
            tuple, the size of the input image
        """
        return self.lq_size

    @property
    def expect_output_file_ext(self):
        """ 
        Returns the expected output file ext. For tasks other than HDR, 
        the output ext is the same with the input ext. For HDR, the output 
        will be 'exr'.

        Returns:
            str, file extension.
        """
        return self.output_ext
    
    @property
    def expect_output_resolution(self):
        """ Returns the expected output size (h, w).

        Returns:
            tuple, the size of the output image
        """
        return (self.scale * self.lq_size[0], self.scale * self.lq_size[1])

    def __len__(self):
        """ 
        Returns the number of samples to infer. When used in multi-device 
        inference.

        Returns:
            int, the number of samples to infer.
        """
        return len(self.sample_list)

    def __getitem__(self, item):
        # No clue on the real input size, and thus does not require 
        # shape checking.
        return self._get_item(item)

    def set_clip_name(self, clipname):
        """ 
        Set the clip name for this dataset.

        Args:
            clipname: str
        """
        self._clipname = clipname

    def _shard_segment(self, rank_size, rank_id):
        """ 
        Shard the data into ${rank_size} segments. For example,
        case 1:
            num_samples = 10, 
            rank_size = 3, 
            shard: [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]], 
            shard len: [4, 3, 3]
        case 2:
            num_samples = 10, 
            rank_size = 2, 
            shard: [[0, 1, 2, 3, 4]. [5, 6, 7, 8, 9]], 
            shard len: [5, 5]
        Args:
            rank_size: int
            rank_id: int, [0, rank_size)
        """
        num_samples = len(self.sample_list)
        res = num_samples % rank_size
        shard_size_base = int(num_samples // rank_size)

        if rank_id < res:
            start_idx = (shard_size_base + 1) * rank_id
            end_idx = start_idx + shard_size_base + 1
        else:
            start_idx = (shard_size_base + 1) * res + \
                        shard_size_base * (rank_id - res)
            end_idx = start_idx + shard_size_base

        if rank_size == rank_id + 1:
            end_idx = num_samples
        logger.info(f'Data shard {start_idx} - {end_idx - 1} (total {num_samples})', 
                    force=True)

        self.sample_list = list(self.sample_list[start_idx:end_idx])

    def _shard_interleave(self, rank_size, rank_id):
        """ 
        Shard the data into ${rank_size} interlaced segments. For example,
        case 1:
            num_samples = 10, 
            rank_size = 3, 
            shard: [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]], 
            shard len: [4, 3, 3]
        case 2:
            num_samples = 10, 
            rank_size = 2, 
            shard: [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]], 
            shard len: [5, 5]
        Args:
            rank_size: int
            rank_id: int, [0, rank_size)
        """
        start_idx = rank_id
        self.sample_list = list(self.sample_list[start_idx::rank_size])

    def shard(self, rank_size, rank_id, segment=True):
        """ 
        Shard the sample list according to the rank_size and rank_id

        Args:
            rank_size: int
            rank_id: int, [0, rank_size)
            segment: boolean, whether to shard into consecutive segments 
                or interlaced segments. Default 'True'
        """ 
        if segment:
            self._shard_segment(rank_size, rank_id)
        else:
            self._shard_interleave(rank_size, rank_id)

    def _get_item(self, item):
        raise NotImplementedError


class VSRTestDataset(_TestDataset):
    """ 
    Test dataset for VSR task.
    """
    def _get_item(self, index):
        center_frame_meta = self.sample_list[index]
        folder = center_frame_meta['source_folder']
        filename = center_frame_meta['filename']
        num_frames = center_frame_meta['num_frames']
        base_index = center_frame_meta['base_index']
        center_frame_index = int(filename.split('.')[0])
        num_digits = len(filename.split('.')[0])

        lq_indices = get_consecutive_frame_indices(
            center_frame_index,
            self.num_lq_frames, 
            num_frames,
            base_index, interval=1, 
            pad_mode='reflect'
            )
        lq_files = [os.path.join(folder, f'{ind:0{num_digits}d}.{self.ext}') 
                    for ind in lq_indices]
        lq = load_batch_image(lq_files, target_color_space=self.color_space)

        if self.cfg.data.normalized and not (self.ext == 'exr'):
            lq = np.clip(lq / 255., 0., 1.)

        # Record the center frame id, which will be used when outputing the results.
        center_frame_name = \
            f'{lq_indices[self.num_lq_frames//2]:0{num_digits}d}.{self.output_ext}'

        # If self._clipname is not empty, i.e., there exist several folders 
        # in the source lq folder
        if self._clipname != '':
            center_frame_name = os.path.join(self._clipname, center_frame_name)

        return dict(output_file=center_frame_name, lq=lq)


class DenoiseTestDataset(VSRTestDataset):
    """ 
    Test dataset for Denoise task.
    """
    def __init__(self, data_dir, cfg):
        super().__init__(data_dir, cfg)
        # Note that the output is the same size as the input in denoise.
        self.scale = 1
        self.frame_rate = 1


# DummyTestDataset for debug
class DummyTestDataset(_TestDataset):
    """ 
    Dummy test daset for debugging.
    """
    def __init__(self, data_dir, cfg):  # pylint: disable=super-init-not-called
        h = cfg.data.inference.input_size[0] + \
            (cfg.data.inference.patch_pad_size*2 
            if cfg.data.inference.eval_using_patch else 0)

        w = cfg.data.inference.input_size[1] + \
            (cfg.data.inference.patch_pad_size*2 
            if cfg.data.inference.eval_using_patch else 0)

        c = 1 if (cfg.data.color_space=='gray') else 3
        shape_lq = (cfg.model.num_net_input_frames, h, w, c)
        self.sample_list = [np.zeros(shape_lq).astype(np.float32)] * 100
        logger.info(f'Using dummy test dataset with {len(self.sample_list)} '
                    f'element (for debug only). with sizeof {shape_lq}')
        self.lq_size = (h, w)

    def _get_item(self, index):
        lq = self.sample_list[index]
        return dict(output_file='dummy.png', lq=lq)


class VFITestDataset(VSRTestDataset):
    """ 
    Test dataset for VFI task.

    The total number of output frames will be:
        `self.num_sample_list * self.frame_rate`
    where `self.num_sample_list + self.frame_rate - 1` frames are directly 
    copied from the input, and `(self.num_sample_list - 1) * (self.frame_rate - 1)` 
    frames are interpolated.
    
    Generally, if a model requires `M` input frames, and output `N*(self.frame_rate-1)` 
    frames each batch, then we set `self.num_lq_frames=M`, `self.num_interp_frames=N`, 
    and the num of key frames equals to `N+1`.

    We must be aware that not all input frames will be inserted with interpolated 
    frames. A model may require 4 input source frames [A, B, C, D], and interpolate 
    only 1 frame between B and C. The number of required input frames is indicated 
    by the `num_lq_frames`. Meanwhile in this case, only the center frames B and C
    are `key frames` while A and D are just auxiliary information frames. The number 
    of `key frames` are given by `self.num_interp_frames + 1` with the assumption
    that the key frames only lie in the center of the input frames. 
    """
    def __init__(self, data_dir, cfg):
        super().__init__(data_dir, cfg)
        # The frame rate is given by the model configuration.
        self.frame_rate = cfg.model.frame_rate
        # The number of the output frames in each batch, 
        # **not multiplying the frame_rate**.
        self.num_interp_frames = cfg.model.num_net_output_frames

        self.num_final_digits = int(
            np.ceil(
                np.log10(self.frame_rate * len(self.sample_list))))
        self.scale = 1

    def __len__(self):
        return len(self.sample_list) - self.num_interp_frames

    def _get_item(self, index):
        # Get the initial key frame metadata
        key_frame_ids = []
        frame_meta = self.sample_list[index]
        lq_path = frame_meta['source_folder']
        filename = frame_meta['filename']
        num_total_frames = frame_meta['num_frames']
        base_index = frame_meta['base_index']
        num_digits = len(filename.split('.')[0])
        start_frame_index = int(filename.split('.')[0])

        # The next self.num_interp_frames+1 frames are key frames
        key_frame_ids.append(start_frame_index)
        for i in range(start_frame_index+1, start_frame_index+self.num_interp_frames+1):
            key_frame_ids.append(i)
        num_final_digits = max(self.num_final_digits, num_digits)

        # Assuming the key frames are in the center of input frames, 
        # get the auxiliary frames
        lq_indices = get_consecutive_frame_indices(
            key_frame_ids, 
            self.num_lq_frames,
            num_total_frames, base_index,
            interval=1, 
            pad_mode='replicate'
            )

        lq_files = [os.path.join(lq_path, f'{ind:0{num_digits}d}.{self.ext}') 
                    for ind in lq_indices]
        lq = load_batch_image(lq_files, target_color_space=self.color_space)
        if self.cfg.data.normalized:
            lq = np.clip(lq / 255., 0., 1.)

        def _format_output_filename(frame_id, _num_digits, ext):
            output_file = f'{frame_id:0{_num_digits}d}.{ext}'
            if self._clipname != '':
                # Format the output file with `${clip}/00000.png` like pattern
                output_file = os.path.join(self._clipname, output_file)
            return output_file

        # Prepare input copies for VFI output.
        # Record both the source-target filename, as well as the key frames data, 
        # in the dict `input_file_copy`:
        #    key: target_output_file
        #    value: [source_input_file, target_output_data]
        # One can use the copy the source_input_file or write out the data to 
        # target_file.
        input_file_copy = dict()
        output_files = []
        for i, k_id in enumerate(key_frame_ids[:-1]):  # leave the last key frame
            source_file = os.path.join(lq_path, f'{k_id:0{num_digits}d}.{self.ext}')

            index_in_indices = lq_indices.index(k_id)
            data = lq[index_in_indices]

            new_frame_id = base_index + (k_id - base_index) * self.frame_rate
            target_file = _format_output_filename(new_frame_id, 
                                                  num_final_digits, 
                                                  self.output_ext)

            input_file_copy[target_file] = [source_file, data]
            output_files.extend([_format_output_filename(new_frame_id+j+1, 
                                                         num_final_digits, 
                                                         self.output_ext)
                                 for j in range(self.frame_rate-1)])

        if index == len(self) - 1:   # copy the last key frame only when reaching the end
            source_file = os.path.join(lq_path, 
                                       f'{key_frame_ids[-1]:0{num_digits}d}.{self.ext}')

            index_in_indices = lq_indices.index(key_frame_ids[-1])
            data = lq[index_in_indices]

            new_frame_id = base_index + (key_frame_ids[-1] - base_index) * self.frame_rate
            target_file = _format_output_filename(
                new_frame_id, 
                num_final_digits, 
                self.output_ext)
            input_file_copy[target_file] = [source_file, data]

            # copy the final frame
            for i in range(self.frame_rate-1):
                new_frame_id = new_frame_id + 1
                target_file = _format_output_filename(
                    new_frame_id, 
                    num_final_digits, 
                    self.output_ext)
                input_file_copy[target_file] = [source_file, data]

        if len(output_files) == 1:
            output_files = output_files[0]

        return dict(output_file=output_files, 
                    lq=lq, 
                    input_copies=input_file_copy)

    def _shard_segment(self, rank_size, rank_id):
        num_samples = len(self.sample_list)
        res = (num_samples - 1) % rank_size
        shard_size_base = int((num_samples - 1) // rank_size)

        if rank_id < res:
            start_idx = (shard_size_base + 1) * rank_id
            # enclose the last as the key frame
            end_idx = start_idx + (shard_size_base + 1) + 1   
        else:
            start_idx = (shard_size_base + 1) * res + shard_size_base * (rank_id - res)
            # enclose the last as the key frame
            end_idx = start_idx + shard_size_base + 1         

        if rank_size == rank_id + 1:
            end_idx = num_samples
        logger.info(f'Data shard {start_idx} - {end_idx - 1} (total {num_samples})', force=True)
        self.sample_list = list(self.sample_list[start_idx:end_idx])
        self.num_samples_shard = len(self.sample_list)
        self.shard_flag = True


class HDRTestDataset(VSRTestDataset):
    """ 
    Test dataset for HDR task. The output ext should be set in 
    'cfg.data.extension'.
    """
    def __init__(self, data_dir, cfg):
        super().__init__(data_dir, cfg)
        self.output_ext = cfg.data.extension   # regardless of the input ext
        self.frame_rate = 1
        self.scale = 1


class ComposedTestDataset(_TestDataset):
    """ Test dataset for a test directory with multiple clips.
    """
    def __init__(self):  # pylint: disable=super-init-not-called
        self._datasets = None
        self.num_samples_list = []

    @staticmethod
    def from_datasets(*datasets):
        """ 
        Construct composed dataset from a collection of sub dataset class.

        Args:
            *datasets: list of test datasets.
        """
        cls = ComposedTestDataset()
        cls._datasets = list(datasets)
        cls.num_samples_list = [len(d) for d in cls._datasets]
        return

    @staticmethod
    def from_datadir(subcls, data_dir, clip_list, cfg):
        """ 
        Construct composed dataset from a collection of dataset folder.

        Args:
            subcls: class type, task class
            data_dir: str, top data folder
            clip_list: list of str, clips in the data_dir
            cfg: yacs node
        """
        cls = ComposedTestDataset()
        datasets = []
        for clip_name in clip_list:
            data_clip = os.path.join(data_dir, clip_name)
            sub_datasets = subcls(data_clip, cfg)
            sub_datasets.set_clip_name(clip_name)
            datasets.append(sub_datasets)
        cls._datasets = datasets
        cls.num_samples_list = [len(d) for d in cls._datasets]
        return cls

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
        data = self._datasets[dataset_id][index]
        return data

    @property
    def expect_task_output_meta_info(self):
        return self._datasets[0].expect_task_output_meta_info

    @property
    def raw_image_size(self):
        return self._datasets[0].raw_image_size

    @property
    def expect_output_file_ext(self):
        return self._datasets[0].expect_output_file_ext

    def __len__(self):
        return np.sum(self.num_samples_list)

    def shard(self, rank_size, rank_id):
        # Shard not supported for now.
        raise NotImplementedError('Composed dataset not support data shard.')


class MixtureTestDataset(_TestDataset):
    """ 
    Test dataset for a test directory with multiple dataset folder.
    """
    def __init__(self):  # pylint: disable=super-init-not-called
        self._datasets = None
        self.num_samples_list = [0]

    @staticmethod
    def from_datasets(*datasets):
        """ 
        Construct mixture dataset from a list of test datasets.

        Args:
            *datasets: list of test datasets instances. Should be return data 
                terms with the same dtype and shape.

        Returns:
            a MixtureDatasets instance
        """
        cls = MixtureTestDataset()
        cls._datasets = list(datasets)
        cls.num_samples_list = [len(d) for d in cls._datasets]
        return

    @staticmethod
    def from_datadir(subcls, data_dir_list, cfg):
        """ 
        Construct mixture dataset from a list of data directories.

        Args:
            subcls: test dataset class type
            data_dir_list: list(str), each is a top directory of a dataset.
                Should be return data terms with the same dtype and shape.
            cfg: yacs Node, global configuration

        Returns:
            a MixtureDatasets instance
        """
        cls = MixtureTestDataset()
        datasets = []
        for data_dir in data_dir_list:
            files = os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, files[0])):
                sub_datasets = ComposedTestDataset.from_datadir(
                    subcls, 
                    data_dir, 
                    files, 
                    cfg
                    )
            else:
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
        dataset_id = 0
        for dataset_id, num in enumerate(self.num_samples_list):
            if index - num < 0:
                break
            index -= num
        return dataset_id, index

    def _get_item(self, item):
        dataset_id, index = self.get_datasets(item)
        return self._datasets[dataset_id][index]

    @property
    def output_meta_info(self):
        return self._datasets[0].output_meta_info

    @property
    def raw_image_size(self):
        return self._datasets[0].raw_image_size

    def __len__(self):
        return np.sum(self.num_samples_list)
