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
from src.utils.file_io import imread
from src.utils.constant import VALID_FILE_EXT


def supported_file_format(filename):
    """ 
    Check whether the image file is supported.

    Args:
        filename: str

    Returns:
        boolean
    """
    ext = filename.split('.')[-1]
    return ext in VALID_FILE_EXT


def gen_pattern(data_dir, meta, split='lq'):
    """ 
    Generate image pattern given metadata and folder structure.

    Args:
        data_dir: str, top dataset folder
        meta: dict, loaded metadata from set file
        split: str, name of data split
    
    Returns:
        str, file pattern of the images
    """
    if meta is None:
        return os.path.join(data_dir, '{}')
    else:
        if meta['prefix']:
            return os.path.join(data_dir,
                                'images',
                                meta['{}_folder'.format(split)],
                                '{}')
        else:
            return os.path.join(data_dir,
                                'images',
                                '{}',
                                meta['{}_folder'.format(split)])


def pad_list(src, head_pad_size, tail_pad_size, mode):
    """ 
    Pad the given list to target length.

    Args:
        src: list[int], indicies of the frames
        head_pad_size: int, length of pad at the head (before `src`)
        tail_pad_size: int, length of pad at the tail (after `src`)
        mode: str, pad mode. Optional in ['reflect', 'replicate']

    Returns:
        list[int], padded indices list which has target size
    """
    num_src = len(src)
    if mode == 'reflect':
        head_pad_value = list(reversed(src[1:]))
        tail_pad_value = list(reversed(src[:-1]))
    elif mode == 'replicate':
        head_pad_value = [src[0]] * head_pad_size
        tail_pad_value = [src[-1]] * tail_pad_size
    else:
        raise NotImplementedError
    src = head_pad_value + src
    src = src[-(num_src + head_pad_size):] + tail_pad_value
    src = src[:num_src+head_pad_size+tail_pad_size]
    return src


def get_consecutive_frame_indices(given_frame_ids, num_frames_required,
                                  max_frames, base_index=0, interval=1,
                                  pad_mode='reflect'):
    """ 
    Get consecutive indices given the center frame index/indices.
    
    There will be padding at the border of the list. Two typical cases
    used in VSR and VFI model:
        Case 1:
            If given ids are like:
                given_frame_ids=[1,2],
                num_frames_required=4,
                pad_mode='replicate'
            should return [1,1,2,3]. [1, 2] are at the center.
        Case 2:
            If given ids are like:
                given_frame_ids=3,
                num_frames_required=5,
                interval=2,
                pad_mode='reflect'
            should return [3,1,3,5,7]. [3] lies at the center.

    Args:
        given_frame_ids: int or list[int], the center frame indices.
        num_frames_required: int, number of the frames required.
        max_frames: int, the total number of the frames in the dataset.
        base_index: int, the base index of the frame. Default is 0.
        iterval: int, the frame iterval. Default is 1.
        pad_mode: str, the pad method if on the border. 
            Optional in ['reflect', 'replicate']

    Returns:
        list[int], length equals to num_frames_required
    """

    # Find the cosecutive frame indices
    if isinstance(given_frame_ids, (list, tuple)):
        # Currently only supported in vfi 
        assert num_frames_required % 2 == 0, \
            f'{len(given_frame_ids)} frame ids are given. ' \
            f'The required number of frames should be even.'
        num_extra_frames = num_frames_required - len(given_frame_ids)
        min_id = min(given_frame_ids)
        max_id = max(given_frame_ids)

    else:
        # Supported in other tasks
        assert num_frames_required % 2 == 1, \
            f'Only the center frame id is given. ' \
            f'The required number of frames should be odd.'
        min_id = max_id = given_frame_ids
        num_extra_frames = num_frames_required

    assert min_id >= base_index and max_id < max_frames + base_index

    # Obtain the indices within the range [base_index, base_index + max_frames]
    index = []
    left_pad = False
    right_pad = False
    for k in range(min_id - interval*(num_extra_frames//2),
                   max_id+interval*(num_extra_frames//2)+1,
                   interval):
        if base_index > k:
            left_pad = True
            continue
        elif k >= (base_index + max_frames):
            right_pad = True
            continue
        else:
            index.append(k)

    index_len = len(index)
    # When the given frames are on the edge, perform padding,
    if left_pad:
        index = pad_list(index, num_frames_required-index_len, 0, pad_mode)

    if right_pad:
        index = pad_list(index, 0, num_frames_required-index_len, pad_mode)

    return index


def batch_dim_squeeze(dim):
    """
    Squeeze the batch dimension when possible.

    Args:
        dim: list[int], shape of the tensor.
    
    Returns:
        list[int], reduced dimension.
    """
    return dim[1:] if dim[0] == 1 else dim


def load_batch_image(files, target_color_space, as_array=True):
    """
    Load batch images, may return as np.ndarray or just a list.

    Args:
        files: list[str], file paths to read.
        target_color_space: str, color space to which the images are converted.
        as_array: boolean, return as list or np.ndarray.
    
    Returns:
        If as_array is True, return the images are np.ndarray. Else, return
        as the list.
    """

    im = []
    for f in files:
        assert os.path.exists(f), f"{f} not exists."
        _im = imread(f, target_color_space).astype(np.float32)
        im.append(_im)
    return np.array(im) if as_array else im
