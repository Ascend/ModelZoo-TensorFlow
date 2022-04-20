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

from src.utils.utils import to_pair

__all__ = [
    'resize', 'depth_to_space', 'space_to_depth', 
    'decimation_up', 'decimation_down'
]


def depth_to_space(x, scale, use_default=False, data_format='NHWC'):
    """Rearrange data from depths to blocks of spatial data.

    Given a tensor of size [N, H, W, C], this operator converts the tensor
    to size [N, H*scale, W*scale, C/(scale*scale)].

    Args:
        x: tensor, which has the shape [N, H, W, C] or [N, C, H, W].
        scale: int, specifying how many blocks the depths is rearrageed. Both
            h and w dimension will be scaled up by this value.
        use_default: boolean, use tensorflow default implementation. If False,
            use a composed operator instead. Default False.
        data_format: str, possible choices in ['NHWC', 'NCHW'].
    
    Returns:
        tensor, which has the shape [N, H*scale, W*scale, C/(scale*scale)].
    """
    if use_default:
        out = tf.nn.depth_to_space(x, scale, data_format=data_format)
    elif data_format == 'NHWC':
        b, h, w, c = x.get_shape().as_list()
        c_scaled = c // (scale**2)
        out = tf.reshape(x, [-1, h, w, scale, scale, c_scaled])
        out = tf.transpose(out, [0, 1, 3, 2, 4, 5])
        out = tf.reshape(out, [-1, h * scale, w * scale, c_scaled])
    elif data_format == 'NCHW':
        b, c, h, w = x.get_shape().as_list()
        c_scaled = c // (scale**2)
        out = tf.reshape(x, [-1, scale, scale, c_scaled, h, w])
        out = tf.transpose(out, [0, 3, 4, 1, 5, 2])
        out = tf.reshape(out, [-1, c_scaled, h * scale, w * scale])
    else:
        raise ValueError(f'Unknown data format `{data_format}`')
    return out


def space_to_depth(x, scale, use_default=False, data_format='NHWC'):
    """Rearrange data from blocks of spatial data to depths.

    Given a tensor of size [N, H, W, C], this operator converts the tensor
    to size [N, H/scale, W/scale, C*(scale*scale)].

    Args:
        x: tensor, which has the shape [N, H, W, C] or [N, C, H, W].
        scale: int, specifying how many blocks the depths is rearrageed. Both
            h and w dimension will be scaled down by this value.
        use_default: boolean, use tensorflow default implementation. If False,
            use a composed operator instead. Default False.
        data_format: str, possible choices in ['NHWC', 'NCHW'].
    
    Returns:
        tensor, which has the shape [N, H/scale, W/scale, C*(scale*scale)].
    """
    if use_default:
        out = tf.nn.space_to_depth(x, scale, data_format=data_format)
    elif data_format == 'NHWC':
        b, h, w, c = x.get_shape().as_list()
        c_scaled = c * (scale**2)
        out = tf.reshape(x, [-1, h//scale, scale, w//scale, scale, c])
        out = tf.transpose(out, [0, 1, 3, 2, 4, 5])
        out = tf.reshape(out, [-1, h//scale, w//scale, c_scaled])
    elif data_format == 'NCHW':
        b, c, h, w = x.get_shape().as_list()
        c_scaled = c * (scale**2)
        out = tf.reshape(x, [-1, c, h//scale, scale, w//scale, scale])
        out = tf.transpose(out, [0, 3, 5, 1, 2, 4])
        out = tf.reshape(out, [-1, c_scaled, h//scale, w//scale])
    else:
        raise ValueError(f'Unknown data format `{data_format}`')
    return out


def resize(x, size, align_corners=False, name=None, half_pixel_centers=False, method='bicubic'):
    """Wrap of tensorflow resize function.

    Args:
        x: tensor, which has the shape [N, H, W, C] or [N, C, H, W].
        size: list[int] of length 2, indicating the target size [H_target, W_target].
        align_corners: boolean, whether to align corners when resizing.
        name: str, the name of the resize operation.
        half_pixel_centers: boolean, whether use the half pixel as the center.
        method: str, resize method. Possible choices in ('bicubic', 'bilinear', 'area')

    Return:
        tensor, the resized version fo x which is of shape [N, H_target, W_target, C].
    """
    if method == 'bicubic':
        upsampling = tf.image.resize_bicubic
    elif method == 'bilinear':
        upsampling = tf.image.resize_bilinear
    elif method == 'area':
        upsampling = tf.image.resize_area
        return upsampling(x, size=size, align_corners=align_corners, name=name)
    else:
        raise ValueError
    return upsampling(x, size=size, align_corners=align_corners, name=name, half_pixel_centers=half_pixel_centers)


def decimation_up(x, scale, data_format='NHWC'):
    """Interpolate the tensor to target scale.
    
    Given a tensor of size [N, H, W, C], this operator converts the tensor
    to size [N, H*scale, W*scale, C]. The interpolateed pixels will be filled 
    with zeros.

    For example, the entries of 2D tensor x is:
    [[1, 2],
     [3, 4]]
    
    When scale=3, the output will be:
    [[1, 0, 0, 2, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [3, 0, 0, 4, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0]]

    Args:
        x: tensor, which has the shape [N, H, W, C] or [N, C, H, W].
        scale: int, specifying the magnification the h and w.
        data_format: str, possible choices in ['NHWC', 'NCHW'].

    Returns:
        tensor, which has the shape [N, H*scale, W*scale, C].
    """
    x_shape = x.get_shape().as_list()

    scale = to_pair(scale, 2)
    sh, sw = scale

    zeros = tf.zeros([*x_shape, sh*sw-1], dtype=x.dtype)
    x_expand = tf.expand_dims(x, -1)
    x_up = tf.concat([x_expand, zeros], axis=-1)
    x_up = tf.reshape(x_up, shape=[*x_shape, sh, sw])
    if data_format == 'NCHW':
        n, c, h, w = x_shape
        x_up = tf.transpose(x_up, (0, 1, 2, 4, 3, 5))
        x_up = tf.reshape(x_up, [n, c, h*sh, w*sw])
    elif data_format == 'NHWC':
        n, h, w, c = x_shape
        x_up = tf.transpose(x_up, (0, 1, 4, 2, 5, 3))
        x_up = tf.reshape(x_up, [n, h*sh, w*sw, c])
    else:
        raise ValueError

    return x_up


def decimation_down(x, scale, data_format='NCHW'):
    """Decimation the tensor with target scale.
    
    Given a tensor of size [N, H, W, C], this operator converts the tensor
    to size [N, H/scale, W/scale, C]. The values remained are the upper-left
    corner value within each block.

    For example, the entries of 2D tensor x is:
    [[ 1,  2,  3,  4],
     [ 5,  6,  7,  8],
     [ 9, 10, 11, 12],
     [13, 14, 15, 16]]
    
    When scale=2, the output will be:
    [[1, 3],
     [9, 11]]

    Args:
        x: tensor, which has the shape [N, H, W, C] or [N, C, H, W].
        scale: int, specifying the down magnification the h and w.
        data_format: str, possible choices in ['NHWC', 'NCHW'].

    Returns:
        tensor, which has the shape [N, H/scale, W/scale, C]
    """
    x_shape = x.get_shape().as_list()

    scale = to_pair(scale, 2)
    sh, sw = scale

    if data_format == 'NCHW':
        b, c, h, w = x_shape
        x_down = tf.reshape(x, [b, c, h//sh, sh, w//sw, sw])
        x_down = tf.slicing(x_down, (0, 0, 0, 0, 0, 0), (-1, -1, -1, 1, -1, 1))
        x_down = tf.squeeze(x_down, axis=(3, 5))
    elif data_format == 'NHWC':
        b, h, w, c = x_shape
        x_down = tf.reshape(x, [b, h//sh, sh, w//sw, sw, c])
        x_down = tf.slicing(x_down, (0, 0, 0, 0, 0, 0), (-1, -1, 1, -1, 1, -1))
        x_down = tf.squeeze(x_down, axis=(2, 4))
    else:
        raise ValueError

    return x_down
