#!/usr/bin/env python
#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
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
#
"""
Copyright 2019, Yao Yao, HKUST.
Differentiable homography related.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *
"""Python layer for image_ops."""
from tensorflow.python.eager import context
from tensorflow.contrib.image.ops import gen_image_ops
from tensorflow.contrib.util import loader
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import resource_loader
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

_IMAGE_DTYPES = set(
    [dtypes.uint8, dtypes.int32, dtypes.int64,
     dtypes.float16, dtypes.float32, dtypes.float64])

def get_homographies(left_cam, right_cam, depth_num, depth_start, depth_interval):
    with tf.name_scope('get_homographies'):
        # cameras (K, R, t) K是内参矩阵，描述相机的内参比例如焦距；R是旋转矩阵，代表相机参考的坐标轴方向；t是平移向量
        R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])#第一维度全取，即所有图像，第二维度只取1，第三维度取前3，第四维度取前三个，[？，1，3，3]
        R_right = tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])#第一维度全取，即所有图像，第二维度只取1，第三维度取前3，第四维度取第三个
        t_right = tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        K_left = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])#第一维度全取，即所有图像，第二维度从第2开始，第三维度取前3，第四维度取前三个
        K_right = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])

        # depth tf.cast：数据类型转换；tf.shape：矩阵格式变换：tf.shape(t,[])将一个单元素张量t重塑为标量,t=[7],result=7
        depth_num = tf.reshape(tf.cast(depth_num, 'int32'), [])
        depth = depth_start + tf.cast(tf.range(depth_num), tf.float32) * depth_interval#张量
        # preparation tf.squeeze会将原始input中所有维度为1的那些维都删掉，例如在tf.slice之后[[1,1,3]]，squeeze(axis=1)之后就变成了[1,1,3]
        num_depth = tf.shape(depth)[0]#depth的第一维度值
        K_left_inv = tf.matrix_inverse(tf.squeeze(K_left, axis=1))#tf.squeeze：去掉所有是维度1的张量，可以给出数组参数去掉指定的维度;tf.matrix_inverse矩阵求逆
        R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])#tf.transpose默认将矩阵转置；perm中的[0,1,2]对应高、行、列，[?,3,3]
        R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])#对0维的矩阵进行转置时perm设置为[0,2,1]，代表将三维数组的行和列进行转置,[?,3,3]

        fronto_direction = tf.slice(tf.squeeze(R_left, axis=1), [0, 2, 0], [-1, 1, 3])          # (B, D, 1, 3)[?,1,3,3]→[?,3,3]全取、第三个、三个全取→[?,1,3]

        c_left = -tf.matmul(R_left_trans, tf.squeeze(t_left, axis=1))#矩阵乘，R_left*t_left,[?,3,3]*[?,3,1]=[?,3,1]
        c_right = -tf.matmul(R_right_trans, tf.squeeze(t_right, axis=1))                        # (B, D, 3, 1)
        c_relative = tf.subtract(c_right, c_left)#R_right*t_right-R_left*t_left [?,3,1]

        # compute
        batch_size = tf.shape(R_left)[0]#样本数
        temp_vec = tf.matmul(c_relative, fronto_direction)#(R_right*t_right-R_left*t_left)*R_left [?,3,1]*[?,1,3] [?,3,3]
        depth_mat = tf.tile(tf.reshape(depth, [batch_size, num_depth, 1, 1]), [1, 1, 3, 3])#在四个维度上各复制[1，1，3，3]次，batchsize样本数,num_depth为depth的第一维度值[batchsize,num_depth,3,3]

        temp_vec = tf.tile(tf.expand_dims(temp_vec, axis=1), [1, num_depth, 1, 1])#axis指定扩展的位置，从0开始,0就是最左,[?,3,3]→[?,(1),3,3]→tile后为[?，num_depth,3,3]
        #?=batch_size 样本数
        middle_mat0 = tf.eye(3, batch_shape=[batch_size, num_depth]) - temp_vec / depth_mat#tf.eye构造[3,batch_size,num_depth]的矩阵，矩阵除单位矩阵部分外用0填充 [?,num_depth,3,3]
        middle_mat1 = tf.tile(tf.expand_dims(tf.matmul(R_left_trans, K_left_inv), axis=1), [1, num_depth, 1, 1])#(R_left*K_left),expand为[?,1,3,3],tile后为[?,num_depth,3,3]
        middle_mat2 = tf.matmul(middle_mat0, middle_mat1)#[?,num_depth,3,3]*[?,num_depth,3,3] [?,num_depth,3,3]

        homographies = tf.matmul(tf.tile(K_right, [1, num_depth, 1, 1])
                     , tf.matmul(tf.tile(R_right, [1, num_depth, 1, 1])
                     , middle_mat2))#k_right[?,1,3,3] [?,num_depth,3,3]

    return homographies

def get_homographies_inv_depth(left_cam, right_cam, depth_num, depth_start, depth_end):
    #与上一个函数效果一致，只是上个函数的输入是depth_start以及间隔，这个的输入是起止
    with tf.name_scope('get_homographies'):
        # cameras (K, R, t)
        R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        R_right = tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        t_right = tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        K_left = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_right = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])

        # depth 
        depth_num = tf.reshape(tf.cast(depth_num, 'int32'), [])
        #逆深度
        inv_depth_start = tf.reshape(tf.div(1.0, depth_start), [])
        inv_depth_end = tf.reshape(tf.div(1.0, depth_end), [])
        inv_depth = tf.lin_space(inv_depth_start, inv_depth_end, depth_num)#返回从start到end中depth_num个均匀间隔的数字
        depth = tf.div(1.0, inv_depth)#张量

        # preparation
        num_depth = tf.shape(depth)[0]
        K_left_inv = tf.matrix_inverse(tf.squeeze(K_left, axis=1))
        R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
        R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])

        fronto_direction = tf.slice(tf.squeeze(R_left, axis=1), [0, 2, 0], [-1, 1, 3])          # (B, D, 1, 3)

        c_left = -tf.matmul(R_left_trans, tf.squeeze(t_left, axis=1))
        c_right = -tf.matmul(R_right_trans, tf.squeeze(t_right, axis=1))                        # (B, D, 3, 1)
        c_relative = tf.subtract(c_right, c_left)        

        # compute
        batch_size = tf.shape(R_left)[0]
        temp_vec = tf.matmul(c_relative, fronto_direction)
        depth_mat = tf.tile(tf.reshape(depth, [batch_size, num_depth, 1, 1]), [1, 1, 3, 3])

        temp_vec = tf.tile(tf.expand_dims(temp_vec, axis=1), [1, num_depth, 1, 1])

        middle_mat0 = tf.eye(3, batch_shape=[batch_size, num_depth]) - temp_vec / depth_mat
        middle_mat1 = tf.tile(tf.expand_dims(tf.matmul(R_left_trans, K_left_inv), axis=1), [1, num_depth, 1, 1])
        middle_mat2 = tf.matmul(middle_mat0, middle_mat1)

        homographies = tf.matmul(tf.tile(K_right, [1, num_depth, 1, 1])
                     , tf.matmul(tf.tile(R_right, [1, num_depth, 1, 1])
                     , middle_mat2))

    return homographies

def get_pixel_grids(height, width):
    # texture coordinate
    x_linspace = tf.linspace(0.5, tf.cast(width, 'float32') - 0.5, width)# (width-0.5)-0.5=width-1
    y_linspace = tf.linspace(0.5, tf.cast(height, 'float32') - 0.5, height)
    x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)#x_coordinates为[height,width]的数组，每行为x_linspace，共重复height次,y为[width,height]
    x_coordinates = tf.reshape(x_coordinates, [-1])#变成一维
    y_coordinates = tf.reshape(y_coordinates, [-1])
    ones = tf.ones_like(x_coordinates)#创建一个全1的数组，大小与x_coordinates相同,为height*width
    indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)#依然是一维数组
    return indices_grid

def repeat_int(x, num_repeats):
    ones = tf.ones((1, num_repeats), dtype='int32')#产生[1,num_repeats]的全1数组
    x = tf.reshape(x, shape=(-1, 1))#x的第二维度的大小为1 [num_repeats,1]
    x = tf.matmul(x, ones)#[num_repeats,num_repeats]
    return tf.reshape(x, [-1])#变成一维数组

def repeat_float(x, num_repeats):
    ones = tf.ones((1, num_repeats), dtype='float')
    x = tf.reshape(x, shape=(-1, 1))
    x = tf.matmul(x, ones)
    return tf.reshape(x, [-1])

def interpolate(image, x, y):
    image_shape = tf.shape(image)
    batch_size = image_shape[0]
    height =image_shape[1]
    width = image_shape[2]

    # image coordinate to pixel coordinate
    x = x - 0.5
    y = y - 0.5
    x0 = tf.cast(tf.floor(x), 'int32')#floor返回不大于x的最大整数
    x1 = x0 + 1#大于或等于x的最小整数
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    max_y = tf.cast(height - 1, dtype='int32')
    max_x = tf.cast(width - 1,  dtype='int32')
    x0 = tf.clip_by_value(x0, 0, max_x)#设置阈值，限制取值范围为[0，max]
    x1 = tf.clip_by_value(x1, 0, max_x)
    y0 = tf.clip_by_value(y0, 0, max_y)
    y1 = tf.clip_by_value(y1, 0, max_y)
    b = repeat_int(tf.range(batch_size), height * width)

    indices_a = tf.stack([b, y0, x0], axis=1)#图像中坐标张量数组
    indices_b = tf.stack([b, y0, x1], axis=1)
    indices_c = tf.stack([b, y1, x0], axis=1)
    indices_d = tf.stack([b, y1, x1], axis=1)

    pixel_values_a = tf.gather_nd(image, indices_a)#输出图像中的张量
    pixel_values_b = tf.gather_nd(image, indices_b)
    pixel_values_c = tf.gather_nd(image, indices_c)
    pixel_values_d = tf.gather_nd(image, indices_d)

    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')
    area_a = tf.expand_dims(((y1 - y) * (x1 - x)), 1)
    area_b = tf.expand_dims(((y1 - y) * (x - x0)), 1)
    area_c = tf.expand_dims(((y - y0) * (x1 - x)), 1)
    area_d = tf.expand_dims(((y - y0) * (x - x0)), 1)
    output = tf.add_n([area_a * pixel_values_a,
                        area_b * pixel_values_b,
                        area_c * pixel_values_c,
                        area_d * pixel_values_d])#四个区域增加
    return output

def homography_warping(input_image, homography):
    with tf.name_scope('warping_by_homography'):#homography为三维数组
        image_shape = tf.shape(input_image)
        batch_size = image_shape[0]
        height = image_shape[1]
        width = image_shape[2]

        # turn homography to affine_mat of size (B, 2, 3) and div_mat of size (B, 1, 3)
        affine_mat = tf.slice(homography, [0, 0, 0], [-1, 2, 3])
        div_mat = tf.slice(homography, [0, 2, 0], [-1, 1, 3])

        # generate pixel grids of size (B, 3, (W+1) x (H+1))
        pixel_grids = get_pixel_grids(height, width)
        pixel_grids = tf.expand_dims(pixel_grids, 0)
        pixel_grids = tf.tile(pixel_grids, [batch_size, 1])
        pixel_grids = tf.reshape(pixel_grids, (batch_size, 3, -1))
        # return pixel_grids

        # affine + divide tranform, output (B, 2, (W+1) x (H+1))
        grids_affine = tf.matmul(affine_mat, pixel_grids)
        grids_div = tf.matmul(div_mat, pixel_grids)
        grids_zero_add = tf.cast(tf.equal(grids_div, 0.0), dtype='float32') * 1e-7 # handle div 0
        grids_div = grids_div + grids_zero_add
        grids_div = tf.tile(grids_div, [1, 2, 1])
        grids_inv_warped = tf.div(grids_affine, grids_div)
        x_warped, y_warped = tf.unstack(grids_inv_warped, axis=1)
        x_warped_flatten = tf.reshape(x_warped, [-1])
        y_warped_flatten = tf.reshape(y_warped, [-1])

        # interpolation
        warped_image = interpolate(input_image, x_warped_flatten, y_warped_flatten)
        warped_image = tf.reshape(warped_image, shape=image_shape, name='warped_feature')
    # return input_image
    return warped_image

def transform(images,
              transforms,
              interpolation="BILINEAR",
              output_shape=None,
              name=None):
  """Applies the given transform(s) to the image(s).
  Args:
    images: A tensor of shape (num_images, num_rows, num_columns, num_channels)
       (NHWC), (num_rows, num_columns, num_channels) (HWC), or
       (num_rows, num_columns) (HW). The rank must be statically known (the
       shape is not `TensorShape(None)`.
    transforms: Projective transform matrix/matrices. A vector of length 8 or
       tensor of size N x 8. If one row of transforms is
       [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the *output* point
       `(x, y)` to a transformed *input* point
       `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
       where `k = c0 x + c1 y + 1`. The transforms are *inverted* compared to
       the transform mapping input points to output points. Note that gradients
       are not backpropagated into transformation parameters.
    interpolation: Interpolation mode. Supported values: "NEAREST", "BILINEAR".
    output_shape: Output dimesion after the transform, [height, width].
       If None, output is the same size as input image.
    name: The name of the op.
  Returns:
    Image(s) with the same type and shape as `images`, with the given
    transform(s) applied. Transformed coordinates outside of the input image
    will be filled with zeros.
  Raises:
    TypeError: If `image` is an invalid type.
    ValueError: If output shape is not 1-D int32 Tensor.
  """
  with ops.name_scope(name, "transform"):
    image_or_images = ops.convert_to_tensor(images, name="images")
    transform_or_transforms = ops.convert_to_tensor(
        transforms, name="transforms", dtype=dtypes.float32)
    if image_or_images.dtype.base_dtype not in _IMAGE_DTYPES:
      raise TypeError("Invalid dtype %s." % image_or_images.dtype)
    elif image_or_images.get_shape().ndims is None:
      raise TypeError("image_or_images rank must be statically known")
    # 填充为4-D张量
    elif len(image_or_images.get_shape()) == 2:
      images = image_or_images[None, :, :, None]
    elif len(image_or_images.get_shape()) == 3:
      images = image_or_images[None, :, :, :]
    elif len(image_or_images.get_shape()) == 4:
      images = image_or_images
    else:
      raise TypeError("Images should have rank between 2 and 4.")

    if output_shape is None:
      output_shape = array_ops.shape(images)[1:3]
      if not context.executing_eagerly():
        output_shape_value = tensor_util.constant_value(output_shape)
        if output_shape_value is not None:
          output_shape = output_shape_value
    #转换为张量
    output_shape = ops.convert_to_tensor(
        output_shape, dtypes.int32, name="output_shape")

    if not output_shape.get_shape().is_compatible_with([2]):
      raise ValueError("output_shape must be a 1-D Tensor of 2 elements: "
                       "new_height, new_width")

    if len(transform_or_transforms.get_shape()) == 1:
      transforms = transform_or_transforms[None]
    elif transform_or_transforms.get_shape().ndims is None:
      raise TypeError(
          "transform_or_transforms rank must be statically known")
    elif len(transform_or_transforms.get_shape()) == 2:
      transforms = transform_or_transforms
    else:
      raise TypeError("Transforms should have rank 1 or 2.")

    output = gen_image_ops.image_projective_transform_v2(
        images,
        output_shape=output_shape,
        transforms=transforms,
        interpolation=interpolation.upper())
    if len(image_or_images.get_shape()) == 2:
      return output[0, :, :, 0]
    elif len(image_or_images.get_shape()) == 3:
      return output[0, :, :, :]
    else:
      return output

def tf_transform_homography(input_image, homography):

	# tf.contrib.image.transform is for pixel coordinate but our
	# homograph parameters are for image coordinate (x_p = x_i + 0.5).
	# So need to change the corresponding homography parameters 
    homography = tf.reshape(homography, [-1, 9])#homography为二维数组
    a0 = tf.slice(homography, [0, 0], [-1, 1])
    a1 = tf.slice(homography, [0, 1], [-1, 1])
    a2 = tf.slice(homography, [0, 2], [-1, 1])
    b0 = tf.slice(homography, [0, 3], [-1, 1])
    b1 = tf.slice(homography, [0, 4], [-1, 1])
    b2 = tf.slice(homography, [0, 5], [-1, 1])
    c0 = tf.slice(homography, [0, 6], [-1, 1])
    c1 = tf.slice(homography, [0, 7], [-1, 1])
    c2 = tf.slice(homography, [0, 8], [-1, 1])
    a_0 = a0 - c0 / 2
    a_1 = a1 - c1 / 2
    a_2 = (a0 + a1) / 2 + a2 - (c0 + c1) / 4 - c2 / 2
    b_0 = b0 - c0 / 2
    b_1 = b1 - c1 / 2
    b_2 = (b0 + b1) / 2 + b2 - (c0 + c1) / 4 - c2 / 2
    c_0 = c0
    c_1 = c1
    c_2 = c2 + (c0 + c1) / 2
    homo = []
    homo.append(a_0)
    homo.append(a_1)
    homo.append(a_2)
    homo.append(b_0)
    homo.append(b_1)
    homo.append(b_2)
    homo.append(c_0)
    homo.append(c_1)
    homo.append(c_2)
    homography = tf.stack(homo, axis=1)
    homography = tf.reshape(homography, [-1, 9])

    homography_linear = tf.slice(homography, begin=[0, 0], size=[-1, 8])
    homography_linear_div = tf.tile(tf.slice(homography, begin=[0, 8], size=[-1, 1]), [1, 8])
    homography_linear = tf.div(homography_linear, homography_linear_div)#变换矩阵

    warped_image = transform(
       input_image, homography_linear, interpolation='BILINEAR')
    # warped_image = tf.transpose
    # warped_image = cv2.warpPerspective(img_numpy, homography_linear, (9, 9))
    # tfa.image.transform_ops
    # return input_image
    return warped_image
"""
init_op = tf.global_variables_initializer()
    config = tf.ConfigProto(allow_soft_placement = True,log_device_placement = True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=npu_config_proto(config_proto=config)) as sess:
        # initialization
        sess.run(init_op)
        img_numpy = input_image.eval(session=sess)
"""