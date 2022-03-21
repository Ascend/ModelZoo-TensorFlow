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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import h5py
import numpy as np
import utils.general
import tensorflow as tf

#the path of h36m test set
image_root = '/root/h36m/images/'
image_list_file = '/root/h36m/annot/valid.txt'
path_to_db = '/root/h36m/annot/valid.h5'
path_to_calib = '/root/h36m/annot/cameras.h5'

objtype=0
crop_noise = False
crop_size_zoom = 1.5
crop_size_zoom_2d = 1.8
crop_size = 368
grid_size = crop_size // 8
crop_scale_noise_sigma = 0.1
crop_offset_noise_sigma = 0.1
crop_scale_noise_sigma_2d = 0.1
crop_offset_noise_sigma_2d = 0.1
 
human36m_to_main = {
    'body': np.array([9, 8, 14, 15, 16, 11, 12, 13, 4, 5, 6, 1, 2, 3, 17, 17, 17, 17, 10, 17], dtype=np.int64)
}

def switch_joint_order(keypoint, order):
    # reorder the joints to the order used in our network
    #assert len(order.shape) == 1, 'order must be 1-dim'
    return keypoint[order, ...]

def project_tf(joint3d, calibK, calibR=None, calibt=None, calibDistCoef=None):
    with tf.name_scope('project_tf'):
        x = joint3d
        if calibR is not None:
            x = tf.matmul(joint3d, calibR, transpose_b=True)
        if calibt is not None:
            x = x + calibt
        xi = tf.divide(x[:, 0], x[:, 2])
        yi = tf.divide(x[:, 1], x[:, 2])

        if calibDistCoef is not None:
            X2 = xi * xi
            Y2 = yi * yi
            XY = X2 * Y2
            R2 = X2 + Y2
            R4 = R2 * R2
            R6 = R4 * R2

            dc = calibDistCoef
            radial = 1.0 + dc[0] * R2 + dc[1] * R4 + dc[4] * R6
            tan_x = 2.0 * dc[2] * XY + dc[3] * (R2 + 2.0 * X2)
            tan_y = 2.0 * dc[3] * XY + dc[2] * (R2 + 2.0 * Y2)

            xi = radial * xi + tan_x
            yi = radial * yi + tan_y

        xp = tf.transpose(tf.stack([xi, yi], axis=0))
        pt = tf.matmul(xp, calibK[:2, :2], transpose_b=True) + calibK[:2, 2]
    return pt, x

def create_multiple_gaussian_map(coords_wh, output_size, sigma, valid_vec=None, extra=False):
    """ Creates a map of size (output_shape[0], output_shape[1]) at (center[0], center[1])
        with variance sigma for multiple coordinates."""
    with tf.name_scope('create_multiple_gaussian_map'):
        sigma = tf.cast(sigma, tf.float32)
        assert len(output_size) == 2
        s = coords_wh.get_shape().as_list()
        coords_wh = tf.cast(coords_wh, tf.int32)
        if valid_vec is not None:
            valid_vec = tf.cast(valid_vec, tf.float32)
            valid_vec = tf.squeeze(valid_vec)
            cond_val = tf.greater(valid_vec, 0.5)
        else:
            cond_val = tf.ones_like(coords_wh[:, 0], dtype=tf.float32)
            cond_val = tf.greater(cond_val, 0.5)

        cond_1_in = tf.logical_and(tf.less(coords_wh[:, 0], output_size[0] - 1), tf.greater(coords_wh[:, 0], 0))
        cond_2_in = tf.logical_and(tf.less(coords_wh[:, 1], output_size[1] - 1), tf.greater(coords_wh[:, 1], 0))
        cond_in = tf.logical_and(cond_1_in, cond_2_in)
        cond = tf.logical_and(cond_val, cond_in)

        coords_wh = tf.cast(coords_wh, tf.float32)

        # create meshgrid
        x_range = tf.expand_dims(tf.range(output_size[0]), 1)
        y_range = tf.expand_dims(tf.range(output_size[1]), 0)

        X = tf.cast(tf.tile(x_range, [1, output_size[1]]), tf.float32)
        Y = tf.cast(tf.tile(y_range, [output_size[0], 1]), tf.float32)

        X.set_shape((output_size[0], output_size[1]))
        Y.set_shape((output_size[0], output_size[1]))

        X = tf.expand_dims(X, -1)
        Y = tf.expand_dims(Y, -1)

        X_b = tf.tile(X, [1, 1, s[0]])
        Y_b = tf.tile(Y, [1, 1, s[0]])

        X_b -= coords_wh[:, 0]
        Y_b -= coords_wh[:, 1]

        dist = tf.square(X_b) + tf.square(Y_b)

        scoremap = tf.exp(-dist / (2 * tf.square(sigma))) * tf.cast(cond, tf.float32)

        if extra:
            negative = 1 - tf.reduce_sum(scoremap, axis=2, keep_dims=True)
            negative = tf.minimum(tf.maximum(negative, 0.0), 1.0)
            scoremap = tf.concat([scoremap, negative], axis=2)

        return scoremap

def calc_crop_scale(keypoints, calibK, calibDC, valid):
    if objtype == 0:
        keypoint_center = (keypoints[8] + keypoints[11]) / 2
        center_valid = tf.logical_and(valid[8], valid[11])
    elif objtype == 1:
        keypoint_center = keypoints[12]
        center_valid = valid[12]
    else:  # objtype == 2
        assert objtype == 2  # conditioned by the shape of input
        if keypoints.shape[0] == 18:
            keypoint_center = (keypoints[8] + keypoints[11]) / 2
            center_valid = tf.logical_and(valid[8], valid[11])
        else:
            keypoint_center = keypoints[12]
            center_valid = valid[12]

    valid_idx = tf.where(valid)[:, 0]
    valid_keypoints = tf.gather(keypoints, valid_idx, name='valid_keypoints')

    min_coord = tf.reduce_min(valid_keypoints, 0, name='min_coord')
    max_coord = tf.reduce_max(valid_keypoints, 0, name='max_coord')

    keypoint_center = tf.cond(center_valid, lambda: keypoint_center, lambda: (min_coord + max_coord) / 2)
    keypoint_center.set_shape((3,))

    fit_size = tf.reduce_max(tf.maximum(max_coord - keypoint_center, keypoint_center - min_coord))
    crop_scale_noise = tf.cast(1.0, tf.float32)
    if crop_noise:
        crop_scale_noise = tf.exp(tf.truncated_normal([], mean=0.0, stddev=crop_scale_noise_sigma))
        crop_scale_noise = tf.maximum(crop_scale_noise, tf.reciprocal(crop_size_zoom))
    crop_size_best = tf.multiply(crop_scale_noise, 2 * fit_size * crop_size_zoom, name='crop_size_best')

    crop_offset_noise = tf.cast(0.0, tf.float32)
    if crop_noise:
        crop_offset_noise = tf.truncated_normal([3], mean=0.0, stddev=crop_offset_noise_sigma) * fit_size * tf.constant([1., 1., 0.], dtype=tf.float32)
        crop_offset_noise = tf.maximum(crop_offset_noise, max_coord + 1e-5 - crop_size_best / 2 - keypoint_center)
        crop_offset_noise = tf.minimum(crop_offset_noise, min_coord - 1e-5 + crop_size_best / 2 - keypoint_center, name='crop_offset_noise')
    crop_center = tf.add(keypoint_center, crop_offset_noise, name='crop_center')

    crop_box_bl = tf.concat([crop_center[:2] - crop_size_best / 2, crop_center[2:]], 0)
    crop_box_ur = tf.concat([crop_center[:2] + crop_size_best / 2, crop_center[2:]], 0)

    crop_box = tf.stack([crop_box_bl, crop_box_ur], 0)
    scale = tf.cast(grid_size, tf.float32) / crop_size_best

    crop_box2d, _ = project_tf(crop_box, calibK, calibDistCoef=calibDC)
    min_coord2d = tf.reduce_min(crop_box2d, 0)
    max_coord2d = tf.reduce_max(crop_box2d, 0)
    crop_size_best2d = tf.reduce_max(max_coord2d - min_coord2d)
    crop_center2d = (min_coord2d + max_coord2d) / 2
    scale2d = tf.cast(crop_size, tf.float32) / crop_size_best2d
    return crop_center, scale, crop_center2d, scale2d

def calc_crop_scale2d(keypoints, valid):
    # assert self.objtype == 2
    if keypoints.shape[0] == 19 or keypoints.shape[0] == 20:
        keypoint_center = (keypoints[8] + keypoints[11]) / 2
        center_valid = tf.logical_and(valid[8], valid[11])
    else:
        keypoint_center = keypoints[12]
        center_valid = valid[12]

    valid_idx = tf.where(valid)[:, 0]
    valid_keypoints = tf.gather(keypoints, valid_idx)
    min_coord = tf.reduce_min(valid_keypoints, 0)
    max_coord = tf.reduce_max(valid_keypoints, 0)

    keypoint_center = tf.cond(center_valid, lambda: keypoint_center, lambda: (min_coord + max_coord) / 2)
    keypoint_center.set_shape((2,))

    fit_size = tf.reduce_max(tf.maximum(max_coord - keypoint_center, keypoint_center - min_coord))
    crop_scale_noise = tf.cast(1.0, tf.float32)
    if crop_noise:
        crop_scale_noise = tf.exp(tf.truncated_normal([], mean=0.0, stddev=crop_scale_noise_sigma_2d))
    crop_size_best = 2 * fit_size * crop_size_zoom_2d * crop_scale_noise

    crop_offset_noise = tf.cast(0.0, tf.float32)
    if crop_noise:
        crop_offset_noise = tf.truncated_normal([2], mean=0.0, stddev=crop_offset_noise_sigma_2d) * fit_size
        crop_offset_noise = tf.maximum(crop_offset_noise, keypoint_center - crop_size_best / 2 - min_coord + 1)
        crop_offset_noise = tf.minimum(crop_offset_noise, keypoint_center + crop_size_best / 2 - max_coord - 1)
    crop_center = keypoint_center + crop_offset_noise
    scale2d = tf.cast(crop_size, tf.float32) / crop_size_best
    return crop_center, scale2d

def crop_image(image, crop_center2d, scale2d):
    image_crop = utils.general.crop_image_from_xy(tf.expand_dims(image, 0), crop_center2d[::-1], crop_size, scale2d)  # crop_center_hw
    image_crop = tf.squeeze(image_crop)
    return image_crop

def update_keypoint2d(keypoint2d, crop_center2d, scale2d):
    keypoint_x = (keypoint2d[:, 0] - crop_center2d[0]) * scale2d + crop_size // 2
    keypoint_y = (keypoint2d[:, 1] - crop_center2d[1]) * scale2d + crop_size // 2
    keypoint2d_local = tf.stack([keypoint_x, keypoint_y], 1)
    keypoint2d_local = keypoint2d_local
    return keypoint2d_local

with open(image_list_file) as f:
    img_list = [_.strip() for _ in f]
fannot = h5py.File(path_to_db, 'r')
annot3d = fannot['S'][:]
annot2d = fannot['part'][:]
fannot.close()
fcalib = h5py.File(path_to_calib, 'r')
calib_data = {}
map_camera = {'54138969': 'camera1', '55011271': 'camera2', '58860488': 'camera3', '60457274': 'camera4'}
for pid in fcalib.keys():
    if pid == '3dtest':
        continue
    person_cam_data = {}
    for camera in map_camera.values():
        cam_data = {_: fcalib[pid][camera][_][:] for _ in fcalib[pid][camera].keys()}
        person_cam_data[camera] = cam_data
    calib_data[pid] = person_cam_data
fcalib.close()

for img_idx, img_name in enumerate(img_list):
    img_dir = os.path.join(image_root, img_name)
    body2d = annot2d[img_idx].astype(np.float32)
    if (body2d >= 1000).any() or (body2d <= 0).any():
        continue
    body3d = annot3d[img_idx].astype(np.float32)
    body3d = np.concatenate((body3d, np.ones((1, 3), dtype=np.float32)), axis=0)  # put dummy values in order_dict
    person = img_name.split('_')[0].replace('S', 'subject')
    camera = img_name.split('.')[1].split('_')[0]
    camera_name = map_camera[camera]
    cam_param = calib_data[person][camera_name]

    K = np.eye(3)
    K[0, 0] = cam_param['f'][0, 0]
    K[1, 1] = cam_param['f'][1, 0]
    K[0, 2] = cam_param['c'][0, 0]
    K[1, 2] = cam_param['c'][1, 0]
    dc = np.zeros((5,))
    dc[:3] = cam_param['k'][:, 0]
    dc[3:] = cam_param['p'][:, 0]

    calib = {'K': [], 'R': [], 't': [], 'distCoef': []}
    calib['K'] = K.astype(np.float32)
    calib['R'] = np.eye(3, dtype=np.float32)
    calib['t'] = np.zeros((3,), dtype=np.float32)
    calib['distCoef'] = dc.astype(np.float32)
    
    body_valid = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0]], dtype=bool)
    body3d = np.array(body3d)
    body3d = switch_joint_order(body3d, human36m_to_main['body'])
    body2d, body3d = project_tf(body3d, calib['K'], calib['R'], calib['t'], calib['distCoef'])
    body3d = tf.cast(body3d, tf.float32)
    body2d = tf.cast(body2d, tf.float32) 
 
    imw = 1000
    imh = 1002
    img_file = tf.read_file(img_dir)
    image = tf.image.decode_image(img_file, channels=3)
    image = tf.image.pad_to_bounding_box(image, 0, 0, imh, imw)
    image.set_shape((imh, imw, 3))
    image = tf.cast(image, tf.float32) / 255.0 - 0.5

    keypoints = body3d
    valid = body_valid
    crop_center3d, scale3d, crop_center2d, scale2d = calc_crop_scale(keypoints, calib['K'], calib['distCoef'], body_valid[0])
    image_crop = crop_image(image, crop_center2d, scale2d)
    image_crop = tf.reshape(image_crop, shape=(135424,3))
    image_crop = tf.Session().run(image_crop)
    image_crop.tofile("TestData/{0:d}_{1}.bin".format(img_idx,img_name.split('.jpg')[0]))
