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
import tensorflow as tf
import numpy as np
import os
import h5py
from pre_process import switch_joint_order, project_tf,update_keypoint2d,calc_crop_scale,crop_image,create_multiple_gaussian_map

#the path of the output bin file
msameoutdir = '/root/output'
binfiles = os.listdir(msameoutdir)
binfiles.sort(key=lambda x:int(x.split('_')[0]))
predicteds = []
for file in binfiles:
    if file.endswith(".bin"):
        idx = int(file.split('_')[0])
        name = file.split('_')[1] + '_' + file.split('_')[2] + '_' + file.split('_')[3] + '_' + file.split('_')[4] + '.jpg'
        outputNum = file.split('output_')[1].split('.')[0]
        tmp = np.fromfile(msameoutdir + '/' + file, dtype='float32')
        list = []
        if idx>=len(predicteds):
            predicteds.append({})
        if outputNum== "0":
            predicted_scoremaps = np.reshape(tmp, [6, 1, 46, 46, 21])
            for i in predicted_scoremaps:
                i = tf.convert_to_tensor(i)
                list.append(i)
            predicteds[idx]['predicted_scoremaps'] = list
        elif outputNum== "1":
            predicteds[idx]['name'] = name
        else:
            predicted_PAFs = np.reshape(tmp, [6, 1, 46, 46, 69])
            for i in predicted_PAFs:
                i = tf.convert_to_tensor(i)
                list.append(i)
            predicteds[idx]['predicted_PAFs'] = list

crop_size = 368
sigma = 7
human36m_to_main = {
    'body': np.array([9, 8, 14, 15, 16, 11, 12, 13, 4, 5, 6, 1, 2, 3, 17, 17, 17, 17, 10, 17], dtype=np.int64)
}

#the path of h36m test set
image_root = '/root/h36m/images/'
image_list_file = '/root/h36m/annot/valid_images.txt'
path_to_db = '/root/h36m/annot/valid.h5'
path_to_calib = '/root/h36m/annot/cameras.h5'

with open(image_list_file) as f:
    img_list = [_.strip() for _ in f]
fannot = h5py.File(path_to_db, 'r')
annot3d = fannot['S'][:]
annot2d = fannot['part'][:]
fannot.close()
#get the camera calib data
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

for img_idx in range(0,len(predicteds)):
    img_name = predicteds[img_idx]['name']
    img_dir = os.path.join(image_root, img_name)
    print(img_dir)
    body2d = annot2d[img_idx].astype(np.float32)

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

    crop_center3d, scale3d, crop_center2d, scale2d = calc_crop_scale(keypoints, calib['K'], calib['distCoef'], valid[0])
    image_crop = crop_image(image, crop_center2d, scale2d)


    body2d_local = update_keypoint2d(body2d, crop_center2d, scale2d)
    scoremap2d = create_multiple_gaussian_map(body2d_local[:, ::-1], (crop_size, crop_size), sigma, valid_vec=valid[0], extra=True)
    mask = tf.ones((imh, imw), dtype=tf.float32)
    mask_crop = crop_image(tf.stack([mask] * 3, axis=2), crop_center2d, scale2d)
    mask_crop = mask_crop[:, :, 0]
    from utils.PAF import createPAF
    PAF = createPAF(body2d_local, body3d, 0, (crop_size, crop_size), True, valid_vec=valid[0])
    PAF_type = tf.ones([], dtype=bool)

    data = {}
    data['scoremap2d'] = scoremap2d
    data['valid'] = valid[0]
    data['mask_crop'] = mask_crop
    data['PAF'] =PAF
    data['PAF_type'] = PAF_type

    scoremap2d = tf.expand_dims(data['scoremap2d'], 0)
    valid = tf.expand_dims(data['valid'], 0)
    mask_crop = tf.expand_dims(data['mask_crop'], 0)
    s = scoremap2d.get_shape().as_list()
    valid = tf.concat([valid, tf.zeros((s[0], 1), dtype=tf.bool)], axis=1)
    valid = tf.cast(valid, tf.float32)

    mask_scoremap = tf.tile(tf.expand_dims(mask_crop, axis=3), [1, 1, 1, s[3]])
    loss_2d = 0.0
                    # multiply mask_scoremap to mask out the invalid areas
    for ip, predicted_scoremap in enumerate(predicted_scoremaps):
        resized_scoremap = tf.image.resize_images(predicted_scoremap, (s[1], s[2]), method=tf.image.ResizeMethod.BICUBIC)
        mean_over_pixel = tf.reduce_sum(tf.square((resized_scoremap - scoremap2d) * mask_scoremap), [1, 2]) / (tf.reduce_sum(mask_scoremap, [1, 2]) + 1e-6)
        loss_2d_ig = tf.reduce_sum(valid * mean_over_pixel) / (tf.reduce_sum(valid) + 1e-6)
        loss_2d += loss_2d_ig
    loss_2d /= len(predicted_scoremaps)
    sess = tf.Session()
    loss_2d = sess.run(loss_2d)

    valid = tf.expand_dims(data['valid'], 0)
    mask_crop = tf.expand_dims(data['mask_crop'], 0)
    PAF = tf.expand_dims(data['PAF'], 0)
    PAF_type = tf.expand_dims(data['PAF_type'], 0)
    loss_PAF = 0.0
    import utils.PAF
    valid_PAF = tf.cast(utils.PAF.getValidPAF(valid, 0, PAFdim=3), tf.float32)
    # multiply mask_PAF to mask out the invalid areas
    s = PAF.get_shape().as_list()
    mask_PAF = tf.tile(tf.expand_dims(mask_crop, axis=3), [1, 1, 1, s[3]])
    mask_PAF = tf.reshape(mask_PAF, [s[0], s[1], s[2], -1, 3])  # detach x, y, z
    mask_PAF2D = mask_PAF * tf.constant([1, 1, 0], dtype=tf.float32)  # for the 2D case
    mask_PAF = tf.where(PAF_type, mask_PAF, mask_PAF2D)  # take out corresponding mask by PAF type
    mask_PAF = tf.reshape(mask_PAF, [s[0], s[1], s[2], -1])
    for ip, pred_PAF in enumerate(predicted_PAFs):
        resized_PAF = tf.image.resize_images(pred_PAF, (s[1], s[2]), method=tf.image.ResizeMethod.BICUBIC)
        channelWisePAF = tf.reshape(resized_PAF, [s[0], s[1], s[2], -1, 3])
        PAF_x2y2 = tf.sqrt(tf.reduce_sum(tf.square(channelWisePAF[:, :, :, :, 0:2]), axis=4)) + 1e-6
        PAF_normed_x = channelWisePAF[:, :, :, :, 0] / PAF_x2y2
        PAF_normed_y = channelWisePAF[:, :, :, :, 1] / PAF_x2y2
        PAF_normed_z = tf.zeros(PAF_normed_x.get_shape(), dtype=tf.float32)
        normed_PAF = tf.stack([PAF_normed_x, PAF_normed_y, PAF_normed_z], axis=4)
        normed_PAF = tf.reshape(normed_PAF, [s[0], s[1], s[2], -1])
        normed_PAF = tf.where(tf.logical_and(tf.not_equal(PAF, 0.0), tf.not_equal(resized_PAF, 0.0)),normed_PAF, tf.zeros((s[0], s[1], s[2], s[3]), dtype=tf.float32))  # use normed_PAF only in pixels where PAF is not zero
        final_PAF = tf.where(PAF_type, resized_PAF, normed_PAF)
        # mean_over_pixel = tf.reduce_sum(tf.square((resized_PAF - data['PAF'][ig]) * mask_PAF), [1, 2]) / (tf.reduce_sum(mask_PAF, [1, 2]) + 1e-6)
        mean_over_pixel = tf.reduce_sum(tf.square((final_PAF - PAF) * mask_PAF), [1, 2]) / (tf.reduce_sum(mask_PAF, [1, 2]) + 1e-6)
        loss_PAF_ig = tf.reduce_sum(valid_PAF * mean_over_pixel) / (tf.reduce_sum(valid_PAF) + 1e-6)
        loss_PAF += loss_PAF_ig
    loss_PAF /= len(predicted_PAFs)
    loss_PAF = sess.run(loss_PAF)
    print("loss_2d: %f \t loss_PAF: %f \t picture_name: %s" % (loss_2d, loss_PAF, img_name))
