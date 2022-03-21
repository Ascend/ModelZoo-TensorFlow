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


"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. 
The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""


import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf
from npu_bridge.npu_init import *

from .util import renderer as vis_util
from .util import image as img_util
from .util import openpose as op_util
from .config import get_config
from .RunModel import RunModel
import os
from os.path import join
import matplotlib.pyplot as plt


flags.DEFINE_string('img_path', 'demo/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')
flags.DEFINE_string('save_path', 'demo/output', 'Output dir')


def visualize(img, proc_param, joints, verts, cam, save_path):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    os.makedirs(save_path, exist_ok=True)
    plt.imsave(join(save_path, 'input.jpg'), img)
    plt.imsave(join(save_path, 'joint_projection.jpg'), skel_img)
    plt.imsave(join(save_path, '3D_Mesh_overlay.jpg'), rend_img_overlay)
    plt.imsave(join(save_path, '3D_mesh.jpg'), rend_img)
    plt.imsave(join(save_path, 'diff_vp1.jpg'), rend_img_vp1)
    plt.imsave(join(save_path, 'diff_vp2.jpg'), rend_img_vp2)

def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def main(config):
    # define sess config
    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # close remap
    sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    sess = tf.Session(config=sess_config)
    model = RunModel(config=config, sess=sess)

    input_img, proc_param, img = preprocess_image(config.img_path, config.json_path)
    # Add batch dimension: 2 x D x D x 3
    input_img = np.stack([input_img] * 2)

    # Theta is the 85D vector holding [camera, pose, shape]
    # where camera is 3D [s, tx, ty]
    # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
    # shape is 10D shape coefficients of SMPL
    joints, verts, cams, joints3d, theta = model.predict(
        input_img, get_theta=True)

    visualize(img, proc_param, joints[0], verts[0], cams[0], config.save_path)


if __name__ == '__main__':
    config = get_config()
    config.batch_size = 2
    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)
    main(config)